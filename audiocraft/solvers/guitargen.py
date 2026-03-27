# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import time
import typing as tp
import warnings

import flashy
import math
import omegaconf
import torch
from torch.nn import functional as F

from . import base, builders
from .compression import CompressionSolver
from .. import metrics as eval_metrics
from .. import models
from ..data.audio_dataset import AudioDataset
from ..data.music_dataset import MusicDataset, MusicInfo, AudioInfo
from ..data.audio_utils import normalize_audio
from ..modules.conditioners import JointEmbedCondition, SegmentWithAttributes, WavCondition, \
            StyleConditioner, _drop_description_condition
from ..utils.cache import CachedBatchWriter, CachedBatchLoader
from ..utils.samples.manager import SampleManager
from ..utils.utils import get_dataset_from_loader, is_jsonable, warn_once, model_hash

from . import musicgen

class GuitarGenSolver(musicgen.MusicGenSolver):
    """Solver for GuitarGen training task.
    """


    def _prepare_tokens_and_attributes(
        self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
        check_synchronization_points: bool = False
    ) -> tp.Tuple[dict, torch.Tensor, torch.Tensor]:
        """Prepare input batchs for language model training.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]): Input batch with audio tensor of shape [B, C, T]
                and corresponding metadata as SegmentWithAttributes (with B items).
            check_synchronization_points (bool): Whether to check for synchronization points slowing down training.
        Returns:
            Condition tensors (dict[str, any]): Preprocessed condition attributes.
            Tokens (torch.Tensor): Audio tokens from compression model, of shape [B, K, T_s],
                with B the batch size, K the number of codebooks, T_s the token timesteps.
            Padding mask (torch.Tensor): Mask with valid positions in the tokens tensor, of shape [B, K, T_s].
        """
        if self.model.training:
            warnings.warn(
                "Up to version 1.0.1, the _prepare_tokens_and_attributes was evaluated with `torch.no_grad()`. "
                "This is inconsistent with how model were trained in the MusicGen paper. We removed the "
                "`torch.no_grad()` in version 1.1.0. Small changes to the final performance are expected. "
                "Really sorry about that.")
        if self._cached_batch_loader is None or self.current_stage != "train":
            audio, infos = batch
            audio = audio.to(self.device)
            audio_tokens = None
            assert audio.size(0) == len(infos), (
                f"Mismatch between number of items in audio batch ({audio.size(0)})",
                f" and in metadata ({len(infos)})"
            )
        else:
            audio = None
            # In that case the batch will be a tuple coming from the _cached_batch_writer bit below.
            infos, = batch  # type: ignore
            assert all([isinstance(info, AudioInfo) for info in infos])
            assert all([info.audio_tokens is not None for info in infos])  # type: ignore
            audio_tokens = torch.stack([info.audio_tokens for info in infos]).to(self.device)  # type: ignore
            audio_tokens = audio_tokens.long()
            for info in infos:
                if isinstance(info, MusicInfo):
                    # Careful here, if you want to use this condition_wav (e.b. chroma conditioning),
                    # then you must be using the chroma cache! otherwise the code will try
                    # to use this segment and fail (by that I mean you will see NaN everywhere).
                    info.self_wav = WavCondition(
                        torch.full([1, info.channels, info.total_frames], float('NaN')),
                        length=torch.tensor([info.n_frames]),
                        sample_rate=[info.sample_rate],
                        path=[info.meta.path],
                        seek_time=[info.seek_time])
                    dataset = get_dataset_from_loader(self.dataloaders['original_train'])
                    assert isinstance(dataset, MusicDataset), type(dataset)
                    if dataset.paraphraser is not None and info.description is not None:
                        # Hackingly reapplying paraphraser when using cache.
                        info.description = dataset.paraphraser.sample_paraphrase(
                            info.meta.path, info.description)
        # prepare attributes
        attributes = [info.to_condition_attributes() for info in infos]
        attributes = self.model.cfg_dropout(attributes)
        attributes = self.model.att_dropout(attributes)
        tokenized = self.model.condition_provider.tokenize(attributes)

        # Now we should be synchronization free.
        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("warn")

        if audio_tokens is None:
            audio_tokens = self._get_audio_tokens(audio)

        with self.autocast:
            condition_tensors = self.model.condition_provider(tokenized)

        # create a padding mask to hold valid vs invalid positions
        padding_mask = torch.ones_like(audio_tokens, dtype=torch.bool, device=audio_tokens.device)
        # replace encodec tokens from padded audio with special_token_id
        if self.cfg.tokens.padding_with_special_token:
            audio_tokens = audio_tokens.clone()
            padding_mask = padding_mask.clone()
            token_sample_rate = self.compression_model.frame_rate
            B, K, T_s = audio_tokens.shape
            for i in range(B):
                n_samples = infos[i].n_frames
                audio_sample_rate = infos[i].sample_rate
                # take the last token generated from actual audio frames (non-padded audio)
                valid_tokens = math.floor(float(n_samples) / audio_sample_rate * token_sample_rate)
                audio_tokens[i, :, valid_tokens:] = self.model.special_token_id
                padding_mask[i, :, valid_tokens:] = 0

        if self.device == "cuda" and check_synchronization_points:
            torch.cuda.set_sync_debug_mode("default")

        if self._cached_batch_writer is not None and self.current_stage == 'train':
            assert self._cached_batch_loader is None
            assert audio_tokens is not None
            for info, one_audio_tokens in zip(infos, audio_tokens):
                assert isinstance(info, AudioInfo)
                if isinstance(info, MusicInfo):
                    assert not info.joint_embed, "joint_embed and cache not supported yet."
                    info.self_wav = None
                assert one_audio_tokens.max() < 2**15, one_audio_tokens.max().item()
                info.audio_tokens = one_audio_tokens.short().cpu()
            self._cached_batch_writer.save(infos)

        return condition_tensors, audio_tokens, padding_mask

    def run_step(self, idx: int, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]], metrics: dict) -> dict:
        """Perform one training or valid step on a given batch."""
        check_synchronization_points = idx == 1 and self.device == 'cuda'

        condition_tensors, audio_tokens, padding_mask = self._prepare_tokens_and_attributes(
            batch, check_synchronization_points)

        self.deadlock_detect.update('tokens_and_conditions')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('warn')

        with self.autocast:
            style_mask = None
            if hasattr(self.model.condition_provider.conditioners, 'self_wav'):
                if isinstance(self.model.condition_provider.conditioners.self_wav, StyleConditioner):
                    style_mask = self.model.condition_provider.conditioners.self_wav.mask

            model_output = self.model.compute_predictions(audio_tokens, [], condition_tensors)  # type: ignore
            logits = model_output.logits
            if style_mask is not None:
                mask = padding_mask & model_output.mask & style_mask
            else:
                mask = padding_mask & model_output.mask
            ce, ce_per_codebook = self._compute_cross_entropy(logits, audio_tokens, mask)
            loss = ce
        self.deadlock_detect.update('loss')

        if check_synchronization_points:
            torch.cuda.set_sync_debug_mode('default')

        if self.is_training:
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            if self.scaler is not None:
                loss = self.scaler.scale(loss)
            self.deadlock_detect.update('scale')
            if self.cfg.fsdp.use:
                loss.backward()
                flashy.distrib.average_tensors(self.model.buffers())
            elif self.cfg.optim.eager_sync:
                with flashy.distrib.eager_sync_model(self.model):
                    loss.backward()
            else:
                # this should always be slower but can be useful
                # for weird use cases like multiple backwards.
                loss.backward()
                flashy.distrib.sync_model(self.model)
            self.deadlock_detect.update('backward')

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.cfg.optim.max_norm:
                if self.cfg.fsdp.use:
                    metrics['grad_norm'] = self.model.clip_grad_norm_(self.cfg.optim.max_norm)  # type: ignore
                else:
                    metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.max_norm
                    )
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            self.deadlock_detect.update('optim')
            if self.scaler is not None:
                scale = self.scaler.get_scale()
                metrics['grad_scale'] = scale
            if not loss.isfinite().all():
                raise RuntimeError("Model probably diverged.")

        metrics['ce'] = ce
        metrics['ppl'] = torch.exp(ce)
        for k, ce_q in enumerate(ce_per_codebook):
            metrics[f'ce_q{k + 1}'] = ce_q
            metrics[f'ppl_q{k + 1}'] = torch.exp(ce_q)

        return metrics

    @torch.no_grad()
    def run_generate_step(self, batch: tp.Tuple[torch.Tensor, tp.List[SegmentWithAttributes]],
                          gen_duration: float, prompt_duration: tp.Optional[float] = None,
                          remove_text_conditioning: bool = False,
                          ) -> dict:
        """Run generate step on a batch of optional audio tensor and corresponding attributes.

        Args:
            batch (tuple[torch.Tensor, list[SegmentWithAttributes]]):
            use_prompt (bool): Whether to do audio continuation generation with prompt from audio batch.
            gen_duration (float): Target audio duration for the generation.
            prompt_duration (float, optional): Duration for the audio prompt to use for continuation.
        Returns:
            gen_outputs (dict): Generation outputs, consisting in audio, audio tokens from both the generation
                and the prompt along with additional information.
        """
        bench_start = time.time()
        audio, meta = batch
        assert audio.size(0) == len(meta), (
            f"Mismatch between number of items in audio batch ({audio.size(0)})",
            f" and in metadata ({len(meta)})"
        )
        # prepare attributes
        attributes = [x.to_condition_attributes() for x in meta]
        if remove_text_conditioning:
            attributes = _drop_description_condition(attributes)

        # prepare audio prompt
        if prompt_duration is None:
            prompt_audio = None
        else:
            assert prompt_duration < gen_duration, "Prompt duration must be lower than target generation duration"
            prompt_audio_frames = int(prompt_duration * self.compression_model.sample_rate)
            prompt_audio = audio[..., :prompt_audio_frames]

        # get audio tokens from compression model
        if prompt_audio is None or prompt_audio.nelement() == 0:
            num_samples = len(attributes)
            prompt_tokens = None
        else:
            num_samples = None
            prompt_audio = prompt_audio.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt_audio)
            assert scale is None, "Compression model in MusicGen should not require rescaling."

        # generate by sampling from the LM
        with self.autocast:
            total_gen_len = math.ceil(gen_duration * self.compression_model.frame_rate)
            gen_tokens = self.model.generate(
                prompt_tokens, attributes, max_gen_len=total_gen_len,
                num_samples=num_samples, **self.generation_params)

        # generate audio from tokens
        assert gen_tokens.dim() == 3
        gen_audio = self.compression_model.decode(gen_tokens, None)

        bench_end = time.time()
        gen_outputs = {
            'rtf': (bench_end - bench_start) / gen_duration,
            'ref_audio': audio,
            'gen_audio': gen_audio,
            'gen_tokens': gen_tokens,
            'prompt_audio': prompt_audio,
            'prompt_tokens': prompt_tokens,
        }
        return gen_outputs

    def generate_audio(self) -> dict:
        """Audio generation stage."""
        generate_stage_name = f'{self.current_stage}'
        sample_manager = SampleManager(self.xp)
        self.logger.info(f"Generating samples in {sample_manager.base_folder}")
        loader = self.dataloaders['generate']
        updates = len(loader)
        lp = self.log_progress(generate_stage_name, loader, total=updates, updates=self.log_updates)

        dataset = get_dataset_from_loader(loader)
        dataset_duration = dataset.segment_duration
        assert dataset_duration is not None
        assert isinstance(dataset, AudioDataset)
        target_duration = self.cfg.generate.lm.gen_duration
        prompt_duration = self.cfg.generate.lm.prompt_duration
        if target_duration is None:
            target_duration = dataset_duration
        if prompt_duration is None:
            prompt_duration = dataset_duration / 4
        assert prompt_duration < dataset_duration, (
            f"Specified prompt duration ({prompt_duration}s) is longer",
            f" than reference audio duration ({dataset_duration}s)"
        )

        def get_hydrated_conditions(meta: tp.List[SegmentWithAttributes]):
            hydrated_conditions = []
            for sample in [x.to_condition_attributes() for x in meta]:
                cond_dict = {}
                for cond_type in sample.__annotations__.keys():
                    for cond_key, cond_val in getattr(sample, cond_type).items():
                        if cond_key not in self.model.condition_provider.conditioners.keys():
                            continue
                        if is_jsonable(cond_val):
                            cond_dict[cond_key] = cond_val
                        elif isinstance(cond_val, WavCondition):
                            cond_dict[cond_key] = cond_val.path
                        elif isinstance(cond_val, JointEmbedCondition):
                            cond_dict[cond_key] = cond_val.text  # only support text at inference for now
                        else:
                            # if we reached this point, it is not clear how to log the condition
                            # so we just log the type.
                            cond_dict[cond_key] = str(type(cond_val))
                            continue
                hydrated_conditions.append(cond_dict)
            return hydrated_conditions

        metrics: dict = {}
        average = flashy.averager()
        for batch in lp:
            audio, meta = batch
            # metadata for sample manager
            hydrated_conditions = get_hydrated_conditions(meta)
            sample_generation_params = {
                **{f'classifier_free_guidance_{k}': v for k, v in self.cfg.classifier_free_guidance.items()},
                **self.generation_params
            }
            if self.cfg.generate.lm.unprompted_samples:
                if self.cfg.generate.lm.gen_gt_samples:
                    # get the ground truth instead of generation
                    self.logger.warn(
                        "Use ground truth instead of audio generation as generate.lm.gen_gt_samples=true")
                    gen_unprompted_audio = audio
                    rtf = 1.
                else:
                    gen_unprompted_outputs = self.run_generate_step(
                        batch, gen_duration=target_duration, prompt_duration=None)
                    gen_unprompted_audio = gen_unprompted_outputs['gen_audio'].cpu()
                    rtf = gen_unprompted_outputs['rtf']
                sample_manager.add_samples(
                    gen_unprompted_audio, self.epoch, hydrated_conditions,
                    ground_truth_wavs=audio, generation_args=sample_generation_params)

            if self.cfg.generate.lm.prompted_samples:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=prompt_duration)
                gen_audio = gen_outputs['gen_audio'].cpu()
                prompt_audio = gen_outputs['prompt_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)
            if self.cfg.generate.lm.no_text_conditioning:
                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration, prompt_duration=None,
                    remove_text_conditioning=self.cfg.generate.lm.no_text_conditioning)
                gen_audio = gen_outputs['gen_audio'].cpu()
                rtf = gen_outputs['rtf']
                # Here, the prompt is the original audio provided for the style conditioning
                prompt_audio = gen_outputs['ref_audio'].cpu()
                sample_manager.add_samples(
                    gen_audio, self.epoch, hydrated_conditions,
                    prompt_wavs=prompt_audio, ground_truth_wavs=audio,
                    generation_args=sample_generation_params)

            metrics['rtf'] = rtf
            metrics = average(metrics)

        flashy.distrib.barrier()
        return metrics

    def evaluate_audio_generation(self) -> dict:
        """Evaluate audio generation with off-the-shelf metrics."""
        evaluate_stage_name = f'{self.current_stage}_generation'
        # instantiate evaluation metrics, if at least one metric is defined, run audio generation evaluation
        fad: tp.Optional[eval_metrics.FrechetAudioDistanceMetric] = None
        kldiv: tp.Optional[eval_metrics.KLDivergenceMetric] = None
        text_consistency: tp.Optional[eval_metrics.TextConsistencyMetric] = None
        chroma_cosine: tp.Optional[eval_metrics.ChromaCosineSimilarityMetric] = None
        should_run_eval = False
        eval_chroma_wavs: tp.Optional[torch.Tensor] = None
        if self.cfg.evaluate.metrics.fad:
            fad = builders.get_fad(self.cfg.metrics.fad).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.kld:
            kldiv = builders.get_kldiv(self.cfg.metrics.kld).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.text_consistency:
            text_consistency = builders.get_text_consistency(self.cfg.metrics.text_consistency).to(self.device)
            should_run_eval = True
        if self.cfg.evaluate.metrics.chroma_cosine:
            chroma_cosine = builders.get_chroma_cosine_similarity(self.cfg.metrics.chroma_cosine).to(self.device)
            # if we have predefind wavs for chroma we should purge them for computing the cosine metric
            has_predefined_eval_chromas = 'self_wav' in self.model.condition_provider.conditioners and \
                                          self.model.condition_provider.conditioners['self_wav'].has_eval_wavs()
            if has_predefined_eval_chromas:
                warn_once(self.logger, "Attempting to run cosine eval for config with pre-defined eval chromas! "
                                       'Resetting eval chromas to None for evaluation.')
                eval_chroma_wavs = self.model.condition_provider.conditioners.self_wav.eval_wavs  # type: ignore
                self.model.condition_provider.conditioners.self_wav.reset_eval_wavs(None)  # type: ignore
            should_run_eval = True

        def get_compressed_audio(audio: torch.Tensor) -> torch.Tensor:
            audio_tokens, scale = self.compression_model.encode(audio.to(self.device))
            compressed_audio = self.compression_model.decode(audio_tokens, scale)
            return compressed_audio[..., :audio.shape[-1]]

        metrics: dict = {}
        if should_run_eval:
            loader = self.dataloaders['evaluate']
            updates = len(loader)
            lp = self.log_progress(f'{evaluate_stage_name} inference', loader, total=updates, updates=self.log_updates)
            average = flashy.averager()
            dataset = get_dataset_from_loader(loader)
            assert isinstance(dataset, AudioDataset)
            self.logger.info(f"Computing evaluation metrics on {len(dataset)} samples")

            for idx, batch in enumerate(lp):
                audio, meta = batch
                assert all([self.cfg.sample_rate == m.sample_rate for m in meta])

                target_duration = audio.shape[-1] / self.cfg.sample_rate
                if self.cfg.evaluate.fixed_generation_duration:
                    target_duration = self.cfg.evaluate.fixed_generation_duration

                gen_outputs = self.run_generate_step(
                    batch, gen_duration=target_duration,
                    remove_text_conditioning=self.cfg.evaluate.get('remove_text_conditioning', False)
                )
                y_pred = gen_outputs['gen_audio'].detach()
                y_pred = y_pred[..., :audio.shape[-1]]

                normalize_kwargs = dict(self.cfg.generate.audio)
                normalize_kwargs.pop('format', None)
                y_pred = torch.stack([normalize_audio(w, **normalize_kwargs) for w in y_pred], dim=0).cpu()
                y = audio.cpu()  # should already be on CPU but just in case
                sizes = torch.tensor([m.n_frames for m in meta])  # actual sizes without padding
                sample_rates = torch.tensor([m.sample_rate for m in meta])  # sample rates for audio samples
                audio_stems = [Path(m.meta.path).stem + f"_{m.seek_time}" for m in meta]

                if fad is not None:
                    if self.cfg.metrics.fad.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    fad.update(y_pred, y, sizes, sample_rates, audio_stems)
                if kldiv is not None:
                    if self.cfg.metrics.kld.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    kldiv.update(y_pred, y, sizes, sample_rates)
                if text_consistency is not None:
                    texts = [m.description for m in meta]
                    if self.cfg.metrics.text_consistency.use_gt:
                        y_pred = y
                    text_consistency.update(y_pred, texts, sizes, sample_rates)
                if chroma_cosine is not None:
                    if self.cfg.metrics.chroma_cosine.use_gt:
                        y_pred = get_compressed_audio(y).cpu()
                    chroma_cosine.update(y_pred, y, sizes, sample_rates)
                    # restore chroma conditioner's eval chroma wavs
                    if eval_chroma_wavs is not None:
                        self.model.condition_provider.conditioners['self_wav'].reset_eval_wavs(eval_chroma_wavs)

            flashy.distrib.barrier()
            if fad is not None:
                metrics['fad'] = fad.compute()
            if kldiv is not None:
                kld_metrics = kldiv.compute()
                metrics.update(kld_metrics)
            if text_consistency is not None:
                metrics['text_consistency'] = text_consistency.compute()
            if chroma_cosine is not None:
                metrics['chroma_cosine'] = chroma_cosine.compute()
            metrics = average(metrics)
            metrics = flashy.distrib.average_metrics(metrics, len(loader))

        return metrics
