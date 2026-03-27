# import pretty_midi
# import torchaudio
# import matplotlib.pyplot as plt
# import numpy as np
# midi = pretty_midi.PrettyMIDI("../GTechs/P1_scales/midi/midi_A.mid")

# audio, fs = torchaudio.load("../GTechs/P1_scales/audio/directinput/directinput_A.wav")
# audio /= audio.abs().max()

# onsets = []
# ends = []
# for i in midi.instruments:
#     for n in i.notes:
#         onsets.append(n.start)
#         ends.append(n.end)
# onsets = np.array(onsets)
# ends = np.array(ends)

# time = np.arange(audio.shape[-1])/fs
# midi_on = np.zeros(audio.shape[-1], dtype=bool)
# for i in range(len(onsets)):
#     midi_on += (time>onsets[i]) *( time<=ends[i] )

# max_len = 5.0
# time_idx = time<max_len

# fig, ax = plt.subplots(1,1, figsize=(10,5))
# ax.plot(time[time_idx], audio[0].numpy()[time_idx], color="grey")
# ax.plot(time[time_idx], midi_on[time_idx], color="blue")
# fig.savefig("../arp/test.png")


from audiocraft.solvers import CompressionSolver

model = CompressionSolver.model_from_checkpoint("/home/vuillemr/audiocraft/exp/xps/electro/checkpoint.th")


out.codes

tfr = StreamingTransformer(
    d_model=512,
    num_heads=8,
    num_layers=8,
    dim_feedforward=2048,
    dropout=0.1,

    causal=True,                 # ✅ VERY IMPORTANT
    cross_attention=False,       # ✅ no conditioning

    positional_embedding='sin',  # or 'learned'
    max_period=10_000,

    bias_ff=True,
    bias_attn=True,
)





from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import ConditioningProvider
from audiocraft.modules.conditioners import ConditionFuser
from audiocraft.models.lm import LMModel
from audiocraft import train
from pathlib import Path
import logging
import os
import sys
from audiocraft.modules.transformer import StreamingTransformer

os.chdir(Path(train.__file__).parent.parent)

fuser = ConditionFuser(
    fuse2cond={
                "prepend": [], 
                "sum":[],
                "cross":[],
            },
    cross_attention_pos_emb=False
)
pattern_provider = DelayedPatternProvider(n_q=8)
condition_provider = ConditioningProvider({})

lm_model = LMModel(
    pattern_provider=pattern_provider,
    condition_provider=condition_provider,
    fuser=fuser,

    n_q=8,
    card=1024,

    dim=256,            # good for your dataset
    num_heads=8,
    hidden_scale=4,
    num_layers=8,

    norm='layer_norm',
    norm_first=True,

    # transformer kwargs ↓↓↓
    causal=True,        # 🔥 REQUIRED
    cross_attention=False,
    positional_embedding='sin',
    max_period=10_000,
    bias_ff=False,
    bias_attn=False,
    bias_proj= False,
    layer_scale=None,
    weight_init="gaussian",
    depthwise_init="current",
    zero_bias_init=True,
    attention_as_float32=False,

    memory_efficient=True,

)

lm_model = lm_model.to("cuda")
solver = train.get_solver_from_sig('ab400dbd', {'device': 'cuda', 'dataset': {'batch_size': 1}})
encodec_model = solver.model
batch = next(iter(solver.dataloaders['train'])).to("cuda")

codes, scale = encodec_model.encode(batch)
print("codes=", codes.shape)

out = lm_model.compute_predictions(codes, conditions={})


for name, module in lm_model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params/1e6:.2f}M params")
for name, module in encodec_model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params/1e6:.2f}M params")