"""Code for the lecture-4 slide "GLU at the same cost": a GLU feed-forward layer with d_ff = (8/3) d has the same
parameters and FLOPs as the plain MLP with d_ff = 4 d.

Plain MLP:  FFN(x) = max(0, x W1) W2            W1: d x 4d, W2: 4d x d          -> 2 * d * 4d     = 8 d^2 parameters
GLU:        FFN(x) = (Swish(x W1) * x V) W2     W1, V: d x (8/3)d, W2: (8/3)d x d -> 3 * d * (8/3)d = 8 d^2 parameters
A matrix of p weights costs 2p FLOPs per token (a multiply and an add per weight), so both cost 16 d^2 FLOPs per token;
the gate adds one element-wise multiply, (8/3) d FLOPs. Shazeer ("GLU Variants Improve Transformer", 2020) shrinks d_ff
by 2/3 for exactly this reason: T5-base, d = 768, MLP d_ff = 3,072, GLU d_ff = 2,048.

Counts both layers in PyTorch (no biases, as in T5 and LLaMA): parameters from the modules, matrix-multiply FLOPs from
torch's FlopCounterMode, for T5-base and for a LLaMA-7B-sized layer (d = 4,096; LLaMA rounds (8/3) d = 10,923 up to
11,008, a multiple of 256, so its GLU has 0.8% more parameters than the 4d MLP).

Run:  python glu_params.py      (writes data/glu_params.json; the slide reads it)
"""
import json
from fractions import Fraction
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

OUT = Path(__file__).parent / "data" / "glu_params.json"


class MLP(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1, self.w2 = nn.Linear(d, d_ff, bias=False), nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class GLU(nn.Module):                                          # SwiGLU, as in LLaMA
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1, self.v = nn.Linear(d, d_ff, bias=False), nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.v(x))


def count(layer, d):
    with torch.device("meta"):                                 # shapes only, no memory or compute
        x = torch.empty(1, d)
        with FlopCounterMode(display=False) as fc:
            layer(x)
    return sum(p.numel() for p in layer.parameters()), fc.get_total_flops()


assert 3 * Fraction(8, 3) == 2 * 4                             # 3 matrices of d x (8/3)d = 2 matrices of d x 4d
res = []
for name, d, ff_mlp, ff_glu in [("T5-base", 768, 3072, 2048), ("LLaMA-7B-sized", 4096, 16384, 11008)]:
    with torch.device("meta"):
        mlp, glu = MLP(d, ff_mlp), GLU(d, ff_glu)
    (p_mlp, f_mlp), (p_glu, f_glu) = count(mlp, d), count(glu, d)
    assert p_mlp == 2 * d * ff_mlp and p_glu == 3 * d * ff_glu and f_mlp == 2 * p_mlp and f_glu == 2 * p_glu
    res.append({"model": name, "d": d, "d_ff_mlp": ff_mlp, "d_ff_glu": ff_glu, "params_mlp": p_mlp, "params_glu": p_glu,
                "flops_per_token_mlp": f_mlp, "flops_per_token_glu": f_glu, "gate_multiply_flops": ff_glu,
                "glu_over_mlp": p_glu / p_mlp})
    print(f"{name:15s} d={d:5d}  MLP d_ff={ff_mlp:6d}: {p_mlp:>12,} params, {f_mlp:>12,} FLOPs/token   "
          f"GLU d_ff={ff_glu:6d}: {p_glu:>12,} params, {f_glu:>12,} FLOPs/token (+{ff_glu:,} for the gate)  "
          f"ratio {p_glu / p_mlp:.4f}")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print("saved", OUT)
