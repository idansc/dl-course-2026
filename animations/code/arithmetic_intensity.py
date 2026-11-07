"""Code for the lecture-4 slide "Arithmetic intensity": FLOPs per number moved to or from memory, for the slide's three
operations, and the break-even point of an H100.

Arithmetic intensity = FLOPs / numbers read from or written to memory. One number is one BF16 value (2 bytes). The
slide counts numbers, not bytes, as Ivanov et al. (MLSys 2021) and Shazeer (2019) do, so its values can be compared
with the slides that follow it (the residual add: 1/3, LayerNorm: 3.5, attention's matrix multiplies: 153).
1. x + y, two vectors of n numbers: n FLOPs; read 2n numbers, write n                          -> 1/3
2. W x, W of shape d x d, one token: 2 d^2 FLOPs (a multiply and an add per weight);
   read d^2 + d numbers, write d                                                              -> about 2
3. W X, X holds n tokens: 2 n d^2 FLOPs; read d^2 + n d numbers, write n d                     -> 2 n d / (d + 2 n)
   (2,731 for n = d = 4,096)
H100 SXM: 989 TFLOP/s (BF16, dense) and 3.35 TB/s = 1.675 T BF16 numbers per second: 590 FLOPs per number moved. An
operation far below that leaves the cores waiting for memory (memory-bound); above it, the cores are the limit
(compute-bound). Decoding with a batch of b sequences is case 3 with n = b: about 2b, so it needs b of a few hundred.

Run:  python arithmetic_intensity.py      (writes data/arithmetic_intensity.json; the slide reads it)
"""
import json
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).parent / "data" / "arithmetic_intensity.json"
D = N = 4096                                                   # model width and tokens (a LLaMA-7B-sized layer)
TFLOPS, TBPS, BYTES = 989.4, 3.35, 2                           # H100 SXM datasheet: BF16 dense, HBM3; BF16 = 2 bytes

numbers_per_s = TBPS * 1e12 / BYTES
break_even = TFLOPS * 1e12 / numbers_per_s
ops = []
for name, flops, read, write in [("x + y", N, 2 * N, N),
                                 ("W x (one token)", 2 * D * D, D * D + D, D),
                                 (f"W X ({N:,} tokens)", 2 * N * D * D, D * D + N * D, N * D)]:
    ai = Fraction(flops, read + write)
    ops.append({"op": name, "flops": flops, "numbers_read": read, "numbers_written": write,
                "intensity": float(ai), "intensity_exact": str(ai),
                "bound": "memory" if ai < break_even else "compute",
                "fraction_of_peak": round(min(1.0, float(ai) / break_even), 5)})
    print(f"{name:20s} FLOPs {flops:>16,}  numbers moved {read + write:>12,}  intensity {float(ai):8.3f} ({ai})  "
          f"-> {ops[-1]['bound']}-bound, at most {100 * ops[-1]['fraction_of_peak']:.2f}% of peak")
print(f"H100: {TFLOPS} TFLOP/s / {numbers_per_s / 1e12:.3f} T numbers/s = {break_even:.1f} FLOPs per number moved")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"gpu": "H100 SXM", "tflops_bf16_dense": TFLOPS, "tb_per_s": TBPS, "bytes_per_number": BYTES,
                           "numbers_per_s": numbers_per_s, "break_even": break_even, "d": D, "n": N, "ops": ops},
                          indent=1))
print("saved", OUT)
