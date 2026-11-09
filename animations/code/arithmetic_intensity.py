"""Code for the lecture-4 slide "Arithmetic intensity": FLOPs per byte moved to or from memory, for the slide's three
operations, and the break-even point of an H100.

Arithmetic intensity = FLOPs / bytes read from or written to memory, as Stanford CS336 (2026, lecture 2) and the
Roofline paper count it. Numbers are BF16, 2 bytes each.
1. x + y, two vectors of n numbers: n FLOPs; read 2n numbers, write n: 6n bytes                  -> 1/6
2. W x, W of shape d x d, one token: 2 d^2 FLOPs (a multiply and an add per weight);
   read d^2 + d numbers, write d: 2 (d^2 + 2d) bytes                                             -> d / (d + 2), about 1
3. W X, X holds n tokens: 2 n d^2 FLOPs; read d^2 + n d numbers, write n d                       -> n d / (d + 2n)
   (1,365 for n = d = 4,096)
H100 SXM: 989 TFLOP/s (BF16, dense) and 3.35 TB/s of memory bandwidth. Computation time = FLOPs / 989e12 s and
communication time = bytes / 3.35e12 s; the operation is memory-bound when communication takes longer, which is exactly
when its intensity is below 989 / 3.35 = 295 FLOPs per byte (the script checks that the two tests agree). Papers that
count numbers (words) instead of bytes, such as Ivanov et al. (MLSys 2021) and Shazeer (2019), get twice these values
and a break-even of 590. Decoding with a batch of b sequences is case 3 with n = b: about b FLOPs per byte, so it needs
b of a few hundred.

Run:  python arithmetic_intensity.py      (writes data/arithmetic_intensity.json; the slide reads it)
"""
import json
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).parent / "data" / "arithmetic_intensity.json"
D = N = 4096                                                   # model width and tokens (a LLaMA-7B-sized layer)
TFLOPS, TBPS, BYTES = 989.4, 3.35, 2                           # H100 SXM datasheet: BF16 dense, HBM3; BF16 = 2 bytes

break_even = TFLOPS / TBPS                                     # FLOPs per byte at which both times are equal
ops = []
for name, flops, read, write in [("x + y", N, 2 * N, N),
                                 ("W x (one token)", 2 * D * D, D * D + D, D),
                                 (f"W X ({N:,} tokens)", 2 * N * D * D, D * D + N * D, N * D)]:
    moved = BYTES * (read + write)
    ai = Fraction(flops, moved)
    t_compute, t_communicate = flops / (TFLOPS * 1e12), moved / (TBPS * 1e12)
    bound = "memory" if t_communicate > t_compute else "compute"
    assert (bound == "memory") == (ai < break_even)            # the time test and the intensity test agree
    ops.append({"op": name, "flops": flops, "numbers_read": read, "numbers_written": write, "bytes_moved": moved,
                "intensity": float(ai), "intensity_exact": str(ai), "intensity_per_number": float(ai * BYTES),
                "computation_time_s": t_compute, "communication_time_s": t_communicate, "bound": bound})
    print(f"{name:20s} FLOPs {flops:>16,}  bytes moved {moved:>12,}  intensity {float(ai):9.4f} ({ai})  "
          f"computation {t_compute * 1e6:9.3f} us  communication {t_communicate * 1e6:9.3f} us  -> {bound}-bound")
print(f"H100: {TFLOPS} TFLOP/s / {TBPS} TB/s = {break_even:.1f} FLOPs per byte "
      f"({break_even * BYTES:.0f} per BF16 number)")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"gpu": "H100 SXM", "tflops_bf16_dense": TFLOPS, "tb_per_s": TBPS, "bytes_per_number": BYTES,
                           "break_even": break_even, "break_even_per_number": break_even * BYTES, "d": D, "n": N,
                           "ops": ops}, indent=1))
print("saved", OUT)
