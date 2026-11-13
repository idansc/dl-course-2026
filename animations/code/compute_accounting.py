"""Code for lecture 5, Part 1 (counting FLOPs and bytes): the napkin math of Stanford CS336 (2026) lecture 2, Meta's
reported GPU-hours for Llama 3.1, memory per parameter, and the points of the roofline slide.

FLOPs. In y = W x every weight does one multiply and one add per token: 2 FLOPs per parameter per token (forward).
The backward pass computes two products of the same size, the gradient for the activations and the gradient for the
weights: 4 FLOPs per parameter per token. Training therefore costs about 6 N D FLOPs (N parameters, D tokens); this
ignores attention's own FLOPs, which are small for short contexts (CS336 lecture 2).
- CS336's question: training a 70B model on 15T tokens on 1,024 H100s at half their peak (989.4 TFLOP/s BF16 dense).
- Llama 3.1 (model card): pretrained on ~15T tokens; 30.84M H100-hours for the 405B model. At full peak the 6 N D FLOPs
  would take 10.2M H100-hours: the run used about a third of the GPUs' peak (failures, communication, memory-bound
  kernels; the card's hours may include more than the main run).
Memory. Training with AdamW in mixed precision keeps, per parameter, BF16 weights (2 bytes), BF16 gradients (2) and two
FP32 moments (4 + 4): 12 bytes (CS336 2026, lecture 2; activations come on top). Serving keeps only the BF16 weights
(2 bytes), plus the KV cache. Eight 80 GB H100s: 53B parameters for training, 320B for serving.
Roofline. Attainable FLOP/s = min(peak, intensity x bandwidth), H100 SXM: 989.4 TFLOP/s and 3.35 TB/s, so the ridge is
at 295 FLOPs per byte. Points: an element-wise add (1/6), and the weight multiply W X of a Llama-2-13B layer
(d = 5,120) with n tokens at once: intensity n d / (d + 2 n) (arithmetic_intensity.py). Decoding with one sequence is
n = 1; with a batch of 64, n = 64; a 1,024-token prompt (prefill) is n = 1,024; a training micro-batch, n = 4,096.

Run:  python compute_accounting.py      (writes data/compute_accounting.json; the slides read it)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "compute_accounting.json"
PEAK, BW, HBM = 989.4e12, 3.35e12, 80e9                        # H100 SXM: BF16 dense FLOP/s, bytes/s, bytes

# ---------------------------------------------------------------- FLOPs
flops_per_param_token = {"forward": 2, "backward": 4, "training": 6}
N70, D15 = 70e9, 15e12
c70 = 6 * N70 * D15
days_1024 = c70 / (PEAK * 0.5 * 1024 * 86400)                  # CS336 lecture 2: 1,024 H100s at half the peak
N405 = 405e9
c405 = 6 * N405 * D15
hours_peak_405 = c405 / PEAK / 3600
reported_405 = 30.84e6                                         # Llama 3.1 model card, H100-80GB hours
print(f"6 N D for 70B on 15T tokens: {c70:.2e} FLOPs -> {days_1024:.0f} days on 1,024 H100s at half the peak")
print(f"405B on 15T: {c405:.2e} FLOPs -> {hours_peak_405 / 1e6:.1f}M H100-hours at peak; reported "
      f"{reported_405 / 1e6:.2f}M -> {hours_peak_405 / reported_405:.0%} of peak")

# ---------------------------------------------------------------- memory per parameter
train_bytes = {"weights (BF16)": 2, "gradients (BF16)": 2, "Adam first moment (FP32)": 4, "Adam second moment (FP32)": 4}
serve_bytes = {"weights (BF16)": 2}
B_TRAIN, B_SERVE = sum(train_bytes.values()), sum(serve_bytes.values())
max_train, max_serve = 8 * HBM / B_TRAIN, 8 * HBM / B_SERVE
print(f"bytes per parameter: training {B_TRAIN}, serving {B_SERVE}; on 8 H100s: train {max_train / 1e9:.0f}B, "
      f"serve {max_serve / 1e9:.0f}B parameters (activations and KV cache not counted)")

# ---------------------------------------------------------------- roofline
ridge = PEAK / BW
D = 5120                                                       # Llama-2-13B
ai_wx = lambda n: n * D / (D + 2 * n)                          # FLOPs per byte of W X, BF16 (arithmetic_intensity.py)
points = [("element-wise add", 1 / 6), ("decode, 1 sequence", ai_wx(1)), ("decode, batch of 64", ai_wx(64)),
          ("prefill, 1,024 tokens", ai_wx(1024)), ("training, 4,096 tokens", ai_wx(4096))]
roof = []
for name, ai in points:
    att = min(PEAK, ai * BW)
    roof.append({"name": name, "intensity": ai, "attainable_tflops": att / 1e12, "fraction_of_peak": att / PEAK,
                 "bound": "memory" if ai < ridge else "compute"})
    print(f"{name:24s} {ai:9.2f} FLOPs/byte -> {att / 1e12:7.1f} TFLOP/s ({att / PEAK:6.1%} of peak), "
          f"{'memory' if ai < ridge else 'compute'}-bound")
print(f"ridge: {ridge:.0f} FLOPs per byte")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"peak_flops": PEAK, "bandwidth": BW, "hbm_bytes": HBM, "ridge": ridge,
                           "flops_per_param_token": flops_per_param_token,
                           "napkin_70b": {"params": N70, "tokens": D15, "flops": c70, "gpus": 1024, "fraction_of_peak": 0.5,
                                          "days": days_1024},
                           "llama31_405b": {"params": N405, "tokens": D15, "flops": c405,
                                            "h100_hours_at_peak": hours_peak_405, "h100_hours_reported": reported_405,
                                            "fraction_of_peak": hours_peak_405 / reported_405},
                           "bytes_per_param": {"training": train_bytes, "serving": serve_bytes},
                           "max_params_8xh100": {"training": max_train, "serving": max_serve},
                           "roofline": roof}, indent=1))
print("saved", OUT)
