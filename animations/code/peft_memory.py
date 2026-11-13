"""Code for lecture 5, Part 6 (fewer bytes: to serve and to fine-tune): what fine-tuning costs in memory, and what
LoRA and QLoRA save, counted the way Part 1 counts (Stanford CS336 2026, lecture 2: 12 bytes per trained parameter for
AdamW in mixed precision, 2 bytes per frozen BF16 parameter).

- A 65B model (LLaMA-65B, the QLoRA paper's case): serving 2 bytes per parameter; full fine-tuning 12 bytes per parameter
  (weights 2 + gradients 2 + Adam's two FP32 moments 8), the paper's "more than 780 GB"; a frozen 16-bit copy (LoRA) and
  a frozen 4-bit copy (QLoRA, 0.5 bytes per parameter before the scales). Adapters and activations come on top.
- One fine-tuned copy per task: 1,000 tasks or customers, each with its own full copy.
- LoRA on one d x d matrix: W x + B A x with A of shape r x d and B of shape d x r: 2 d r trainable weights instead of
  d^2 (d = 4,096, r = 16).
- Published results used on the slides (not recomputed): LoRA (Hu et al., ICLR 2022) on GPT-3 175B: 10,000x fewer
  trainable parameters, training memory 1.2 TB -> 350 GB, a checkpoint per task 350 GB -> 35 MB; QLoRA (Dettmers et al.,
  NeurIPS 2023): fine-tuning a 65B model in < 48 GB.

Run:  python peft_memory.py      (writes data/peft_memory.json; the slides read it)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "peft_memory.json"
N = 65e9
BYTES = {"serve (BF16)": 2, "full fine-tuning": 2 + 2 + 4 + 4, "frozen BF16": 2, "frozen 4-bit": 0.5}
mem = {k: N * b / 1e9 for k, b in BYTES.items()}
for k, v in mem.items():
    print(f"65B, {k:18s}: {v:6.1f} GB")
tasks = 1000
print(f"{tasks:,} fine-tuned copies at {mem['serve (BF16)']:.0f} GB each: {tasks * mem['serve (BF16)'] / 1e3:,.0f} TB")
d, r = 4096, 16
full, lora = d * d, 2 * d * r
print(f"LoRA, d = {d:,}, r = {r}: {lora:,} trainable weights vs {full:,} ({lora / full:.2%})")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"params": N, "bytes_per_param": BYTES, "gb": mem, "tasks": tasks,
                           "tasks_tb": tasks * mem["serve (BF16)"] / 1e3,
                           "lora": {"d": d, "r": r, "full": full, "lora": lora, "fraction": lora / full},
                           "lora_paper": {"trainable_reduction": 10000, "vram_tb_before": 1.2, "vram_gb_after": 350,
                                          "checkpoint_gb": 350, "checkpoint_mb": 35},
                           "qlora_paper": {"full_gb": 780, "qlora_gb_max": 48}}, indent=1))
print("saved", OUT)
