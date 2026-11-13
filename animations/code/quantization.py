"""Code for lecture 5, Part 6 (fewer bytes per number): what quantization buys a memory-bound decoder, and why 8-bit
activations broke for LLMs, measured on GPT-2.

1. Bytes per weight. Decoding reads every weight once per step (inference_cost.py), so with the KV cache kept in BF16,
   Llama-2-13B on one H100 (3.35 TB/s) at batch 1 takes (weights + KV cache) / bandwidth per token, for weights in
   16, 8 and 4 bits.
2. Outliers, measured. GPT-2 small, the hidden state after block 6, for a 28-token sentence. One channel (447)
   reaches 2,873 on the first token, while a typical channel stays below 5 (the 'massive activations' of Sun et al.
   2024; Dettmers et al. 2022 found outlier features in every large model from 6.7B parameters on). Absmax INT8 with
   one scale for the whole tensor (scale = max |x| / 127) rounds almost every other value to zero. Finer scales fix
   it: one scale per token (row), as LLM.int8() does for activations, or one per block of 32 values, as the MX
   formats (MXFP4) do.
3. Published results used on the slides (not recomputed): GPTQ (Frantar et al., ICLR 2023, Table 5), OPT-175B WikiText2
   perplexity; gpt-oss-120b (OpenAI 2025 model card): 116.8B parameters, MoE weights in MXFP4 (4.25 bits), checkpoint
   60.8 GiB.

Run:  python quantization.py      (writes data/quantization.json; the slides read it; downloads GPT-2 small once)
"""
import json
from pathlib import Path

import torch
from transformers import GPT2Model, GPT2TokenizerFast

OUT = Path(__file__).parent / "data" / "quantization.json"
IC = json.loads((Path(__file__).parent / "data" / "inference_cost.json").read_text())
BW = IC["bandwidth"]

# ---------------------------------------------------------------- 1. bytes per weight -> time per token
r1 = next(r for r in IC["mha"] if r["B"] == 1)
params, kv = r1["params"], r1["kv_per_seq_gb"] * 1e9
formats = []
for name, bits in (("BF16", 16), ("FP8 / INT8", 8), ("INT4", 4)):
    w = params * bits / 8
    formats.append({"format": name, "bits": bits, "weights_gb": w / 1e9, "ms_per_token": (w + kv) / BW * 1e3})
    print(f"{name:10s} weights {w / 1e9:5.1f} GB -> {(w + kv) / BW * 1e3:4.1f} ms per token at batch 1")

# ---------------------------------------------------------------- 2. GPT-2's outlier channel and INT8
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2").eval()
TEXT = ("Deep learning models are trained on large datasets, and the cost of serving them depends on how many bytes "
        "each step has to read from memory.")
ids = tok(TEXT, return_tensors="pt").input_ids
with torch.no_grad():
    x = model(ids, output_hidden_states=True).hidden_states[6][0]           # (tokens, 768), after block 6
chan_max = x.abs().max(0).values
OUTLIER = int(chan_max.argmax())


def int8(v, dim=None):
    s = v.abs().max() / 127 if dim is None else v.abs().amax(dim, keepdim=True) / 127
    return torch.round(v / s) * s


def scores(q):
    return {"zero_fraction": float((q == 0).float().mean()), "rel_error": float((q - x).norm() / x.norm())}


blocks = x.reshape(x.shape[0], -1, 32)
res = {"tokens": ids.shape[1], "layer": 6, "outlier_channel": OUTLIER, "outlier_max": float(chan_max[OUTLIER]),
       "outlier_max_after_first_token": float(x[1:, OUTLIER].abs().max()),
       "median_channel_max": float(chan_max.median()), "channel_max": [round(float(v), 3) for v in chan_max],
       "one_scale": scores(int8(x)) | {"scale": float(x.abs().max() / 127)},
       "per_token": scores(int8(x, 1)),
       "per_block_32": scores(int8(blocks, 2).reshape(x.shape))}
print(f"GPT-2 layer 6: channel {OUTLIER} reaches {res['outlier_max']:.0f} (first token), "
      f"{res['outlier_max_after_first_token']:.0f} after it; median channel max {res['median_channel_max']:.2f}")
for k in ("one_scale", "per_token", "per_block_32"):
    v = res[k]
    print(f"  {k:12s}: {v['zero_fraction']:.1%} of values round to 0, error {v['rel_error']:.1%}")

gptq = {"source": "Frantar et al., GPTQ, Table 5 (WikiText2 perplexity)",
        "OPT-175B": {"FP16": 8.34, "RTN 4-bit": 10.54, "GPTQ 4-bit": 8.37, "RTN 3-bit": 7.3e3, "GPTQ 3-bit": 8.68},
        "BLOOM-176B": {"FP16": 8.11, "RTN 4-bit": 8.37, "GPTQ 4-bit": 8.21, "RTN 3-bit": 571, "GPTQ 3-bit": 8.64}}
gpt_oss = {"params": 116.8e9, "bf16_gb": 116.8e9 * 2 / 1e9, "checkpoint_gib": 60.8, "checkpoint_gb": 60.8 * 2 ** 30 / 1e9,
           "mxfp4_bits": 4.25, "source": "gpt-oss model card (arXiv 2508.10925), Sec. 2.1 and Table 1"}
print(f"gpt-oss-120b: BF16 {gpt_oss['bf16_gb']:.0f} GB vs released {gpt_oss['checkpoint_gb']:.1f} GB")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"model": IC["model"], "formats": formats, "gpt2": res, "gptq": gptq, "gpt_oss": gpt_oss},
                          indent=1))
print("saved", OUT)
