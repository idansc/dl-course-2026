"""Code for lecture 5 (compute & efficiency), parts 1 and 2: what one generated token costs, and how much each way of
sharing keys and values shrinks the KV cache.

Part 1 follows Stanford CS336 (2026), lecture 10 (inference): Llama-2-13B on one H100 (3.35 TB/s, 80 GB), BF16.
- Arithmetic intensity (FLOPs per byte moved) of the two stages, for S tokens of context and T new tokens:
    MLP: B T (every sequence reads the same weights);  attention: S T / (S + T) (every sequence has its own K, V).
    Prefill (T = S): MLP B S, attention S / 2.  Decoding (T = 1): MLP B, attention S / (S + 1) < 1.
- One decoding step reads every weight once and every sequence's KV cache once, so
    latency = (2 bytes x parameters + B x KV cache per sequence) / bandwidth,  throughput = B / latency.
Part 2: values cached per token per layer for a 64-head model with d_head = 128 (Llama-3-70B), and the cache at
128k tokens over its 80 layers: MHA (64 K, V heads), MQA (1), GQA (8), MLA (one 512-d latent + a 64-d RoPE key, as in
DeepSeek-V2/V3), and GQA with each K, V shared by two adjacent layers (cross-layer attention, CLA2).

Run:  python inference_cost.py      (writes data/inference_cost.json; the slides read it)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "inference_cost.json"
BW, HBM, BYTES = 3.35e12, 80e9, 2                              # H100 SXM: bytes/s, memory; BF16

# ---------------------------------------------------------------- part 1: Llama-2-13B (CS336 lecture 10's config)
S, D, F, N, H, L, V = 1024, 5120, 13824, 40, 128, 40, 32000


def stats(B, K):
    params = 2 * V * D + 3 * D * F * L + (2 * D * N * H + 2 * D * K * H) * L
    weights = BYTES * params
    kv_seq = S * K * H * L * 2 * BYTES                         # keys and values, every layer, S tokens
    memory = weights + B * kv_seq
    latency = memory / BW
    return {"B": B, "K": K, "params": params, "weights_gb": weights / 1e9, "kv_per_seq_gb": kv_seq / 1e9,
            "kv_gb": B * kv_seq / 1e9, "memory_gb": memory / 1e9, "fits": memory <= HBM,
            "latency_ms": latency * 1e3, "tokens_per_s": B / latency}


intensity = {"prefill_mlp": f"B·S = {S:,} at B = 1", "prefill_attention": S / 2,
             "decode_mlp": "B", "decode_attention": S / (S + 1), "h100_break_even": 989.4e12 / BW}
batches = [1, 16, 32, 64, 96, 128, 256]
mha = [stats(B, 40) for B in batches]
gqa = [stats(B, 8) for B in batches]
for name, rows in (("MHA, 40 K, V heads", mha), ("GQA, 8 K, V heads", gqa)):
    print(name)
    for r in rows:
        print(f"  B = {r['B']:3d}: weights {r['weights_gb']:5.1f} GB + KV {r['kv_gb']:6.1f} GB = {r['memory_gb']:6.1f} GB "
              f"({'fits' if r['fits'] else 'does not fit'} in 80 GB), {r['latency_ms']:5.1f} ms per step, "
              f"{r['tokens_per_s']:7,.0f} tokens/s")
print("intensity:", intensity)

# ---------------------------------------------------------------- part 2: values cached per token per layer
NH, DH, LAYERS, CTX = 64, 128, 80, 131072                      # Llama-3-70B: 64 heads of 128, 80 layers; 128k tokens
per_layer = {"MHA (2017)": 2 * NH * DH, "MQA (2019)": 2 * DH, "GQA-8 (2023)": 2 * 8 * DH,
             "MLA (2024)": 512 + 64, "GQA-8 + cross-layer (2024)": 2 * 8 * DH // 2}
kv_128k = {k: v * LAYERS * BYTES * CTX / 1e9 for k, v in per_layer.items()}
for k in per_layer:
    print(f"{k:28s} {per_layer[k]:6,} values per token per layer -> {kv_128k[k]:6.1f} GB at 128k tokens")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"model": "Llama-2-13B", "S": S, "bandwidth": BW, "hbm_bytes": HBM, "intensity": intensity,
                           "mha": mha, "gqa": gqa, "per_layer": per_layer, "kv_128k_gb": kv_128k,
                           "per_layer_model": "64 heads x 128, 80 layers (Llama-3-70B), BF16, 131,072 tokens"},
                          indent=1))
print("saved", OUT)
