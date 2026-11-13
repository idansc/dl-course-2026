"""Code for lecture 5, Parts 7 and 8: what a token stores and what it uses, for a mixture of experts and a looped
transformer, counted from the published configurations.

DeepSeek-V3 (technical report, Sec. 4.2): 61 layers, d = 7,168; the first 3 layers dense (MLP width 18,432), the other
58 are MoE layers with 1 shared and 256 routed experts of width 2,048, 8 routed experts per token; MLA attention with
128 heads of 128 (+ a 64-d RoPE key), query compression 1,536, KV compression 512; vocabulary 129,280 (input and
output embeddings untied). Counting every matrix gives the report's 671B total and 37B activated parameters (the
multi-token-prediction module is not counted, as in the report).
In a batch of B tokens each routed expert sees on average B k / E of them (k = 8 of E = 256: B / 32): the batch that
lifts a dense MLP's arithmetic intensity to B lifts each expert's only to B / 32.
Huginn-0125 (Geiping et al. 2025; config.json on Hugging Face): d = 5,280, 55 heads of 96, gated MLP of width 17,920,
2 prelude + 4 recurrent + 2 coda layers, tied 65,536-token embeddings, and a linear adapter (2d -> d) that injects the
input at every iteration. Stored parameters: all of these once. Used per token at r iterations: prelude, coda and output
layer once, the core and adapter r times; r = 32 (its training mean) uses as many FLOPs as a dense model of ~54B
parameters (the paper: 'a computation load equivalent to 50 billion parameters').

Run:  python experts_and_loops.py      (writes data/experts_and_loops.json; the slides read it)
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "data" / "experts_and_loops.json"

# ---------------------------------------------------------------- DeepSeek-V3
d, L, DENSE, V = 7168, 61, 3, 129280
ff_dense, ff_exp, E, K, SHARED = 18432, 2048, 256, 8, 1
H_, DH, DR, QC, KVC = 128, 128, 64, 1536, 512
mla = d * QC + QC * H_ * (DH + DR) + d * (KVC + DR) + KVC * H_ * (DH + DH) + H_ * DH * d   # q down/up, kv down/up, out
mlp = lambda f: 3 * d * f                                      # gated MLP: gate, up, down
moe_total = (SHARED + E) * mlp(ff_exp) + E * d                 # experts + router
moe_active = (SHARED + K) * mlp(ff_exp) + E * d
emb = 2 * V * d
total = L * mla + DENSE * mlp(ff_dense) + (L - DENSE) * moe_total + emb
active = L * mla + DENSE * mlp(ff_dense) + (L - DENSE) * moe_active + emb
print(f"DeepSeek-V3: {total / 1e9:.1f}B parameters, {active / 1e9:.1f}B per token ({active / total:.1%}); "
      f"routed experts hold {(L - DENSE) * E * mlp(ff_exp) / total:.0%} of all parameters; each expert sees B/{E // K}")

# ---------------------------------------------------------------- Huginn-0125
dh, ffh, VH = 5280, 17920, 65536
layer = 4 * dh * dh + 3 * dh * ffh                             # attention (q, k, v, out) + gated MLP
adapter = 2 * dh * dh
stored = 8 * layer + adapter + VH * dh
used = lambda r: 4 * layer + r * (4 * layer + adapter) + VH * dh        # parameters touched per token (FLOPs / 2)
print(f"Huginn: {stored / 1e9:.2f}B stored; per token at r = 32: {used(32) / 1e9:.1f}B parameter-uses "
      f"(= FLOPs of a {used(32) / 1e9:.0f}B dense model); at r = 4: {used(4) / 1e9:.1f}B")

stored_used = [{"model": "Llama-2-13B (dense)", "stored": 13.0e9, "used": 13.0e9},
               {"model": "DeepSeek-V3 (MoE)", "stored": total, "used": active},
               {"model": "Huginn-3.5B, r = 32 (looped)", "stored": stored, "used": used(32)}]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"deepseek_v3": {"total": total, "active": active, "experts": E, "active_experts": K,
                                           "shared": SHARED, "tokens_per_expert_fraction": K / E,
                                           "routed_share": (L - DENSE) * E * mlp(ff_exp) / total},
                           "huginn": {"stored": stored, "used_r": {r: used(r) for r in (1, 4, 8, 16, 32, 64)}},
                           "stored_used": stored_used}, indent=1))
print("saved", OUT)
