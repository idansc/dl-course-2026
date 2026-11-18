"""Code for lecture 6 (compute & efficiency), Part 9: serving, more tokens per read of the weights.

The running example of Part 1 (Stanford CS336 (2026), lecture 10): Llama-2-13B in BF16 on one H100 SXM
(3.35 TB/s, 80 GB, 989.4 dense BF16 TFLOP/s), 1,024 tokens of context per sequence.

1. One decoding step on the chip: reading the weights takes 7.8 ms; one token's math takes 0.026 ms. Up to
   989.4e12 / 3.35e12 = 295 tokens can share one read before the math becomes the bottleneck.
2. Static vs continuous batching (Orca, OSDI 2022): 8 requests, 4 slots, the schedule of each.
3. Paged KV cache (vLLM, SOSP 2023): how many sequences fit when each one reserves its maximum length (2,048 tokens)
   vs pages filled as it grows (512 tokens on average, 16 tokens per page), and the tokens/s of each batch.
4. Speculative decoding (Leviathan et al., ICML 2023; Chen et al., 2023): a drafter proposes k tokens, the model
   checks them in one read. With acceptance rate a, a check yields (1 - a^(k+1)) / (1 - a) tokens on average, and the
   time improves by that / (1 + c k), c = drafter cost / model cost (Leviathan et al., Theorem 3.8 and Eq. 1).
   Checked here by simulation: (a) the accepted tokens follow the model's distribution exactly (rejection sampling,
   random distributions, 10^6 draws); (b) the tokens per check match the formula.
5. Prefill on the decode GPU: an 8,000-token prompt costs 2 x 13B x 8,000 FLOPs, about 0.21 s at peak, during which
   no sequence in the batch gets its next token; its KV cache (6.6 GB) is what disaggregated serving ships.

Run:  python serving_cost.py      (writes data/serving_cost.json; the slides read it)
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data" / "serving_cost.json"
BW, HBM, PEAK, BYTES = 3.35e12, 80e9, 989.4e12, 2               # H100 SXM; BF16
PARAMS = 13_015_449_600                                         # Llama-2-13B, as in inference_cost.py
L, KVH, DH, S = 40, 40, 128, 1024                               # layers, K/V heads, head size, context
KV_TOK = 2 * L * KVH * DH * BYTES                               # bytes of keys and values per token: 819,200
W = PARAMS * BYTES

# ---------------------------------------------------------------- 1. one decoding step on the chip
read_ms = W / BW * 1e3
kv_ms = S * KV_TOK / BW * 1e3
tok_ms = 2 * PARAMS / PEAK * 1e3
chip = {"weights_gb": W / 1e9, "read_ms": read_ms, "kv_ms": kv_ms, "step_ms": read_ms + kv_ms, "token_math_ms": tok_ms,
        "ridge_tokens": PEAK / BW, "busy_pct_b1": 100 * tok_ms / (read_ms + kv_ms),
        "math_ms": {b: b * tok_ms for b in (1, 16, 64, 295)}}
print(f"chip: read {read_ms:.2f} ms (+{kv_ms:.2f} KV), one token's math {tok_ms:.4f} ms, ridge {PEAK / BW:.0f} tokens, "
      f"cores busy {chip['busy_pct_b1']:.2f}% at batch 1")

# ---------------------------------------------------------------- 2. static vs continuous batching
LENS, SLOTS = [3, 9, 5, 7, 2, 8, 4, 6], 4                       # tokens each request generates, in arrival order


def schedule(lens, slots, continuous):
    """(request, slot, first step, last step + 1) for each request; static batches start together and wait for the
    longest; continuous batching refills a slot on the step after it frees."""
    out, free, queue, t = [], [0] * slots, list(range(len(lens))), 0
    if not continuous:
        for b in range(0, len(lens), slots):
            group = queue[b:b + slots]
            for j, r in enumerate(group):
                out.append((r, j, t, t + lens[r]))
            t += max(lens[r] for r in group)
        return out, t
    for r in queue:
        j = min(range(slots), key=lambda i: free[i])
        out.append((r, j, free[j], free[j] + lens[r]))
        free[j] += lens[r]
    return out, max(free)


batching = {}
for name, cont in (("static", False), ("continuous", True)):
    sch, steps = schedule(LENS, SLOTS, cont)
    batching[name] = {"schedule": sch, "steps": steps, "busy_pct": 100 * sum(LENS) / (steps * SLOTS)}
    print(f"{name:10s}: {steps} steps for {sum(LENS)} tokens, slots busy {batching[name]['busy_pct']:.0f}%")

# ---------------------------------------------------------------- 3. paged KV cache
MAXLEN, MEAN, PAGE = 2048, 512, 16
free_b = HBM - W


def step(B, tokens_read):
    t = (W + B * tokens_read * KV_TOK) / BW
    return {"B": B, "step_ms": t * 1e3, "tokens_per_s": B / t}


n_res = int(free_b // (MAXLEN * KV_TOK))
n_pag = int(free_b // ((MEAN + PAGE / 2) * KV_TOK))             # half a page unused per sequence, on average
paged = {"kv_mb_per_token": KV_TOK / 1e6, "free_gb": free_b / 1e9, "maxlen": MAXLEN, "mean": MEAN, "page": PAGE,
         "reserved": {"n": n_res, "gb_per_seq": MAXLEN * KV_TOK / 1e9, "used_pct": 100 * MEAN / MAXLEN,
                      **step(n_res, MEAN)},
         "paged": {"n": n_pag, "gb_per_seq": (MEAN + PAGE / 2) * KV_TOK / 1e9, "used_pct": 100 * MEAN / (MEAN + PAGE / 2),
                   **step(n_pag, MEAN)},
         "prefix": {"tokens": 1000, "requests": 64, "copies_gb": 64 * 1000 * KV_TOK / 1e9, "shared_gb": 1000 * KV_TOK / 1e9}}
print(f"paged: reserved {n_res} sequences ({paged['reserved']['tokens_per_s']:,.0f} tok/s), paged {n_pag} "
      f"({paged['paged']['tokens_per_s']:,.0f} tok/s); a 1,000-token prompt for 64 requests: "
      f"{paged['prefix']['copies_gb']:.1f} GB copied vs {paged['prefix']['shared_gb']:.2f} GB shared")

# ---------------------------------------------------------------- 4. speculative decoding
DRAFT = 1.1e9                                                   # a 1.1B drafter with the same tokenizer
C = DRAFT / PARAMS
E = lambda a, k: (1 - a ** (k + 1)) / (1 - a)
SPEED = lambda a, k, c=C: E(a, k) / (1 + c * k)
ALPHAS, KS = (0.6, 0.7, 0.8, 0.9), list(range(0, 13))
spec = {"c": C, "drafter_params": DRAFT, "draft_ms": DRAFT * BYTES / BW * 1e3,
        "tokens": {str(a): [E(a, k) for k in KS] for a in ALPHAS},
        "speedup": {str(a): [SPEED(a, k) for k in KS] for a in ALPHAS}, "ks": KS,
        "best": {str(a): max(KS, key=lambda k: SPEED(a, k)) for a in ALPHAS}}
spec["best_speedup"] = {a: SPEED(float(a), k) for a, k in spec["best"].items()}
a0, k0 = 0.8, 4
rnd = k0 * spec["draft_ms"] + chip["step_ms"]
spec["example"] = {"alpha": a0, "k": k0, "tokens": E(a0, k0), "round_ms": rnd, "ms_per_token": rnd / E(a0, k0),
                   "plain_ms_per_token": chip["step_ms"], "speedup": chip["step_ms"] / (rnd / E(a0, k0))}
print(f"speculative, c = {C:.3f}: best k {spec['best']}, speedups "
      + ", ".join(f"{a}: {v:.2f}x" for a, v in spec["best_speedup"].items()))
ex = spec["example"]
print(f"  example a = {a0}, k = {k0}: {ex['tokens']:.2f} tokens per check, {ex['round_ms']:.1f} ms per round, "
      f"{ex['ms_per_token']:.2f} vs {ex['plain_ms_per_token']:.2f} ms per token: {ex['speedup']:.2f}x")

# the worked example (two tokens): drafter p, model q
p, q = np.array([0.8, 0.2]), np.array([0.6, 0.4])
keep = np.minimum(1, q / p)
resid = np.maximum(q - p, 0) / np.maximum(q - p, 0).sum()
final = p * keep + (1 - (p * keep).sum()) * resid
spec["ab"] = {"p": p.tolist(), "q": q.tolist(), "keep": keep.tolist(), "rejected": float(1 - (p * keep).sum()),
              "resid": resid.tolist(), "final": final.tolist()}
print("  A, B: keep", keep, "rejected mass", round(1 - (p * keep).sum(), 3), "-> final", final)

# (a) exactness by simulation: random p, q over 10 tokens, one speculative step, 10^6 draws
rng = np.random.default_rng(0)
tv = []
for trial in range(5):
    p, q = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))
    n = 1_000_000
    x = rng.choice(10, n, p=p)
    acc = rng.random(n) < np.minimum(1, q[x] / p[x])
    r = np.maximum(q - p, 0)
    y = np.where(acc, x, rng.choice(10, n, p=r / r.sum()))
    emp = np.bincount(y, minlength=10) / n
    tv.append(0.5 * np.abs(emp - q).sum())
spec["exact_tv"] = max(tv)
spec["draft_tv"] = 0.5 * float(np.abs(p - q).sum())
print(f"  exactness: total-variation distance to the model's distribution <= {max(tv):.4f} over 5 random pairs "
      f"(the drafter alone is {spec['draft_tv']:.2f} away)")

# (b) tokens per check by simulation: each draft accepted with probability a until the first rejection
sim = {}
for a in ALPHAS:
    acc = rng.random((200_000, k0)) < a
    first_rej = np.where(acc.all(1), k0, acc.argmin(1))         # accepted drafts before the first rejection
    sim[str(a)] = float((first_rej + 1).mean())                 # + the model's own token (correction or bonus)
spec["sim_tokens_k4"] = sim
print("  tokens per check at k = 4, simulated vs formula:",
      ", ".join(f"{a}: {sim[str(a)]:.3f} / {E(a, k0):.3f}" for a in ALPHAS))

# ---------------------------------------------------------------- 5. prefill on the decode GPU
P_TOK = 8000
pre_s = 2 * PARAMS * P_TOK / PEAK
prefill = {"tokens": P_TOK, "tflop": 2 * PARAMS * P_TOK / 1e12, "seconds_at_peak": pre_s,
           "decode_steps_lost": pre_s * 1e3 / chip["step_ms"], "kv_gb": P_TOK * KV_TOK / 1e9,
           "ship_s_400gbps": P_TOK * KV_TOK / 50e9, "ship_ms_nvlink": P_TOK * KV_TOK / 900e9 * 1e3}
print(f"prefill of {P_TOK:,} tokens: {prefill['tflop']:.0f} TFLOP, {pre_s:.2f} s at peak = "
      f"{prefill['decode_steps_lost']:.0f} decode steps; its KV cache {prefill['kv_gb']:.1f} GB, "
      f"{prefill['ship_s_400gbps']:.2f} s over 400 Gb/s, {prefill['ship_ms_nvlink']:.0f} ms over NVLink")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"model": "Llama-2-13B", "chip": chip, "batching": batching, "lens": LENS, "slots": SLOTS,
                           "paged": paged, "spec": spec, "prefill": prefill}, indent=1))
print("wrote", OUT)
