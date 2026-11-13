"""Code for lecture 5, Part 2 (scaling laws): Kaplan's power laws, Chinchilla's IsoFLOP curves, and the shift from
train-optimal to inference-optimal model sizes. Everything is computed from published fitted laws (no new training).

1. Kaplan et al. 2020, Eqs. 1.1-1.3: L(N) = (8.8e13 / N)^0.076 (non-embedding parameters), L(D) = (5.4e13 / D)^0.095
   (tokens), L(C) = (3.1e8 / C)^0.050 (C in PF-days, 1 PF-day = 8.64e19 FLOPs). Lines over the ranges they fit:
   768 to 1.5B parameters, 22M to 23B tokens, eight orders of magnitude of compute.
2. Chinchilla's parametric loss L(N, D) = E + A / N^alpha + B / D^beta, with the re-fit of Besiroglu et al. 2024
   (E = 1.8172, A = 482.01, B = 2085.43, alpha = 0.3478, beta = 0.3658), which agrees with Chinchilla's approaches 1 and 2
   (about 20 tokens per parameter); Hoffmann et al.'s own published fit implies about 70. IsoFLOP curves: for a budget
   C = 6 N D, the loss as a function of N (D = C / 6N). The minimum of each curve is the compute-optimal model:
   N_opt = G (C/6)^a, a = beta / (alpha + beta), G = (alpha A / (beta B))^(1 / (alpha + beta)).
   Gopher (280B parameters, 300B tokens) and Chinchilla (70B, 1.4T) used the same compute, 5.76e23 FLOPs.
3. Inference-optimal (Sardana, Portes, Doubov and Frankle, ICML 2024): reach a target loss with the fewest total FLOPs,
   training (6 N D_train) plus serving (2 N per generated or read token, D_inf tokens over the model's life). Target: the
   loss of the compute-optimal 13B model under the same law. For each N, D_train(N) solves L(N, D_train) = target.
4. Tokens per parameter of well-known models (CS336 2026 lecture 9's list; model cards; Qwen3 report: all Qwen3 models
   trained on 36T tokens).

Run:  python scaling_laws.py      (writes data/scaling_laws.json; the slides read it)
"""
import json
import math
from pathlib import Path

OUT = Path(__file__).parent / "data" / "scaling_laws.json"
PFDAY = 8.64e19

# ---------------------------------------------------------------- 1. Kaplan's power laws
kaplan = {"N": {"c": 8.8e13, "alpha": 0.076, "range": [768, 1.5e9]},
          "D": {"c": 5.4e13, "alpha": 0.095, "range": [2.2e7, 2.3e10]},
          "C": {"c": 3.1e8, "alpha": 0.050, "range": [1e-8, 1e0]}}                 # C in PF-days
for k, v in kaplan.items():
    lo, hi = v["range"]
    v["loss_at_range"] = [(v["c"] / lo) ** v["alpha"], (v["c"] / hi) ** v["alpha"]]
    v["loss_factor_per_10x"] = 10 ** -v["alpha"]
    print(f"Kaplan L({k}): {v['loss_at_range'][0]:.2f} -> {v['loss_at_range'][1]:.2f}; x10 multiplies the loss by "
          f"{v['loss_factor_per_10x']:.3f}")

# ---------------------------------------------------------------- 2. Chinchilla (Besiroglu et al. re-fit)
E, A, B, AL, BE = 1.8172, 482.01, 2085.43, 0.3478, 0.3658
L = lambda n, d: E + A / n ** AL + B / d ** BE
a_exp = BE / (AL + BE)
G = (AL * A / (BE * B)) ** (1 / (AL + BE))
n_opt = lambda c: G * (c / 6) ** a_exp
BUDGETS = [1e19, 1e20, 1e21, 1e22, 1e23, 5.76e23]
iso = []
for c in BUDGETS:
    ns = [10 ** (7 + 0.05 * i) for i in range(int((12.3 - 7) / 0.05) + 1)]
    curve = [(n, L(n, c / (6 * n))) for n in ns if c / (6 * n) >= 1e8]
    no = n_opt(c)
    iso.append({"C": c, "curve": curve, "n_opt": no, "d_opt": c / (6 * no), "tokens_per_param": c / (6 * no) / no,
                "loss_opt": L(no, c / (6 * no))})
    print(f"C = {c:.2e}: N_opt {no / 1e9:7.2f}B, D_opt {c / (6 * no) / 1e9:9.1f}B tokens "
          f"({c / (6 * no) / no:4.1f} tokens/param), loss {L(no, c / (6 * no)):.3f}")
C_G = 5.76e23
gopher = {"N": 280e9, "D": 300e9, "loss": L(280e9, 300e9)}
chinchilla = {"N": 70e9, "D": 1.4e12, "loss": L(70e9, 1.4e12)}
print(f"same compute: Gopher 280B/300B loss {gopher['loss']:.3f}; Chinchilla 70B/1.4T loss {chinchilla['loss']:.3f}")

# ---------------------------------------------------------------- 3. train- vs inference-optimal
c13 = 6 * (13e9 / G) ** (1 / a_exp)                            # the budget whose optimum is 13B
d13 = c13 / 6 / 13e9
target = L(13e9, d13)
d_train = lambda n: (B / (target - E - A / n ** AL)) ** (1 / BE) if target - E - A / n ** AL > 0 else math.inf
DEMANDS = [0, 2e12, 2e13]                                      # tokens served over the model's life
grid = [10 ** (9 + 0.01 * i) for i in range(int((10.6 - 9) / 0.01) + 1)]
infopt = []
for dinf in DEMANDS:
    tot = [(n, 6 * n * d_train(n) + 2 * n * dinf) for n in grid if d_train(n) < math.inf]
    nb, cb = min(tot, key=lambda t: t[1])
    c_13 = 6 * 13e9 * d13 + 2 * 13e9 * dinf
    infopt.append({"D_inf": dinf, "curve": tot, "n_best": nb, "d_best": d_train(nb), "tokens_per_param": d_train(nb) / nb,
                   "total_best": cb, "total_13b": c_13, "saving": 1 - cb / c_13})
    print(f"serve {dinf:.0e} tokens: best N {nb / 1e9:5.1f}B trained on {d_train(nb) / 1e12:5.2f}T tokens "
          f"({d_train(nb) / nb:5.0f} tokens/param); total {cb:.2e} vs {c_13:.2e} with the 13B ({1 - cb / c_13:.0%} less)")
print(f"target: the compute-optimal 13B ({d13 / 1e9:.0f}B tokens, {d13 / 13e9:.0f} tokens/param), loss {target:.4f}")

# ---------------------------------------------------------------- 4. tokens per parameter
MODELS = [("GPT-3", 2020, 175e9, 300e9), ("Chinchilla", 2022, 70e9, 1.4e12), ("LLaMA-65B", 2023, 65e9, 1.4e12),
          ("Llama 2 70B", 2023, 70e9, 2e12), ("Llama 3 70B", 2024, 70e9, 15e12), ("Llama 3 8B", 2024, 8e9, 15e12),
          ("Qwen3-0.6B", 2025, 0.6e9, 36e12)]
tpp = [{"model": m, "year": y, "params": n, "tokens": d, "tokens_per_param": d / n} for m, y, n, d in MODELS]
for r in tpp:
    print(f"{r['model']:12s} {r['year']}  {r['tokens_per_param']:9,.1f} tokens per parameter")

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps({"kaplan": kaplan, "chinchilla_fit": {"E": E, "A": A, "B": B, "alpha": AL, "beta": BE,
                                                                 "source": "Besiroglu et al. 2024, Table 1"},
                           "isoflop": iso, "gopher": gopher, "chinchilla": chinchilla,
                           "inference_optimal": {"target_loss": target, "target_n": 13e9, "target_d": d13,
                                                 "runs": infopt},
                           "tokens_per_param": tpp}, indent=1))
print("saved", OUT)
