"""Code for the static Muon slides of lecture 2: every number and chart on them. The matrix is the one of muon_cifar.py,
G = W1's SGD-momentum buffer (100 x 3072) after one epoch of the 2-layer MLP on CIFAR-10.

  "More recent methods: Muon"        update_spectrum, one_minibatch: few directions carry the update, and why
                                     (dL/dW1 = sum over the batch of ds_i x_i^T: every row is a mix of input images)
  "Adam evens out the weights ..."   sign_example: 8 entries of G and their signs; G's singular values -> U V^T (all 1)
  "Why the same step ..."            same_output_budget: first-order loss decrease of steps with the same sigma_max
  "Computing U V^T fast"             newton_schulz_curves: what 1, 2, 3, 5 Newton-Schulz steps do to a singular value,
                                     and where the 100 singular values of G land after 5 steps

Run:  python muon_explained.py   -> prints the numbers, writes data/muon_explained.json, saves figures/muon_explained.png
Try:  same_output_budget(torch.randn(100, 3072)): a random matrix has a flat spectrum, and U V^T gains only about 1.2x
      over the gradient (3.1x on the real update); one_minibatch(N=16): 16 outer products, so the gradient has rank at most 16.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cifar_experiment import Xtr, Ytr, mlp
from muon_cifar import momentum_buffer, zeropower_via_newtonschulz5
from muon_demo import muon_iterations, ns_map

OUT = Path(__file__).with_name("data") / "muon_explained.json"


def energy(s, k):
    """Share of the squared singular values in the top k directions."""
    return float((s[:k] ** 2).sum() / (s ** 2).sum())


def update_spectrum(G):
    s = torch.linalg.svdvals(G.double())
    return dict(shape=list(G.shape), sv_rel=(s / s[0]).tolist(), top1=energy(s, 1), top5=energy(s, 5),
                top10=energy(s, 10))


def one_minibatch(N=128, seed=0):
    """dL/dW1 for one minibatch at initialisation, from autograd and as the sum of N outer products ds_i x_i^T
    (ds_i = dL/ds for image i, s = W1 x + b1). Its rows are mixes of the N images, so its spectrum follows theirs."""
    torch.manual_seed(seed)
    model = mlp()
    X, y = Xtr[:N], Ytr[:N]
    s = model[0](X)
    s.retain_grad()
    F.cross_entropy(model[2](torch.relu(s)), y).backward()
    G = model[0].weight.grad.double()
    outer = sum(torch.outer(s.grad[i].double(), X[i].double()) for i in range(N))
    sg, sx = torch.linalg.svdvals(G), torch.linalg.svdvals(X.double())
    return dict(N=N, max_abs_err=(outer - G).abs().max().item(), max_abs=G.abs().max().item(),
                grad_top1=energy(sg, 1), grad_top5=energy(sg, 5), images_top1=energy(sx, 1), images_top5=energy(sx, 5))


def sign_example(G, k=8, seed=0):
    """k random entries of G (scaled to max |g| = 1) and their signs: Adam with beta1 = beta2 = 0 steps by sign(g)."""
    idx = torch.randperm(G.numel(), generator=torch.Generator().manual_seed(seed))[:k]
    g = G.flatten()[idx].double()
    g = g / g.abs().max()
    return dict(g=g.tolist(), sign=torch.sign(g).tolist())


def same_output_budget(G):
    """First-order loss decrease <G, D> of a step D with sigma_max(D) = 1, i.e. |D x| <= |x| for every input x:
    D = G / sigma_max(G) (the gradient), sign(G) / sigma_max(sign(G)) (Adam-like), U V^T (Muon, the best such step)
    and Newton-Schulz's approximation of U V^T (what Muon computes), scaled the same way."""
    G = G.double()
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    smax = lambda D: torch.linalg.matrix_norm(D, ord=2)
    steps = {"gradient": G, "sign": torch.sign(G), "U V^T": U @ Vh,
             "Newton-Schulz": zeropower_via_newtonschulz5(G.float(), steps=5).double()}
    dec = {k: ((G * D).sum() / smax(D)).item() for k, D in steps.items()}
    assert abs(dec["U V^T"] - S.sum().item()) < 1e-6 * S.sum().item()      # <G, U V^T> = sum of the singular values
    return dict(decrease=dec, ratio={k: v / dec["gradient"] for k, v in dec.items()})


def newton_schulz_curves(G, steps=(1, 2, 3, 5), n=201):
    """sigma -> sigma after t Newton-Schulz steps (the scalar map, iterated), and the 100 singular values of G:
    sigma_k / |G|_F before (what the iteration starts from) and u_k^T X_5 v_k after 5 steps on the matrix."""
    grid = np.linspace(0, 1, n)
    curves, x = {}, grid.copy()
    for t in range(1, max(steps) + 1):
        x = ns_map(x)
        if t in steps:
            curves[str(t)] = x.tolist()
    r = muon_iterations(steps=5, M=G)
    real_in, real_out = r["sv"][0], r["sv"][-1]
    assert np.allclose(real_out, [ns_map(ns_map(ns_map(ns_map(ns_map(s))))) for s in real_in], atol=1e-3)
    return dict(grid=grid.tolist(), curves=curves, real_in=real_in.tolist(), real_out=real_out.tolist())


if __name__ == "__main__":
    G = momentum_buffer()
    res = dict(spectrum=update_spectrum(G), one_minibatch=one_minibatch(), sign_example=sign_example(G),
               budget=same_output_budget(G), newton_schulz=newton_schulz_curves(G))
    sp, mb, bd, ns = res["spectrum"], res["one_minibatch"], res["budget"], res["newton_schulz"]
    print(f"G = W1 momentum buffer {sp['shape']}: top-1 direction {100 * sp['top1']:.0f}% of the energy, "
          f"top-5 {100 * sp['top5']:.0f}%, top-10 {100 * sp['top10']:.0f}%")
    print(f"one minibatch at init: sum of {mb['N']} outer products = autograd (max error {mb['max_abs_err']:.1e}, "
          f"entries up to {mb['max_abs']:.1e}); gradient top-1 {100 * mb['grad_top1']:.0f}%, top-5 "
          f"{100 * mb['grad_top5']:.0f}%; the images themselves top-1 {100 * mb['images_top1']:.0f}%, top-5 "
          f"{100 * mb['images_top5']:.0f}%")
    print("8 entries of G:", " ".join(f"{v:+.2f}" for v in res["sign_example"]["g"]))
    print("first-order decrease with sigma_max(step) = 1: " +
          ", ".join(f"{k} {v:.2f} ({bd['ratio'][k]:.2f}x)" for k, v in bd["decrease"].items()))
    print(f"Newton-Schulz on G: sigma / |G|_F in [{min(ns['real_in']):.3f}, {max(ns['real_in']):.3f}] -> after 5 steps "
          f"[{min(ns['real_out']):.2f}, {max(ns['real_out']):.2f}]")
    OUT.write_text(json.dumps(res))
    print("saved", OUT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    k = np.arange(1, len(sp["sv_rel"]) + 1)
    ax[0].bar(k, sp["sv_rel"], color="#7C3AED")
    ax[0].set(xlabel="direction k", ylabel="σ_k / σ_1", title=f"G: top-1 {100 * sp['top1']:.0f}%, top-5 "
                                                               f"{100 * sp['top5']:.0f}% of the update")
    g = res["sign_example"]
    ax[1].bar(np.arange(8) - 0.2, g["g"], 0.4, color="#7C3AED", label="g")
    ax[1].bar(np.arange(8) + 0.2, g["sign"], 0.4, color="#16A34A", label="sign(g)")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].legend(); ax[1].set(title="Adam (β1 = β2 = 0): sign(g)")
    ax[2].bar(list(bd["ratio"]), list(bd["ratio"].values()), color=["#9CA3AF", "#9CA3AF", "#16A34A", "#86EFAC"])
    ax[2].set(ylabel="loss decrease / gradient's", title="same σ_max(step): first-order decrease")
    for t, c in ns["curves"].items():
        ax[3].plot(ns["grid"], c, color=plt.cm.Purples(0.3 + 0.12 * int(t)), label=f"{t} step{'s' * (t != '1')}")
    ax[3].plot(ns["real_in"], ns["real_out"], "o", ms=3, color="#16A34A", label="G's 100 σ, 5 steps")
    ax[3].axhline(1, color="#9CA3AF", ls="--", lw=1)
    ax[3].set(xlabel="σ before (G / |G|_F)", ylabel="σ after", title="Newton–Schulz"); ax[3].legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_name("figures") / "muon_explained.png"
    fig.savefig(out, dpi=120)
    print("saved", out)
