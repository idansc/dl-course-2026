"""Code for the clip MuonOrthogonalize (renders/w01/MuonOrthogonalize.mp4): Muon's orthogonalization step by step,
on the matrix of the Muon slides: M = W1's SGD-momentum buffer (100 x 3072) after one epoch of the 2-layer MLP on
CIFAR-10. A thin wrapper around muon_cifar.py (the slides' code): the same M (momentum_buffer) and the same
zeropower_via_newtonschulz5 (Keller Jordan, github.com/KellerJordan/Muon, MIT license; float32 on a CPU).

Newton-Schulz keeps the singular vectors of M = U S V^T and maps every singular value s to
3.4445 s - 4.7750 s^3 + 2.0315 s^5, five times, starting from M / |M|_F. The result is close to U V^T: every direction
gets about the same step size.

Run:  python muon_demo.py      -> prints the singular values after each step, saves figures/muon_orthogonalize.png
Try:  steps = 10 (the values keep moving inside a band around 1: the coefficients trade exactness for speed);
      the cubic iteration a, b, c = 1.5, -0.5, 0 (converges to exactly 1, but needs many more steps);
      the momentum buffer after more epochs (muon_cifar.momentum_buffer trains one epoch).
"""
from pathlib import Path

import numpy as np
import torch

from muon_cifar import momentum_buffer, zeropower_via_newtonschulz5    # imports CIFAR-10 via cifar_experiment.py

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)                            # a, b, c in zeropower_via_newtonschulz5


def ns_map(s):
    """What one Newton-Schulz step does to a singular value."""
    a, b, c = NS_COEFFICIENTS
    return a * s + b * s ** 3 + c * s ** 5


def muon_iterations(steps=5, M=None):
    """Singular values of the Newton-Schulz iterate X_t after t = 0, ..., steps, per direction: entry [t, k] is
    u_k^T X_t v_k for the k-th singular pair (u_k, v_k) of M, largest first (X_0 = M / |M|_F)."""
    M = momentum_buffer() if M is None else M
    U, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    iterates = [zeropower_via_newtonschulz5(M, steps=t).double() for t in range(steps + 1)]
    sv = np.array([((U.T @ X) * Vh).sum(1).numpy() for X in iterates])          # u_k^T X_t v_k for every k
    sv_sorted = np.array([torch.linalg.svdvals(X).numpy() for X in iterates])
    assert np.allclose(np.sort(sv, 1)[:, ::-1], sv_sorted, atol=1e-4)          # the same values, unsorted
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    UV = U @ Vh                                                                 # the exact orthogonalization
    O = iterates[-1]
    return dict(shape=tuple(M.shape), sv_rel=(S / S[0]).numpy(), sv=sv, top1=energy[0].item(),
                top5=energy[4].item(), top10=energy[9].item(), max_over_fro=(S[0] / S.norm()).item(),
                rel_err_UV=((O - UV).norm() / UV.norm()).item(),
                cos_UV=((O * UV).sum() / (O.norm() * UV.norm())).item())


if __name__ == "__main__":
    r = muon_iterations()
    print(f"M = W1 momentum buffer {r['shape']}: top-1 direction {100 * r['top1']:.0f}% of the energy, "
          f"top-5 {100 * r['top5']:.0f}%, top-10 {100 * r['top10']:.0f}%")
    print(f"sigma / sigma_max from 1 down to {r['sv_rel'].min():.3f}; sigma_max / |M|_F = {r['max_over_fro']:.3f}")
    for t, s in enumerate(r["sv"]):
        print(f"step {t}: singular values in [{s.min():.3f}, {s.max():.3f}]")
    print(f"after 5 steps: |X_5 - UV^T| / |UV^T| = {r['rel_err_UV']:.2f}, cosine {r['cos_UV']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1.7, 1]))
    k = np.arange(1, len(r["sv_rel"]) + 1)
    ax.plot(k, r["sv_rel"], color="#9CA3AF", lw=2, label="σ / σ_max")
    for t, s in enumerate(r["sv"]):
        ax.plot(k, s, color=plt.cm.Purples(0.35 + 0.13 * t), lw=1.5, label=f"Newton–Schulz step {t}")
    ax.axhline(1, color="#9CA3AF", ls="--", lw=1)
    ax.set_xlabel("direction k (largest σ first)"); ax.set_ylabel("singular value")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    s = np.linspace(0, 1.25, 300)
    ax2.plot(s, ns_map(s), color="#7C3AED", label="3.4445σ − 4.7750σ³ + 2.0315σ⁵")
    ax2.plot(s, s, color="#9CA3AF", ls=":", label="σ")
    ax2.axhline(1, color="#9CA3AF", ls="--", lw=1)
    ax2.set_xlabel("σ before a step"); ax2.set_ylabel("σ after the step"); ax2.legend(fontsize=8)
    fig.suptitle(f"Muon on W1's momentum buffer {r['shape'][0]}×{r['shape'][1]}: after 5 steps every σ is in "
                 f"[{r['sv'][-1].min():.2f}, {r['sv'][-1].max():.2f}]")
    fig.tight_layout()
    out = Path(__file__).with_name("figures") / "muon_orthogonalize.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)
