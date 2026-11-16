"""Code for lecture 5 (self-supervised learning), the slide "Every direction: informative, not a copy": a correlated
2-D embedding cloud before and after whitening (after Jia-Bin Huang's "How AI Learned to Teach Itself [JEPA]",
YouTube 2026, 32:46).

Correlated: z2 is nearly a copy of z1 (correlation 0.9), so one direction (the diagonal) holds almost all of the
spread and the other almost none. Whitened (ZCA: subtract the mean, multiply by the inverse square root of the
covariance): every direction has variance 1 and no direction repeats another, covariance = I.

Printed: the covariance, the correlation and the share of the spread along each principal direction, before and
after. Run:  python whitening_cloud.py OUT.png      (also writes data/whitening_cloud.json)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

rng = np.random.default_rng(3)
N, RHO = 80, 0.9
Z = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], size=N)
Z = (Z - Z.mean(0)) / Z.std(0, ddof=1)                         # each coordinate alone: mean 0, variance 1
C = np.cov(Z.T)
evals, evecs = np.linalg.eigh(C)
W = Z @ evecs @ np.diag(evals ** -0.5) @ evecs.T                  # ZCA whitening: covariance exactly I
CW = np.cov(W.T)


def describe(name, cov):
    ev = np.linalg.eigvalsh(cov)[::-1]
    share = ev / ev.sum()
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    print(f"{name:10s} covariance [[{cov[0, 0]:.2f}, {cov[0, 1]:.2f}], [{cov[1, 0]:.2f}, {cov[1, 1]:.2f}]]  "
          f"correlation {corr:.2f}  spread along the principal directions {share[0]:.0%} / {share[1]:.0%}")
    return {"cov": cov.round(4).tolist(), "corr": round(float(corr), 4), "share": share.round(4).tolist()}


out = {"N": N, "rho": RHO, "correlated": describe("correlated", C), "whitened": describe("whitened", CW)}
OUTP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("whitening_cloud.png")
data = Path(__file__).parent / "data"
data.mkdir(exist_ok=True)
(data / "whitening_cloud.json").write_text(json.dumps(out, indent=1))

# ---------------------------------------------------------------- the figure: two square panels and an arrow
plt.rcParams["font.family"] = "Arial"
BLUE, GRID, AXIS = "#2F5597", "#E3E3E3", "#A6A6A6"
fig = plt.figure(figsize=(10.4, 4.0))
panels = [fig.add_axes([0.02, 0.02, 0.37, 0.96]), fig.add_axes([0.61, 0.02, 0.37, 0.96])]
for ax, pts, cov in zip(panels, (Z, W), (C, CW)):
    for v in np.arange(-3, 3.1, 1.0):
        ax.axhline(v, color=GRID, lw=0.8, zorder=0)
        ax.axvline(v, color=GRID, lw=0.8, zorder=0)
    ax.axhline(0, color=AXIS, lw=1.2, zorder=1)
    ax.axvline(0, color=AXIS, lw=1.2, zorder=1)
    ev, V = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(ev[1]), 2 * 2 * np.sqrt(ev[0]), angle=ang, facecolor=BLUE,
                         alpha=0.10, edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((0, 0), 2 * 2 * np.sqrt(ev[1]), 2 * 2 * np.sqrt(ev[0]), angle=ang, facecolor="none",
                         edgecolor=BLUE, lw=2.6, zorder=3))
    ax.scatter(pts[:, 0], pts[:, 1], s=26, color=BLUE, alpha=0.85, lw=0, zorder=4)
    ax.set(xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), aspect="equal", xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(3.2, -0.12, "$z_1$", ha="right", va="top", fontsize=15, color="#595959")
    ax.text(0.12, 3.2, "$z_2$", ha="left", va="top", fontsize=15, color="#595959")
fig.patches.append(FancyArrowPatch((0.425, 0.5), (0.575, 0.5), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=26, lw=2.2, color="#7F7F7F"))
fig.text(0.5, 0.57, "whitening", ha="center", fontsize=15, color="#595959")
fig.savefig(OUTP, dpi=220, facecolor="white")
print("saved", OUTP)
