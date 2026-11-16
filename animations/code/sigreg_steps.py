"""Code for lecture 5 (self-supervised learning), the slide "SIGReg: make every 1-D slice Gaussian": what SIGReg
computes, before and after training (LeJEPA, Balestriero & LeCun, 2025, arXiv 2511.08544, Definition 2 and
Algorithm 1).

Top: a lumpy 2-D embedding cloud (four clumps, as an encoder that groups images into a few tight clusters).
Both clouds are whitened exactly (mean 0, covariance I), so VICReg's variance and covariance terms are 0 for both:
VICReg cannot tell them apart, SIGReg can.
Bottom: a sample of N(0, I), the round cloud that minimizing SIGReg pushes the embeddings towards (an
illustration of the target, not the output of a trained encoder). For three random unit
directions the 1-D projections ("slices") are drawn against N(0, 1), with their Epps-Pulley statistic (the paper's
Algorithm 1: 17 points on [-5, 5], Gaussian window exp(-t^2/2), trapezoid rule, times N); SIGReg is the average of
that statistic over random directions, here 1,024.

Run:  python sigreg_steps.py OUT.png      (also writes data/sigreg_steps.json)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T = np.linspace(-5, 5, 17)
WINDOW = np.exp(-0.5 * T ** 2)
trapz = getattr(np, "trapezoid", None) or np.trapz


def epps_pulley(z):                                   # same as sigreg_slices.py: Algorithm 1 for one direction
    ecf = np.exp(1j * z[:, None] * T[None, :]).mean(0)
    return float(trapz(np.abs(ecf - WINDOW) ** 2 * WINDOW, T) * len(z))


def sigreg(x, m, rng):
    a = rng.standard_normal((x.shape[1], m))
    a /= np.linalg.norm(a, axis=0)
    z = x @ a
    return float(np.mean([epps_pulley(z[:, j]) for j in range(m)]))


rng = np.random.default_rng(7)
N = 512
centers = np.array([[-1.4, 1.0], [1.1, 1.4], [1.5, -0.8], [-0.9, -1.3]])
lumpy = centers[rng.integers(0, 4, N)] + 0.2 * rng.standard_normal((N, 2))
gauss = rng.standard_normal((N, 2))


def whiten(x):                                                       # mean 0, covariance exactly I
    x = x - x.mean(0)
    ev, V = np.linalg.eigh(np.cov(x.T))
    return x @ V @ np.diag(ev ** -0.5) @ V.T


def vicreg(x, mu=25.0, nu=1.0, eps=1e-4):
    """VICReg's anti-collapse terms (Bardes et al., 2022, Eqs. 1-4, weights of its code): mu * variance hinge +
    nu * covariance; the invariance term needs two views and is left out."""
    d = x.shape[1]
    v = np.mean(np.maximum(0.0, 1.0 - np.sqrt(x.var(0, ddof=1) + eps)))
    c = np.cov(x.T)
    return float(mu * v + nu * ((c ** 2).sum() - (np.diag(c) ** 2).sum()) / d)


lumpy, gauss = whiten(lumpy), whiten(gauss)
ANGLES = (10.0, 100.0, 165.0)          # three directions where the lumps show (130-155 deg would look Gaussian-ish)
DIRS = [np.array([math.cos(math.radians(a)), math.sin(math.radians(a))]) for a in ANGLES]

stats = {}
for name, x in (("before", lumpy), ("after", gauss)):
    stats[name] = {"slices": [epps_pulley(x @ u) for u in DIRS], "sigreg_1024": sigreg(x, 1024, np.random.default_rng(1)),
                   "vicreg": vicreg(x)}
    print(f"{name:6s}: slices " + ", ".join(f"{v:6.1f}" for v in stats[name]["slices"]) +
          f"   SIGReg over 1,024 directions {stats[name]['sigreg_1024']:.1f}   VICReg terms {stats[name]['vicreg']:.2f}")
OUTP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sigreg_steps.png")
data = Path(__file__).parent / "data"
data.mkdir(exist_ok=True)
(data / "sigreg_steps.json").write_text(json.dumps({"N": N, "angles": ANGLES, "stats": stats}, indent=1))

# ---------------------------------------------------------------- figure: 2 rows (before, after) x 5 columns
plt.rcParams["font.family"] = "Arial"
COLS = ["#2F5597", "#ED7D31", "#548235"]
INK, GREY, RED = "#000000", "#7F7F7F", "#CC0000"
fig = plt.figure(figsize=(12.6, 5.2))
gs = fig.add_gridspec(2, 5, width_ratios=[1.25, 1, 1, 1, 1.05], hspace=0.35, wspace=0.28,
                      left=0.075, right=0.995, top=0.86, bottom=0.04)
grid = np.linspace(-4, 4, 200)
pdf = np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi)
for r, (name, x) in enumerate((("before", lumpy), ("after", gauss))):
    ax = fig.add_subplot(gs[r, 0])
    ax.scatter(x[:, 0], x[:, 1], s=6, color="#A6A6A6", lw=0)
    for u, c in zip(DIRS, COLS):
        ax.annotate("", xy=2.7 * u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4, mutation_scale=15))
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    ax.set_box_aspect(1)
    for s in ax.spines.values():
        s.set_color("#D0D0D0")
    for k, (u, c) in enumerate(zip(DIRS, COLS)):
        a = fig.add_subplot(gs[r, 1 + k])
        a.hist(x @ u, bins=np.linspace(-4, 4, 33), density=True, color=c, alpha=0.8, lw=0)
        a.plot(grid, pdf, color=INK, lw=1.5, ls="--")
        a.set(xlim=(-4, 4), ylim=(0, 0.75), xticks=[], yticks=[])
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.text(0.97, 0.95, f"{stats[name]['slices'][k]:.1f}", transform=a.transAxes, ha="right", va="top",
               fontsize=14, color=c)
    v = fig.add_subplot(gs[r, 4])
    v.axis("off")
    big = stats[name]["sigreg_1024"]
    v.text(0.5, 0.80, f"{big:.1f}", ha="center", va="center", fontsize=32, color=RED if r == 0 else "#2F5597",
           fontweight="bold")
    v.text(0.5, 0.56, "SIGReg", ha="center", va="center", fontsize=15, color=GREY)
    v.text(0.5, 0.28, f"{stats[name]['vicreg']:.2f}", ha="center", va="center", fontsize=22, color=GREY,
           fontweight="bold")
    v.text(0.5, 0.06, "VICReg", ha="center", va="center", fontsize=15, color=GREY)
fig.text(0.012, 0.69, "lumpy", rotation=90, va="center", fontsize=17, color=INK)
fig.text(0.012, 0.24, "Gaussian", rotation=90, va="center", fontsize=17, color=INK)
for xpos, num, label in ((0.10, "1", "random directions"), (0.40, "2", "each slice vs N(0, 1): a test value"),
                         (0.875, "3", "average")):
    fig.text(xpos, 0.955, num, ha="center", va="center", fontsize=14, color="white", fontweight="bold",
             bbox=dict(boxstyle="circle,pad=0.32", fc=COLS[0], ec="none"))
    fig.text(xpos + 0.02, 0.955, label, ha="left", va="center", fontsize=16, color=INK)
fig.savefig(OUTP, dpi=200, facecolor="white")
print("saved", OUTP)
