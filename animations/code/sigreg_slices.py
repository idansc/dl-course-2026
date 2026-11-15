"""Code for lecture 5 (self-supervised learning), the SIGReg slides: why matching the mean and covariance does not
make a cloud Gaussian, and how SIGReg notices (LeJEPA, Balestriero & LeCun, 2025, arXiv 2511.08544: Fig. 5,
Definition 2 and Algorithm 1).

The "X" cloud: half of the points lie on the line y = x, half on y = -x, with t ~ N(0, 1) along the line, plus a little
noise; the result is rescaled so that each coordinate is N(0, 1). Its mean is 0 and its covariance is I, so whitening,
Barlow Twins and VICReg are all satisfied, yet it is not an isotropic Gaussian: projected onto the diagonal, half of
the points land at 0.

SIGReg projects a batch onto random unit directions and scores every 1-D sample with the Epps-Pulley statistic, here
exactly as in the paper's Algorithm 1 (17 points on [-5, 5], Gaussian window exp(-t^2/2), trapezoid rule, times N),
then averages over directions. Printed: mean and covariance of the cloud, the statistic on the x axis, the y axis and
the diagonal, and SIGReg over 1,024 random directions, each for the X cloud and for a true N(0, I) sample.

Run:  python sigreg_slices.py OUT.png      (also writes data/sigreg_slices.json)
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
WINDOW = np.exp(-0.5 * T ** 2)                  # the N(0, 1) characteristic function, also used as the weight
trapz = getattr(np, "trapezoid", None) or np.trapz


def epps_pulley(z):
    """Epps-Pulley statistic of 1-D samples z against N(0, 1), as in Algorithm 1 of the paper (one direction)."""
    ecf = np.exp(1j * z[:, None] * T[None, :]).mean(0)          # empirical characteristic function
    err = np.abs(ecf - WINDOW) ** 2 * WINDOW
    return float(trapz(err, T) * len(z))


def sigreg(x, m, rng):
    """SIGReg (Definition 2): average Epps-Pulley statistic over m random unit directions."""
    a = rng.standard_normal((x.shape[1], m))
    a /= np.linalg.norm(a, axis=0)
    z = x @ a
    return float(np.mean([epps_pulley(z[:, j]) for j in range(m)]))


rng = np.random.default_rng(0)
N, NOISE = 2048, 0.08
t = rng.standard_normal(N // 2)
X = np.concatenate([np.stack([t, t], 1), np.stack([t, -t], 1)])     # the two arms of the X
X = X + NOISE * rng.standard_normal(X.shape)
X = X - X.mean(0)                                                  # then whiten exactly: mean 0, covariance I,
evals, evecs = np.linalg.eigh(np.cov(X.T))                         # so whitening, Barlow Twins and VICReg have
X = X @ evecs @ np.diag(evals ** -0.5) @ evecs.T                   # nothing left to fix
G = rng.standard_normal((N, 2))                                    # a true isotropic Gaussian, for reference

mean, cov = X.mean(0), np.cov(X.T)
print(f"X cloud: mean = [{mean[0]:+.3f}, {mean[1]:+.3f}],  covariance = [[{cov[0, 0]:.3f}, {cov[0, 1]:+.3f}], "
      f"[{cov[1, 0]:+.3f}, {cov[1, 1]:.3f}]]")
DIRS = {"x axis": 0.0, "y axis": 90.0, "diagonal": 45.0}
stats = {}
for name, deg in DIRS.items():
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    stats[name] = {"X": epps_pulley(X @ u), "gaussian": epps_pulley(G @ u)}
    print(f"  Epps-Pulley on the {name:8s} ({deg:4.0f} deg):  X cloud {stats[name]['X']:7.2f}   "
          f"N(0, I) {stats[name]['gaussian']:5.2f}")
sig = {"X": sigreg(X, 1024, np.random.default_rng(1)), "gaussian": sigreg(G, 1024, np.random.default_rng(1))}
print(f"  SIGReg over 1,024 random directions:  X cloud {sig['X']:.2f}   N(0, I) {sig['gaussian']:.2f}")

OUTP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sigreg_slices.png")
data = Path(__file__).parent / "data"
data.mkdir(exist_ok=True)
(data / "sigreg_slices.json").write_text(json.dumps(
    {"N": N, "noise": NOISE, "mean": mean.tolist(), "cov": cov.tolist(), "epps_pulley": stats, "sigreg_1024": sig},
    indent=1))

# ---------------------------------------------------------------- the figure: the cloud and three of its slices
plt.rcParams["font.family"] = "Arial"
BLUE, RED, GREY, INK = "#2F5597", "#CC0000", "#A6A6A6", "#000000"
fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), gridspec_kw=dict(wspace=0.28))
for a in axes:
    a.set_box_aspect(1)                                           # four equal squares
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=3, color=GREY, alpha=0.5, lw=0)
for name, deg in DIRS.items():
    c = RED if name == "diagonal" else BLUE
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    ax.annotate("", xy=3.2 * u, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2, mutation_scale=16))
ax.set(xlim=(-3.6, 3.6), ylim=(-3.6, 3.6), xticks=[], yticks=[])
ax.set_title("mean 0, covariance I", fontsize=15)
for s in ax.spines.values():
    s.set_color("#BFBFBF")
grid = np.linspace(-4, 4, 200)
pdf = np.exp(-grid ** 2 / 2) / math.sqrt(2 * math.pi)
for a, (name, deg) in zip(axes[1:], DIRS.items()):
    u = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
    c = RED if name == "diagonal" else BLUE
    a.hist(X @ u, bins=np.linspace(-4, 4, 41), density=True, color=c, alpha=0.75, lw=0)
    a.plot(grid, pdf, color=INK, lw=1.6, ls="--")
    a.set(xlim=(-4, 4), ylim=(0, 0.8), yticks=[], xticks=[-2, 0, 2])
    a.tick_params(labelsize=12)
    a.set_title(f"{name}: test {stats[name]['X']:.1f}", fontsize=15, color=c)
    for s in ("top", "right", "left"):
        a.spines[s].set_visible(False)
    if name == "diagonal":
        a.annotate("half of the\npoints at 0", xy=(0.15, 0.77), xytext=(1.3, 0.55), fontsize=12, color=RED,
                   arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))
fig.text(0.62, 0.02, "dashed: N(0, 1)", fontsize=12, color="#595959", ha="center")
fig.savefig(OUTP, dpi=220, bbox_inches="tight", facecolor="white")
print("saved", OUTP)
