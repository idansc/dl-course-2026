"""Code for the clip SpaceWarping (renders/w01/SpaceWarping.mp4): one hidden layer h = ReLU(Wx + b) makes XOR
linearly separable. Same points, W and b as the clip.

Run:  python space_warping.py      -> prints accuracies, saves figures/space_warping.png (x -> z -> h)
Try:  remove the ReLU (h = z) and check that no line separates the classes any more.
"""
from pathlib import Path
import numpy as np

rng = np.random.default_rng(4)
centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
labels = np.array([0, 0, 1, 1])                     # XOR: 1 when the signs differ
X = np.concatenate([c + rng.normal(0, 0.13, (22, 2)) for c in centers])
y = np.repeat(labels, 22)
W = np.array([[0.85, -0.65], [-0.65, 0.85]])
b = np.array([-0.5, -0.5])


def hidden(X, relu=True):
    Z = X @ W.T + b                                 # linear part: stretch, squash, shift
    return np.maximum(Z, 0) if relu else Z          # ReLU folds negative values onto the axes


def best_linear_accuracy(F, y, n_angles=3600):
    """Best accuracy of any line w·f + c = 0 (brute force over directions and thresholds)."""
    best = 0.0
    for t in np.linspace(0, np.pi, n_angles, endpoint=False):
        proj = F @ np.array([np.cos(t), np.sin(t)])
        for thr in np.unique(proj):
            for sign in (1, -1):
                acc = np.mean((sign * (proj - thr) > 0) == y)
                best = max(best, acc)
    return best


if __name__ == "__main__":
    H = hidden(X)
    pred = (H[:, 0] + H[:, 1] > 0.5).astype(int)    # the clip's linear read-out in h-space
    print(f"read-out [h1 + h2 > 0.5]: accuracy {np.mean(pred == y):.3f}")
    print(f"best line in x-space:      {best_linear_accuracy(X, y, 720):.3f}")
    print(f"best line in z-space (no ReLU): {best_linear_accuracy(hidden(X, relu=False), y, 720):.3f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, F, name in zip(axes, [X, hidden(X, relu=False), H], ["x (input)", "z = Wx + b", "h = ReLU(z)"]):
        ax.scatter(F[y == 1, 0], F[y == 1, 1], c="#16A34A", s=18, label="class 1")
        ax.scatter(F[y == 0, 0], F[y == 0, 1], marker="s", facecolors="white", edgecolors="#EA580C", s=18, label="class 0")
        ax.axhline(0, color="#9CA3AF", lw=0.8); ax.axvline(0, color="#9CA3AF", lw=0.8)
        ax.set_title(name); ax.set_aspect("equal")
    t = np.linspace(-0.3, 1.0, 2)
    axes[2].plot(t, 0.5 - t, color="#16A34A", lw=2)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_name("figures") / "space_warping.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print("saved", out)
