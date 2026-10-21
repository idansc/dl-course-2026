"""Code for the clip MatrixBackprop (renders/w01/MatrixBackprop.mp4): backprop through the linear layer S = X W^T + b
with matrices, on one CIFAR-10 minibatch (N = 128), checked against autograd. Every gradient has the shape of its
variable, and the formula is the only product of the available matrices whose shapes fit.

Run:  python matrix_backprop.py    -> prints the shapes, the error vs autograd and the Jacobian size shown in the clip,
                                      saves figures/matrix_backprop.png
Try:  dS @ X (the clip's rejected product) raises a shape error; dS.T @ X is the product that fits.
"""
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[2] / "tutorials/data/cifar-10-batches-py"


def minibatch(N=128, root=DATA):
    """X: the first N images of data_batch_1 as rows (N x 3072, float64, pixels/255 minus their mean); y: labels."""
    if not root.exists():                                   # first run: download CIFAR-10, python version (163 MB)
        import torchvision
        torchvision.datasets.CIFAR10(root.parent, download=True)
    with open(root / "data_batch_1", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    X = torch.tensor(d[b"data"][:N].astype(np.float64) / 255.0)
    return X - X.mean(0), torch.tensor(d[b"labels"][:N])


def backward(X, y, W, S):
    """The backward pass of S = X W^T + b followed by the mean cross-entropy, written with matrix products."""
    N = X.shape[0]
    dS = (torch.softmax(S, dim=1) - F.one_hot(y, 10).to(S.dtype)) / N    # dL/dS: N x 10, the shape of S
    dW = dS.T @ X                   # dL/dW = (dL/dS)^T X: (10 x N)(N x 3072) = 10 x 3072, the shape of W
    db = dS.sum(0)                  # dL/db: sum of the rows of dL/dS, 10
    dX = dS @ W                     # dL/dX = (dL/dS) W: (N x 10)(10 x 3072) = N x 3072, the shape of X
    return dS, dW, db, dX


def check(N=128, seed=0):
    """The formulas vs autograd on a real minibatch, and the size of the Jacobian dS/dW they never build."""
    X, y = minibatch(N)
    torch.manual_seed(seed)
    W = (0.001 * torch.randn(10, 3072, dtype=torch.float64)).requires_grad_()
    b = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    Xr = X.clone().requires_grad_()
    S = Xr @ W.T + b                                # forward: N x 10 scores
    L = F.cross_entropy(S, y)
    L.backward()                                    # autograd fills W.grad, b.grad, Xr.grad
    with torch.no_grad():
        dS, dW, db, dX = backward(X, y, W, S)
    err = lambda a, g: (a - g).abs().max().item()
    return dict(N=N, D=X.shape[1], C=W.shape[0], loss=L.item(), shapes=dict(X=tuple(X.shape), W=tuple(W.shape),
                S=tuple(S.shape), dS=tuple(dS.shape), dW=tuple(dW.shape), dX=tuple(dX.shape)),
                max_abs_err_dW=err(dW, W.grad), max_abs_err_db=err(db, b.grad), max_abs_err_dX=err(dX, Xr.grad),
                jacobian_entries=S.numel() * W.numel(), jacobian_MB_fp32=S.numel() * W.numel() * 4 / 1e6,
                rejected=rejected_product(dS, X))


def rejected_product(dS, X):
    """the first attempt in the clip: dL/dS times X"""
    try:
        dS @ X
    except RuntimeError as e:
        return str(e)


def figure(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle
    N, D, C = r["N"], r["D"], r["C"]
    size = {N: 1.25, D: 2.9, C: 0.42, 1: 0.13}              # drawn sizes: 3072 wide, 128 medium, 10 thin
    blue, purple, ink = "#2563EB", "#7C3AED", "#1F2937"
    fig, ax = plt.subplots(figsize=(8.6, 6.6))

    def row(y, items):
        """draw a product left to right: (rows, cols, name, colour) blocks and operator strings"""
        x = 0.0
        for it in items:
            if isinstance(it, str):
                ax.text(x + 0.25, y, it, ha="center", va="center", fontsize=16)
                x += 0.5
                continue
            rows, cols, name, color = it
            w, h = size[cols], size[rows]
            ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=to_rgba(color, 0.18), ec=color, lw=2))
            ax.text(x + w / 2, y + h / 2 + 0.12, name, ha="center", va="bottom", color=color, fontsize=13)
            ax.text(x + w / 2, y - h / 2 - 0.12, f"{rows} x {cols}", ha="center", va="top", fontsize=10)
            x += w

    dS, dW, dX = r"$\partial L/\partial S$", r"$\partial L/\partial W$", r"$\partial L/\partial X$"
    row(4.9, [(N, C, "$S$", blue), "=", (N, D, "$X$", ink), r"$\cdot$", (D, C, r"$W^\top$", blue), "+",
              (1, C, "$b$", blue)])
    row(1.85, [(C, D, dW, purple), "=", (C, N, r"$(\partial L/\partial S)^\top$", purple), r"$\cdot$",
               (N, D, "$X$", ink)])
    row(-0.4, [(N, D, dX, purple), "=", (N, C, dS, purple), r"$\cdot$", (C, D, "$W$", blue)])
    ax.text(0, -1.75, f"max |formula - autograd| on {N} CIFAR-10 images: {r['max_abs_err_dW']:.1e} (dW), "
                      f"{r['max_abs_err_dX']:.1e} (dX)", fontsize=11)
    ax.text(0, -2.2, f"Jacobian dS/dW, never formed: {N * C:,} x {C * D:,} = {r['jacobian_entries']:,} entries "
                     f"({r['jacobian_MB_fp32']:.0f} MB in float32)", fontsize=11)
    ax.set_xlim(-0.2, 8.4)
    ax.set_ylim(-2.5, 6.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    r = check()
    s = r["shapes"]
    print(f"forward : X {s['X']}, W {s['W']} -> S = X W^T + b {s['S']}, loss {r['loss']:.4f}")
    print(f"backward: dL/dS {s['dS']}")
    print(f"          dL/dS @ X -> {r['rejected']}")
    print(f"          dL/dW = (dL/dS)^T X {s['dW']} = shape of W;  dL/dX = (dL/dS) W {s['dX']} = shape of X")
    print(f"max |formula - autograd|: dW {r['max_abs_err_dW']:.1e}, db {r['max_abs_err_db']:.1e}, "
          f"dX {r['max_abs_err_dX']:.1e}")
    print(f"Jacobian dS/dW: {r['jacobian_entries']:,} entries = {r['jacobian_MB_fp32']:.0f} MB in float32 (never built)")
    out = Path(__file__).with_name("figures") / "matrix_backprop.png"
    out.parent.mkdir(exist_ok=True)
    figure(r, out)
    print("saved", out)
