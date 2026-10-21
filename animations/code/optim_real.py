"""Code for the week-1 optimization clips (renders/w01/<Scene>.mp4):

    LearningRateRegimes, IllConditioning, NoisyGradients, SaddlePoint, MomentumNesterov, AdaptiveMethods

Every number and path in these clips comes from this file: the Manim scenes (../w01_optimization.py) import it.

Data: scikit-learn's bundled diabetes dataset, 442 patients (Efron et al. 2004, "Least Angle Regression"; no
download). Two predictors: bmi (body-mass index, kg/m^2, std 4.41) and s5 (log serum triglycerides, std 0.52).
Target: disease progression one year after baseline. Features and target are centred, so the model has no bias:

    s = w1 * bmi + w2 * s5,        L(w) = 1/2 * mean((s - y)^2)

L is an exact quadratic bowl: its Hessian is H = E[x x^T] (the same at every w), its minimum w* solves H w = E[x y].

Run:  python optim_real.py      -> prints every number shown in the clips, saves figures/<clip>.png
Try:  the learning rates, batch sizes, rho, start points and the sweep grid are arguments of the clip functions
      below (see README.md); the optimizers are written exactly as on the lecture slides.
"""
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_diabetes

DTYPE = torch.float64                  # double precision: the paths are exact up to ~1e-13
RAW_WINDOW = dict(w1=(-5.0, 22.0), w2=(-6.0, 76.0))   # part of the raw bowl drawn in the clips (and the sweep box)


# =====================================================================================================
# Data, loss, gradient
def load_data(standardize=False):
    """x = (bmi, s5) centred (and divided by their std if standardize), y = disease progression, centred."""
    data = load_diabetes(scaled=False)                 # bundled with scikit-learn
    cols = [data.feature_names.index("bmi"), data.feature_names.index("s5")]
    X = torch.tensor(data.data[:, cols], dtype=DTYPE)
    y = torch.tensor(data.target, dtype=DTYPE)
    X = X - X.mean(0)
    y = y - y.mean()
    if standardize:
        X = X / X.std(0, correction=0)                 # each feature: mean 0, std 1
    return X, y


def loss(w, X, y):
    """L(w) = 1/2 mean((X w - y)^2)."""
    return 0.5 * ((X @ w - y) ** 2).mean()


def gradient(w, X, y):
    """dL/dw = X^T (X w - y) / N."""
    return X.T @ (X @ w - y) / len(y)


def bowl(X, y):
    """The exact quadratic: Hessian H = X^T X / N, minimum w*, L(w*), eigenvalues and condition number of H."""
    H = X.T @ X / len(y)
    w_star = torch.linalg.solve(H, X.T @ y / len(y))
    eig = torch.linalg.eigvalsh(H)                     # ascending
    return dict(H=H, w_star=w_star, L_star=loss(w_star, X, y), eig=eig, kappa=eig[-1] / eig[0])


def losses(path, X, y):
    return torch.stack([loss(w, X, y) for w in path])


def steps_to(gap, fraction=0.01):
    """First step at which the loss gap L - L* is at most `fraction` of its initial value (None if never)."""
    hit = np.nonzero(np.asarray(gap) <= fraction * float(gap[0]))[0]
    return int(hit[0]) if len(hit) else None


def full_batch(X, y):
    return lambda w: gradient(w, X, y)


def minibatches(X, y, batch_size, seed=0):
    """compute_gradient(w) on the next minibatch of a shuffled pass over the data (reshuffled every epoch)."""
    g = torch.Generator().manual_seed(seed)
    state = dict(order=torch.randperm(len(y), generator=g), pos=0)

    def compute_gradient(w):
        if state["pos"] + batch_size > len(y):         # epoch done: new shuffle
            state["order"], state["pos"] = torch.randperm(len(y), generator=g), 0
        idx = state["order"][state["pos"]:state["pos"] + batch_size]
        state["pos"] += batch_size
        return gradient(w, X[idx], y[idx])
    return compute_gradient


# =====================================================================================================
# Optimizers, as on the lecture slides (CS231n style). compute_gradient(w) returns dL/dw at w.
# Each returns the path: a (num_steps + 1) x len(w) tensor with the weights before the first and after every step.
def sgd(compute_gradient, w, learning_rate, num_steps):
    w, path = w.clone(), [w.clone()]
    for t in range(num_steps):
        dw = compute_gradient(w)
        w -= learning_rate * dw
        path.append(w.clone())
    return torch.stack(path)


def sgd_momentum(compute_gradient, w, learning_rate, num_steps, rho=0.9):
    w, path = w.clone(), [w.clone()]
    v = torch.zeros_like(w)
    for t in range(num_steps):
        dw = compute_gradient(w)
        v = rho * v + dw                               # velocity: running sum of past gradients
        w -= learning_rate * v
        path.append(w.clone())
    return torch.stack(path)


def nesterov(compute_gradient, w, learning_rate, num_steps, rho=0.9):
    w, path = w.clone(), [w.clone()]
    v = torch.zeros_like(w)
    for t in range(num_steps):
        dw = compute_gradient(w + rho * v)             # gradient at the look-ahead point
        v = rho * v - learning_rate * dw
        w += v
        path.append(w.clone())
    return torch.stack(path)


def adagrad(compute_gradient, w, learning_rate, num_steps):
    w, path = w.clone(), [w.clone()]
    grad_squared = torch.zeros_like(w)
    for t in range(num_steps):
        dw = compute_gradient(w)
        grad_squared += dw * dw                        # only grows: the steps keep shrinking
        w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
        path.append(w.clone())
    return torch.stack(path)


def rmsprop(compute_gradient, w, learning_rate, num_steps, decay_rate=0.99):
    w, path = w.clone(), [w.clone()]
    grad_squared = torch.zeros_like(w)
    for t in range(num_steps):
        dw = compute_gradient(w)
        grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw   # leaky AdaGrad
        w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
        path.append(w.clone())
    return torch.stack(path)


def adam(compute_gradient, w, learning_rate, num_steps, beta1=0.9, beta2=0.999):
    w, path = w.clone(), [w.clone()]
    moment1, moment2 = torch.zeros_like(w), torch.zeros_like(w)
    for t in range(1, num_steps + 1):
        dw = compute_gradient(w)
        moment1 = beta1 * moment1 + (1 - beta1) * dw          # momentum
        moment2 = beta2 * moment2 + (1 - beta2) * dw * dw     # AdaGrad / RMSProp
        moment1_unbias = moment1 / (1 - beta1 ** t)           # bias correction
        moment2_unbias = moment2 / (1 - beta2 ** t)           # bias correction
        w -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)
        path.append(w.clone())
    return torch.stack(path)


def _np(t):
    return t.detach().numpy() if torch.is_tensor(t) else np.asarray(t)


def _f(t):
    return float(t)


# =====================================================================================================
# Clip 1. LearningRateRegimes: one weight, s = w * bmi. L(w) = L* + a/2 (w - w*)^2 with a = mean(bmi^2), so one GD
# step multiplies the distance to w* by (1 - lr * a): it converges iff |1 - lr * a| < 1, i.e. lr < 2/a.
def learning_rate_regimes(factors=(0.1, 1.0, 1.9, 2.02), num_steps=30, w0=0.0):
    """GD on the one-weight parabola with lr = factor / a for each factor."""
    X, y = load_data()
    x = X[:, :1]                                       # bmi only (centred): N x 1
    a = (x[:, 0] ** 2).mean()                          # curvature of the parabola, (kg/m^2)^2
    w_star = (x[:, 0] * y).mean() / a
    runs = []
    for factor in factors:
        learning_rate = factor / a
        path = sgd(full_batch(x, y), torch.tensor([w0], dtype=DTYPE), learning_rate, num_steps)
        runs.append(dict(factor=factor, lr=_f(learning_rate), path=_np(path[:, 0]),
                         loss=_np(losses(path, x, y)), contraction=_f(1 - learning_rate * a)))
    return dict(a=_f(a), w_star=_f(w_star), L_star=_f(loss(w_star.view(1), x, y)), threshold=_f(2 / a),
                w0=w0, runs=runs)


# =====================================================================================================
# Clip 2. IllConditioning: GD from w = 0 on the raw-centred bowl (kappa ~ 90) and on the standardized bowl (kappa ~ 2.6).
# Each uses the best fixed learning rate for a quadratic, lr = 2 / (lambda_min + lambda_max): every direction then
# shrinks by (kappa - 1)/(kappa + 1) per step, so the steps to 1% of the loss gap depend only on kappa.
def ill_conditioning(num_steps=(120, 8), target=0.01):
    out = {}
    for name, standardize, n in (("raw", False, num_steps[0]), ("standardized", True, num_steps[1])):
        X, y = load_data(standardize)
        b = bowl(X, y)
        learning_rate = 2 / (b["eig"][0] + b["eig"][-1])
        path = sgd(full_batch(X, y), torch.zeros(2, dtype=DTYPE), learning_rate, n)
        gap = losses(path, X, y) - b["L_star"]
        out[name] = dict(H=_np(b["H"]), w_star=_np(b["w_star"]), L_star=_f(b["L_star"]), eig=_np(b["eig"]),
                         kappa=_f(b["kappa"]), lr=_f(learning_rate), path=_np(path), gap=_np(gap),
                         steps=steps_to(_np(gap), target))
    out["corr"] = float(np.corrcoef(_np(load_data()[0]).T)[0, 1])   # corr(bmi, s5): tilts the raw valley
    out["target"] = target
    return out


# =====================================================================================================
# Clip 3. NoisyGradients: standardized bowl, minibatch SGD from w = 0 with batch sizes 1, 16 and 442 (= full batch).
# Each step uses the true gradient of the next minibatch of a shuffled pass over the patients (fixed seed).
def noisy_gradients(batch_sizes=(442, 16, 1), learning_rate=0.05, num_steps=150, seed=0):
    X, y = load_data(standardize=True)
    b = bowl(X, y)
    runs = []
    for batch_size in batch_sizes:
        path = sgd(minibatches(X, y, batch_size, seed), torch.zeros(2, dtype=DTYPE), learning_rate, num_steps)
        gap = losses(path, X, y) - b["L_star"]
        runs.append(dict(batch_size=batch_size, path=_np(path), gap=_np(gap),
                         gap_last50=float(_np(gap)[-50:].mean())))
    return dict(H=_np(b["H"]), w_star=_np(b["w_star"]), L_star=_f(b["L_star"]), kappa=_f(b["kappa"]),
                lr=learning_rate, num_steps=num_steps, runs=runs, N=len(y))


# =====================================================================================================
# Clip 4. SaddlePoint: a network with one hidden unit, s = w2 * tanh(w1 * x), on standardized bmi -> standardized
# progression. At w = 0 the gradient is zero and the Hessian is [[0, -m], [-m, 0]], m = mean(x y): a saddle with
# curvatures +m and -m. The minima are a mirror pair (w, -w): tanh is odd.
def saddle_data():
    X, y = load_data(standardize=True)
    return X[:, 0], y / y.std(correction=0)            # standardized bmi, standardized progression


def saddle_loss(w, x, y):
    s = w[1] * torch.tanh(w[0] * x)
    return 0.5 * ((s - y) ** 2).mean()


def saddle_gradient(w, x, y):
    """Backprop by hand: h = tanh(w1 x), s = w2 h."""
    h = torch.tanh(w[0] * x)
    ds = (w[1] * h - y) / len(y)                       # dL/ds
    return torch.stack([(ds * w[1] * (1 - h ** 2) * x).sum(),   # dL/dw1 = dL/ds * w2 * tanh' * x
                        (ds * h).sum()])                        # dL/dw2 = dL/ds * h


def saddle_point(start=(0.05, 0.02), learning_rate=0.1, rho=0.9, num_steps=100, grid=(161, 201)):
    x, y = saddle_data()
    f = lambda w: saddle_loss(w, x, y)
    origin = torch.zeros(2, dtype=DTYPE)
    hess0 = torch.autograd.functional.hessian(f, origin)
    # the minimum: Newton's method (exact autograd Hessian) from a point in the valley, then its mirror image
    w = torch.tensor([0.3, 2.0], dtype=DTYPE)
    for _ in range(50):
        w = w - torch.linalg.solve(torch.autograd.functional.hessian(f, w), saddle_gradient(w, x, y))
    w_min = w
    w0 = torch.tensor(start, dtype=DTYPE)
    compute_gradient = lambda w: saddle_gradient(w, x, y)
    runs = {}
    for name, path in (("GD", sgd(compute_gradient, w0, learning_rate, num_steps)),
                       ("momentum", sgd_momentum(compute_gradient, w0, learning_rate, num_steps, rho))):
        L = torch.stack([f(w) for w in path])
        halfway = 0.5 * (f(origin) + f(w_min))         # "escaped": the loss is halfway down from the saddle
        below = torch.nonzero(L < halfway)
        runs[name] = dict(path=_np(path), loss=_np(L), escape=int(below[0, 0]) if len(below) else None)
    # the loss surface on a grid, for the contour lines
    w1s = torch.linspace(-1.6, 1.6, grid[0], dtype=DTYPE)
    w2s = torch.linspace(-4.4, 4.4, grid[1], dtype=DTYPE)
    S = w2s[:, None, None] * torch.tanh(w1s[None, :, None] * x[None, None, :])     # (len w2, len w1, N)
    Z = 0.5 * ((S - y) ** 2).mean(-1)
    return dict(hessian0=_np(hess0), mean_xy=_f((x * y).mean()), grad0=_np(saddle_gradient(origin, x, y)),
                L0=_f(f(origin)), w_min=_np(w_min), L_min=_f(f(w_min)), hessian_min=_np(torch.linalg.eigvalsh(
                    torch.autograd.functional.hessian(f, w_min))), start=start, lr=learning_rate, rho=rho,
                runs=runs, w1s=_np(w1s), w2s=_np(w2s), Z=_np(Z))


# =====================================================================================================
# Clip 5. MomentumNesterov: the raw bowl again, from w = 0, the same learning rate for all three.
# (0.05 is stable for all three: SGD needs lr < 2/lambda_max = 0.102, Nesterov with rho 0.9 lr < 0.069.)
def momentum_nesterov(learning_rate=0.05, rho=0.9, num_steps=150, target=0.01):
    X, y = load_data()
    b = bowl(X, y)
    w0 = torch.zeros(2, dtype=DTYPE)
    runs = []
    for name, path in (("SGD", sgd(full_batch(X, y), w0, learning_rate, num_steps)),
                       ("SGD+Momentum", sgd_momentum(full_batch(X, y), w0, learning_rate, num_steps, rho)),
                       ("Nesterov", nesterov(full_batch(X, y), w0, learning_rate, num_steps, rho))):
        gap = _np(losses(path, X, y) - b["L_star"])
        runs.append(dict(name=name, path=_np(path), gap=gap, steps=steps_to(gap, target), final_gap=float(gap[-1])))
    return dict(H=_np(b["H"]), w_star=_np(b["w_star"]), L_star=_f(b["L_star"]), kappa=_f(b["kappa"]),
                lr=learning_rate, rho=rho, runs=runs, target=target)


# =====================================================================================================
# Clip 6. AdaptiveMethods: the raw bowl, from w = 0. Each optimizer gets its own learning rate from a small sweep:
# the lowest loss after num_steps among the grid values whose path stays inside RAW_WINDOW (the part of the bowl the
# clip shows). AdaGrad, RMSProp and Adam divide each coordinate of the step by its own gradient scale.
LR_GRID = tuple(m * 10.0 ** e for e in range(-3, 3) for m in (1, 3))      # 0.001, 0.003, ..., 30, 100, 300
OPTIMIZERS = {"SGD+Momentum": sgd_momentum, "AdaGrad": adagrad, "RMSProp": rmsprop, "Adam": adam}


def in_window(path, window=RAW_WINDOW):
    """True if the path is finite and (unless window is None) stays inside the window."""
    p = _np(path)
    if not np.all(np.isfinite(p)):
        return False
    return window is None or bool(np.all((p[:, 0] >= window["w1"][0]) & (p[:, 0] <= window["w1"][1])
                                         & (p[:, 1] >= window["w2"][0]) & (p[:, 1] <= window["w2"][1])))


def adaptive_methods(num_steps=100, lr_grid=LR_GRID, window=RAW_WINDOW, target=0.01):
    X, y = load_data()
    b = bowl(X, y)
    w0 = torch.zeros(2, dtype=DTYPE)
    runs, sweeps = [], {}
    for name, optimizer in OPTIMIZERS.items():
        sweep = []
        for learning_rate in lr_grid:
            path = optimizer(full_batch(X, y), w0, learning_rate, num_steps)
            final = _f(loss(path[-1], X, y) - b["L_star"]) if in_window(path, window) else float("nan")
            sweep.append((learning_rate, final))
        sweeps[name] = sweep
        best_lr = min((s for s in sweep if np.isfinite(s[1])), key=lambda s: s[1])[0]
        path = optimizer(full_batch(X, y), w0, best_lr, num_steps)
        gap = _np(losses(path, X, y) - b["L_star"])
        runs.append(dict(name=name, lr=best_lr, path=_np(path), gap=gap, steps=steps_to(gap, target),
                         final_gap=float(gap[-1])))
    return dict(H=_np(b["H"]), w_star=_np(b["w_star"]), L_star=_f(b["L_star"]), kappa=_f(b["kappa"]),
                runs=runs, sweeps=sweeps, num_steps=num_steps, target=target)


# =====================================================================================================
# Static figures (one per clip) and the printout
COLORS = dict(SGD="#1F2937", **{"SGD+Momentum": "#2563EB"}, Nesterov="#16A34A", AdaGrad="#EA580C",
              RMSProp="#DC2626", Adam="#7C3AED")


def _ellipses(ax, H, w_star, gaps, swap=True, **kw):
    """Contour lines of a quadratic bowl: {w : (w - w*)^T H (w - w*) / 2 = gap}. swap: draw w2 horizontally."""
    evals, evecs = np.linalg.eigh(H)
    th = np.linspace(0, 2 * np.pi, 200)
    for g in gaps:
        pts = w_star[:, None] + evecs @ (np.sqrt(2 * g / evals)[:, None] * np.stack([np.cos(th), np.sin(th)]))
        ax.plot(*(pts[::-1] if swap else pts), **kw)


def figures(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(exist_ok=True)
    saved = []

    r = results["LearningRateRegimes"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    ws = np.linspace(-10, 30, 300)
    for ax, run, c in zip(axes, r["runs"], ["#2563EB", "#16A34A", "#EA580C", "#DC2626"]):
        ax.plot(ws, r["L_star"] + 0.5 * r["a"] * (ws - r["w_star"]) ** 2, color="#9CA3AF")
        ax.plot(run["path"], run["loss"], "-o", color=c, ms=3, lw=1)
        ax.set_title(f"lr = {run['factor']:g}/a = {run['lr']:.4f}", color=c)
        ax.set_xlabel("w"); ax.set_ylim(1700, 6000)
    axes[0].set_ylabel("L(w)")
    fig.suptitle(f"s = w·bmi: parabola with curvature a = {r['a']:.2f}; GD converges iff lr < 2/a = {r['threshold']:.4f}")
    fig.tight_layout(); fig.savefig(outdir / "learning_rate_regimes.png", dpi=130); plt.close(fig)
    saved.append("learning_rate_regimes.png")

    r = results["IllConditioning"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[2.4, 1]))
    for ax, key in zip(axes, ["raw", "standardized"]):
        d = r[key]
        gap0 = d["gap"][0]
        _ellipses(ax, d["H"], d["w_star"], gap0 * 0.35 ** np.arange(8), color="#9CA3AF", lw=0.8)
        _ellipses(ax, d["H"], d["w_star"], [r["target"] * gap0], color="#16A34A", lw=1.5)
        p = d["path"][: d["steps"] + 1]
        ax.plot(p[:, 1], p[:, 0], "-o", color="#2563EB", ms=2, lw=1)
        ax.plot(d["w_star"][1], d["w_star"][0], "*", color="#1F2937", ms=10)
        ax.set_aspect("equal"); ax.set_xlabel("w2 (s5)"); ax.set_ylabel("w1 (bmi)")
        ax.set_title(f"{key}: κ = {d['kappa']:.1f}, lr = {d['lr']:.3g}: 1% of the gap after {d['steps']} steps")
    axes[0].set_xlim(*RAW_WINDOW["w2"]); axes[0].set_ylim(*RAW_WINDOW["w1"])
    fig.tight_layout(); fig.savefig(outdir / "ill_conditioning.png", dpi=130); plt.close(fig)
    saved.append("ill_conditioning.png")

    r = results["NoisyGradients"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    gap0 = r["runs"][0]["gap"][0]
    _ellipses(ax, r["H"], r["w_star"], gap0 * 0.5 ** np.arange(9), color="#9CA3AF", lw=0.8)
    for run, c in zip(r["runs"], ["#1F2937", "#7C3AED", "#EA580C"]):
        ax.plot(run["path"][:, 1], run["path"][:, 0], color=c, lw=1, label=f"batch {run['batch_size']}")
        ax2.semilogy(run["gap"], color=c, lw=1, label=f"batch {run['batch_size']}")
    ax.plot(r["w_star"][1], r["w_star"][0], "*", color="#1F2937", ms=10)
    ax.set_aspect("equal"); ax.set_xlabel("w2 (s5, standardized)"); ax.set_ylabel("w1 (bmi, standardized)")
    ax.legend(); ax2.legend(); ax2.set_xlabel("step"); ax2.set_ylabel("L − L* (full data)")
    fig.suptitle(f"minibatch SGD, lr = {r['lr']}, same start and seed")
    fig.tight_layout(); fig.savefig(outdir / "noisy_gradients.png", dpi=130); plt.close(fig)
    saved.append("noisy_gradients.png")

    r = results["SaddlePoint"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1, 1.2]))
    lv = np.unique(np.round(np.concatenate([np.linspace(0.335, 0.5, 8), [0.33, 0.6, 0.8, 1.1, 1.6, 2.4, 3.6]]), 4))
    ax.contour(r["w1s"], r["w2s"], r["Z"], levels=lv, colors="#9CA3AF", linewidths=0.7)
    for (name, run), c in zip(r["runs"].items(), ["#1F2937", "#2563EB"]):
        ax.plot(run["path"][:, 0], run["path"][:, 1], "-o", color=c, ms=1.5, lw=1, label=name)
        ax2.plot(run["loss"], color=c, label=f"{name}: halfway down at step {run['escape']}")
    ax.plot([r["w_min"][0], -r["w_min"][0]], [r["w_min"][1], -r["w_min"][1]], "*", color="#16A34A", ms=12)
    ax.plot(0, 0, "x", color="#DC2626", ms=10, mew=2)
    ax.set_xlabel("w1"); ax.set_ylabel("w2"); ax.legend()
    ax2.set_xlabel("step"); ax2.set_ylabel("loss"); ax2.legend()
    fig.suptitle(f"s = w2·tanh(w1·x): saddle at 0 (Hessian eigenvalues ±{r['mean_xy']:.3f}), lr = {r['lr']}")
    fig.tight_layout(); fig.savefig(outdir / "saddle_point.png", dpi=130); plt.close(fig)
    saved.append("saddle_point.png")

    for key, fname in (("MomentumNesterov", "momentum_nesterov.png"), ("AdaptiveMethods", "adaptive_methods.png")):
        r = results[key]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw=dict(width_ratios=[2.4, 1]))
        gap0 = r["runs"][0]["gap"][0]
        _ellipses(ax, r["H"], r["w_star"], gap0 * 0.35 ** np.arange(8), color="#9CA3AF", lw=0.8)
        for run in r["runs"]:
            lab = run["name"] + (f" (lr {run['lr']:g})" if "lr" in run else "")
            ax.plot(run["path"][:, 1], run["path"][:, 0], "-o", color=COLORS[run["name"]], ms=1.5, lw=1, label=lab)
            ax2.semilogy(np.maximum(run["gap"], 1e-12), color=COLORS[run["name"]], lw=1.2, label=lab)
        ax.plot(r["w_star"][1], r["w_star"][0], "*", color="#1F2937", ms=10)
        ax.set_aspect("equal"); ax.set_xlim(*RAW_WINDOW["w2"]); ax.set_ylim(*RAW_WINDOW["w1"])
        ax.set_xlabel("w2 (s5)"); ax.set_ylabel("w1 (bmi)"); ax.legend(fontsize=8)
        ax2.set_xlabel("step"); ax2.set_ylabel("L − L*"); ax2.set_ylim(1e-6, 3e3)
        fig.tight_layout(); fig.savefig(outdir / fname, dpi=130); plt.close(fig)
        saved.append(fname)
    return saved


def main():
    results = dict(LearningRateRegimes=learning_rate_regimes(), IllConditioning=ill_conditioning(),
                   NoisyGradients=noisy_gradients(), SaddlePoint=saddle_point(),
                   MomentumNesterov=momentum_nesterov(), AdaptiveMethods=adaptive_methods())
    X, y = load_data()
    print(f"data: {len(y)} patients; std bmi {X[:, 0].std(correction=0):.2f} kg/m^2, std s5 {X[:, 1].std(correction=0):.2f}, "
          f"std progression {y.std(correction=0):.1f}")

    r = results["LearningRateRegimes"]
    print(f"\n[LearningRateRegimes] s = w*bmi: a = mean(bmi^2) = {r['a']:.2f}, w* = {r['w_star']:.2f}, "
          f"L(w*) = {r['L_star']:.1f}, threshold 2/a = {r['threshold']:.4f}")
    for run in r["runs"]:
        print(f"  lr = {run['factor']:>4g}/a = {run['lr']:.4f}: distance to w* x {run['contraction']:+.2f} per step, "
              f"after {len(run['path']) - 1} steps w = {run['path'][-1]:.2f}, L = {run['loss'][-1]:.1f}")

    r = results["IllConditioning"]
    print(f"\n[IllConditioning] corr(bmi, s5) = {r['corr']:.3f}")
    for key in ("raw", "standardized"):
        d = r[key]
        print(f"  {key:12s}: eigenvalues {d['eig'].round(3)}, kappa = {d['kappa']:.2f}, w* = {d['w_star'].round(2)}, "
              f"L* = {d['L_star']:.1f}, lr = 2/(l_min + l_max) = {d['lr']:.4f}, 1% of the gap after {d['steps']} steps")

    r = results["NoisyGradients"]
    print(f"\n[NoisyGradients] standardized bowl (kappa {r['kappa']:.2f}), lr = {r['lr']}, {r['num_steps']} steps, "
          f"w* = {r['w_star'].round(2)}, L* = {r['L_star']:.1f}")
    for run in r["runs"]:
        print(f"  batch {run['batch_size']:>3d}: final w = {run['path'][-1].round(2)}, mean L - L* over the last 50 "
              f"steps = {run['gap_last50']:.3g}")

    r = results["SaddlePoint"]
    print(f"\n[SaddlePoint] s = w2*tanh(w1*x): L(0) = {r['L0']:.4f}, gradient at 0 = {r['grad0']}, "
          f"Hessian at 0 = {r['hessian0'].round(4).tolist()} (eigenvalues +-{r['mean_xy']:.4f} = mean(x*y))")
    print(f"  minima at +-{r['w_min'].round(4)}, L = {r['L_min']:.4f}, Hessian eigenvalues there {r['hessian_min'].round(4)}")
    for name, run in r["runs"].items():
        print(f"  {name:8s} lr {r['lr']} from {r['start']}: loss halfway down at step {run['escape']}, "
              f"after {len(run['loss']) - 1} steps w = {run['path'][-1].round(3)}, L = {run['loss'][-1]:.4f}")

    r = results["MomentumNesterov"]
    print(f"\n[MomentumNesterov] raw bowl, lr = {r['lr']} for all, rho = {r['rho']}")
    for run in r["runs"]:
        print(f"  {run['name']:13s}: 1% of the gap after {run['steps']} steps, gap after {len(run['gap']) - 1} steps "
              f"= {run['final_gap']:.2e}")

    r = results["AdaptiveMethods"]
    print(f"\n[AdaptiveMethods] raw bowl, {r['num_steps']} steps, lr from the sweep {[f'{v:g}' for v in LR_GRID]}")
    for run in r["runs"]:
        print(f"  {run['name']:13s}: lr = {run['lr']:g}, 1% of the gap after {run['steps']} steps, gap after "
              f"{r['num_steps']} steps = {run['final_gap']:.2e}")

    out = Path(__file__).with_name("figures")
    for f in figures(results, out):
        print("saved", out / f)


if __name__ == "__main__":
    main()
