"""Code for the clip BackpropFlow (renders/w01/BackpropFlow.mp4): backprop by hand on one neuron,
f = sigmoid(w0*x0 + w1*x1 + w2), checked against PyTorch autograd.

Run:  python backprop_flow.py      -> prints every forward value and gradient shown in the clip, checks them against
                                      autograd and against the numeric gradient (gradient check),
                                      saves figures/backprop_flow.png
Try:  change the inputs below; each gate only needs its local derivative (downstream = local x upstream).
"""
from pathlib import Path
import math

w0, x0, w1, x1, w2 = 2.0, -1.0, -3.0, -2.0, -3.0      # the numbers in the clip


def forward_backward(w0, x0, w1, x1, w2):
    # forward pass: one simple gate at a time
    a = w0 * x0                    # multiply gate
    b = w1 * x1                    # multiply gate
    c = a + b                      # add gate
    d = c + w2                     # add gate
    f = 1 / (1 + math.exp(-d))     # sigmoid gate
    # backward pass: downstream gradient = local gradient x upstream gradient
    df = 1.0
    dd = f * (1 - f) * df          # sigmoid: local gradient f(1 - f)
    dc, dw2 = dd, dd               # add gate distributes the gradient
    da, db = dc, dc
    dw0, dx0 = x0 * da, w0 * da    # multiply gate swaps the inputs
    dw1, dx1 = x1 * db, w1 * db
    values = dict(a=a, b=b, c=c, d=d, f=f)
    grads = dict(w0=dw0, x0=dx0, w1=dw1, x1=dx1, w2=dw2)
    return values, grads


def numeric_gradient(h=1e-5):
    """Gradient check: centered differences (f(v + h) - f(v - h)) / 2h, one input at a time."""
    v = dict(w0=w0, x0=x0, w1=w1, x1=x1, w2=w2)
    out = {}
    for k in v:
        up, down = dict(v), dict(v)
        up[k] += h
        down[k] -= h
        out[k] = (forward_backward(**up)[0]["f"] - forward_backward(**down)[0]["f"]) / (2 * h)
    return out


def check_with_autograd():
    import torch
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in dict(w0=w0, x0=x0, w1=w1, x1=x1, w2=w2).items()}
    f = torch.sigmoid(t["w0"] * t["x0"] + t["w1"] * t["x1"] + t["w2"])
    f.backward()
    return {k: v.grad.item() for k, v in t.items()}


def figure(values, grads, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = {"w0": (0, 4), "x0": (0, 3), "w1": (0, 2), "x1": (0, 1), "w2": (0, 0), "*0": (2, 3.5), "*1": (2, 1.5),
           "+1": (4, 2.5), "+2": (6, 1.5), "sig": (8, 1.5), "f": (10, 1.5)}
    edges = [("w0", "*0"), ("x0", "*0"), ("w1", "*1"), ("x1", "*1"), ("*0", "+1"), ("*1", "+1"), ("+1", "+2"),
             ("w2", "+2"), ("+2", "sig"), ("sig", "f")]
    fwd = dict(w0=w0, x0=x0, w1=w1, x1=x1, w2=w2, **{"*0": values["a"], "*1": values["b"], "+1": values["c"],
                                                       "+2": values["d"], "sig": values["f"]})
    bwd = dict(grads, **{"*0": grads["w2"], "*1": grads["w2"], "+1": grads["w2"], "+2": grads["w2"], "sig": 1.0})
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(-1.4, 10.6)
    leaves = ("w0", "x0", "w1", "x1", "w2")
    for s, t in edges:
        (x0_, y0_), (x1_, y1_) = pos[s], pos[t]
        ax.plot([x0_, x1_], [y0_, y1_], color="#9CA3AF", lw=1.5, zorder=1)
        if s in leaves:             # inputs: value and gradient left of the box
            ax.text(x0_ - 0.45, y0_ + 0.1, f"{fwd[s]:.2f}", color="#16A34A", ha="right", fontsize=10)
            ax.text(x0_ - 0.45, y0_ - 0.3, f"{bwd[s]:.2f}", color="#DC2626", ha="right", fontsize=10)
            continue
        mx, my = (x0_ + x1_) / 2, (y0_ + y1_) / 2
        ax.text(mx, my + 0.18, f"{fwd[s]:.2f}", color="#16A34A", ha="center", fontsize=10)
        ax.text(mx, my - 0.32, f"{bwd[s]:.2f}", color="#DC2626", ha="center", fontsize=10)
    for k, (x, y) in pos.items():
        label = {"*0": "×", "*1": "×", "+1": "+", "+2": "+", "sig": "σ"}.get(k, k)
        ax.text(x, y, label, ha="center", va="center", fontsize=12,
                bbox=dict(boxstyle="circle" if k in ("*0", "*1", "+1", "+2", "sig") else "round",
                          fc="white", ec="#1F2937"))
    ax.set_title("forward values (green) and gradients ∂f/∂(·) (red)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    values, grads = forward_backward(w0, x0, w1, x1, w2)
    print("forward :", {k: round(v, 4) for k, v in values.items()})
    print("backward:", {k: round(v, 4) for k, v in grads.items()})
    auto = check_with_autograd()
    print("autograd:", {k: round(v, 4) for k, v in auto.items()})
    print("max |by hand - autograd| =", max(abs(grads[k] - auto[k]) for k in grads))
    num = numeric_gradient()
    rel = max(abs(grads[k] - num[k]) / max(abs(grads[k]), abs(num[k])) for k in grads)
    print(f"gradient check: max relative error |analytic - numeric| / max(|a|, |n|) = {rel:.1e}")
    out = Path(__file__).with_name("figures") / "backprop_flow.png"
    out.parent.mkdir(exist_ok=True)
    figure(values, grads, out)
    print("saved", out)
