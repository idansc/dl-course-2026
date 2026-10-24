"""Code for the lecture-2 "Adam" slide: SGD+momentum, AdaGrad, RMSProp and Adam training our 2-layer MLP on CIFAR-10
(the same loop as the opening slide; only the optimizer line changes). Each optimizer gets its best learning rate from a
small grid (lowest training loss after 10 epochs, mean of 3 seeds); the slide shows the training loss per epoch.

Why not the diabetes bowl (optim_real.adaptive_methods)? With 2 weights, a convex loss and exact gradients, Adam's
momentum only overshoots and RMSProp wins: the clip contradicted the slide. Adam's momentum pays off with noisy
minibatch gradients in a real network, which is the setting the slide is about.

Run:  python adaptive_cifar.py   -> prints the sweep, writes data/adaptive_cifar.json, saves figures/adaptive_cifar.png
Try:  epochs=20 (the gaps shrink: all four keep improving); batch=32 (noisier gradients); Adam without bias correction
      (betas only, no unbias lines, see adam_bias_correction.py).
"""
import json, time
from pathlib import Path

import numpy as np
import torch

from muon_cifar import train                              # the training loop of the slides (45k / 5k / 10k split)

OUT = Path(__file__).with_name("data") / "adaptive_cifar.json"
OPTIMIZERS = {                                            # name: (optimizer for lr, learning-rate grid)
    "SGD+Momentum": (lambda lr: lambda m: [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)], [0.01, 0.03, 0.1]),
    "AdaGrad": (lambda lr: lambda m: [torch.optim.Adagrad(m.parameters(), lr=lr)], [0.01, 0.03, 0.1, 0.3]),
    "RMSProp": (lambda lr: lambda m: [torch.optim.RMSprop(m.parameters(), lr=lr, alpha=0.99)], [3e-4, 1e-3, 3e-3]),
    "Adam": (lambda lr: lambda m: [torch.optim.Adam(m.parameters(), lr=lr)], [1e-3, 3e-3, 1e-2]),
}


def sweep(epochs=10, seeds=(0, 1, 2)):
    runs, best = {}, {}
    for name, (make, lrs) in OPTIMIZERS.items():
        for lr in lrs:
            hs = [train(make(lr), epochs=epochs, seed=s)[1] for s in seeds]
            runs[f"{name}|{lr}"] = {k: np.mean([h[k] for h in hs], 0).tolist() for k in ("train_loss", "val_acc")}
        lr = min(lrs, key=lambda lr: runs[f"{name}|{lr}"]["train_loss"][-1])
        assert lr not in (lrs[0], lrs[-1]), f"{name}: best lr {lr} is on the edge of the grid"
        best[name] = dict(lr=lr, **runs[f"{name}|{lr}"])
    return dict(epochs=epochs, seeds=list(seeds), runs=runs, best=best)


if __name__ == "__main__":
    t0 = time.time()
    res = sweep()
    for name, r in res["best"].items():
        tl, va = r["train_loss"], r["val_acc"]
        print(f"{name:13s} lr {r['lr']:<6g} train loss  epoch 1 {tl[0]:.3f}  epoch 3 {tl[2]:.3f}  epoch 10 {tl[-1]:.3f}  |  "
              f"val acc epoch 1 {100 * va[0]:.1f}%  best {100 * max(va):.1f}%")
    res["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(res))
    print(f"saved {OUT} ({res['seconds']:.0f}s)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"SGD+Momentum": "#2563EB", "AdaGrad": "#EA580C", "RMSProp": "#DC2626", "Adam": "#7C3AED"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in res["best"].items():
        ax.plot(range(1, res["epochs"] + 1), r["train_loss"], "-o", ms=3, color=colors[name], label=f"{name} (lr {r['lr']:g})")
    ax.set(xlabel="epoch", ylabel="training loss", title="CIFAR-10, 2-layer MLP: best lr per optimizer, mean of 3 seeds")
    ax.legend(); fig.tight_layout()
    out = Path(__file__).with_name("figures") / "adaptive_cifar.png"
    fig.savefig(out, dpi=130)
    print("saved", out)
