"""Precompute the reflow (rectified-flow) model for FewStepSampling (run once, ~1-3 min on CPU):
    python w09_precompute.py   -> data/w09_reflow.npz
1. exact flow-matching ODE (closed-form velocity of the 8-mode ring, 200 Euler steps) maps noise x_0 to T(x_0);
2. a small MLP is trained on straight paths between the pairs (x_0, T(x_0))  (1-reflow, Liu et al. 2022);
3. for 300 display noise points: 50-step paths and 1/2/4/8-step Euler paths, exact field vs reflowed model.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from w09_gmm import ring, euler

torch.set_num_threads(4)
DATA = Path(__file__).parent / "data"


def main(steps=8000):
    g = ring()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((20000, 2))
    x1 = euler(g.velocity, x0, 200)
    print("ODE pairs done", flush=True)
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(3, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                        nn.Linear(256, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X0, X1 = torch.tensor(x0, dtype=torch.float32), torch.tensor(x1, dtype=torch.float32)
    for it in range(steps):
        idx = torch.randint(0, len(X0), (512,))
        a, b = X0[idx], X1[idx]
        t = torch.rand(512, 1)
        xt = (1 - t) * a + t * b
        loss = ((net(torch.cat([xt, t], 1)) - (b - a)) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == steps - 1:
            print(it, f"{loss.item():.4f}", flush=True)

    def v_net(x, t):
        with torch.no_grad():
            xx = torch.tensor(x, dtype=torch.float32)
            return net(torch.cat([xx, torch.full((len(x), 1), float(t))], 1)).numpy().astype(float)

    z = np.random.default_rng(7).standard_normal((300, 2))
    out = {"z": z, "M": g.M, "S": g.S}
    for name, fn in [("exact", g.velocity), ("reflow", v_net)]:
        out[f"{name}_path"] = euler(fn, z, 50, record=True)
        for n in [1, 2, 4, 8]:
            out[f"{name}_{n}_path"] = euler(fn, z, n, record=True)
            e = out[f"{name}_{n}_path"][-1]
            out[f"{name}_{n}"] = e
            d = np.sqrt(((e[:, None] - g.M[None]) ** 2).sum(-1)).min(1)
            out[f"{name}_{n}_hit"] = (d < 3 * g.S[0]).mean()
            print(name, n, "on a mode:", out[f"{name}_{n}_hit"])
    np.savez(DATA / "w09_reflow.npz", **out)
    torch.save(net.state_dict(), DATA / "w09_reflow_net.pt")


if __name__ == "__main__":
    main()
