"""Precompute the trained models behind the week-8 clips (run once, ~1-2 min on CPU):
    python w08_precompute.py            -> data/w08_aevae.npz, data/w08_gan.npz
AE vs VAE: sklearn's 8x8 handwritten digits (UCI, 1797 images, ships with sklearn), all 10 classes, 2-D latent.
GAN: a real 1-D GAN (MLP generator and discriminator), unimodal target, then a bimodal target.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
CLASSES = list(range(10))


def mlp(sizes, act=nn.ReLU, last=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if last is not None:
        layers.append(last)
    return nn.Sequential(*layers)


def aevae(steps=15000, seed=0):
    from sklearn.datasets import load_digits
    d = load_digits()
    keep = np.isin(d.target, CLASSES)
    X = torch.tensor(d.data[keep] / 16.0, dtype=torch.float32)
    y = d.target[keep]
    out = {"X": X.numpy(), "y": y}
    for kind in ["ae", "vae"]:
        torch.manual_seed(seed)
        enc = mlp([64, 256, 128, 4 if kind == "vae" else 2])
        dec = mlp([2, 128, 256, 64])
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        for it in range(steps):
            idx = torch.randint(0, len(X), (256,))
            xb = X[idx]
            h = enc(xb)
            if kind == "vae":
                mu, logvar = h[:, :2], h[:, 2:]
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()
            else:
                z, kl = h, torch.tensor(0.0)
            rec = F.binary_cross_entropy_with_logits(dec(z), xb, reduction="none").sum(1).mean()
            loss = rec + kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 1000 == 0 or it == steps - 1:
                print(kind, it, f"rec {rec.item():.2f} kl {kl.item():.2f}", flush=True)
        with torch.no_grad():
            h = enc(X)
            Z = h[:, :2].numpy()
            out[f"{kind}_z"] = Z
            out[f"{kind}_rec"] = rec.item()
            out[f"{kind}_kl"] = kl.item()
            # decoder on a fine grid of the latent box (for interpolation lines and random samples)
            lo, hi = np.percentile(Z, 0.5, 0) - 0.6 * Z.std(0), np.percentile(Z, 99.5, 0) + 0.6 * Z.std(0)
            out[f"{kind}_box"] = np.stack([lo, hi])
        torch.save({"dec": dec.state_dict(), "enc": enc.state_dict()}, DATA / f"w08_{kind}.pt")
    np.savez(DATA / "w08_aevae.npz", **out)
    aevae_post()


def aevae_post(pair=(0, 3), n_stat=2000):
    """Walks and prior draws from the saved models. A logistic-regression digit classifier trained on the real
    images (98% train accuracy) labels decoded images, so the clip can quote real numbers."""
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    out = dict(np.load(DATA / "w08_aevae.npz"))
    X, y = out["X"], out["y"]
    dg = load_digits()
    clf = LogisticRegression(max_iter=3000).fit(dg.data / 16.0, dg.target)
    # walk endpoints: the most typical 0 and 3 (closest to their class mean in pixel space)
    med = {c: np.flatnonzero(y == c)[np.argmin(((X[y == c] - X[y == c].mean(0)) ** 2).sum(1))] for c in pair}
    ends = [med[pair[0]], med[pair[1]]]
    out["ends"], out["end_classes"] = np.array(ends), np.array(pair)
    ts = np.linspace(0, 1, 9)
    eps = np.random.default_rng(1).standard_normal((8, 2))           # the same 8 draws z ~ N(0, I) for both
    eps_stat = np.random.default_rng(2).standard_normal((n_stat, 2))
    for kind in ["ae", "vae"]:
        dec = mlp([2, 128, 256, 64])
        dec.load_state_dict(torch.load(DATA / f"w08_{kind}.pt")["dec"])
        D = lambda z: torch.sigmoid(dec(torch.tensor(z, dtype=torch.float32))).detach().numpy()
        Z = out[f"{kind}_z"]
        line = Z[ends[0]][None] * (1 - ts[:, None]) + Z[ends[1]][None] * ts[:, None]
        out[f"{kind}_line"], out[f"{kind}_line_img"] = line, D(line)
        out[f"{kind}_line_conf"] = clf.predict_proba(out[f"{kind}_line_img"]).max(1)
        out[f"{kind}_draw"], out[f"{kind}_draw_img"] = eps, D(eps)
        out[f"{kind}_draw_cls"] = clf.predict(out[f"{kind}_draw_img"])
        out[f"{kind}_stat_hist"] = np.bincount(clf.predict(D(eps_stat)), minlength=10)
        print(kind, "walk conf", np.round(out[f"{kind}_line_conf"], 2), "draw classes", out[f"{kind}_draw_cls"],
              f"{n_stat} N(0,I) draws per class", out[f"{kind}_stat_hist"])
    np.savez(DATA / "w08_aevae.npz", **out)


def gan(target, steps, seed, snap_every, lr_g=1e-3, lr_d=1e-3, init=-2.5, zdim=1, hid=32, d_hid=64, r1=0.0, d_act=nn.LeakyReLU):
    torch.manual_seed(seed)
    np.random.seed(seed)
    G = mlp([zdim, hid, hid, 1], act=nn.LeakyReLU)
    D = mlp([1, d_hid, d_hid, 1], act=d_act)
    with torch.no_grad():                       # start the generator as a narrow blob near x = init
        G[-1].weight.mul_(0.1)
        G[-1].bias.fill_(init)
    og = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    grid = torch.linspace(-5, 5, 201)[:, None]
    zfix = torch.randn(4000, zdim)
    snaps_g, snaps_d, snaps_step = [], [], []

    def kde(s):
        s = s[:, 0]
        bw = 0.12
        return (torch.exp(-0.5 * ((grid - s[None]) / bw) ** 2).mean(1) / (bw * np.sqrt(2 * np.pi))).numpy()

    for it in range(steps + 1):
        if it % snap_every == 0:
            with torch.no_grad():
                snaps_g.append(kde(G(zfix)))
                snaps_d.append(torch.sigmoid(D(grid))[:, 0].numpy())
                snaps_step.append(it)
        xr = target(256).requires_grad_(r1 > 0)
        xf = G(torch.randn(256, zdim))
        dr = D(xr)
        ld = F.softplus(-dr).mean() + F.softplus(D(xf.detach())).mean()
        if r1 > 0:                              # R1 penalty (Mescheder et al. 2018): keeps D smooth on the data
            g, = torch.autograd.grad(dr.sum(), xr, create_graph=True)
            ld = ld + 0.5 * r1 * (g ** 2).sum(1).mean()
        od.zero_grad(); ld.backward(); od.step()
        lg = F.softplus(-D(G(torch.randn(256, zdim)))).mean()      # non-saturating generator loss
        og.zero_grad(); lg.backward(); og.step()
    return np.array(snaps_g), np.array(snaps_d), np.array(snaps_step), grid[:, 0].numpy()


def gans():
    uni = lambda n: 1.5 + 0.6 * torch.randn(n, 1)
    def bi(n):
        s = torch.where(torch.rand(n, 1) < 0.5, -2.0, 2.0)
        return s + 0.45 * torch.randn(n, 1)
    kw = dict(lr_g=1e-3, lr_d=1e-3, r1=1.0, d_act=nn.Tanh)
    g1, d1, s1, grid = gan(uni, 1200, 0, 5, init=-2.5, **kw)
    g2, d2, s2, _ = gan(bi, 2400, 0, 10, init=0.0, **kw)
    np.savez(DATA / "w08_gan.npz", uni_g=g1, uni_d=d1, uni_s=s1, bi_g=g2, bi_d=d2, bi_s=s2, grid=grid)
    for nm, g, s in [("uni", g1, s1), ("bi", g2, s2)]:
        mass_left = g[:, grid < 0].sum(1) * (grid[1] - grid[0])
        print(nm, "mass left of 0 every 10th snapshot:", np.round(mass_left[::10], 2))
    print("final D range uni", d1[-1].min(), d1[-1].max())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "aevae"):
        aevae()
    if what == "aevae_post":
        aevae_post()
    if what in ("all", "gan"):
        gans()
