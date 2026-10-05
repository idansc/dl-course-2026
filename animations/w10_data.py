"""Precompute data for w10_world_models.py (run once; writes data/w10_*.npz).

    python w10_data.py splat   # 2-D Gaussian splatting fit of grace_hopper.jpg, snapshots over steps
    python w10_data.py world   # bouncing-ball world: true simulator, a learned MLP world model, rollouts
    python w10_data.py mpc     # random-shooting MPC around a wall, every replanning round recorded
"""
import sys
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------------------------- splatting
def splat_render(p, res, bg=1.0):
    """Front-to-back alpha compositing of N 2-D Gaussians on a res×res grid in [0,1]^2 (y down)."""
    import torch
    mu, log_s, th, col, op = p["mu"], p["log_s"], p["theta"], p["color"], p["opacity"]
    g = (torch.arange(res, dtype=torch.float32) + 0.5) / res
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    pix = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)                 # [P,2]
    d = pix[None] - mu[:, None]                                              # [N,P,2]
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    u = (c * d[..., 0] + s * d[..., 1]) / torch.exp(log_s[:, 0:1])           # coords in the Gaussian's frame
    v = (-s * d[..., 0] + c * d[..., 1]) / torch.exp(log_s[:, 1:2])
    a = (torch.sigmoid(op)[:, None] * torch.exp(-0.5 * (u * u + v * v))).clamp(max=0.99)   # [N,P]
    T = torch.cumprod(torch.cat([torch.ones_like(a[:1]), 1 - a[:-1]], 0), 0)  # transmittance before i
    w = T * a
    img = w.T @ torch.sigmoid(col) + (T[-1] * (1 - a[-1]))[:, None] * bg
    return img.reshape(res, res, 3)


def fit_splats(n=256, res=64, steps=1000, snaps=(0, 2, 5, 10, 15, 25, 40, 60, 100, 150, 250, 400, 600, 1000), seed=0):
    import torch
    import matplotlib.cbook as cbook
    from PIL import Image
    torch.manual_seed(seed)
    im = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB").crop((26, 20, 486, 480))
    target = torch.tensor(np.asarray(im.resize((res, res), Image.LANCZOS)), dtype=torch.float32) / 255
    hi = np.asarray(im.resize((256, 256), Image.LANCZOS))
    p = {"mu": torch.rand(n, 2), "log_s": torch.full((n, 2), np.log(0.035)), "theta": torch.rand(n) * np.pi,
         "color": torch.zeros(n, 3), "opacity": torch.zeros(n)}
    for t in p.values():
        t.requires_grad_(True)
    lrs = {"mu": 0.004, "log_s": 0.02, "theta": 0.03, "color": 0.05, "opacity": 0.05}
    opt = torch.optim.Adam([{"params": [p[k]], "lr": lrs[k]} for k in p])
    rec = {k: [] for k in ["mu", "log_s", "theta", "color", "opacity", "psnr", "step"]}
    renders = []
    for it in range(steps + 1):
        img = splat_render(p, res)
        loss = ((img - target) ** 2).mean()
        if it in snaps:
            with torch.no_grad():
                for k in ["mu", "log_s", "theta", "color", "opacity"]:
                    rec[k].append(p[k].detach().numpy().copy())
                rec["psnr"].append(float(-10 * np.log10(loss.item())))
                rec["step"].append(it)
                renders.append((splat_render(p, 192).clamp(0, 1).numpy() * 255).astype(np.uint8))
            print(f"step {it:5d}  mse {loss.item():.5f}  psnr {rec['psnr'][-1]:.2f}")
        if it == steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    np.savez_compressed(DATA / "w10_splats.npz", target_lo=(target.numpy() * 255).astype(np.uint8), target_hi=hi,
                        renders=np.stack(renders), **{k: np.array(v) for k, v in rec.items()})


# ----------------------------------------------------------------------------------------------- bouncing ball
BOX_W, BOX_H, DT, G, REST = 4.0, 2.4, 0.05, -9.8, 0.85


def sim_step(s, a):
    """True physics. s=(x,y,vx,vy), a = horizontal push in {-1,0,1}."""
    x, y, vx, vy = s
    vx = vx + 4.0 * a * DT
    vy = vy + G * DT
    x, y = x + vx * DT, y + vy * DT
    if x < 0: x, vx = -x, -vx * REST
    if x > BOX_W: x, vx = 2 * BOX_W - x, -vx * REST
    if y < 0: y, vy = -y, -vy * REST
    if y > BOX_H: y, vy = 2 * BOX_H - y, -vy * REST
    return np.array([x, y, vx, vy])


def build_world(seed=0):
    import torch
    rng = np.random.default_rng(seed)
    # training data: one-step transitions from random states and actions
    S = np.stack([rng.uniform(0, BOX_W, 40000), rng.uniform(0, BOX_H, 40000),
                  rng.uniform(-4, 4, 40000), rng.uniform(-7, 7, 40000)], 1)
    A = rng.integers(-1, 2, 40000).astype(float)
    S1 = np.stack([sim_step(s, a) for s, a in zip(S, A)])
    scale = np.array([BOX_W, BOX_H, 4, 7])
    X = torch.tensor(np.c_[S / scale, A], dtype=torch.float32)
    Y = torch.tensor((S1 - S) / scale, dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 4))
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for it in range(3000):
        idx = torch.randint(0, len(X), (512,))
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print("one-step mse", loss.item())

    def model_step(s, a):
        with torch.no_grad():
            d = net(torch.tensor(np.r_[s / scale, a], dtype=torch.float32)).numpy()
        return s + d * scale

    s0 = np.array([0.6, 1.9, 1.6, 0.0])
    acts = np.array(([1] * 6 + [0] * 8 + [-1] * 4 + [0] * 12) * 2)[:50].astype(float)
    real, imag = [s0], [s0]
    for a in acts:
        real.append(sim_step(real[-1], a))
        imag.append(model_step(imag[-1], a))
    real, imag = np.array(real), np.array(imag)
    err = np.linalg.norm(real[:, :2] - imag[:, :2], axis=1)
    # average error vs horizon over many random starts / action sequences
    H = 50
    errs = []
    for k in range(200):
        s = np.array([rng.uniform(0.3, 3.7), rng.uniform(0.5, 2.2), rng.uniform(-2, 2), rng.uniform(-2, 2)])
        aa = rng.integers(-1, 2, H).astype(float)
        r, m, e = s, s, [0.0]
        for a in aa:
            r, m = sim_step(r, a), model_step(m, a)
            e.append(np.linalg.norm(r[:2] - m[:2]))
        errs.append(e)
    errs = np.array(errs)
    print("err at h=1,10,25,50:", errs.mean(0)[[1, 10, 25, 50]], "this traj", err[[1, 10, 25, 50]])
    np.savez(DATA / "w10_world.npz", real=real, imag=imag, acts=acts, err=err, mean_err=errs.mean(0),
             one_step_mse=loss.item())


# ----------------------------------------------------------------------------------------------- MPC
def build_mpc(seed=2, K=60, H=12, rounds=150, noise=1.0, mult=0.6, save=True):
    """Point mass, a = acceleration; a wall at x=0 below y=WALL_TOP. Random-shooting MPC with true dynamics as the model."""
    rng = np.random.default_rng(seed)
    goal = np.array([4.5, -1.5])
    wall_top, dt, amax = 1.0, 0.3, 3.0

    def roll(s, acts):
        p, v, traj, hit = s[:2].copy(), s[2:].copy(), [s[:2].copy()], False
        for a in acts:
            v = 0.8 * v + a * dt
            p_new = p + v * dt
            if p[0] * p_new[0] <= 0 and min(p[1], p_new[1]) < wall_top:
                hit = True
            if abs(p_new[0]) > 6 or abs(p_new[1]) > 2.8:
                hit = True
            p = p_new
            traj.append(p.copy())
        return np.array(traj), hit, np.r_[p, v]

    def score(traj, hit):
        return -np.linalg.norm(traj - goal, axis=1)[-4:].mean() - (10.0 if hit else 0.0)

    s = np.array([-4.5, -1.5, 0.0, 0.0])
    out = {"plans": [], "scores": [], "best": [], "path": [s[:2].copy()]}
    for r in range(rounds):
        A = np.clip(np.cumsum(rng.normal(0, noise, (K, H, 2)), 1) * mult, -amax, amax)
        trajs, scores = [], []
        for k in range(K):
            tr, hit, _ = roll(s, A[k])
            trajs.append(tr); scores.append(score(tr, hit))
        b = int(np.argmax(scores))
        out["plans"].append(np.array(trajs)); out["scores"].append(np.array(scores)); out["best"].append(b)
        _, _, s = roll(s, A[b][:1])                     # execute only the first action, then replan
        out["path"].append(s[:2].copy())
        if np.linalg.norm(s[:2] - goal) < 0.6:
            break
    print("rounds", len(out["best"]), "final", s)
    if not save:
        return len(out["best"])
    np.savez(DATA / "w10_mpc.npz", plans=np.array(out["plans"]), scores=np.array(out["scores"]),
             best=np.array(out["best"]), path=np.array(out["path"]), goal=goal, wall_top=wall_top)


if __name__ == "__main__":
    what = sys.argv[1:] or ["splat", "world", "mpc"]
    if "splat" in what: fit_splats()
    if "world" in what: build_world()
    if "mpc" in what: build_mpc()
