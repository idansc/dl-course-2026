"""Code for the clips MomentumOvershoot and NesterovLookAhead (renders/w01/<Scene>.mp4): why Nesterov damps the swing
that momentum keeps. The runs are the ones of MomentumNesterov (optim_real.momentum_nesterov(): the raw diabetes bowl,
raw-centred bmi and s5, from w = 0, lr 0.05, rho 0.9); this file only looks at them along the eigenvectors of H.

The bowl is an exact quadratic, L(w) = L* + 1/2 (w - w*)^T H (w - w*), and every optimizer here is linear in the
gradient H (w - w*). In the eigenbasis of H the two coordinates therefore decouple:

    e = (w - w*) . u_max    across the valley (curvature lambda_max = 19.53)
    f = (w - w*) . u_min    along it          (curvature lambda_min = 0.217)

and the 2-D run, projected on u_max, is exactly the 1-D run of the same optimizer on the parabola
L(e) = 1/2 lambda_max e^2: a cut across the valley in its steep direction (checked below to ~1e-14). Per step, with
lr = learning rate and lam = the curvature of the direction, both methods are 2nd-order linear recurrences:

    momentum (optim_real.sgd_momentum: v = rho v + dw,  w -= lr v)
        e[t+1] = (1 + rho - lr lam) e[t] - rho e[t-1]
    Nesterov (optim_real.nesterov: dw = gradient at w + rho v,  v = rho v - lr dw,  w += v)
        e[t+1] = (1 + rho)(1 - lr lam) e[t] - rho (1 - lr lam) e[t-1]

so e[t] = sum of c_i z_i^t over the two roots z_i of z^2 - b z + c = 0 (b, c the two coefficients above). When the
roots are complex, e[t] swings across the floor with period 2 pi / angle(z), and every swing shrinks by |z| = sqrt(c)
per step: momentum sqrt(rho) = 0.949 whatever the curvature (as long as it swings), Nesterov sqrt(rho (1 - lr lam))
= 0.145 across the valley. Nesterov takes the gradient at the look-ahead point e + rho v (v = the step just taken):
across the valley lr lam = 0.977, so the gradient step there takes back 97.7% of the look-ahead point's distance to
the floor, e[t+1] = (1 - lr lam)(e[t] + rho v) = 0.023 (e[t] + rho v), and the swing dies within two steps.

Run:  python nesterov_damping.py   -> prints the numbers shown in the clips, saves figures/nesterov_damping.png
Try:  steep_cut(learning_rate=0.03): Nesterov's swing now shrinks by 0.61 per step, momentum's still by sqrt(rho);
      steep_cut(rho=0.5): momentum 0.71 per step, Nesterov 0.11; steep_cut(learning_rate=0.07): Nesterov's steep
      root is -1.02, so e flips sign every step and grows: Nesterov diverges across the valley.
"""
from pathlib import Path

import numpy as np
import torch

import optim_real as opt

METHODS = ("SGD+Momentum", "Nesterov")


def eigen_directions(H):
    """Eigenvalues and unit eigenvectors of the 2x2 Hessian, with fixed signs: u_max points to larger w1 (bmi),
    u_min to larger w2 (s5)."""
    lam, U = np.linalg.eigh(H)                         # ascending
    u_min, u_max = U[:, 0] * np.sign(U[1, 0]), U[:, 1] * np.sign(U[0, 1])
    return dict(lam_min=float(lam[0]), lam_max=float(lam[1]), u_min=u_min, u_max=u_max)


def recurrence(method, learning_rate, rho, curvature):
    """(b, c) with e[t+1] = b e[t] - c e[t-1] on a parabola of the given curvature."""
    a = learning_rate * curvature
    if method == "SGD+Momentum":
        return 1 + rho - a, rho
    if method == "Nesterov":
        return (1 + rho) * (1 - a), rho * (1 - a)
    raise ValueError(method)


def roots(b, c):
    """Roots of z^2 - b z + c: the per-step factor |z| and, for complex roots, the swing period 2 pi / angle(z)."""
    z = np.roots([1.0, -b, c])
    z = z[np.argmax(np.abs(z))]                        # the slower root sets the decay
    period = 2 * np.pi / abs(np.angle(z)) if abs(z.imag) > 1e-12 else None
    return dict(b=float(b), c=float(c), z=complex(z), factor=float(abs(z)), period=period)


def fit_recurrence(e, floor):
    """Least-squares fit of e[t+1] = b e[t] - c e[t-1] on the steps where |e| is above `floor` (the rounding floor)."""
    n = int(np.nonzero(np.abs(e) > floor)[0][-1])     # last step above the floor
    A = np.stack([e[1:n], -e[:n - 1]], 1)
    (b, c), *_ = np.linalg.lstsq(A, e[2:n + 1], rcond=None)
    return dict(roots(b, c), steps_used=n)


def sign_flips(e, floor=0.0):
    """Steps t at which e changes sign from t-1 to t (crossings of the valley floor), counted while |e| > floor."""
    big = np.abs(e) > floor
    return [t for t in range(1, len(e)) if big[t] and big[t - 1] and e[t] * e[t - 1] < 0]


def run_1d(method, e0, learning_rate, rho, curvature, num_steps):
    """The same optimizer function from optim_real, run on the 1-D parabola L(e) = 1/2 curvature e^2."""
    optimizer = dict(zip(METHODS, (opt.sgd_momentum, opt.nesterov)))[method]
    path = optimizer(lambda e: curvature * e, torch.tensor([e0], dtype=opt.DTYPE), learning_rate, num_steps, rho)
    return path[:, 0].numpy()


def look_ahead(path, w_star, u, rho, learning_rate, X, y):
    """Nesterov's look-ahead, in optim_real.nesterov's convention (v = w[t] - w[t-1], the step just taken; v = 0 at
    t = 0): the point w + rho v, the gradient there (optim_real.gradient on the data), and both projected on u."""
    v = np.vstack([np.zeros_like(path[:1]), np.diff(path, axis=0)])
    points = path + rho * v
    grads = np.stack([opt.gradient(torch.tensor(p, dtype=opt.DTYPE), X, y).numpy() for p in points])
    return dict(v=v @ u, point=(points - w_star) @ u, grad=grads @ u, grad_step=-learning_rate * grads @ u)


def steep_cut(learning_rate=0.05, rho=0.9, num_steps=150):
    """The runs of MomentumNesterov projected on both eigenvectors, the per-step factors (roots of the recurrences
    and a fit to the projected runs), the sign flips across the floor, and Nesterov's look-ahead points."""
    r = opt.momentum_nesterov(learning_rate, rho, num_steps)
    H, w_star = r["H"], r["w_star"]
    d = eigen_directions(H)
    X, y = opt.load_data()
    out = dict(d, kappa=d["lam_max"] / d["lam_min"], lr=learning_rate, rho=rho, num_steps=num_steps, H=H, w_star=w_star,
               runs={})
    for run in r["runs"]:
        if run["name"] not in METHODS:
            continue
        P = run["path"]
        e, f = (P - w_star) @ d["u_max"], (P - w_star) @ d["u_min"]
        floor = 1e3 * np.finfo(float).eps * np.abs(w_star).max()       # ~1e-11: well above the rounding floor
        res = dict(e=e, f=f, steps=run["steps"], gap=run["gap"],
                   flips=sign_flips(e), flips_flat=sign_flips(f),
                   steep=roots(*recurrence(run["name"], learning_rate, rho, d["lam_max"])),
                   flat=roots(*recurrence(run["name"], learning_rate, rho, d["lam_min"])),
                   fit=fit_recurrence(e, floor), floor=floor,
                   check_1d=float(np.abs(run_1d(run["name"], e[0], learning_rate, rho, d["lam_max"], num_steps)
                                         - e).max()))
        if run["name"] == "Nesterov":
            res["look_ahead"] = look_ahead(P, w_star, d["u_max"], rho, learning_rate, X, y)
        else:                       # momentum's step: the gradient step at w plus rho times the step just taken
            v = np.concatenate([[0.0], np.diff(e)])
            res["push"], res["grad_step"] = rho * v, -learning_rate * d["lam_max"] * e
        out["runs"][run["name"]] = res
    out["e0"], out["f0"] = float(out["runs"]["Nesterov"]["e"][0]), float(out["runs"]["Nesterov"]["f"][0])
    return out


# =====================================================================================================
def figure(r, path, shown=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"SGD+Momentum": "#2563EB", "Nesterov": "#16A34A"}
    e0, lam = r["e0"], r["lam_max"]
    M, N = r["runs"]["SGD+Momentum"], r["runs"]["Nesterov"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.1]))

    ax = axes[0]                    # the cut: L(e) = 1/2 lam e^2, the first steps and Nesterov's look-ahead points
    es = np.linspace(-1.1, 1.1, 300) * abs(e0)
    ax.plot(es, 0.5 * lam * es ** 2, color="#9CA3AF", lw=2)
    for name, k in (("SGD+Momentum", 7), ("Nesterov", 4)):
        e = r["runs"][name]["e"][:k]
        ax.plot(e, 0.5 * lam * e ** 2, "-o", color=col[name], ms=6, lw=1, alpha=0.8, label=f"{name}, steps 0–{k - 1}")
    la = N["look_ahead"]["point"][1:3]
    ax.plot(la, 0.5 * lam * la ** 2, "o", mfc="none", mec=col["Nesterov"], ms=11, mew=1.8, label="look-ahead e + ρv")
    ax.set_xlabel("e = (w − w*)·u_max"); ax.set_ylabel("L − L*")
    ax.set_title(f"cut across the valley: L = ½·{lam:.2f}·e²")
    ax.legend(fontsize=8, loc="upper center")

    ax = axes[1]                    # e_t vs step with the envelopes |e0| |z|^t
    t = np.arange(shown + 1)
    for name, run in (("SGD+Momentum", M), ("Nesterov", N)):
        ax.plot(t, run["e"][:shown + 1] / abs(e0), "-o", color=col[name], ms=3.5, lw=1.5,
                label=f"{name}: ×{run['steep']['factor']:.3f} per step")
        env = run["steep"]["factor"] ** t
        ax.plot(t, env, "--", color=col[name], lw=0.8, alpha=0.6)
        ax.plot(t, -env, "--", color=col[name], lw=0.8, alpha=0.6)
    ax.axhline(0, color="#9CA3AF", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("e / |e₀|")
    ax.set_title("steep direction: the swing across the floor")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]                    # |e| and |f| over all steps, log scale
    t = np.arange(r["num_steps"] + 1)
    for name, run in (("SGD+Momentum", M), ("Nesterov", N)):
        ax.semilogy(t, np.maximum(np.abs(run["e"]), 1e-17), color=col[name], lw=1.3, label=f"{name}, steep |e|")
        ax.semilogy(t, np.abs(run["f"]), color=col[name], lw=1.3, ls=":", label=f"{name}, flat |f|")
    ax.set_ylim(1e-16, 1e3)
    ax.set_xlabel("step"); ax.set_title(f"per step: steep ×{M['steep']['factor']:.3f} vs ×{N['steep']['factor']:.3f}, "
                                        f"flat ×{M['flat']['factor']:.3f} vs ×{N['flat']['factor']:.3f}", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"raw diabetes bowl (κ = {r['kappa']:.1f}), lr = {r['lr']}, ρ = {r['rho']}, from w = 0: "
                 "the runs of MomentumNesterov along the eigenvectors of H", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def swing(d):
    return f"swing period {d['period']:.2f} steps" if d["period"] else "real roots: no swing"


def main():
    r = steep_cut()
    M, N = r["runs"]["SGD+Momentum"], r["runs"]["Nesterov"]
    print(f"raw diabetes bowl: H = {np.round(r['H'], 3).tolist()}, w* = {np.round(r['w_star'], 2)}")
    print(f"  curvatures lambda_max = {r['lam_max']:.2f} along u_max = {np.round(r['u_max'], 4)}, "
          f"lambda_min = {r['lam_min']:.3f} along u_min = {np.round(r['u_min'], 4)}, kappa = {r['kappa']:.1f}")
    print(f"  start w = 0: e0 = (0 - w*).u_max = {r['e0']:.3f} (across), f0 = {r['f0']:.2f} (along)")
    print(f"  lr = {r['lr']}, rho = {r['rho']}: lr*lambda_max = {r['lr'] * r['lam_max']:.4f}, "
          f"lr*lambda_min = {r['lr'] * r['lam_min']:.4f}")
    for name, run in r["runs"].items():
        s, fl, ft = run["steep"], run["flat"], run["fit"]
        print(f"\n[{name}] 1% of the loss gap after {run['steps']} steps")
        print(f"  projection on u_max = the 1-D run on L(e) = 1/2*{r['lam_max']:.2f}*e^2: max difference "
              f"{run['check_1d']:.1e}")
        print(f"  steep: e[t+1] = {s['b']:.4f} e[t] - {s['c']:.4f} e[t-1]; slower root {s['z']:.4f}: "
              f"|z| = {s['factor']:.4f} per step, {swing(s)}")
        print(f"         fit to the projected run (steps 0-{ft['steps_used']}): b = {ft['b']:.4f}, c = {ft['c']:.4f}, "
              f"|z| = {ft['factor']:.4f}, {swing(ft)}")
        print(f"  flat:  f[t+1] = {fl['b']:.4f} f[t] - {fl['c']:.4f} f[t-1]: |z| = {fl['factor']:.4f} per step, "
              f"{swing(fl)}")
        flips = run["flips"]
        print(f"  crossings of the valley floor (sign flips of e) in {r['num_steps']} steps: {len(flips)}"
              + (f", the last at step {flips[-1]}" if flips else "") + "; |e| at steps 0/2/5/10/20/24: "
              + ", ".join(f"{abs(run['e'][k]):.3g}" for k in (0, 2, 5, 10, 20, 24)))
        print(f"  first 25 steps: {sum(t <= 24 for t in flips)} crossings; along the valley (f): "
              f"{len(run['flips_flat'])} in {r['num_steps']} steps")
    if N["flips"]:
        last = N["flips"][-1]
        print(f"\nNesterov crosses the floor at steps {N['flips']}, landing at |e|/|e0| = "
              + ", ".join(f"{abs(N['e'][t] / r['e0']):.0e}" for t in N["flips"])
              + f"; from step {last + 1} on, |e| <= {np.abs(N['e'][last + 1:]).max():.1e}, the rounding level of w "
              f"(eps·|w*| = {np.finfo(float).eps * np.abs(r['w_star']).max():.1e})")
    la = N["look_ahead"]
    print("\nNesterov's look-ahead (e in units of |e0|; v = the step just taken; look-ahead = e + rho v; "
          "gradient step = -lr * gradient there, from optim_real.gradient on the data):")
    print("   t        e        v   look-ahead   gradient   grad step   e[t+1]   | momentum: e[t+1]")
    u = abs(r["e0"])
    for t in range(5):
        print(f"  {t:2d} {N['e'][t] / u:+8.4f} {la['v'][t] / u:+8.4f} {la['point'][t] / u:+10.4f} {la['grad'][t]:+10.2f} "
              f"{la['grad_step'][t] / u:+10.4f} {N['e'][t + 1] / u:+9.4f}   | {M['e'][t + 1] / u:+8.4f}")
    err = np.abs(la["point"][:-1] + la["grad_step"][:-1] - N["e"][1:]).max()
    print(f"  look-ahead + gradient step = e[t+1] (max error {err:.1e}); the gradient there = lambda_max * look-ahead "
          f"(max error {np.abs(la['grad'] - r['lam_max'] * la['point']).max():.1e})")
    out = Path(__file__).with_name("figures") / "nesterov_damping.png"
    out.parent.mkdir(exist_ok=True)
    figure(r, out)
    print("saved", out)


if __name__ == "__main__":
    main()
