"""Code for the clip AdamBiasCorrection (renders/w01/AdamBiasCorrection.mp4): why Adam divides its moments by
(1 - beta**t). With a constant gradient dw = 1 both moments should be 1, but they start at 0, so without the
correction the first steps are several times larger than learning_rate. Same update as the "Adam (full form)" slide.

Run:  python adam_bias_correction.py   -> prints the step sizes shown in the clip, saves figures/adam_bias_correction.png
Try:  beta2 = 0.99 or 0.9999 (how long does the uncorrected step stay too large?), or a noisy gradient dw.
"""
from pathlib import Path
import numpy as np

beta1, beta2 = 0.9, 0.999
learning_rate = 1.0             # = 1, so every step below is measured in units of the learning rate


def adam_constant_gradient(num_steps=3000, dw=1.0):
    """Adam's two moments and its step size, with and without bias correction, for t = 1..num_steps."""
    moment1, moment2 = 0, 0
    rows = []
    for t in range(1, num_steps + 1):
        moment1 = beta1 * moment1 + (1 - beta1) * dw            # momentum
        moment2 = beta2 * moment2 + (1 - beta2) * dw * dw       # AdaGrad / RMSProp
        moment1_unbias = moment1 / (1 - beta1 ** t)             # bias correction
        moment2_unbias = moment2 / (1 - beta2 ** t)             # bias correction
        step_uncorrected = learning_rate * moment1 / (np.sqrt(moment2) + 1e-7)
        step = learning_rate * moment1_unbias / (np.sqrt(moment2_unbias) + 1e-7)
        rows.append((t, moment1, moment2, moment1_unbias, moment2_unbias, step_uncorrected, step))
    names = ["t", "moment1", "moment2", "moment1_unbias", "moment2_unbias", "step_uncorrected", "step"]
    return {k: np.array(v) for k, v in zip(names, zip(*rows))}


def figure(a, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = a["t"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(t, a["moment1"], color="#2563EB", lw=2)
    ax1.plot(t, a["moment2"], color="#7C3AED", lw=2)
    ax1.plot(t, a["moment1_unbias"], color="#1F2937", lw=1.5, ls="--")
    ax1.text(60, 0.9, "moment1", color="#2563EB")
    ax1.text(15, 0.12, "moment2", color="#7C3AED")
    ax1.text(1.1, 1.04, "divided by 1 - beta**t: both are exactly 1", color="#1F2937")
    ax1.set_title("moments, constant gradient dw = 1")
    ax1.set_ylim(0, 1.15)
    ax2.plot(t, a["step_uncorrected"], color="#DC2626", lw=2, label="without correction")
    ax2.plot(t, a["step"], color="#16A34A", lw=2, label="with correction")
    for tt in (1, int(t[np.argmax(a["step_uncorrected"])]), 100, 1000):
        s = a["step_uncorrected"][tt - 1]
        ax2.plot(tt, s, "o", color="#DC2626")
        ax2.annotate(f"{s:.2f}" + (f" at t = {tt}" if tt not in (1, 100, 1000) else ""), (tt, s),
                     textcoords="offset points", xytext=(8, -14) if tt == 1 else (6, 6), color="#DC2626")
    ax2.set_title("step size / learning rate")
    ax2.set_ylim(0, 7.5)
    ax2.legend(frameon=False, loc="upper right")
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("step t")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    a = adam_constant_gradient()
    at = lambda k, t: a[k][t - 1]
    tpk = int(a["t"][np.argmax(a["step_uncorrected"])])
    print(f"t = 1: moment1 = {at('moment1', 1):.3f}, moment2 = {at('moment2', 1):.3f} -> step without correction "
          f"0.1 / sqrt(0.001) = {at('step_uncorrected', 1):.2f} x lr")
    print(f"largest step without correction: {at('step_uncorrected', tpk):.2f} x lr at t = {tpk}")
    for t in (100, 1000):
        print(f"t = {t}: step without correction {at('step_uncorrected', t):.2f} x lr")
    print(f"moment2 at t = 1000: {at('moment2', 1000):.2f}")
    print(f"with correction: {a['step'].min():.4f} to {a['step'].max():.4f} x lr at every t")
    out = Path(__file__).with_name("figures") / "adam_bias_correction.png"
    out.parent.mkdir(exist_ok=True)
    figure(a, out)
    print("saved", out)
