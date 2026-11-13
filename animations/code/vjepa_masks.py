"""Code for lecture 5 (self-supervised learning), the V-JEPA masking slide: V-JEPA's multi-block masks, sampled with
the rule and settings of the official code (facebookresearch/jepa: src/masks/multiblock3d.py, configs/pretrain/vitl16.yaml).

A 16-frame 224 x 224 clip is cut into tubelets of 2 frames x 16 x 16 pixels: a grid of 8 (time) x 14 x 14 tokens.
- short-range mask: 8 blocks, each covering 15% of the frame (int(196 * 0.15) = 29 tokens), aspect ratio in (0.75, 1.5);
- long-range mask: 2 blocks, each covering 70% of the frame (137 tokens);
every block spans all 8 time steps (temporal scale 1.0), so a hidden region is hidden in every frame. The hidden part y is
the union of the blocks; the visible part x is the rest. As in the code, one block size is drawn per batch and the block
positions per clip.

Prints the hidden fraction of each mask type and draws one example of each. Per clip, the union hides about 63%
(short-range) and 82% (long-range) of the tokens. The official collator then cuts every clip's visible set to the
smallest one in the batch (24 clips per GPU), so the encoder sees fewer tokens: about 77% and 95% hidden, 86% on average.
This is the paper's "average masking ratio of ~90%".

Run:  python vjepa_masks.py OUT.png      (also writes data/vjepa_masks.json)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T, H, W = 8, 14, 14
TYPES = {"short-range": dict(n=8, scale=0.15), "long-range": dict(n=2, scale=0.70)}
AR = (0.75, 1.5)
rng = np.random.default_rng(0)


def block_size(scale):
    keep = int(H * W * scale)
    ar = AR[0] + rng.random() * (AR[1] - AR[0])
    return min(H, int(round(math.sqrt(keep * ar)))), min(W, int(round(math.sqrt(keep / ar))))


def sample(n, size):
    hidden = np.zeros((H, W), bool)
    h, w = size
    for _ in range(n):
        top, left = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        hidden[top:top + h, left:left + w] = True
    return hidden                                              # the same spatial mask at all T time steps


BATCH = 24                                                     # clips per GPU in configs/pretrain/vitl16.yaml
stats = {}
for name, c in TYPES.items():
    fr, eff = [], []
    for _ in range(2000):                                       # 2,000 batches of 24 clips
        size = block_size(c["scale"])
        hid = [sample(c["n"], size).mean() for _ in range(BATCH)]
        fr += hid
        eff.append(max(hid))                                    # context cut to the batch's smallest visible set
    stats[name] = {"blocks": c["n"], "block_fraction": c["scale"], "mean_hidden": float(np.mean(fr)),
                   "effective_hidden": float(np.mean(eff))}
    print(f"{name:11s}: {c['n']} blocks of {c['scale']:.0%} -> {np.mean(fr):.1%} of tokens hidden per clip, "
          f"{np.mean(eff):.1%} after the batch-minimum cut")
both = (stats["short-range"]["effective_hidden"] + stats["long-range"]["effective_hidden"]) / 2
print(f"mean of the two (effective): {both:.1%}  (the paper: 'an average masking ratio of ~90%')")

OUTP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vjepa_masks.png")
data = Path(__file__).parent / "data"
data.mkdir(exist_ok=True)
(data / "vjepa_masks.json").write_text(json.dumps({"grid": [T, H, W], "types": stats, "mean_both": both}, indent=1))

# ---------------------------------------------------------------- one example of each type, as stacked frames
plt.rcParams["font.family"] = "Arial"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
BLUE, WHITE, EDGE = "#2F5597", "#FFFFFF", "#C9D3E6"
ex = {"short-range": None, "long-range": None}
for name in ex:
    for _ in range(200):                                        # an example whose hidden share is near the mean
        m = sample(TYPES[name]["n"], block_size(TYPES[name]["scale"]))
        if abs(m.mean() - stats[name]["mean_hidden"]) < 0.02:
            break
    ex[name] = m
for ax, (name, m) in zip(axes, ex.items()):
    for k in range(3, -1, -1):                                  # four of the 8 time steps, back to front
        dx, dy = 1.1 * k, 1.1 * k
        for i in range(H):
            for j in range(W):
                ax.add_patch(plt.Rectangle((j + dx, H - 1 - i + dy), 1, 1, facecolor=WHITE if m[i, j] else BLUE,
                                           edgecolor=EDGE if m[i, j] else WHITE, lw=0.6))
        ax.add_patch(plt.Rectangle((dx, dy), W, H, fill=False, edgecolor="#404040", lw=1.2))
    ax.set_xlim(-0.5, W + 4.0)
    ax.set_ylim(-0.5, H + 4.0)
    ax.set_aspect("equal")
    ax.axis("off")
    c = TYPES[name]
    ax.set_title(f"{name}: union of {c['n']} blocks,\n{c['scale']:.0%} of the frame each", fontsize=17)
fig.tight_layout()
fig.savefig(OUTP, dpi=220, facecolor="white")
print("saved", OUTP)
