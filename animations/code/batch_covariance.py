"""Code for lecture 5 (self-supervised learning), the slide "The covariance matrix of a batch": what the d x d matrix
of whitening, Barlow Twins and VICReg is (after Jia-Bin Huang's "How AI Learned to Teach Itself [JEPA]", YouTube 2026,
the Whitened MSE part).

Left: a batch of N images, one column per image, each column the image's embedding z (d numbers z_1 ... z_d; d = 4
drawn). A row is one number across the batch. The caption names N and d (his "you mention N,D but you never define
them?", 18 Nov 2026). Right: the covariance matrix C (d x d) compares every two rows: C_44 = how much z_4 varies,
C_24 = whether z_2 and z_4 go up and down together. Drawn as the target I (blue diagonal), as on the next slide.

Run:  python batch_covariance.py OUT.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "stix"
BLUE, LIGHT, TINT, GREY, INK = "#2F5597", "#EDEDED", "#DDE7F3", "#7F7F7F", "#000000"
ORANGE = "#ED7D31"
W, H = 11.6, 3.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
ax.axis("off")

D, ROW, TOP = 4, 0.56, 3.3                        # d = 4 numbers per image; row spacing; y of z_1
rows_y = [TOP - j * ROW for j in range(D)]
cols_x = [0.85 + k * 0.62 for k in range(5)] + [0.85 + 4 * 0.62 + 0.94]   # five columns, a gap (...), column N

for j, c in ((1, BLUE), (3, ORANGE)):                                    # the two rows the callouts talk about
    ax.add_patch(FancyBboxPatch((0.6, rows_y[j] - 0.23), cols_x[-1] + 0.4 - 0.6, 0.46,
                                boxstyle="round,pad=0,rounding_size=0.12", fc=c, ec="none", alpha=0.18, zorder=1))
for k, x in enumerate(cols_x):
    ax.add_patch(FancyBboxPatch((x - 0.24, rows_y[-1] - 0.3), 0.48, rows_y[0] - rows_y[-1] + 0.6,
                                boxstyle="round,pad=0,rounding_size=0.1", fc="none", ec="#A6A6A6", lw=1.6, zorder=2))
    for y in rows_y:
        ax.add_patch(Circle((x, y), 0.15, fc="white", ec="#595959", lw=1.5, zorder=3))
    ax.text(x, rows_y[-1] - 0.52, "$N$" if k == len(cols_x) - 1 else str(k + 1), ha="center", va="center",
            fontsize=15 if k == len(cols_x) - 1 else 14, color=GREY)
ax.text((cols_x[4] + cols_x[5]) / 2, (rows_y[0] + rows_y[-1]) / 2, "$\\cdots$", ha="center", va="center", fontsize=22,
        color=GREY)
for j, y in enumerate(rows_y):
    ax.text(0.35, y, f"$z_{j + 1}$", ha="center", va="center", fontsize=21, color=INK)
Y1, Y2 = rows_y[-1] - 0.95, rows_y[-1] - 1.3                            # the two caption lines
ax.text((cols_x[0] + cols_x[-1]) / 2, Y1, "a batch of $N$ images, one column each", ha="center", va="center",
        fontsize=16, color=INK)
ax.text((cols_x[0] + cols_x[-1]) / 2, Y2, "each column: $d$ numbers (here $d = 4$)", ha="center", va="center",
        fontsize=16, color=INK)

YM = (rows_y[0] + rows_y[-1]) / 2
ax.add_patch(FancyArrowPatch((cols_x[-1] + 0.5, YM), (6.05, YM), arrowstyle="-|>", mutation_scale=24, lw=2.2,
                             color=GREY))
ax.text((cols_x[-1] + 0.5 + 6.05) / 2, YM + 0.18, "compare\nevery two rows", ha="center", va="bottom", fontsize=15,
        color="#595959", linespacing=1.15)

S, G, X0 = 0.5, 0.06, 6.6                                                # the d x d grid
gy = [rows_y[j] - S / 2 for j in range(D)]
for i in range(D):
    for j in range(D):
        ax.add_patch(Rectangle((X0 + j * (S + G), gy[i]), S, S, fc=BLUE if i == j else LIGHT, ec="none", zorder=2))
    ax.text(X0 - 0.25, rows_y[i], f"$z_{i + 1}$", ha="center", va="center", fontsize=17, color=INK)
    ax.text(X0 + i * (S + G) + S / 2, rows_y[0] + 0.5, f"$z_{i + 1}$", ha="center", va="center", fontsize=17,
            color=INK)
for (i, c) in ((1, BLUE), (3, ORANGE)):                                  # cells (2, 4) and (4, 4)
    ax.add_patch(Rectangle((X0 + 3 * (S + G) - 0.04, gy[i] - 0.04), S + 0.08, S + 0.08, fc="none", ec=c, lw=3,
                           zorder=4))
xr = X0 + 4 * (S + G) + 0.1
ax.text(xr, rows_y[1], "$z_2$ with $z_4$:\ndo they move together?", ha="left", va="center", fontsize=15,
        color=INK, linespacing=1.2)
ax.text(xr, rows_y[3], "$z_4$ with itself:\nhow much does it vary?", ha="left", va="center", fontsize=15, color=INK,
        linespacing=1.2)
ax.text(X0 + 2 * (S + G) - G / 2, Y1, "$C$:  $d \\times d$", ha="center", va="center", fontsize=17, color=INK)

OUTP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("batch_covariance.png")
fig.savefig(OUTP, dpi=220, facecolor="white")
print("saved", OUTP)
