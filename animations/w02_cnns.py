"""Week 2 (convolutional networks). Render one scene:
    manim -qh w02_cnns.py ConvSliding
Real numbers: ConvSliding convolves matplotlib's grace_hopper.jpg with numpy; ResidualHighway and GradCAMReveal read
data/w02_resgrads.json and data/w02_gradcam.npz, written by data/w02_make_data.py.
"""
import json
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


def gray_image(n=40):
    """Grace Hopper (matplotlib sample), face crop, n x n grayscale in [0, 1]."""
    import matplotlib.cbook as cbook
    from PIL import Image
    img = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("L")
    img = img.crop((96, 40, 416, 360)).resize((n, n), Image.LANCZOS)
    return np.asarray(img, dtype=float) / 255.0


def correlate_valid(img, k):
    """What a conv layer computes (cross-correlation, no padding, stride 1)."""
    H, W = img.shape
    out = np.zeros((H - 2, W - 2))
    for i in range(H - 2):
        for j in range(W - 2):
            out[i, j] = (img[i:i + 3, j:j + 3] * k).sum()
    return out


def signed_rgb(v, vmax):
    """negative -> blue, 0 -> white, positive -> orange."""
    t = np.clip(v / vmax, -1, 1)[..., None]
    white = np.array([255, 255, 255.0])
    pos = np.array([234, 88, 12.0])
    neg = np.array([37, 99, 235.0])
    return np.where(t >= 0, white + t * (pos - white), white + (-t) * (neg - white))


def pixel_image(rgb, height):
    m = ImageMobject(np.clip(rgb, 0, 255).astype(np.uint8))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.height = height
    return m


# =====================================================================================================
class ConvSliding(Scene):
    """A real 3×3 Sobel kernel slides over a real photo; at each position: patch ⊙ kernel, summed; the feature map
    fills in. Then a horizontal-edge kernel on the same image. Target: p3 slides 53–60."""

    def construct(self):
        title(self, "Convolution: one kernel, slid everywhere", "Grace Hopper (matplotlib sample image), 40×40 grayscale")
        n = 40
        img = gray_image(n)
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
        ky = kx.T.copy()
        outs = {"x": correlate_valid(img, kx), "y": correlate_valid(img, ky)}
        vmax = max(np.percentile(np.abs(o), 99) for o in outs.values())
        m = n - 2
        H = 4.0
        px = H / n
        in_img = pixel_image(np.repeat(img[..., None] * 255, 3, axis=2), H).move_to([-4.6, -0.35, 0])
        in_lbl = Text("input  40×40", font_size=22).next_to(in_img, DOWN, buff=0.12)
        out_c = np.array([4.6, -0.35, 0])
        out_frame = Rectangle(width=m * px, height=m * px, stroke_color=MUTED, stroke_width=1).move_to(out_c)
        out_lbl = Text("feature map  38×38", font_size=22).next_to(out_frame, DOWN, buff=0.12)

        def in_cell(i, j):          # centre of input pixel (row i, col j)
            return in_img.get_corner(UL) + np.array([(j + 0.5) * px, -(i + 0.5) * px, 0])

        def out_cell(i, j):
            return out_frame.get_corner(UL) + np.array([(j + 0.5) * px, -(i + 0.5) * px, 0])

        state = {"kernel": "x", "revealed": np.zeros((m, m), bool)}
        k = ValueTracker(-1)        # sweep index (row-major); -1 = nothing swept yet
        pos = ValueTracker(0)       # current window position (flattened index)

        def fmap():
            o = outs[state["kernel"]]
            idx = np.arange(m * m).reshape(m, m)
            shown = state["revealed"] | (idx <= k.get_value())
            rgb = signed_rgb(o, vmax)
            rgb[~shown] = [243, 244, 246]
            return pixel_image(rgb, m * px).move_to(out_c)

        fm = always_redraw(fmap)

        def ij():
            p = int(round(pos.get_value()))
            return divmod(min(max(p, 0), m * m - 1), m)

        def window():
            i, j = ij()
            return Square(3 * px, color=RED_, stroke_width=4).move_to(in_cell(i + 1, j + 1))

        def out_marker():
            i, j = ij()
            return Square(px * 1.6, color=RED_, stroke_width=3).move_to(out_cell(i, j))

        # middle panel: patch ⊙ kernel = value
        cell = 0.52
        mid_y = 0.5

        def grid3(vals, colors, x, fmt, txt_color=None):
            g = VGroup()
            for r in range(3):
                for c in range(3):
                    sq = Square(cell, stroke_color=MUTED, stroke_width=1.5, fill_color=colors[r][c], fill_opacity=1)
                    sq.move_to([x + (c - 1) * cell, mid_y - (r - 1) * cell, 0])
                    tc = txt_color[r][c] if txt_color else INK
                    g.add(VGroup(sq, Text(fmt.format(vals[r, c]), font_size=17, color=tc).move_to(sq)))
            return g

        def patch_grid():
            i, j = ij()
            P = img[i:i + 3, j:j + 3]
            cols = [[rgb_to_color(np.array([P[r, c]] * 3)) for c in range(3)] for r in range(3)]
            tcol = [[WHITE if P[r, c] < 0.5 else INK for c in range(3)] for r in range(3)]
            return grid3(P, cols, -1.05, "{:.2f}", tcol)

        def kernel_grid():
            K = kx if state["kernel"] == "x" else ky
            cols = [[heat_color(abs(K[r, c]) / 2, ORANGE_ if K[r, c] > 0 else BLUE_) for c in range(3)] for r in range(3)]
            return grid3(K, cols, 1.05, "{:+.0f}")

        def result():
            i, j = ij()
            v = outs[state["kernel"]][i, j]
            col = ORANGE_ if v > 0.05 else (BLUE_ if v < -0.05 else INK)
            return Text(f"sum = {v:+.2f}".replace("-", "−"), font_size=32, color=col).move_to([0, mid_y - 1.45, 0])

        odot = MathTex(r"\odot", font_size=44).move_to([0, mid_y, 0])
        p_lbl = Text("patch", font_size=20, color=MUTED).move_to([-1.05, mid_y + 1.05, 0])
        k_lbl = always_redraw(lambda: Text("vertical-edge kernel" if state["kernel"] == "x" else "horizontal-edge kernel",
                                           font_size=20, color=MUTED).move_to([1.05, mid_y + 1.05, 0]))
        pg, kg, res = always_redraw(patch_grid), always_redraw(kernel_grid), always_redraw(result)
        win, om = always_redraw(window), always_redraw(out_marker)

        cap = caption("a photo is a grid of numbers: 40×40 pixels, brightness in [0, 1]")
        self.play(FadeIn(in_img), FadeIn(in_lbl), FadeIn(cap))
        self.wait(1.2)
        cap = swap_caption(self, cap, "a 3×3 kernel: here the Sobel vertical-edge detector")
        self.play(FadeIn(kg), FadeIn(k_lbl))
        self.wait(1.2)

        # three hand-picked positions crossing a strong edge
        o = outs["x"]
        r0 = 17
        c_star = int(np.argmax(np.abs(o[r0, 4:-4]))) + 4
        demo = [r0 * m + c_star - 3, r0 * m + c_star - 1, r0 * m + c_star]
        pos.set_value(demo[0])
        cap = swap_caption(self, cap, "at one position: multiply the 3×3 patch by the kernel, entry by entry, and sum")
        self.play(Create(win), FadeIn(pg), FadeIn(odot), FadeIn(p_lbl), FadeIn(out_frame), FadeIn(out_lbl), FadeIn(fm))
        self.play(FadeIn(res), FadeIn(om))
        for t, d in enumerate(demo):
            if t:
                self.play(pos.animate.set_value(d), run_time=0.8, rate_func=lambda a: np.floor(a * 3) / 3 if a < 1 else 1)
            i, j = divmod(d, m)
            state["revealed"][i, j] = True
            self.wait(1.3)
        cap = swap_caption(self, cap, "the sum is one pixel of the feature map: large where brightness changes from left to right")
        self.wait(1.4)

        # sweep
        cap = swap_caption(self, cap, "slide it over every position: the feature map fills in")
        pos.add_updater(lambda mob: mob.set_value(max(k.get_value(), 0)))
        self.add(pos)
        self.play(k.animate.set_value(m * m - 1), run_time=7, rate_func=linear)
        pos.clear_updaters()
        legend = VGroup(Square(0.22, stroke_width=0, fill_color=ORANGE_, fill_opacity=1),
                        Text("dark → bright", font_size=18),
                        Square(0.22, stroke_width=0, fill_color=BLUE_, fill_opacity=1),
                        Text("bright → dark", font_size=18)).arrange(RIGHT, buff=0.1)
        legend[2].shift(0.25 * RIGHT); legend[3].shift(0.25 * RIGHT)
        legend.next_to(out_frame, UP, buff=0.15)
        cap = swap_caption(self, cap, "the outlines of the face and the cap light up; flat regions stay near 0")
        self.play(FadeIn(legend))
        self.wait(2)

        # horizontal kernel
        cap = swap_caption(self, cap, "a different kernel (transposed: horizontal edges) gives a different feature map")
        state["kernel"] = "y"
        state["revealed"][:] = False
        k.set_value(-1)
        pos.add_updater(lambda mob: mob.set_value(max(k.get_value(), 0)))
        self.add(pos)
        self.play(k.animate.set_value(m * m - 1), run_time=4, rate_func=linear)
        pos.clear_updaters()
        self.wait(1)
        cap = swap_caption(self, cap, "a conv layer learns many kernels, each one reused at every position",
                           color=PURPLE_)
        self.wait(2.5)


# =====================================================================================================
class ReceptiveField(Scene):
    """Three stacked 3×3 convolutions: one unit of conv 3 sees 3×3 of conv 2, 5×5 of conv 1 and 7×7 of the input.
    Target: p3 slides 72–74."""

    def construct(self):
        title(self, "Receptive field of stacked 3×3 convolutions", "stride 1, 'same' padding; every map is 9×9")
        n, cell = 9, 0.3
        xs = [-4.95, -1.65, 1.65, 4.95]
        y0 = -0.25
        names = ["input", "conv 1", "conv 2", "conv 3"]
        cols = [BLUE_, PURPLE_, ORANGE_, GREEN_]
        grids = VGroup()
        for x in xs:
            g = VGroup(*[Square(cell, stroke_color=MUTED, stroke_width=1, fill_color=WHITE, fill_opacity=1)
                         .move_to([x + (c - 4) * cell, y0 - (r - 4) * cell, 0]) for r in range(n) for c in range(n)])
            grids.add(g)
        lbls = VGroup(*[Text(nm, font_size=24, color=cl).next_to(g, DOWN, buff=0.18) for nm, g, cl in zip(names, grids, cols)])
        arrows = VGroup(*[Arrow(grids[i].get_right() + 0.05 * RIGHT, grids[i + 1].get_left() + 0.05 * LEFT, buff=0.05,
                                color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.25) for i in range(3)])
        alab = VGroup(*[Text("3×3", font_size=18, color=MUTED).next_to(a, UP, buff=0.05) for a in arrows])

        r, c = ValueTracker(4), ValueTracker(4)
        level = ValueTracker(0)        # how many layers back the field is shown (0..3)

        def cellpos(L, i, j):
            return np.array([xs[L] + (j - 4) * cell, y0 - (i - 4) * cell, 0])

        def region(L, rad):
            i, j = int(round(r.get_value())), int(round(c.get_value()))
            i0, i1 = max(i - rad, 0), min(i + rad, n - 1)
            j0, j1 = max(j - rad, 0), min(j + rad, n - 1)
            ul = cellpos(L, i0, j0) + np.array([-cell / 2, cell / 2, 0])
            lr = cellpos(L, i1, j1) + np.array([cell / 2, -cell / 2, 0])
            return ul, lr

        def fields():
            g = VGroup()
            lv = int(round(level.get_value()))
            for back in range(lv + 1):
                L = 3 - back
                ul, lr = region(L, back)
                rect = Rectangle(width=lr[0] - ul[0], height=ul[1] - lr[1], stroke_color=cols[L], stroke_width=4,
                                 fill_color=cols[L], fill_opacity=0.3).move_to((ul + lr) / 2)
                g.add(rect)
                if back < lv:
                    ul2, lr2 = region(L - 1, back + 1)
                    g.add(Line([lr2[0], ul2[1], 0], ul, color=cols[L - 1], stroke_width=2, stroke_opacity=0.7),
                          Line(lr2, [ul[0], lr[1], 0], color=cols[L - 1], stroke_width=2, stroke_opacity=0.7))
            return g

        def sizes():
            g = VGroup()
            lv = int(round(level.get_value()))
            for back in range(1, lv + 1):
                L = 3 - back
                s = 2 * back + 1
                g.add(Text(f"{s}×{s}", font_size=26, color=cols[L], weight=BOLD).next_to(grids[L], UP, buff=0.42))
            return g

        cap = caption("four feature maps; each is a 3×3 convolution of the one before")
        self.play(FadeIn(grids), FadeIn(lbls), *[GrowArrow(a) for a in arrows], FadeIn(alab), FadeIn(cap))
        self.wait(1)
        fl, sz = always_redraw(fields), always_redraw(sizes)
        cap = swap_caption(self, cap, "pick one unit of conv 3", color=GREEN_)
        self.add(fl, sz)
        self.play(Indicate(grids[3][4 * n + 4], color=GREEN_, scale_factor=1.6))
        self.wait(0.6)
        steps = ["it is computed from a 3×3 window of conv 2",
                 "each of those units sees 3×3 of conv 1: together a 5×5 window",
                 "and, through them, a 7×7 window of the input"]
        for t, txt in enumerate(steps):
            cap = swap_caption(self, cap, txt, color=cols[2 - t])
            fl.suspend_updating(); sz.suspend_updating()
            level.set_value(t + 1)
            self.play(Transform(fl, fields()), Transform(sz, sizes()), run_time=0.9)
            fl.resume_updating(); sz.resume_updating()
            self.wait(1.1)
        formula = MathTex(r"\text{receptive field after } L \text{ layers} = 1 + 2L", font_size=36, color=INK
                          ).move_to([0, 2.05, 0])
        cap = swap_caption(self, cap, "each 3×3 layer adds one pixel on every side")
        self.play(FadeIn(formula))
        self.wait(1.5)

        cap = swap_caption(self, cap, "move the unit: its field moves with it, and is cut off by the padding at the border")
        self.play(c.animate.set_value(1), run_time=1.8, rate_func=lambda a: np.round(a * 3) / 3)
        self.play(r.animate.set_value(7), run_time=1.8, rate_func=lambda a: np.round(a * 3) / 3)
        self.play(c.animate.set_value(6), r.animate.set_value(3), run_time=1.8, rate_func=lambda a: np.round(a * 4) / 4)
        self.play(c.animate.set_value(4), r.animate.set_value(4), run_time=1.2)
        self.wait(0.5)
        cap = swap_caption(self, cap, "three 3×3 layers: a 7×7 view with 27 weights instead of 49, plus two extra ReLUs",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# =====================================================================================================
class ResidualHighway(Scene):
    """Plain 30-layer MLP vs the same net with skip connections: real per-layer gradient norms at init (log scale),
    the identity path as a gradient highway, and real training curves. Data: data/w02_resgrads.json.
    Target: p4 (ResNet)."""

    def construct(self):
        d = json.loads((DATA / "w02_resgrads.json").read_text())
        gp, gr = np.array(d["plain_grad"]), np.array(d["residual_grad"])
        D = len(gp)
        title(self, "Residual connections: a highway for the gradient",
              f"two {D}-layer MLPs, width 64, PyTorch default init, same data; gradient norm per layer at init")
        lo, hi = -14, 2                     # log10 range of the bar axis
        bw, gap = 0.17, 0.03
        x0 = -6.0
        bar_h = 1.55

        def lg(v):
            return (np.clip(np.log10(v), lo, hi) - lo) / (hi - lo) * bar_h

        rows = {}
        for name, base_y, g, col, form in [("plain", 0.15, gp, RED_, r"\text{plain: } h \leftarrow \mathrm{ReLU}(W_l h)"),
                                           ("residual", -2.35, gr, GREEN_, r"\text{residual: } h \leftarrow h + W_l\,\mathrm{ReLU}(h)")]:
            axis = Line([x0 - 0.1, base_y, 0], [x0 + D * (bw + gap), base_y, 0], color=MUTED, stroke_width=2)
            yax = Line([x0 - 0.1, base_y, 0], [x0 - 0.1, base_y + bar_h, 0], color=MUTED, stroke_width=2)
            ticks = VGroup()
            for e in (-12, -6, 0):
                y = base_y + (e - lo) / (hi - lo) * bar_h
                ticks.add(Line([x0 - 0.16, y, 0], [x0 - 0.04, y, 0], color=MUTED, stroke_width=2),
                          MathTex(f"10^{{{e}}}", font_size=20, color=MUTED).move_to([x0 - 0.5, y, 0]))
            bars = VGroup(*[Rectangle(width=bw, height=max(lg(v), 0.01), stroke_width=0, fill_color=col, fill_opacity=0.9)
                            .move_to([x0 + i * (bw + gap) + bw / 2, base_y + max(lg(v), 0.01) / 2, 0])
                            for i, v in enumerate(g)])
            lab = MathTex(form, font_size=30, color=col).move_to([x0 + D * (bw + gap) / 2 - 0.2, base_y + bar_h + 0.35, 0])
            nums = VGroup(Text("layer 1", font_size=16, color=MUTED).next_to(bars[0], DOWN, buff=0.08).set_y(base_y - 0.15),
                          Text(f"layer {D}", font_size=16, color=MUTED).set_y(base_y - 0.15).set_x(bars[-1].get_x()))
            rows[name] = dict(axis=axis, yax=yax, ticks=ticks, bars=bars, lab=lab, nums=nums, base=base_y, g=g, col=col)

        def show_row(name):
            r = rows[name]
            self.play(FadeIn(r["lab"]), Create(r["axis"]), Create(r["yax"]), FadeIn(r["ticks"]), FadeIn(r["nums"]), run_time=0.8)

        def sweep(name, highway=False):
            r = rows[name]
            t = ValueTracker(D + 0.5)
            for i, b in enumerate(r["bars"]):
                b.add_updater(lambda m, i=i: m.set_opacity(0.9 if i >= t.get_value() - 0.5 else 0))
            pulse = always_redraw(lambda: Dot([x0 + max(t.get_value() - 0.5, 0) * (bw + gap) + bw / 2,
                                               r["base"] + bar_h + 0.05, 0], radius=0.09, color=r["col"]))
            self.add(r["bars"], pulse)
            hw = None
            if highway:
                hw = Arrow([x0 + D * (bw + gap), r["base"] + bar_h + 0.05, 0], [x0 - 0.05, r["base"] + bar_h + 0.05, 0],
                           buff=0, color=GREEN_, stroke_width=6, max_tip_length_to_length_ratio=0.04)
                self.play(GrowArrow(hw), run_time=0.8)
            self.play(t.animate.set_value(0.5), run_time=4, rate_func=linear)
            for b in r["bars"]:
                b.clear_updaters()
            self.remove(pulse)
            return hw

        cap = caption(f"two {D}-layer nets, identical except for a skip connection around every layer")
        self.play(FadeIn(cap))
        show_row("plain")
        show_row("residual")
        self.wait(1)

        cap = swap_caption(self, cap, "backprop through the plain net: the gradient shrinks at every layer", color=RED_)
        sweep("plain")
        ann_p = Text(f"layer {D}: {gp[-1]:.0e}  →  layer 1: {gp[0]:.0e}", font_size=20, color=RED_
                     ).move_to([x0 + 2.3, rows["plain"]["base"] + 1.2, 0])
        self.play(FadeIn(ann_p))
        ratio = gp[-1] / gp[0]
        cap = swap_caption(self, cap, f"layer 1's gradient is ~{ratio / 1e9:.0f} billion times smaller than layer {D}'s: it cannot learn",
                           color=RED_)
        self.wait(1.5)

        cap = swap_caption(self, cap, "with skips each layer's Jacobian is I + ∂f/∂h: the identity carries the gradient back",
                           color=GREEN_, size=26)
        sweep("residual", highway=True)
        ann_r = Text(f"every layer: {gr.min():.0f} to {gr.max():.0f}", font_size=20, color=GREEN_
                     ).next_to(rows["residual"]["axis"], DOWN, buff=0.12)
        self.play(FadeIn(ann_r))
        self.wait(1.5)

        # training curves
        lp, lr_ = np.array(d["plain_loss"]), np.array(d["residual_loss"])
        S = 300
        lax = Axes(x_range=[0, S, 100], y_range=[0, 3, 1], x_length=4.2, y_length=2.6,
                   axis_config={"stroke_color": MUTED, "include_tip": False, "font_size": 20},
                   x_axis_config={"numbers_to_include": [100, 200, 300]},
                   y_axis_config={"numbers_to_include": [1, 2, 3]}).move_to([4.15, 0.25, 0])
        for num in list(lax.x_axis.numbers) + list(lax.y_axis.numbers):
            num.set_color(INK)
        lhdr = Text("training loss, 3-class spiral", font_size=22).next_to(lax, UP, buff=0.15)
        xl = Text("SGD step", font_size=18, color=MUTED).next_to(lax.x_axis, DOWN, buff=0.35)
        ln3 = DashedLine(lax.c2p(0, np.log(3)), lax.c2p(S, np.log(3)), color=MUTED, stroke_width=2)
        ln3l = MathTex(r"\ln 3", font_size=24, color=MUTED).next_to(lax.c2p(S, np.log(3)), RIGHT, buff=0.08)
        tt = ValueTracker(1)

        def curve(L, col):
            k = int(tt.get_value())
            pts = [lax.c2p(i, min(L[i], 3)) for i in range(max(k, 2))]
            return VMobject(stroke_color=col, stroke_width=4).set_points_as_corners(pts)

        cp = always_redraw(lambda: curve(lp, RED_))
        cr = always_redraw(lambda: curve(lr_, GREEN_))
        cap = swap_caption(self, cap, "train both (SGD + momentum, best of 3 learning rates)")
        self.play(Create(lax), FadeIn(lhdr), FadeIn(xl), Create(ln3), FadeIn(ln3l))
        self.add(cp, cr)
        self.play(tt.animate.set_value(S), run_time=4, rate_func=linear)
        leg = VGroup(Text(f"plain: stuck at chance ({d['plain_acc']:.0%} acc.)", font_size=19, color=RED_),
                     Text(f"residual: {d['residual_acc']:.0%} training acc.", font_size=19, color=GREEN_)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(xl, DOWN, buff=0.2)
        self.play(FadeIn(leg))
        cap = swap_caption(self, cap, "skip connections make depth trainable: the gradient always has a way back",
                           color=PURPLE_)
        self.wait(2.5)


# =====================================================================================================
def cam_overlay(cam, size, alpha=0.5):
    """Upsample a Grad-CAM map to size×size and colour it with the usual jet map as an RGBA overlay."""
    from PIL import Image
    import matplotlib
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR), float) / 255
    rgba = matplotlib.colormaps["jet"](c)
    rgba[..., 3] = alpha
    return (rgba * 255).astype(np.uint8)


class GradCAMReveal(Scene):
    """Grad-CAM on a real pretrained ResNet-18 (heat over the evidence for each class), then a tiny CNN trained on a
    dataset with a leaked label tag: Grad-CAM shows it looks at the tag, and swapping the tag flips its answer.
    Data: data/w02_gradcam.npz. Target: exam B (Grad-CAM question)."""

    def construct(self):
        z = np.load(DATA / "w02_gradcam.npz")
        title(self, "Grad-CAM: where is the evidence?", "heat = class-score gradient × last conv feature maps")
        H = 3.6
        photo = ImageMobject(z["img"]).set(height=H).move_to([-4.1, -0.45, 0])
        names, probs = [str(s) for s in z["names"]], z["probs"]
        pick = [names.index("military uniform"), names.index("bow tie")]       # top-1 and top-3 of ResNet-18
        overlays = [ImageMobject(cam_overlay(z["cams"][i], 224)).set(height=H).move_to(photo) for i in pick]
        formula = MathTex(r"\alpha_k = \frac{1}{HW}\sum_{i,j}\frac{\partial y^{c}}{\partial A^k_{ij}}", r"\qquad",
                          r"L^{c} = \mathrm{ReLU}\Big(\sum_k \alpha_k A^k\Big)", font_size=32).move_to([2.3, 1.95, 0])

        def plabel(i):
            return Text(f"ResNet-18 → “{names[i]}”  p = {probs[i]:.2f}", font_size=22, color=RED_
                        ).next_to(photo, DOWN, buff=0.15)

        cap = caption("a pretrained ResNet-18 (ImageNet) looks at a photo")
        self.play(FadeIn(photo), FadeIn(cap))
        self.wait(0.8)
        cap = swap_caption(self, cap, "Grad-CAM: weight each feature map by how much it raises the class score")
        self.play(FadeIn(formula))
        self.wait(1.5)
        lab = plabel(pick[0])
        cap = swap_caption(self, cap, f"top class “{names[pick[0]]}”: heat on the cap, the face and the shoulder", color=RED_)
        self.play(FadeIn(overlays[0]), FadeIn(lab), run_time=1.2)
        self.wait(2)
        lab2 = plabel(pick[1])
        cap = swap_caption(self, cap, f"ask about “{names[pick[1]]}” instead: the heat moves down to the collar", color=RED_)
        self.play(FadeOut(overlays[0]), FadeIn(overlays[1]), Transform(lab, lab2), run_time=1.2)
        self.wait(2)

        # shortcut model
        s = 2.5
        def gray(a):
            a = np.clip((a - a.min()) / (a.max() - a.min()), 0, 1)
            return pixel_image(np.repeat(a[..., None] * 255, 3, axis=2), s)
        right_c = np.array([3.0, -0.45, 0])
        img_same = gray(z["sc_img_same"]).move_to(right_c + 1.5 * LEFT)
        img_swap = gray(z["sc_img_swap"]).move_to(right_c + 1.5 * RIGHT)
        ov_same = ImageMobject(cam_overlay(z["sc_cam_same"], 64, 0.5)).set(height=s).move_to(img_same)
        ov_swap = ImageMobject(cam_overlay(z["sc_cam_swap"], 64, 0.5)).set(height=s).move_to(img_swap)
        cls = ["circle", "square"]
        hdr = Text("tiny CNN, shapes dataset with a leaked tag:\nbottom-right bar  —  horizontal = circle, vertical = square",
                   font_size=19, color=INK, line_spacing=0.8).move_to(right_c + 1.9 * UP)
        l_same = Text(f"says “{cls[int(z['sc_cls_same'])]}” (p={float(z['sc_prob_same']):.2f})", font_size=19,
                      color=GREEN_).next_to(img_same, DOWN, buff=0.15)
        l_swap = Text(f"says “{cls[int(z['sc_cls_swap'])]}” (p={float(z['sc_prob_swap']):.2f})", font_size=19,
                      color=RED_).next_to(img_swap, DOWN, buff=0.15)
        t_same = Text("circle + horizontal bar", font_size=17, color=MUTED).next_to(img_same, UP, buff=0.1)
        t_swap = Text("same circle, vertical bar", font_size=17, color=MUTED).next_to(img_swap, UP, buff=0.1)

        cap = swap_caption(self, cap, "now a model trained on shapes whose label also leaks through a corner tag")
        self.play(FadeOut(formula), FadeIn(hdr))
        self.play(FadeIn(img_same), FadeIn(t_same), FadeIn(l_same))
        self.wait(1.2)
        cap = swap_caption(self, cap, "it answers correctly; Grad-CAM puts heat on the circle and on the tag")
        self.play(FadeIn(ov_same), run_time=1.2)
        self.wait(2)
        cap = swap_caption(self, cap, "same circle, swapped tag: the heat is on the tag and the answer flips", color=RED_)
        self.play(FadeIn(img_swap), FadeIn(t_swap))
        self.play(FadeIn(ov_swap), FadeIn(l_swap))
        self.wait(1.5)
        acc = VGroup(
            Text(f"test accuracy: {float(z['sc_acc_same']):.0%} with honest tags, {float(z['sc_acc_swap']):.1%} with swapped tags",
                 font_size=19, color=RED_),
            Text(f"on average {float(z['sc_mass_same']):.0%} of the heat is in the tag corner (6% of the image)",
                 font_size=19, color=RED_)).arrange(DOWN, buff=0.1).move_to(right_c + 2.05 * DOWN + 0.4 * LEFT)
        cap = swap_caption(self, cap, "over 300 test images: the tag, not the shape, decides", color=RED_)
        self.play(FadeIn(acc))
        self.wait(2)
        cap = swap_caption(self, cap, "a shortcut: high accuracy for the wrong reason — Grad-CAM exposes it", color=PURPLE_)
        self.wait(2.5)
