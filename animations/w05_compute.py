"""Week 5 (compute and efficiency). Render one scene:
    manim -qh w05_compute.py Roofline
Data: data/w05_weights.npz (made by data/w05_07_precompute.py).
"""
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"
TEAL = "#0891B2"


def log_axes(x_rng, y_rng, xl, yl, x_ticks, y_ticks, pos):
    """Axes in log10 units with hand-written tick labels (x_ticks / y_ticks: list of (value, label))."""
    ax = Axes(x_range=[*x_rng, 1], y_range=[*y_rng, 1], x_length=xl, y_length=yl, tips=False,
              axis_config={"stroke_color": MUTED, "include_ticks": False}).move_to(pos)
    lab = VGroup()
    for v, s in x_ticks:
        p = ax.c2p(np.log10(v), y_rng[0])
        lab.add(Line(p, p + 0.08 * DOWN, color=MUTED), Text(s, font_size=18, color=MUTED).next_to(p, DOWN, buff=0.14))
    for v, s in y_ticks:
        p = ax.c2p(x_rng[0], np.log10(v))
        lab.add(Line(p, p + 0.08 * LEFT, color=MUTED), Text(s, font_size=18, color=MUTED).next_to(p, LEFT, buff=0.14))
    return ax, lab


class Roofline(Scene):
    """Roofline model on (approximate) H100 numbers: decoding at batch 1 sits on the bandwidth slope, batching
    climbs it, prefill sits on the compute roof; the A100 ridge for comparison. Target: L11b inference slides."""

    def construct(self):
        title(self, "The roofline", "is a layer limited by arithmetic or by memory traffic?")
        PEAK, BW = 989.0, 3.35            # H100 SXM, dense bf16 TFLOP/s and HBM TB/s (approx.)
        A_PEAK, A_BW = 312.0, 2.0         # A100 80GB (approx.)
        d = 8192
        I_of = lambda B: B * d / (d + 2 * B)      # y = xW, W d×d bf16: 2Bd² FLOPs / 2(d²+2Bd) bytes
        perf = lambda I, pk=PEAK, bw=BW: min(pk, bw * I)

        ax, ticks = log_axes([0, 4.2], [0, 3.3], 7.4, 4.0,
                             [(1, "1"), (10, "10"), (100, "100"), (1000, "1k"), (10000, "10k")],
                             [(1, "1"), (10, "10"), (100, "100"), (1000, "1000")], [-2.4, 0.0, 0])
        xlab = Text("arithmetic intensity I  (FLOPs per byte read from memory)", font_size=20).next_to(ax, DOWN, buff=0.45)
        ylab = Text("TFLOP/s", font_size=20).rotate(PI / 2).next_to(ax, LEFT, buff=0.6)
        P = lambda I, y: ax.c2p(np.log10(I), np.log10(y))

        def roof(pk, bw, color, dashed=False):
            ridge = pk / bw
            pts = [P(1, bw), P(ridge, pk), P(10 ** 4.2, pk)]
            line = VMobject(stroke_color=color, stroke_width=6).set_points_as_corners(pts)
            return DashedVMobject(line, num_dashes=40) if dashed else line

        h100 = roof(PEAK, BW, INK)
        ridge = PEAK / BW
        lab_peak = Text("compute peak ≈ 989 TFLOP/s", font_size=20).next_to(P(PEAK / BW, PEAK), LEFT, buff=0.35).shift(0.22 * UP)
        lab_bw = Text("memory slope: 3.35 TB/s × I", font_size=20).rotate(np.arctan2(*(P(100, 335) - P(1, 3.35))[[1, 0]]))
        lab_bw.move_to(P(12, 40.2)).shift(0.32 * UP + 0.18 * LEFT)
        ridge_dot = VGroup(Dot(P(ridge, PEAK), color=INK, radius=0.06),
                           DashedLine(P(ridge, 1), P(ridge, 150), color=MUTED, stroke_width=2))
        ridge_lab = Text(f"ridge ≈ {ridge:.0f} FLOP/B", font_size=18, color=MUTED).next_to(P(ridge, 2), RIGHT, buff=0.1)

        rx = 4.55
        f1 = MathTex(r"\text{FLOP/s} = \min(\text{peak},\ \text{BW}\times I)", font_size=30).move_to([rx, 1.7, 0])
        hw = Text("H100, bf16 (approx.)", font_size=20, color=MUTED).next_to(f1, DOWN, buff=0.2)
        cap = caption("1. speed = min(compute peak, memory bandwidth × intensity)")
        self.play(Create(ax), FadeIn(ticks), FadeIn(xlab), FadeIn(ylab), FadeIn(cap))
        self.play(Create(h100), run_time=1.5)
        self.play(FadeIn(lab_peak), FadeIn(lab_bw), FadeIn(ridge_dot), FadeIn(ridge_lab), FadeIn(f1), FadeIn(hw))
        self.wait(1.2)

        cap = swap_caption(self, cap, "2. a linear layer reuses each weight once per token: I ≈ B")
        f2 = MathTex(r"y = xW,\ \ W\in\mathbb{R}^{d\times d}", font_size=28).move_to([rx, 0.55, 0])
        f3 = MathTex(r"I = \frac{2Bd^2}{2(d^2 + 2Bd)} \approx B", font_size=30).next_to(f2, DOWN, buff=0.25)
        f4 = Text("B = tokens processed per weight read", font_size=18, color=MUTED).next_to(f3, DOWN, buff=0.15)
        self.play(FadeIn(f2), FadeIn(f3), FadeIn(f4))
        self.wait(1.5)

        B = ValueTracker(1)
        dot = always_redraw(lambda: Dot(P(I_of(B.get_value()), perf(I_of(B.get_value()))), color=RED_, radius=0.11))
        dlab = always_redraw(lambda: Text(f"decode, B = {B.get_value():.0f}", font_size=20, color=RED_)
                             .next_to(dot, RIGHT + DOWN, buff=0.08))

        def readout():
            b = B.get_value(); I = I_of(b); p = perf(I)
            return VGroup(Text(f"B = {b:.0f}   I ≈ {I:.0f}", font_size=22, color=RED_),
                          Text(f"{p:.1f} TFLOP/s = {p / PEAK:.1%} of peak", font_size=22, color=RED_)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([rx, -1.95, 0])
        ro = always_redraw(readout)
        cap = swap_caption(self, cap, "3. decoding one sequence: I ≈ 1, so it is memory-bound", color=RED_)
        self.play(FadeIn(dot, scale=2), FadeIn(dlab), FadeIn(ro))
        self.wait(1.5)
        cap = swap_caption(self, cap, "4. batching B users reuses each weight read: the point climbs")
        self.play(B.animate.set_value(256), run_time=5, rate_func=lambda t: (np.exp(t * np.log(256)) - 1) / 255)
        self.wait(1)

        Ipre = I_of(2048)
        pdot = Dot(P(Ipre, perf(Ipre)), color=GREEN_, radius=0.11)
        plab = Text("prefill, 2048 tokens", font_size=20, color=GREEN_).next_to(pdot, UP, buff=0.15)
        cap = swap_caption(self, cap, f"5. prefill: the whole prompt at once, I ≈ {Ipre:.0f}, compute-bound", color=GREEN_)
        self.play(FadeIn(pdot, scale=2), FadeIn(plab))
        self.wait(1.5)

        a100 = roof(A_PEAK, A_BW, MUTED, dashed=True)
        alab = Text(f"A100 (dashed): 312 TFLOP/s,\n2.0 TB/s, ridge ≈ {A_PEAK / A_BW:.0f} FLOP/B", font_size=16, color=MUTED,
                    line_spacing=0.8).move_to(P(3000, 25))
        cap = swap_caption(self, cap, "6. GPUs gain compute faster than bandwidth: the ridge moves right")
        self.play(Create(a100), FadeIn(alab), FadeOut(dlab), run_time=1.5)
        self.wait(1.5)
        cap = swap_caption(self, cap, "decode ≈ bandwidth / model bytes: 3.35 TB/s ÷ 16 GB ≈ 200 tokens/s",
                           color=PURPLE_)
        self.wait(3)


class MemoryBudget(Scene):
    """GPU memory as a stacked bar: weights, gradients, Adam states, activations, KV cache, while the model and the
    context grow; bf16 vs fp32; the 2N / 6N FLOPs rule. Target: new slide (compute & memory budget)."""

    def construct(self):
        title(self, "Where the GPU memory goes", "Llama-3-style models, approximate GB (1 GB = 10⁹ bytes)")
        names = ["weights", "gradients", "Adam m, v + fp32 master", "activations", "KV cache"]
        cols = [BLUE_, ORANGE_, PURPLE_, GREEN_, TEAL]
        T = [ValueTracker(0.0) for _ in names]
        ymax = ValueTracker(160.0)
        x0, bw, y0, H = -3.7, 1.7, -2.75, 4.55
        axis_x = x0 - bw / 2 - 0.45

        def nice(m):
            raw = m / 4
            k = 10 ** np.floor(np.log10(raw))
            return next(s * k for s in (1, 2, 5, 10) if s * k >= raw)

        def axis():
            m = ymax.get_value()
            st = nice(m)
            g = VGroup(Line([axis_x, y0, 0], [axis_x, y0 + H, 0], color=MUTED, stroke_width=2))
            v = 0.0
            while v <= m + 1e-6:
                y = y0 + v / m * H
                g.add(Line([axis_x - 0.08, y, 0], [axis_x, y, 0], color=MUTED, stroke_width=2),
                      Text(f"{v:.0f}", font_size=16, color=MUTED).next_to([axis_x - 0.08, y, 0], LEFT, buff=0.08))
                v += st
            g.add(Text("GB", font_size=18, color=MUTED).move_to([axis_x - 0.35, y0 + H + 0.25, 0]))
            return g

        def stack():
            m, y, g = ymax.get_value(), y0, VGroup()
            for t, c in zip(T, cols):
                h = t.get_value() / m * H
                if h > 1e-3:
                    g.add(Rectangle(width=bw, height=h, stroke_color=WHITE, stroke_width=1.5, fill_color=c,
                                    fill_opacity=0.9).move_to([x0, y + h / 2, 0]))
                y += h
            tot = sum(t.get_value() for t in T)
            g.add(Text(f"{tot:,.0f} GB", font_size=24, weight=BOLD).move_to([x0, min(y, y0 + H) + 0.25, 0]))
            return g

        def gpu_line():
            m = ymax.get_value()
            y = y0 + 80 / m * H
            return VGroup(DashedLine([axis_x, y, 0], [x0 + bw / 2 + 0.15, y, 0], color=RED_, stroke_width=3),
                          Text("80 GB", font_size=18, color=RED_).next_to([x0 + bw / 2 + 0.15, y, 0], RIGHT, buff=0.08))

        bpp = ["2 B/param", "2 B/param", "12 B/param", "∝ batch·ctx·width·depth", "∝ batch·ctx·layers·kv-width"]

        def legend():
            g, lx, top = VGroup(), -0.95, 1.55
            for r, i in enumerate(reversed(range(5))):
                v = T[i].get_value()
                y = top - r * 0.78
                row = VGroup(Square(0.26, fill_color=cols[i], fill_opacity=0.9, stroke_width=0).move_to([lx + 0.13, y, 0]),
                             Text(names[i], font_size=20).move_to([lx + 0.4, y, 0], aligned_edge=LEFT),
                             Text(f"{v:,.1f} GB", font_size=20, weight=BOLD).move_to([lx + 5.0, y, 0], aligned_edge=RIGHT),
                             Text(bpp[i], font_size=15, color=MUTED).move_to([lx + 0.4, y - 0.27, 0], aligned_edge=LEFT))
                g.add(row.set_opacity(1.0 if v > 1e-3 else 0.25))
            return g

        ax_, st_, gl_, lg_ = always_redraw(axis), always_redraw(stack), always_redraw(gpu_line), always_redraw(legend)
        setup = Text("Llama-3 8B, fp32", font_size=26, color=BLUE_).move_to([6.6, 2.3, 0], aligned_edge=RIGHT)
        self.add(ax_, st_, gl_, lg_)
        cap = caption("1. weights: 8B parameters × 4 bytes in fp32 = 32 GB")
        self.play(FadeIn(cap), FadeIn(setup), T[0].animate.set_value(32), run_time=1.5)
        self.wait(1)
        cap = swap_caption(self, cap, "bf16 halves it: 2 bytes per parameter = 16 GB (inference fits one GPU)")
        self.play(T[0].animate.set_value(16), Transform(setup, Text("Llama-3 8B, bf16", font_size=26, color=BLUE_).move_to([6.6, 2.3, 0], aligned_edge=RIGHT)))
        self.wait(1.2)
        cap = swap_caption(self, cap, "2. training adds gradients + Adam (m, v, fp32 master): 16 bytes/param")
        self.play(T[1].animate.set_value(16), run_time=1)
        self.play(T[2].animate.set_value(96), run_time=1.5)
        self.wait(1.2)
        cap = swap_caption(self, cap, "3. activations saved for backward: ≈34·tokens·width bytes per layer")
        self.play(T[3].animate.set_value(34 * 4096 * 4096 * 32 / 1e9),
                  Transform(setup, Text("8B, bf16, 4k tokens", font_size=26, color=BLUE_).move_to([6.6, 2.3, 0], aligned_edge=RIGHT)), run_time=1.5)
        self.wait(1.2)
        cap = swap_caption(self, cap, "4. scale to 70B: ≈1.2 TB of training state, at least 16 H100s before any data", color=RED_)
        self.play(T[0].animate.set_value(140), T[1].animate.set_value(140), T[2].animate.set_value(840),
                  T[3].animate.set_value(34 * 4096 * 8192 * 80 / 1e9), ymax.animate.set_value(1400),
                  Transform(setup, Text("70B, bf16, 4k tokens", font_size=26, color=BLUE_).move_to([6.6, 2.3, 0], aligned_edge=RIGHT)), run_time=3)
        self.wait(1.5)

        kv_tok = 2 * 32 * 8 * 128 * 2 / 1e9      # K and V × 32 layers × 8 kv heads × 128 dims × 2 bytes (GQA)
        cap = swap_caption(self, cap, "5. inference (8B): only weights + KV cache; the cache grows with context")
        ctx = ValueTracker(1024)
        bsz = ValueTracker(1)
        setup2 = always_redraw(lambda: Text(f"8B inference, batch {bsz.get_value():.0f}, context {ctx.get_value() / 1024:.0f}k", font_size=24,
                                            color=TEAL).move_to([6.6, 2.3, 0], aligned_edge=RIGHT))
        self.play(*[T[i].animate.set_value(0) for i in (1, 2, 3)], T[0].animate.set_value(16),
                  T[4].animate.set_value(kv_tok * 1024), ymax.animate.set_value(100), FadeOut(setup), FadeIn(setup2),
                  run_time=2)
        self.play(ctx.animate.set_value(131072), T[4].animate.set_value(kv_tok * 131072), run_time=3, rate_func=linear)
        self.wait(1)
        cap = swap_caption(self, cap, f"serve 16 users at 32k: {kv_tok * 16 * 32768:.0f} GB of cache, more than the weights", color=TEAL)
        self.play(T[4].animate.set_value(kv_tok * 16 * 32768), ctx.animate.set_value(32768), bsz.animate.set_value(16), run_time=2)
        self.wait(1.5)

        box = VGroup(
            MathTex(r"\text{forward: } 2N \text{ FLOPs per token}", font_size=30),
            MathTex(r"\text{training: } 6N \text{ FLOPs per token}", font_size=30),
            MathTex(r"8\text{B}\times 15\text{T tokens}: 6ND \approx 7\times10^{23}", font_size=28),
            Text("≈ 0.5M H100-hours at 40% utilization", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        frame = SurroundingRectangle(box, color=PURPLE_, buff=0.25, corner_radius=0.1, fill_color=WHITE, fill_opacity=1)
        grp = VGroup(frame, box).move_to([4.15, -1.95, 0])
        cap = swap_caption(self, cap, "FLOPs: 2N per token forward, 6N with backward → training ≈ 6·N·D", color=PURPLE_)
        self.play(FadeIn(grp, shift=0.2 * UP), FadeOut(lg_))
        self.wait(3.5)


class Quantization(Scene):
    """Real weights (SmolLM2-135M, one MLP matrix) snapped to an absmax int8 grid, then int4; one outlier ruins
    per-tensor int4, a scale per group of 128 fixes it; relative error vs bits. Target: L11b quantization slide."""

    def construct(self):
        z = np.load(DATA / "w05_weights.npz")
        sample, group, absmax = z["sample"], z["group"], float(z["absmax"])
        bits, e_t, e_g = list(z["bits"]), z["err_tensor"], z["err_group"]
        title(self, "Quantization", "SmolLM2-135M, layer 10 MLP down-projection (576×1536 real weights)")
        R = 0.9                                   # visible weight range [-R, R]
        L, Rx, base, hmax = -6.6, 0.6, -2.35, 2.6
        X = lambda w: L + (np.clip(w, -R, R) + R) / (2 * R) * (Rx - L)
        nbins = 73
        edges = np.linspace(-R, R, nbins + 1)
        dens, _ = np.histogram(sample, edges, density=True)
        dmax = dens.max() * 1.1

        def hist_bars(values, positions=None, width=None, color=BLUE_):
            g = VGroup()
            for v, p in zip(values, positions):
                h = min(hmax + 0.25, v / dmax * hmax)
                if h > 0.005:
                    g.add(Rectangle(width=width, height=h, stroke_width=0, fill_color=color, fill_opacity=0.85)
                          .move_to([X(p), base + h / 2, 0]))
            return g

        bw_ = (Rx - L) / nbins
        orig = hist_bars(dens, (edges[:-1] + edges[1:]) / 2, bw_ * 0.92)
        axis = Line([L, base, 0], [Rx, base, 0], color=MUTED, stroke_width=2)
        tl = VGroup(*[VGroup(Line([X(v), base, 0], [X(v), base - 0.08, 0], color=MUTED),
                             Text(f"{v:g}", font_size=16, color=MUTED).move_to([X(v), base - 0.25, 0]))
                      for v in (-0.8, -0.4, 0, 0.4, 0.8)])
        out_arr = Arrow([Rx - 0.9, 0.5, 0], [Rx + 0.15, 0.5, 0], color=RED_, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.25)
        out_lab = Text(f"one outlier at {absmax:.2f}\n(= {absmax / float(z['std']):.0f} std)", font_size=17, color=RED_,
                       line_spacing=0.8).next_to(out_arr, UP, buff=0.08)
        hl = Text(f"std {float(z['std']):.2f}", font_size=18, color=MUTED).move_to([L + 0.9, 0.2, 0])

        cap = caption("1. trained weights: bell-shaped around 0, stored in bf16 (16 bits each)")
        self.play(FadeIn(cap), Create(axis), FadeIn(tl))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in orig], lag_ratio=0.01), run_time=1.5)
        self.play(FadeIn(hl), GrowArrow(out_arr), FadeIn(out_lab))
        self.wait(1)

        # a row of real weights (one contiguous group) on a number line above the histogram
        rng = np.random.default_rng(1)
        pts = np.sort(rng.choice(group, 26, replace=False))
        ly = 1.45
        nl = Line([L, ly, 0], [Rx, ly, 0], color=MUTED, stroke_width=1.5)
        nl_lab = Text("26 weights from one row", font_size=17, color=MUTED).next_to(nl, UP, buff=0.12).align_to(nl, LEFT)
        dots = VGroup(*[Dot([X(w), ly, 0], radius=0.06, color=INK) for w in pts])
        self.play(Create(nl), FadeIn(nl_lab), LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.03), run_time=1)

        def grid(scale, q, color):
            g = VGroup()
            for k in range(-q, q + 1):
                v = k * scale
                if abs(v) <= R:
                    g.add(Line([X(v), base, 0], [X(v), ly + 0.18, 0], color=color, stroke_width=1.2, stroke_opacity=0.45))
            return g

        def snap(scale, q):
            return np.clip(np.round(pts / scale), -q, q) * scale

        def qhist(scale, q, color, wpos=None):
            lv = np.arange(-q, q + 1) * scale
            src = sample if wpos is None else wpos
            qv = np.clip(np.round(src / scale), -q, q)
            cnt = np.array([(qv == k).sum() for k in range(-q, q + 1)]) / len(src) / scale
            keep = np.abs(lv) <= R
            return hist_bars(cnt[keep], lv[keep], min(bw_ * 0.92, scale / (2 * R) * (Rx - L) * 0.7), color)

        # right panel: error vs bits
        ex, ey, ew, eh = 2.1, -2.35, 4.0, 3.6
        lo, hi = -2.3, 0.1                        # log10 relative error range
        EP = lambda b, e: np.array([ex + (b - 2) / 6 * ew, ey + (np.log10(e) - lo) / (hi - lo) * eh, 0])
        eax = VGroup(Line([ex, ey, 0], [ex + ew, ey, 0], color=MUTED), Line([ex, ey, 0], [ex, ey + eh, 0], color=MUTED))
        for b in bits:
            eax.add(Text(str(b), font_size=16, color=MUTED).move_to([EP(b, 1)[0], ey - 0.22, 0]))
        for e, s in [(1, "100%"), (0.1, "10%"), (0.01, "1%")]:
            eax.add(Text(s, font_size=16, color=MUTED).next_to([ex, EP(2, e)[1], 0], LEFT, buff=0.1),
                    Line([ex, EP(2, e)[1], 0], [ex + ew, EP(2, e)[1], 0], color=MUTED, stroke_width=1, stroke_opacity=0.3))
        eax.add(Text("bits per weight", font_size=18).move_to([ex + ew / 2, ey - 0.55, 0]),
                Text("relative RMS error", font_size=18).move_to([ex + ew / 2, ey + eh + 0.3, 0]))

        def err_dot(b, e, color, dirn=LEFT + DOWN):
            d = Dot(EP(b, e), color=color, radius=0.08)
            return VGroup(d, Text(f"{e:.1%}", font_size=17, color=color).next_to(d, dirn, buff=0.06))

        # int8, per-tensor
        s8 = absmax / 127
        g8 = grid(s8, 127, PURPLE_)
        cap = swap_caption(self, cap, f"2. int8: scale = max|w| / 127, round every weight to the nearest of 255 levels", color=PURPLE_)
        self.play(FadeIn(g8), Create(eax))
        tgt8 = snap(s8, 127)
        self.play(*[d.animate.move_to([X(t), ly, 0]).set_color(PURPLE_) for d, t in zip(dots, tgt8)], run_time=1)
        q8 = qhist(s8, 127, PURPLE_)
        self.play(Transform(orig, q8), FadeIn(err_dot(8, e_t[6], PURPLE_)), run_time=1.2)
        self.wait(1)

        # int4 per tensor: the outlier sets the step
        s4 = absmax / 7
        g4 = grid(s4, 7, RED_)
        tgt4 = snap(s4, 7)
        cap = swap_caption(self, cap, f"3. int4 per tensor: 15 levels, step {s4:.2f} set by the outlier; almost every weight → 0", color=RED_, size=26)
        self.play(FadeOut(g8), FadeIn(g4))
        self.play(*[d.animate.move_to([X(t), ly, 0]).set_color(RED_) for d, t in zip(dots, tgt4)], run_time=1)
        self.play(Transform(orig, qhist(s4, 7, RED_)), run_time=1.2)
        t_curve = VMobject(stroke_color=RED_, stroke_width=4).set_points_as_corners([EP(b, e) for b, e in zip(bits, e_t)])
        self.play(FadeIn(err_dot(4, e_t[2], RED_)), Create(t_curve), run_time=1.2)
        tl_lab = Text("one scale\nper tensor", font_size=17, color=RED_, line_spacing=0.8).move_to(EP(6.7, 0.6))
        self.play(FadeIn(tl_lab))
        self.wait(1.2)

        # int4 per group of 128
        sg = np.abs(group).max() / 7
        gg = grid(sg, 7, GREEN_)
        tgtg = snap(sg, 7)
        # histogram with every 128-weight group quantized on its own scale (the random sample stands in for the groups)
        gq = sample.reshape(-1, 128)
        sc = np.abs(gq).max(1, keepdims=True) / 7
        deq = (np.round(gq / sc) * sc).flatten()
        dens_g, _ = np.histogram(deq, edges, density=True)
        cap = swap_caption(self, cap, f"4. int4 with one scale per group of 128 weights: step {sg:.2f}, the shape survives", color=GREEN_, size=26)
        self.play(FadeOut(g4), FadeIn(gg))
        self.play(*[d.animate.move_to([X(t), ly, 0]).set_color(GREEN_) for d, t in zip(dots, tgtg)], run_time=1)
        self.play(Transform(orig, hist_bars(dens_g, (edges[:-1] + edges[1:]) / 2, bw_ * 0.92, GREEN_)), run_time=1.2)
        g_curve = VMobject(stroke_color=GREEN_, stroke_width=4).set_points_as_corners([EP(b, e) for b, e in zip(bits, e_g)])
        gl_lab = Text("one scale per\n128 weights", font_size=17, color=GREEN_, line_spacing=0.8).move_to(EP(3.3, 0.02))
        self.play(FadeIn(err_dot(4, e_g[2], GREEN_)), Create(g_curve), FadeIn(gl_lab), run_time=1.2)
        self.wait(1)
        cap = swap_caption(self, cap, "each bit halves the error; int4 + group scales = 4× smaller than bf16 at ~13% error",
                           color=PURPLE_, size=26)
        self.wait(3)


class Parallelism(Scene):
    def swap_head(self, old, new, *anims, run_time=1.0):
        self.play(FadeOut(old), FadeIn(new), *anims, run_time=run_time)
        return new

    """Four ways to split training over 4 GPUs: data, tensor, pipeline, FSDP. What each GPU stores, which part of
    the batch it sees, and what goes over the wire. Target: new slide (distributed training)."""

    def construct(self):
        title(self, "Splitting training over 4 GPUs", "a 4-layer model, a batch of 4 samples")
        LC = [BLUE_, ORANGE_, GREEN_, PURPLE_]
        gx = [-4.8, -1.6, 1.6, 4.8]
        gw, lh = 2.5, 0.48
        ly = [0.9, 0.25, -0.4, -1.05]       # layer slots (top = layer 1)
        gpus = VGroup()
        for i, x in enumerate(gx):
            box = RoundedRectangle(corner_radius=0.15, width=gw, height=3.15, stroke_color=INK, stroke_width=2,
                                   fill_color=SOFT, fill_opacity=1).move_to([x, -0.1, 0])
            lab = Text(f"GPU {i}", font_size=20, weight=BOLD).next_to(box, UP, buff=0.08)
            gpus.add(VGroup(box, lab))
        slots = VGroup(*[VGroup(*[Rectangle(width=2.0, height=lh, stroke_color=MUTED, stroke_width=1.2)
                                  .set_fill(WHITE, 1).move_to([x, y, 0]) for y in ly]) for x in gx])
        self.play(FadeIn(gpus), FadeIn(slots))

        def layer(gi, li, frac=(0, 1), op=0.9):
            a, b = frac
            return Rectangle(width=2.0 * (b - a), height=lh, stroke_width=0, fill_color=LC[li], fill_opacity=op).move_to(
                [gx[gi] - 1.0 + 2.0 * (a + b) / 2, ly[li], 0])

        def lname(gi, li, dx=0.0):
            return Text(f"L{li + 1}", font_size=16, color=WHITE, weight=BOLD).move_to([gx[gi] + dx, ly[li], 0])

        def sample(k, pos):
            sq = Square(0.3, stroke_width=1.5, stroke_color=INK, fill_color=GRAY_B, fill_opacity=1).move_to(pos)
            return VGroup(sq, Text(f"x{k}", font_size=14).move_to(pos))

        def mem(txt):
            return VGroup(*[Text(txt, font_size=17).move_to([x, -1.87, 0]) for x in gx])

        def heading(name, color, sends):
            h = VGroup(Text(name, font_size=26, weight=BOLD, color=color),
                       Text("sends: " + sends, font_size=22, color=RED_)).arrange(RIGHT, buff=0.4)
            return h.move_to([-6.6, 2.3, 0], aligned_edge=LEFT)

        def gap_arrows(y, width=6):
            return VGroup(*[DoubleArrow([gx[i] + 0.95, y, 0], [gx[i + 1] - 0.95, y, 0], buff=0, color=RED_,
                                        stroke_width=width, tip_length=0.16, max_tip_length_to_length_ratio=0.4)
                            for i in range(3)])

        dy = -2.47          # data row
        cap = caption("1. data parallel: full model on every GPU, 1/4 of the batch each")
        head = heading("Data parallel", BLUE_, "gradients (all-reduce), once per step")
        content = VGroup(*[layer(g, l) for g in range(4) for l in range(4)], *[lname(g, l) for g in range(4) for l in range(4)])
        data = VGroup(*[sample(g, [gx[g], dy, 0]) for g in range(4)])
        mm = mem("100% params")
        self.play(FadeIn(cap), FadeIn(head[0]), FadeIn(content), FadeIn(data, shift=0.2 * UP), FadeIn(mm))
        self.wait(1)
        cap = swap_caption(self, cap, "after backward: all-reduce (sum) the gradients, then identical updates")
        arr = gap_arrows(-0.1, 10)
        self.play(FadeIn(head[1]), GrowFromCenter(arr), run_time=0.8)
        self.play(*[Indicate(c, color=RED_, scale_factor=1.03) for c in content[:16]], run_time=1)
        self.wait(0.6)
        self.play(FadeOut(arr))

        # tensor parallel
        cap = swap_caption(self, cap, "2. tensor parallel: each weight matrix is cut into 4 column slices")
        tp = VGroup(*[layer(g, l, (g / 4, (g + 1) / 4)) for g in range(4) for l in range(4)])
        data_all = VGroup(*[VGroup(*[sample(k, [gx[g] - 0.54 + 0.36 * k, dy, 0]) for k in range(4)]) for g in range(4)])
        head = self.swap_head(head, heading("Tensor parallel", ORANGE_, "activations (all-reduce), every layer"),
                  ReplacementTransform(content, tp), ReplacementTransform(data, data_all),
                  Transform(mm, mem("25% params")), run_time=1.5)
        self.wait(0.6)
        cap = swap_caption(self, cap, "same batch everywhere; partial outputs combined after every layer")
        for l in range(4):
            arr = gap_arrows(ly[l])
            self.play(FadeIn(arr), Indicate(VGroup(*[tp[g * 4 + l] for g in range(4)]), color=RED_, scale_factor=1.0), run_time=0.45)
            self.play(FadeOut(arr), run_time=0.25)
        cap = swap_caption(self, cap, "traffic every layer: only fast within one node (NVLink)")
        self.wait(1)

        # pipeline parallel
        cap = swap_caption(self, cap, "3. pipeline: GPU i holds layer i; micro-batches flow stage to stage")
        pp = VGroup(*[layer(g, g) for g in range(4)], *[lname(g, g, -0.45) for g in range(4)])
        head = self.swap_head(head, heading("Pipeline parallel", GREEN_, "activations to the next GPU only"),
                  FadeOut(tp), FadeIn(pp), FadeOut(data_all), Transform(mm, mem("25% params")), run_time=1.2)
        mbs = [sample(k, [gx[0] - 1.65, ly[0], 0]) for k in range(4)]
        idle = VGroup()
        for t in range(8):           # at tick t, micro-batch k is at stage t-k
            anims = []
            for k in range(4):
                s = t - k
                if 0 <= s <= 3:
                    if s == 0:
                        self.add(mbs[k])
                    anims.append(mbs[k].animate.move_to([gx[s] + 0.55, ly[s], 0]))
                elif s == 4:
                    anims.append(FadeOut(mbs[k]))
            busy = {t - k for k in range(4) if 0 <= t - k <= 3}
            new_idle = VGroup(*[Text("idle", font_size=18, color=WHITE, weight=BOLD).move_to([gx[g] + 0.55, ly[g], 0])
                                for g in range(4) if g not in busy])
            anims += [FadeOut(idle), FadeIn(new_idle)]
            idle = new_idle
            self.play(*anims, run_time=0.6)
            if t == 1:
                cap = swap_caption(self, cap, "stages wait at the start and the end: the pipeline bubble")
        self.play(FadeOut(idle))
        self.wait(0.4)

        # FSDP
        cap = swap_caption(self, cap, "4. FSDP: data parallel, but weights, grads, Adam states sharded 4 ways")
        fs = VGroup(*[layer(g, l, (g / 4, (g + 1) / 4)) for g in range(4) for l in range(4)])
        data = VGroup(*[sample(g, [gx[g], dy, 0]) for g in range(4)])
        head = self.swap_head(head, heading("FSDP (ZeRO-3)", PURPLE_, "weights (all-gather), grads (reduce-scatter)"),
                  FadeOut(pp), FadeIn(fs), FadeIn(data, shift=0.2 * UP), Transform(mm, mem("25% params + states")),
                  run_time=1.2)
        self.wait(0.6)
        cap = swap_caption(self, cap, "before each layer: all-gather its full weights, compute, free them")
        for l in range(4):
            full = VGroup(*[layer(g, l, (s / 4, (s + 1) / 4), op=0.45) for g in range(4) for s in range(4) if s != g])
            self.play(FadeIn(full), FadeIn(gap_arrows(ly[l]), rate_func=there_and_back), run_time=0.45)
            self.play(FadeOut(full), run_time=0.3)
        cap = swap_caption(self, cap, "FSDP: ~1/4 of the memory per GPU for ~1.5× the traffic of data parallel",
                           color=PURPLE_, size=26)
        self.wait(3)
