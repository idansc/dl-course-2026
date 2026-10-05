"""Week 5 (compute and scaling laws). Render one scene:
    manim -qh w05_scaling.py PowerLawFrontier
Every curve and number is computed from the published Chinchilla fit (Hoffmann et al. 2022, Approach 3):
    L(N, D) = E + A / N^alpha + B / D^beta,  E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28,  C = 6 N D.
"""
from style import *

TEAL = "#0891B2"
E_, A_, B_, AL, BE = 1.69, 406.4, 410.7, 0.34, 0.28
SRC = "fit: Hoffmann et al. 2022 (Chinchilla), Approach 3"
G_ = (AL * A_ / (BE * B_)) ** (1 / (AL + BE))     # N_opt = G (C/6)^(beta/(alpha+beta))
EXP_N = BE / (AL + BE)                           # ≈ 0.452
EXP_L = AL * BE / (AL + BE)                      # ≈ 0.154: (L_opt - E) ∝ C^-EXP_L


def loss(N, D):
    return E_ + A_ / N ** AL + B_ / D ** BE


def n_opt(C):
    return G_ * (C / 6) ** EXP_N


def loss_opt(C):
    N = n_opt(C)
    return loss(N, C / (6 * N))


def sci(x, digits=1):
    """2.3e+08 -> 2.3×10⁸ as LaTeX."""
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return rf"{m:.{digits}f}\times10^{{{e}}}"


def sup(x):
    """7.2e19 -> '7×10¹⁹' (plain text)."""
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"{m:.0f}×10" + str(e).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))


def human(n):
    for v, s in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if n >= v:
            x = n / v
            return f"{x:.0f}{s}" if x >= 9.95 else f"{x:.1f}{s}".replace(".0" + s, s)
    return f"{n:.0f}"


def pow10(e, size=26, color=MUTED):
    return MathTex(rf"10^{{{e}}}", font_size=size, color=color)


def polyline(pts, color, width=4, opacity=1.0):
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    if len(pts) >= 2:
        m.set_points_as_corners(pts)
    return m


def frame_axes(x0, y0, w, h):
    """Plain L-shaped axes in screen units; the caller maps data to [x0, x0+w] × [y0, y0+h]."""
    return VGroup(Line([x0, y0, 0], [x0 + w, y0, 0], color=MUTED, stroke_width=2),
                  Line([x0, y0, 0], [x0, y0 + h, 0], color=MUTED, stroke_width=2))


class PowerLawFrontier(Scene):
    """Loss vs training compute (log-log) for 7 model sizes, each a learning curve L(N, C/6N); their lower envelope is
    the compute-efficient frontier; after subtracting E it is a straight line (power law). Target: w05 scaling-laws slide."""

    def construct(self):
        title(self, "The compute-efficient frontier", SRC)
        Ns = [1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11]
        names = ["100M", "300M", "1B", "3B", "10B", "30B", "100B"]
        cols = color_gradient([ManimColor("#93C5FD"), ManimColor(BLUE_), ManimColor(PURPLE_), ManimColor("#3B0764")], len(Ns))
        X0, Y0, W, H = -5.9, -2.05, 8.0, 4.1
        cx = (18.0, 25.0)
        Lbot, Ltop = 1.82, 6.0
        s = ValueTracker(0.0)      # 0: y = log L,  1: y = log(L - E)

        def P(lc, L, sv=None):
            sv = s.get_value() if sv is None else sv
            lo, hi = np.log10(Lbot - sv * E_), np.log10(Ltop - sv * E_)
            return np.array([X0 + (lc - cx[0]) / (cx[1] - cx[0]) * W,
                             Y0 + (np.log10(L - sv * E_) - lo) / (hi - lo) * H, 0])

        lcs = np.linspace(cx[0], cx[1], 400)
        Lc = [loss(N, 10 ** lcs / (6 * N)) for N in Ns]
        env = np.min(Lc, axis=0)

        def curve(L, color, width=4, opacity=1.0, sv=None):
            keep = (L <= Ltop) & (L >= Lbot)
            return polyline([P(x, y, sv) for x, y in zip(lcs[keep], L[keep])], color, width, opacity)

        axes = frame_axes(X0, Y0, W, H)
        xt = VGroup()
        for e in range(18, 26):
            p = P(e, 3, 0) * np.array([1, 0, 0]) + np.array([0, Y0, 0])
            xt.add(Line(p, p + 0.08 * DOWN, color=MUTED), pow10(e).next_to(p, DOWN, buff=0.12))
        xlab = Text("training compute C (FLOPs)", font_size=20).next_to(axes, DOWN, buff=0.55)

        def yticks(vals, sv, fmt):
            g = VGroup()
            for v in vals:
                y = P(cx[0], v + sv * E_, sv)[1]
                g.add(Line([X0, y, 0], [X0 - 0.08, y, 0], color=MUTED),
                      Text(fmt(v), font_size=20, color=MUTED).next_to([X0 - 0.08, y, 0], LEFT, buff=0.1))
            return g

        yt0 = yticks([2, 3, 4, 5, 6], 0, lambda v: f"{v:g}")
        yt1 = yticks([0.2, 0.5, 1, 2, 4], 1, lambda v: f"{v:g}")
        ylab0 = Text("loss L (nats/token)", font_size=20).rotate(PI / 2).move_to([X0 - 0.85, Y0 + H / 2, 0])
        ylab1 = Text("reducible loss L − E", font_size=20).rotate(PI / 2).move_to([X0 - 0.85, Y0 + H / 2, 0])

        rx = 4.85
        f1 = MathTex(r"L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}", font_size=32).move_to([rx, 1.75, 0])
        f2 = MathTex(r"E{=}1.69,\ A{=}406.4,\ B{=}410.7", font_size=27, color=MUTED).next_to(f1, DOWN, buff=0.2)
        f3 = MathTex(r"\alpha{=}0.34,\ \beta{=}0.28,\quad D = C/6N", font_size=27, color=MUTED).next_to(f2, DOWN, buff=0.1)
        legend = VGroup()
        for i, (nm, c) in enumerate(zip(names, cols)):
            legend.add(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=c, stroke_width=6),
                              Text(f"N = {nm}", font_size=19)).arrange(RIGHT, buff=0.15))
        legend.arrange_in_grid(rows=4, cols=2, col_alignments="ll", buff=(0.5, 0.17), flow_order="dr")
        legend.move_to([rx, -0.75, 0])
        lhdr = Text("model size (parameters)", font_size=19, color=MUTED).next_to(legend, UP, buff=0.2)

        cap = caption("1. the Chinchilla fit: loss of N parameters trained on D = C/6N tokens")
        self.play(FadeIn(cap), Create(axes), FadeIn(xt), FadeIn(yt0), FadeIn(xlab), FadeIn(ylab0), FadeIn(f1),
                  FadeIn(f2), FadeIn(f3))
        self.wait(1.2)

        curves = VGroup(*[always_redraw(lambda L=L, c=c: curve(L, c)) for L, c in zip(Lc, cols)])
        cap = swap_caption(self, cap, "2. one learning curve per model size: more compute = more tokens seen")
        self.play(FadeIn(lhdr))
        drawn = [curve(Lc[i], cols[i]) for i in range(len(Ns))]
        for i in range(len(Ns)):
            self.play(Create(drawn[i]), FadeIn(legend[i]), run_time=0.45)
        self.remove(*drawn)
        self.add(curves)
        self.wait(0.8)

        envl = always_redraw(lambda: curve(env, INK, 9, 0.35))
        cap = swap_caption(self, cap, "3. the lower envelope: best loss reachable at each compute budget")
        e0 = curve(env, INK, 9, 0.35)
        self.play(Create(e0), run_time=1.8)
        self.remove(e0)
        self.add(envl)
        self.wait(1)

        # window where the 1B model is the best of the seven
        k = names.index("1B")
        best = np.argmin(Lc, axis=0) == k
        lo_c, hi_c = lcs[best][0], lcs[best][-1]
        band = always_redraw(lambda: Rectangle(width=P(hi_c, 3)[0] - P(lo_c, 3)[0], height=H, stroke_width=0,
                                               fill_color=cols[k], fill_opacity=0.12)
                             .move_to([(P(hi_c, 3)[0] + P(lo_c, 3)[0]) / 2, Y0 + H / 2, 0]))
        hl = always_redraw(lambda: curve(Lc[k], cols[k], 8))
        blab = Text(f"1B is best for\n{sup(10 ** lo_c)} – {sup(10 ** hi_c)} FLOPs", font_size=19,
                    color=cols[k], line_spacing=0.8).move_to([(P(hi_c, 3)[0] + P(lo_c, 3)[0]) / 2, Y0 + 0.45, 0])
        cap = swap_caption(self, cap, "4. each model size is the best choice only inside a window of compute")
        self.play(FadeIn(band), FadeIn(hl), FadeIn(blab))
        self.wait(1.2)
        cap = swap_caption(self, cap, "left of it: a smaller model is better; right of it: the 1B model has plateaued", size=26)
        self.wait(2)
        self.play(FadeOut(band), FadeOut(hl), FadeOut(blab))

        # morph to log(L - E)
        cap = swap_caption(self, cap, f"5. subtract the irreducible loss E = {E_}: the frontier becomes a straight line")
        self.play(FadeOut(yt0), FadeOut(ylab0), run_time=0.5)
        self.play(s.animate.set_value(1.0), run_time=2.5)
        self.play(FadeIn(yt1), FadeIn(ylab1), run_time=0.5)
        opt = DashedVMobject(polyline([P(x, loss_opt(10 ** x), 1.0) for x in lcs], RED_, 4), num_dashes=60)
        slope = MathTex(rf"L_{{\text{{opt}}}} - E \propto C^{{-{EXP_L:.3f}}}", font_size=32, color=RED_)
        slope.move_to(P(21.0, loss_opt(1e21), 1.0) + np.array([-0.6, -0.8, 0]))
        self.play(Create(opt), run_time=1.5)
        self.play(FadeIn(slope))
        self.wait(1.2)
        r10 = 1 - 10 ** -EXP_L
        cap = swap_caption(self, cap, f"a power law, exponent αβ/(α+β) = {EXP_L:.3f}: 10× compute cuts "
                                      f"reducible loss by {r10:.0%}", color=PURPLE_, size=26)
        self.wait(3)


class IsoFLOPProfiles(Scene):
    """IsoFLOP valleys: for each fixed budget C, loss vs N has a minimum; the minima follow N_opt ∝ C^0.452 and
    D_opt ∝ C^0.548; then the 3-line derivation. Target: w05 scaling-laws slide (Chinchilla)."""

    def construct(self):
        title(self, "IsoFLOP profiles", SRC)
        budgets = [19, 20, 21, 22, 23, 24]
        cols = color_gradient([ManimColor("#93C5FD"), ManimColor(BLUE_), ManimColor(PURPLE_), ManimColor("#3B0764")], len(budgets))
        X0, Y0, W, H = -5.7, -2.05, 6.6, 4.1
        nx, ly = (7.0, 12.0), (1.8, 4.0)
        P = lambda ln, L: np.array([X0 + (ln - nx[0]) / (nx[1] - nx[0]) * W, Y0 + (L - ly[0]) / (ly[1] - ly[0]) * H, 0])
        lns = np.linspace(nx[0], nx[1], 400)

        axes = frame_axes(X0, Y0, W, H)
        ticks = VGroup()
        for e in range(7, 13):
            p = P(e, ly[0])
            ticks.add(Line(p, p + 0.08 * DOWN, color=MUTED), pow10(e).next_to(p, DOWN, buff=0.12))
        for v in [2.0, 2.5, 3.0, 3.5, 4.0]:
            p = P(nx[0], v)
            ticks.add(Line(p, p + 0.08 * LEFT, color=MUTED), Text(f"{v:.1f}", font_size=20, color=MUTED).next_to(p, LEFT, buff=0.1))
        xlab = Text("model size N (parameters), budget fixed: D = C/6N", font_size=20).next_to(axes, DOWN, buff=0.55)
        ylab = Text("loss L", font_size=20).rotate(PI / 2).move_to([X0 - 0.75, Y0 + H / 2, 0])

        def valley(lc):
            C = 10.0 ** lc
            L = loss(10 ** lns, C / (6 * 10 ** lns))
            keep = L <= ly[1]
            return lns[keep], L[keep]

        def vcurve(i):
            x, L = valley(budgets[i])
            g = polyline([P(a, b) for a, b in zip(x, L)], cols[i], 4)
            lab = MathTex(rf"C{{=}}10^{{{budgets[i]}}}", font_size=26, color=cols[i])
            return g, lab

        def minimum(i):
            C = 10.0 ** budgets[i]
            N = n_opt(C)
            return np.log10(N), loss_opt(C)

        cap = caption("1. fix the compute budget C = 10²¹ FLOPs and vary the model size N")
        self.play(FadeIn(cap), Create(axes), FadeIn(ticks), FadeIn(xlab), FadeIn(ylab))
        i0 = budgets.index(21)
        g, lab = vcurve(i0)
        lab.next_to(g.get_end(), RIGHT, buff=0.1)
        self.play(Create(g), FadeIn(lab), run_time=1.5)
        x_ok = valley(21)[0]
        small = Text("← too small: underfits", font_size=20, color=MUTED).move_to(P(7.05, 1.95), aligned_edge=LEFT)
        big = Text("too large: too few tokens →", font_size=20, color=MUTED).move_to(P(12.0, 1.95), aligned_edge=RIGHT)
        cap = swap_caption(self, cap, "2. a valley: small models underfit, large models see too few tokens")
        self.play(FadeIn(small), FadeIn(big))
        self.wait(1.2)
        mn, mL = minimum(i0)
        d0 = Dot(P(mn, mL), color=RED_, radius=0.09)
        Nn = n_opt(1e21)
        mlab = MathTex(rf"N_{{\text{{opt}}}} = {sci(Nn)}", font_size=28, color=RED_).next_to(d0, UP, buff=0.15)
        cap = swap_caption(self, cap, f"3. the minimum: {human(Nn)} parameters on {human(1e21 / 6 / Nn)} tokens is the best use of 10²¹ FLOPs",
                           size=26)
        self.play(FadeIn(d0, scale=2), FadeIn(mlab))
        self.wait(1.5)

        cap = swap_caption(self, cap, "4. repeat for budgets 10¹⁹ … 10²⁴: bigger budgets, lower and wider-right valleys")
        self.play(FadeOut(small), FadeOut(big), FadeOut(mlab))
        curves, labs, dots = VGroup(g), VGroup(lab), VGroup()
        for i in range(len(budgets)):
            if i == i0:
                continue
            gi, li = vcurve(i)
            li.next_to(gi.get_end(), RIGHT, buff=0.1)
            curves.add(gi); labs.add(li)
            self.play(Create(gi), FadeIn(li), run_time=0.55)
        for i in range(len(budgets)):
            dots.add(Dot(P(*minimum(i)), color=RED_, radius=0.08))
        self.play(FadeOut(d0), LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.15), run_time=1.2)
        self.wait(0.8)

        # right panel: N_opt and D_opt vs C (log-log)
        RX0, RY0, RW, RH = 2.55, -2.0, 4.0, 3.6
        cxr, cyr = (18.6, 24.4), (8.0, 13.0)
        Q = lambda lc, lv: np.array([RX0 + (lc - cxr[0]) / (cxr[1] - cxr[0]) * RW, RY0 + (lv - cyr[0]) / (cyr[1] - cyr[0]) * RH, 0])
        rax = frame_axes(RX0, RY0, RW, RH)
        for e in budgets:
            p = Q(e, cyr[0])
            rax.add(Line(p, p + 0.08 * DOWN, color=MUTED), pow10(e, 22).next_to(p, DOWN, buff=0.1))
        for e in range(8, 14):
            p = Q(cxr[0], e)
            rax.add(Line(p, p + 0.08 * LEFT, color=MUTED), pow10(e, 22).next_to(p, LEFT, buff=0.08))
        rax.add(Text("budget C (FLOPs)", font_size=20).next_to(Q(21.5, cyr[0]), DOWN, buff=0.45))
        lcr = np.linspace(cxr[0], cxr[1], 50)
        nline = polyline([Q(x, np.log10(n_opt(10 ** x))) for x in lcr], RED_, 4)
        dline = polyline([Q(x, np.log10(10 ** x / 6 / n_opt(10 ** x))) for x in lcr], TEAL, 4)
        ndots = VGroup(*[Dot(Q(e, np.log10(n_opt(10.0 ** e))), color=RED_, radius=0.07) for e in budgets])
        ddots = VGroup(*[Dot(Q(e, np.log10(10.0 ** e / 6 / n_opt(10.0 ** e))), color=TEAL, radius=0.07) for e in budgets])
        nlab = MathTex(rf"N_{{\text{{opt}}}} \propto C^{{{EXP_N:.3f}}}", font_size=26, color=RED_).move_to(Q(22.9, 9.25))
        dlab = MathTex(rf"D_{{\text{{opt}}}} \propto C^{{{1 - EXP_N:.3f}}}", font_size=26, color=TEAL).move_to(Q(20.4, 12.1))
        rhdr = Text("optimal size and tokens", font_size=20).next_to(rax, UP, buff=0.25)

        cap = swap_caption(self, cap, f"5. the minima lie on a line in log-log: N_opt ∝ C^{EXP_N:.3f}  (= β/(α+β))", color=RED_)
        self.play(Create(rax), FadeIn(rhdr))
        self.play(*[TransformFromCopy(dots[i], ndots[i]) for i in range(len(budgets))], run_time=1.2)
        self.play(Create(nline), FadeIn(nlab))
        self.wait(1)
        r_lo = 10.0 ** budgets[0] / 6 / n_opt(10.0 ** budgets[0]) ** 2
        r_hi = 10.0 ** budgets[-1] / 6 / n_opt(10.0 ** budgets[-1]) ** 2
        cap = swap_caption(self, cap, f"6. tokens grow faster: {r_lo:.0f} tokens/param at 10¹⁹ → {r_hi:.0f} at 10²⁴ in this fit",
                           color=TEAL)
        self.play(FadeIn(ddots), Create(dline), FadeIn(dlab))
        b0 = Q(budgets[0], np.log10(n_opt(10.0 ** budgets[0])))
        b1 = Q(budgets[0], np.log10(10.0 ** budgets[0] / 6 / n_opt(10.0 ** budgets[0])))
        e0 = Q(budgets[-1], np.log10(n_opt(10.0 ** budgets[-1])))
        e1 = Q(budgets[-1], np.log10(10.0 ** budgets[-1] / 6 / n_opt(10.0 ** budgets[-1])))
        br0 = VGroup(DoubleArrow(b0, b1, buff=0.1, color=MUTED, stroke_width=3, tip_length=0.14),
                     Text(f"{r_lo:.0f}×", font_size=18, color=INK).next_to((b0 + b1) / 2, RIGHT, buff=0.08))
        br1 = VGroup(DoubleArrow(e0, e1, buff=0.1, color=MUTED, stroke_width=3, tip_length=0.14),
                     Text(f"{r_hi:.0f}×", font_size=18, color=INK).next_to((e0 + e1) / 2, LEFT, buff=0.08))
        self.play(FadeIn(br0), FadeIn(br1))
        self.wait(2)

        # derivation
        left = VGroup(axes, ticks, xlab, ylab, curves, labs, dots)
        right = VGroup(rax, rhdr, nline, dline, ndots, ddots, nlab, dlab, br0, br1)
        cap = swap_caption(self, cap, "7. derive it: substitute D = C/6N, set ∂L/∂N = 0")
        self.play(FadeOut(left), right.animate.scale(0.95).move_to([-3.4, -0.2, 0]), run_time=1.2)
        dx = 3.0
        eqs = VGroup(
            MathTex(r"L(N) = E + A\,N^{-\alpha} + B\left(\tfrac{C}{6}\right)^{-\beta} N^{\beta}", font_size=32),
            MathTex(r"\frac{\partial L}{\partial N} = -\alpha A\,N^{-\alpha-1} + \beta B\left(\tfrac{C}{6}\right)^{-\beta} N^{\beta-1} = 0",
                    font_size=32),
            MathTex(r"N_{\text{opt}} = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}"
                    r"\left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}", font_size=32),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT).move_to([dx, 0.35, 0])
        for eq in eqs:
            self.play(Write(eq), run_time=1.3)
            self.wait(0.9)
        num = MathTex(rf"= {G_:.2f}\,\left(\tfrac{{C}}{{6}}\right)^{{{EXP_N:.3f}}}", font_size=32, color=RED_)
        num.next_to(eqs[2], DOWN, buff=0.35, aligned_edge=LEFT).shift(1.6 * RIGHT)
        box = SurroundingRectangle(VGroup(eqs[2], num), color=RED_, buff=0.18, corner_radius=0.08)
        self.play(FadeIn(num), Create(box))
        cap = swap_caption(self, cap, f"exponent β/(α+β) = 0.28/0.62 = {EXP_N:.3f}: 10× compute → {10 ** EXP_N:.1f}× params, "
                                      f"{10 ** (1 - EXP_N):.1f}× tokens", color=PURPLE_, size=26)
        self.wait(3)


class TrainVsInferenceOptimal(Scene):
    """Fixed target loss (the fit's prediction for Llama 3 8B on 15T tokens): smaller models need more tokens and
    more training FLOPs but are cheaper per generated token; lifetime cost 6ND + 2NT has its optimum moving to
    smaller, over-trained models as the inference volume T grows. Target: w05 scaling-laws slide (beyond Chinchilla)."""

    def construct(self):
        title(self, "Training-optimal vs inference-optimal", SRC + "; same target loss for every model")
        N8, D8 = 8.03e9, 15e12
        Lt = loss(N8, D8)
        Nmin = (A_ / (Lt - E_)) ** (1 / AL)

        def D_of(N):
            return (B_ / (Lt - E_ - A_ / N ** AL)) ** (1 / BE)

        X0, Y0, W, H = -5.6, -2.05, 6.6, 4.1
        nx, cy = (9.3, 11.3), (23.0, 25.0)
        P = lambda ln, lc: np.array([X0 + (ln - nx[0]) / (nx[1] - nx[0]) * W, Y0 + (lc - cy[0]) / (cy[1] - cy[0]) * H, 0])
        lns = np.linspace(max(nx[0], np.log10(Nmin) + 1e-3), nx[1], 600)
        Nv = 10 ** lns
        Dv = D_of(Nv)
        train = 6 * Nv * Dv

        def cost_curve(T, color, width=4, opacity=1.0):
            tot = train + 2 * Nv * T
            keep = (np.log10(tot) <= cy[1])
            return polyline([P(a, np.log10(b)) for a, b in zip(lns[keep], tot[keep])], color, width, opacity)

        def argmin_N(T):
            # fine 1-D search on the analytic cost
            ln = np.linspace(np.log10(Nmin) + 1e-4, 12, 20001)
            N = 10 ** ln
            tot = 6 * N * D_of(N) + 2 * N * T
            j = np.argmin(tot)
            return N[j], tot[j]

        axes = frame_axes(X0, Y0, W, H)
        ticks = VGroup()
        for v, s in [(3e9, "3B"), (1e10, "10B"), (3e10, "30B"), (1e11, "100B"), (2e11, "200B")]:
            p = P(np.log10(v), cy[0])
            ticks.add(Line(p, p + 0.08 * DOWN, color=MUTED), Text(s, font_size=20, color=MUTED).next_to(p, DOWN, buff=0.12))
        for e in range(23, 26):
            p = P(nx[0], e)
            ticks.add(Line(p, p + 0.08 * LEFT, color=MUTED), pow10(e).next_to(p, LEFT, buff=0.1))
        xlab = Text("model size N (parameters)", font_size=20).next_to(axes, DOWN, buff=0.55)
        ylab = Text("FLOPs over the model's life", font_size=20).rotate(PI / 2).move_to([X0 - 0.95, Y0 + H / 2, 0])

        rx = 1.6
        tgt = MathTex(rf"L^* = {Lt:.3f}", font_size=36).move_to([rx, 1.95, 0], aligned_edge=LEFT)
        tgt2 = Text("the fit's loss for Llama 3 8B on 15T tokens", font_size=19, color=MUTED).next_to(tgt, DOWN, buff=0.12, aligned_edge=LEFT)
        dform = MathTex(r"D(N)\ \text{solves}\ \ E + \tfrac{A}{N^{\alpha}} + \tfrac{B}{D^{\beta}} = L^*", font_size=28).next_to(tgt2, DOWN, buff=0.3, aligned_edge=LEFT)

        cap = caption(f"1. fix a target loss L* = {Lt:.3f}; each N trains on just enough tokens to reach it")
        self.play(FadeIn(cap), Create(axes), FadeIn(ticks), FadeIn(xlab), FadeIn(ylab), FadeIn(tgt), FadeIn(tgt2), FadeIn(dform))
        tr = cost_curve(0, INK, 5)
        trlab = MathTex(r"\text{training } 6ND", font_size=24).move_to(P(10.95, 23.85))
        self.play(Create(tr), FadeIn(trlab), run_time=1.6)
        wall = DashedLine(P(np.log10(Nmin), cy[0]), P(np.log10(Nmin), cy[1]), color=MUTED, stroke_width=2)
        wlab = Text(f"below {human(Nmin)} params\nL* is unreachable", font_size=18, color=MUTED, line_spacing=0.8)
        wlab.next_to(P(np.log10(Nmin), cy[0] + 0.25), RIGHT, buff=0.12)
        self.play(Create(wall), FadeIn(wlab))
        self.wait(0.8)

        Nc, Cc = argmin_N(0)
        Dc = D_of(Nc)
        dc = Dot(P(np.log10(Nc), np.log10(Cc)), color=GREEN_, radius=0.1)
        dcl = Text("Chinchilla-optimal", font_size=18, color=GREEN_, weight=BOLD).next_to(dc, DOWN, buff=0.15).shift(0.4 * RIGHT)
        Ct8 = 6 * N8 * D8
        d8 = Dot(P(np.log10(N8), np.log10(Ct8)), color=ORANGE_, radius=0.1)
        d8l = Text("Llama 3 8B", font_size=18, color=ORANGE_, weight=BOLD).next_to(d8, RIGHT, buff=0.15)

        def card(color, rows):
            g = VGroup(*[Text(r, font_size=21, color=color, weight=BOLD) if j == 0 else Text(r, font_size=19)
                         for j, r in enumerate(rows)])
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        cc = card(GREEN_, ["Chinchilla-optimal",
                           f"N = {human(Nc)},  D = {human(Dc)}  ({Dc / Nc:.0f} tokens/param)",
                           f"training: {Cc / 1e23:.1f}×10²³ FLOPs"]).move_to([rx, 0.0, 0], aligned_edge=LEFT)
        c8 = card(ORANGE_, ["Llama 3 8B",
                            f"N = 8.03B,  D = 15T  ({D8 / N8:,.0f} tokens/param)",
                            f"training: {Ct8 / 1e23:.1f}×10²³ FLOPs  ({Ct8 / Cc:.1f}×)"]).next_to(cc, DOWN, buff=0.35, aligned_edge=LEFT)
        cap = swap_caption(self, cap, f"2. the cheapest way to train to L*: {human(Nc)} params on {human(Dc)} tokens")
        self.play(FadeIn(dc, scale=2), FadeIn(dcl), FadeIn(cc))
        self.wait(1.2)
        cap = swap_caption(self, cap, f"3. 8B reaches the same loss with {D8 / Dc:.1f}× the tokens: {Ct8 / Cc:.1f}× the training FLOPs",
                           color=ORANGE_)
        self.play(FadeIn(d8, scale=2), FadeIn(d8l), FadeIn(c8))
        self.wait(1.5)
        inf = Text(f"serving: 2N FLOPs/token → {Nc / N8:.1f}× cheaper", font_size=20, color=ORANGE_).next_to(c8, DOWN, buff=0.12, aligned_edge=LEFT)
        cap = swap_caption(self, cap, f"4. but each generated token costs 2N FLOPs: 8B serves {Nc / N8:.1f}× cheaper than {human(Nc)}")
        self.play(FadeIn(inf))
        self.wait(1.5)

        # lifetime cost with T tokens served
        cap = swap_caption(self, cap, "5. lifetime cost = 6ND + 2N·T, for T tokens generated over the model's life")
        self.play(FadeOut(dcl), FadeOut(d8l), FadeOut(tgt2), FadeOut(dform),
                  VGroup(cc, c8, inf).animate.shift(1.0 * UP))
        lT = ValueTracker(11.0)
        tot = always_redraw(lambda: cost_curve(10 ** lT.get_value(), PURPLE_, 5))
        mdot = always_redraw(lambda: Dot(P(np.log10(argmin_N(10 ** lT.get_value())[0]),
                                           np.log10(argmin_N(10 ** lT.get_value())[1])), color=PURPLE_, radius=0.1))

        def readout():
            T = 10 ** lT.get_value()
            N, _ = argmin_N(T)
            return VGroup(Text(f"T = {T / 1e12:,.1f}T tokens served", font_size=21, color=PURPLE_, weight=BOLD),
                          Text(f"cheapest: N = {human(N)},  {D_of(N) / N:,.0f} tokens/param", font_size=20, color=PURPLE_)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([rx, -1.75, 0], aligned_edge=LEFT)
        ro = always_redraw(readout)
        self.play(FadeIn(tot), FadeIn(mdot), FadeIn(ro))
        ghosts = VGroup()
        for e in [12, 13, 14]:
            self.play(lT.animate.set_value(e), run_time=1.6, rate_func=linear)
            gh = cost_curve(10.0 ** e, PURPLE_, 3, 0.35)
            Ng, Cg = argmin_N(10.0 ** e)
            gd = Dot(P(np.log10(Ng), np.log10(Cg)), color=PURPLE_, radius=0.07).set_opacity(0.5)
            gl = MathTex(rf"T{{=}}10^{{{e}}}", font_size=24, color=PURPLE_).next_to(gh.get_end(), LEFT, buff=0.1).shift(0.15 * DOWN)
            ghosts.add(VGroup(gh, gd, gl))
            self.add(ghosts[-1])
            if e == 12:
                cap = swap_caption(self, cap, "6. the more the model is used, the further the optimum slides to smaller models",
                                   color=PURPLE_)
        self.wait(0.5)

        # T at which 8B is the optimum: d/dN [6 N D(N)] + 2T = 0
        h = 1e-4
        dND = (N8 * (1 + h) * D_of(N8 * (1 + h)) - N8 * (1 - h) * D_of(N8 * (1 - h))) / (2 * N8 * h)
        T8 = -3 * dND
        cap = swap_caption(self, cap, f"7. Llama 3 8B is the cost-optimal size if it serves ≈{T8 / 1e12:.0f}T tokens in its lifetime",
                           color=ORANGE_)
        self.play(lT.animate.set_value(np.log10(T8)), run_time=1.2)
        ring = Circle(radius=0.2, color=ORANGE_, stroke_width=5).move_to(P(np.log10(N8), np.log10(argmin_N(T8)[1])))
        self.play(Create(ring))
        self.wait(1.5)
        cap = swap_caption(self, cap, "for heavily served models, smaller and over-trained beats Chinchilla-optimal",
                           color=PURPLE_, size=26)
        self.wait(3)
