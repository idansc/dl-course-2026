"""Week 4 (Transformers): positional encoding. Sinusoidal PE as clocks, where position enters the attention score,
and RoPE length extrapolation (position interpolation, NTK / YaRN). Complements RoPERotation and
PermutationEquivariance in w04_transformers.py. Render one scene:
    manim -qh --disable_caching -o SinusoidalClock w04_positions.py SinusoidalClock
"""
from style import *


def div_color(v, scale=1.0):
    """Diverging colour for a signed number: negative → blue, 0 → white, positive → orange."""
    v = float(np.clip(v / scale, -1, 1))
    return interpolate_color(ManimColor(WHITE), ManimColor(ORANGE_ if v > 0 else BLUE_), abs(v))


def div_image(M, scale, up=8):
    """Signed matrix → nearest-neighbour ImageMobject with the same diverging colours as div_color."""
    t = np.clip(np.asarray(M, dtype=float) / scale, -1, 1)[..., None]
    pos = np.array(ManimColor(ORANGE_).to_rgb())
    neg = np.array(ManimColor(BLUE_).to_rgb())
    rgb = 1 - np.abs(t) * (1 - np.where(t > 0, pos, neg))
    rgb = np.repeat(np.repeat((rgb * 255).astype(np.uint8), up, 0), up, 1)
    im = ImageMobject(rgb)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return im


def sin_pe(T, d, base=10000.0):
    """Sinusoidal PE as a d × T matrix: rows 2j = sin(mθ_j), 2j+1 = cos(mθ_j)."""
    th = base ** (-2 * np.arange(d // 2) / d)
    a = np.outer(th, np.arange(T))
    P = np.zeros((d, T))
    P[0::2], P[1::2] = np.sin(a), np.cos(a)
    return P, th


def matrix(M, cell, colfn, mask=None):
    """Grid of squares coloured by colfn(value); masked cells drawn as light grey."""
    g = VGroup()
    n, k = M.shape
    for i in range(n):
        for j in range(k):
            fill = SOFT if (mask is not None and mask[i, j]) else colfn(M[i, j])
            g.add(Square(cell, stroke_color=MUTED, stroke_width=0.8, fill_color=fill, fill_opacity=1)
                  .move_to([j * cell, -i * cell, 0]))
    return g


# ----------------------------------------------------------------------------------------------------------------------
class SinusoidalClock(Scene):
    """Target: p7 71–77 (positional encodings). Sinusoidal PE with d = 16, base 10000 as 8 clock hands, one per
    (sin, cos) pair, turning at θ_j = 10000^(−2j/d): as the position m advances, each hand sits at angle m·θ_j and
    the d × T heatmap fills in column by column (a continuous binary counter). Shifting by k turns every hand by
    k·θ_j, i.e. PE(m+k) = R_k PE(m) (checked in numpy), so PE(m)·PE(n) = Σ cos((m−n)θ_j) depends only on the
    offset: the 64 × 64 similarity matrix is banded and constant along diagonals."""

    def construct(self):
        d, T = 16, 64
        J = d // 2
        PE, th = sin_pe(T, d)
        lam = 2 * np.pi / th
        title(self, "Sinusoidal positions as clocks", "one hand per (sin, cos) pair, at geometric frequencies")

        cw, ch = 0.13, 0.28                       # heatmap cell width / height
        x0, y0 = -4.35, 2.2                       # top-left cell centre
        clock_x, r = -4.85, 0.24

        def row_y(i):
            return y0 - i * ch

        # heatmap (cells start invisible and are revealed column by column)
        cells = [[Rectangle(width=cw, height=ch, stroke_width=0, fill_color=div_color(PE[i, c]), fill_opacity=0)
                  .move_to([x0 + c * cw, row_y(i), 0]) for c in range(T)] for i in range(d)]
        frame = Rectangle(width=T * cw, height=d * ch, stroke_color=MUTED, stroke_width=1.5)
        frame.move_to([x0 + (T - 1) * cw / 2, y0 - (d - 1) * ch / 2, 0])
        seps = VGroup(*[Line([x0 - cw / 2, row_y(2 * j) + ch / 2, 0], [x0 + (T - 0.5) * cw, row_y(2 * j) + ch / 2, 0],
                             stroke_color=MUTED, stroke_width=1) for j in range(1, J)])
        heat = VGroup(*[cells[i][c] for i in range(d) for c in range(T)])
        m = ValueTracker(0.0)

        def reveal(mob):
            mm = m.get_value()
            for i in range(d):
                for c in range(T):
                    cells[i][c].set_fill(opacity=1.0 if c <= mm + 1e-6 else 0.0)

        heat.add_updater(reveal)

        # clocks, one per pair, next to the two rows it produces
        faces = VGroup()
        lam_lab = VGroup()
        for j in range(J):
            cy = (row_y(2 * j) + row_y(2 * j + 1)) / 2
            faces.add(VGroup(Circle(r, stroke_color=INK, stroke_width=2, fill_color=WHITE, fill_opacity=1)
                             .move_to([clock_x, cy, 0]),
                             Line([clock_x, cy + r, 0], [clock_x, cy + r - 0.06, 0], stroke_color=INK, stroke_width=2)))
            lam_lab.add(Text(f"λ = {lam[j]:,.0f}" if lam[j] >= 10 else f"λ = {lam[j]:.1f}", font_size=16, color=MUTED)
                        .move_to([clock_x - r - 0.15, cy, 0], aligned_edge=RIGHT))
        lam_hdr = Text("wavelength", font_size=16, color=MUTED).move_to([clock_x - 0.9, y0 + 0.35, 0])

        def hands():
            g = VGroup()
            mm = m.get_value()
            for j in range(J):
                c = faces[j][0].get_center()
                a = mm * th[j]
                tip = c + 0.85 * r * np.array([np.sin(a), np.cos(a), 0])
                g.add(Line(c, tip, stroke_color=PURPLE_, stroke_width=3.5), Dot(c, radius=0.025, color=INK))
            return g

        hnd = always_redraw(hands)

        def marker():
            c = min(T - 1, int(np.floor(m.get_value() + 1e-6)))
            return Rectangle(width=cw, height=d * ch + 0.08, stroke_color=PURPLE_, stroke_width=3).move_to(
                [x0 + c * cw, frame.get_center()[1], 0])

        mark = always_redraw(marker)
        counter = always_redraw(lambda: Text(f"position m = {int(np.floor(m.get_value() + 1e-6))}", font_size=28,
                                             color=PURPLE_).move_to([5.55, 0.9, 0]))
        ticks = VGroup(*[Text(str(c), font_size=18, color=MUTED).move_to([x0 + c * cw, row_y(d - 1) - 0.35, 0])
                         for c in (0, 16, 32, 48, 63)])
        xlab = Text("position m →", font_size=20, color=MUTED).move_to([x0 + 31.5 * cw, row_y(d - 1) - 0.65, 0])
        rowlab = VGroup(Text("rows 2j, 2j+1", font_size=18, color=MUTED),
                        Text("= sin, cos of hand j", font_size=18, color=MUTED)).arrange(DOWN, buff=0.08)
        rowlab.move_to([5.55, -1.6, 0])
        formula = VGroup(MathTex(r"PE_{2j}(m) = \sin(m\theta_j)", font_size=30),
                         MathTex(r"PE_{2j+1}(m) = \cos(m\theta_j)", font_size=30),
                         MathTex(r"\theta_j = 10000^{-2j/d},\ d = 16", font_size=30)).arrange(DOWN, buff=0.18,
                                                                                            aligned_edge=LEFT)
        formula.scale_to_fit_width(2.75).move_to([5.55, 2.0, 0])

        cap = caption("each position m is a setting of 8 clock hands: hand j sits at angle m·θ, its own rate θ")
        self.play(FadeIn(cap), FadeIn(faces), FadeIn(lam_lab), FadeIn(formula), run_time=1.2)
        self.add(hnd)
        self.play(Create(frame), FadeIn(seps), FadeIn(ticks), FadeIn(xlab), FadeIn(rowlab), FadeIn(counter),
                  run_time=0.8)
        self.add(heat, mark)
        self.wait(0.6)
        cap = swap_caption(self, cap, "advance m: the fast hand spins every step, the slow ones barely move")
        self.play(m.animate.set_value(20), run_time=4.0, rate_func=linear)
        cap = swap_caption(self, cap, "like a binary counter in continuous form: slow hands mark coarse position")
        self.play(m.animate.set_value(T - 1), run_time=5.0, rate_func=linear)
        self.wait(1.2)

        # --- shift by k -------------------------------------------------------------------------------------------
        heat.clear_updaters()
        self.play(*[FadeOut(x) for x in (heat, frame, seps, ticks, xlab, rowlab, counter, mark, hnd, faces, lam_lab,
                                         formula)], run_time=0.8)
        m0, k_end = 10, 5
        R = 0.6
        cxs = [-5.6 + 1.6 * j for j in range(J)]
        cy = 0.95
        big = VGroup(*[VGroup(Circle(R, stroke_color=INK, stroke_width=2.5).move_to([x, cy, 0]),
                              Line([x, cy + R, 0], [x, cy + R - 0.1, 0], stroke_color=INK, stroke_width=2.5))
                       for x in cxs])
        jl = VGroup(*[MathTex(rf"j={j}", font_size=26, color=MUTED).move_to([x, cy + R + 0.3, 0])
                      for j, x in enumerate(cxs)])
        k = ValueTracker(0.0)

        def big_hands():
            g = VGroup()
            kk = k.get_value()
            for j, x in enumerate(cxs):
                c = np.array([x, cy, 0])
                a0, a1 = m0 * th[j], (m0 + kk) * th[j]
                g.add(Line(c, c + 0.88 * R * np.array([np.sin(a0), np.cos(a0), 0]), stroke_color=PURPLE_,
                           stroke_width=4, stroke_opacity=0.3))
                if kk * th[j] > 0.02:
                    g.add(Arc(radius=0.42 * R, start_angle=PI / 2 - a0, angle=-kk * th[j], arc_center=c,
                              color=GREEN_, stroke_width=4))
                g.add(Line(c, c + 0.88 * R * np.array([np.sin(a1), np.cos(a1), 0]), stroke_color=PURPLE_,
                           stroke_width=5), Dot(c, radius=0.04, color=INK))
                g.add(Text(f"+{kk * th[j]:.3f} rad" if kk * th[j] < 0.995 else f"+{kk * th[j]:.2f} rad",
                           font_size=18, color=GREEN_).move_to([x, cy - R - 0.3, 0]))
            return g

        bh = always_redraw(big_hands)
        hdr = Text(f"PE(m) at m = {m0}", font_size=26, color=PURPLE_).move_to([0, 2.2, 0])
        cap = swap_caption(self, cap, f"zoom in on one position, m = {m0}")
        self.play(FadeIn(big), FadeIn(jl), FadeIn(bh), FadeIn(hdr))
        self.wait(0.6)
        hdr2 = Text(f"PE(m + k) at m = {m0}, k = {k_end}", font_size=26, color=PURPLE_).move_to(hdr)
        cap = swap_caption(self, cap, f"move by k = {k_end}: every hand turns by {k_end}·θ at its own rate, whatever m is")
        self.play(k.animate.set_value(k_end), Transform(hdr, hdr2), run_time=2.5)
        # numpy check: PE(m+k) = R_k PE(m) for every m
        P2, _ = sin_pe(T + k_end, d)
        err = 0.0
        for mm in range(T):
            for j in range(J):
                b = k_end * th[j]
                Rk = np.array([[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]])   # turns (sin, cos) clockwise
                err = max(err, np.abs(P2[2 * j:2 * j + 2, mm + k_end] - Rk @ P2[2 * j:2 * j + 2, mm]).max())
        eq = MathTex(r"PE(m+k) = R_k\,PE(m),\qquad R_k = \mathrm{diag}\big(R(k\theta_0),\dots,R(k\theta_7)\big)",
                     font_size=34).move_to([0, -0.75, 0])
        e_m, e_x = f"{err:.0e}".split("e")
        chk = MathTex(rf"\text{{numpy, all }} m < {T}:\ \max|PE(m{{+}}k) - R_k\,PE(m)| = {e_m}\times 10^{{{int(e_x)}}}",
                      font_size=28, color=MUTED)
        chk.next_to(eq, DOWN, buff=0.25)
        cap = swap_caption(self, cap, "a shift is a fixed rotation per pair, independent of m")
        self.play(FadeIn(eq, shift=0.1 * UP))
        self.play(FadeIn(chk))
        self.wait(1.2)
        eq2 = MathTex(r"PE(m)^\top PE(n) = \sum_j \cos\big((m-n)\,\theta_j\big)", font_size=36, color=PURPLE_)
        eq2.next_to(chk, DOWN, buff=0.35)
        cap = swap_caption(self, cap, "so the dot product of two positions depends only on their offset m − n",
                           color=PURPLE_)
        self.play(FadeIn(eq2, shift=0.1 * UP))
        self.wait(1.8)

        # --- similarity matrix --------------------------------------------------------------------------------------
        S = PE.T @ PE
        assert np.allclose(S[5, 15], S[30, 40])
        Sn = (S - S.min()) / (S.max() - S.min())
        rgb = np.array([[ManimColor(heat_color(v)).to_rgb() for v in row] for row in Sn])
        rgb = np.repeat(np.repeat((rgb * 255).astype(np.uint8), 8, 0), 8, 1)
        img = ImageMobject(rgb)
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        img.set(height=4.3).move_to([-3.3, -0.35, 0])
        cbar = VGroup(Text(f"white = {S.min():.1f}", font_size=16, color=MUTED),
                      Text(f"purple = {S.max():.0f}", font_size=16, color=PURPLE_)).arrange(RIGHT, buff=0.4)
        cbar.next_to(img, DOWN, buff=0.12)
        bx = SurroundingRectangle(img, buff=0, color=MUTED, stroke_width=1.5)
        ml = Text("position m", font_size=20, color=MUTED).rotate(PI / 2).next_to(img, LEFT, buff=0.2)
        nl = Text("position n", font_size=20, color=MUTED).next_to(img, UP, buff=0.15)
        ax = Axes(x_range=[-63, 63, 21], y_range=[0, 8, 2], x_length=5.6, y_length=3.2,
                  axis_config={"stroke_color": MUTED, "include_ticks": True}, tips=False).move_to([3.6, -0.2, 0])
        ks = np.arange(-63, 64)
        prof = np.array([np.cos(kk * th).sum() for kk in ks])
        curve = ax.plot_line_graph(ks, prof, add_vertex_dots=False, line_color=PURPLE_, stroke_width=4)
        axl = VGroup(Text("offset m − n", font_size=20, color=MUTED).next_to(ax, DOWN, buff=0.15),
                     *[Text(str(v), font_size=16, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
                       for v in (-63, 63)],
                     *[Text(str(v), font_size=16, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1)
                       for v in (4, 8)])
        ylab = MathTex(r"\textstyle\sum_j\cos((m{-}n)\theta_j)", font_size=26, color=PURPLE_).next_to(ax, UP, buff=0.3)
        cap = swap_caption(self, cap, f"PE(m)·PE(n) for all {T} × {T} pairs of positions: a band around the diagonal")
        self.play(FadeOut(VGroup(big, jl, hdr, eq, chk, eq2)), FadeOut(bh), run_time=0.6)
        self.play(FadeIn(img), Create(bx), FadeIn(ml), FadeIn(nl), FadeIn(cbar), run_time=1.0)
        self.wait(1.0)
        off = 12
        p, s = img.get_corner(UL), img.height / T
        diag = Line(p + np.array([(off + 0.5) * s, -0.5 * s, 0]), p + np.array([(T - 0.5) * s, -(T - off - 0.5) * s, 0]),
                    color=GREEN_, stroke_width=5)
        dot = Dot(ax.c2p(-off, prof[ks == -off][0]), color=GREEN_, radius=0.08)
        dl = Text(f"offset {off}: {prof[ks == -off][0]:.2f} on every cell", font_size=18, color=GREEN_)
        dl.next_to(axl[0], DOWN, buff=0.15)
        cap = swap_caption(self, cap, "constant along every diagonal: the same profile of the offset, repeated")
        self.play(Create(ax), FadeIn(axl), FadeIn(ylab), Create(curve), run_time=1.5)
        self.play(Create(diag), FadeIn(dot), FadeIn(dl))
        self.wait(2.0)
        cap = swap_caption(self, cap, "peak at offset 0; with only d = 16 the falloff with distance is not monotone",
                           size=26)
        self.play(Indicate(curve, color=PURPLE_, scale_factor=1.03), run_time=1.0)
        self.wait(1.5)


# ----------------------------------------------------------------------------------------------------------------------
class WherePositionEnters(Scene):
    """Target: p7 71–77 / L8b. The attention pipeline x → W_Q, W_K → q·k → + bias / mask → softmax, and the four places
    position can enter it: (a) absolute x + p before the projections (score = content + mixed + position-position
    terms; the position-position term shown for sinusoidal p and random W, d = 16), (b) a relative bias b(i−j) on the
    score matrix (T5 buckets / ALiBi −m(i−j)), (c) rotary R_i q · R_j k = q·R_{j−i}k (Toeplitz; see RoPERotation),
    (d) NoPE: only the causal mask. Ends with ALiBi on 4 heads (slopes 1/2, 1/8, 1/32, 1/128 of an 8-head model):
    the linear penalty and the resulting attention, fading off the diagonal at head-specific rates."""

    def construct(self):
        title(self, "Where does position enter attention?")
        colors = {"a": BLUE_, "b": ORANGE_, "c": GREEN_, "d": RED_}

        # pipeline
        py = 2.15
        xs = [-5.4, -2.75, -0.1, 2.75, 5.5]
        labels = [MathTex(r"x_i,\ x_j", font_size=32), MathTex(r"W_Q,\ W_K", font_size=32),
                  MathTex(r"q_i^\top k_j", font_size=32), Text("+ bias, mask", font_size=22),
                  Text("softmax", font_size=22)]
        boxes = VGroup()
        for x, l in zip(xs, labels):
            rr = RoundedRectangle(corner_radius=0.08, width=1.9, height=0.62, stroke_color=INK, stroke_width=2,
                                  fill_color=SOFT, fill_opacity=1).move_to([x, py, 0])
            boxes.add(VGroup(rr, l.move_to(rr)))
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.06, color=INK, stroke_width=3,
                                max_tip_length_to_length_ratio=0.25) for i in range(4)])

        def tag(letter, tex, x, target_x):
            t = VGroup(Text(f"({letter})", font_size=22, color=colors[letter], weight=BOLD),
                       MathTex(tex, font_size=30, color=colors[letter])).arrange(RIGHT, buff=0.1)
            t.move_to([x, py - 0.8, 0])
            a = Arrow([target_x, py - 0.6, 0], [target_x, py - 0.33, 0], buff=0, color=colors[letter],
                      stroke_width=4, max_tip_length_to_length_ratio=0.35)
            return VGroup(t, a)

        tags = {"a": tag("a", r"+\,p_i", xs[0], xs[0]),
                "c": tag("c", r"R_i q,\ R_j k", (xs[1] + xs[2]) / 2, (xs[1] + xs[2]) / 2),
                "b": tag("b", r"+\,b(i-j)", xs[3] - 0.55, xs[3] - 0.55),
                "d": tag("d", r"\text{mask}", xs[3] + 0.95, xs[3] + 0.55)}
        tags["d"][0].move_to([xs[3] + 1.25, py - 0.8, 0])

        cap = caption("attention: a score for every query i and key j, then mask and softmax")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.15), run_time=1.2)
        self.play(*[GrowArrow(a) for a in arrows], run_time=0.6)
        self.wait(0.6)

        # panels
        T, d = 8, 16
        rng = np.random.default_rng(3)
        P, th = sin_pe(T, d)
        P = P.T                                                       # T × d
        Wq, Wk = rng.normal(size=(d, d)) / np.sqrt(d), rng.normal(size=(d, d)) / np.sqrt(d)
        Spp = (P @ Wq) @ (P @ Wk).T / np.sqrt(d)
        ii, jj = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
        causal = jj > ii
        Bal = -0.5 * (ii - jj)
        q = rng.normal(size=d)

        def rope(v, p):
            a = p * th
            x, y = v[0::2], v[1::2]
            o = np.empty_like(v)
            o[0::2], o[1::2] = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
            return o

        Srot = np.array([[rope(q, i) @ rope(q, j) for j in range(T)] for i in range(T)]) / np.sqrt(d)
        assert np.allclose(Srot[1, 4], Srot[3, 6])
        cell = 0.27
        pxs = {"a": -5.25, "b": -1.75, "c": 1.75, "d": 5.25}
        heads = {"a": "absolute", "b": "relative bias", "c": "rotary", "d": "NoPE"}
        forms = {"a": r"(x_i{+}p_i)W_QW_K^\top(x_j{+}p_j)^\top",
                 "b": r"q_i^\top k_j + b(i-j)",
                 "c": r"(R_iq_i)^\top R_jk_j = q_i^\top R_{j-i}k_j",
                 "d": r"q_i^\top k_j,\quad j \le i"}
        mats = {"a": matrix(Spp, cell, lambda v: div_color(v, np.abs(Spp).max())),
                "b": matrix(Bal, cell, lambda v: div_color(v, 3.5), mask=causal),
                "c": matrix(Srot, cell, lambda v: div_color(v, np.abs(Srot).max())),
                "d": matrix(np.zeros((T, T)), cell, lambda v: WHITE, mask=None)}
        for i in range(T):                                           # NoPE: masked cells red, rest plain
            for j in range(T):
                if causal[i, j]:
                    mats["d"][i * T + j].set_fill(RED_, opacity=0.35)
        subs = {"a": "position–position term", "b": "ALiBi: −m(i−j), m = 1/2", "c": "q · R_{j−i} q: Toeplitz",
                "d": "no position signal at all"}
        caps = {"a": "(a) absolute: add a position vector p to each input, before the projections",
                "b": "(b) relative bias: add b(i−j) to the scores (T5: learned buckets, ALiBi: linear)",
                "c": "(c) rotary: rotate q and k by position; the score sees only the offset j − i",
                "d": "(d) NoPE: only the causal mask; order is inferred from what each token can see"}
        panels = {}
        for key in "abcd":
            x = pxs[key]
            h = Text(f"({key}) {heads[key]}", font_size=24, color=colors[key], weight=BOLD).move_to([x, 0.62, 0])
            f = MathTex(forms[key], font_size=28)
            if f.width > 3.2:
                f.scale_to_fit_width(3.2)
            f.move_to([x, 0.1, 0])
            mt = mats[key].move_to([x, -1.45, 0])
            st = Text(subs[key].replace("R_{j−i}", "R(j−i)"), font_size=17, color=MUTED).move_to([x, -2.88, 0])
            panels[key] = VGroup(h, f, mt, st)

        for key in "abcd":
            cap = swap_caption(self, cap, caps[key], color=colors[key], size=26)
            self.play(FadeIn(tags[key]), FadeIn(panels[key][:2]), run_time=0.9)
            self.play(LaggedStart(*[FadeIn(c) for c in panels[key][2]], lag_ratio=0.01), FadeIn(panels[key][3]),
                      run_time=1.0)
            self.wait(1.8)

        # ALiBi heads
        cap = swap_caption(self, cap, "ALiBi: each head gets its own slope m; the penalty grows linearly with distance",
                           color=ORANGE_, size=26)
        self.play(*[FadeOut(panels[k]) for k in "abcd"], *[tags[k].animate.set_opacity(0.25) for k in "acd"],
                  run_time=0.7)
        T2 = 12
        ii, jj = np.meshgrid(np.arange(T2), np.arange(T2), indexing="ij")
        c2 = jj > ii
        slopes = [2 ** -1, 2 ** -3, 2 ** -5, 2 ** -7]
        sl_txt = ["1/2", "1/8", "1/32", "1/128"]
        hx = [-5.1, -1.7, 1.7, 5.1]
        cell2 = 0.2
        hlabs, bmats, amats, reads = VGroup(), [], [], VGroup()
        for h, (sl, x) in enumerate(zip(slopes, hx)):
            B = -sl * (ii - jj)
            A = softmax(np.where(c2, -1e9, B), axis=-1)
            hlabs.add(Text(f"head {2 * h + 1} of 8:  m = {sl_txt[h]}", font_size=22, color=ORANGE_).move_to([x, 0.62, 0]))
            bmats.append(matrix(B, cell2, lambda v: div_color(v, 5.5), mask=c2).move_to([x, -1.0, 0]))
            An = A / A.max(axis=1, keepdims=True)                     # each row scaled to its own max
            amats.append(matrix(An, cell2, lambda v: heat_color(v), mask=c2).move_to([x, -1.0, 0]))
            reads.add(Text(f"8 back vs. self: ×{np.exp(-8 * sl):.3f}", font_size=18, color=INK).move_to([x, -2.45, 0]))
        btag = Text("bias −m(i−j) added to the scores (darker blue = larger penalty)", font_size=18, color=MUTED)
        btag.move_to([0, -2.88, 0])
        atag = Text("softmax(bias) with equal content scores; each row scaled to its max", font_size=18, color=MUTED)
        atag.move_to(btag)
        self.play(FadeIn(hlabs), *[FadeIn(b) for b in bmats], FadeIn(btag), run_time=1.0)
        self.wait(1.6)
        cap = swap_caption(self, cap, "after softmax: steep heads look locally, shallow heads look far back",
                           color=ORANGE_, size=26)
        self.play(*[Transform(b, a) for b, a in zip(bmats, amats)], Transform(btag, atag), run_time=1.4)
        self.play(FadeIn(reads))
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------------
class LengthExtrapolation(Scene):
    """Target: L8b (long context). RoPE with d = 128, base 10000, trained to L = 4096, run at L′ = 16384. The rotation
    angle m·θ_j (mod 2π) vs position for pairs j = 40, 48, 56, 63 with the training range shaded; on the right each
    pair's circle shows the arc of angles seen in training. Pairs with wavelength 2π/θ_j > L (18 of 64) never complete
    a turn inside training, so past L they reach unseen angles (red). Position interpolation (m → m·L/L′) squeezes
    everything back into the seen range. Then wavelengths λ_j for the original base, PI (×4), NTK-aware base
    10000·4^(128/126) and YaRN's ramp (α = 1, β = 32): fast pairs untouched, slow pairs stretched."""

    def construct(self):
        d, base, L, L2 = 128, 10000.0, 4096, 16384
        th_all = base ** (-2 * np.arange(d // 2) / d)
        lam_all = 2 * np.pi / th_all
        n_slow = int((lam_all > L).sum())
        js = [40, 48, 56, 63]
        cols = [BLUE_, GREEN_, ORANGE_, PURPLE_]
        th = th_all[js]
        title(self, "RoPE beyond the training length", f"d = {d}, base 10000, trained to L = {L:,} tokens")

        ax = Axes(x_range=[0, L2, L], y_range=[0, TAU, PI], x_length=7.2, y_length=3.5,
                  axis_config={"stroke_color": MUTED}, tips=False).move_to([-2.4, 0.15, 0])
        shade = Rectangle(width=ax.c2p(L, 0)[0] - ax.c2p(0, 0)[0], height=ax.y_length, stroke_width=0,
                          fill_color=GREEN_, fill_opacity=0.08)
        shade.move_to(ax.c2p(L / 2, PI))
        tr_lab = Text("training range", font_size=18, color=GREEN_).move_to(ax.c2p(L / 2, TAU) + 0.22 * UP)
        xt = VGroup(*[Text(s, font_size=17, color=MUTED).next_to(ax.c2p(v, 0), DOWN, buff=0.12)
                      for v, s in [(0, "0"), (L, "L = 4096"), (2 * L, "8192"), (3 * L, "12288"), (L2, "L′ = 16384")]])
        yt = VGroup(*[MathTex(s, font_size=26, color=MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.12)
                      for v, s in [(0, "0"), (PI, r"\pi"), (TAU, r"2\pi")]])
        xl = Text("position m", font_size=19, color=MUTED).next_to(xt, DOWN, buff=0.1)
        yl = MathTex(r"m\theta_j \bmod 2\pi", font_size=26, color=MUTED).rotate(PI / 2)
        yl.next_to(yt, LEFT, buff=0.15)
        s = ValueTracker(1.0)                     # position scale (1 = plain RoPE, L/L′ = interpolation)
        pos = ValueTracker(0.0)

        def curves():
            g = VGroup()
            ms = np.linspace(0, L2, 2049)
            for t, c in zip(th, cols):
                a = ms * s.get_value() * t
                turn = np.floor(a / TAU)
                for k in np.unique(turn):
                    sel = turn == k
                    if sel.sum() < 2:
                        continue
                    pts = [ax.c2p(mm, aa - k * TAU) for mm, aa in zip(ms[sel], a[sel])]
                    g.add(VMobject(stroke_color=c, stroke_width=4).set_points_as_corners(pts))
            return g

        cv = always_redraw(curves)

        # circles
        cc = [np.array([3.35, 0.95, 0]), np.array([5.75, 0.95, 0]), np.array([3.35, -1.65, 0]),
              np.array([5.75, -1.65, 0])]
        R = 0.66
        circ = VGroup()
        for t, c, col, j in zip(th, cc, cols, js):
            seen = min(L * t, TAU)
            circ.add(VGroup(DashedVMobject(Circle(R, stroke_color=MUTED, stroke_width=2).move_to(c), num_dashes=30),
                            Arc(radius=R, start_angle=0, angle=seen, arc_center=c, color=col, stroke_width=8),
                            Text(f"j = {j},  λ = {2 * np.pi / t:,.0f}", font_size=18, color=col).move_to(c + (R + 0.27) * UP)))

        def is_unseen(t, mm):
            return L * t < TAU and mm * s.get_value() > L + 1e-6

        def live():
            g = VGroup()
            mm = pos.get_value()
            x = ax.c2p(mm, 0)[0]
            g.add(DashedLine([x, ax.c2p(0, 0)[1], 0], [x, ax.c2p(0, TAU)[1], 0], color=INK, stroke_width=2))
            for t, c, col in zip(th, cc, cols):
                a = mm * s.get_value() * t
                dc = RED_ if is_unseen(t, mm) else col
                g.add(Dot(ax.c2p(mm, a % TAU), radius=0.07, color=dc))
                g.add(Line(c, c + R * np.array([np.cos(a), np.sin(a), 0]), color=dc, stroke_width=3),
                      Dot(c + R * np.array([np.cos(a), np.sin(a), 0]), radius=0.08, color=dc))
            g.add(Text(f"position m = {mm:,.0f}", font_size=24, color=INK).move_to([4.55, 2.3, 0]))
            return g

        cap = caption("each pair j of q and k turns by angle m·θ_j at position m (shown mod 2π)")
        self.play(FadeIn(cap), Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), FadeIn(yl), run_time=1.0)
        self.play(FadeIn(shade), FadeIn(tr_lab), FadeIn(cv), run_time=1.0)
        self.wait(1.2)
        cap = swap_caption(self, cap, f"slow pairs (λ > L) never finish a turn in training: {n_slow} of the 64 pairs",
                           size=26)
        self.play(LaggedStart(*[FadeIn(c) for c in circ], lag_ratio=0.2), run_time=1.4)
        self.wait(1.5)
        lv = always_redraw(live)
        cap = swap_caption(self, cap, "run past L: slow pairs reach angles the model never saw (red)", color=RED_,
                           size=26)
        self.add(lv)
        self.play(pos.animate.set_value(L), run_time=2.0, rate_func=linear)
        self.play(pos.animate.set_value(L2), run_time=3.5, rate_func=linear)
        self.wait(1.0)

        pi_lab = MathTex(r"m \;\to\; m\cdot L/L' = m/4", font_size=32, color=GREEN_).move_to([ax.get_x(), -2.6, 0])
        cap = swap_caption(self, cap, "position interpolation: rescale positions by L/L′ = 1/4", color=GREEN_)
        self.play(pos.animate.set_value(0), run_time=0.8)
        self.play(FadeIn(pi_lab), s.animate.set_value(L / L2), run_time=2.5)
        cap = swap_caption(self, cap, "every angle up to L′ now falls inside the seen arcs", color=GREEN_)
        self.play(pos.animate.set_value(L2), run_time=3.5, rate_func=linear)
        self.wait(0.6)
        cap = swap_caption(self, cap, "price: neighbours are 4× closer in angle, also in the fast pairs", size=26)
        self.wait(1.8)

        # wavelength view: PI vs NTK-aware vs YaRN
        sc = L2 / L
        base_ntk = base * sc ** (d / (d - 2))
        r = L / lam_all
        gam = np.clip((r - 1) / (32 - 1), 0, 1)
        th_yarn = (1 - gam) * th_all / sc + gam * th_all
        lam_sets = [(lam_all, INK, "original, base 10000", False),
                    (lam_all * sc, GREEN_, "PI: every λ × 4", True),
                    (2 * np.pi / base_ntk ** (-2 * np.arange(d // 2) / d), ORANGE_,
                     f"NTK-aware: base {base_ntk:,.0f}", False),
                    (2 * np.pi / th_yarn, PURPLE_, "YaRN: ramp, α = 1, β = 32", False)]
        self.play(*[FadeOut(m) for m in (ax, xt, yt, xl, yl, shade, tr_lab, cv, lv, circ, pi_lab)], run_time=0.8)
        ax2 = Axes(x_range=[0, 63, 8], y_range=[0, 6, 1], x_length=8.2, y_length=4.2,
                   axis_config={"stroke_color": MUTED}, tips=False).move_to([-1.6, -0.05, 0])
        xt2 = VGroup(*[Text(str(v), font_size=17, color=MUTED).next_to(ax2.c2p(v, 0), DOWN, buff=0.12)
                       for v in (0, 16, 32, 48, 63)])
        yt2 = VGroup(*[MathTex(f"10^{v}", font_size=26, color=MUTED).next_to(ax2.c2p(0, v), LEFT, buff=0.12)
                       for v in range(0, 7, 2)])
        xl2 = Text("pair index j  (fast → slow)", font_size=19, color=MUTED).next_to(xt2, DOWN, buff=0.1)
        yl2 = Text("wavelength λ = 2π/θ (log scale)", font_size=19, color=MUTED).next_to(ax2, UP, buff=0.12).align_to(ax2, LEFT).shift(0.3 * RIGHT)
        hl = VGroup()
        for v, s_, c in [(L, "L = 4096", GREEN_), (L2, "L′ = 16384", RED_)]:
            y = np.log10(v)
            hl.add(DashedLine(ax2.c2p(0, y), ax2.c2p(63, y), color=c, stroke_width=2),
                   Text(s_, font_size=17, color=c).next_to(ax2.c2p(0, y), UR, buff=0.06).shift(0.1 * RIGHT))
        jj = np.arange(d // 2)

        def lam_line(lam, c, dashed):
            ln = ax2.plot_line_graph(jj, np.log10(lam), add_vertex_dots=False, line_color=c, stroke_width=4)
            ln = ln["line_graph"]
            return DashedVMobject(ln, num_dashes=50) if dashed else ln

        lines = [lam_line(l, c, dsh) for l, c, _, dsh in lam_sets]
        leg = VGroup()
        for _, c, t, _ in lam_sets:
            leg.add(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=c, stroke_width=5), Text(t, font_size=19, color=c))
                    .arrange(RIGHT, buff=0.15))
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([4.75, 0.6, 0])
        cap = swap_caption(self, cap, "the same story in wavelengths: λ grows geometrically with the pair index j", size=26)
        self.play(Create(ax2), FadeIn(xt2), FadeIn(yt2), FadeIn(xl2), FadeIn(yl2), FadeIn(hl), run_time=1.0)
        self.play(Create(lines[0]), FadeIn(leg[0]), run_time=1.0)
        self.wait(0.4)
        cap = swap_caption(self, cap, "PI stretches every wavelength by 4, the fast ones too", color=GREEN_, size=26)
        self.play(Create(lines[1]), FadeIn(leg[1]), run_time=1.0)
        self.wait(1.0)
        cap = swap_caption(self, cap, f"NTK-aware: raise the base to {base_ntk:,.0f}; fast pairs stay, slow pairs × 4",
                           color=ORANGE_, size=26)
        self.play(Create(lines[2]), FadeIn(leg[2]), run_time=1.2)
        self.wait(1.4)
        cap = swap_caption(self, cap, "YaRN: keep pairs with > 32 turns in L, interpolate those with < 1, ramp between",
                           color=PURPLE_, size=26)
        self.play(Create(lines[3]), FadeIn(leg[3]), run_time=1.2)
        self.wait(2.5)
