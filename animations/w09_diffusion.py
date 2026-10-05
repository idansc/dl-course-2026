"""Week 9 (diffusion and flow matching). Render one scene:
    manim -qh --disable_caching -o ForwardReverseDiffusion w09_diffusion.py ForwardReverseDiffusion
All data are 2-D Gaussian mixtures, so scores, denoisers and flow-matching velocities are exact (w09_gmm.py).
FewStepSampling also needs the reflowed MLP from `python w09_precompute.py` -> data/w09_reflow.npz.
"""
from pathlib import Path
from style import *
from w09_gmm import GMM, two_moons, ring, cosine_abar, ddpm_sample, euler

DATA = Path(__file__).parent / "data"


def plane(x_range, y_range, w, h, at):
    return NumberPlane(x_range=x_range, y_range=y_range, x_length=w, y_length=h,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.2, "stroke_width": 1},
                       axis_config={"stroke_color": MUTED, "stroke_width": 1.5}).move_to(at)


def cloud(P, X, color=BLUE_, r=0.035, opacity=0.75):
    return VGroup(*[Dot(P.c2p(*x), radius=r, color=color, fill_opacity=opacity) for x in X])


def place(c, P, X):
    for dt, x in zip(c, X):
        dt.move_to(P.c2p(*x))


def polyline(points, color, width=2, opacity=1.0):
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(points)
    return m


def gauss_ring(P, center, radius, color, width=2):
    unit = P.c2p(1, 0)[0] - P.c2p(0, 0)[0]
    return DashedVMobject(Circle(radius=radius * unit, color=color, stroke_width=width), num_dashes=36).move_to(P.c2p(*center))


def interp_frames(F, k):
    """Linear interpolation of a stack of frames F (T, n, 2) at fractional index k."""
    k = float(np.clip(k, 0, len(F) - 1))
    i0 = int(np.floor(k)); i1 = min(i0 + 1, len(F) - 1); f = k - i0
    return (1 - f) * F[i0] + f * F[i1]


# ----------------------------------------------------------------------------------------------------------------
class ForwardReverseDiffusion(Scene):
    """Two moons (a 32-component Gaussian mixture) dissolve into N(0, I) under x_t = sqrt(abar) x_0 + sqrt(1-abar) eps,
    then fresh noise is denoised back by 100 ancestral DDPM steps with the exact score; one particle shows its
    x0-prediction moving from the data mean to a moon. Target: p10 (DDPM forward/reverse)."""

    def construct(self):
        g = two_moons()
        rng = np.random.default_rng(0)
        X0, _ = g.sample(400, rng)
        E = rng.standard_normal(X0.shape)
        title(self, "Diffusion: data → noise → data", "two moons (a Gaussian mixture), exact score, 100 DDPM steps")
        P = plane([-3.5, 3.5, 1], [-3.5, 3.5, 1], 5.2, 5.2, [-3.5, -0.4, 0])
        t = ValueTracker(0.0)

        # right panel: formula, schedule, readout
        f = MathTex(r"x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon", font_size=40).move_to([3.6, 1.75, 0])
        fsub = Text("x_0: data     ε ~ N(0, I): fresh noise", font_size=20, color=MUTED).next_to(f, DOWN, buff=0.15)
        A = Axes(x_range=[0, 1, 0.5], y_range=[0, 1, 0.5], x_length=4.4, y_length=2.0,
                 axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False, "font_size": 20},
                 x_axis_config={"numbers_to_include": [0, 0.5, 1], "decimal_number_config": {"color": MUTED, "num_decimal_places": 1}},
                 y_axis_config={"numbers_to_include": [0, 1], "decimal_number_config": {"color": MUTED, "num_decimal_places": 0}},
                 ).move_to([3.6, -0.5, 0])
        ts = np.linspace(0, 1, 120)
        ab = cosine_abar(ts)
        sig = polyline([A.c2p(a, b) for a, b in zip(ts, np.sqrt(ab))], BLUE_, 4)
        noi = polyline([A.c2p(a, b) for a, b in zip(ts, np.sqrt(1 - ab))], RED_, 4)
        sig_l = MathTex(r"\sqrt{\bar\alpha_t}", font_size=28, color=BLUE_).next_to(A.c2p(0.1, 1.0), RIGHT, buff=0.1).shift(0.15 * UP)
        noi_l = MathTex(r"\sqrt{1-\bar\alpha_t}", font_size=28, color=RED_).next_to(A.c2p(0.62, 1.0), RIGHT, buff=0.1).shift(0.15 * UP)
        t_l = MathTex("t", font_size=28, color=MUTED).next_to(A.c2p(1, 0), RIGHT, buff=0.15)
        cursor = always_redraw(lambda: DashedLine(A.c2p(t.get_value(), 0), A.c2p(t.get_value(), 1.0), color=INK, stroke_width=2))
        readout = always_redraw(lambda: Text(
            f"t = {t.get_value():.2f}     signal {np.sqrt(cosine_abar(t.get_value())):.2f}     noise {np.sqrt(1 - cosine_abar(t.get_value())):.2f}",
            font_size=22).move_to([3.6, -2.15, 0]))

        dots = cloud(P, X0, BLUE_)
        cap = caption("1. data: 400 points from a two-moons distribution")
        self.play(FadeIn(cap), Create(P))
        self.play(LaggedStart(*[FadeIn(dt) for dt in dots], lag_ratio=0.003), run_time=1.5)
        self.add(dots)
        self.wait(0.6)

        def fwd(m):
            a = cosine_abar(t.get_value())
            place(m, P, np.sqrt(a) * X0 + np.sqrt(1 - a) * E)

        cap = swap_caption(self, cap, "2. forward process: shrink the data by √ᾱ_t and mix in Gaussian noise")
        self.play(Write(f), FadeIn(fsub))
        self.play(Create(A), Create(sig), Create(noi), FadeIn(sig_l), FadeIn(noi_l), FadeIn(t_l), FadeIn(cursor), FadeIn(readout))
        dots.add_updater(fwd)
        self.play(t.animate.set_value(0.5), run_time=5, rate_func=linear)
        self.wait(0.5)
        cap = swap_caption(self, cap, "3. by t = 1 the signal is gone: x_1 is pure N(0, I) noise")
        self.play(t.animate.set_value(1.0), run_time=3.5, rate_func=linear)
        dots.remove_updater(fwd)
        ring2 = gauss_ring(P, (0, 0), 2, INK)
        ring2_l = Text("2σ of N(0, I)", font_size=18, color=INK).next_to(ring2, UR, buff=-0.45)
        self.play(Create(ring2), FadeIn(ring2_l))
        self.wait(0.8)

        # reverse
        XT = np.random.default_rng(1).standard_normal((400, 2))
        _, XS, X0S = ddpm_sample(g.eps_hat, XT, T=100, rng=np.random.default_rng(2), record=True)
        k = ValueTracker(0.0)
        new = cloud(P, XT, BLUE_)
        cap = swap_caption(self, cap, "4. reverse: start from fresh noise; the model predicts the clean point x̂_0")
        self.play(FadeOut(dots), FadeIn(new), FadeOut(ring2), FadeOut(ring2_l))
        j = int(np.argmin(((XS[-1] - np.array([-1.6, 0.4])) ** 2).sum(1)))    # a particle that ends on the upper moon
        hl = always_redraw(lambda: Dot(P.c2p(*interp_frames(XS, k.get_value())[j]), radius=0.09, color=ORANGE_))
        x0r = always_redraw(lambda: Circle(radius=0.13, color=PURPLE_, stroke_width=5).move_to(
            P.c2p(*interp_frames(X0S, min(k.get_value(), 99))[j])))
        x0l = always_redraw(lambda: DashedLine(P.c2p(*interp_frames(XS, k.get_value())[j]),
                                               P.c2p(*interp_frames(X0S, min(k.get_value(), 99))[j]), color=PURPLE_, stroke_width=3))
        x0t = MathTex(r"\hat x_0", font_size=30, color=PURPLE_)
        x0t.add_updater(lambda m: m.next_to(P.c2p(*interp_frames(X0S, min(k.get_value(), 99))[j]), UR, buff=0.16))
        new.add_updater(lambda m: place(m, P, interp_frames(XS, k.get_value())))
        t.add_updater(lambda m: m.set_value(1 - k.get_value() / 100))
        self.play(FadeIn(hl), FadeIn(x0r), FadeIn(x0l), FadeIn(x0t))
        self.wait(1.0)
        cap = swap_caption(self, cap, "5. each step: move a little toward x̂_0, add a little fresh noise")
        self.play(k.animate.set_value(45), run_time=6, rate_func=linear)
        cap = swap_caption(self, cap, "6. early x̂_0 is a blurry average; later it commits to one moon")
        self.play(k.animate.set_value(100), run_time=6, rate_func=linear)
        new.clear_updaters()
        t.clear_updaters()
        ghost = cloud(P, X0, GREEN_, r=0.03, opacity=0.35)
        cap = swap_caption(self, cap, "100 small denoising steps turn noise into new two-moons samples", color=PURPLE_)
        gl = VGroup(Dot(radius=0.06, color=GREEN_, fill_opacity=0.5), Text("training data", font_size=20, color=GREEN_),
                    Dot(radius=0.06, color=BLUE_), Text("new samples", font_size=20, color=BLUE_)).arrange(RIGHT, buff=0.12)
        gl[2].shift(0.3 * RIGHT); gl[3].shift(0.3 * RIGHT)
        gl.move_to([3.6, -2.65, 0])
        self.play(FadeIn(ghost), FadeIn(gl), FadeOut(x0r), FadeOut(x0l), FadeOut(x0t))
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------
class FlowMatchingPaths(Scene):
    """Flow matching on two moons: straight conditional paths x_t = (1-t) x_0 + t x_1, crossing paths averaged into
    the exact marginal velocity v(x, t) = E[x_1 - x_0 | x_t = x], the field over t, and Euler sampling.
    Target: L10b (flow matching)."""

    def construct(self):
        g = two_moons()
        rng = np.random.default_rng(3)
        title(self, "Flow matching", "noise x_0 ~ N(0, I) → data x_1 (two moons); exact velocity for this mixture")
        P = plane([-3.5, 3.5, 1], [-3.5, 3.5, 1], 5.2, 5.2, [-3.5, -0.4, 0])
        n = 60
        X1, _ = g.sample(n, rng)
        X0 = rng.standard_normal((n, 2))
        f1 = MathTex(r"x_t = (1-t)\,x_0 + t\,x_1", font_size=40).move_to([3.6, 1.75, 0])
        f2 = MathTex(r"\tfrac{d x_t}{dt} = x_1 - x_0", font_size=36).next_to(f1, DOWN, buff=0.3)

        noise_c = cloud(P, X0, MUTED, 0.045, 0.9)
        data_c = cloud(P, X1, GREEN_, 0.045, 0.9)
        lg = VGroup(VGroup(Dot(radius=0.07, color=MUTED), Text("noise x_0", font_size=20, color=MUTED)).arrange(RIGHT, buff=0.1),
                    VGroup(Dot(radius=0.07, color=GREEN_), Text("data x_1", font_size=20, color=GREEN_)).arrange(RIGHT, buff=0.1)
                    ).arrange(RIGHT, buff=0.4).move_to([3.6, 0.45, 0])
        cap = caption("1. two clouds: Gaussian noise x_0 (gray) and data x_1 (green)")
        self.play(FadeIn(cap), Create(P), FadeIn(lg))
        self.play(FadeIn(noise_c), FadeIn(data_c))
        self.wait(0.6)

        cap = swap_caption(self, cap, "2. pair each noise point with a random data point; join them by a straight line")
        lines = VGroup(*[Line(P.c2p(*a), P.c2p(*b), color=MUTED, stroke_width=1.5, stroke_opacity=0.6) for a, b in zip(X0, X1)])
        self.play(Create(lines), Write(f1), run_time=1.5)
        self.wait(0.5)

        cap = swap_caption(self, cap, "3. along its line a point moves at constant velocity x_1 − x_0")
        tt = ValueTracker(0)
        movers = cloud(P, X0, BLUE_, 0.06, 1)
        movers.add_updater(lambda m: place(m, P, (1 - tt.get_value()) * X0 + tt.get_value() * X1))
        self.play(FadeIn(movers), Write(f2))
        self.play(tt.animate.set_value(1), run_time=3, rate_func=linear)
        movers.clear_updaters()
        self.play(FadeOut(movers))
        self.wait(0.3)

        # crossing directions at one point
        q, tq, rad = np.array([0.8, 0.6]), 0.5, 0.18
        big1, _ = g.sample(60000, rng)
        big0 = rng.standard_normal(big1.shape)
        xt = (1 - tq) * big0 + tq * big1
        sel = np.flatnonzero(((xt - q) ** 2).sum(1) < rad ** 2)
        V = (big1 - big0)[sel]
        vbar, vex = V.mean(0), g.velocity(q[None], tq)[0]
        sc = 0.6
        cap = swap_caption(self, cap, f"4. lines cross: at one point (t = {tq}) the {len(sel)} passing pairs point in different directions")
        self.play(lines.animate.set_stroke(opacity=0.15), noise_c.animate.set_opacity(0.25), data_c.animate.set_opacity(0.25))
        qc = Circle(radius=rad * (P.c2p(1, 0)[0] - P.c2p(0, 0)[0]), color=INK, stroke_width=3).move_to(P.c2p(*q))
        arrows = VGroup(*[Arrow(P.c2p(*q), P.c2p(*(q + sc * v)), buff=0, color=ORANGE_, stroke_width=2, stroke_opacity=0.55,
                                max_tip_length_to_length_ratio=0.12) for v in V[:40]])
        self.play(Create(qc), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.03), run_time=2)
        self.wait(0.6)
        cap = swap_caption(self, cap, "5. the network regresses onto their average: v(x, t) = E[x_1 − x_0 | x_t = x]")
        mean_arrow = Arrow(P.c2p(*q), P.c2p(*(q + sc * vex)), buff=0, color=PURPLE_, stroke_width=8)
        nums = VGroup(
            Text(f"average of the {len(sel)} pairs:  ({vbar[0]:+.2f}, {vbar[1]:+.2f})", font_size=22, color=ORANGE_),
            Text(f"exact v(x, t) for the mixture:  ({vex[0]:+.2f}, {vex[1]:+.2f})", font_size=22, color=PURPLE_),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([3.6, -0.5, 0])
        self.play(GrowArrow(mean_arrow), FadeIn(nums))
        self.wait(2)

        # field over t
        cap = swap_caption(self, cap, "6. that average is a velocity field; it changes with t")
        self.play(FadeOut(VGroup(arrows, qc, mean_arrow, lines, noise_c, data_c)), FadeOut(nums))
        grid = np.array([[x, y] for y in np.linspace(-3, 3, 13) for x in np.linspace(-3, 3, 13)])
        tf = ValueTracker(0.0)

        def field():
            v = g.velocity(grid, tf.get_value())
            L = np.linalg.norm(v, axis=1, keepdims=True)
            vs = v / np.maximum(L, 1e-9) * np.minimum(L, 3.0) * 0.14
            return VGroup(*[Arrow(P.c2p(*p), P.c2p(*(p + d)), buff=0, color=PURPLE_, stroke_width=2.5,
                                  max_tip_length_to_length_ratio=0.35, max_stroke_width_to_length_ratio=12)
                            for p, d in zip(grid, vs)])
        fld = always_redraw(field)
        tread = always_redraw(lambda: Text(f"t = {tf.get_value():.2f}", font_size=30).move_to([3.6, -0.5, 0]))
        self.play(FadeIn(fld), FadeIn(tread))
        self.play(tf.animate.set_value(0.95), run_time=4, rate_func=linear)
        self.play(tf.animate.set_value(0.0), run_time=1)

        # Euler sampling
        cap = swap_caption(self, cap, "7. sample: fresh noise, 40 Euler steps x ← x + Δt · v(x, t)")
        Z = np.random.default_rng(5).standard_normal((300, 2))
        steps = 40
        path = euler(g.velocity, Z, steps, record=True)
        parts = cloud(P, Z, BLUE_, 0.04, 0.9)
        k = ValueTracker(0)
        ntr = 30

        def trails():
            kk = k.get_value()
            i0 = int(np.floor(kk))
            g_ = VGroup()
            for p in range(ntr):
                pts = [P.c2p(*path[i, p]) for i in range(i0 + 1)] + [P.c2p(*interp_frames(path, kk)[p])]
                g_.add(polyline(pts, BLUE_, 2, 0.6))
            return g_
        tr = always_redraw(trails)
        self.play(FadeIn(parts))
        self.add(tr)
        parts.add_updater(lambda m: place(m, P, interp_frames(path, k.get_value())))
        tf.add_updater(lambda m: m.set_value(min(k.get_value() / steps, 0.999)))
        self.play(k.animate.set_value(steps), run_time=7, rate_func=linear)
        parts.clear_updaters()
        tf.clear_updaters()
        cap = swap_caption(self, cap, "8. the averaged field bends the paths, and the particles land on the data", color=PURPLE_)
        self.play(FadeOut(fld), FadeOut(tread))
        ghost = cloud(P, g.sample(300, np.random.default_rng(9))[0], GREEN_, 0.03, 0.35)
        self.play(FadeIn(ghost))
        self.wait(2.2)


# ----------------------------------------------------------------------------------------------------------------
class GuidanceScale(Scene):
    """Classifier-free guidance on a 2-D class-conditional Gaussian mixture with exact conditional and unconditional
    noise predictions. Class A has a distinctive mode and a B-like mode; as w grows, samples abandon the B-like part
    and bunch up (spread and B-like share measured over 300 samples). Target: p10 59-62 (CFG)."""

    def construct(self):
        M = np.array([[-1.6, -0.3], [0.3, -0.4], [1.4, -0.6], [0.0, 1.5]])
        S = np.array([0.45, 0.4, 0.5, 0.5])
        full = GMM(M, S, np.array([0.6, 0.4, 1.0, 1.0]))
        cond = full.subset([0, 1])                                           # class A = components 0 and 1
        Ws = np.linspace(0, 8, 33)
        XT = np.random.default_rng(0).standard_normal((300, 2))
        runs = []
        for w in Ws:
            fn = lambda x, a, w=w: (1 + w) * cond.eps_hat(x, a) - w * full.eps_hat(x, a)
            runs.append(ddpm_sample(fn, XT, T=100, rng=np.random.default_rng(1)))
        runs = np.array(runs)
        spread = np.array([np.sqrt(((R - R.mean(0)) ** 2).sum(1).mean()) for R in runs])
        blike = np.array([((((R[:, None] - M[None]) ** 2).sum(-1) / S[None] ** 2).argmin(1) == 1).mean() for R in runs])

        title(self, "Classifier-free guidance", "exact noise predictions for a 2-D mixture, 100 DDPM steps, 300 samples of class A")
        P = plane([-3.5, 3.5, 1], [-3.5, 3.5, 1], 5.2, 5.2, [-3.5, -0.4, 0])
        rings = VGroup(*[gauss_ring(P, m, 2 * sd, col, 2.5) for m, sd, col in zip(M, S, [BLUE_, BLUE_, ORANGE_, GREEN_])])
        labs = VGroup(
            Text("class A", font_size=22, color=BLUE_, weight=BOLD).next_to(P.c2p(*(M[0] + [0, 2 * S[0]])), UP, buff=0.05),
            Text("A, B-like", font_size=18, color=BLUE_).next_to(P.c2p(*(M[1] - [0, 2 * S[1]])), DOWN, buff=0.05),
            Text("class B", font_size=22, color=ORANGE_, weight=BOLD).next_to(P.c2p(*(M[2] + [0.3, 2 * S[2]])), UP, buff=0.05),
            Text("class C", font_size=22, color=GREEN_, weight=BOLD).next_to(P.c2p(*(M[3] + [0, 2 * S[3]])), UP, buff=0.05))
        cap = caption("1. three classes; some class-A examples look like class B")
        self.play(FadeIn(cap), Create(P))
        self.play(FadeIn(rings), FadeIn(labs))
        self.wait(1.0)

        wt = ValueTracker(0.0)
        cur = lambda: interp_frames(runs, wt.get_value() / (Ws[1] - Ws[0]))
        dots = cloud(P, runs[0], BLUE_, 0.04, 0.8)
        cap = swap_caption(self, cap, f"2. w = 0: conditional sampling covers all of class A ({blike[0]:.0%} in the B-like part)")
        self.play(LaggedStart(*[FadeIn(d_) for d_ in dots], lag_ratio=0.004), run_time=1.5)
        self.add(dots)
        self.wait(1.0)

        f = MathTex(r"\tilde\varepsilon = \hat\varepsilon(x\,|\,A) + w\,\big(\hat\varepsilon(x\,|\,A) - \hat\varepsilon(x)\big)",
                    font_size=34).move_to([3.6, 2.05, 0])
        fsub = Text("push along 'more A than average'", font_size=20, color=MUTED).next_to(f, DOWN, buff=0.12)
        cap = swap_caption(self, cap, "3. guidance: extrapolate away from the unconditional prediction, scale w")
        self.play(Write(f), FadeIn(fsub))

        sl = NumberLine(x_range=[0, 8, 1], length=4.6, color=MUTED, include_numbers=True, font_size=20,
                        decimal_number_config={"color": MUTED, "num_decimal_places": 0}).move_to([3.7, 0.45, 0])
        sl_l = MathTex("w", font_size=32).next_to(sl, LEFT, buff=0.2)
        knob = always_redraw(lambda: Triangle(color=PURPLE_, fill_opacity=1).scale(0.12).rotate(PI).move_to(sl.n2p(wt.get_value()) + 0.2 * UP))
        wval = always_redraw(lambda: Text(f"w = {wt.get_value():.1f}", font_size=26, color=PURPLE_).move_to([3.7, 1.05, 0]))

        def chart(vals, yhi, at, label, color, fmt):
            ax = Axes(x_range=[0, 8, 2], y_range=[0, yhi, yhi / 2], x_length=2.3, y_length=1.45,
                      axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False, "font_size": 16},
                      x_axis_config={"numbers_to_include": [0, 4, 8], "decimal_number_config": {"color": MUTED, "num_decimal_places": 0}},
                      ).move_to(at)
            lab = Text(label, font_size=19, color=color).next_to(ax, UP, buff=0.22)

            def cur_curve():
                wv = wt.get_value()
                m = Ws <= wv + 1e-9
                xs = list(Ws[m]) + [wv]
                ys = list(vals[m]) + [np.interp(wv, Ws, vals)]
                pts = [ax.c2p(x, y) for x, y in zip(xs, ys)]
                g_ = VGroup(Dot(pts[-1], radius=0.06, color=color),
                            Text(fmt.format(ys[-1]), font_size=18, color=color).next_to(pts[-1], RIGHT, buff=0.08))
                if len(pts) > 1:
                    g_.add(polyline(pts, color, 3))
                return g_
            return VGroup(ax, lab), always_redraw(cur_curve)

        c1, c1c = chart(spread, 1.2, [2.25, -1.55, 0], "spread (diversity)", BLUE_, "{:.2f}")
        c2, c2c = chart(blike, 0.4, [5.35, -1.55, 0], "share in A's B-like part", RED_, "{:.0%}")
        self.play(Create(sl), FadeIn(sl_l), FadeIn(knob), FadeIn(wval), FadeIn(c1), FadeIn(c2), FadeIn(c1c), FadeIn(c2c))
        dots.add_updater(lambda m: place(m, P, cur()))
        self.wait(0.5)
        cap = swap_caption(self, cap, "4. w > 0: samples leave the part of A that overlaps class B")
        self.play(wt.animate.set_value(2.0), run_time=4.5, rate_func=linear)
        self.wait(0.6)
        cap = swap_caption(self, cap, "5. large w: everything piles onto A's most distinctive mode")
        self.play(wt.animate.set_value(8.0), run_time=4.5, rate_func=linear)
        self.wait(1.0)
        cap = swap_caption(self, cap, f"guidance buys class fidelity with diversity: spread {spread[0]:.2f} → {spread[-1]:.2f}",
                           color=PURPLE_)
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------
class FewStepSampling(Scene):
    """8-mode ring: Euler sampling with 1/2/4/8 steps on the exact (curved) flow-matching field vs a reflowed MLP
    trained on (noise, ODE endpoint) pairs (nearly straight paths). One exact Euler step lands every sample on the
    data mean. Target: L10b (few-step sampling, rectified flow)."""

    def construct(self):
        d = np.load(DATA / "w09_reflow.npz")
        M, S = d["M"], d["S"]
        title(self, "Few-step sampling needs straight paths", "8-mode ring; 300 noise samples; Euler steps on the velocity field")
        Ps = {}
        for name, cx, hdr in [("exact", -3.65, "flow matching (exact field)"), ("reflow", 3.65, "after reflow (trained MLP)")]:
            P = plane([-3.6, 3.6, 1], [-3.6, 3.6, 1], 4.4, 4.4, [cx, -0.2, 0])
            modes = VGroup(*[gauss_ring(P, m, 3 * S[0], GREEN_, 2) for m in M])
            h = Text(hdr, font_size=24, weight=BOLD).next_to(P, UP, buff=0.1)
            Ps[name] = dict(P=P, modes=modes, hdr=h)
        mid = Text("Euler\nsteps", font_size=22, color=MUTED, line_spacing=0.8).move_to([0, 1.3, 0])
        chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=0.8, height=0.55, stroke_color=MUTED, stroke_width=2,
                                                 fill_color=WHITE, fill_opacity=1), Text(str(n), font_size=26, color=MUTED))
                         for n in [1, 2, 4, 8]]).arrange(DOWN, buff=0.15).move_to([0, -0.6, 0])

        def chip_on(i):
            anims = []
            for j, c in enumerate(chips):
                on = j == i
                anims += [c[0].animate.set_fill(PURPLE_ if on else WHITE).set_stroke(PURPLE_ if on else MUTED),
                          c[1].animate.set_color(WHITE if on else MUTED)]
            return anims

        def metric(name, n):
            P = Ps[name]["P"]
            return Text(f"{n} step{'s' if n > 1 else ''}: {d[f'{name}_{n}_hit']:.0%} land on a mode", font_size=22,
                        color=RED_ if d[f"{name}_{n}_hit"] < 0.5 else GREEN_).next_to(P, DOWN, buff=0.12)

        ntr = 40

        def path_lines(Pth, P, color=BLUE_, op=0.5, w=2):
            return VGroup(*[polyline([P.c2p(*x) for x in Pth[:, i]], color, w, op) for i in range(ntr)])

        ex = Ps["exact"]
        cap = caption("1. exact flow-matching field: 50 small Euler steps follow curved paths")
        self.play(FadeIn(cap), Create(ex["P"]), FadeIn(ex["modes"]), FadeIn(ex["hdr"]))
        z_ex = cloud(ex["P"], d["z"], BLUE_, 0.035, 0.8)
        self.play(FadeIn(z_ex))
        trails = path_lines(d["exact_path"], ex["P"])
        self.play(Create(trails), *[dt.animate.move_to(ex["P"].c2p(*x)) for dt, x in zip(z_ex, d["exact_path"][-1])], run_time=2.5)
        self.wait(0.8)

        def show_n(name, n, dots, segs, run_time=1.4):
            P = Ps[name]["P"]
            pth = d[f"{name}_{n}_path"]
            new_segs = path_lines(pth, P, ORANGE_, 0.8, 2.5)
            for dt, x in zip(dots, d["z"]):
                dt.move_to(P.c2p(*x))
            return [Transform(segs, new_segs), *[dt.animate.move_to(P.c2p(*x)) for dt, x in zip(dots, pth[-1])]]

        cap = swap_caption(self, cap, "2. one step: x_0 + v(x_0, 0) = the data mean, so every sample lands in the empty center",
                           size=26)
        self.play(trails.animate.set_stroke(opacity=0.12), FadeIn(mid), FadeIn(chips), *chip_on(0))
        segs_ex = VGroup()
        self.add(segs_ex)
        self.play(*show_n("exact", 1, z_ex, segs_ex), run_time=1.6)
        m_ex = metric("exact", 1)
        self.play(FadeIn(m_ex))
        self.wait(1.5)
        cap = swap_caption(self, cap, "3. 2, 4, 8 steps: closer, but straight steps cut the curves short")
        for i, n in enumerate([2, 4, 8]):
            self.play(*chip_on(i + 1), *show_n("exact", n, z_ex, segs_ex), Transform(m_ex, metric("exact", n)), run_time=1.4)
            self.wait(0.8)

        # reflow
        rf = Ps["reflow"]
        cap = swap_caption(self, cap, "4. reflow: retrain on (noise, its own ODE endpoint) pairs → nearly straight paths")
        self.play(Create(rf["P"]), FadeIn(rf["modes"]), FadeIn(rf["hdr"]))
        z_rf = cloud(rf["P"], d["z"], BLUE_, 0.035, 0.8)
        self.play(FadeIn(z_rf))
        trails_rf = path_lines(d["reflow_path"], rf["P"])
        self.play(Create(trails_rf), *[dt.animate.move_to(rf["P"].c2p(*x)) for dt, x in zip(z_rf, d["reflow_path"][-1])], run_time=2.5)
        self.wait(0.8)
        cap = swap_caption(self, cap, f"5. now one step already works: {d['reflow_1_hit']:.0%} on a mode vs {d['exact_1_hit']:.0%}")
        self.play(trails_rf.animate.set_stroke(opacity=0.12))
        segs_rf = VGroup()
        self.add(segs_rf)
        m_rf = metric("reflow", 1)
        self.play(*chip_on(0), *show_n("exact", 1, z_ex, segs_ex), *show_n("reflow", 1, z_rf, segs_rf),
                  Transform(m_ex, metric("exact", 1)), FadeIn(m_rf), run_time=1.6)
        self.wait(1.8)
        for i, n in enumerate([2, 4, 8]):
            self.play(*chip_on(i + 1), *show_n("exact", n, z_ex, segs_ex), *show_n("reflow", n, z_rf, segs_rf),
                      Transform(m_ex, metric("exact", n)), Transform(m_rf, metric("reflow", n)), run_time=1.2)
            self.wait(0.6)
        cap = swap_caption(self, cap, "straight paths = few-step sampling (rectified flow, distillation)", color=PURPLE_)
        self.wait(2.2)
