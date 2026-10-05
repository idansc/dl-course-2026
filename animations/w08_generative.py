"""Week 8 (generative models: AE/VAE, GANs, VQ). Render one scene:
    manim -qh --disable_caching -o Reparameterization w08_generative.py Reparameterization
Trained models (AE/VAE on sklearn's 8x8 digits, the 1-D GANs) come from `python w08_precompute.py` -> data/w08_*.npz.
"""
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


def plane(x_range, y_range, w, h, at):
    return NumberPlane(x_range=x_range, y_range=y_range, x_length=w, y_length=h,
                       background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.2, "stroke_width": 1},
                       axis_config={"stroke_color": MUTED, "stroke_width": 2}).move_to(at)


def digit_img(v, size=0.62):
    """8x8 digit (values in [0,1], ink = 1) as a dark-on-white image with a thin frame."""
    a = (255 * (1 - np.clip(np.asarray(v).reshape(8, 8), 0, 1))).astype(np.uint8)
    im = ImageMobject(np.repeat(np.repeat(a, 8, 0), 8, 1))
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = size
    fr = Rectangle(width=size, height=size, stroke_color=MUTED, stroke_width=1.5).move_to(im)
    return Group(im, fr)


def polyline(points, color, width=4, opacity=1.0):
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(points)
    return m


# ----------------------------------------------------------------------------------------------------------------
class AEvsVAELatent(Scene):
    """Real 2-D latent spaces of an autoencoder and a VAE (same MLPs, 15k Adam steps) trained on sklearn's 8x8
    digits: the AE spreads its codes over an arbitrary range with gaps, the VAE's KL term packs them into N(0, I).
    A 0 -> 3 walk and 8 prior draws are decoded by the real decoders; a logistic-regression digit classifier
    (trained on the real images) supplies the numbers. Target: p9 (autoencoders -> VAE), mm9."""

    def construct(self):
        d = np.load(DATA / "w08_aevae.npz")
        y = d["y"]
        cols = [BLUE_, ORANGE_, GREEN_, RED_, PURPLE_, "#92400E", "#DB2777", "#6B7280", "#CA8A04", "#0891B2"]
        title(self, "Autoencoder vs VAE latent space", "real models: 8×8 handwritten digits → 2 numbers")
        legend = VGroup(*[VGroup(Dot(radius=0.06, color=cols[k]), Text(str(k), font_size=22, color=cols[k], weight=BOLD)).arrange(RIGHT, buff=0.1)
                          for k in range(10)]).arrange(DOWN, buff=0.1)
        legend = VGroup(Text("digit", font_size=20, color=MUTED), legend).arrange(DOWN, buff=0.15).move_to([0, 0.3, 0])
        side = {}
        for kind, cx in [("ae", -3.55), ("vae", 3.55)]:
            Z = d[f"{kind}_z"]
            if kind == "ae":
                lo, hi = np.percentile(Z, 1, 0), np.percentile(Z, 99, 0)
                c, r, step = (lo + hi) / 2, (hi - lo).max() / 2 * 1.08, 10
            else:
                c, r, step = np.zeros(2), 3.3, 1
            ax = Axes(x_range=[c[0] - r, c[0] + r, step], y_range=[c[1] - r, c[1] + r, step], x_length=3.6, y_length=3.6,
                      axis_config={"color": MUTED, "stroke_width": 1.5, "include_tip": False, "tick_size": 0.05}).move_to([cx, 0.3, 0])
            frame = SurroundingRectangle(ax, buff=0, color=MUTED, stroke_width=1.5)
            ticks = VGroup()
            for v in np.arange(np.ceil((c[0] - r) / step) * step, c[0] + r, step):
                if abs(v) > 1e-9:
                    ticks.add(Text(f"{v:g}", font_size=14, color=MUTED).next_to(ax.c2p(v, c[1] - r), DOWN, buff=0.06))
            for v in np.arange(np.ceil((c[1] - r) / step) * step, c[1] + r, step):
                if abs(v) > 1e-9:
                    ticks.add(Text(f"{v:g}", font_size=14, color=MUTED).next_to(ax.c2p(c[0] - r, v), LEFT, buff=0.06))
            box = VGroup(ax, frame, ticks)
            inside = (np.abs(Z - c) < r).all(1)
            dots = VGroup(*[Dot(ax.c2p(*z), radius=0.022, color=cols[int(t)], fill_opacity=0.8) for z, t, ok in zip(Z, y, inside) if ok])
            name = "autoencoder" if kind == "ae" else "variational autoencoder (VAE)"
            hdr = Text(name, font_size=24, weight=BOLD).next_to(frame, UP, buff=0.08)
            side[kind] = dict(p=ax, box=box, dots=dots, hdr=hdr)
        ae, vae = side["ae"], side["vae"]

        cap = caption("1. an autoencoder squeezes each 64-pixel digit into a 2-D code z = enc(x)")
        self.play(FadeIn(cap), FadeIn(ae["box"]), FadeIn(ae["hdr"]), FadeIn(legend))
        self.play(LaggedStart(*[FadeIn(dt) for dt in ae["dots"]], lag_ratio=0.0008), run_time=2)
        self.wait(0.6)
        cap = swap_caption(self, cap, "2. it is only trained to reconstruct: codes spread over an arbitrary range, with gaps")
        self.play(Indicate(ae["box"][2], color=INK, scale_factor=1.15), run_time=1.2)
        self.wait(1.2)

        cap = swap_caption(self, cap, "3. a VAE encodes a Gaussian q(z|x) and pays KL(q ‖ N(0, I))")
        unit = vae["p"].c2p(1, 0)[0] - vae["p"].c2p(0, 0)[0]
        ring = VGroup(*[DashedVMobject(Circle(radius=rr * unit, color=INK, stroke_width=2), num_dashes=40).move_to(vae["p"].c2p(0, 0))
                        for rr in (1, 2)])
        ring_lab = Text("N(0, I): 1σ, 2σ", font_size=16, color=INK).move_to(vae["p"].c2p(2.2, 2.85))
        self.play(FadeIn(vae["box"]), FadeIn(vae["hdr"]))
        self.play(LaggedStart(*[FadeIn(dt) for dt in vae["dots"]], lag_ratio=0.0008), Create(ring), FadeIn(ring_lab), run_time=2)
        cap = swap_caption(self, cap, "4. KL pulls every code toward the origin: all ten digits share one compact disk")
        self.wait(1.8)

        def strip(imgs, pl, notes=None, color=MUTED):
            g = Group(*[digit_img(v, 0.42) for v in imgs]).arrange(RIGHT, buff=0.07)
            g.next_to(pl, DOWN, buff=0.12).set_x(pl.get_x())
            if notes is not None:
                g.add(*[Text(n, font_size=15, color=color).next_to(im, DOWN, buff=0.06) for n, im in zip(notes, g)])
            return g

        def walk(s):
            line, pl = d[f"{s}_line"], side[s]["p"]
            seg = DashedLine(pl.c2p(*line[0]), pl.c2p(*line[-1]), color=INK, stroke_width=3)
            marks = VGroup(*[Dot(pl.c2p(*z), radius=0.045, color=INK) for z in line])
            st = strip(d[f"{s}_line_img"], side[s]["box"], [f"{c:.2f}" for c in d[f"{s}_line_conf"]])
            self.play(Create(seg), run_time=0.5)
            for i in range(len(line)):
                self.play(FadeIn(marks[i], scale=1.6), FadeIn(st[i], shift=0.1 * DOWN), FadeIn(st[len(line) + i]), run_time=0.25)
            return VGroup(seg, marks), st

        a, b = d["end_classes"]
        cap = swap_caption(self, cap, f"5. walk a straight line from a {a} to a {b} in each space and decode every point")
        ae_line, ae_strip = walk("ae")
        vae_line, vae_strip = walk("vae")
        note = Text("under each image: a digit classifier's confidence", font_size=17, color=MUTED).move_to([0, -2.75, 0])
        self.play(FadeIn(note))
        cap = swap_caption(self, cap, "6. the AE path crosses a gap and decodes smudges; the VAE path stays digit-like")
        self.play(*[Indicate(ae_strip[i], color=RED_, scale_factor=1.25) for i in (3, 4)], run_time=1.2)
        self.wait(1.4)

        cap = swap_caption(self, cap, "7. generate: draw z ~ N(0, I) and decode (the same 8 draws on both sides)")
        self.play(FadeOut(ae_line), FadeOut(vae_line), FadeOut(ae_strip), FadeOut(vae_strip), FadeOut(note))
        strips = []
        for s in ["ae", "vae"]:
            pl = side[s]["p"]
            xs = VGroup(*[Cross(scale_factor=0.06, stroke_color=INK, stroke_width=3).move_to(pl.c2p(*z)) for z in d[f"{s}_draw"]])
            st = strip(d[f"{s}_draw_img"], side[s]["box"], [f"→ {c}" for c in d[f"{s}_draw_cls"]], INK)
            self.play(LaggedStart(*[FadeIn(x, scale=2) for x in xs], lag_ratio=0.1),
                      LaggedStart(*[FadeIn(im) for im in st], lag_ratio=0.05), run_time=1.4)
            strips.append(st)
        h_ae, h_vae = d["ae_stat_hist"], d["vae_stat_hist"]
        n = h_ae.sum()
        top = int(np.argmax(h_ae))
        cap = swap_caption(self, cap, "8. the AE has no prior: N(0, I) is a speck in its space, so draws repeat one digit", color=RED_, size=26)
        stat = VGroup(
            Text(f"{n} draws: {h_ae[top] / n:.0%} decode to a {top}", font_size=19, color=RED_).next_to(strips[0], DOWN, buff=0.1),
            Text(f"{n} draws: every digit {h_vae.min() / n:.0%}–{h_vae.max() / n:.0%}", font_size=19, color=GREEN_).next_to(strips[1], DOWN, buff=0.1))
        self.play(FadeIn(stat))
        self.wait(2)
        cap = swap_caption(self, cap, "a VAE latent is a space you can sample from and walk through", color=PURPLE_)
        self.wait(2)


# ----------------------------------------------------------------------------------------------------------------
class Reparameterization(Scene):
    """z = mu + sigma * eps: sampling inside the graph blocks the gradient; moved outside, gradients reach mu and sigma.
    Real numbers for L = (z - 2)^2, mu = 0.5, sigma = 1, eps = 0.8; then Monte-Carlo averages vs the exact gradient.
    Target: mm9 (VAE training)."""

    def construct(self):
        title(self, "The reparameterization trick", "toy loss L = (z − 2)²,  μ = 0.5,  σ = 1")
        mu0, sg0, eps0 = 0.5, 1.0, 0.8
        z0 = mu0 + sg0 * eps0
        dLdz = 2 * (z0 - 2)

        def node(tex, pos, color=INK, r=0.36):
            pos = np.array([pos[0], pos[1], 0.0])
            c = Circle(radius=r, color=color, stroke_width=3, fill_color=WHITE, fill_opacity=1).move_to(pos)
            return VGroup(c, MathTex(tex, font_size=34, color=color).move_to(pos))

        def box(tex, pos, color=INK, w=2.3):
            pos = np.array([pos[0], pos[1], 0.0])
            t = MathTex(tex, font_size=32, color=color)
            r = RoundedRectangle(corner_radius=0.12, width=max(w, t.width + 0.4), height=0.75, color=color,
                                 stroke_width=3, fill_color=WHITE, fill_opacity=1).move_to(pos)
            return VGroup(r, t.move_to(pos))

        Y = 0.85
        mu = node(r"\mu", [-5.6, Y + 0.65], BLUE_)
        sg = node(r"\sigma", [-5.6, Y - 0.65], BLUE_)
        samp = box(r"z \sim \mathcal N(\mu, \sigma^2)", [-2.2, Y], RED_, 2.9)
        z = node("z", [1.0, Y])
        L = box(r"L = (z-2)^2", [3.9, Y], INK, 2.4)
        e1 = Arrow(mu.get_right(), samp.get_left(), buff=0.08, color=INK, stroke_width=3)
        e2 = Arrow(sg.get_right(), samp.get_left(), buff=0.08, color=INK, stroke_width=3)
        e3 = Arrow(samp.get_right(), z.get_left(), buff=0.08, color=INK, stroke_width=3)
        e4 = Arrow(z.get_right(), L.get_left(), buff=0.08, color=INK, stroke_width=3)
        enc = Text("encoder outputs", font_size=18, color=BLUE_).rotate(PI / 2).next_to(VGroup(mu, sg), LEFT, buff=0.15)
        graphA = VGroup(mu, sg, samp, z, L, e1, e2, e3, e4)

        cap = caption("1. a VAE samples z from the encoder's Gaussian, then computes the loss")
        self.play(FadeIn(cap), FadeIn(graphA), FadeIn(enc))
        self.wait(0.8)

        g4 = Arrow(L.get_left() + 0.45 * DOWN, z.get_right() + 0.45 * DOWN, buff=0.05, color=ORANGE_, stroke_width=5)
        g4l = MathTex(rf"\tfrac{{\partial L}}{{\partial z}} = {dLdz:.1f}", font_size=28, color=ORANGE_).next_to(g4, DOWN, buff=0.08)
        g3 = Arrow(z.get_left() + 0.45 * DOWN, samp.get_right() + 0.45 * DOWN, buff=0.05, color=ORANGE_, stroke_width=5)
        block = Cross(scale_factor=0.3, stroke_color=RED_, stroke_width=8).move_to(samp.get_bottom() + 0.45 * DOWN + 1.0 * LEFT)
        blk_l = Text("random draw: no ∂z/∂μ, no ∂z/∂σ", font_size=22, color=RED_).next_to(block, DOWN, buff=0.18)
        cap = swap_caption(self, cap, "2. backprop reaches z, then hits the random draw: μ and σ get no gradient", color=RED_)
        self.play(GrowArrow(g4), FadeIn(g4l))
        self.play(GrowArrow(g3))
        self.play(FadeIn(block, scale=1.5), FadeIn(blk_l))
        self.wait(1.6)

        # rewire: eps outside the gradient path
        cap = swap_caption(self, cap, "3. reparameterize: draw ε ~ N(0, 1) outside, then z = μ + σ·ε is plain arithmetic")
        eps = box(r"\varepsilon \sim \mathcal N(0,1)", [-2.5, Y - 1.75], MUTED, 2.4)
        mul = node(r"\times", [-3.3, Y - 0.65], INK, 0.3)
        add = node("+", [-1.3, Y], INK, 0.3)
        f1 = Arrow(sg.get_right(), mul.get_left(), buff=0.06, color=INK, stroke_width=3)
        f2 = Arrow(eps.get_top(), mul.get_bottom(), buff=0.06, color=MUTED, stroke_width=3)
        f3 = Arrow(mul.get_right(), add.get_left(), buff=0.06, color=INK, stroke_width=3)
        f4 = Arrow(mu.get_right(), add.get_left(), buff=0.06, color=INK, stroke_width=3)
        f5 = Arrow(add.get_right(), z.get_left(), buff=0.08, color=INK, stroke_width=3)
        self.play(FadeOut(VGroup(g4, g4l, g3, block, blk_l, samp, e1, e2, e3)))
        self.play(FadeIn(eps, shift=0.2 * UP), FadeIn(mul), FadeIn(add), *[GrowArrow(a) for a in (f1, f2, f3, f4, f5)])
        vals = VGroup(
            MathTex(rf"\varepsilon = {eps0}", font_size=26, color=MUTED).next_to(eps, RIGHT, buff=0.15),
            MathTex(rf"z = {mu0} + {sg0:.0f}\cdot{eps0} = {z0:.1f}", font_size=28).next_to(z, UP, buff=0.15),
        )
        self.play(FadeIn(vals))
        self.wait(1.2)

        cap = swap_caption(self, cap, "4. now the gradient flows back to both: ∂z/∂μ = 1, ∂z/∂σ = ε", color=ORANGE_)
        gz = Arrow(L.get_left() + 0.45 * DOWN, z.get_right() + 0.45 * DOWN, buff=0.05, color=ORANGE_, stroke_width=5)
        gzl = MathTex(rf"\tfrac{{\partial L}}{{\partial z}} = 2(z-2) = {dLdz:.1f}", font_size=26, color=ORANGE_).next_to(gz, DOWN, buff=0.08)
        gmu = CurvedArrow(add.get_top() + 0.05 * UP, mu.get_right() + 0.1 * UP, angle=0.6, color=ORANGE_, stroke_width=5)
        gmul = MathTex(rf"\tfrac{{\partial L}}{{\partial \mu}} = {dLdz:.1f}\cdot 1 = {dLdz:.2f}", font_size=26,
                       color=ORANGE_).next_to(gmu, UP, buff=0.05)
        gsg = CurvedArrow(mul.get_left() + 0.05 * LEFT + 0.1 * DOWN, sg.get_bottom() + 0.05 * DOWN, angle=-0.9, color=ORANGE_, stroke_width=5)
        gsgl = MathTex(rf"\tfrac{{\partial L}}{{\partial \sigma}} = {dLdz:.1f}\cdot {eps0} = {dLdz * eps0:.2f}", font_size=26,
                       color=ORANGE_).move_to([-5.3, Y - 1.75, 0])
        self.play(GrowArrow(gz), FadeIn(gzl))
        self.play(Create(gmu), FadeIn(gmul))
        self.play(Create(gsg), FadeIn(gsgl))
        self.wait(1.8)

        # number line: fixed eps, moving mu and sigma moves the samples
        cap = swap_caption(self, cap, "5. with the ε's fixed, every sample moves smoothly as μ and σ change")
        nl = NumberLine(x_range=[-3, 5, 1], length=8.2, color=MUTED, include_numbers=True, font_size=22,
                        decimal_number_config={"color": MUTED, "num_decimal_places": 0}).move_to([-2.2, -2.65, 0])
        rng = np.random.default_rng(4)
        epss = np.concatenate([[eps0], rng.standard_normal(24)])
        mt, st = ValueTracker(mu0), ValueTracker(sg0)
        target = DashedLine(nl.n2p(2) + 0.05 * DOWN, nl.n2p(2) + 1.0 * UP, color=GREEN_, stroke_width=3)
        tl = MathTex("z=2", font_size=24, color=GREEN_).next_to(target.get_top(), RIGHT, buff=0.1)

        def dens():
            m, s = mt.get_value(), st.get_value()
            xs = np.linspace(-3, 5, 160)
            ys = np.exp(-0.5 * ((xs - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            pts = [nl.n2p(x) + UP * 1.4 * yy for x, yy in zip(xs, ys)]
            return polyline(pts, BLUE_, 3)

        def samples():
            m, s = mt.get_value(), st.get_value()
            g = VGroup(*[Dot(nl.n2p(np.clip(m + s * e, -3, 5)), radius=0.05, color=BLUE_, fill_opacity=0.45) for e in epss[1:]])
            g.add(Dot(nl.n2p(m + s * eps0), radius=0.09, color=ORANGE_))
            return g

        dcur, sdots = always_redraw(dens), always_redraw(samples)
        readout = always_redraw(lambda: Text(f"μ = {mt.get_value():.2f}   σ = {st.get_value():.2f}",
                                             font_size=24, color=BLUE_).move_to([-5.0, -1.75, 0]))
        self.play(Create(nl), FadeIn(dcur), FadeIn(sdots), FadeIn(readout), Create(target), FadeIn(tl))
        self.play(mt.animate.set_value(1.6), run_time=2)
        self.play(st.animate.set_value(0.45), run_time=1.6)
        self.play(mt.animate.set_value(mu0), st.animate.set_value(sg0), run_time=1.4)

        # Monte-Carlo average vs exact gradient of E[L] = (mu-2)^2 + sigma^2
        cap = swap_caption(self, cap, "6. one ε gives a noisy gradient; the average over ε is the exact gradient of E[L]")
        big = rng.standard_normal(10000)
        big[0] = eps0
        gm = 2 * (mu0 + sg0 * big - 2)
        gs = gm * big
        rows = [["samples", r"\partial/\partial\mu", r"\partial/\partial\sigma"]]
        for n in [1, 10, 100, 10000]:
            rows.append([f"{n}", f"{gm[:n].mean():+.2f}", f"{gs[:n].mean():+.2f}"])
        rows.append(["exact", f"{2 * (mu0 - 2):+.2f}", f"{2 * sg0:+.2f}"])
        tab = VGroup()
        for i, r in enumerate(rows):
            for j, c in enumerate(r):
                if i == 0 and j > 0:
                    t = MathTex(c, font_size=28, color=ORANGE_)
                elif i == 0:
                    t = Text(c, font_size=20, color=MUTED)
                else:
                    t = Text(c, font_size=22, color=PURPLE_ if i == len(rows) - 1 else INK,
                             weight=BOLD if i == len(rows) - 1 else NORMAL)
                tab.add(t.move_to([3.6 + 1.35 * j, -0.4 - 0.4 * i, 0]))
        rule = Line([2.9, -0.6, 0], [6.7, -0.6, 0], color=MUTED, stroke_width=1.5)
        rule2 = Line([2.9, -2.2, 0], [6.7, -2.2, 0], color=MUTED, stroke_width=1.5)
        self.play(FadeIn(tab[:3]), Create(rule))
        for i in range(1, len(rows)):
            if i == len(rows) - 1:
                self.play(Create(rule2), run_time=0.3)
            self.play(FadeIn(tab[3 * i:3 * i + 3], shift=0.1 * UP), run_time=0.5)
            self.wait(0.3)
        self.wait(1.2)
        cap = swap_caption(self, cap, "the randomness is an input, not a node: ordinary backprop trains μ and σ", color=PURPLE_)
        self.wait(2.2)


# ----------------------------------------------------------------------------------------------------------------
class GANDynamics(Scene):
    """A real 1-D GAN (MLP G and D, non-saturating loss, R1 penalty): the generator density chases the data,
    D flattens to 1/2; then a two-mode target where G collapses onto one mode. Target: new slide (GANs)."""

    def construct(self):
        d = np.load(DATA / "w08_gan.npz")
        grid = d["grid"]
        title(self, "GAN training in 1-D", "a real GAN: MLP generator and discriminator, densities from 4000 samples")
        ax = Axes(x_range=[-4.5, 4.5, 1], y_range=[0, 1.6, 0.5], x_length=11.5, y_length=4.2,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_tip": False},
                  x_axis_config={"numbers_to_include": range(-4, 5), "numbers_to_exclude": [], "font_size": 22,
                                 "decimal_number_config": {"color": MUTED, "num_decimal_places": 0}},
                  ).move_to([0, -0.6, 0])
        ax.y_axis.set_opacity(0)
        m = (grid >= -4.5) & (grid <= 4.5)
        gx = grid[m]
        YMAX = 1.6

        def curve(ys, color, width=5):
            ys = np.minimum(ys[m], YMAX)
            return polyline([ax.c2p(x, v) for x, v in zip(gx, ys)], color, width)

        def area(ys, color):
            ys = ys[m]
            pts = [ax.c2p(gx[0], 0)] + [ax.c2p(x, v) for x, v in zip(gx, ys)] + [ax.c2p(gx[-1], 0)]
            return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=0.22)

        pdf = lambda x, mu, s: np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        uni = pdf(grid, 1.5, 0.6)
        bi = 0.5 * pdf(grid, -2, 0.45) + 0.5 * pdf(grid, 2, 0.45)

        def legend():
            items = [(GREEN_, "data density"), (BLUE_, "generator density"), (PURPLE_, "D(x) = P(real)")]
            g = VGroup(*[VGroup(Line(ORIGIN, 0.45 * RIGHT, color=c, stroke_width=6), Text(t, font_size=20, color=c)).arrange(RIGHT, buff=0.12)
                         for c, t in items]).arrange(RIGHT, buff=0.5)
            return g.move_to([-2.2, 1.95, 0])

        snap = ValueTracker(0)
        cur = {"key": "uni"}

        def interp(name):
            arr, i = d[f"{cur['key']}_{name}"], snap.get_value()
            i0 = int(np.floor(i)); i1 = min(i0 + 1, len(arr) - 1); f = i - i0
            return (1 - f) * arr[i0] + f * arr[i1]

        gcur = always_redraw(lambda: curve(interp("g"), BLUE_, 5))
        dcur = always_redraw(lambda: curve(interp("d"), PURPLE_, 4))
        stepno = always_redraw(lambda: Text(f"training step {int(round(np.interp(snap.get_value(), np.arange(len(d[cur['key'] + '_s'])), d[cur['key'] + '_s'])))}",
                                            font_size=24, color=INK).move_to([4.9, 1.95, 0]))
        half = DashedLine(ax.c2p(-4.5, 0.5), ax.c2p(4.5, 0.5), color=PURPLE_, stroke_width=2, stroke_opacity=0.5)
        half_l = MathTex(r"\tfrac12", font_size=26, color=PURPLE_).next_to(ax.c2p(-4.5, 0.5), LEFT, buff=0.12)

        data_c, data_a = curve(uni, GREEN_, 4), area(uni, GREEN_)
        cap = caption("1. data: samples from an unknown density (here a Gaussian at x = 1.5)")
        self.play(FadeIn(cap), Create(ax))
        self.play(FadeIn(data_a), Create(data_c), FadeIn(legend()[0]))
        self.wait(0.6)
        lg = legend()
        cap = swap_caption(self, cap, "2. generator: noise z ~ N(0, 1) → MLP → x; its samples start far from the data")
        self.play(FadeIn(gcur), FadeIn(lg[1]), run_time=1)
        self.wait(0.6)
        cap = swap_caption(self, cap, "3. discriminator D(x): probability that x is real; untrained ≈ ½ everywhere")
        self.play(FadeIn(dcur), FadeIn(lg[2]), Create(half), FadeIn(half_l), FadeIn(stepno))
        self.wait(0.8)
        cap = swap_caption(self, cap, "4. D learns where the data is; G follows D's slope toward 'real'")
        n1 = len(d["uni_s"])
        k300 = int(np.searchsorted(d["uni_s"], 300))
        self.play(snap.animate.set_value(k300), run_time=7, rate_func=linear)
        cap = swap_caption(self, cap, "5. once G overlaps the data, D cannot tell them apart: D(x) → ½")
        self.play(snap.animate.set_value(n1 - 1), run_time=5, rate_func=linear)
        dfin = d["uni_d"][-1][np.abs(grid - 1.5) < 1.2]
        cap = swap_caption(self, cap, f"6. after 1200 steps D ∈ [{dfin.min():.2f}, {dfin.max():.2f}] over the data: equilibrium", color=PURPLE_)
        self.wait(1.8)

        # bimodal target
        cap = swap_caption(self, cap, "7. a harder target: two modes; the generator starts in between")
        cur["key"] = "bi"
        snap.set_value(0)
        data_c2, data_a2 = curve(bi, GREEN_, 4), area(bi, GREEN_)
        self.play(Transform(data_c, data_c2), Transform(data_a, data_a2), run_time=1.2)
        dx = grid[1] - grid[0]
        mass = always_redraw(lambda: VGroup(
            Text(f"generator mass  left {interp('g')[grid < 0].sum() * dx:.0%}", font_size=22, color=BLUE_),
            Text(f"right {interp('g')[grid >= 0].sum() * dx:.0%}", font_size=22, color=BLUE_)).arrange(RIGHT, buff=0.25)
            .move_to(ax.c2p(2.4, 1.45)))
        self.play(FadeIn(mass))
        self.wait(0.6)
        cap = swap_caption(self, cap, "8. mode collapse: G settles on one mode and stays there", color=RED_)
        self.play(snap.animate.set_value(len(d["bi_s"]) - 1), run_time=9, rate_func=lambda t: smooth(t) ** 0.6)
        cap = swap_caption(self, cap, "D ≈ 1 on the empty mode, but D is flat where G sits: no gradient pulls G across", color=RED_, size=26)
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------
class VQSnap(Scene):
    """Vector quantization in 2-D: encoder vectors snap to their nearest codebook entry; straight-through gradient;
    codebook loss pulls used codes toward their points, dead codes never move until re-initialized.
    Real k-means-style updates of the VQ-VAE codebook loss on fixed encoder outputs. Target: p9 (VQ-VAE)."""

    def construct(self):
        rng = np.random.default_rng(2)
        centers = np.array([[-2.0, 1.6], [1.8, 1.9], [2.2, -1.5], [-1.6, -1.9], [0.2, 0.1], [-2.6, -0.2]])
        sizes = [70, 60, 70, 60, 50, 50]
        Ze = np.concatenate([c + 0.38 * rng.standard_normal((n, 2)) for c, n in zip(centers, sizes)])
        K = 8
        E0 = np.random.default_rng(109).uniform(-3.8, 3.8, (K, 2))      # uniform init over the box

        def assign(E):
            return np.argmin(((Ze[:, None] - E[None]) ** 2).sum(-1), 1)

        def train(E, steps, lr=0.25):
            hist = [E.copy()]
            for _ in range(steps):
                a = assign(E)
                for k in range(K):
                    if (a == k).any():
                        # grad of ||sg(z_e) - e_k||^2 averaged over the code's points
                        E[k] += lr * (Ze[a == k].mean(0) - E[k])
                hist.append(E.copy())
            return hist

        title(self, "Vector quantization (VQ-VAE)", "an encoder output z_e is replaced by its nearest codebook vector")
        P = plane([-4, 4, 1], [-4, 4, 1], 5.0, 5.0, [-3.4, -0.45, 0])
        dots = VGroup(*[Dot(P.c2p(*z), radius=0.04, color=BLUE_, fill_opacity=0.65) for z in Ze])
        Et = [E0.copy()]

        def code_mobs(E, used):
            g = VGroup()
            for k, e in enumerate(E):
                sq = Square(0.22, color=ORANGE_ if used[k] else MUTED, fill_color=ORANGE_ if used[k] else WHITE,
                            fill_opacity=1, stroke_width=3).rotate(PI / 4).move_to(P.c2p(*e))
                g.add(sq)
            return g

        cap = caption("1. the encoder maps each input to a continuous vector z_e (420 vectors here)")
        self.play(FadeIn(cap), Create(P))
        self.play(LaggedStart(*[FadeIn(dt) for dt in dots], lag_ratio=0.003), run_time=1.5)
        self.wait(0.5)

        cap = swap_caption(self, cap, "2. a codebook of K = 8 learnable vectors e_k, initialized at random")
        used = np.bincount(assign(E0), minlength=K) > 0
        codes = code_mobs(E0, np.ones(K, bool))
        labels = VGroup(*[Text(f"{k + 1}", font_size=16, color=INK).next_to(c, UR, buff=0.02) for k, c in enumerate(codes)])
        self.play(LaggedStart(*[FadeIn(c, scale=2) for c in codes], lag_ratio=0.08), FadeIn(labels))
        self.wait(0.5)

        cap = swap_caption(self, cap, "3. quantize: z_q = e_k with k = argmin ‖z_e − e_k‖ (nearest code)")
        a0 = assign(E0)
        lines = VGroup(*[Line(P.c2p(*z), P.c2p(*E0[a]), color=MUTED, stroke_width=1, stroke_opacity=0.5) for z, a in zip(Ze, a0)])
        self.play(Create(lines), run_time=1.2)
        ghosts = dots.copy()
        self.play(*[g.animate.move_to(P.c2p(*E0[a])) for g, a in zip(ghosts, a0)], run_time=1.5)
        self.wait(0.6)
        self.play(FadeOut(ghosts), run_time=0.5)

        # usage bars (right side)
        def usage_bars(E):
            cnt = np.bincount(assign(E), minlength=K)
            p = cnt / cnt.sum()
            nz = p[p > 0]
            perp = np.exp(-(nz * np.log(nz)).sum())
            g = VGroup()
            base = np.array([1.9, -2.6, 0])
            for k in range(K):
                h = max(0.02, cnt[k] / 140 * 2.2)
                r = Rectangle(width=0.42, height=h, stroke_width=0, fill_color=ORANGE_ if cnt[k] else MUTED, fill_opacity=0.9)
                r.move_to(base + RIGHT * (k * 0.58) + UP * h / 2)
                g.add(r, Text(f"{cnt[k]}", font_size=17, color=INK if cnt[k] else RED_).next_to(r, UP, buff=0.05),
                      Text(f"{k + 1}", font_size=17, color=MUTED).move_to(base + RIGHT * (k * 0.58) + 0.22 * DOWN))
            g.add(Text(f"codes used: {(cnt > 0).sum()}/{K}    perplexity {perp:.1f}", font_size=22,
                       color=INK).move_to(base + RIGHT * 2.03 + UP * 2.85))
            return g

        # straight-through
        cap = swap_caption(self, cap, "4. argmin has no gradient: straight-through copies ∂L/∂z_q onto z_e")
        i = 15
        ze, zq = Ze[i], E0[a0[i]]
        hl = Circle(radius=0.12, color=BLUE_, stroke_width=4).move_to(P.c2p(*ze))
        g_dir = np.array([0.55, -0.85])
        gq = Arrow(P.c2p(*zq), P.c2p(*(zq + g_dir)), buff=0, color=RED_, stroke_width=6)
        f = MathTex(r"z_q = z_e + \mathrm{sg}[\,z_q - z_e\,]", font_size=34).move_to([3.95, 1.7, 0])
        f2 = Text("forward: value of z_q     backward: identity to z_e", font_size=19, color=MUTED).next_to(f, DOWN, buff=0.18)
        f3 = Text("(sg = stop-gradient; arrow: an example decoder gradient)", font_size=17, color=MUTED).next_to(f2, DOWN, buff=0.1)
        self.play(Create(hl), FadeIn(f), FadeIn(f2), FadeIn(f3))
        self.play(GrowArrow(gq))
        ge = gq.copy()
        self.play(ge.animate.shift(P.c2p(*ze) - P.c2p(*zq)), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(VGroup(hl, gq, ge, f2, f3)))

        cap = swap_caption(self, cap, "5. codebook usage: codes that win no vectors are dead", color=RED_)
        ub = usage_bars(E0)
        self.play(FadeIn(ub), codes.animate.become(code_mobs(E0, used)))
        self.wait(1.5)

        cap = swap_caption(self, cap, "6. codebook loss: each used code moves toward the mean of its vectors")
        f_new = MathTex(r"L = \|x-\hat x\|^2 + \|\mathrm{sg}[z_e]-e\|^2 + \beta\|z_e-\mathrm{sg}[e]\|^2", font_size=28).move_to(f)
        self.play(Transform(f, f_new))
        hist = train(E0.copy(), 12)
        for E in hist[1:]:
            a = assign(E)
            new_lines = VGroup(*[Line(P.c2p(*z), P.c2p(*E[k]), color=MUTED, stroke_width=1, stroke_opacity=0.5) for z, k in zip(Ze, a)])
            u = np.bincount(a, minlength=K) > 0
            self.play(codes.animate.become(code_mobs(E, u)), labels.animate.become(
                VGroup(*[Text(f"{k + 1}", font_size=16, color=INK).next_to(P.c2p(*e), UR, buff=0.1) for k, e in enumerate(E)])),
                lines.animate.become(new_lines), ub.animate.become(usage_bars(E)), run_time=0.35)
        E1 = hist[-1]
        self.wait(0.8)
        dead = np.flatnonzero(np.bincount(assign(E1), minlength=K) == 0)
        cap = swap_caption(self, cap, f"7. dead codes get no gradient: {len(dead)} of {K} never moved", color=RED_)
        self.play(*[Indicate(codes[k], color=RED_, scale_factor=1.8) for k in dead])
        self.wait(1.2)

        cap = swap_caption(self, cap, "8. fix: restart dead codes at random encoder outputs, keep training")
        E2 = E1.copy()
        for k in dead:
            E2[k] = Ze[rng.integers(len(Ze))] + 0.05 * rng.standard_normal(2)
        hist2 = [E2] + train(E2.copy(), 12)[1:]
        for E in hist2:
            a = assign(E)
            new_lines = VGroup(*[Line(P.c2p(*z), P.c2p(*E[k]), color=MUTED, stroke_width=1, stroke_opacity=0.5) for z, k in zip(Ze, a)])
            u = np.bincount(a, minlength=K) > 0
            self.play(codes.animate.become(code_mobs(E, u)), labels.animate.become(
                VGroup(*[Text(f"{k + 1}", font_size=16, color=INK).next_to(P.c2p(*e), UR, buff=0.1) for k, e in enumerate(E)])),
                lines.animate.become(new_lines), ub.animate.become(usage_bars(E)), run_time=0.35)
        cap = swap_caption(self, cap, "all codes in use: each input becomes one of K discrete tokens", color=PURPLE_)
        self.wait(2.2)
