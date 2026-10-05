"""Week 1–2 (foundations: perceptron, MLPs, backprop, optimization, initialization). Render one scene:
    manim -qh w01_foundations.py PerceptronLearning
"""
from style import *


def halfplane_poly(w, b, box):
    """Clip the box [x0,x1]x[y0,y1] to the half-plane w·p + b > 0 (Sutherland–Hodgman, one edge)."""
    x0, x1, y0, y1 = box
    pts = [np.array(p, float) for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    f = lambda p: w @ p + b
    out = []
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        fp, fq = f(p), f(q)
        if fp > 0:
            out.append(p)
        if (fp > 0) != (fq > 0):
            out.append(p + (q - p) * fp / (fp - fq))
    return out


def line_in_box(w, b, box):
    """Endpoints of the line w·p + b = 0 inside the box, or None."""
    x0, x1, y0, y1 = box
    pts = []
    corners = [np.array(p, float) for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        fp, fq = w @ p + b, w @ q + b
        if (fp > 0) != (fq > 0):
            pts.append(p + (q - p) * fp / (fp - fq))
    return pts[:2] if len(pts) >= 2 else None


# =====================================================================================================
class PerceptronLearning(Scene):
    """Perceptron rule Δw = η(y−ŷ)x on AND (each mistake turns/shifts the boundary, converges) and on XOR
    (the boundary cycles forever). Target: p1 slides 17–18, 29."""

    def construct(self):
        title(self, "The perceptron learning rule", "one point at a time; only mistakes change the weights")
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
        box = (-0.6, 1.6, -0.6, 1.6)
        ax = Axes(x_range=[-0.6, 1.6, 1], y_range=[-0.6, 1.6, 1], x_length=4.6, y_length=4.6,
                  axis_config={"stroke_color": MUTED, "include_ticks": True, "tip_length": 0.18},
                  ).shift(3.3 * LEFT + 0.35 * DOWN)
        axl = VGroup(MathTex("x_1", font_size=30).next_to(ax.x_axis, RIGHT, buff=0.1),
                     MathTex("x_2", font_size=30).next_to(ax.y_axis, UP, buff=0.1))
        frame = Rectangle(width=4.6, height=4.6, stroke_color=MUTED, stroke_width=1).move_to(ax.c2p(0.5, 0.5))

        w0, w1, bb = ValueTracker(-0.8), ValueTracker(-0.8), ValueTracker(-0.3)
        W = lambda: np.array([w0.get_value(), w1.get_value()])
        eta = 0.5

        def shade():
            poly = halfplane_poly(W(), bb.get_value(), box)
            if len(poly) < 3:
                return VMobject()
            return Polygon(*[ax.c2p(*p) for p in poly], stroke_width=0, fill_color=PURPLE_, fill_opacity=0.12)

        def bline():
            seg = line_in_box(W(), bb.get_value(), box)
            if seg is None:
                return VMobject()
            return Line(ax.c2p(*seg[0]), ax.c2p(*seg[1]), color=PURPLE_, stroke_width=5)

        def warrow():
            w, b = W(), bb.get_value()
            n = np.linalg.norm(w) + 1e-9
            seg = line_in_box(w, b, box)
            base = (seg[0] + seg[1]) / 2 if seg is not None else np.array([0.5, 0.5])
            tip = base + 0.45 * w / n
            return Arrow(ax.c2p(*base), ax.c2p(*tip), buff=0, color=PURPLE_, stroke_width=5,
                         max_tip_length_to_length_ratio=0.35)

        def dots(y):
            g = VGroup()
            for p, t in zip(X, y):
                if t == 1:
                    g.add(Dot(ax.c2p(*p), radius=0.14, color=GREEN_))
                else:
                    g.add(Square(0.26, color=ORANGE_, fill_color=WHITE, fill_opacity=1, stroke_width=5).move_to(ax.c2p(*p)))
            return g

        legend = VGroup(VGroup(Dot(radius=0.1, color=GREEN_), Text("y = 1", font_size=20)).arrange(RIGHT, buff=0.1),
                        VGroup(Square(0.2, color=ORANGE_, stroke_width=4), Text("y = 0", font_size=20)).arrange(RIGHT, buff=0.1),
                        VGroup(Square(0.2, stroke_width=0, fill_color=PURPLE_, fill_opacity=0.25),
                               Text("predicted ŷ = 1", font_size=20)).arrange(RIGHT, buff=0.1)
                        ).arrange(RIGHT, buff=0.4).move_to([3.0, -2.55, 0])

        y_and = np.array([0, 0, 0, 1])
        pts = dots(y_and)
        sh, bl, wa = always_redraw(shade), always_redraw(bline), always_redraw(warrow)

        # right panel: rule and live weights
        rx = 3.0
        rule = VGroup(MathTex(r"\hat y = [\,w\cdot x + b > 0\,]", font_size=36),
                      MathTex(r"w \leftarrow w + \eta\,(y-\hat y)\,x", font_size=36),
                      MathTex(r"b \leftarrow b + \eta\,(y-\hat y)", font_size=36),
                      MathTex(r"\eta = 0.5", font_size=30, color=MUTED)).arrange(DOWN, buff=0.2).move_to([rx, 1.65, 0])

        def wtext():
            w, b = W(), bb.get_value()
            return Text(f"w = ({w[0]:+.2f}, {w[1]:+.2f})    b = {b:+.2f}".replace("-", "−"), font_size=26,
                        color=PURPLE_).move_to([rx, 0.3, 0])
        wt = always_redraw(wtext)
        epoch_lbl = Text("epoch 1", font_size=24, color=MUTED).move_to([rx, -1.4, 0])

        cap = caption("a perceptron predicts 1 on the side the weight vector w points to")
        self.play(Create(ax), FadeIn(axl), FadeIn(frame), FadeIn(pts), FadeIn(legend), FadeIn(cap))
        self.play(FadeIn(sh), Create(bl), FadeIn(wa), FadeIn(rule), FadeIn(wt))
        self.wait(1.2)

        def run(y, epochs, slow=True, err_lbl=None):
            nonlocal cap, epoch_lbl
            ring = Circle(0.3, color=INK, stroke_width=4).move_to(ax.c2p(*X[0])).set_opacity(0)
            self.add(ring)
            history = []
            for ep in range(epochs):
                if err_lbl is None:
                    new_lbl = Text(f"epoch {ep + 1}", font_size=24, color=MUTED).move_to(epoch_lbl)
                    self.play(Transform(epoch_lbl, new_lbl), run_time=0.25)
                errs = 0
                for p, t in zip(X, y):
                    w, b = W(), bb.get_value()
                    yh = int(w @ p + b > 0)
                    self.play(ring.animate.move_to(ax.c2p(*p)).set_stroke(INK, opacity=1), run_time=0.25 if slow else 0.15)
                    if yh == t:
                        self.play(ring.animate.set_stroke(GREEN_), run_time=0.15 if slow else 0.1)
                        continue
                    errs += 1
                    d = t - yh
                    self.play(ring.animate.set_stroke(RED_), run_time=0.2)
                    calc = MathTex(rf"x=({p[0]:.0f},{p[1]:.0f}),\ y-\hat y = {d:+d}\ \Rightarrow\ "
                                   rf"\Delta w = ({eta * d * p[0] + 0:+.1f},{eta * d * p[1] + 0:+.1f}),\ \Delta b = {eta * d:+.1f}",
                                   font_size=26, color=RED_)
                    txt = ("mistake: w turns toward x, b goes up" if d > 0
                           else "mistake: w turns away from x, b goes down")
                    calc = VGroup(calc, Text(txt, font_size=20, color=RED_)).arrange(DOWN, buff=0.12).move_to([rx, -0.5, 0])
                    self.play(FadeIn(calc), run_time=0.3 if slow else 0.15)
                    self.play(w0.animate.increment_value(eta * d * p[0]), w1.animate.increment_value(eta * d * p[1]),
                              bb.animate.increment_value(eta * d), run_time=0.9 if slow else 0.45)
                    self.wait(0.25 if slow else 0.1)
                    self.play(FadeOut(calc), run_time=0.2 if slow else 0.1)
                history.append(errs)
                if err_lbl is not None:
                    err_lbl(ep, errs)
                if errs == 0:
                    break
            self.play(FadeOut(ring), run_time=0.2)
            return history

        cap = swap_caption(self, cap, "AND: visit the points in order; only a mistake moves the boundary")
        hist = run(y_and, 6)
        cap = swap_caption(self, cap, f"AND is linearly separable: after {len(hist)} epochs an error-free pass, learning stops",
                           color=GREEN_)
        self.wait(1.5)

        # XOR
        y_xor = np.array([0, 1, 1, 0])
        new_pts = dots(y_xor)
        cap = swap_caption(self, cap, "XOR: green and orange sit on opposite corners — no line separates them")
        self.play(*[Transform(pts[i], new_pts[i]) for i in range(4)])
        self.play(w0.animate.set_value(-0.8), w1.animate.set_value(-0.8), bb.animate.set_value(-0.3))
        self.wait(0.8)
        errs_txt = VGroup()
        def err_lbl(ep, e):
            t = Text(f"epoch {ep + 1}: {e} mistakes", font_size=22, color=RED_)
            if len(errs_txt):
                t.next_to(errs_txt[-1], DOWN, buff=0.1, aligned_edge=LEFT)
            else:
                t.move_to([rx, -1.35, 0])
            errs_txt.add(t)
            self.add(t)
        cap = swap_caption(self, cap, "the same rule on XOR: every epoch still makes mistakes")
        self.remove(epoch_lbl)
        run(y_xor, 3, slow=False, err_lbl=err_lbl)
        cap = swap_caption(self, cap, "the boundary cycles forever: one perceptron cannot represent XOR", color=RED_)
        self.wait(2.5)


# =====================================================================================================
class SpaceWarping(Scene):
    """One hidden layer h = ReLU(Wx + b) warps and folds the plane so XOR becomes linearly separable.
    Target: p3 slides 22–25."""

    def construct(self):
        title(self, "A hidden layer warps space", "h = ReLU(Wx + b) with 2 hidden units, applied to XOR")
        rng = np.random.default_rng(4)
        centers = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float)
        labels = np.array([0, 0, 1, 1])          # XOR: 1 when the signs differ
        P = np.concatenate([c + rng.normal(0, 0.13, (22, 2)) for c in centers])
        Y = np.repeat(labels, 22)
        # W squashes the (1,1) diagonal (eigenvalue 0.2) and stretches the (1,-1) diagonal (eigenvalue 1.5)
        W = np.array([[0.85, -0.65], [-0.65, 0.85]])
        b = np.array([-0.5, -0.5])

        unit = 0.95
        zoom = ValueTracker(1.0)
        origin = np.array([-2.6, 0.0, 0])
        to_s = lambda p: origin + unit * zoom.get_value() * np.array([p[0], p[1], 0])
        # fixed reference axes (the h1/h2 axes after the map)
        ref = always_redraw(lambda: VGroup(
            Line(origin + [-2.9 * unit, 0, 0], origin + [2.9 * unit, 0, 0], color=MUTED, stroke_width=2),
            Line(origin + [0, -2.6 * unit, 0], origin + [0, 2.4 * unit, 0], color=MUTED, stroke_width=2)))

        stage = ValueTracker(0.0)

        def phi(p):
            s = stage.get_value()
            if s <= 1:
                return (1 - s) * p + s * (W @ p + b)
            L = W @ p + b
            u = min(s - 1, 1)
            return (1 - u) * L + u * np.maximum(L, 0)

        lim = 1.4
        ts = np.linspace(-lim, lim, 81)
        grid_src = []
        for c in np.arange(-lim, lim + 1e-6, 0.35):
            grid_src.append(np.c_[np.full_like(ts, c), ts])
            grid_src.append(np.c_[ts, np.full_like(ts, c)])

        def grid():
            g = VGroup()
            for k, line in enumerate(grid_src):
                m = VMobject(stroke_color=BLUE_, stroke_width=1.6, stroke_opacity=0.35)
                m.set_points_as_corners([to_s(phi(p)) for p in line])
                g.add(m)
            return g

        def points():
            g = VGroup()
            for p, y in zip(P, Y):
                q = to_s(phi(p))
                if y == 1:
                    g.add(Dot(q, radius=0.075, color=GREEN_))
                else:
                    g.add(Square(0.13, color=ORANGE_, stroke_width=3, fill_color=WHITE, fill_opacity=1).move_to(q))
            return g

        gr, pts = always_redraw(grid), always_redraw(points)
        axn = VGroup(MathTex("x_1", font_size=28, color=MUTED).move_to(origin + unit * np.array([3.15, 0.25, 0])),
                     MathTex("x_2", font_size=28, color=MUTED).move_to(origin + unit * np.array([0.35, 2.5, 0])))
        cap = caption("XOR: green where the signs differ — no straight line separates the classes")
        self.play(Create(ref), FadeIn(axn), FadeIn(gr), FadeIn(pts), FadeIn(cap))
        # one failed linear attempt
        tryl = DashedLine(to_s((-2.0, -1.4)), to_s((2.0, 1.0)), color=RED_, stroke_width=4)
        self.play(Create(tryl), run_time=0.8)
        self.play(Rotate(tryl, PI / 2.2, about_point=to_s((0, 0))), run_time=1.2)
        self.play(FadeOut(tryl), run_time=0.4)

        rx = 4.0
        mats = VGroup(
            MathTex(r"W = \begin{bmatrix} 0.85 & -0.65 \\ -0.65 & 0.85 \end{bmatrix},\ b = \begin{bmatrix} -0.5 \\ -0.5 \end{bmatrix}",
                    font_size=30),
            MathTex(r"z = Wx + b", font_size=36, color=BLUE_),
            MathTex(r"h = \mathrm{ReLU}(z) = \max(z, 0)", font_size=36, color=PURPLE_),
            MathTex(r"\hat y = [\,h_1 + h_2 > 0.5\,]", font_size=36, color=GREEN_),
        ).arrange(DOWN, buff=0.38).move_to([rx, 0.3, 0])
        self.play(FadeIn(mats[0]))
        cap = swap_caption(self, cap, "1. the linear part Wx + b stretches, squashes and shifts the whole plane")
        self.play(FadeIn(mats[1]))
        self.play(stage.animate.set_value(1.0), run_time=3.5, rate_func=smooth)
        newl = VGroup(MathTex("z_1", font_size=28, color=MUTED).move_to(axn[0]),
                      MathTex("z_2", font_size=28, color=MUTED).move_to(axn[1]))
        self.play(Transform(axn, newl))
        cap = swap_caption(self, cap, "still not separable: a linear map keeps straight lines straight")
        self.wait(1.4)

        cap = swap_caption(self, cap, "2. ReLU folds everything negative onto the axes", color=PURPLE_)
        self.play(FadeIn(mats[2]))
        self.play(stage.animate.set_value(2.0), run_time=3.5, rate_func=smooth)
        newl = VGroup(MathTex("h_1", font_size=28, color=MUTED).move_to(axn[0]),
                      MathTex("h_2", font_size=28, color=MUTED).move_to(axn[1]))
        self.play(Transform(axn, newl))
        cap = swap_caption(self, cap, "both orange clusters land on the corner (0, 0); green lands on the axes")
        self.play(zoom.animate.set_value(1.5), run_time=1.2)
        self.wait(1.0)

        sep = Line(to_s((-0.4, 0.9)), to_s((0.9, -0.4)), color=GREEN_, stroke_width=5)
        half = Polygon(to_s((-0.4, 0.9)), to_s((0.9, -0.4)), to_s((1.7, -0.4)), to_s((1.7, 1.55)), to_s((-0.4, 1.55)),
                       stroke_width=0, fill_color=GREEN_, fill_opacity=0.08)
        cap = swap_caption(self, cap, "3. now one straight line separates the classes", color=GREEN_)
        self.play(FadeIn(mats[3]), Create(sep), FadeIn(half))
        self.wait(1.2)
        cap = swap_caption(self, cap, "a hidden layer learns coordinates in which the problem becomes linear", color=PURPLE_)
        self.wait(2.5)


# =====================================================================================================
class BackpropFlow(Scene):
    """Forward values flow left→right, gradients flow right→left on f = σ(w0·x0 + w1·x1 + w2); each gate multiplies the
    upstream gradient by its local derivative. Target: p3 slides 30–38."""

    def construct(self):
        title(self, "Backpropagation on a computational graph", "f = σ(w0·x0 + w1·x1 + w2)")
        v = dict(w0=2.0, x0=-1.0, w1=-3.0, x1=-2.0, w2=-3.0)
        a, bm = v["w0"] * v["x0"], v["w1"] * v["x1"]
        c = a + bm
        d = c + v["w2"]
        f = 1 / (1 + np.exp(-d))
        g = {"f": 1.0}
        g["d"] = f * (1 - f) * g["f"]
        g["c"], g["w2"] = g["d"], g["d"]
        g["a"], g["b"] = g["c"], g["c"]
        g["w0"], g["x0"] = v["x0"] * g["a"], v["w0"] * g["a"]
        g["w1"], g["x1"] = v["x1"] * g["b"], v["w1"] * g["b"]

        pos = {"w0": [-5.0, 2.0], "x0": [-5.0, 1.05], "w1": [-5.0, 0.1], "x1": [-5.0, -0.85], "w2": [-5.0, -1.8],
               "m0": [-2.9, 1.5], "m1": [-2.9, -0.4], "s1": [-0.7, 0.55], "s2": [1.5, -0.2], "sg": [3.6, -0.2],
               "out": [5.7, -0.2]}
        pos = {k: np.array(p + [0]) for k, p in pos.items()}
        nodes = {}
        for k in ["w0", "x0", "w1", "x1", "w2"]:
            nodes[k] = VGroup(RoundedRectangle(corner_radius=0.1, width=0.85, height=0.55, stroke_color=BLUE_,
                                               fill_color=SOFT, fill_opacity=1, stroke_width=2),
                              MathTex(f"{k[0]}_{k[1]}", font_size=30, color=BLUE_)).move_to(pos[k])
        for k, s in [("m0", r"\times"), ("m1", r"\times"), ("s1", "+"), ("s2", "+"), ("sg", r"\sigma")]:
            nodes[k] = VGroup(Circle(0.36, stroke_color=INK, fill_color=WHITE, fill_opacity=1, stroke_width=3),
                              MathTex(s, font_size=36)).move_to(pos[k])
        nodes["out"] = MathTex("f", font_size=40).move_to(pos["out"])

        edges = [("w0", "m0", "w0", "w0"), ("x0", "m0", "x0", "x0"), ("w1", "m1", "w1", "w1"), ("x1", "m1", "x1", "x1"),
                 ("m0", "s1", "a", "a"), ("m1", "s1", "b", "b"), ("s1", "s2", "c", "c"), ("w2", "s2", "w2", "w2"),
                 ("s2", "sg", "d", "d"), ("sg", "out", "f", "f")]
        val = dict(v, a=a, b=bm, c=c, d=d, f=f)
        lines = {}
        for s, t, vk, gk in edges:
            p0 = nodes[s].get_right() if s in v else nodes[s][0].get_boundary_point(normalize(pos[t] - pos[s]))
            p1 = nodes[t][0].get_boundary_point(normalize(pos[s] - pos[t])) if t != "out" else nodes[t].get_left() + 0.1 * LEFT
            lines[vk] = Line(p0, p1, color=MUTED, stroke_width=3)

        def lab(k, txt, color, side):
            if k in v:                       # leaves: write the numbers left of the box
                m = MathTex(txt, font_size=28, color=color)
                m.next_to(nodes[k], LEFT, buff=0.15).shift((0.17 if side > 0 else -0.17) * UP)
                return m
            ln = lines[k]
            n = normalize(rotate_vector(ln.get_unit_vector(), PI / 2))
            if n[1] < 0:
                n = -n
            m = MathTex(txt, font_size=28, color=color)
            m.move_to(ln.point_from_proportion(0.5) + (0.28 if side > 0 else -0.28) * n)
            return m

        key = VGroup(VGroup(Line(ORIGIN, 0.4 * RIGHT, color=GREEN_, stroke_width=4), Text("forward value", font_size=22, color=GREEN_)).arrange(RIGHT, buff=0.12),
                     VGroup(Line(ORIGIN, 0.4 * RIGHT, color=RED_, stroke_width=4), MathTex(r"\partial f/\partial(\cdot)\ \text{gradient}", font_size=28, color=RED_)).arrange(RIGHT, buff=0.12)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([4.6, 1.9, 0])
        self.play(*[FadeIn(n) for n in nodes.values()], *[Create(l) for l in lines.values()], FadeIn(key[0]), run_time=1.2)
        cap = caption("a neuron as a graph of simple gates: two ×, two +, one σ")
        self.play(FadeIn(cap))
        self.wait(1)

        # forward
        cap = swap_caption(self, cap, "forward pass: each gate computes its output from its inputs", color=GREEN_)
        fwd = {}
        order = [["w0", "x0", "w1", "x1", "w2"], ["a", "b"], ["c"], ["d"], ["f"]]
        for grp in order:
            anims = []
            for k in grp:
                fwd[k] = lab(k, f"{val[k]:.2f}" if k in ("f",) else f"{val[k]:g}", GREEN_, +1)
                dot = Dot(lines[k].get_start(), radius=0.07, color=GREEN_)
                anims.append(Succession(MoveAlongPath(dot, lines[k], run_time=0.6), FadeOut(dot, run_time=0.05)))
                anims.append(FadeIn(fwd[k], run_time=0.6))
                anims.append(lines[k].animate.set_color(GREEN_))
            self.play(*anims, run_time=0.7)
        self.wait(0.8)

        panel_y = -2.6
        def show_calc(tex, old=None):
            m = MathTex(tex, font_size=32, color=RED_).move_to([1.0, panel_y, 0])
            if old is None:
                self.play(FadeIn(m), run_time=0.4)
            else:
                self.play(FadeOut(old), FadeIn(m), run_time=0.4)
            return m

        def back(keys, run=0.7):
            anims = []
            for k in keys:
                lab_k = lab(k, f"{g[k]:.2f}", RED_, -1)
                dot = Dot(lines[k].get_end(), radius=0.07, color=RED_)
                anims += [Succession(MoveAlongPath(dot, lines[k].copy().reverse_points(), run_time=run),
                                     FadeOut(dot, run_time=0.05)), FadeIn(lab_k, run_time=run)]
            self.play(*anims, run_time=run + 0.05)

        cap = swap_caption(self, cap, "backward pass: start from ∂f/∂f = 1 and walk the graph right to left", color=RED_)
        self.play(FadeIn(key[1]), run_time=0.4)
        back(["f"])
        calc = show_calc(r"\text{downstream} = \text{local gradient} \times \text{upstream}")
        self.wait(1.2)
        cap = swap_caption(self, cap, "σ gate: local gradient σ(d)(1 − σ(d)) = 0.73 · 0.27", color=RED_)
        calc = show_calc(rf"\frac{{\partial f}}{{\partial d}} = \sigma(1-\sigma)\cdot 1 = {f:.2f}\cdot{1 - f:.2f} = {g['d']:.2f}", calc)
        back(["d"])
        self.wait(1.2)
        cap = swap_caption(self, cap, "+ gate: local gradient 1 — it copies the upstream gradient to every input", color=RED_)
        calc = show_calc(rf"\frac{{\partial f}}{{\partial c}} = \frac{{\partial f}}{{\partial w_2}} = 1\cdot {g['d']:.2f}", calc)
        back(["c", "w2"])
        self.wait(0.6)
        back(["a", "b"])
        self.wait(1.0)
        cap = swap_caption(self, cap, "× gate: the gradient for one input is the other input's value × upstream", color=RED_)
        calc = show_calc(rf"\frac{{\partial f}}{{\partial w_0}} = x_0\cdot{g['a']:.2f} = {g['w0']:.2f},\quad"
                         rf"\frac{{\partial f}}{{\partial x_0}} = w_0\cdot{g['a']:.2f} = {g['x0']:.2f}", calc)
        back(["w0", "x0"])
        self.wait(1.2)
        calc = show_calc(rf"\frac{{\partial f}}{{\partial w_1}} = x_1\cdot{g['b']:.2f} = {g['w1']:.2f},\quad"
                         rf"\frac{{\partial f}}{{\partial x_1}} = w_1\cdot{g['b']:.2f} = {g['x1']:.2f}", calc)
        back(["w1", "x1"])
        self.wait(1.2)
        self.play(FadeOut(calc))
        cap = swap_caption(self, cap, "every gate needs only its local derivative: downstream = local × upstream", color=PURPLE_)
        self.wait(2.5)


# =====================================================================================================
class OptimizerPaths(Scene):
    """SGD vs momentum vs Adam on an ill-conditioned quadratic valley: live paths on the contours and the loss curves.
    Each optimizer uses its best learning rate from a sweep (40 steps, path must stay in view). Target: p2 slides 45–57."""

    def construct(self):
        title(self, "Optimizers in a narrow valley",
              "quadratic bowl, curvature 0.2 along and 10 across: condition number 50")
        A = np.diag([0.2, 10.0])
        fval = lambda p: 0.5 * p @ A @ p
        grad = lambda p: A @ p
        p0 = np.array([-4.5, 1.5])
        N = 40
        xr, yr = (-5.2, 5.2), (-2.6, 2.6)

        def sgd(lr):
            p = p0.copy(); P = [p]
            for _ in range(N):
                p = p - lr * grad(p); P.append(p)
            return np.array(P)

        def mom(lr, beta):
            p = p0.copy(); vel = np.zeros(2); P = [p]
            for _ in range(N):
                vel = beta * vel + grad(p); p = p - lr * vel; P.append(p)
            return np.array(P)

        def adam(lr, b1=0.9, b2=0.999):
            p = p0.copy(); m = np.zeros(2); s = np.zeros(2); P = [p]
            for t in range(1, N + 1):
                gg = grad(p); m = b1 * m + (1 - b1) * gg; s = b2 * s + (1 - b2) * gg ** 2
                p = p - lr * (m / (1 - b1 ** t)) / (np.sqrt(s / (1 - b2 ** t)) + 1e-8); P.append(p)
            return np.array(P)

        inview = lambda P: np.all(np.abs(P[:, 0]) < xr[1] - 0.1) and np.all(np.abs(P[:, 1]) < yr[1] - 0.1)
        def best(cands):
            ok = [(hp, P) for hp, P in cands if inview(P)]
            return min(ok, key=lambda c: fval(c[1][-1]))
        lrs = np.round(np.arange(0.01, 1.0, 0.01), 2)
        s_hp, s_P = best([(lr, sgd(lr)) for lr in lrs if lr < 0.2])
        m_hp, m_P = best([((lr, b), mom(lr, b)) for lr in lrs if lr < 0.2 for b in (0.5, 0.6, 0.7, 0.8, 0.9)])
        a_hp, a_P = best([(lr, adam(lr)) for lr in lrs])
        runs = [("SGD", f"lr {s_hp:g}", s_P, BLUE_), ("momentum", f"lr {m_hp[0]:g}, β {m_hp[1]:g}", m_P, ORANGE_),
                ("Adam", f"lr {a_hp:g}", a_P, PURPLE_)]

        # contour background as an image
        wpx, hpx = 520, 260
        xs, ys = np.linspace(*xr, wpx), np.linspace(*yr, hpx)[::-1]
        XX, YY = np.meshgrid(xs, ys)
        F = 0.5 * (A[0, 0] * XX ** 2 + A[1, 1] * YY ** 2)
        lv = np.log2(F + 1e-3)
        band = (np.floor(lv) % 2).astype(float)
        edge = np.abs(lv - np.round(lv)) < 0.06
        shade = np.clip(1 - np.log1p(F) / np.log1p(F.max()), 0, 1)
        base = np.array([255, 255, 255], float)
        tint = np.array([219, 234, 254], float)
        rgb = base[None, None] * (1 - shade[..., None] * 0.9) + tint[None, None] * shade[..., None] * 0.9
        rgb = rgb - band[..., None] * 6
        rgb[edge] = [170, 182, 200]
        img = ImageMobject(rgb.clip(0, 255).astype(np.uint8))
        img.width = 7.4
        img.move_to([-2.85, -0.3, 0])
        frame = Rectangle(width=img.width, height=img.height, stroke_color=MUTED, stroke_width=1).move_to(img)
        sx = img.width / (xr[1] - xr[0])
        to_s = lambda p: img.get_center() + np.array([p[0] * sx, p[1] * sx, 0])
        star = Star(5, outer_radius=0.16, color=GREEN_, fill_opacity=1).move_to(to_s((0, 0)))
        start = Dot(to_s(p0), radius=0.08, color=INK)

        # loss panel: y = log10(loss) + 6
        lax = Axes(x_range=[0, N, 10], y_range=[0, 8, 2], x_length=4.3, y_length=2.7,
                   axis_config={"stroke_color": MUTED, "include_tip": False, "font_size": 20},
                   y_axis_config={"numbers_to_include": [0, 2, 4, 6, 8]},
                   x_axis_config={"numbers_to_include": [0, 10, 20, 30, 40]}).move_to([3.95, 0.35, 0])
        for num in lax.x_axis.numbers:
            num.set_color(INK)
        for num in lax.y_axis.numbers:
            val = int(round(lax.y_axis.p2n(num.get_center()))) - 6
            num.become(MathTex(f"10^{{{val}}}", font_size=22).move_to(num).shift(0.08 * LEFT))
        lhdr = Text("loss (log scale) vs step", font_size=22).next_to(lax, UP, buff=0.1)
        ly = lambda v: np.clip(np.log10(v) + 6, 0, 8)

        cap = caption("a long narrow valley: steep across, almost flat along")
        self.play(FadeIn(img), FadeIn(frame), FadeIn(star), FadeIn(start), FadeIn(cap))
        self.play(Create(lax), FadeIn(lhdr))
        self.wait(1.2)

        texts = {
            "SGD": "SGD: the gradient points across the valley — it zigzags, then crawls along the floor",
            "momentum": "momentum: velocity builds up along the valley while the zigzags cancel out",
            "Adam": "Adam: divides each coordinate by its gradient RMS — both axes take similar steps",
        }
        legend = VGroup()
        for name, hp, P, col in runs:
            t = ValueTracker(0)

            def path(P=P, col=col, t=t):
                k = t.get_value()
                i = int(np.floor(k))
                pts = [to_s(p) for p in P[:i + 1]]
                if i < N:
                    pts.append(to_s(P[i] + (k - i) * (P[i + 1] - P[i])))
                m = VMobject(stroke_color=col, stroke_width=3.5)
                m.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + 1e-4 * RIGHT])
                head = Dot(pts[-1], radius=0.07, color=col)
                return VGroup(m, head)

            def curve(P=P, col=col, t=t):
                k = t.get_value()
                i = int(np.floor(k))
                L = [ly(fval(p)) for p in P[:i + 2]]
                pts = [lax.c2p(j, L[j]) for j in range(min(i + 1, N + 1))]
                if i < N:
                    pts.append(lax.c2p(k, L[i] + (k - i) * (L[i + 1] - L[i])))
                m = VMobject(stroke_color=col, stroke_width=3)
                m.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + 1e-4 * RIGHT])
                return m

            pm, cm = always_redraw(path), always_redraw(curve)
            cap = swap_caption(self, cap, texts[name], color=col, size=26)
            self.add(pm, cm)
            self.play(t.animate.set_value(N), run_time=6, rate_func=linear)
            pm.clear_updaters(); cm.clear_updaters()
            if name != "Adam":
                self.play(pm[0].animate.set_stroke(opacity=0.35), pm[1].animate.set_opacity(0.35), run_time=0.4)
            fl = fval(P[-1])
            entry = VGroup(Line(ORIGIN, 0.35 * RIGHT, color=col, stroke_width=5),
                           Text(f"{name} ({hp}): {fl:.1e}", font_size=19, color=col)).arrange(RIGHT, buff=0.12)
            if len(legend):
                entry.next_to(legend[-1], DOWN, buff=0.12, aligned_edge=LEFT)
            else:
                entry.next_to(lax, DOWN, buff=0.6).align_to(lax, LEFT).shift(0.1 * LEFT)
                lg_hdr = Text("best swept learning rate: loss after 40 steps", font_size=17, color=MUTED
                              ).next_to(entry, UP, buff=0.08, aligned_edge=LEFT)
                self.play(FadeIn(lg_hdr), run_time=0.3)
            legend.add(entry)
            self.play(FadeIn(entry), run_time=0.4)
            self.wait(0.8)
        cap = swap_caption(self, cap, "Adam's fixed step size keeps it circling the minimum unless the learning rate decays",
                           color=PURPLE_, size=26)
        self.wait(2)
        cap = swap_caption(self, cap, "ill-conditioning, not the size of the gradient, is what makes plain SGD slow",
                           color=INK, size=26)
        self.wait(2.5)


# =====================================================================================================
class InitSignal(Scene):
    """Activations of a 10-layer ReLU MLP for weight std too small / too large / Kaiming √(2/n): the signal vanishes,
    explodes, or keeps its scale. Target: T01 §5."""

    def construct(self):
        title(self, "Initialization sets the signal scale", "10-layer ReLU MLP, 256 units, 1000 Gaussian inputs, no bias")
        n, depth = 256, 10
        rng = np.random.default_rng(0)
        Ws = [rng.standard_normal((n, n)) * np.sqrt(2 / n) for _ in range(depth)]
        h = rng.standard_normal((1000, n))
        H = []
        for Wl in Ws:
            h = np.maximum(h @ Wl, 0)
            H.append(h)
        # ReLU nets without bias are positively homogeneous: scaling every W by g scales layer l by g^l
        kstd = np.sqrt(2 / n)
        std = ValueTracker(0.05)
        g = lambda: std.get_value() / kstd
        show = [1, 2, 4, 6, 8, 10]
        nbins, hmax = 16, 4.0
        edges = np.linspace(0, hmax, nbins + 1)
        pos_vals = [Hl[Hl > 0] for Hl in H]
        panel_w, ph = 1.75, 1.45
        xs = np.linspace(-5.4, 5.4, len(show))
        y0 = 0.3

        frames = VGroup()
        for j, l in enumerate(show):
            base = Line([xs[j] - panel_w / 2, y0, 0], [xs[j] + panel_w / 2, y0, 0], color=MUTED, stroke_width=2)
            lbl = Text(f"layer {l}", font_size=20).next_to(base, DOWN, buff=0.12)
            ticks = VGroup(Text("0", font_size=14, color=MUTED).move_to([xs[j] - panel_w / 2, y0 - 0.15, 0]),
                           Text("4+", font_size=14, color=MUTED).move_to([xs[j] + panel_w / 2, y0 - 0.15, 0]))
            frames.add(VGroup(base, lbl, ticks))

        def hists():
            gg = g()
            grp = VGroup()
            bw = panel_w / (nbins + 1)
            for j, l in enumerate(show):
                vals = pos_vals[l - 1] * gg ** l
                cnt, _ = np.histogram(np.clip(vals, 0, hmax * 2), bins=edges)
                over = (vals > hmax).sum()
                tot = len(vals)
                for k in range(nbins + 1):
                    c = cnt[k] if k < nbins else over
                    hgt = max(0.01, min(c / tot * 4.0, 1.0) * ph)
                    col = PURPLE_ if k < nbins else RED_
                    grp.add(Rectangle(width=bw * 0.9, height=hgt, stroke_width=0, fill_color=col, fill_opacity=0.9)
                            .move_to([xs[j] - panel_w / 2 + (k + 0.5) * bw, y0 + hgt / 2, 0]))
            return grp

        # std per layer, log axis
        # y = log10(std) + 3, so the x-axis sits at the bottom (std = 1e-3)
        sax = Axes(x_range=[0, 10, 1], y_range=[0, 5, 1], x_length=6.0, y_length=1.8,
                   axis_config={"stroke_color": MUTED, "include_tip": False, "font_size": 18},
                   x_axis_config={"numbers_to_include": list(range(1, 11))},
                   y_axis_config={"numbers_to_include": [1, 3, 5]}).move_to([-0.6, -1.85, 0])
        for num in sax.x_axis.numbers:
            num.set_color(INK)
        for num in sax.y_axis.numbers:
            val = int(round(sax.y_axis.p2n(num.get_center()))) - 3
            num.become(MathTex(f"10^{{{val}}}", font_size=20).move_to(num).shift(0.08 * LEFT))
        sl = Text("activation std per layer (log scale)", font_size=20).next_to(sax, UP, buff=0.12)
        base_std = np.array([Hl.std() for Hl in H])
        ref_line = DashedLine(sax.c2p(0, np.log10(base_std.mean()) + 3), sax.c2p(10, np.log10(base_std.mean()) + 3),
                              color=GREEN_, stroke_width=2)

        def std_curve():
            gg = g()
            vals = np.clip(np.log10(base_std * gg ** np.arange(1, 11)) + 3, 0, 5)
            pts = [sax.c2p(l + 1, v) for l, v in enumerate(vals)]
            m = VMobject(stroke_color=PURPLE_, stroke_width=4).set_points_as_corners(pts)
            return VGroup(m, *[Dot(p, radius=0.05, color=PURPLE_) for p in pts])

        def std_label():  # uses kai_lbl defined below
            s = std.get_value()
            col = GREEN_ if abs(s - kstd) < 0.004 else (BLUE_ if s < kstd else RED_)
            return VGroup(Text(f"weight std = {s:.3f}", font_size=26, color=col),
                          kai_lbl.copy()
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([4.7, -1.75, 0])

        kai_lbl = MathTex(rf"\text{{Kaiming }}\sqrt{{2/n}} = {kstd:.3f}", font_size=26, color=MUTED)
        hg, sc, slab = always_redraw(hists), always_redraw(std_curve), always_redraw(std_label)
        hdr = Text("histogram of the nonzero activations (red bar: values beyond 4)", font_size=20, color=MUTED
                   ).move_to([0, 2.15, 0])
        cap = caption("weights too small: every layer shrinks the signal by the same factor, and it dies", color=BLUE_)
        self.play(FadeIn(frames), FadeIn(hg), FadeIn(hdr), FadeIn(cap))
        self.play(Create(sax), FadeIn(sl), FadeIn(sc), FadeIn(slab))
        self.wait(2.5)
        cap = swap_caption(self, cap, "raise the weight scale: the deep layers wake up layer by layer")
        self.play(std.animate.set_value(0.13), run_time=6, rate_func=linear)
        cap = swap_caption(self, cap, "weights too large: the scale grows geometrically and the deep layers explode", color=RED_)
        self.wait(2.5)
        cap = swap_caption(self, cap, f"Kaiming: Var(w) = 2/n keeps the scale constant through depth", color=GREEN_)
        self.play(std.animate.set_value(kstd), run_time=3)
        self.play(Create(ref_line))
        self.wait(1.5)
        cap = swap_caption(self, cap, "the same argument keeps gradients from vanishing or exploding going backward",
                           color=GREEN_, size=26)
        self.wait(2.5)
