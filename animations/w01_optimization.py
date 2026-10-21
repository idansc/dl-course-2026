"""Week 1, lecture 2 (optimization): the optimizer clips on real data with real optimizer runs.
Every number and path comes from code/optim_real.py (diabetes data, bundled with scikit-learn) and code/muon_demo.py
(a wrapper around code/muon_cifar.py, CIFAR-10); the scenes import them, so `python code/optim_real.py` and
`python code/muon_demo.py` print exactly what the clips show. Code table: code/README.md.
Render one scene:
    manim -qh --disable_caching -o IllConditioning w01_optimization.py IllConditioning
Supersedes the synthetic OptimizerPaths (w01_foundations.py) and the third-party GIFs / screen recordings in the p2 deck.
"""
import sys
from pathlib import Path

import contourpy

from style import *

sys.path.insert(0, str(Path(__file__).parent / "code"))     # `code` is also a stdlib module: import the files directly
import optim_real as opt                                    # noqa: E402

OPT_COLOR = {"SGD": INK, "SGD+Momentum": BLUE_, "Nesterov": GREEN_, "AdaGrad": ORANGE_, "RMSProp": RED_,
             "Adam": PURPLE_}
CONTOUR = "#C3CAD5"
GAPS = 10.0 ** (-np.arange(9) / 2)       # contour lines at L - L* = gap0 * 10^(-k/2): 1, 0.32, 0.1, 0.032, 0.01, ...


# =====================================================================================================
# helpers
def mtext(markup, size=24, color=INK):
    """Arial text with Pango markup (<sub>, <b>): Arial has no glyphs for subscript digits."""
    return MarkupText(markup, font=FONT, font_size=size, color=color)


def head(scene, text, sub):
    t = Text(text, font_size=40, weight=BOLD).to_corner(UL, buff=0.45)
    s = mtext(sub, 24, MUTED).next_to(t, DOWN, aligned_edge=LEFT, buff=0.15)
    scene.play(FadeIn(VGroup(t, s), shift=0.2 * DOWN), run_time=0.8)
    return VGroup(t, s)


def cap(markup, color=INK, size=28):
    return mtext(markup, size, color).to_edge(DOWN, buff=0.45)


def swap(scene, old, markup, color=INK, size=28):
    new = cap(markup, color, size)
    if old is None:
        scene.play(FadeIn(new, shift=0.1 * UP), run_time=0.6)
    else:
        scene.play(FadeOut(old, shift=0.1 * UP), FadeIn(new, shift=0.1 * UP), run_time=0.6)
    return new


class Window:
    """Maps parameter coordinates (h, v) in [h0, h1] x [v0, v1] onto a screen rectangle. Give the width or the height
    (one scale for both axes, so contour lines keep their true shape) or both (separate scales)."""

    def __init__(self, hr, vr, center, width=None, height=None):
        self.hr, self.vr = (float(hr[0]), float(hr[1])), (float(vr[0]), float(vr[1]))
        dh, dv = self.hr[1] - self.hr[0], self.vr[1] - self.vr[0]
        self.sx = width / dh if width else height / dv
        self.sy = height / dv if height else self.sx
        self.width, self.height = self.sx * dh, self.sy * dv
        self.center = np.array([center[0], center[1], 0.0])
        self.box = (*self.hr, *self.vr)

    def __call__(self, h, v):
        return self.center + np.array([(h - sum(self.hr) / 2) * self.sx, (v - sum(self.vr) / 2) * self.sy, 0.0])

    def w(self, w):
        """bowl weights (w1, w2): w2 (s5) horizontal, w1 (bmi) vertical"""
        return self(w[1], w[0])

    def frame(self, color=MUTED, width=1.2):
        return Rectangle(width=self.width, height=self.height, stroke_color=color, stroke_width=width).move_to(self.center)


def clip_segment(a, b, box):
    """Liang-Barsky: the part of segment ab inside box = (h0, h1, v0, v1), or None."""
    h0, h1, v0, v1 = box
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - h0), (d[0], h1 - a[0]), (-d[1], a[1] - v0), (d[1], v1 - a[1])):
        if p == 0:
            if q < 0:
                return None
        elif p < 0:
            t0 = max(t0, q / p)
        else:
            t1 = min(t1, q / p)
    return None if t0 > t1 else (a + t0 * d, a + t1 * d)


def clip_polyline(P, box):
    """Pieces of the polyline P (n x 2) inside box, cut exactly at the border."""
    pieces, cur = [], []
    for a, b in zip(P[:-1], P[1:]):
        seg = clip_segment(np.asarray(a, float), np.asarray(b, float), box)
        if seg is None:
            if len(cur) > 1:
                pieces.append(cur)
            cur = []
            continue
        p, q = seg
        if not cur or np.linalg.norm(cur[-1] - p) > 1e-12:
            if len(cur) > 1:
                pieces.append(cur)
            cur = [p]
        cur.append(q)
        if np.linalg.norm(q - b) > 1e-12:          # left the box
            pieces.append(cur)
            cur = []
    if len(cur) > 1:
        pieces.append(cur)
    return pieces


def polyline(pts, color, width=2.0, opacity=1.0):
    pts = [np.asarray(p, float) for p in pts]
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + 1e-4 * RIGHT])
    return m


def bowl_contours(win, H, w_star, gaps, color=CONTOUR, width=1.6, n=480):
    """True contour lines of the quadratic L = L* + (w - w*)^T H (w - w*) / 2 at the given gaps L - L*."""
    evals, evecs = np.linalg.eigh(H)
    th = np.linspace(0, 2 * np.pi, n)
    grp = VGroup()
    for g in gaps:
        W = np.asarray(w_star)[:, None] + evecs @ (np.sqrt(2 * g / evals)[:, None] * np.stack([np.cos(th), np.sin(th)]))
        hv = np.stack([W[1], W[0]], 1)                   # h = w2, v = w1
        for piece in clip_polyline(hv, win.box):
            grp.add(polyline([win(*p) for p in piece], color, width))
    return grp


def grid_contours(win, xs, ys, Z, levels, colors, width=1.5):
    cg = contourpy.contour_generator(xs, ys, Z)
    grp = VGroup()
    for lv, col in zip(levels, colors):
        for line in cg.lines(lv):
            for piece in clip_polyline(line, win.box):
                grp.add(polyline([win(*p) for p in piece], col, width))
    return grp


class Trace(VGroup):
    """A path revealed step by step: animate trace.t from 0 to len(pts) - 1 (fractional values interpolate)."""

    def __init__(self, pts, color, width=3.0, dot_r=0.0, head_r=0.07, opacity=1.0):
        super().__init__()
        self.pts = [np.asarray(p, float) for p in pts]
        self.n = len(self.pts) - 1
        self.t = ValueTracker(0.0)
        self.template = polyline(self.pts, color)
        self.line = polyline([self.pts[0]], color, width, opacity)
        self.dots = VGroup(*[Dot(p, radius=dot_r, color=color).set_opacity(0) for p in self.pts]) if dot_r else VGroup()
        self.head = Dot(self.pts[0], radius=head_r, color=color) if head_r else VGroup()
        self.add(self.line, self.dots, self.head)
        self.line.add_updater(self._line)
        if dot_r:
            self.dots.add_updater(self._dots)
        if head_r:
            self.head.add_updater(lambda d: d.move_to(self.at(self.t.get_value())))

    def at(self, k):
        if self.n == 0:
            return self.pts[0]
        k = float(np.clip(k, 0, self.n))
        i = min(int(k), self.n - 1)
        return self.pts[i] + (k - i) * (self.pts[i + 1] - self.pts[i])

    def _line(self, m):
        a = self.t.get_value() / max(self.n, 1)
        if a <= 1e-6:
            m.set_points_as_corners([self.pts[0], self.pts[0] + 1e-4 * RIGHT])
        else:
            m.pointwise_become_partial(self.template, 0, min(a, 1.0))

    def _dots(self, g):
        k = self.t.get_value()
        for i, d in enumerate(g):
            d.set_opacity(1.0 if i <= k + 1e-6 else 0.0)

    def freeze(self):
        for m in (self.line, self.dots, self.head):
            m.clear_updaters()
        return self


def run_steps(scene, traces, stops, times, rate=linear):
    """Advance all traces through the step counts in `stops`, one play per stop."""
    for s, rt in zip(stops, times):
        scene.play(*[tr.t.animate.set_value(min(s, tr.n)) for tr in traces], run_time=rt, rate_func=rate)


def step_counter(tracker, pos, color=INK, size=22, fmt="step {}"):
    return always_redraw(lambda: Text(fmt.format(int(tracker.get_value() + 1e-6)), font_size=size, color=color)
                         .move_to(pos, aligned_edge=LEFT))


def star(p, color=INK, r=0.13):
    return Star(5, outer_radius=r, color=color, fill_opacity=1, stroke_width=0).move_to(p)


class LogPlot:
    """Loss (log10) vs step axes; x ticks every x_step, y labels 10^k at the given exponents."""

    def __init__(self, center, width, height, n_steps, x_step, y_range, y_labels, header=None):
        self.y_range = y_range                         # in log10 units; the axes run from 0 so the x-axis is at the bottom
        self.ax = Axes(x_range=[0, n_steps, x_step], y_range=[0, y_range[1] - y_range[0], 1], x_length=width,
                       y_length=height, tips=False,
                       axis_config={"stroke_color": MUTED, "stroke_width": 2, "include_tip": False},
                       x_axis_config={"numbers_to_include": list(range(x_step, n_steps + 1, x_step)), "font_size": 20})
        self.ax.move_to([center[0], center[1], 0])
        for num in self.ax.x_axis.numbers:
            num.set_color(INK)
        self.ylab = VGroup(*[MathTex(f"10^{{{k}}}", font_size=22).next_to(self.ax.c2p(0, k - y_range[0]), LEFT,
                                                                          buff=0.12) for k in y_labels])
        self.xlab = Text("step", font_size=20, color=MUTED).next_to(self.ax.x_axis.get_right(), DOWN, buff=0.32)
        self.group = VGroup(self.ax, self.ylab, self.xlab)
        if header:
            self.header = mtext(header, 22).next_to(self.ax, UP, buff=0.18)
            self.group.add(self.header)

    def pts(self, values):
        lv = np.clip(np.log10(np.maximum(np.asarray(values, float), 1e-300)), *self.y_range) - self.y_range[0]
        return [self.ax.c2p(i, v) for i, v in enumerate(lv)]


def fmt_e(x):
    """1.3e-05 -> '1.3·10<sup>−5</sup>' (Pango markup)"""
    m, e = f"{x:.1e}".split("e")
    e = int(e)
    return m if e == 0 else f"{m}·10<sup>{'−' if e < 0 else ''}{abs(e)}</sup>"


# =====================================================================================================
class LearningRateRegimes(Scene):
    """One weight, s = w·bmi on the diabetes data: L(w) is an exact parabola with curvature a = mean(bmi²) = 19.48.
    GD from w = 0 with lr = 0.1/a, 1/a, 1.9/a, 2.02/a: slow, one step, oscillating, diverging (threshold 2/a = 0.1027).
    Numbers: optim_real.learning_rate_regimes(). Target: p2 "Learning rate behavior" (replaces the gbhat.com GIF)."""

    def construct(self):
        r = opt.learning_rate_regimes()
        head(self, "Learning rate: four regimes",
             "diabetes data, 442 patients · s = w·bmi (bmi in kg/m², centred) · L(w) = ½·mean((s − y)²)")
        a, ws, Ls = r["a"], r["w_star"], r["L_star"]
        L = lambda w: Ls + 0.5 * a * (w - ws) ** 2
        xr, yr = (-10.0, 30.0), (1700.0, 6100.0)
        cols = [BLUE_, GREEN_, ORANGE_, RED_]
        xs = [-5.1, -1.7, 1.7, 5.1]
        wins, curves, bases, ticks, headers, traces, factors = [], VGroup(), VGroup(), VGroup(), [], [], []
        for k, run in enumerate(r["runs"]):
            win = Window(xr, yr, center=(xs[k], -0.05), width=3.0, height=2.75)
            wins.append(win)
            curves.add(ParametricFunction(lambda u, win=win: win(u, L(u)), t_range=[xr[0], xr[1]], color=MUTED,
                                          stroke_width=3))
            bases.add(Line(win(xr[0], yr[0]), win(xr[1], yr[0]), color=MUTED, stroke_width=1.5))
            for wv, lab in ((0.0, "0"), (ws, "w*")):
                ticks.add(Line(win(wv, yr[0]), win(wv, yr[0]) + 0.08 * DOWN, color=MUTED, stroke_width=1.5),
                          Text(lab, font_size=18, color=MUTED).next_to(win(wv, yr[0]), DOWN, buff=0.12))
            col = cols[k]
            h1 = Text(f"lr = {run['factor']:g} / a", font_size=28, color=col, weight=BOLD)
            h2 = Text(f"= {run['lr']:.4f}", font_size=24, color=col)
            headers.append(VGroup(h1, h2).arrange(DOWN, buff=0.1).next_to(win(sum(xr) / 2, yr[1]), UP, buff=0.12))
            traces.append(Trace([win(w, l) for w, l in zip(run["path"], run["loss"])], col, width=2.2, dot_r=0.035,
                                head_r=0.07))
            factors.append(Text(f"×{run['contraction']:.2f} per step".replace("-", "−"), font_size=22, color=col)
                           .next_to(win(sum(xr) / 2, yr[0]), DOWN, buff=0.55))
        starts = VGroup(*[Dot(win(r["w0"], L(r["w0"])), radius=0.07, color=INK) for win in wins])

        c = swap(self, None, f"L(w) is a parabola: curvature a = mean(bmi²) = {a:.1f}")
        self.play(Create(curves), FadeIn(bases), FadeIn(ticks), run_time=1.4)
        self.play(FadeIn(starts), run_time=0.4)
        self.wait(1.0)
        texts = ["small steps: a slow descent", "lands exactly on w* in one step", "overshoots every step, but converges",
                 "overshoots more every step: diverges"]
        times = [[3.2], [1.0, 0.4], [3.6], [3.6]]
        stops = [[30], [1, 30], [30], [30]]
        for k in range(4):
            c = swap(self, c, f"lr = {r['runs'][k]['factor']:g}/a: " + texts[k], color=cols[k])
            self.play(FadeIn(headers[k], shift=0.1 * DOWN), FadeOut(starts[k]), run_time=0.4)
            self.add(traces[k])
            run_steps(self, [traces[k]], stops[k], times[k])
            traces[k].freeze()
            self.wait(0.5)
        c = swap(self, c, "each step multiplies the distance to w* by 1 − lr·a")
        self.play(LaggedStart(*[FadeIn(f, shift=0.1 * UP) for f in factors], lag_ratio=0.25), run_time=1.4)
        self.wait(1.6)
        c = swap(self, c, f"it converges only if lr &lt; 2/a = {r['threshold']:.4f}")
        self.wait(2.8)


# =====================================================================================================
class IllConditioning(Scene):
    """Problem #1. GD on the raw-centred (bmi, s5) bowl: curvatures 19.53 and 0.22 (kappa 89.8); GD with its best fixed
    lr 2/(l_min + l_max) zigzags across the valley and crawls along it: 104 steps to 1% of the loss gap. Standardized
    features: kappa 2.6, the same rule needs 3 steps. Numbers: optim_real.ill_conditioning(). Target: p2 "Problem #1"."""

    def construct(self):
        r = opt.ill_conditioning()
        R, S = r["raw"], r["standardized"]
        head(self, "Problem #1: ill-conditioning",
             "diabetes data, 442 patients · s = w<sub>1</sub>·bmi + w<sub>2</sub>·s5 (centred) · "
             "L(w) = ½·mean((s − y)²)")
        winR = Window((-4, 70), (-3, 23), center=(-2.45, 0.0), width=8.2)
        winS = Window((-8, 52), (-8, 52), center=(4.85, 0.0), width=3.3)
        parts = {}
        for key, d, win in (("raw", R, winR), ("std", S, winS)):
            gap0 = d["gap"][0]
            p = {}
            p["frame"] = win.frame()
            p["contours"] = bowl_contours(win, d["H"], d["w_star"], gap0 * GAPS[~np.isclose(GAPS, r["target"])])
            p["target"] = bowl_contours(win, d["H"], d["w_star"], [r["target"] * gap0], color=GREEN_, width=3)
            p["star"] = star(win.w(d["w_star"]))
            p["start"] = Dot(win.w([0, 0]), radius=0.07, color=INK)
            p["hlab"] = mtext("w<sub>2</sub> (s5)", 20, MUTED).next_to(p["frame"], DOWN, buff=0.1).align_to(p["frame"], RIGHT)
            p["vlab"] = mtext("w<sub>1</sub> (bmi)", 20, MUTED).rotate(PI / 2).next_to(p["frame"], LEFT, buff=0.1) \
                .align_to(p["frame"], UP)
            path = d["path"][: d["steps"] + 1]
            p["trace"] = Trace([win.w(q) for q in path], BLUE_, width=2.2, dot_r=0.03, head_r=0.075)
            parts[key] = p
        hdrR = mtext(f"raw features:  κ = {R['kappa']:.1f}", 24).next_to(winR.frame(), UP, buff=0.15) \
            .align_to(winR.frame(), LEFT)
        hdrS = mtext(f"standardized:  κ = {S['kappa']:.1f}", 24).next_to(winS.frame(), UP, buff=0.15) \
            .align_to(winS.frame(), LEFT)
        tlab = mtext("1% of the loss gap", 18, GREEN_).next_to(winR.w(R["w_star"]), DOWN, buff=0.55)

        P = parts["raw"]
        c = swap(self, None, f"curvature {R['eig'][1]:.1f} across the valley, {R['eig'][0]:.2f} along it")
        self.play(FadeIn(P["frame"]), Create(P["contours"]), FadeIn(hdrR), FadeIn(P["hlab"]), FadeIn(P["vlab"]),
                  run_time=1.6)
        self.play(FadeIn(P["start"]), FadeIn(P["star"]), Create(P["target"]), FadeIn(tlab), run_time=0.8)
        self.wait(1.0)
        c = swap(self, c, f"GD, best fixed lr ({R['lr']:.3f}): zigzags across, crawls along")
        cnt = step_counter(P["trace"].t, winR.frame().get_corner(UR) + np.array([-1.25, 0.27, 0]), BLUE_)
        self.add(P["trace"], cnt)
        run_steps(self, [P["trace"]], [10, R["steps"]], [3.6, 4.2])
        P["trace"].freeze()
        cnt.clear_updaters()
        resR = mtext(f"<b>{R['steps']} steps</b> to 1% of the gap", 24, BLUE_).next_to(P["frame"], DOWN, buff=0.45) \
            .align_to(P["frame"], LEFT)
        self.play(FadeIn(resR, shift=0.1 * UP), run_time=0.5)
        self.wait(0.8)

        Q = parts["std"]
        c = swap(self, c, f"standardize each feature: curvatures {S['eig'][1]:.2f} and {S['eig'][0]:.2f}")
        self.play(FadeIn(Q["frame"]), Create(Q["contours"]), FadeIn(hdrS), FadeIn(Q["hlab"]), FadeIn(Q["vlab"]),
                  run_time=1.6)
        self.play(FadeIn(Q["start"]), FadeIn(Q["star"]), Create(Q["target"]), run_time=0.8)
        self.wait(0.5)
        c = swap(self, c, f"the same rule (lr = {S['lr']:.1f}) needs {S['steps']} steps")
        self.add(Q["trace"])
        run_steps(self, [Q["trace"]], [1, 2, 3], [0.9, 0.9, 0.9])
        Q["trace"].freeze()
        resS = mtext(f"<b>{S['steps']} steps</b>", 24, BLUE_).next_to(Q["frame"], DOWN, buff=0.45) \
            .align_to(Q["frame"], LEFT)
        self.play(FadeIn(resS, shift=0.1 * UP), run_time=0.5)
        self.wait(1.0)
        c = swap(self, c, f"κ = largest / smallest curvature: {R['kappa']:.1f} → {S['kappa']:.1f}")
        self.wait(2.8)


# =====================================================================================================
class NoisyGradients(Scene):
    """Problem #3. Standardized bowl, minibatch SGD (lr 0.05, 150 steps, same start and seed) with B = 442 (full batch),
    16 and 1: the paths wobble around the full-batch path; the full-data loss gap settles at a floor that grows as B
    shrinks. Numbers: optim_real.noisy_gradients(). Target: p2 "Problem #3" (replaces the screen recording)."""

    def construct(self):
        r = opt.noisy_gradients()
        head(self, "Problem #3: noisy gradients",
             f"diabetes data, 442 patients · standardized bmi and s5 · minibatch SGD, lr = {r['lr']}")
        win = Window((-6, 44), (-6, 58), center=(-3.3, -0.35), height=4.6)
        frame = win.frame()
        gap0 = r["runs"][0]["gap"][0]
        contours = bowl_contours(win, r["H"], r["w_star"], gap0 * GAPS)
        st = star(win.w(r["w_star"]))
        start = Dot(win.w([0, 0]), radius=0.07, color=INK)
        hlab = mtext("w<sub>2</sub> (s5)", 20, MUTED).next_to(frame, DOWN, buff=0.1).align_to(frame, RIGHT)
        vlab = mtext("w<sub>1</sub> (bmi)", 20, MUTED).rotate(PI / 2).next_to(frame, LEFT, buff=0.1).align_to(frame, UP)
        lp = LogPlot(center=(3.85, 0.4), width=5.0, height=2.6, n_steps=r["num_steps"], x_step=50, y_range=(-4, 4),
                     y_labels=[-4, -2, 0, 2, 4], header="L − L* on all 442 patients")
        cols = {442: INK, 16: PURPLE_, 1: ORANGE_}
        texts = {442: "B = 442, the full batch: the exact gradient",
                 16: "B = 16: a noisy estimate, the path wobbles",
                 1: "B = 1: one patient per step, a random walk"}
        legend_hdr = Text("mean L − L*, last 50 steps", font_size=19, color=MUTED)
        legend_hdr.move_to([lp.ax.get_left()[0], -1.55, 0], aligned_edge=LEFT)
        rows = []
        for i, run in enumerate(r["runs"]):
            B = run["batch_size"]
            val = f"{run['gap_last50']:.3f}" if run["gap_last50"] < 0.1 else f"{run['gap_last50']:.1f}"
            y = -1.92 - 0.36 * i
            key = VGroup(Line(ORIGIN, 0.35 * RIGHT, color=cols[B], stroke_width=5),
                         Text(f"B = {B}", font_size=21, color=cols[B])).arrange(RIGHT, buff=0.15)
            key.move_to([lp.ax.get_left()[0], y, 0], aligned_edge=LEFT)
            num = Text(val, font_size=21, color=cols[B]).move_to([lp.ax.get_left()[0] + 3.4, y, 0], aligned_edge=RIGHT)
            rows.append(VGroup(key, num))

        c = swap(self, None, "each step: the gradient of B random patients")
        self.play(FadeIn(frame), Create(contours), FadeIn(hlab), FadeIn(vlab), run_time=1.4)
        self.play(FadeIn(start), FadeIn(st), Create(lp.ax), FadeIn(lp.ylab), FadeIn(lp.xlab), FadeIn(lp.header),
                  run_time=0.9)
        self.wait(0.6)
        for i, run in enumerate(r["runs"]):
            B = run["batch_size"]
            c = swap(self, c, texts[B], color=cols[B] if B != 442 else INK)
            tr = Trace([win.w(q) for q in run["path"]], cols[B], width=2.0 if B != 442 else 3.0, head_r=0.07)
            lt = Trace(lp.pts(run["gap"]), cols[B], width=2.0, head_r=0.04)
            self.add(tr, lt)
            run_steps(self, [tr, lt], [r["num_steps"]], [4.6])
            tr.freeze(), lt.freeze()
            self.play(FadeIn(rows[i]) if i else AnimationGroup(FadeIn(legend_hdr), FadeIn(rows[i])), run_time=0.5)
        c = swap(self, c, "smaller batch: noisier steps, higher loss floor")
        self.wait(2.8)


# =====================================================================================================
class SaddlePoint(Scene):
    """Problem #2. One hidden unit, s = w2·tanh(w1·x), standardized bmi -> standardized progression: the real loss
    surface has a saddle at w = 0 (zero gradient, Hessian eigenvalues ±mean(x·y) = ±0.586) and two mirror-image minima.
    From (0.05, 0.02), GD (lr 0.1) stalls on the plateau; momentum (rho 0.9) escapes: halfway down the loss at step 52
    vs 16. Numbers: optim_real.saddle_point(). Target: p2 "Problem #2"."""

    def construct(self):
        r = opt.saddle_point()
        head(self, "Problem #2: saddle points",
             "one hidden unit: s = w<sub>2</sub>·tanh(w<sub>1</sub>·x) · x = bmi, y = progression (standardized), "
             "442 patients")
        win = Window((-1.6, 1.6), (-4.4, 4.4), center=(-3.75, -0.25), width=4.6, height=4.9)
        frame = win.frame()
        levels = [0.3285, 0.331, 0.336, 0.345, 0.36, 0.38, 0.41, 0.45, 0.49, 0.51, 0.56, 0.65, 0.8, 1.05, 1.5, 2.2, 3.3,
                  5.0]
        lc = [interpolate_color(ManimColor(BLUE_), ManimColor(CONTOUR), min(1.0, i / 9)) if lv < 0.5 else CONTOUR
              for i, lv in enumerate(levels)]
        contours = grid_contours(win, r["w1s"], r["w2s"], r["Z"], levels, lc, width=1.5)
        axes0 = VGroup(DashedLine(win(-1.6, 0), win(1.6, 0), color=RED_, stroke_width=1.6, dash_length=0.08),
                       DashedLine(win(0, -4.4), win(0, 4.4), color=RED_, stroke_width=1.6, dash_length=0.08))
        sad = VGroup(Line(UL, DR), Line(UR, DL)).scale(0.11).set_stroke(RED_, 5).move_to(win(0, 0))
        wm = r["w_min"]
        mins = VGroup(star(win(wm[0], wm[1]), GREEN_, 0.15), star(win(-wm[0], -wm[1]), GREEN_, 0.15))
        minlab = Text("minimum", font_size=20, color=GREEN_).next_to(mins[0], RIGHT, buff=0.12)
        minlab2 = Text("minimum", font_size=20, color=GREEN_).next_to(mins[1], LEFT, buff=0.12)
        for m in (minlab, minlab2):
            m.add_background_rectangle(color=WHITE, opacity=0.9, buff=0.05)
        hlab = mtext("w<sub>1</sub>", 22, MUTED).next_to(frame, DOWN, buff=0.1).align_to(frame, RIGHT)
        vlab = mtext("w<sub>2</sub>", 22, MUTED).next_to(frame, LEFT, buff=0.1).align_to(frame, UP)

        # loss vs step
        n = len(r["runs"]["GD"]["loss"]) - 1
        ax = Axes(x_range=[0, n, 20], y_range=[0.30, 0.52, 0.05], x_length=5.2, y_length=3.0, tips=False,
                  axis_config={"stroke_color": MUTED, "stroke_width": 2, "include_tip": False},
                  x_axis_config={"numbers_to_include": list(range(20, n + 1, 20)), "font_size": 20},
                  y_axis_config={"numbers_to_include": [0.35, 0.4, 0.45, 0.5], "font_size": 20,
                                 "decimal_number_config": {"num_decimal_places": 2}}).move_to([3.55, 0.3, 0])
        for num in [*ax.x_axis.numbers, *ax.y_axis.numbers]:
            num.set_color(INK)
        ahdr = Text("loss", font_size=22).next_to(ax, UP, buff=0.15)
        axl = Text("step", font_size=20, color=MUTED).next_to(ax.x_axis.get_right(), DOWN, buff=0.32)
        half = 0.5 * (r["L0"] + r["L_min"])
        lines = VGroup(DashedLine(ax.c2p(0, r["L0"]), ax.c2p(n, r["L0"]), color=RED_, stroke_width=1.5),
                       DashedLine(ax.c2p(0, r["L_min"]), ax.c2p(n, r["L_min"]), color=GREEN_, stroke_width=1.5))
        llab = VGroup(Text(f"saddle {r['L0']:.3f}", font_size=17, color=RED_).next_to(ax.c2p(n, r["L0"]), UP, buff=0.06)
                      .align_to(ax.c2p(n, 0), RIGHT),
                      Text(f"minimum {r['L_min']:.3f}", font_size=17, color=GREEN_)
                      .next_to(ax.c2p(n, r["L_min"]), DOWN, buff=0.06).align_to(ax.c2p(n, 0), RIGHT))

        cols = {"GD": INK, "momentum": BLUE_}
        tr, lt, marks = {}, {}, {}
        for name, run in r["runs"].items():
            tr[name] = Trace([win(*q) for q in run["path"]], cols[name], width=2.6, dot_r=0.022, head_r=0.075)
            lt[name] = Trace([ax.c2p(i, v) for i, v in enumerate(run["loss"])], cols[name], width=2.6, head_r=0.05)
            e = run["escape"]
            dot = Dot(ax.c2p(e, half), radius=0.06, color=cols[name])
            lab = Text(f"step {e}", font_size=20, color=cols[name]).next_to(dot, RIGHT, buff=0.12)
            marks[name] = VGroup(dot, lab)

        c = swap(self, None, "the real loss surface: a saddle and two minima")
        self.play(FadeIn(frame), Create(contours), FadeIn(hlab), FadeIn(vlab), run_time=2.0)
        self.play(FadeIn(mins), FadeIn(minlab), FadeIn(minlab2), run_time=0.6)
        self.wait(0.6)
        c = swap(self, c, f"at w = 0: gradient 0, curvatures +{r['mean_xy']:.2f} and −{r['mean_xy']:.2f}")
        self.play(Create(axes0), FadeIn(sad, scale=1.5), run_time=1.0)
        self.wait(1.4)
        c = swap(self, c, f"GD from ({r['start'][0]}, {r['start'][1]}), lr {r['lr']}: it stalls on the plateau")
        self.play(Create(ax), FadeIn(ahdr), FadeIn(axl), FadeIn(lines), FadeIn(llab), run_time=0.8)
        self.add(tr["GD"], lt["GD"])
        run_steps(self, [tr["GD"], lt["GD"]], [n], [6.0])
        tr["GD"].freeze(), lt["GD"].freeze()
        self.play(FadeIn(marks["GD"]), run_time=0.4)
        c = swap(self, c, f"momentum, ρ = {r['rho']}: speed builds up, it escapes", color=BLUE_)
        self.add(tr["momentum"], lt["momentum"])
        run_steps(self, [tr["momentum"], lt["momentum"]], [n], [6.0])
        tr["momentum"].freeze(), lt["momentum"].freeze()
        self.play(FadeIn(marks["momentum"]), run_time=0.4)
        c = swap(self, c, f"halfway down the loss: step {r['runs']['GD']['escape']} vs step "
                          f"{r['runs']['momentum']['escape']}")
        self.wait(2.8)


# =====================================================================================================
def bowl_panel(r, center=(-2.6, 0.35), width=8.0):
    win = Window(opt.RAW_WINDOW["w2"], opt.RAW_WINDOW["w1"], center=center, width=width)
    frame = win.frame()
    gap0 = r["runs"][0]["gap"][0]
    grp = dict(win=win, frame=frame,
               contours=bowl_contours(win, r["H"], r["w_star"], gap0 * GAPS),
               star=star(win.w(r["w_star"])), start=Dot(win.w([0, 0]), radius=0.07, color=INK),
               hlab=mtext("w<sub>2</sub> (s5)", 20, MUTED).next_to(frame, DOWN, buff=0.1).align_to(frame, RIGHT),
               vlab=mtext("w<sub>1</sub> (bmi)", 20, MUTED).rotate(PI / 2).next_to(frame, LEFT, buff=0.1)
               .align_to(frame, UP))
    return grp


class MomentumNesterov(Scene):
    """The raw diabetes bowl (kappa 89.8), from w = 0, lr 0.05 for all: SGD crawls (1% of the loss gap after 147 steps),
    SGD+Momentum (rho 0.9) speeds along the valley but keeps swinging across (22 steps), Nesterov damps the swing
    (18 steps). Numbers: optim_real.momentum_nesterov(). Target: p2 "Solves the previous problems" and the Nesterov
    slides (replaces the gbhat.com GIFs and screen recordings)."""

    def construct(self):
        r = opt.momentum_nesterov()
        head(self, "Momentum and Nesterov",
             f"diabetes bowl: raw bmi and s5 (κ = {r['kappa']:.1f}) · full-batch gradients · lr = {r['lr']} for all, "
             f"ρ = {r['rho']}")
        B = bowl_panel(r)
        n = len(r["runs"][0]["gap"]) - 1
        lp = LogPlot(center=(4.6, 0.35), width=3.9, height=2.6, n_steps=n, x_step=50, y_range=(-6, 4),
                     y_labels=[-6, -3, 0, 3], header="L − L*")
        texts = {"SGD": "SGD lands on the valley floor, then crawls along it",
                 "SGD+Momentum": "momentum: speeds along the valley, keeps swinging across",
                 "Nesterov": "Nesterov: the look-ahead gradient damps the swing"}
        legend_hdr = Text("steps to 1% of the loss gap", font_size=19, color=MUTED)
        legend_hdr.move_to([B["frame"].get_left()[0], -1.75, 0], aligned_edge=LEFT)
        rows = []
        for i, run in enumerate(r["runs"]):
            col = OPT_COLOR[run["name"]]
            row = VGroup(Line(ORIGIN, 0.35 * RIGHT, color=col, stroke_width=5), Text(run["name"], font_size=21, color=col),
                         Text(f"{run['steps']}", font_size=21, color=col, weight=BOLD)).arrange(RIGHT, buff=0.15)
            row.move_to([B["frame"].get_left()[0] + 3.6 * i, -2.15, 0], aligned_edge=LEFT)
            rows.append(row)

        c = swap(self, None, "the narrow valley of Problem #1, start w = 0")
        self.play(FadeIn(B["frame"]), Create(B["contours"]), FadeIn(B["hlab"]), FadeIn(B["vlab"]), run_time=1.4)
        self.play(FadeIn(B["start"]), FadeIn(B["star"]), Create(lp.ax), FadeIn(lp.ylab), FadeIn(lp.xlab),
                  FadeIn(lp.header), run_time=0.9)
        self.wait(0.5)
        prev = []
        for i, run in enumerate(r["runs"]):
            col = OPT_COLOR[run["name"]]
            c = swap(self, c, texts[run["name"]], color=col)
            tr = Trace([B["win"].w(q) for q in run["path"]], col, width=2.4, dot_r=0.025, head_r=0.07)
            lt = Trace(lp.pts(run["gap"]), col, width=2.0, head_r=0.04)
            self.add(tr, lt)
            run_steps(self, [tr, lt], [n], [5.2])
            tr.freeze(), lt.freeze()
            anims = [FadeIn(rows[i])] + ([FadeIn(legend_hdr)] if i == 0 else [])
            anims += [p.animate.set_opacity(0.3) for p in prev]
            self.play(*anims, run_time=0.5)
            prev.append(VGroup(tr.line, tr.dots, tr.head))
        steps = [run["steps"] for run in r["runs"]]
        c = swap(self, c, f"1% of the loss gap: {steps[0]}, {steps[1]} and {steps[2]} steps")
        self.wait(2.8)


# =====================================================================================================
class AdaptiveMethods(Scene):
    """The raw diabetes bowl again: SGD+Momentum vs AdaGrad vs RMSProp vs Adam, each with its best learning rate from a
    sweep (lowest loss after 100 steps, path inside the plotted window). The adaptive methods divide each coordinate by
    its own gradient scale, so w2 (s5) moves as fast as w1 (bmi); AdaGrad's steps shrink, RMSProp's do not.
    Numbers: optim_real.adaptive_methods(). Target: p2 RMSProp and Adam slides (replaces the screen recordings)."""

    def construct(self):
        r = opt.adaptive_methods()
        head(self, "Adaptive learning rates",
             f"diabetes bowl: raw bmi and s5 (κ = {r['kappa']:.1f}) · full batch · lr of each from a sweep")
        B = bowl_panel(r)
        n = r["num_steps"]
        lp = LogPlot(center=(4.6, 0.35), width=3.9, height=2.6, n_steps=n, x_step=50, y_range=(-6, 4),
                     y_labels=[-6, -3, 0, 3], header="L − L*")
        texts = {"SGD+Momentum": "SGD+Momentum: swings across the valley",
                 "AdaGrad": "AdaGrad: a step size per coordinate, shrinking over time",
                 "RMSProp": "RMSProp: a leaky average, the steps keep their size",
                 "Adam": "Adam: RMSProp plus momentum, with bias correction"}
        legend_hdr = Text(f"learning rate · L − L* after {n} steps", font_size=19, color=MUTED)
        legend_hdr.move_to([B["frame"].get_left()[0], -1.7, 0], aligned_edge=LEFT)
        rows = []
        for i, run in enumerate(r["runs"]):
            col = OPT_COLOR[run["name"]]
            row = VGroup(Line(ORIGIN, 0.35 * RIGHT, color=col, stroke_width=5),
                         mtext(f"{run['name']}  lr {run['lr']:g} · {fmt_e(run['final_gap'])}", 21, col)
                         ).arrange(RIGHT, buff=0.15)
            row.move_to([B["frame"].get_left()[0] + 5.4 * (i % 2), -2.08 - 0.4 * (i // 2), 0], aligned_edge=LEFT)
            rows.append(row)

        c = swap(self, None, "the same narrow valley, start w = 0")
        self.play(FadeIn(B["frame"]), Create(B["contours"]), FadeIn(B["hlab"]), FadeIn(B["vlab"]), run_time=1.4)
        self.play(FadeIn(B["start"]), FadeIn(B["star"]), Create(lp.ax), FadeIn(lp.ylab), FadeIn(lp.xlab),
                  FadeIn(lp.header), run_time=0.9)
        self.wait(0.5)
        prev = []
        for i, run in enumerate(r["runs"]):
            col = OPT_COLOR[run["name"]]
            c = swap(self, c, texts[run["name"]], color=col)
            tr = Trace([B["win"].w(q) for q in run["path"]], col, width=2.4, dot_r=0.025, head_r=0.07)
            lt = Trace(lp.pts(run["gap"]), col, width=2.0, head_r=0.04)
            self.add(tr, lt)
            run_steps(self, [tr, lt], [n], [4.6])
            tr.freeze(), lt.freeze()
            anims = [FadeIn(rows[i])] + ([FadeIn(legend_hdr)] if i == 0 else [])
            anims += [p.animate.set_opacity(0.3) for p in prev]
            self.play(*anims, run_time=0.5)
            prev.append(VGroup(tr.line, tr.dots, tr.head))
        c = swap(self, c, "per-coordinate scaling undoes the bad feature scales")
        self.wait(2.8)


# =====================================================================================================
class MuonOrthogonalize(Scene):
    """Muon on the matrix of the Muon slides: M = W1's SGD-momentum buffer (100 x 3072) after one epoch of the 2-layer
    MLP on CIFAR-10. Its 100 singular values are lopsided (top direction 33% of the energy, top 5: 73%); five
    Newton-Schulz steps (Keller Jordan's zeropower_via_newtonschulz5, MIT license) apply 3.4445s - 4.7750s^3 + 2.0315s^5
    to every singular value and leave all 100 in [0.68, 1.13]: the update is ~ U V^T. Numbers: muon_demo.muon_iterations()
    (a wrapper around muon_cifar.py). Target: p2 "More recent methods (Muon)" (replaces the YouTube video)."""

    def construct(self):
        import muon_demo                     # imports CIFAR-10 and trains one epoch: only this scene needs it
        r = muon_demo.muon_iterations()
        sv, rel = r["sv"], r["sv_rel"]       # sv: (6, 100) per direction after 0..5 steps; rel: sigma / sigma_max
        n = len(rel)
        head(self, "Muon: orthogonalize the update",
             f"CIFAR-10, 2-layer MLP: W1's SGD-momentum buffer M ({r['shape'][0]} × {r['shape'][1]}) after one epoch")
        base_y, top_y, ymax = -1.95, 1.65, 1.25
        x0, x1 = -6.2, 1.55
        pitch = (x1 - x0) / (n - 1)
        bw = 0.68 * pitch
        hy = lambda v: max((top_y - base_y) * v / ymax, 0.006)

        def bars_at(vals):
            return VGroup(*[Rectangle(width=bw, height=hy(v), stroke_width=0, fill_color=PURPLE_, fill_opacity=0.9)
                            .move_to([x0 + k * pitch, base_y + hy(v) / 2, 0]) for k, v in enumerate(vals)])

        axis = Line([x0 - 0.2, base_y, 0], [x1 + 0.2, base_y, 0], color=MUTED, stroke_width=2)
        one = DashedLine([x0 - 0.2, base_y + hy(1), 0], [x1 + 0.2, base_y + hy(1), 0], color=MUTED, stroke_width=1.5,
                         dash_length=0.08)
        one_lab = Text("1", font_size=20, color=MUTED).next_to(one, LEFT, buff=0.1)
        ticks = VGroup(*[Text(f"{k}", font_size=18, color=MUTED).move_to([x0 + (k - 1) * pitch, base_y - 0.25, 0])
                         for k in (1, 25, 50, 75, 100)])
        klab = Text("direction (largest σ first)", font_size=18, color=MUTED).next_to(ticks, DOWN, buff=0.12)
        bars = bars_at(rel)
        hdr = mtext("σ / σ<sub>max</sub>", 24).move_to([(x0 + x1) / 2, 2.05, 0])

        # one Newton-Schulz step on a singular value: p(s) = a s + b s^3 + c s^5
        pc, pw = np.array([4.75, -0.2, 0]), 3.2
        P = lambda s, v: pc + np.array([(s / ymax - 0.5) * pw, (v / ymax - 0.5) * pw, 0])
        pax = VGroup(Line(P(0, 0), P(ymax, 0), color=MUTED, stroke_width=2),
                     Line(P(0, 0), P(0, ymax), color=MUTED, stroke_width=2))
        ident = DashedLine(P(0, 0), P(ymax, ymax), color=MUTED, stroke_width=1.5, dash_length=0.08)
        curve = ParametricFunction(lambda s: P(s, muon_demo.ns_map(s)), t_range=[0, 1.205], color=PURPLE_,
                                   stroke_width=4)
        ptick = VGroup(*[Text(f"{v:g}", font_size=18, color=MUTED).next_to(P(v, 0), DOWN, buff=0.1) for v in (0, 0.5, 1)],
                       *[Text(f"{v:g}", font_size=18, color=MUTED).next_to(P(0, v), LEFT, buff=0.1) for v in (0.5, 1)])
        pxl = Text("σ before the step", font_size=19, color=MUTED).next_to(P(ymax / 2, 0), DOWN, buff=0.42)
        pyl = Text("σ after the step", font_size=19, color=MUTED).next_to(P(0, ymax), UP, buff=0.12) \
            .align_to(P(0, ymax), LEFT)
        credit = Text("Newton–Schulz code: Keller Jordan, Muon (MIT license)", font_size=15, color=MUTED) \
            .next_to(pxl, DOWN, buff=0.3).align_to(P(ymax, 0), RIGHT)

        c = swap(self, None, f"the {n} singular values of M, divided by the largest")
        self.play(FadeIn(axis), FadeIn(ticks), FadeIn(klab), FadeIn(hdr),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02), run_time=2.0)
        self.play(FadeIn(one), FadeIn(one_lab), run_time=0.4)
        self.wait(0.6)
        c = swap(self, c, f"top direction: {100 * r['top1']:.0f}% of the energy, top 5: {100 * r['top5']:.0f}%")
        self.play(*[Indicate(b, color=RED_, scale_factor=1.0) for b in bars[:5]], run_time=1.2)
        self.wait(0.8)
        c = swap(self, c, "Newton–Schulz starts from M / ‖M‖: every σ below 1")
        ns_hdr = Text("Newton–Schulz step 0", font_size=24).move_to(hdr)
        self.play(Transform(bars, bars_at(sv[0])), FadeOut(hdr), FadeIn(ns_hdr), run_time=1.0)
        num = ns_hdr[-1]                                     # the step number: swapped alone below
        self.wait(0.6)
        c = swap(self, c, "each step: σ ← 3.4445σ − 4.7750σ<sup>3</sup> + 2.0315σ<sup>5</sup>", color=PURPLE_)
        self.play(FadeIn(pax), FadeIn(ptick), FadeIn(pxl), FadeIn(pyl), Create(ident), FadeIn(credit), run_time=0.6)
        self.play(Create(curve), run_time=1.2)
        dots = VGroup(*[Dot(P(v, v), radius=0.035, color=PURPLE_) for v in sv[0]])
        self.play(FadeIn(dots), run_time=0.4)
        for t in range(1, len(sv)):
            if t == 1:
                c = swap(self, c, "small σ grow ×3.4 per step, large σ fall back")
            slow = 1.0 if t <= 2 else 0.6
            arrows = VGroup(*[Line(P(sv[t - 1][k], sv[t - 1][k]), P(sv[t - 1][k], sv[t][k]), color=PURPLE_,
                                   stroke_width=1.2, stroke_opacity=0.6) for k in range(n)])
            self.play(Create(arrows), *[d.animate.move_to(P(sv[t - 1][k], sv[t][k])) for k, d in enumerate(dots)],
                      run_time=0.8 * slow)
            new_num = Text(f"{t}", font_size=24).move_to(num, aligned_edge=LEFT)
            self.play(Transform(bars, bars_at(sv[t])), FadeOut(num, shift=0.15 * UP), FadeIn(new_num, shift=0.15 * UP),
                      *[d.animate.move_to(P(sv[t][k], sv[t][k])) for k, d in enumerate(dots)], FadeOut(arrows),
                      run_time=0.9 * slow)
            num = new_num
        self.wait(0.4)
        c = swap(self, c, f"after 5 steps: all {n} σ between {sv[-1].min():.2f} and {sv[-1].max():.2f}")
        self.wait(1.6)
        c = swap(self, c, "update ≈ UV<sup>T</sup>: every direction gets about the same step")
        self.wait(2.8)
