"""Week 1, lecture 2 (optimization): why Nesterov, shown on a cut across the valley. Two clips that set up the
Nesterov slides: MomentumOvershoot (the problem: momentum keeps swinging across the steep direction) and
NesterovLookAhead (the fix: the gradient taken at the look-ahead point already points back, so the swing dies).

The cut: the runs of MomentumNesterov (raw diabetes bowl, lr 0.05, rho 0.9, from w = 0) projected on the steep
eigenvector of H are exactly 1-D runs on the parabola L(e) = 1/2 * 19.53 * e^2. Every position, look-ahead point and
gradient step comes from code/nesterov_damping.py (the scenes import it, so `python code/nesterov_damping.py` prints
exactly what the clips show). Code table: code/README.md.
Render one scene:
    manim -qh --disable_caching -o MomentumOvershoot w01_nesterov.py MomentumOvershoot
Slide version without words except a small legend (renders/w01/clean/; its own media dir, parallel runs collide on
the text cache):
    DLC_CLEAN=1 manim -qh --disable_caching --media_dir media_clean_MomentumOvershoot -o MomentumOvershoot \
        w01_nesterov.py MomentumOvershoot
"""
import sys
from pathlib import Path

from style import *

sys.path.insert(0, str(Path(__file__).parent / "code"))     # `code` is also a stdlib module: import the file directly
import nesterov_damping as nd                               # noqa: E402

N_SHOWN = 24                                    # steps shown in both clips
FAINT = 0.35                                    # opacity of the momentum ball next to Nesterov
LOOK = "#15803D"                                # look-ahead ring: a darker green than the ball

if CLEAN:      # slide version: the frame has no title or caption, so the cut and the strip are larger
    CUT = dict(x0=0.0, y0=-0.3, sx=5.7, sy=3.25)
    STRIP = dict(x0=-5.7, x1=5.7, yc=-2.55, hu=1.05)
    BALL_R, RING_R, LEG_FS, LEG_Y, ARROW_W, OFFSET = 0.2, 0.28, 28, 3.05, 7.0, 0.36
else:
    CUT = dict(x0=0.0, y0=-0.55, sx=5.0, sy=2.35)
    STRIP = dict(x0=-5.0, x1=5.0, yc=-2.1, hu=0.75)
    BALL_R, RING_R, LEG_FS, LEG_Y, ARROW_W, OFFSET = 0.17, 0.24, 22, 1.75, 6.0, 0.32


# =====================================================================================================
# helpers
def mtext(markup, size=24, color=INK):
    """Arial text with Pango markup (<sub>, <sup>)."""
    return MarkupText(markup, font=FONT, font_size=size, color=color)


def head(scene, text, sub):
    if CLEAN:                                  # slide version: the slide has its own title
        return VGroup()
    t = Text(text, font_size=40, weight=BOLD).to_corner(UL, buff=0.45)
    s = mtext(sub, 24, MUTED).next_to(t, DOWN, aligned_edge=LEFT, buff=0.15)
    scene.play(FadeIn(VGroup(t, s), shift=0.2 * DOWN), run_time=0.8)
    return VGroup(t, s)


def cap(markup, color=INK, size=28):
    if CLEAN:
        return VMobject()
    return mtext(markup, size, color).to_edge(DOWN, buff=0.45)


class Cut:
    """The parabola L(e) = 1/2 lambda_max e^2 on screen, in units of the start's distance |e0|: u = e / |e0| sits at
    x = x0 + sx·u, y = y0 + sy·u² (a vertical scale of L)."""

    def __init__(self, x0, y0, sx, sy, umax=1.1):
        self.x0, self.y0, self.sx, self.sy, self.umax = x0, y0, sx, sy, umax

    def p(self, u):
        return np.array([self.x0 + self.sx * u, self.y0 + self.sy * u * u, 0.0])

    def normal(self, u):
        """unit normal pointing into the bowl"""
        n = np.array([-2 * self.sy * u, self.sx, 0.0])
        return n / np.linalg.norm(n)

    def curve(self, width=4.0):
        return ParametricFunction(self.p, t_range=[-self.umax, self.umax], color=MUTED, stroke_width=width)

    def arrow(self, ua, ub, color, offset=0.0, width=6.0, tip=0.3, trim_end=0.0, frac=1.0):
        """An arrow along the parabola (shifted by `offset` into the bowl) from u = ua to u = ub, ending `trim_end`
        before ub (at a ring's rim); frac < 1 draws only its first part (to grow it)."""
        if abs(ub - ua) < 1e-9:
            return VMobject()
        P = np.array([self.p(u) + offset * self.normal(u) for u in np.linspace(ua, ub, 160)])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        end = frac * max(s[-1] - trim_end, 0.0)
        if end < 0.03:
            return VMobject()
        at = lambda d: np.array([np.interp(d, s, P[:, i]) for i in range(3)])
        tl = min(tip, 0.6 * end)                       # short arrows: a smaller tip
        apex, base = at(end), at(end - tl)
        d = (apex - base) / max(np.linalg.norm(apex - base), 1e-9)
        side = np.array([-d[1], d[0], 0.0]) * 0.47 * tl
        grp = VGroup()
        if end - tl > 0.01:
            shaft = VMobject(stroke_color=color, stroke_width=width)
            shaft.set_points_as_corners([at(x) for x in np.linspace(0, end - tl, 60)])
            grp.add(shaft)
        grp.add(Polygon(apex, base + side, base - side, stroke_width=0, fill_color=color, fill_opacity=1))
        return grp


class Strip:
    """e_t / |e0| vs step: x = steps 0..n, y = yc + hu·u. A grey zero line (the valley floor) and a left axis."""

    def __init__(self, x0, x1, yc, hu, n=N_SHOWN):
        self.x0, self.x1, self.yc, self.hu, self.n = x0, x1, yc, hu, n

    def p(self, k, u):
        return np.array([self.x0 + (self.x1 - self.x0) * k / self.n, self.yc + self.hu * u, 0.0])

    def axes(self):
        return VGroup(Line(self.p(0, 0), self.p(self.n, 0), color=MUTED, stroke_width=2),
                      Line(self.p(0, -1.1), self.p(0, 1.1), color=MUTED, stroke_width=2))

    def trace(self, us, kf, color, opacity=1.0, width=3.0, dot_r=0.04):
        """the run up to the fractional step kf (the head interpolates between steps, like the ball)"""
        k = int(np.floor(kf + 1e-9))
        pts = [self.p(i, us[i]) for i in range(min(k, self.n) + 1)]
        if kf > k + 1e-6 and k < self.n:
            s = kf - k
            pts.append(self.p(kf, us[k] + s * (us[k + 1] - us[k])))
        line = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
        line.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + 1e-4 * RIGHT])
        dots = VGroup(*[Dot(q, radius=dot_r, color=color, fill_opacity=opacity) for q in pts[:k + 1]])
        head_dot = Dot(pts[-1], radius=1.6 * dot_r, color=color, fill_opacity=opacity)
        return VGroup(line, dots, head_dot)


def ball(color, opacity=1.0):
    return Dot(radius=BALL_R, color=color, fill_opacity=opacity, stroke_width=0)


def ring():
    """the look-ahead point: a dashed hollow circle"""
    return DashedVMobject(Circle(radius=RING_R, stroke_color=LOOK, stroke_width=4), num_dashes=12, dashed_ratio=0.6)


def legend(entries):
    """A row of (icon, name, colour) entries, placed inside the bowl above its floor."""
    row = VGroup()
    for icon, name, color in entries:
        t = Text(name, font_size=LEG_FS, color=color)
        icon.move_to(ORIGIN)
        row.add(VGroup(icon, t).arrange(RIGHT, buff=0.14))
    row.arrange(RIGHT, buff=0.55)
    cap_h = Text("H", font_size=LEG_FS).height        # same baseline for every name (descenders differ)
    for e in row:
        e[1].shift((e[0].get_y() - cap_h / 2 - e[1][0].get_bottom()[1]) * UP)
    return row.move_to([CUT["x0"], LEG_Y, 0])


def lerp(us, kf):
    """position at the fractional step kf (linear between steps)"""
    k = min(int(np.floor(kf + 1e-9)), len(us) - 2)
    return us[k] + (kf - k) * (us[k + 1] - us[k])


def step_times(n, slow, t_slow, t_fast0, t_fast1):
    """per-step durations: the first `slow` steps at t_slow, then from t_fast0 down to t_fast1. Every run time is a
    multiple of 0.1 s (3 frames): other lengths get ms-rounded in the partial movie files, which makes the frame
    timing of the joined clip uneven."""
    return [t_slow] * slow + [round(x, 1) for x in np.linspace(t_fast0, t_fast1, n - slow)]


# =====================================================================================================
class MomentumOvershoot(Scene):
    """The problem Nesterov fixes. The MomentumNesterov run of SGD+Momentum (raw diabetes bowl, lr 0.05, rho 0.9),
    projected on the steep eigenvector of H: exactly a 1-D run on the parabola L(e) = 1/2·19.53·e² (a cut across the
    valley). Released at the start's steep coordinate, the ball reaches the bottom in one step but its velocity carries
    it up the other side; it swings back and forth, the swing shrinking only ×0.949 per step (|z| = sqrt(rho);
    period 5.9 steps, 8 crossings of the floor in 24 steps). The arrow is its velocity: the step it is about to take.
    Numbers: nesterov_damping.steep_cut(). Target: p2 Nesterov slides (before the look-ahead)."""

    def construct(self):
        r = nd.steep_cut()
        run = r["runs"]["SGD+Momentum"]
        u = run["e"][:N_SHOWN + 1] / abs(r["e0"])
        head(self, "Momentum overshoots",
             f"the diabetes valley, cut across in its steep direction: L(e) = ½·{r['lam_max']:.2f}·e<sup>2</sup> · "
             f"lr = {r['lr']}, ρ = {r['rho']}")
        cut, strip = Cut(**CUT), Strip(**STRIP)
        curve, axes = cut.curve(4.5 if CLEAN else 4.0), strip.axes()
        kf, hu = ValueTracker(0.0), ValueTracker(u[0])        # fractional step of the ball; head of the arrow
        b = ball(BLUE_).move_to(cut.p(u[0])).set_z_index(3)
        b.add_updater(lambda m: m.move_to(cut.p(lerp(u, kf.get_value()))))
        arrow = always_redraw(lambda: cut.arrow(lerp(u, kf.get_value()), hu.get_value(), BLUE_, width=ARROW_W)
                              .set_z_index(2))
        trace = always_redraw(lambda: strip.trace(u, kf.get_value(), BLUE_))
        leg = legend([(ball(BLUE_).scale(0.55), "momentum", BLUE_)])
        c = cap(f"momentum carries it past the bottom: the swing shrinks only ×{run['steep']['factor']:.2f} per step")

        self.play(Create(curve), run_time=1.0)
        self.play(FadeIn(b, scale=0.6), FadeIn(axes), FadeIn(leg), run_time=0.6)
        self.add(trace)
        self.wait(0.5)
        self.add(arrow)
        if not CLEAN:
            self.play(FadeIn(c, shift=0.1 * UP), run_time=0.6)
        for t, dt in enumerate(step_times(N_SHOWN, 7, 1.0, 0.6, 0.3)):
            grow = max(0.1, round(0.3 * dt, 1))
            self.play(hu.animate.set_value(u[t + 1]), run_time=grow, rate_func=smooth)            # the velocity
            self.play(kf.animate.set_value(t + 1), run_time=round(dt - grow, 1), rate_func=smooth)  # the step
        b.clear_updaters()
        self.remove(arrow)
        self.wait(1.6)


# =====================================================================================================
class NesterovLookAhead(Scene):
    """The fix. The same cut, now with the Nesterov run of MomentumNesterov next to the momentum ball (faint). Nesterov
    takes the gradient at the look-ahead point e + rho·v (v = the step just taken; optim_real.nesterov's convention):
    step 1 starts at rest, so the look-ahead point is the ball itself and the step is a plain gradient step (the same
    as momentum's); step 2 starts at the bottom with velocity 0.98|e0|: the look-ahead point is 0.86|e0| up the other
    side, where the gradient points back, and the step lands at 0.02|e0|; momentum, whose gradient is taken at the
    bottom, flies on to 0.88|e0|. Nesterov's swing shrinks ×0.145 per step (|z| = sqrt(rho·(1 − lr·lambda))),
    momentum's ×0.949. Numbers: nesterov_damping.steep_cut(). Target: p2 Nesterov slides."""

    def construct(self):
        r = nd.steep_cut()
        M, N = r["runs"]["SGD+Momentum"], r["runs"]["Nesterov"]
        scale = abs(r["e0"])
        um, un = M["e"][:N_SHOWN + 1] / scale, N["e"][:N_SHOWN + 1] / scale
        la = N["look_ahead"]["point"] / scale            # e + rho v
        head(self, "Nesterov momentum",
             "the same cut, lr and ρ · Nesterov takes the gradient at the look-ahead point e + ρv")
        cut, strip = Cut(**CUT), Strip(**STRIP)
        curve, axes = cut.curve(4.5 if CLEAN else 4.0), strip.axes()
        kf = ValueTracker(0.0)
        bm = ball(BLUE_, FAINT).move_to(cut.p(um[0])).set_z_index(3)
        bn = ball(GREEN_).move_to(cut.p(un[0])).set_z_index(4)
        bm.add_updater(lambda m: m.move_to(cut.p(lerp(um, kf.get_value()))))
        bn.add_updater(lambda m: m.move_to(cut.p(lerp(un, kf.get_value()))))
        tm = always_redraw(lambda: strip.trace(um, kf.get_value(), BLUE_, opacity=0.45))
        tn = always_redraw(lambda: strip.trace(un, kf.get_value(), GREEN_))
        leg = legend([(ball(BLUE_, FAINT).scale(0.55), "momentum", BLUE_),
                      (ball(GREEN_).scale(0.55), "Nesterov", GREEN_),
                      (ring().scale(0.55), "look-ahead", LOOK)])
        c = cap(f"the gradient at the look-ahead point points back: ×{N['steep']['factor']:.2f} per step, "
                f"not ×{M['steep']['factor']:.2f}")

        self.play(Create(curve), run_time=1.0)
        self.play(FadeIn(bm, scale=0.6), FadeIn(bn, scale=0.6), FadeIn(axes), FadeIn(leg), run_time=0.6)
        self.add(tm, tn)
        self.wait(0.4)
        if not CLEAN:
            self.play(FadeIn(c, shift=0.1 * UP), run_time=0.6)

        def look_ahead_step(t, grow_v, grow_g, slide, hold):
            """velocity arrow to the look-ahead ring, the gradient step there, then the step itself"""
            g_arrow = lambda f: cut.arrow(la[t], un[t + 1], PURPLE_, offset=OFFSET, width=ARROW_W, frac=f) \
                .set_z_index(2)
            rg = ring().move_to(cut.p(la[t])).set_z_index(3)
            gv, gg = ValueTracker(0.0), ValueTracker(0.0)
            parts = [rg]
            if abs(la[t] - un[t]) > 1e-9:            # moving: an arrow from the ball along the cut to the ring
                v_arrow = always_redraw(lambda: cut.arrow(un[t], la[t], GREEN_, width=ARROW_W, trim_end=RING_R,
                                                          frac=gv.get_value()).set_z_index(2))
                self.add(v_arrow)
                self.play(gv.animate.set_value(1.0), run_time=grow_v, rate_func=smooth)
                v_arrow.clear_updaters()
                parts.append(v_arrow)
            self.play(FadeIn(rg, scale=1.4), run_time=0.4)
            ga = always_redraw(lambda: g_arrow(gg.get_value()))
            self.add(ga)
            self.play(gg.animate.set_value(1.0), run_time=grow_g, rate_func=smooth)
            ga.clear_updaters()
            parts.append(ga)
            self.wait(hold)
            self.play(kf.animate.set_value(t + 1), run_time=slide, rate_func=smooth)
            self.wait(hold)
            self.play(*[FadeOut(p) for p in parts], run_time=0.4)

        look_ahead_step(0, 0.0, 0.8, 0.9, 0.3)       # at rest: the look-ahead point is the ball itself
        look_ahead_step(1, 1.0, 0.9, 1.1, 0.5)       # at the bottom, moving fast: the look-ahead point is up the wall
        for t, dt in enumerate(step_times(N_SHOWN - 2, 2, 0.8, 0.6, 0.3), start=2):
            self.play(kf.animate.set_value(t + 1), run_time=dt, rate_func=smooth)
        for m in (bm, bn):
            m.clear_updaters()
        self.wait(1.6)
