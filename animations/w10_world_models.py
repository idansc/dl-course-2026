"""Week 10 (world models, 3-D scene representations). Data comes from `python w10_data.py` (data/w10_*.npz).
Render one scene:
    manim -qh --disable_caching -o GaussianSplatting w10_world_models.py GaussianSplatting
"""
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


def zcap(scene, old, text, z=10, **kw):
    """swap_caption that keeps the caption above masks (z_index)."""
    new = caption(text, **kw).set_z_index(z)
    anims = [FadeIn(new, shift=0.1 * UP)]
    if old is not None:
        anims.insert(0, FadeOut(old, shift=0.1 * UP))
    scene.play(*anims, run_time=0.6)
    return new


def rgb_hex(c):
    c = np.clip(np.asarray(c, dtype=float), 0, 1)
    return rgb_to_hex(c)


class GaussianSplatting(Scene):
    """A real photo fitted by 256 2-D Gaussians (real torch fit, 1000 Adam steps): the splats move, stretch and
    recolour while the alpha-composited render sharpens. Target slide: w10 'Gaussian splatting' (new)."""

    def construct(self):
        d = np.load(DATA / "w10_splats.npz")
        steps, psnr = d["step"], d["psnr"]
        S, cy = 3.4, -0.4                                       # panel size and centre height
        xs = [-4.6, 0.0, 4.6]
        hdr = title(self, "Gaussian splatting", "a photo explained by 256 Gaussians")
        hdr.set_z_index(10)

        def to_scene(u, v):
            return np.array([xs[1] + (u - 0.5) * S, cy + (0.5 - v) * S, 0])

        def splats(k, k_sigma=1.2):
            mu, ls, th = d["mu"][k], np.exp(d["log_s"][k]), d["theta"][k]
            col, op = 1 / (1 + np.exp(-d["color"][k])), 1 / (1 + np.exp(-d["opacity"][k]))
            g = VGroup()
            for i in range(len(mu)):
                w = max(2 * k_sigma * ls[i, 0] * S, 0.02)
                h = max(2 * k_sigma * ls[i, 1] * S, 0.02)
                e = Ellipse(width=w, height=h, stroke_width=0.6, stroke_color=rgb_hex(col[i] * 0.6),
                            stroke_opacity=0.6, fill_color=rgb_hex(col[i]), fill_opacity=0.85 * op[i])
                e.rotate(-th[i]).move_to(to_scene(*mu[i]))
                g.add(e)
            return VGroup(*reversed(g.submobjects))             # index 0 is in front: draw it last

        def img(arr, x):
            m = ImageMobject(arr).set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            return m.set_height(S).move_to([x, cy, 0]).set_z_index(5)

        frame = [Square(S, stroke_color=MUTED, stroke_width=2).move_to([x, cy, 0]).set_z_index(6) for x in xs]
        # white masks clip the splat view to its panel
        big = 20
        L, R, T, B = xs[1] - S / 2, xs[1] + S / 2, cy + S / 2, cy - S / 2
        masks = VGroup(Rectangle(width=big, height=big).move_to([0, T + big / 2, 0]),
                       Rectangle(width=big, height=big).move_to([0, B - big / 2, 0]),
                       Rectangle(width=big, height=big).move_to([L - big / 2, 0, 0]),
                       Rectangle(width=big, height=big).move_to([R + big / 2, 0, 0]))
        masks.set_style(fill_color=WHITE, fill_opacity=1, stroke_width=0).set_z_index(4)
        labels = VGroup(Text("target photo (64×64 pixels)", font_size=22),
                        Text("the 256 Gaussians", font_size=22, color=PURPLE_),
                        Text("their rendering", font_size=22, color=GREEN_))
        for lab, x in zip(labels, xs):
            lab.move_to([x, cy + S / 2 + 0.28, 0]).set_z_index(10)

        target = img(d["target_hi"], xs[0])
        cap = zcap(self, None, "the data: one photo, 64×64×3 = 12,288 numbers to explain")
        self.add(masks)
        self.play(FadeIn(target), Create(frame[0]), FadeIn(labels[0]))
        self.wait(0.6)

        G = splats(0)
        cap = zcap(self, cap, "the model: 256 Gaussians, each with position, shape, colour, opacity")
        self.play(Create(frame[1]), FadeIn(labels[1]), LaggedStart(*[FadeIn(e, scale=0.5) for e in G], lag_ratio=0.004),
                  run_time=1.6)
        one = G[-30]
        ring = Circle(radius=0.32, color=INK, stroke_width=3).move_to(one).set_z_index(7)
        self.play(Create(ring), run_time=0.5)
        self.wait(0.8)
        self.play(FadeOut(ring), run_time=0.3)

        f1 = MathTex(r"C(x) = \sum_i c_i\,\alpha_i G_i(x)\prod_{j<i}\bigl(1-\alpha_j G_j(x)\bigr)", font_size=30)
        f2 = MathTex(r"\mathcal{L} = \textstyle\sum_x \lVert C(x) - I(x)\rVert^2", font_size=30)
        f1.to_corner(UR, buff=0.45).set_z_index(10)
        f2.next_to(f1, DOWN, buff=0.15).align_to(f1, RIGHT).set_z_index(10)
        rend = img(d["renders"][0], xs[2])
        cap = zcap(self, cap, "render: alpha-composite the splats front to back, pixel by pixel")
        self.play(Write(f1), run_time=1.2)
        self.play(FadeIn(rend), Create(frame[2]), FadeIn(labels[2]))
        self.wait(0.5)
        cap = zcap(self, cap, "differentiable: the pixel error backpropagates into every Gaussian")
        self.play(Write(f2), run_time=0.9)
        stat = Text(f"step {steps[0]}    PSNR {psnr[0]:.1f} dB", font_size=24).move_to([xs[2], cy - S / 2 - 0.3, 0]).set_z_index(10)
        self.play(FadeIn(stat))
        self.wait(0.6)

        cap = zcap(self, cap, f"Adam steps move, stretch and recolour the Gaussians: PSNR {psnr[0]:.0f} → {psnr[-1]:.0f} dB", color=PURPLE_)
        for k in range(1, len(steps)):
            Gk = splats(k)
            new_r = img(d["renders"][k], xs[2])
            new_s = Text(f"step {steps[k]}    PSNR {psnr[k]:.1f} dB", font_size=24).move_to(stat).set_z_index(10)
            rt = 1.3 if k < 8 else 1.0
            self.play(Transform(G, Gk), FadeIn(new_r), FadeOut(rend), FadeTransform(stat, new_s), run_time=rt)
            rend, stat = new_r, new_s
            if k == 6:
                cap = zcap(self, cap, "big blobs settle the background first, small thin ones add edges later", size=26)
        self.wait(0.8)
        cap = zcap(self, cap, "3-D Gaussian splatting: the same fit with 3-D Gaussians and many camera views", color=GREEN_, size=26)
        self.wait(2.5)


class WorldModelRollout(Scene):
    """A learned world model (MLP on (state, action) → next state, bouncing-ball physics) imagines a rollout; the
    real and imagined balls separate as the horizon grows. Target slide: w10 'World models: learning to imagine'."""

    def construct(self):
        d = np.load(DATA / "w10_world.npz")
        real, imag, acts, err, merr = d["real"], d["imag"], d["acts"], d["err"], d["mean_err"]
        title(self, "A world model imagines the future", "bouncing ball: state = (x, y, vx, vy), action = push left / none / right")

        # ---- phase A: the model, then a filmstrip of its own predictions
        f1 = MathTex(r"\hat s_{t+1} = f_\theta(s_t,\ a_t)", font_size=40).move_to([0, 1.75, 0])
        cap = caption("a world model maps the current state and an action to the next state")
        self.play(FadeIn(cap), Write(f1))
        self.wait(0.6)

        sc_f = 0.38
        def mini(s, x, imagined, label):
            box = Rectangle(width=4.0 * sc_f, height=2.4 * sc_f, stroke_color=BLUE_ if imagined else INK, stroke_width=3)
            if imagined:
                box = DashedVMobject(box, num_dashes=40)
            box.move_to([x, 0.1, 0])
            o = box.get_corner(DL)
            ball = Dot(o + np.array([s[0] * sc_f, s[1] * sc_f, 0]), radius=0.09, color=BLUE_ if imagined else INK)
            lab = MathTex(label, font_size=30, color=BLUE_ if imagined else INK).next_to(box, DOWN, buff=0.15)
            return VGroup(box, ball, lab)

        ts = [0, 6, 12, 18, 24]
        xs = np.linspace(-5.6, 5.6, len(ts))
        frames = [mini(real[0], xs[0], False, r"s_0\ \text{(observed)}")]
        for j, t in enumerate(ts[1:], 1):
            frames.append(mini(imag[t], xs[j], True, rf"\hat s_{{{t}}}"))
        arrows = VGroup()
        for j in range(len(ts) - 1):
            a = acts[ts[j]:ts[j + 1]]
            glyph = {1: "→", 0: "·", -1: "←"}
            lab = Text("".join(glyph[int(v)] for v in a), font_size=18, color=ORANGE_)
            ar = Arrow(frames[j][0].get_right(), frames[j + 1][0].get_left(), buff=0.08, color=MUTED, stroke_width=3,
                       max_tip_length_to_length_ratio=0.25)
            if lab.width > 0.9 * ar.get_length():
                lab.scale_to_fit_width(0.9 * ar.get_length())
            lab.next_to(ar, UP, buff=0.08)
            ft = MathTex(r"f_\theta\!\times\!6", font_size=24, color=PURPLE_).next_to(ar, DOWN, buff=0.05)
            arrows.add(VGroup(ar, lab, ft))
        self.play(FadeIn(frames[0]))
        cap = swap_caption(self, cap, "feed each prediction back in: a whole future imagined without acting")
        f2 = MathTex(r"\hat s_{t+2} = f_\theta(\hat s_{t+1},\ a_{t+1}),\ \ \dots", font_size=34, color=BLUE_).move_to([0, -1.6, 0])
        self.play(Write(f2), run_time=0.8)
        for j in range(1, len(ts)):
            self.play(GrowArrow(arrows[j - 1][0]), FadeIn(arrows[j - 1][1:]), FadeIn(frames[j], shift=0.2 * RIGHT), run_time=0.7)
        acts_lab = Text("orange: the 6 actions between frames (→ push right, · none, ← push left)", font_size=20,
                        color=ORANGE_).move_to([0, -2.35, 0])
        self.play(FadeIn(acts_lab))
        self.wait(1.2)
        self.play(*[FadeOut(m) for m in [*frames, arrows, f2, acts_lab]], f1.animate.scale(0.75).move_to([4.4, 2.95, 0]))

        # ---- phase B: real vs imagined, error vs horizon
        k = 1.25
        box = Rectangle(width=4.0 * k, height=2.4 * k, stroke_color=INK, stroke_width=3).move_to([-3.4, -0.25, 0])
        o = box.get_corner(DL)
        P = lambda s: o + np.array([s[0] * k, s[1] * k, 0])
        ax = Axes(x_range=[0, 50, 10], y_range=[0, 1.8, 0.5], x_length=5.0, y_length=3.0, tips=False,
                  axis_config={"color": MUTED, "include_numbers": True, "font_size": 22,
                               "decimal_number_config": {"color": INK, "num_decimal_places": 1}},
                  x_axis_config={"decimal_number_config": {"color": INK, "num_decimal_places": 0}},
                  ).move_to([3.7, -0.2, 0])
        xl = Text("horizon (steps imagined)", font_size=20).next_to(ax, DOWN, buff=0.1)
        yl = Text("position error", font_size=20).rotate(PI / 2).next_to(ax, LEFT, buff=0.25)
        H = 45
        t = ValueTracker(0)
        rdot = always_redraw(lambda: Dot(P(real[int(t.get_value())]), radius=0.13, color=INK))
        idot = always_redraw(lambda: Dot(P(imag[int(t.get_value())]), radius=0.13, color=BLUE_))
        rtrail = always_redraw(lambda: VMobject(stroke_color=INK, stroke_width=3).set_points_as_corners(
            [P(s) for s in real[:max(2, int(t.get_value()) + 1)]]))
        itrail = always_redraw(lambda: DashedVMobject(VMobject().set_points_as_corners(
            [P(s) for s in imag[:max(2, int(t.get_value()) + 1)]]), num_dashes=max(2, int(t.get_value()) * 2)).set_stroke(BLUE_, 3))
        gap = always_redraw(lambda: DashedLine(P(real[int(t.get_value())]), P(imag[int(t.get_value())]),
                                               color=RED_, stroke_width=3) if err[int(t.get_value())] > 0.05 else VMobject())
        curve = always_redraw(lambda: ax.plot_line_graph(np.arange(int(t.get_value()) + 1), merr[:int(t.get_value()) + 1],
                                                         line_color=RED_, stroke_width=4, add_vertex_dots=False))
        leg = VGroup(VGroup(Dot(color=INK, radius=0.09), Text("real world", font_size=20)).arrange(RIGHT, buff=0.12),
                     VGroup(Dot(color=BLUE_, radius=0.09), Text("imagined by the world model", font_size=20)).arrange(RIGHT, buff=0.12)
                     ).arrange(RIGHT, buff=0.5).next_to(box, UP, buff=0.15)
        cap = swap_caption(self, cap, "same start, same actions: the real ball (black) and the imagined one (blue)")
        mlab = Text("red: mean gap over 200 random rollouts", font_size=18, color=RED_).next_to(ax, UP, buff=0.1)
        self.play(Create(box), FadeIn(leg), Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(mlab))
        self.add(rtrail, itrail, gap, rdot, idot, curve)
        self.play(t.animate.set_value(12), run_time=3, rate_func=linear)
        cap = swap_caption(self, cap, f"one-step error is tiny ({merr[1]:.2f}), but every step builds on the last prediction",
                           size=26)
        self.play(t.animate.set_value(25), run_time=3.5, rate_func=linear)
        cap = swap_caption(self, cap, f"errors compound: mean gap {merr[10]:.2f} at 10 steps, {merr[25]:.2f} at 25, "
                                      f"{merr[45]:.2f} at 45 (200 rollouts)", color=RED_, size=26)
        self.play(t.animate.set_value(H), run_time=5, rate_func=linear)
        self.wait(1)
        cap = swap_caption(self, cap, "so: imagine short horizons, then re-observe the real world (next: MPC)", color=PURPLE_)
        self.wait(2.5)


class MPCImagination(Scene):
    """Random-shooting model-predictive control: 60 imagined action sequences, scored, the first action of the
    best executed, then replanning from the new real state. Target slide: w10 'Planning in imagination (MPC)'."""

    def construct(self):
        d = np.load(DATA / "w10_mpc.npz")
        plans, scores, best, path, goal, wall_top = d["plans"], d["scores"], d["best"], d["path"], d["goal"], float(d["wall_top"])
        R = len(best)
        title(self, "Planning in imagination (MPC)", "imagine many futures, act one step, look again")
        k, c0 = 0.72, np.array([-2.2, -0.35, 0])
        P = lambda p: c0 + np.array([p[0] * k, p[1] * k, 0])
        arena = Rectangle(width=12 * k, height=5.6 * k, stroke_color=MUTED, stroke_width=2).move_to(c0)
        wall = Rectangle(width=0.12, height=(wall_top + 2.8) * k, stroke_width=0, fill_color=INK, fill_opacity=1)
        wall.move_to(P([0, (wall_top - 2.8) / 2]))
        star = Star(n=5, outer_radius=0.22, inner_radius=0.1, color=GREEN_, fill_opacity=1).move_to(P(goal))
        glab = Text("goal", font_size=20, color=GREEN_).next_to(star, UP, buff=0.1)
        agent = Dot(P(path[0]), radius=0.13, color=BLUE_).set_z_index(5)
        alab = Text("agent", font_size=20, color=BLUE_).next_to(agent, DOWN, buff=0.1)
        cap = caption("reach the goal; the wall is open only at the top")
        self.play(FadeIn(cap), Create(arena), FadeIn(wall), FadeIn(star), FadeIn(glab), FadeIn(agent), FadeIn(alab))
        self.wait(0.6)

        # right column: the algorithm, one line per step
        xr = 4.75
        lines = [MathTex(r"a^{(k)}_{1:H} \sim p(a),\ \ k=1..60", font_size=28),
                 MathTex(r"\hat s^{(k)}_{1:H} = \mathrm{rollout}_{f_\theta}(s_t, a^{(k)})", font_size=28),
                 MathTex(r"k^\star = \arg\max_k R(\hat s^{(k)})", font_size=28),
                 MathTex(r"\text{execute } a^{(k^\star)}_1,\ \text{observe } s_{t+1}", font_size=28)]
        for i, m in enumerate(lines):
            m.move_to([xr, 1.6 - 0.75 * i, 0])
        rnote = MathTex(r"R = -\text{dist to goal} - 10\cdot\text{crash}", font_size=24, color=MUTED).next_to(lines[2], DOWN, buff=0.1)
        lines[3].next_to(rnote, DOWN, buff=0.25)
        lo, hi = -12.0, float(scores.max())

        def clip(tr):
            """Cut an imagined trajectory where it crashes (arena edge or wall), as the rollout would."""
            out = [tr[0]]
            for a, b in zip(tr[:-1], tr[1:]):
                if a[0] * b[0] <= 0 and min(a[1], b[1]) < wall_top:
                    t = a[0] / (a[0] - b[0] + 1e-9)
                    out.append(a + t * (b - a)); break
                if abs(b[0]) > 6 or abs(b[1]) > 2.8:
                    out.append(np.clip(b, [-6, -2.8], [6, 2.8])); break
                out.append(b)
            return out

        def plan_group(r):
            g = VGroup()
            order = np.argsort(scores[r])
            for j in order:
                s = scores[r][j]
                if s < -9:
                    col, op = RED_, 0.25
                else:
                    col, op = interpolate_color(ManimColor(MUTED), ManimColor(PURPLE_), np.clip((s - lo) / (hi - lo), 0, 1) ** 2), 0.55
                g.add(VMobject(stroke_color=col, stroke_width=2, stroke_opacity=op).set_points_as_corners([P(p) for p in clip(plans[r][j])]))
            b = VMobject(stroke_color=GREEN_, stroke_width=6).set_points_as_corners([P(p) for p in plans[r][best[r]]])
            return g, b

        cnt = Text("replan 1   real steps 0", font_size=24).move_to([xr, -2.25, 0])
        trail = VMobject(stroke_color=BLUE_, stroke_width=4).set_points_as_corners([P(path[0]), P(path[0])])

        # round 1, slowly
        g, b = plan_group(0)
        cap = swap_caption(self, cap, "1. imagine: sample 60 action sequences, roll each out in the world model")
        self.play(Write(lines[0]), FadeOut(alab))
        self.play(Write(lines[1]), LaggedStart(*[Create(m) for m in g], lag_ratio=0.03), run_time=2.2)
        self.add(trail)
        cap = swap_caption(self, cap, "2. score every imagined future: near the goal is good, a crash (red) is bad")
        self.play(Write(lines[2]), FadeIn(rnote))
        self.wait(1.0)
        cap = swap_caption(self, cap, "3. commit to only the first action of the best plan", color=GREEN_)
        self.play(Create(b), Write(lines[3]), FadeIn(cnt))
        self.play(agent.animate.move_to(P(path[1])), trail.animate.set_points_as_corners([P(path[0]), P(path[1])]),
                  FadeTransform(cnt, cnt := Text("replan 1   real steps 1", font_size=24).move_to(cnt)), run_time=0.8)
        self.wait(0.5)
        cap = swap_caption(self, cap, "4. observe the real new state and replan from scratch")
        self.play(FadeOut(g), FadeOut(b), run_time=0.4)
        for r in range(1, 4):
            g, b = plan_group(r)
            self.play(LaggedStart(*[Create(m) for m in g], lag_ratio=0.01), run_time=0.9)
            self.play(Create(b), run_time=0.4)
            self.play(agent.animate.move_to(P(path[r + 1])), trail.animate.set_points_as_corners([P(p) for p in path[:r + 2]]),
                      FadeTransform(cnt, cnt := Text(f"replan {r + 1}   real steps {r + 1}", font_size=24).move_to(cnt)), run_time=0.5)
            self.play(FadeOut(g), FadeOut(b), run_time=0.3)

        cap = swap_caption(self, cap, f"repeat: {R} replans, each with a fresh 12-step imagination", size=26)
        for r in range(4, R, 2):
            g, b = plan_group(r)
            self.add(g, b)
            self.play(agent.animate.move_to(P(path[r + 1])), trail.animate.set_points_as_corners([P(p) for p in path[:r + 2]]),
                      Transform(cnt, Text(f"replan {r + 1}   real steps {r + 1}", font_size=24).move_to(cnt)),
                      run_time=0.14, rate_func=linear)
            self.remove(g, b)
        self.play(agent.animate.move_to(P(path[-1])), trail.animate.set_points_as_corners([P(p) for p in path]),
                  Transform(cnt, Text(f"replan {R}   real steps {R}", font_size=24).move_to(cnt)), run_time=0.3)
        cap = swap_caption(self, cap, "MPC: plan far, commit one step, re-observe: model errors never pile up",
                           color=PURPLE_, size=26)
        self.wait(2.5)
