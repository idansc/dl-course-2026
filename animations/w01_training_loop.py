"""Lecture 2, "the training loop" (linear classifier + 2-layer MLP on CIFAR-10, loss, backprop, optimizers).
Render one scene:
    manim -qh --disable_caching -o TrainingLoop w01_training_loop.py TrainingLoop
Slide version without words (renders/w01/clean/): DLC_CLEAN=1 and a separate --media_dir per parallel run.
Every number on screen comes from the runnable student code in code/ (single source of truth; this file only draws):
code/training_loop.py (re-runs the lecture's CIFAR-10 loop, ~10 s; identical to p2_training_loop_src/results.json),
code/adam_bias_correction.py, code/matrix_backprop.py (identical to p2_training_loop_src/bp_check.json).
Colours are the deck's five parts of the loop: Model BLUE_, Loss ORANGE_, Gradient PURPLE_, Update GREEN_, Repeat TEAL.
"""
import sys
from pathlib import Path
from style import *

sys.path.insert(0, str(Path(__file__).parent / "code"))     # not a package: `code` is also a stdlib module
import training_loop as loop_code              # noqa: E402
import adam_bias_correction as adam_code       # noqa: E402
import matrix_backprop as mb_code              # noqa: E402

TEAL = "#0E7490"      # part 5 of the loop (repeat / validation) in the p2 deck
GRID = "#D1D5DB"      # inactive outline in the deck's loop diagram


def mcaption(markup, color=INK, size=28):
    """caption() with Pango markup, for sub/superscripts that Arial has no Unicode glyph for."""
    if CLEAN:
        return VMobject()
    return MarkupText(markup, font=FONT, font_size=size, color=color).to_edge(DOWN, buff=0.45)


def swap_mcaption(scene, old, markup, **kw):
    new = mcaption(markup, **kw)
    if CLEAN:                                  # the same pause, no text
        scene.wait(0.6)
        return new
    scene.play(FadeOut(old, shift=0.1 * UP), FadeIn(new, shift=0.1 * UP), run_time=0.6)
    return new


def title_markup(scene, text, sub):
    """title() with a Pango-markup subtitle."""
    if CLEAN:
        return VGroup()
    t = Text(text, font_size=40, weight=BOLD).to_corner(UL, buff=0.45)
    s = MarkupText(sub, font=FONT, font_size=24, color=MUTED).next_to(t, DOWN, aligned_edge=LEFT, buff=0.15)
    grp = VGroup(t, s)
    scene.play(FadeIn(grp, shift=0.2 * DOWN), run_time=0.8)
    return grp


def rgba(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])


def picture(rgb, h, nearest=False):
    m = ImageMobject(rgba(rgb))
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    m.height = h
    return m


def purple_heat(v, gamma=0.7):
    """0 -> white, 1 -> PURPLE_ (gradient magnitude maps)."""
    c = np.array(ManimColor(PURPLE_).to_rgb()) * 255
    v = np.clip(np.asarray(v, float), 0, 1)[..., None] ** gamma
    return (255 + v * (c - 255)).astype(np.uint8)


def grid_pos(center, cell, gap):
    """centres of a 2 x 5 grid (one cell per CIFAR-10 class, in label order)"""
    return [np.asarray(center, float) + np.array([(c - 2) * (cell + gap), (0.5 - r) * (cell + gap), 0])
            for r in range(2) for c in range(5)]


def image_grid(arrs, cell, gap, center):
    return Group(*[picture(a, cell).move_to(p) for a, p in zip(arrs, grid_pos(center, cell, gap))])


def polyline(pts, **kw):
    m = VMobject(**kw)
    pts = [np.asarray(p, float) for p in pts]
    m.set_points_as_corners(pts if len(pts) > 1 else [pts[0], pts[0] + 1e-4 * RIGHT])
    return m


# =====================================================================================================
class TrainingLoop(Scene):
    """The loop of lecture 2 running for real: model = nn.Linear(3072, 10), cross-entropy, loss.backward(), SGD step
    (lr 1e-3, momentum 0.9), next minibatch. Three real iterations (1, 11, 21: their CIFAR-10 images, the scores of
    the first image, the minibatch loss, |dL/dW| per class, W as 10 templates), then the 20-epoch run: minibatch
    losses, training loss per epoch, W turning into class templates, test accuracy. Target: p2 slide 2 (the loop)."""

    def construct(self):
        data, _, hist = loop_code.run()                 # the lecture's loop, re-run (code/training_loop.py)
        watched, ipe, n_it = hist["watched"], hist["iters_per_epoch"], hist["iterations"]
        shown = [dict(iteration=w["step"], loss=w["loss"], labels=w["y"].tolist(), scores=w["scores"].numpy())
                 for w in watched]
        thumbs = [[loop_code.to_image(x, data["mean"]) for x in w["x"]] for w in watched]
        gmaps = [w["grad_W"].abs().reshape(10, 3, 32, 32).sum(1).numpy() for w in watched]     # |dL/dW| per class
        W_init = loop_code.templates(hist["W_init"])
        W_after = [loop_code.templates(w["W_after"]) for w in watched]
        W_epoch = [loop_code.templates(W) for W in hist["W"]]
        hdr = title(self, "The training loop", "linear classifier on CIFAR-10 · SGD, lr 1e-3, momentum 0.9")

        # ---- the loop diagram (as on the deck's slide 2)
        NAMES = ["Model", "Loss", "Gradient", "Update"]
        FORMS = [r"s = Wx + b", r"L = -\log p_y", r"\partial L / \partial W",
                 r"W \leftarrow W - \alpha\, \partial L / \partial W"]
        COLS = [BLUE_, ORANGE_, PURPLE_, GREEN_]
        if CLEAN:      # slide version: names only in the boxes, no counter; the loop higher, its outputs larger
            BW, BH, GAP, X0, RY = 2.5, 1.0, 0.45, -4.9, 2.55
            SX, SY = -5.95, -0.8
        else:
            BW, BH, GAP, X0, RY = 2.5, 1.08, 0.45, -4.9, 1.55
            SX, SY = -5.95, -1.2              # minibatch slot x; y of the row of per-step outputs
        xs = [X0 + BW / 2 + i * (BW + GAP) for i in range(4)]
        boxes = VGroup()
        for i in range(4):
            r = RoundedRectangle(corner_radius=0.12, width=BW, height=BH, stroke_color=COLS[i], stroke_width=2,
                                 fill_color=COLS[i], fill_opacity=1).move_to([xs[i], RY, 0])
            if CLEAN:
                nm, fm = Text(NAMES[i], font_size=34, weight=BOLD, color=WHITE).move_to(r), VMobject()
            else:
                nm = Text(NAMES[i], font_size=29, weight=BOLD, color=WHITE)
                fm = MathTex(FORMS[i], font_size=31, color=WHITE)
                if fm.width > BW - 0.24:
                    fm.scale_to_fit_width(BW - 0.24)
                VGroup(nm, fm).arrange(DOWN, buff=0.13).move_to(r)
            boxes.add(VGroup(r, nm, fm))

        def arrow(a, b, col):
            return Arrow(a, b, buff=0, stroke_width=4, tip_length=0.15, max_tip_length_to_length_ratio=0.5,
                         max_stroke_width_to_length_ratio=20, color=col)

        arrows = VGroup(arrow([SX + 0.62, RY, 0], [X0 - 0.04, RY, 0], BLUE_))
        for i in range(3):
            arrows.add(arrow([xs[i] + BW / 2 + 0.04, RY, 0], [xs[i + 1] - BW / 2 - 0.04, RY, 0], COLS[i + 1]))
        yb, yr, yend = RY - BH / 2, RY - BH / 2 - 0.42, RY - 0.78
        ret_pts = [np.array(p, float) for p in [[xs[3], yb - 0.03, 0], [xs[3], yr, 0], [SX, yr, 0], [SX, yend, 0]]]
        ret_line = polyline(ret_pts, stroke_color=TEAL, stroke_width=4)
        ret_tip = Triangle(stroke_width=0, fill_color=TEAL, fill_opacity=1).stretch_to_fit_width(0.24)
        ret_tip.stretch_to_fit_height(0.2).move_to([SX, yend, 0], aligned_edge=DOWN)
        rep_lab = Text("repeat", font_size=26 if CLEAN else 22, color=TEAL).move_to([-0.2, yr, 0]).set_z_index(7)
        rep_bg = BackgroundRectangle(rep_lab, color=WHITE, fill_opacity=1, buff=0.1).set_z_index(6)

        def focus(k):
            """k = 0..3: that box in colour; 4: repeat; None: all. The rest grey, as on the deck's part dividers."""
            an = []
            for i in range(4):
                on = k is None or k == i
                r, nm, fm = boxes[i]
                an += [r.animate.set_fill(COLS[i] if on else SOFT, 1).set_stroke(COLS[i] if on else GRID),
                       nm.animate.set_color(WHITE if on else MUTED), fm.animate.set_color(WHITE if on else MUTED),
                       arrows[i].animate.set_color(COLS[i] if on else GRID)]
            on = k is None or k == 4
            an += [ret_line.animate.set_stroke(TEAL if on else GRID), ret_tip.animate.set_fill(TEAL if on else GRID),
                   rep_lab.animate.set_color(TEAL if on else MUTED)]
            return an

        def pulse(i):
            return boxes[i].animate(rate_func=there_and_back).scale(1.07)

        def travel(pts, col, run_time):
            """a dot carrying one step's output to the next part of the loop"""
            dot = Dot(pts[0], radius=0.085, color=col).set_z_index(5)
            return Succession(MoveAlongPath(dot, polyline(pts), run_time=run_time, rate_func=smooth),
                              FadeOut(dot, run_time=0.05))

        # ---- per-step outputs
        def stack(thumbs):
            g = Group()
            n = len(thumbs)
            for j in reversed(range(n)):                       # back card first, image 0 on top
                im = picture(thumbs[j], 0.72, nearest=True)
                fr = SurroundingRectangle(im, buff=0.0, color=WHITE, stroke_width=3)
                fr2 = SurroundingRectangle(im, buff=0.03, color=MUTED, stroke_width=1)
                g.add(Group(im, fr, fr2).move_to([SX + 0.1 * (j - (n - 1) / 2), RY + 0.1 * (j - (n - 1) / 2), 0]))
            return g

        smax = max(np.abs(s["scores"]).max() for s in shown)
        bw_, bstep, bh = (0.2, 0.25, 1.3) if CLEAN else (0.18, 0.235, 0.8)
        bx0 = xs[0] - (9 * bstep + bw_) / 2

        def score_bars(scores):
            g = VGroup()
            for i, s in enumerate(scores):
                h = max(0.03, abs(s) / smax * bh)
                g.add(Rectangle(width=bw_, height=h, stroke_width=0, fill_color=BLUE_, fill_opacity=0.9)
                      .move_to([bx0 + bw_ / 2 + i * bstep, SY + (h / 2 if s >= 0 else -h / 2), 0]))
            return g

        baseline = Line([bx0 - 0.1, SY, 0], [bx0 + 9 * bstep + bw_ + 0.1, SY, 0], color=MUTED, stroke_width=2)
        CELL, CGAP = (0.52, 0.05) if CLEAN else (0.44, 0.05)
        grads = [image_grid([purple_heat(g) for g in gm / gm.max()], CELL, CGAP, [xs[2], SY, 0]) for gm in gmaps]
        Wgrid = image_grid(W_init, CELL, CGAP, [xs[3], SY, 0])
        lab_y = SY + CELL + CGAP / 2 + 0.3
        g_lab = MathTex(r"\partial L/\partial W", font_size=34 if CLEAN else 30, color=PURPLE_).move_to([xs[2], lab_y, 0])
        W_lab = MathTex("W", font_size=36 if CLEAN else 32, color=GREEN_).move_to([xs[3], lab_y, 0])

        def loss_tex(v):
            return MathTex(rf"L = {v:.3f}", font_size=58 if CLEAN else 46, color=ORANGE_).move_to([xs[1], SY, 0])

        it = ValueTracker(1)
        if not CLEAN:
            counter = always_redraw(lambda: Text(f"iteration {int(round(it.get_value())):,}", font_size=28,
                                                 color=TEAL).to_edge(RIGHT, buff=0.5).match_y(hdr[0]))
        state = {"mb": stack(thumbs[0])}

        def forward(k, rt, pre=()):
            """the minibatch flies into the Model; the 10 scores of its first image come out"""
            mb = state["mb"]
            ref = picture(thumbs[k][0], 1.1 if CLEAN else 0.8, nearest=True).move_to([SX, SY, 0])
            ref = Group(ref, SurroundingRectangle(ref, buff=0.03, color=MUTED, stroke_width=1))
            bars = score_bars(shown[k]["scores"])
            anims = [FadeOut(c, target_position=boxes[0].get_left() + 0.3 * RIGHT, scale=0.15) for c in mb]
            anims.append(FadeIn(ref, target_position=mb[-1].get_center(), scale=0.9))
            if "ref" in state:
                anims.append(FadeOut(state["ref"]))
            self.play(*anims, *pre, run_time=0.6 * rt)
            if "bars" not in state:
                self.play(pulse(0), FadeIn(baseline), LaggedStart(*[GrowFromEdge(b, DOWN if s >= 0 else UP)
                          for b, s in zip(bars, shown[k]["scores"])], lag_ratio=0.06), run_time=0.8 * rt)
                state["bars"] = bars
            else:
                self.play(pulse(0), Transform(state["bars"], bars), run_time=0.6 * rt)
            state["ref"] = ref

        def loss_step(k, rt, pre=()):
            """the true-class score is what the loss looks at; the minibatch loss comes out"""
            bar = state["bars"][shown[k]["labels"][0]]
            self.play(bar.animate.set_fill(ORANGE_), *pre, run_time=0.3 * rt)
            path = [bar.get_top() + 0.1 * UP, [bar.get_x(), yr - 0.25, 0], [xs[1] - 0.7, yb, 0]]
            self.play(travel(path, ORANGE_, 0.5 * rt), run_time=0.5 * rt)
            new = loss_tex(shown[k]["loss"])
            if "loss" not in state:
                self.play(pulse(1), FadeIn(new, shift=0.2 * DOWN), run_time=0.5 * rt)
                state["loss"] = new
            else:
                self.play(pulse(1), Transform(state["loss"], new), run_time=0.5 * rt)

        def grad_step(k, rt, pre=()):
            src = state["loss"]
            path = [src.get_top() + 0.1 * UP + 0.5 * RIGHT, [xs[1] + 0.9, yr - 0.25, 0], [xs[2] - 0.7, yb, 0]]
            self.play(travel(path, PURPLE_, 0.45 * rt), *pre, run_time=0.45 * rt)
            anims = [pulse(2), FadeIn(grads[k], lag_ratio=0.05)]
            anims.append(FadeOut(state["grad"]) if "grad" in state else FadeIn(g_lab))
            self.play(*anims, run_time=0.6 * rt)
            state["grad"] = grads[k]

        def update_step(k, rt, pre=()):
            src = state["grad"]
            path = [src.get_top() + 0.1 * UP + 0.6 * RIGHT, [xs[2] + 0.9, yr - 0.25, 0], [xs[3] - 0.7, yb, 0]]
            first = [FadeIn(Wgrid, lag_ratio=0.05), FadeIn(W_lab)] if k == 0 else []
            self.play(travel(path, GREEN_, 0.45 * rt), *first, *pre, run_time=(0.7 if k == 0 else 0.45) * rt)
            fl = SurroundingRectangle(Wgrid, buff=0.08, color=GREEN_, stroke_width=4, corner_radius=0.06)
            self.play(pulse(3), FadeOut(src.copy(), target_position=Wgrid.get_center(), scale=0.6), FadeIn(fl),
                      run_time=0.6 * rt)
            for cell, a in zip(Wgrid, W_after[k]):               # W after this opt.step()
                cell.pixel_array = rgba(a)
            self.play(FadeOut(fl), run_time=0.35 * rt)

        def repeat_step(k_next, rt, pre=()):
            self.play(travel(ret_pts, TEAL, 0.9 * rt), it.animate(rate_func=linear).set_value(shown[k_next]["iteration"]),
                      *pre, run_time=0.9 * rt)
            state["mb"] = stack(thumbs[k_next])
            self.play(FadeIn(state["mb"], shift=0.3 * UP), run_time=0.4 * rt)

        def next_caption(old, text, color, part):
            new = caption(text, color=color)
            self.play(FadeOut(old, shift=0.1 * UP), FadeIn(new, shift=0.1 * UP), *focus(part), run_time=0.6)
            return new

        # ---- draw the loop
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * DOWN) for b in boxes], lag_ratio=0.15),
                  LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.15),
                  Create(ret_line), FadeIn(ret_tip), FadeIn(rep_bg), FadeIn(rep_lab), run_time=1.5)
        cap = caption("minibatch: 128 CIFAR-10 images", color=BLUE_)
        self.play(FadeIn(state["mb"], shift=0.4 * RIGHT), FadeIn(cap), *([] if CLEAN else [FadeIn(counter)]),
                  run_time=0.8)
        self.wait(0.4)

        # ---- iteration 1, slowly
        cap = next_caption(cap, "forward: scores", BLUE_, 0)
        forward(0, 1.0)
        self.wait(0.3)
        cap = next_caption(cap, "loss: how wrong", ORANGE_, 1)
        loss_step(0, 1.0)
        self.wait(0.3)
        cap = next_caption(cap, "backward: ∂L/∂W", PURPLE_, 2)
        grad_step(0, 1.0)
        self.wait(0.3)
        cap = next_caption(cap, "step: W ← W − α∂L/∂W", GREEN_, 3)
        update_step(0, 1.0)
        self.wait(0.2)
        cap = next_caption(cap, "repeat: next minibatch", TEAL, 4)
        repeat_step(1, 1.0)

        # ---- iterations 11 and 21, quickly
        for k, rt in [(1, 0.55), (2, 0.45)]:
            for part, step in enumerate([forward, loss_step, grad_step, update_step]):
                step(k, rt, pre=focus(part))
            if k == 1:
                repeat_step(2, rt, pre=focus(4))
        self.wait(0.3)

        # ---- fast-forward: 20 epochs
        afs = 26 if CLEAN else 22
        ax = Axes(x_range=[0, 20, 5], y_range=[1.4, 2.4, 0.2], x_length=7.6 if CLEAN else 7.3,
                  y_length=3.2 if CLEAN else 2.15,
                  axis_config={"stroke_color": MUTED, "include_tip": False, "font_size": afs},
                  x_axis_config={"numbers_to_include": [0, 5, 10, 15, 20]},
                  y_axis_config={"numbers_to_include": [1.6, 2.0, 2.4],
                                 "decimal_number_config": {"num_decimal_places": 1}}) \
            .move_to([-2.0, -1.75, 0] if CLEAN else [-2.1, -1.45, 0])
        for num in [*ax.x_axis.numbers, *ax.y_axis.numbers]:
            num.set_color(INK)
        xl = Text("epoch", font_size=afs, color=MUTED).next_to(ax.x_axis.get_right(), RIGHT, buff=0.15)
        yl = Text("loss", font_size=afs, color=ORANGE_).next_to(ax.y_axis.get_top(), UP, buff=0.1)
        il = np.array(hist["iter_loss"])
        il_it = np.array([e * ipe + 10 * j for e in range(20) for j in range(len(il) // 20)])   # 0-based iteration
        tl = np.array(hist["train_loss"])
        Wkeys = np.concatenate([[shown[2]["iteration"]], ipe * np.arange(1, 21)])               # iteration of each W
        Wimgs = np.array([W_after[2]] + W_epoch, dtype=float)

        def trace():
            k = it.get_value()
            n = max(1, int(np.searchsorted(il_it, k - 1, side="right")))
            pts = [ax.c2p(il_it[j] / ipe, il[j]) for j in range(n)]
            g = VGroup(polyline(pts, stroke_color=ORANGE_, stroke_width=1.6, stroke_opacity=0.4))
            ne = int(k // ipe)
            if ne >= 1:
                epts = [ax.c2p(e + 1, tl[e]) for e in range(ne)]
                g.add(polyline(epts, stroke_color=ORANGE_, stroke_width=4),
                      *[Dot(p, radius=0.045, color=ORANGE_) for p in epts])
            g.add(Dot(pts[-1], radius=0.06, color=ORANGE_))
            return g

        def set_W(_):
            k = it.get_value()
            j = int(np.clip(np.searchsorted(Wkeys, k, side="right") - 1, 0, len(Wkeys) - 2))
            f = np.clip((k - Wkeys[j]) / (Wkeys[j + 1] - Wkeys[j]), 0, 1)
            for cell, a in zip(Wgrid, (1 - f) * Wimgs[j] + f * Wimgs[j + 1]):
                cell.pixel_array = rgba(np.round(a))

        start_dot = Dot(ax.c2p(il_it[2] / ipe, il[2]), radius=0.06, color=ORANGE_)
        CELL2, CGAP2, WC = (0.7, 0.07, np.array([4.95, -1.75, 0])) if CLEAN else (0.6, 0.06, np.array([4.95, -1.35, 0]))
        cap = next_caption(cap, f"repeat: 20 epochs, {n_it:,} steps", TEAL, None)
        self.play(FadeOut(state["bars"]), FadeOut(baseline), FadeOut(state["ref"]), FadeOut(state["grad"]),
                  FadeOut(g_lab), Create(ax), FadeIn(xl), FadeIn(yl),
                  FadeOut(state["loss"], target_position=start_dot.get_center(), scale=0.1), FadeIn(start_dot),
                  *[cell.animate.scale(CELL2 / CELL).move_to(p) for cell, p in zip(Wgrid, grid_pos(WC, CELL2, CGAP2))],
                  W_lab.animate.move_to(WC + (CELL2 + CGAP2 / 2 + 0.3) * UP), run_time=1.2)
        tr = always_redraw(trace)
        self.remove(start_dot)
        self.add(tr)
        Wgrid.add_updater(set_W)
        phase = ValueTracker(0)
        runners = always_redraw(lambda: VGroup(*[Dot(ret_line.point_from_proportion((phase.get_value() + q / 3) % 1),
                                                     radius=0.07, color=TEAL) for q in range(3)]).set_z_index(5))
        self.add(runners)
        self.play(it.animate(rate_func=lambda u: 0.85 * u + 0.15 * smooth(u)).set_value(n_it),
                  phase.animate(rate_func=linear).set_value(6), run_time=6.0)
        Wgrid.clear_updaters()
        tr.clear_updaters()
        if CLEAN:      # no end value
            self.play(FadeOut(runners), FadeOut(tr[-1]), run_time=0.5)
        else:
            end = MathTex(f"{tl[-1]:.2f}", font_size=32, color=ORANGE_).next_to(ax.c2p(20, tl[-1]), RIGHT, buff=0.15)
            self.play(FadeOut(runners), FadeOut(tr[-1]), FadeIn(end, shift=0.1 * RIGHT), run_time=0.5)
        cap = swap_caption(self, cap, f"after 20 epochs: test accuracy {100 * hist['test_acc']:.1f}%", color=TEAL)
        self.wait(2.5)


# =====================================================================================================
class AdamBiasCorrection(Scene):
    """Why Adam divides by 1 − βᵗ: with a constant gradient g = 1 both moments should be 1, but they start at 0
    (β1 = 0.9, β2 = 0.999). The uncorrected step lr·m1/√m2 is 3.16·lr at t = 1, 6.57·lr at t = 12, 3.24·lr at
    t = 100, 1.26·lr at t = 1000; the corrected step lr·m̂1/√m̂2 is lr from step 1. Target: p2 Adam (full form) slide."""

    def construct(self):
        T = 3000
        a = adam_code.adam_constant_gradient(T)                     # code/adam_bias_correction.py, the deck's update
        full = np.stack([a[k] for k in ("t", "moment1", "moment2", "moment1_unbias", "moment2_unbias",
                                        "step_uncorrected", "step")], axis=1)
        at = lambda col, tt: float(full[int(tt) - 1, col])          # columns: t m1 m2 m1h m2h unc cor
        title_markup(self, "Adam's bias correction", f"constant gradient g = 1,  β<sub>1</sub> = {adam_code.beta1:g},"
                                                     f"  β<sub>2</sub> = {adam_code.beta2:g}")
        tpk = int(full[np.argmax(full[:, 5]), 0])                   # the peak of the uncorrected step
        keep = np.unique(np.concatenate([np.arange(1, 31), np.round(np.logspace(np.log10(30), np.log10(T), 160)),
                                         [100, 1000, T]])).astype(int)
        S = full[keep - 1]
        uu = np.log10(S[:, 0])
        U = np.log10(T)

        if CLEAN:      # slide version: taller panels, no values at the marked steps; curve labels and legend only
            XL, YL, PY, HY, XC, z = 5.6, 4.9, -0.4, 2.55, (-3.5, 3.6), 1.15
        else:
            XL, YL, PY, HY, XC, z = 5.3, 3.3, -0.6, 1.45, (-3.45, 3.55), 1

        def panel(xc, ymax, ystep, nums):
            ax = Axes(x_range=[0, U, 1], y_range=[0, ymax, ystep], x_length=XL, y_length=YL,
                      axis_config={"stroke_color": MUTED, "include_tip": False, "font_size": 22 * z},
                      y_axis_config={"numbers_to_include": nums,
                                     "decimal_number_config": {"num_decimal_places": 1 if ystep < 1 else 0}}
                      ).move_to([xc, PY, 0])
            for num in ax.y_axis.numbers:
                num.set_color(INK)
            xl = VGroup(*[MathTex(s, font_size=24 * z).next_to(ax.c2p(u, 0), DOWN, buff=0.15)
                          for u, s in [(0, "1"), (1, "10"), (2, "100"), (3, "1000")]])
            tl = MathTex("t", font_size=28 * z, color=MUTED).next_to(ax.c2p(U, 0), RIGHT, buff=0.15)
            return ax, VGroup(ax, xl, tl)

        axL, gL = panel(XC[0], 1.2, 0.5, [0.5, 1.0])
        axR, gR = panel(XC[1], 8, 1, [1, 2, 4, 6, 8])
        gL.add(Text("moments", font_size=30 if CLEAN else 24).move_to([axL.get_center()[0], HY, 0]))
        gR.add(Text("step size / lr", font_size=30 if CLEAN else 24).move_to([axR.get_center()[0], HY, 0]))

        def curve(ax, col, color, u_max=U, width=4):
            """polyline of column col up to log10(t) = u_max (end point interpolated)"""
            ys = S[:, col]
            n = int(np.searchsorted(uu, u_max, side="right"))
            pts = [ax.c2p(uu[j], ys[j]) for j in range(n)]
            if n < len(uu) and u_max > uu[n - 1]:
                f = (u_max - uu[n - 1]) / (uu[n] - uu[n - 1])
                pts.append(ax.c2p(u_max, ys[n - 1] + f * (ys[n] - ys[n - 1])))
            return polyline(pts, stroke_color=color, stroke_width=width)

        # ---- the two moments
        one_L = DashedLine(axL.c2p(0, 1), axL.c2p(U, 1), color=MUTED, stroke_width=2, dash_length=0.08)
        cap = caption("gradient g = 1 every step; moments start at 0")
        self.play(FadeIn(gL), FadeIn(cap), run_time=0.8)
        self.play(Create(one_L), run_time=0.5)
        u = ValueTracker(0)
        c1 = always_redraw(lambda: curve(axL, 1, BLUE_, u.get_value()))
        c2 = always_redraw(lambda: curve(axL, 2, PURPLE_, u.get_value()))
        h1 = always_redraw(lambda: Dot(c1.get_end(), radius=0.06, color=BLUE_))
        h2 = always_redraw(lambda: Dot(c2.get_end(), radius=0.06, color=PURPLE_))
        self.add(c1, c2, h1, h2)
        lab1 = MathTex("m_1", font_size=34 * z, color=BLUE_).move_to(axL.c2p(0.62, 0.86))
        lab2 = MathTex("m_2", font_size=34 * z, color=PURPLE_).move_to(axL.c2p(2.05, 0.27))
        self.play(u.animate(rate_func=linear, run_time=1.6).set_value(1.0), FadeIn(lab1, run_time=0.8))
        self.play(u.animate(rate_func=linear, run_time=2.6).set_value(U),
                  Succession(Wait(1.0), FadeIn(lab2, run_time=0.8)))
        for m in (c1, c2, h1, h2):
            m.clear_updaters()
        self.remove(h1, h2)
        p63 = Dot(axL.c2p(3, at(2, 1000)), radius=0.07, color=PURPLE_)
        p63l = MathTex(f"{at(2, 1000):.2f}", font_size=28, color=PURPLE_).next_to(p63, UP + LEFT, buff=0.06)
        cap = swap_caption(self, cap, f"moment2 is only {at(2, 1000):.2f} at t = 1000", color=PURPLE_)
        if not CLEAN:
            self.play(FadeIn(p63, scale=0.5), FadeIn(p63l), run_time=0.5)
        self.wait(1.2)

        # ---- the step size without correction
        one_R = DashedLine(axR.c2p(0, 1), axR.c2p(U, 1), color=MUTED, stroke_width=2, dash_length=0.08)
        self.play(FadeIn(gR), Create(one_R), run_time=0.8)
        v = ValueTracker(0)
        cr = always_redraw(lambda: curve(axR, 5, RED_, v.get_value()))
        hr = always_redraw(lambda: Dot(cr.get_end(), radius=0.06, color=RED_))
        leg_r = MathTex(r"m_1/\sqrt{m_2}", font_size=32 * z, color=RED_).move_to(axR.c2p(2.75, 6.6))
        p1 = Dot(axR.c2p(0, at(5, 1)), radius=0.07, color=RED_)
        p1l = MathTex(f"{at(5, 1):.2f}", font_size=28, color=RED_).next_to(p1, RIGHT, buff=0.12).shift(0.2 * DOWN)
        cap = swap_caption(self, cap, f"step 1:  {at(1, 1):.1f} / √{at(2, 1):.3f} = {at(5, 1):.2f} × lr", color=RED_)
        self.add(cr, hr)
        self.play(*([] if CLEAN else [FadeIn(p1, scale=0.5), FadeIn(p1l)]), FadeIn(leg_r), run_time=0.6)
        self.wait(0.6)
        for tt, run, side in [(tpk, 1.3, UP), (100, 1.2, UP + RIGHT), (1000, 1.2, UP + RIGHT)]:
            self.play(v.animate(rate_func=linear).set_value(np.log10(tt)), run_time=run)
            if CLEAN:  # no values: the curve keeps drawing
                continue
            p = Dot(axR.c2p(np.log10(tt), at(5, tt)), radius=0.07, color=RED_)
            txt = f"{at(5, tt):.2f}" + (rf"\ \text{{at}}\ t = {tt}" if tt == tpk else "")
            pl = MathTex(txt, font_size=28, color=RED_).next_to(p, side, buff=0.1)
            self.play(FadeIn(p, scale=0.5), FadeIn(pl), run_time=0.35)
        self.play(v.animate(rate_func=linear).set_value(U), run_time=0.8)
        cr.clear_updaters()
        self.remove(hr)
        cap = swap_caption(self, cap, f"uncorrected: steps up to {at(5, tpk):.1f}× too large", color=RED_)
        self.wait(1.2)

        # ---- the correction: the raw moments stay as faint ghosts
        cap = swap_caption(self, cap, "divide by 1 − βᵗ: both moments become exactly 1")
        flat1 = curve(axL, 3, BLUE_)
        flat2 = DashedVMobject(curve(axL, 4, PURPLE_), num_dashes=40)
        lab_h = MathTex(r"\hat m_1 = \hat m_2 = 1", font_size=34 * z).move_to(axL.c2p(1.75, 1.12))
        self.play(TransformFromCopy(c1, flat1), TransformFromCopy(c2, flat2),
                  c1.animate.set_stroke(opacity=0.25), c2.animate.set_stroke(opacity=0.25),
                  lab1.animate.set_opacity(0.35), lab2.animate.set_opacity(0.35),
                  *([] if CLEAN else [FadeOut(p63), FadeOut(p63l)]), FadeIn(lab_h, shift=0.1 * UP), run_time=1.4)
        self.wait(0.6)
        cg = curve(axR, 6, GREEN_, width=5)
        leg_g = MathTex(r"\hat m_1/\sqrt{\hat m_2}", font_size=32 * z, color=GREEN_).next_to(leg_r, DOWN, buff=0.25,
                                                                                          aligned_edge=LEFT)
        cap = swap_caption(self, cap, "corrected step: exactly lr from step 1", color=GREEN_)
        self.play(Create(cg), FadeIn(leg_g), run_time=1.3)
        excess = VMobject(stroke_width=0, fill_color=RED_, fill_opacity=0.12)
        excess.set_points_as_corners([axR.c2p(a, b) for a, b in zip(uu, S[:, 5])] +
                                     [axR.c2p(a, 1) for a in uu[::-1]] + [axR.c2p(uu[0], S[0, 5])])
        self.play(FadeIn(excess), run_time=0.6)
        self.wait(0.8)
        cap = swap_caption(self, cap, "without correction, early steps are too large")
        self.wait(3)


# =====================================================================================================
class MatrixBackprop(Scene):
    """Shapes make the formula. Forward S = XWᵀ + b with X: 128×3072, Wᵀ: 3072×10, S: 128×10. Backward: ∂L/∂S
    (128×10) arrives; ∂L/∂W must have W's shape 10×3072, and (∂L/∂S)ᵀX is the only product of the available
    matrices that fits (∂L/∂S·X is rejected); likewise ∂L/∂X = (∂L/∂S)W. Ends with the autograd check on a real
    CIFAR-10 minibatch (code/matrix_backprop.py) and the size of the Jacobian never formed. Target: p2 'Backprop with matrices'."""

    def construct(self):
        bp = mb_code.check()                        # code/matrix_backprop.py: formulas vs autograd, Jacobian size
        N = bp["N"]
        SZ = {"N": 1.25, "D": 2.9, "C": 0.42, "1": 0.13}       # drawn sizes: 3072 wide, 128 medium, 10 thin
        LAB = {"N": str(N), "D": str(bp["D"]), "C": str(bp["C"]), "1": "1"}
        n_n, n_d, n_c = LAB["N"], LAB["D"], LAB["C"]
        GCOL = PURPLE_
        clean_frame(self, center=(-1.3, 0.05), height=7.7)    # slide version: no formulas on the right, so centre
        title(self, "Backprop with matrices", f"one linear layer, a minibatch of N = {N} CIFAR-10 images")

        def block(rows, cols, name, color, inside=False, fill=0.16, dashed=False):
            h, w = SZ[rows], SZ[cols]
            if dashed:
                r = DashedVMobject(Rectangle(width=w, height=h, stroke_color=color, stroke_width=3),
                                   num_dashes=int(10 * (w + h)))
            else:
                r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=3, fill_color=color,
                              fill_opacity=fill)
            nm = MathTex(name, font_size=34, color=color)
            nm.move_to(r) if inside else nm.next_to(r, UP, buff=0.12)
            sh = VGroup(Text(LAB[rows], font_size=22), Text("×", font_size=22), Text(LAB[cols], font_size=22)
                        ).arrange(RIGHT, buff=0.08).next_to(r, DOWN, buff=0.12)
            g = VGroup(r, nm, sh)
            g.rect, g.name, g.shape = r, nm, sh
            return g

        def at(g, pos):
            """g positioned so that its rectangle is centred at pos"""
            return g.shift(np.array(pos, float) - g.rect.get_center())

        def to(g, pos):
            return g.animate.shift(np.array(pos, float) - g.rect.get_center())

        def op(s, pos):
            return MathTex(s, font_size=40).move_to(pos)

        def pop(*labels, color):
            self.play(*[m.animate.set_color(color).scale(1.3) for m in labels], run_time=0.35)
            self.play(*[m.animate.scale(1 / 1.3) for m in labels], run_time=0.3)

        def fill_slot(srcs, slot, res, extra=()):
            """the two factors pulse, the dashed slot fills in as the solid result"""
            self.play(*[s.animate(rate_func=there_and_back).scale(1.06) for s in srcs], FadeIn(res.rect, scale=0.9),
                      FadeOut(slot.rect), *extra, run_time=0.8)
            self.remove(slot.name, slot.shape)
            self.add(res)

        # ---- forward: S = X Wᵀ + b
        if CLEAN:      # three rows: the matrices on top, then dL/dW and dL/dX, so both results stay on screen
            Y0, TOP, YB, YB2 = 0.3, 2.45, 0.0, -2.5
        else:
            Y0, TOP, YB = 0.35, 1.15, -1.55
        S = at(block("N", "C", "S", BLUE_), [-3.2, Y0, 0])
        eq = op("=", [-2.55, Y0, 0])
        X = at(block("N", "D", "X", INK, inside=True), [-0.65, Y0, 0])
        dot1 = op(r"\cdot", [1.1, Y0, 0])
        Wt = at(block("D", "C", r"W^{\top}", BLUE_), [1.5, Y0, 0])
        plus = op("+", [2.15, Y0, 0])
        b = at(block("1", "C", "b", BLUE_), [2.7, Y0, 0])
        cap = caption("forward: S = XWᵀ + b", color=BLUE_)
        self.play(FadeIn(X), FadeIn(dot1), FadeIn(Wt), FadeIn(cap), run_time=0.7)
        self.play(FadeIn(plus), FadeIn(b), run_time=0.4)
        self.play(FadeIn(eq), ReplacementTransform(VGroup(X.rect.copy(), Wt.rect.copy()), S.rect),
                  FadeIn(S.name), FadeIn(S.shape), run_time=0.8)
        self.add(S)
        self.wait(0.3)
        cap = swap_caption(self, cap, f"inner sizes must match: {n_d} = {n_d}", color=GREEN_)
        pop(X.shape[2], Wt.shape[0], color=GREEN_)
        self.wait(0.5)

        # ---- backward: dL/dS arrives from the loss
        cap = swap_caption(self, cap, f"backward: ∂L/∂S arrives, shape {n_n} × {n_c}", color=PURPLE_)
        Lsym = MathTex("L", font_size=44, color=ORANGE_).move_to([-5.5, Y0, 0])
        ar = Arrow(Lsym.get_right() + 0.1 * RIGHT, S.rect.get_left() + 0.1 * LEFT, buff=0, color=PURPLE_,
                   stroke_width=5, max_tip_length_to_length_ratio=0.2)
        G = at(block("N", "C", r"\partial L/\partial S", GCOL, fill=0.3), S.rect.get_center())
        self.play(FadeIn(Lsym), GrowArrow(ar), run_time=0.6)
        self.play(ReplacementTransform(S, G), run_time=0.7)
        self.wait(0.3)

        # the available matrices go to the top row; Wᵀ turns back into W
        W = at(block("C", "D", "W", BLUE_), [2.05, TOP, 0])
        X.generate_target()
        X.target.shift(np.array([-2.95, TOP, 0]) - X.rect.get_center())
        X.target.shape[2].set_color(INK)
        self.play(FadeOut(VGroup(eq, dot1, plus, b, Lsym, ar, Wt.name, Wt.shape)), to(G, [-5.65, TOP, 0]),
                  MoveToTarget(X), run_time=0.8)
        self.play(Rotate(Wt.rect, -PI / 2), run_time=0.5)
        self.play(Wt.rect.animate.move_to(W.rect), run_time=0.45)
        self.play(FadeIn(W.name), FadeIn(W.shape), run_time=0.3)
        self.remove(Wt.rect)
        self.add(W)

        # ---- dL/dW: which product of the available matrices has W's shape?
        slot = at(block("C", "D", r"\partial L/\partial W", GCOL, dashed=True), [1.6, YB, 0])
        qm = MathTex("?", font_size=36, color=GCOL).move_to(slot.rect)
        cap = swap_caption(self, cap, f"∂L/∂W needs the shape of W: {n_c} × {n_d}", color=PURPLE_)
        self.play(FadeOut(W.rect.copy(), target_position=slot.rect.get_center()), FadeIn(slot), FadeIn(qm),
                  run_time=0.8)
        self.wait(0.4)

        # wrong attempt: dL/dS · X
        A, B = G.copy(), X.copy()
        dotB, eqB = op(r"\cdot", [-4.7, YB, 0]), op("=", [-0.45, YB, 0])
        self.play(to(A, [-5.65, YB, 0]), to(B, [-2.5, YB, 0]), FadeIn(dotB), FadeIn(eqB), run_time=0.8)
        cap = swap_caption(self, cap, f"∂L/∂S · X: inner sizes {n_c} ≠ {n_n}", color=RED_)
        cross = VGroup(Line(UL, DR), Line(DL, UR)).set_stroke(RED_, 7).scale(0.2).move_to(dotB)
        pop(A.shape[2], B.shape[0], color=RED_)
        self.play(Create(cross), Wiggle(A.rect, scale_value=1.05, rotation_angle=0.03 * TAU), run_time=0.6)
        self.wait(0.4)

        # transpose: (dL/dS)ᵀ · X
        At = at(block("C", "N", r"(\partial L/\partial S)^{\top}", GCOL, fill=0.3), [-5.45, YB, 0])
        self.play(FadeOut(cross), FadeOut(A.name), FadeOut(A.shape), B.shape[0].animate.set_color(INK),
                  Rotate(A.rect, -PI / 2), dotB.animate.move_to([-4.39, YB, 0]), run_time=0.7)
        self.play(A.rect.animate.move_to(At.rect), FadeIn(At.name), FadeIn(At.shape), run_time=0.5)
        self.remove(A.rect)
        self.add(At)
        cap = swap_caption(self, cap, f"transpose: (∂L/∂S)ᵀ X fits, {n_c} × {n_d}", color=PURPLE_)
        check = polyline([[-0.18, 0.0, 0], [-0.05, -0.14, 0], [0.2, 0.18, 0]], stroke_color=GREEN_,
                         stroke_width=7).move_to(dotB)
        pop(At.shape[2], B.shape[0], color=GREEN_)
        self.play(Create(check), run_time=0.4)
        res = at(block("C", "D", r"\partial L/\partial W", GCOL, fill=0.3), slot.rect.get_center())
        fill_slot([At.rect, B.rect], slot, res, extra=[FadeOut(qm)])
        if not CLEAN:
            f1 = MathTex(r"\frac{\partial L}{\partial W} = \Big(\frac{\partial L}{\partial S}\Big)^{\!\top} X",
                         font_size=36, color=GCOL).move_to([5.0, YB, 0])
            self.play(Write(f1), run_time=0.7)
        self.wait(0.6)

        # ---- dL/dX = dL/dS · W, by the same rule
        cap = swap_caption(self, cap, f"same rule: ∂L/∂X = (∂L/∂S) W, {n_n} × {n_d}", color=PURPLE_)
        if CLEAN:      # its own row below; the dL/dW row stays
            YX = YB2
            A2, B2 = G.copy(), W.copy()
            dotC, eqC = op(r"\cdot", [-4.7, YX, 0]), op("=", [-0.45, YX, 0])
            slot2 = at(block("N", "D", r"\partial L/\partial X", GCOL, dashed=True), [1.6, YX, 0])
            self.play(to(A2, [-5.65, YX, 0]), to(B2, [-2.5, YX, 0]), FadeIn(slot2), FadeIn(dotC), FadeIn(eqC),
                      run_time=0.7)
            check2 = check.copy().move_to(dotC)
        else:
            self.play(FadeOut(VGroup(At, B, check, res, f1)), dotB.animate.move_to([-4.7, YB, 0]), run_time=0.5)
            A2, B2 = G.copy(), W.copy()
            slot2 = at(block("N", "D", r"\partial L/\partial X", GCOL, dashed=True), [1.6, YB, 0])
            self.play(to(A2, [-5.65, YB, 0]), to(B2, [-2.5, YB, 0]), FadeIn(slot2), run_time=0.7)
            check2 = check.copy().move_to(dotB)
        pop(A2.shape[2], B2.shape[0], color=GREEN_)
        self.play(Create(check2), run_time=0.3)
        res2 = at(block("N", "D", r"\partial L/\partial X", GCOL, fill=0.3), slot2.rect.get_center())
        fill_slot([A2.rect, B2.rect], slot2, res2)
        if not CLEAN:
            f2 = MathTex(r"\frac{\partial L}{\partial X} = \frac{\partial L}{\partial S}\, W", font_size=36,
                         color=GCOL).move_to([5.0, YB, 0])
            self.play(Write(f2), run_time=0.6)
        self.wait(0.5)
        if CLEAN:      # no autograd check, no Jacobian: hold the two results
            self.wait(2.5)
            return

        # ---- check against autograd, and the Jacobian that is never formed
        f1.move_to([-2.6, -0.95, 0])
        e = bp["max_abs_err_dW"]
        ex = int(np.floor(np.log10(e)))
        chk = MathTex(rf"\max\,|\text{{formula}} - \text{{autograd}}| = {e / 10 ** ex:.1f}\times 10^{{{ex}}}",
                      font_size=32).move_to([0, -2.25, 0])
        self.play(FadeOut(VGroup(A2, B2, check2, res2, dotB, eqB)), f2.animate.move_to([2.6, -0.95, 0]), FadeIn(f1),
                  run_time=0.7)
        cap = swap_mcaption(self, cap, "checked vs autograd: max error 10<sup>−17</sup>", color=GREEN_)
        self.play(FadeIn(chk, shift=0.1 * UP), run_time=0.5)
        self.wait(1.0)
        J = Rectangle(width=13.4, height=5.2, stroke_color=MUTED, stroke_width=2, fill_color=SOFT, fill_opacity=0.97
                      ).move_to([0, -0.35, 0])
        Jg = VGroup(*[Line(J.get_corner(DL) + RIGHT * x, J.get_corner(UL) + RIGHT * x) for x in np.arange(0.2, 13.4, 0.2)],
                    *[Line(J.get_corner(DL) + UP * y, J.get_corner(DR) + UP * y) for y in np.arange(0.2, 5.2, 0.2)]
                    ).set_stroke(MUTED, 1, opacity=0.35)
        comma = lambda n: f"{n:,}".replace(",", "{,}")
        jl = MathTex(rf"\frac{{\partial S}}{{\partial W}}:\ {comma(N * 10)} \times {comma(10 * 3072)} = "
                     rf"{bp['jacobian_entries'] / 1e6:.1f}\,\text{{M entries}}\ \ ({bp['jacobian_MB_fp32']:.0f}\,\text{{MB}})",
                     font_size=40).move_to(J)
        jl.set_z_index(3)
        jl_bg = BackgroundRectangle(jl, color=SOFT, fill_opacity=1, buff=0.2).set_z_index(2)   # label stays above the cross
        cap = swap_caption(self, cap, f"never form the Jacobian: {bp['jacobian_entries'] / 1e6:.0f} M entries")
        self.play(GrowFromPoint(VGroup(J, Jg), f1.get_center()), run_time=0.8)
        self.play(FadeIn(jl_bg), FadeIn(jl), run_time=0.4)
        self.wait(1.2)
        jx = VGroup(Line(J.get_corner(UL), J.get_corner(DR)), Line(J.get_corner(DL), J.get_corner(UR))
                    ).set_stroke(RED_, 6, opacity=0.8)
        self.play(Create(jx), run_time=0.5)
        self.play(FadeOut(VGroup(J, Jg, jl_bg, jl, jx)), run_time=0.6)
        self.wait(2.2)
