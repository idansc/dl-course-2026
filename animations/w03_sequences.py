"""Week 3 (sequences): RNNs, the seq2seq bottleneck, subword tokenization. Render one scene:
    manim -qh --disable_caching -o RNNUnroll w03_sequences.py RNNUnroll
"""
from collections import Counter
from style import *


def div_color(v, scale=1.0):
    """Diverging colour for a signed number: negative → blue, 0 → white, positive → orange."""
    v = float(np.clip(v / scale, -1, 1))
    return interpolate_color(ManimColor(WHITE), ManimColor(ORANGE_ if v > 0 else BLUE_), abs(v))


def vec_column(v, cell=0.22, scale=1.0):
    """A vector drawn as a column of coloured squares."""
    g = VGroup(*[Square(cell, stroke_color=MUTED, stroke_width=1, fill_color=div_color(x, scale), fill_opacity=1)
                 for x in v])
    return g.arrange(DOWN, buff=0)


# ----------------------------------------------------------------------------------------------------------------------
class RNNUnroll(Scene):
    """Target: p6 (RNNs, vanishing gradients). A scalar tanh RNN unrolled over 8 steps; backprop multiplies by
    w·tanh'(a_t) at every step, so the gradient reaching early steps shrinks geometrically. Real numbers from the
    RNN below (gradients by torch autograd, checked against the product formula); sweeping w shows that a larger w
    does not help because tanh saturates."""

    X = np.array([0.8, -0.5, 0.3, 0.9, -0.2, 0.6, -0.7, 0.4])
    U = 1.0

    def run(self, w):
        """Forward pass h_t = tanh(w h_{t-1} + u x_t); returns h, factors w·tanh'(a_t), dh_T/dh_t."""
        h, hs = 0.0, []
        for x in self.X:
            h = np.tanh(w * h + self.U * x)
            hs.append(h)
        hs = np.array(hs)
        f = w * (1 - hs ** 2)                       # dh_t / dh_{t-1}
        g = np.ones(len(hs))
        for t in range(len(hs) - 2, -1, -1):
            g[t] = g[t + 1] * f[t + 1]
        return hs, f, g

    def construct(self):
        import torch
        w0 = 0.8
        # check the product formula against autograd once
        h = torch.zeros((), requires_grad=True)
        hs_t = []
        for x in self.X:
            h = torch.tanh(w0 * h + self.U * float(x))
            h.retain_grad()
            hs_t.append(h)
        hs_t[-1].backward()
        assert np.allclose([float(v.grad) for v in hs_t], self.run(w0)[2], atol=1e-6)

        title(self, "RNNs unrolled through time", "one cell, reused at every step")
        T = len(self.X)
        xs = [-6.0 + 1.45 * i for i in range(T)]
        cy = -0.75
        w = ValueTracker(w0)
        fwd = ValueTracker(0)        # how many steps of the forward pass are shown
        gp = ValueTracker(T)         # backward pulse position: bars i >= gp are shown

        rec = MathTex(r"h_t = \tanh(w\,h_{t-1} + u\,x_t)", font_size=34).to_corner(UR, buff=0.5)

        # --- rolled form
        rolled = RoundedRectangle(corner_radius=0.12, width=1.0, height=0.95, stroke_color=INK, stroke_width=3,
                                  fill_color=SOFT, fill_opacity=1).move_to([0, cy, 0])
        rl = MathTex("h_t", font_size=34).move_to(rolled)
        loop = CurvedArrow(rolled.get_right() + 0.2 * UP, rolled.get_left() + 0.2 * UP, angle=1.45 * PI,
                           color=PURPLE_, stroke_width=4)
        wl = MathTex("w", font_size=32, color=PURPLE_).next_to(loop, UP, buff=0.1)
        xin = MathTex("x_t", font_size=32, color=BLUE_).move_to([0, -2.3, 0])
        xar = Arrow(xin.get_top(), rolled.get_bottom(), buff=0.1, color=BLUE_, stroke_width=4)
        cap = caption("an RNN applies the same cell at every step, carrying a hidden state h")
        self.play(FadeIn(cap), FadeIn(rec), FadeIn(rolled), FadeIn(rl), Create(loop), FadeIn(wl), FadeIn(xin), GrowArrow(xar))
        self.wait(1.2)

        # --- unrolled form
        cells = VGroup(*[RoundedRectangle(corner_radius=0.12, width=1.0, height=0.95, stroke_color=INK, stroke_width=3,
                                          fill_color=SOFT, fill_opacity=1).move_to([x, cy, 0]) for x in xs])
        names = VGroup(*[MathTex(f"h_{i + 1}", font_size=30).move_to([x, cy + 0.2, 0]) for i, x in enumerate(xs)])
        xlab = VGroup(*[MathTex(rf"x_{i + 1}\!=\!{v:.1f}", font_size=26, color=BLUE_).move_to([x, -2.25, 0])
                        for i, (x, v) in enumerate(zip(xs, self.X))])
        xars = VGroup(*[Arrow([x, -2.0, 0], [x, cy - 0.48, 0], buff=0.05, color=BLUE_, stroke_width=3,
                              max_tip_length_to_length_ratio=0.25) for x in xs])
        hars = VGroup(*[Arrow([xs[i] + 0.5, cy, 0], [xs[i + 1] - 0.5, cy, 0], buff=0.02, color=PURPLE_, stroke_width=4,
                              max_tip_length_to_length_ratio=0.35) for i in range(T - 1)])
        cap = swap_caption(self, cap, "unrolled: the same weights w, u copied across time")
        self.play(ReplacementTransform(VGroup(rolled), cells), ReplacementTransform(rl, names),
                  FadeOut(loop), FadeOut(wl), ReplacementTransform(xin, xlab), ReplacementTransform(xar, xars),
                  run_time=1.5)
        self.add(names)
        self.play(LaggedStart(*[GrowArrow(a) for a in hars], lag_ratio=0.1), run_time=1)

        # --- forward pass: values appear left to right
        def hvals():
            hs = self.run(w.get_value())[0]
            g = VGroup()
            for i in range(T):
                if i < fwd.get_value() - 1e-6:
                    g.add(Text(f"{hs[i]:+.2f}", font_size=20, color=INK).move_to([xs[i], cy - 0.22, 0]))
            return g
        hv = always_redraw(hvals)
        self.add(hv)
        cap = swap_caption(self, cap, "forward: each h_t mixes the new input with the previous state")
        self.play(fwd.animate.set_value(T), run_time=3, rate_func=linear)

        loss = RoundedRectangle(corner_radius=0.1, width=0.7, height=0.7, stroke_color=RED_, stroke_width=3,
                                fill_color=WHITE, fill_opacity=1).move_to([5.8, cy, 0])
        ll = MathTex("L", font_size=34, color=RED_).move_to(loss)
        lar = Arrow([xs[-1] + 0.5, cy, 0], loss.get_left(), buff=0.02, color=RED_, stroke_width=4,
                    max_tip_length_to_length_ratio=0.3)
        self.play(GrowArrow(lar), FadeIn(loss), FadeIn(ll))

        # --- backward pass
        grad_f = MathTex(r"\frac{\partial h_8}{\partial h_t} = \prod_{k=t+1}^{8} w\,\tanh'(a_k)", font_size=32)
        grad_f.to_corner(UR, buff=0.4)
        base, maxh = 0.35, 1.55

        def factors():
            f = self.run(w.get_value())[1]
            g = VGroup()
            for i in range(1, T):        # factor on the arrow (i-1) -> i
                if gp.get_value() <= i - 1 + 1e-6:
                    g.add(Text(f"×{f[i]:.2f}", font_size=19, color=PURPLE_).move_to([(xs[i - 1] + xs[i]) / 2, 0.0, 0]))
            return g

        def gbars():
            gr = self.run(w.get_value())[2]
            g = VGroup()
            for i in range(T):
                if gp.get_value() <= i + 1e-6:
                    hgt = max(0.015, gr[i] * maxh)
                    r = Rectangle(width=0.6, height=hgt, stroke_width=0, fill_color=RED_, fill_opacity=0.85)
                    r.move_to([xs[i], base + hgt / 2, 0])
                    s = f"{gr[i]:.2f}" if gr[i] >= 0.095 else (f"{gr[i]:.3f}" if gr[i] >= 0.00095 else f"{gr[i]:.0e}")
                    g.add(r, Text(s, font_size=20, color=RED_).next_to(r, UP, buff=0.07))
            return g

        def pulse():
            p = float(np.clip(gp.get_value(), 0, T - 1))
            if gp.get_value() >= T - 1e-6 or gp.get_value() <= 1e-6:
                return VGroup()
            return Dot([np.interp(p, range(T), xs), cy + 0.55, 0], radius=0.09, color=RED_)

        fb, gb, pd = always_redraw(factors), always_redraw(gbars), always_redraw(pulse)
        self.add(fb, gb, pd)
        cap = swap_caption(self, cap, "backward: each step back multiplies the gradient by w·tanh'(a_t) < 1")
        self.play(FadeOut(rec), FadeIn(grad_f))
        self.play(gp.animate.set_value(T - 1), run_time=0.6)
        self.play(gp.animate.set_value(0), run_time=5, rate_func=linear)
        g1 = self.run(w0)[2][0]
        cap = swap_caption(self, cap, f"the product shrinks geometrically: only {g1:.3f} of the signal reaches step 1",
                           color=RED_)
        self.wait(2.2)

        # --- sweep w
        wread = always_redraw(lambda: Text(f"w = {w.get_value():.2f}", font_size=28, color=PURPLE_)
                              .next_to(grad_f, DOWN, buff=0.15).align_to(grad_f, RIGHT))
        self.play(FadeIn(wread))
        cap = swap_caption(self, cap, "smaller w: every factor shrinks, the gradient vanishes faster")
        self.play(w.animate.set_value(0.5), run_time=2.5)
        self.wait(0.8)
        cap = swap_caption(self, cap, "larger w does not help: tanh saturates (h → ±1), tanh' → 0")
        self.play(w.animate.set_value(2.0), run_time=3.5)
        self.wait(1.2)
        self.play(w.animate.set_value(w0), run_time=1.5)
        cap = swap_caption(self, cap, "early inputs get almost no credit: the motivation for LSTMs, GRUs and attention",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------------
class Seq2SeqBottleneck(Scene):
    """Target: p7 2–17. An encoder RNN reads "we are eating bread" and the decoder must translate from one context
    vector (the bottleneck). Then attention: each decoder word mixes all encoder states with alignment weights,
    drawn as arc thickness and filled into the alignment matrix. Hidden states come from a small random-weight tanh
    RNN (real arithmetic, untrained); the alignment weights are illustrative (a trained model gives this pattern)."""

    def construct(self):
        title(self, "Seq2seq: bottleneck, then attention", "English → Spanish with an encoder–decoder RNN")
        src, tgt = ["we", "are", "eating", "bread"], ["estamos", "comiendo", "pan"]
        d = 4
        rng = np.random.default_rng(7)
        E = rng.normal(size=(len(src), d))
        Wh, Ux = rng.normal(size=(d, d)) * 0.6, rng.normal(size=(d, d)) * 0.9
        H, h = [], np.zeros(d)
        for e in E:
            h = np.tanh(Wh @ h + Ux @ e)
            H.append(h)
        H = np.array(H)
        # alignment: "estamos" ← we are, "comiendo" ← eating, "pan" ← bread (softmax of illustrative scores)
        A = softmax(np.array([[2.0, 2.1, -0.2, -0.6], [-0.9, -0.2, 2.6, -0.4], [-1.2, -1.1, 0.2, 2.9]]))

        cy, wy = -0.95, -2.2
        ex = [-6.2, -5.0, -3.8, -2.6]
        cx = -1.0
        dx = [0.75, 2.35, 3.95]

        def cell(v, x, color=INK):
            col = vec_column(v)
            box = RoundedRectangle(corner_radius=0.08, width=0.5, height=col.height + 0.22, stroke_color=color,
                                   stroke_width=3, fill_color=WHITE, fill_opacity=1)
            return VGroup(box, col).move_to([x, cy, 0])

        enc = VGroup(*[cell(H[i], ex[i], BLUE_) for i in range(4)])
        words = VGroup(*[Text(w, font_size=24, color=BLUE_).move_to([ex[i], wy, 0]) for i, w in enumerate(src)])
        warr = VGroup(*[Arrow([ex[i], wy + 0.22, 0], enc[i].get_bottom(), buff=0.06, color=BLUE_, stroke_width=3,
                              max_tip_length_to_length_ratio=0.3) for i in range(4)])
        earr = VGroup(*[Arrow(enc[i].get_right(), enc[i + 1].get_left(), buff=0.04, color=MUTED, stroke_width=3,
                              max_tip_length_to_length_ratio=0.35) for i in range(3)])
        enc_lab = Text("encoder", font_size=22, color=BLUE_).next_to(enc, UP, buff=0.25)

        cap = caption("the encoder reads the sentence one word at a time")
        self.play(FadeIn(cap), FadeIn(enc_lab))
        for i in range(4):
            anims = [FadeIn(words[i], shift=0.1 * UP), GrowArrow(warr[i]), FadeIn(enc[i])]
            if i:
                anims.append(GrowArrow(earr[i - 1]))
            self.play(*anims, run_time=0.55)

        ctx = cell(H[-1], cx, PURPLE_)
        ctx[0].set_stroke(PURPLE_, 4)
        clab = MathTex("c", font_size=34, color=PURPLE_).next_to(ctx, UP, buff=0.15)
        carr = Arrow(enc[-1].get_right(), ctx.get_left(), buff=0.04, color=PURPLE_, stroke_width=4,
                     max_tip_length_to_length_ratio=0.3)
        cap = swap_caption(self, cap, "the whole sentence must fit into one fixed-size context vector c", color=PURPLE_)
        squeeze = [enc[i][1].copy() for i in range(4)]
        self.play(GrowArrow(carr), FadeIn(ctx[0]), FadeIn(clab),
                  *[ReplacementTransform(squeeze[i], ctx[1].copy()) for i in range(4)], run_time=1.4)
        self.add(ctx)
        self.wait(1.0)

        # decoder from c only
        dec_states, s = [], H[-1].copy()
        Wd = rng.normal(size=(d, d)) * 0.7
        for i in range(3):
            s = np.tanh(Wd @ s + 0.5 * H[-1])
            dec_states.append(s)
        dec = VGroup(*[cell(dec_states[i], dx[i], GREEN_) for i in range(3)])
        dwords = VGroup(*[Text(w, font_size=24, color=GREEN_).move_to([dx[i], wy, 0]) for i, w in enumerate(tgt)])
        dwarr = VGroup(*[Arrow(dec[i].get_bottom(), [dx[i], wy + 0.22, 0], buff=0.06, color=GREEN_, stroke_width=3,
                               max_tip_length_to_length_ratio=0.3) for i in range(3)])
        darr = VGroup(Arrow(ctx.get_right(), dec[0].get_left(), buff=0.04, color=PURPLE_, stroke_width=4,
                            max_tip_length_to_length_ratio=0.3),
                      *[Arrow(dec[i].get_right(), dec[i + 1].get_left(), buff=0.04, color=MUTED, stroke_width=3,
                              max_tip_length_to_length_ratio=0.35) for i in range(2)])
        dec_lab = Text("decoder", font_size=22, color=GREEN_).next_to(dec, UP, buff=0.25)
        cap = swap_caption(self, cap, "the decoder generates the translation from c alone")
        self.play(FadeIn(dec_lab))
        for i in range(3):
            self.play(GrowArrow(darr[i]), FadeIn(dec[i]), GrowArrow(dwarr[i]), FadeIn(dwords[i], shift=0.1 * DOWN),
                      run_time=0.55)
        cap = swap_caption(self, cap, "long sentences overflow c: early words are forgotten", color=RED_)
        self.play(*[enc[i].animate.set_opacity(0.35) for i in range(3)], run_time=0.8)
        self.wait(1.2)
        self.play(*[enc[i].animate.set_opacity(1) for i in range(3)], run_time=0.5)

        # --- attention
        cap = swap_caption(self, cap, "attention: keep every encoder state and let each output word look back")
        k = ValueTracker(0)                      # which decoder step is looking (continuous)

        def alpha():
            t = float(np.clip(k.get_value(), 0, 2))
            i = int(min(np.floor(t), 1))
            return (1 - (t - i)) * A[i] + (t - i) * A[i + 1], t

        def arcs():
            a, t = alpha()
            x0 = np.interp(t, [0, 1, 2], dx)
            g = VGroup()
            for j in range(4):
                g.add(ArcBetweenPoints([x0, cy + 0.68, 0], [ex[j], cy + 0.68, 0], angle=PI / 2.6,
                                       color=PURPLE_, stroke_width=1.5 + 13 * a[j], stroke_opacity=0.35 + 0.65 * a[j]))
                g.add(Text(f"α={a[j]:.2f}", font_size=20, color=PURPLE_).move_to([ex[j], wy - 0.5, 0]))
            return g

        def ctx_now():
            a, _ = alpha()
            c = cell(a @ H, cx, PURPLE_)
            c[0].set_stroke(PURPLE_, 4)
            return c

        def focus():
            _, t = alpha()
            return SurroundingRectangle(VGroup(dec[0][0]).copy().move_to([np.interp(t, [0, 1, 2], dx), cy, 0]),
                                        color=PURPLE_, buff=0.07, stroke_width=4)

        clab2 = MathTex(r"c_t = \sum_i \alpha_{t,i}\, h_i", font_size=26, color=PURPLE_).move_to([cx, wy - 0.05, 0])
        # alignment matrix (top right)
        cellw = 0.36
        am = heatmap(np.zeros((3, 4)), cell=cellw).move_to([5.75, 1.2, 0])
        am_cols = VGroup(*[Text(w, font_size=16, color=BLUE_).rotate(PI / 3).next_to(am[j], UP, buff=0.08)
                           for j, w in enumerate(src)])
        am_rows = VGroup(*[Text(w, font_size=16, color=GREEN_).next_to(am[i * 4], LEFT, buff=0.1)
                           for i, w in enumerate(tgt)])
        ar, cn, fo = always_redraw(arcs), always_redraw(ctx_now), always_redraw(focus)
        self.play(FadeOut(enc_lab), FadeOut(dec_lab), FadeOut(clab), FadeOut(carr), FadeOut(darr[0]),
                  FadeIn(am), FadeIn(am_cols), FadeIn(am_rows))
        self.remove(ctx)
        self.add(cn)
        self.play(FadeIn(ar), FadeIn(fo), FadeIn(clab2))
        self.play(*[am[j].animate.set_fill(heat_color(A[0, j])) for j in range(4)], run_time=0.6)
        cap = swap_caption(self, cap, '"estamos" looks back at "we are": weights shown as line thickness')
        self.wait(1.4)
        texts = {1: '"comiendo" looks at "eating": the context vector is recomputed per word',
                 2: '"pan" looks at "bread": no single vector has to hold the whole sentence'}
        for t in (1, 2):
            cap = swap_caption(self, cap, texts[t])
            self.play(k.animate.set_value(t), run_time=1.6)
            self.play(*[am[t * 4 + j].animate.set_fill(heat_color(A[t, j])) for j in range(4)], run_time=0.5)
            self.wait(1.4)
        cap = swap_caption(self, cap, "the alignment matrix α is learned, not given: soft, differentiable alignment",
                           color=PURPLE_, size=26)
        self.play(Indicate(am, color=PURPLE_, scale_factor=1.08))
        self.wait(2.2)


# ----------------------------------------------------------------------------------------------------------------------
class BPEMerges(Scene):
    """Target: T04 (tokenization). Byte-pair encoding on the classic toy corpus (Sennrich et al. 2016): count adjacent
    pairs weighted by word frequency, merge the most frequent, repeat. The merges are computed in the scene; at the
    end the learned merges split an unseen word ("lowest" → low + est_)."""

    CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    N_MERGES = 9

    @staticmethod
    def pair_counts(seg, counts):
        c = Counter()
        for w, n in counts.items():
            s = seg[w]
            for a, b in zip(s, s[1:]):
                c[(a, b)] += n
        return c

    @staticmethod
    def apply(s, pair):
        out, i, hit = [], 0, []
        while i < len(s):
            if i < len(s) - 1 and (s[i], s[i + 1]) == pair:
                out.append(s[i] + s[i + 1])
                hit.append(i)
                i += 2
            else:
                out.append(s[i])
                i += 1
        return out, hit

    def construct(self):
        title(self, "Byte-pair encoding", "learn subword tokens by merging the most frequent pair")
        counts = self.CORPUS
        seg = {w: list(w) + ["_"] for w in counts}
        words = list(counts)
        ys = [1.45, 0.45, -0.55, -1.55]
        x0 = -5.5

        def row(tokens, y, hl=()):
            r = token_row(tokens, size=24, buff=0.08)
            for i in hl:
                r[i][0].set_stroke(PURPLE_, 3).set_fill(heat_color(0.25))
            return r.move_to([x0, y, 0], aligned_edge=LEFT)

        hdr_c = Text("corpus (word × count)", font_size=22, color=MUTED).move_to([-4.1, 2.2, 0])
        cnts = VGroup(*[Text(f"×{counts[w]}", font_size=24, color=MUTED).move_to([-6.25, ys[i], 0])
                        for i, w in enumerate(words)])
        rows = [row(seg[w], ys[i]) for i, w in enumerate(words)]

        vx = 3.7
        hdr_v = Text("vocabulary", font_size=22, color=MUTED).move_to([vx, 2.2, 0], aligned_edge=LEFT)
        base = sorted({ch for w in words for ch in seg[w]})
        base_chips = token_row(base, size=20, buff=0.06)
        base_chips = VGroup(VGroup(*base_chips[:6]).copy(), VGroup(*base_chips[6:]).copy())
        base_chips[0].arrange(RIGHT, buff=0.06).move_to([vx, 1.7, 0], aligned_edge=LEFT)
        base_chips[1].arrange(RIGHT, buff=0.06).move_to([vx, 1.2, 0], aligned_edge=LEFT)
        vsize = [len(base)]
        vcount = always_redraw(lambda: Text(f"size {vsize[0]}", font_size=22, color=PURPLE_)
                               .next_to(hdr_v, RIGHT, buff=0.2))

        cap = caption("start: every word is split into characters (_ marks the end of a word)")
        self.play(FadeIn(cap), FadeIn(hdr_c), FadeIn(cnts), *[FadeIn(r) for r in rows])
        self.play(FadeIn(hdr_v), FadeIn(base_chips), FadeIn(vcount))
        self.wait(1.0)

        px = 0.0
        hdr_p = Text("pair counts", font_size=22, color=MUTED).move_to([px + 1.3, 2.2, 0])

        def pair_panel(c):
            top = sorted(c.items(), key=lambda kv: -kv[1])[:5]
            g = VGroup()
            for r, ((a, b), n) in enumerate(top):
                y = 1.55 - 0.62 * r
                lab = Text(f"{a} {b}", font_size=22, color=PURPLE_ if r == 0 else INK)
                lab.move_to([px + 0.15, y, 0], aligned_edge=RIGHT)
                bar = Rectangle(width=0.25 * n, height=0.36, stroke_width=0,
                                fill_color=PURPLE_ if r == 0 else MUTED, fill_opacity=0.9)
                bar.move_to([px + 0.3, y, 0], aligned_edge=LEFT)
                g.add(lab, bar, Text(str(n), font_size=20, color=INK).next_to(bar, RIGHT, buff=0.1))
            return g

        c = self.pair_counts(seg, counts)
        pp = pair_panel(c)
        cap = swap_caption(self, cap, "count every adjacent pair, weighted by how often its word occurs")
        self.play(FadeIn(hdr_p), FadeIn(pp, shift=0.1 * RIGHT))
        self.wait(1.2)

        merges, rules = VGroup(), []
        for m in range(self.N_MERGES):
            c = self.pair_counts(seg, counts)
            best = max(c.items(), key=lambda kv: kv[1])[0]
            n_best = c[best]
            if m:
                new_pp = pair_panel(c)
                self.play(FadeOut(pp), FadeIn(new_pp), run_time=0.35 if m > 2 else 0.6)
                pp = new_pp
            # highlight occurrences
            hits = {w: self.apply(seg[w], best)[1] for w in words}
            boxes = VGroup()
            for i, w in enumerate(words):
                for j in hits[w]:
                    boxes.add(SurroundingRectangle(VGroup(rows[i][j], rows[i][j + 1]), color=PURPLE_, buff=0.05,
                                                   stroke_width=4))
            fast = m > 2
            if m == 0:
                cap = swap_caption(self, cap, f"merge the most frequent pair into one new token: {best[0]} {best[1]} → "
                                              f"{best[0] + best[1]}  ({n_best}×)", color=PURPLE_)
            elif m == 1:
                cap = swap_caption(self, cap, "recount and repeat: merged tokens can merge again")
            elif m == 3:
                cap = swap_caption(self, cap, "each merge adds one token to the vocabulary")
            self.play(Create(boxes), run_time=0.3 if fast else 0.6)
            # merge animation per row
            anims = []
            for i, w in enumerate(words):
                new, hit = self.apply(seg[w], best)
                if not hit:
                    continue
                nr = row(new, ys[i])
                k_old = 0
                for k_new in range(len(new)):
                    if k_old in hit:
                        anims.append(ReplacementTransform(VGroup(rows[i][k_old], rows[i][k_old + 1]), nr[k_new]))
                        k_old += 2
                    else:
                        anims.append(ReplacementTransform(rows[i][k_old], nr[k_new]))
                        k_old += 1
                rows[i] = nr
                seg[w] = new
            entry = Text(f"{best[0]} {best[1]} → {best[0] + best[1]}", font_size=20,
                         t2c={f"→ {best[0] + best[1]}": PURPLE_})
            entry.move_to([vx, 0.62 - 0.4 * m, 0], aligned_edge=LEFT)
            vsize[0] += 1
            self.play(*anims, FadeOut(boxes), FadeIn(entry, shift=0.1 * LEFT), run_time=0.5 if fast else 0.9)
            merges.add(entry)
            rules.append(best)
            self.wait(0.25 if fast else 0.8)

        cap = swap_caption(self, cap, "frequent words become single tokens; rarer words stay as reusable pieces")
        self.play(*[Indicate(rows[i], color=PURPLE_, scale_factor=1.05) for i in (0, 2)])
        self.wait(1.2)

        # apply learned merges to an unseen word
        self.play(FadeOut(pp), FadeOut(hdr_p))
        new_word = list("lowest") + ["_"]
        nr = row(new_word, -2.55)
        lab = Text("new:", font_size=24, color=MUTED).move_to([-6.25, -2.55, 0])
        cap = swap_caption(self, cap, 'an unseen word is split by replaying the merges in order: "lowest"')
        self.play(FadeIn(lab), FadeIn(nr))
        seq = new_word
        for entry, (a, b) in zip(merges, rules):
            new, hit = self.apply(seq, (a, b))
            if not hit:
                continue
            r2 = row(new, -2.55)
            anims, k_old = [], 0
            for k_new in range(len(new)):
                if k_old in hit:
                    anims.append(ReplacementTransform(VGroup(nr[k_old], nr[k_old + 1]), r2[k_new]))
                    k_old += 2
                else:
                    anims.append(ReplacementTransform(nr[k_old], r2[k_new]))
                    k_old += 1
            self.play(Indicate(entry, color=PURPLE_, scale_factor=1.15), *anims, run_time=0.7)
            nr, seq = r2, new
        cap = swap_caption(self, cap, f'"lowest" → {" + ".join(seq)}: no unknown words, a fixed-size vocabulary',
                           color=PURPLE_)
        self.wait(2.5)
