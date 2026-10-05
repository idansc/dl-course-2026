"""Week 3–4 (attention, Transformers). Render one scene:
    manim -qh w03_attention.py SoftLookup
"""
import json
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


class SoftLookup(Scene):
    """Attention as a soft dictionary lookup: rotate the query, watch scores, weights and the output move."""

    def construct(self):
        title(self, "Attention = soft lookup", "compare a query with every key, average the values")
        keys = np.array([[2.0, 0.6], [1.4, 1.6], [-1.6, 1.1], [-0.6, -1.9]])
        names = ["cat", "dog", "car", "tree"]
        vals = np.array([[1.6, 0.4], [1.2, 1.3], [-1.5, 0.8], [0.2, -1.6]])
        ang = ValueTracker(0.35)

        plane = NumberPlane(x_range=[-2.5, 2.5], y_range=[-2.5, 2.5], x_length=4.2, y_length=4.2,
                            background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                            axis_config={"stroke_color": MUTED}).shift(4.6 * LEFT + 0.2 * DOWN)
        hdr1 = Text("query q and keys k", font_size=24).next_to(plane, UP, buff=0.15)
        karrows = VGroup(*[Arrow(plane.c2p(0, 0), plane.c2p(*k), buff=0, color=ORANGE_, stroke_width=5)
                           for k in keys])
        klabels = VGroup(*[Text(n, font_size=22, color=ORANGE_).next_to(plane.c2p(*k), (k / np.linalg.norm(k)).tolist() + [0], buff=0.1)
                           for k, n in zip(keys, names)])

        def q_vec():
            a = ang.get_value()
            return 2.0 * np.array([np.cos(a), np.sin(a)])

        qarrow = always_redraw(lambda: Arrow(plane.c2p(0, 0), plane.c2p(*q_vec()), buff=0, color=BLUE_, stroke_width=7))
        qlabel = always_redraw(lambda: Text("q", font_size=28, color=BLUE_, weight=BOLD).move_to(plane.c2p(*(q_vec() * 1.22))))

        self.play(Create(plane), FadeIn(hdr1))
        self.play(LaggedStart(*[GrowArrow(a) for a in karrows], lag_ratio=0.15), FadeIn(klabels))
        self.play(GrowArrow(qarrow.copy()), run_time=0.5)
        self.add(qarrow, qlabel)

        # middle: scores and softmax weights
        base_x = -0.9
        def score_bars():
            s = keys @ q_vec()
            g = VGroup()
            for i, v in enumerate(s):
                h = abs(v) / 4.5 * 0.8
                r = Rectangle(width=0.42, height=max(h, 0.02), stroke_width=0,
                              fill_color=ORANGE_ if v >= 0 else MUTED, fill_opacity=0.9)
                r.move_to([base_x + i * 0.6, 0.9 + (h / 2 if v >= 0 else -h / 2), 0])
                g.add(r)
            return g

        def weight_bars():
            w = softmax(keys @ q_vec())
            g = VGroup()
            for i, v in enumerate(w):
                h = v * 1.35
                r = Rectangle(width=0.42, height=max(h, 0.02), stroke_width=0, fill_color=PURPLE_, fill_opacity=0.9)
                r.move_to([base_x + i * 0.6, -2.3 + h / 2, 0])
                g.add(r, Text(f"{v:.2f}", font_size=18, color=PURPLE_).next_to(r, UP, buff=0.05))
            return g

        sline = Line([base_x - 0.4, 0.9, 0], [base_x + 2.2, 0.9, 0], color=MUTED, stroke_width=2)
        wline = Line([base_x - 0.4, -2.3, 0], [base_x + 2.2, -2.3, 0], color=MUTED, stroke_width=2)
        snames = VGroup(*[Text(n, font_size=18, color=MUTED).move_to([base_x + i * 0.6, -2.55, 0]) for i, n in enumerate(names)])
        shdr = MathTex(r"s_i = q\cdot k_i", font_size=34).move_to([base_x + 0.9, 2.05, 0])
        whdr = MathTex(r"w = \mathrm{softmax}(s)", font_size=34).move_to([base_x + 0.9, -0.3, 0])
        sb, wb = always_redraw(score_bars), always_redraw(weight_bars)

        cap = caption("1. score: dot product of the query with every key")
        self.play(FadeIn(cap), FadeIn(shdr), Create(sline), FadeIn(sb))
        self.wait(0.6)
        cap = swap_caption(self, cap, "2. softmax turns scores into weights that sum to 1")
        self.play(FadeIn(whdr), Create(wline), FadeIn(wb), FadeIn(snames))
        self.wait(0.6)

        # right: values and the weighted average
        vplane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], x_length=3.4, y_length=3.4,
                             background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                             axis_config={"stroke_color": MUTED}).shift(4.9 * RIGHT + 0.2 * DOWN)
        vhdr = Text("values v and output", font_size=24).next_to(vplane, UP, buff=0.15)
        varrows = VGroup(*[Arrow(vplane.c2p(0, 0), vplane.c2p(*v), buff=0, color=GREEN_, stroke_width=4,
                                 stroke_opacity=0.45) for v in vals])
        vlabels = VGroup(*[Text(nm, font_size=18, color=GREEN_).set_opacity(0.7).next_to(vplane.c2p(*v), (v / np.linalg.norm(v)).tolist() + [0], buff=0.08)
                           for v, nm in zip(vals, names)])
        out = always_redraw(lambda: Arrow(vplane.c2p(0, 0), vplane.c2p(*(softmax(keys @ q_vec()) @ vals)), buff=0,
                                          color=INK, stroke_width=8))
        ohdr = MathTex(r"z = \sum_i w_i\, v_i", font_size=34).next_to(vplane, DOWN, buff=0.2)
        cap = swap_caption(self, cap, "3. output = weighted average of the values")
        self.play(Create(vplane), FadeIn(vhdr), LaggedStart(*[GrowArrow(a) for a in varrows], lag_ratio=0.1), FadeIn(vlabels))
        self.play(FadeIn(out), FadeIn(ohdr))
        self.wait(0.5)

        cap = swap_caption(self, cap, "move the query: the weights and the output follow smoothly")
        self.play(ang.animate.set_value(0.35 + 2 * PI), run_time=9, rate_func=linear)
        self.play(ang.animate.set_value(0.35 + 2 * PI + 0.55), run_time=1.5)
        cap = swap_caption(self, cap, "a soft, differentiable dictionary lookup", color=PURPLE_)
        self.wait(2)


class ScaledScores(Scene):
    """Why divide by sqrt(d): dot products of random vectors grow like sqrt(d) and saturate the softmax."""

    def construct(self):
        title(self, "Why scale by √d", "random q and 8 keys, unit-variance entries; avg over 1000 draws")
        rng = np.random.default_rng(3)
        dims, cols = [4, 64, 512], [-4.6, 0.0, 4.6]
        bw, gap = 0.2, 0.07

        def weight_bars(w, x, y0, color):
            g = VGroup()
            for i, v in enumerate(w):
                h = max(0.02, v * 1.3)
                g.add(Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
                      .move_to([x + (i - 3.5) * (bw + gap), y0 + h / 2, 0]))
            g.add(Line([x - 1.15, y0, 0], [x + 1.15, y0, 0], color=MUTED, stroke_width=2))
            return g

        rows = []
        for d, x in zip(dims, cols):
            draws = [(rng.standard_normal(d), rng.standard_normal((8, d))) for _ in range(1000)]
            mx = np.array([softmax(K @ q).max() for q, K in draws])
            mx_sc = np.array([softmax(K @ q / np.sqrt(d)).max() for q, K in draws])
            q, K = draws[int(np.argmin(np.abs(mx - mx.mean())))]      # a representative draw
            s_ = K @ q
            raw, sc = softmax(s_), softmax(s_ / np.sqrt(d))
            hdr = MathTex(rf"d = {d}", font_size=38).move_to([x, 1.95, 0])
            std = Text(f"scores spread ≈ ±{np.sqrt(d):.0f}", font_size=20, color=MUTED).move_to([x, 1.5, 0])
            gr = weight_bars(raw, x, -0.05, RED_)
            lr = Text(f"softmax(q·k)   avg max {mx.mean():.2f}", font_size=19, color=RED_).move_to([x, -0.3, 0])
            gs = weight_bars(sc, x, -2.35, PURPLE_)
            ls = Text(f"softmax(q·k/√d)   avg max {mx_sc.mean():.2f}", font_size=19, color=PURPLE_).move_to([x, -2.6, 0])
            rows.append((hdr, std, gr, lr, gs, ls))
        cap = caption("dot products of random d-dim vectors have spread √d")
        self.play(FadeIn(cap), *[FadeIn(r[0]) for r in rows], *[FadeIn(r[1]) for r in rows])
        self.wait(0.8)
        cap = swap_caption(self, cap, "without scaling, the softmax collapses onto one key as d grows")
        self.play(*[FadeIn(r[2], shift=0.2 * UP) for r in rows], *[FadeIn(r[3]) for r in rows], run_time=1.2)
        self.wait(1.8)
        cap = swap_caption(self, cap, "divide by √d: scores stay O(1), weights stay spread")
        self.play(*[FadeIn(r[4], shift=0.2 * UP) for r in rows], *[FadeIn(r[5]) for r in rows], run_time=1.2)
        self.wait(1.8)
        cap = swap_caption(self, cap, "a saturated softmax has ~zero gradient: the model stops learning where to look", color=RED_, size=26)
        self.wait(2.5)


class SelfAttentionHeads(Scene):
    """Real GPT-2 attention for one sentence: the matrix fills row by row, then three heads with different jobs."""

    def construct(self):
        data = json.loads((DATA / "gpt2_attention.json").read_text())
        toks = data["tokens"]
        n = len(toks)
        title(self, "Self-attention in a real model", "GPT-2 small, one sentence, three of its 144 heads")
        A = np.array(data["heads"]["4_3"])
        cell = 0.42
        hm = heatmap(np.zeros((n, n)), cell=cell).shift(0.6 * RIGHT + 0.75 * DOWN)
        cols = VGroup(*[Text(t, font_size=17).rotate(PI / 3).next_to(hm[j], UP, buff=0.12) for j, t in enumerate(toks)])
        rows = VGroup(*[Text(t, font_size=17).next_to(hm[i * n], LEFT, buff=0.15) for i, t in enumerate(toks)])
        ylab = Text("query (who is looking)", font_size=20, color=BLUE_).rotate(PI / 2).next_to(rows, LEFT, buff=0.25)
        xlab = Text("key (who is looked at)", font_size=20, color=ORANGE_).next_to(cols, RIGHT, buff=0.3).align_to(cols, DOWN)
        self.play(FadeIn(hm), FadeIn(cols), FadeIn(rows), FadeIn(ylab), FadeIn(xlab))
        cap = caption("each row: one token's attention weights over the tokens before it (sums to 1)")
        self.play(FadeIn(cap))

        def fill(M, rt=0.12):
            anims = []
            for i in range(n):
                anims.append(AnimationGroup(*[hm[i * n + j].animate.set_fill(heat_color(M[i, j])) for j in range(n)]))
            self.play(LaggedStart(*anims, lag_ratio=0.6), run_time=rt * n)

        fill(A, 0.25)
        it, an = toks.index("it"), toks.index("animal")
        box = SurroundingRectangle(VGroup(*[hm[it * n + j] for j in range(n)]), color=BLUE_, buff=0.03, stroke_width=4)
        tgt = SurroundingRectangle(hm[it * n + an], color=PURPLE_, buff=0.02, stroke_width=6)
        cap = swap_caption(self, cap, f'layer 4, head 3: "it" puts {A[it, an]:.0%} of its attention on "animal"', color=PURPLE_)
        self.play(Create(box))
        self.play(Create(tgt), Indicate(rows[it], color=BLUE_), Indicate(cols[an], color=PURPLE_))
        self.wait(1.5)
        self.play(FadeOut(box), FadeOut(tgt))

        heads = [("4_11", "layer 4, head 11: always the previous token (position bookkeeping)"),
                 ("1_10", "layer 1, head 10: spread out, gathers broad context"),
                 ("4_3", "different heads learn different jobs; multi-head = several lookups in parallel")]
        for key, text in heads:
            M = np.array(data["heads"][key])
            cap = swap_caption(self, cap, text, size=26)
            self.play(*[hm[i * n + j].animate.set_fill(heat_color(M[i, j])) for i in range(n) for j in range(n)], run_time=1.5)
            self.wait(1.8)
        self.wait(1)


class CausalMask(Scene):
    """Masked self-attention: a token may only look at itself and the past; generation one token at a time."""

    def construct(self):
        title(self, "Masked (causal) self-attention", "used for language modeling: predict the next token")
        toks = ["The", "cat", "sat", "on", "the", "mat"]
        n = len(toks)
        rng = np.random.default_rng(0)
        S = rng.normal(size=(n, n)) * 1.2
        mask = np.tril(np.ones((n, n)))
        W = softmax(np.where(mask > 0, S, -1e9))
        cell = 0.62
        hm = heatmap(np.zeros((n, n)), cell=cell).shift(1.5 * LEFT + 0.4 * DOWN)
        cols = VGroup(*[Text(t, font_size=22).next_to(hm[j], UP, buff=0.12) for j, t in enumerate(toks)])
        rows = VGroup(*[Text(t, font_size=22).next_to(hm[i * n], LEFT, buff=0.15) for i, t in enumerate(toks)])
        self.play(FadeIn(hm), FadeIn(cols), FadeIn(rows))
        cap = caption("future positions are set to −∞ before the softmax → weight 0")
        self.play(FadeIn(cap))
        crosses = VGroup()
        for i in range(n):
            for j in range(i + 1, n):
                crosses.add(VGroup(hm[i * n + j].copy().set_fill(RED_, 0.15).set_stroke(RED_, 1),
                                   Text("−∞", font_size=18, color=RED_).move_to(hm[i * n + j])))
        self.play(LaggedStart(*[FadeIn(c) for c in crosses], lag_ratio=0.04), run_time=1.5)
        self.play(*[hm[i * n + j].animate.set_fill(heat_color(W[i, j])) for i in range(n) for j in range(i + 1)], run_time=1.2)
        self.wait(0.8)

        cap = swap_caption(self, cap, "generation: each new token attends to everything before it, then is appended")
        out = VGroup()
        anchor = np.array([3.4, 1.6, 0])
        for i in range(n):
            row = SurroundingRectangle(VGroup(*[hm[i * n + j] for j in range(n)]), color=BLUE_, buff=0.02, stroke_width=4)
            nxt = toks[i + 1] if i + 1 < n else "."
            word = Text("→ " + nxt, font_size=28, color=GREEN_)
            word.next_to(hm[i * n + n - 1], RIGHT, buff=0.5)
            arrow = VMobject()
            self.play(Create(row), run_time=0.4)
            self.play(FadeIn(word, shift=0.1 * RIGHT), run_time=0.5)
            self.play(FadeOut(row), run_time=0.3)
            out.add(word)
        cap = swap_caption(self, cap, "training uses all rows at once: n next-token predictions in one forward pass", color=PURPLE_, size=26)
        self.wait(2.5)
