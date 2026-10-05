"""Week 4 (Transformers): permutation equivariance, the residual stream, RoPE, ViT patches. Render one scene:
    manim -qh --disable_caching -o TransformerBlock w04_transformers.py TransformerBlock
"""
from style import *


def div_color(v, scale=1.0):
    """Diverging colour for a signed number: negative → blue, 0 → white, positive → orange."""
    v = float(np.clip(v / scale, -1, 1))
    return interpolate_color(ManimColor(WHITE), ManimColor(ORANGE_ if v > 0 else BLUE_), abs(v))


def strip(v, cell=0.2, scale=1.0, horizontal=True):
    g = VGroup(*[Square(cell, stroke_color=MUTED, stroke_width=1, fill_color=div_color(x, scale), fill_opacity=1)
                 for x in v])
    return g.arrange(RIGHT if horizontal else DOWN, buff=0)


# ----------------------------------------------------------------------------------------------------------------------
class PermutationEquivariance(Scene):
    """Target: p7 71–77. Self-attention computed with numpy on "dog bites man" and "man bites dog": without positions
    the attention matrix of the reordered sentence is the old one with rows and columns permuted, and every token's
    output is identical (the model sees a bag of words). Blending in sinusoidal positional encodings breaks this."""

    WORDS = ["dog", "bites", "man"]

    def setup_math(self):
        rng = np.random.default_rng(11)
        d = 4
        self.d = d
        self.E = {w: rng.normal(size=d) for w in self.WORDS}
        self.Wq, self.Wk, self.Wv = (rng.normal(size=(d, d)) * s for s in (1.1, 1.1, 0.8))
        pos = np.arange(3)[:, None]
        self.P = np.concatenate([np.sin(pos), np.cos(pos), np.sin(pos / 10), np.cos(pos / 10)], axis=1) * 1.2

    def attn(self, toks, alpha):
        X = np.array([self.E[t] for t in toks]) + alpha * self.P[:len(toks)]
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        A = softmax(Q @ K.T / np.sqrt(self.d))
        return A, A @ V

    C = 0.72

    def cell_pos(self, ox, i, j):
        return np.array([ox + (j - 1) * self.C, 0.15 - (i - 1) * self.C, 0])

    def panel(self, toks, ox, alpha):
        """Returns (group, cells[i][j], rowlabs, collabs, outs)."""
        A, Z = self.attn(toks, alpha)
        c = self.C
        cells = [[None] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                sq = Square(c, stroke_color=MUTED, stroke_width=1, fill_color=heat_color(A[i, j]), fill_opacity=1)
                t = Text(f"{A[i, j]:.2f}", font_size=20, color=WHITE if A[i, j] > 0.6 else INK)
                cells[i][j] = VGroup(sq, t).move_to(self.cell_pos(ox, i, j))
        rowlabs = [token_row([w], size=22, color=BLUE_).move_to(self.cell_pos(ox, i, 0) + 0.55 * LEFT, aligned_edge=RIGHT)
                   for i, w in enumerate(toks)]
        collabs = [Text(w, font_size=20, color=ORANGE_).move_to(self.cell_pos(ox, 0, j) + (0.5 * c + 0.25) * UP)
                   for j, w in enumerate(toks)]
        outs = [strip(Z[i], cell=0.27, scale=1.5).move_to(self.cell_pos(ox, i, 2) + (0.5 * c + 0.3 + 0.54) * RIGHT)
                for i in range(3)]
        g = VGroup(*[cells[i][j] for i in range(3) for j in range(3)], *rowlabs, *collabs, *outs)
        return g, cells, rowlabs, collabs, outs

    def construct(self):
        self.setup_math()
        title(self, "Attention has no sense of order", "self-attention without positions is permutation-equivariant")
        L, R = self.WORDS, ["man", "bites", "dog"]
        perm = [R.index(w) for w in L]           # old row index -> new row index
        oxL, oxR = -3.0, 3.6
        A0, Z0 = self.attn(L, 0)
        A1, Z1 = self.attn(R, 0)
        assert np.allclose(A1[np.ix_(perm, perm)], A0) and np.allclose(Z1[perm], Z0)

        gL, cL, rL, kL, oL = self.panel(L, oxL, 0)
        hdrL = Text('"dog bites man"', font_size=26).move_to([oxL + 0.4, 1.95, 0])
        zl = Text("output z", font_size=18, color=GREEN_).next_to(oL[0], UP, buff=0.32)
        cap = caption("self-attention on one sentence: weights (rows = queries) and outputs")
        self.play(FadeIn(cap), FadeIn(hdrL))
        self.play(LaggedStart(*[FadeIn(m) for m in [*rL, *kL]], lag_ratio=0.08),
                  LaggedStart(*[FadeIn(cL[i][j]) for i in range(3) for j in range(3)], lag_ratio=0.05), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(o, shift=0.2 * RIGHT) for o in oL], lag_ratio=0.2), FadeIn(zl))
        self.wait(1.0)

        # copy to the right and reorder
        gR, cR, rR, kR, oR = self.panel(L, oxR, 0)   # same order first, then permute
        hdrR = Text('"man bites dog"', font_size=26).move_to([oxR + 0.4, 1.95, 0])
        zr = zl.copy().next_to(oR[0], UP, buff=0.32)
        cap = swap_caption(self, cap, "reorder the words: man bites dog")
        self.play(TransformFromCopy(gL, gR), FadeIn(hdrR), FadeIn(zr), run_time=1.2)
        cap = swap_caption(self, cap, "rows move with their query word…")
        self.play(*[cR[i][j].animate.move_to(self.cell_pos(oxR, perm[i], j)) for i in range(3) for j in range(3)],
                  *[rR[i].animate.move_to(self.cell_pos(oxR, perm[i], 0) + 0.55 * LEFT, aligned_edge=RIGHT) for i in range(3)],
                  *[oR[i].animate.move_to(self.cell_pos(oxR, perm[i], 2) + (0.5 * self.C + 0.84) * RIGHT) for i in range(3)],
                  run_time=1.6)
        cap = swap_caption(self, cap, "…and columns move with their key word")
        self.play(*[cR[i][j].animate.move_to(self.cell_pos(oxR, perm[i], perm[j])) for i in range(3) for j in range(3)],
                  *[kR[j].animate.move_to(self.cell_pos(oxR, 0, perm[j]) + (0.5 * self.C + 0.25) * UP) for j in range(3)],
                  run_time=1.6)
        ok = Text("= computed directly on the new order ✓", font_size=22, color=GREEN_).move_to([oxR + 0.4, -1.55, 0])
        self.play(FadeIn(ok, shift=0.1 * UP))
        cap = swap_caption(self, cap, "same numbers, just permuted: A' = P A Pᵀ and z' = P z")
        self.wait(1.6)

        # live comparison of the two "dog" outputs
        alpha = ValueTracker(0.0)

        def readout():
            a = alpha.get_value()
            zL, zR = self.attn(L, a)[1], self.attn(R, a)[1]
            diff = np.linalg.norm(zL[0] - zR[2])
            col = GREEN_ if diff < 1e-3 else RED_
            g = VGroup(Text(f"input = word embedding + {a:.2f} × position encoding", font_size=22, color=PURPLE_),
                       Text(f"‖z(dog, first word) − z(dog, last word)‖ = {diff:.3f}", font_size=24, color=col))
            return g.arrange(DOWN, buff=0.15).move_to([0.3, -2.3, 0])

        ro = always_redraw(readout)
        cap = swap_caption(self, cap, "each word gets the same output wherever it stands: a bag of words", color=RED_)
        self.play(FadeOut(ok), FadeIn(ro), Indicate(oL[0], color=GREEN_), Indicate(oR[0], color=GREEN_))
        self.wait(1.8)

        # add positions: replace static panels with live ones (identical at alpha = 0)
        liveL = always_redraw(lambda: self.panel(L, oxL, alpha.get_value())[0])
        liveR = always_redraw(lambda: self.panel(R, oxR, alpha.get_value())[0])
        self.remove(gL, gR, *[m for row in cR for m in row], *rR, *kR, *oR)
        self.add(liveL, liveR)
        cap = swap_caption(self, cap, "add a position encoding p_i to each input: the symmetry breaks")
        self.play(alpha.animate.set_value(1.0), run_time=4, rate_func=smooth)
        self.wait(1.0)
        cap = swap_caption(self, cap, "now the same word at a different position gets a different output: order matters",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------------
class TransformerBlock(Scene):
    """Target: p7 84–92. The residual stream as one vertical lane per token. LayerNorm normalises each lane, causal
    attention moves information across lanes (arc thickness = real attention weights), the result is added back;
    the MLP then acts within each lane with shared weights and is added back too. All vectors come from a numpy
    pre-LN block with random weights (d = 6); colours show the real numbers."""

    def construct(self):
        title(self, "A Transformer block", "the residual stream: one lane per token")
        toks = ["the", "cat", "sat", "down"]
        n, d = 4, 6
        rng = np.random.default_rng(5)
        X0 = rng.normal(size=(n, d)) * 0.9 + rng.normal(size=d) * 0.6

        def ln(X):
            return (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)

        Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) * s for s in (1.3, 1.3, 1.0, 1.0))
        N1 = ln(X0)
        S = (N1 @ Wq) @ (N1 @ Wk).T / np.sqrt(d)
        A = softmax(np.where(np.tril(np.ones((n, n))) > 0, S, -1e9))
        U1 = A @ (N1 @ Wv) @ Wo
        X1 = X0 + U1
        N2 = ln(X1)
        W1, W2 = rng.normal(size=(d, 4 * d)) / np.sqrt(d), rng.normal(size=(4 * d, d)) / np.sqrt(4 * d)
        U2 = np.maximum(N2 @ W1, 0) @ W2
        X2 = X1 + U2
        sc = 1.8

        lanes = [-5.0, -2.8, -0.6, 1.6]
        bx = 0.78                                        # branch offset to the right of a lane
        y_tok, y_x0, y_b1, y_ln1, y_band, y_add1 = -2.8, -2.15, -1.75, -1.35, -0.5, 0.3
        y_b2, y_ln2, y_mlp, y_add2, y_top = 0.62, 0.92, 1.42, 1.92, 2.45
        sw = 0.15                                        # strip cell size

        lane_lines = VGroup(*[Arrow([x, y_tok + 0.28, 0], [x, y_top, 0], buff=0, color=MUTED, stroke_width=6,
                                    max_tip_length_to_length_ratio=0.04) for x in lanes])
        tok_lab = VGroup(*[Text(t, font_size=24, color=BLUE_).move_to([x, y_tok, 0]) for x, t in zip(lanes, toks)])

        def xstrip(v, x, y):
            return strip(v, cell=sw, scale=sc).move_to([x - 0.12 - 3 * sw, y, 0])

        state = [X0.copy()]
        ty = ValueTracker(y_x0)

        def stream():
            y = ty.get_value()
            V = X0 if y < y_add1 - 1e-3 else (X1 if y < y_add2 - 1e-3 else X2)
            return VGroup(*[xstrip(V[i], lanes[i], y) for i in range(n)])

        st = always_redraw(stream)
        cap = caption("each token has its own lane: the residual stream carries its vector x_i upward")
        self.play(FadeIn(cap), FadeIn(tok_lab), Create(lane_lines), run_time=1.2)
        self.play(FadeIn(st))
        self.wait(0.8)

        # block structure
        def box(x, y, w, h, label, color):
            r = RoundedRectangle(corner_radius=0.06, width=w, height=h, stroke_color=color, stroke_width=2.5,
                                 fill_color=WHITE, fill_opacity=1).move_to([x, y, 0])
            return VGroup(r, Text(label, font_size=16, color=color).move_to(r))

        band = RoundedRectangle(corner_radius=0.12, width=lanes[-1] - lanes[0] + bx + 1.65, height=0.7,
                                stroke_color=PURPLE_, stroke_width=2, fill_color=PURPLE_, fill_opacity=0.07)
        band.move_to([(lanes[0] + lanes[-1] + bx) / 2 + 0.05, y_band, 0])
        band_lab = Text("causal\nself-attention", font_size=18, color=PURPLE_, line_spacing=0.8)
        band_lab.next_to(band, RIGHT, buff=0.15)
        ln1 = VGroup(*[box(x + bx, y_ln1, 0.5, 0.32, "LN", INK) for x in lanes])
        ln2 = VGroup(*[box(x + bx, y_ln2, 0.5, 0.3, "LN", INK) for x in lanes])
        mlp = VGroup(*[box(x + bx, y_mlp, 0.6, 0.4, "MLP", GREEN_) for x in lanes])
        adds = VGroup()
        wires = VGroup()
        for x in lanes:
            for ya in (y_add1, y_add2):
                c = Circle(0.14, color=INK, stroke_width=2.5).set_fill(WHITE, 1).move_to([x, ya, 0])
                adds.add(VGroup(c, MathTex("+", font_size=26).move_to(c)))
            wl = dict(color=MUTED, stroke_width=2.5)
            wires.add(VMobject(**wl).set_points_as_corners([[x, y_b1, 0], [x + bx, y_b1, 0], [x + bx, y_ln1 - 0.16, 0]]),
                      Line([x + bx, y_ln1 + 0.16, 0], [x + bx, y_band - 0.35, 0], **wl),
                      VMobject(**wl).set_points_as_corners([[x + bx, y_band + 0.35, 0], [x + bx, y_add1, 0], [x + 0.14, y_add1, 0]]),
                      VMobject(**wl).set_points_as_corners([[x, y_b2, 0], [x + bx, y_b2, 0], [x + bx, y_ln2 - 0.15, 0]]),
                      Line([x + bx, y_ln2 + 0.15, 0], [x + bx, y_mlp - 0.2, 0], **wl),
                      VMobject(**wl).set_points_as_corners([[x + bx, y_mlp + 0.2, 0], [x + bx, y_add2, 0], [x + 0.14, y_add2, 0]]))
        eq1 = MathTex(r"x_i \leftarrow x_i + \mathrm{Attn}(\mathrm{LN}(x))_i", font_size=30)
        eq2 = MathTex(r"x_i \leftarrow x_i + \mathrm{MLP}(\mathrm{LN}(x_i))", font_size=30)
        eqs = VGroup(eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([4.85, 1.55, 0])
        eq1.set_opacity(0.25); eq2.set_opacity(0.25)
        cap = swap_caption(self, cap, "a block branches off each lane, computes, and adds its result back")
        self.add_foreground_mobject(st)
        self.play(FadeIn(band), FadeIn(band_lab), Create(wires), FadeIn(ln1), FadeIn(ln2), FadeIn(mlp), FadeIn(adds),
                  FadeIn(eqs), run_time=1.8)
        self.wait(0.6)

        # LayerNorm
        cap = swap_caption(self, cap, "LayerNorm: each token's vector rescaled to mean 0, variance 1 (per lane)")
        copies = [xstrip(X0[i], lanes[i], y_x0) for i in range(n)]
        ln_out = [strip(N1[i], cell=sw, scale=sc).move_to([lanes[i] + bx, y_ln1 + 0.42, 0]) for i in range(n)]
        self.play(*[c.animate.move_to([lanes[i] + bx, y_ln1 + 0.42, 0]) for i, c in enumerate(copies)],
                  *[Indicate(b, color=INK, scale_factor=1.15) for b in ln1], run_time=1.2)
        self.play(*[Transform(copies[i], ln_out[i]) for i in range(n)], run_time=0.8)
        stats = Text(f"{toks[1]}: mean {X0[1].mean():+.2f}, std {X0[1].std():.2f}  →  mean {N1[1].mean():+.2f}, "
                     f"std {N1[1].std():.2f}", font_size=20, color=INK)
        stats = VGroup(Text(f'"{toks[1]}" before LN:', font_size=19),
                       Text(f'mean {X0[1].mean():+.2f}, std {X0[1].std():.2f}', font_size=19),
                       Text(f'after: mean {abs(N1[1].mean()):.2f}, std {N1[1].std():.2f}', font_size=19))
        stats.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([3.05, 0.55, 0], aligned_edge=LEFT)
        self.play(FadeIn(stats))
        self.wait(1.2)
        self.play(FadeOut(stats), *[FadeOut(c) for c in copies])

        # attention across lanes
        eq1.set_opacity(1)
        q = ValueTracker(0)

        def arcs():
            t = q.get_value()
            qi = int(round(t))
            g = VGroup()
            xq = lanes[qi] + bx
            for j in range(qi + 1):
                a = A[qi, j]
                xj = lanes[j] + bx
                if j == qi:
                    g.add(Circle(0.12 + 0.0 * a, color=PURPLE_, stroke_width=1.5 + 10 * a).move_to([xq, y_band, 0]))
                else:
                    L = abs(xq - xj)
                    g.add(ArcBetweenPoints([xj, y_band - 0.05, 0], [xq, y_band - 0.05, 0], angle=-4 * np.arctan(0.5 / L),
                                           color=PURPLE_, stroke_width=1.5 + 12 * a, stroke_opacity=0.4 + 0.6 * a))
                g.add(Text(f"{a:.2f}", font_size=16, color=PURPLE_).move_to([xj + (0.33 if j != qi else 0.35), y_band - 0.22, 0]))
            return g

        ac = always_redraw(arcs)
        cap = swap_caption(self, cap, "attention moves information across lanes (causal: only from earlier tokens)")
        first = arcs()
        self.play(FadeIn(first))
        self.remove(first)
        self.add(ac)
        for t in range(1, n):
            self.wait(0.8)
            self.play(q.animate.set_value(t), run_time=0.3)
        self.wait(1.4)
        snap = arcs()
        self.remove(ac)
        self.add(snap)
        self.play(FadeOut(snap))

        # residual add 1
        cap = swap_caption(self, cap, "residual add: the attention output is added into each lane")
        ups = [strip(U1[i], cell=sw, scale=sc).move_to([lanes[i] + bx, y_band + 0.6, 0]) for i in range(n)]
        self.play(*[FadeIn(u, shift=0.15 * UP) for u in ups], run_time=0.6)
        self.play(*[u.animate.move_to([lanes[i] - 0.12 - 3 * sw, y_add1, 0]).set_opacity(0) for i, u in enumerate(ups)],
                  ty.animate.set_value(y_add1), run_time=1.6)
        self.wait(0.6)

        # MLP
        self.play(eq1.animate.set_opacity(0.35), eq2.animate.set_opacity(1))
        cap = swap_caption(self, cap, "the MLP acts within each lane: same weights, each token on its own")
        copies = [xstrip(X1[i], lanes[i], y_add1) for i in range(n)]
        self.play(*[c.animate.move_to([lanes[i] + bx, y_b2 - 0.0, 0]).set_opacity(0) for i, c in enumerate(copies)],
                  LaggedStart(*[Indicate(b, color=INK, scale_factor=1.15) for b in ln2], lag_ratio=0), run_time=0.9)
        self.play(*[Indicate(b, color=GREEN_, scale_factor=1.2) for b in mlp], run_time=1.0)
        ups = [strip(U2[i], cell=sw, scale=sc).move_to([lanes[i] + bx, y_mlp + 0.37, 0]) for i in range(n)]
        self.play(*[FadeIn(u, shift=0.1 * UP) for u in ups], run_time=0.5)
        cap = swap_caption(self, cap, "residual add again: the stream is only ever written into, never replaced")
        self.play(*[u.animate.move_to([lanes[i] - 0.12 - 3 * sw, y_add2, 0]).set_opacity(0) for i, u in enumerate(ups)],
                  ty.animate.set_value(y_add2), run_time=1.6)
        self.play(ty.animate.set_value(y_top - 0.15), run_time=0.8)
        self.play(eq1.animate.set_opacity(1))
        nb = Text("× N blocks: each reads from\nand writes to the same stream", font_size=22, color=PURPLE_,
                  line_spacing=0.9).move_to([4.85, 0.4, 0])
        cap = swap_caption(self, cap, "stack N blocks; the final x_i feed the output head", color=PURPLE_)
        self.play(FadeIn(nb))
        self.wait(2.5)


# ----------------------------------------------------------------------------------------------------------------------
class RoPERotation(Scene):
    """Target: L8b. Rotary position embedding in 2-D: q at position m is rotated by mθ, k at n by nθ, so the score
    depends only on n − m. Two pairs at different absolute positions with the same offset keep identical scores while
    both slide; changing the offset changes the score. Ends with the d = 64 check (numpy, θ_i = 10000^(−2i/d))."""

    def construct(self):
        title(self, "Rotary position embedding (RoPE)", "rotate q and k by an angle proportional to position")
        th = 0.33
        q0 = np.array([1.75, 0.35])
        k0 = 1.5 * np.array([np.cos(-1.64), np.sin(-1.64)])
        g = ValueTracker(0.0)          # 0: unrotated, 1: rotated
        s = ValueTracker(0.0)          # shift of both pairs
        off = ValueTracker(3.0)        # offset n - m
        bases = [2.0, 6.0]
        oxs = [-3.4, 3.4]
        oy = -0.45

        def rot(v, a):
            c, si = np.cos(a), np.sin(a)
            return np.array([c * v[0] - si * v[1], si * v[0] + c * v[1]])

        def pos(p):
            m = bases[p] + s.get_value()
            return m, m + off.get_value()

        planes = [NumberPlane(x_range=[-2.2, 2.2], y_range=[-2.2, 2.2], x_length=3.7, y_length=3.7,
                              background_line_style={"stroke_color": MUTED, "stroke_opacity": 0.25},
                              axis_config={"stroke_color": MUTED}).move_to([ox, oy, 0]) for ox in oxs]

        def pair(p):
            pl = planes[p]
            m, n = pos(p)
            a_q, a_k = g.get_value() * m * th, g.get_value() * n * th
            qv, kv = rot(q0, a_q), rot(k0, a_k)
            grp = VGroup()
            grp.add(Arrow(pl.c2p(0, 0), pl.c2p(*qv), buff=0, color=BLUE_, stroke_width=7))
            grp.add(Arrow(pl.c2p(0, 0), pl.c2p(*kv), buff=0, color=ORANGE_, stroke_width=7))
            grp.add(Text("q", font_size=30, color=BLUE_, slant=ITALIC).move_to(pl.c2p(*(qv * 1.2))))
            grp.add(Text("k", font_size=30, color=ORANGE_, slant=ITALIC).move_to(pl.c2p(*(kv * 1.2))))
            ang_q, ang_k = np.arctan2(qv[1], qv[0]), np.arctan2(kv[1], kv[0])
            dlt = (ang_k - ang_q + PI) % TAU - PI
            grp.add(Arc(radius=0.55, start_angle=ang_q, angle=dlt, arc_center=pl.c2p(0, 0), color=PURPLE_, stroke_width=4))
            hdr = Text(f"query at m = {m:.0f},  key at n = {n:.0f}", font_size=24,
                       t2c={f"m = {m:.0f}": BLUE_, f"n = {n:.0f}": ORANGE_}).move_to([oxs[p], 1.85, 0])
            score = float(qv @ kv)
            sc = Text(f"q·k = {score:+.3f}", font_size=28, color=PURPLE_).move_to([oxs[p], -2.75, 0])
            grp.add(hdr, sc)
            return grp

        ghosts = VGroup(*[VGroup(Arrow(pl.c2p(0, 0), pl.c2p(*q0), buff=0, color=BLUE_, stroke_width=4, stroke_opacity=0.25),
                                 Arrow(pl.c2p(0, 0), pl.c2p(*k0), buff=0, color=ORANGE_, stroke_width=4,
                                       stroke_opacity=0.25)) for pl in planes])
        pA, pB = always_redraw(lambda: pair(0)), always_redraw(lambda: pair(1))
        cap = caption("two query–key pairs: (2, 5) and (6, 9), the same vectors q and k before rotation")
        self.play(FadeIn(cap), *[Create(pl) for pl in planes], run_time=1.2)
        self.add(ghosts)
        self.play(FadeIn(pA), FadeIn(pB))
        self.wait(1.2)

        cap = swap_caption(self, cap, "RoPE rotates each vector by its position × θ (here θ = 0.33 rad)")
        self.play(g.animate.set_value(1.0), run_time=3)
        self.wait(0.5)
        formula = VGroup(MathTex(r"(R_{m\theta}q)^\top (R_{n\theta}k)", font_size=30),
                         MathTex(r"= q^\top R_{(n-m)\theta}\,k", font_size=30)).arrange(DOWN, buff=0.2)
        formula.move_to([0, -0.45, 0])
        fbox = SurroundingRectangle(formula, color=PURPLE_, buff=0.15, corner_radius=0.08, stroke_width=2)
        formula = VGroup(formula, fbox)
        cap = swap_caption(self, cap, "the angle between them is (n − m)·θ: the same offset gives the same score",
                           color=PURPLE_)
        self.play(FadeIn(formula))
        self.wait(1.8)

        cap = swap_caption(self, cap, "slide both pairs: everything rotates, the score does not change")
        for k in range(1, 4):
            self.play(s.animate.set_value(k), run_time=1.0)
        self.wait(0.8)

        cap = swap_caption(self, cap, "change the offset: the angle and the score change")
        self.play(off.animate.set_value(1), run_time=1.5)
        self.wait(0.4)
        self.play(off.animate.set_value(6), run_time=2.0)
        self.wait(0.4)
        self.play(off.animate.set_value(3), s.animate.set_value(0), run_time=1.5)

        # full-dimensional check
        d = 64
        rng = np.random.default_rng(0)
        qd, kd = rng.normal(size=d), rng.normal(size=d)
        freqs = 10000.0 ** (-np.arange(0, d, 2) / d)

        def rope(v, p):
            a = p * freqs
            x, y = v[0::2], v[1::2]
            out = np.empty_like(v)
            out[0::2], out[1::2] = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
            return out

        sc = [rope(qd, m) @ rope(kd, m + 3) for m in (2, 6, 100)]
        sc_other = rope(qd, 2) @ rope(kd, 7)
        chk = VGroup(Text("d = 64, θᵢ = 10000^(−2i/d), random q and k (numpy)", font_size=26, color=MUTED),
                     Text(f"offset 3:   score(2, 5) = {sc[0]:.3f}    score(6, 9) = {sc[1]:.3f}    "
                          f"score(100, 103) = {sc[2]:.3f}", font_size=26, color=PURPLE_),
                     Text(f"offset 5:   score(2, 7) = {sc_other:.3f}", font_size=26, color=INK))
        chk.arrange(DOWN, buff=0.35).move_to([0, 0.0, 0])
        cap = swap_caption(self, cap, "real RoPE: each pair of coordinates rotates at its own frequency; the same holds")
        self.play(FadeOut(pA), FadeOut(pB), FadeOut(ghosts), FadeOut(formula), *[FadeOut(pl) for pl in planes])
        self.play(FadeIn(chk, shift=0.1 * UP))
        self.wait(2.5)
        cap = swap_caption(self, cap, "relative position enters attention for free, with no extra parameters",
                           color=PURPLE_)
        self.wait(2.2)


# ----------------------------------------------------------------------------------------------------------------------
class ViTPatches(Scene):
    """Target: p8 61. A real photo (matplotlib's grace_hopper.jpg, cropped to 128×128) cut into 16×16 patches; one
    patch is flattened to 768 numbers and linearly projected to a token; all 64 patches become a sequence, get a
    [CLS] token and position embeddings, and go through a Transformer encoder. The projection is a random matrix."""

    def construct(self):
        from PIL import Image
        import matplotlib.cbook as cbook
        with cbook.get_sample_data("grace_hopper.jpg") as f:
            img = Image.open(f).convert("RGB")
        w, h = img.size
        side = min(w, h)
        img = img.crop(((w - side) // 2, 0, (w - side) // 2 + side, side)).resize((128, 128), Image.LANCZOS)
        arr = np.asarray(img)
        P, G = 16, 8
        title(self, "Vision Transformer: an image as tokens", "cut into 16×16 patches, one token per patch")

        size = 3.8
        ps = size / G
        center = np.array([-4.2, -0.3, 0])

        def patch_img(r, c):
            im = ImageMobject(arr[r * P:(r + 1) * P, c * P:(c + 1) * P])
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            return im.set(height=ps)

        def grid_pos(r, c, gap=0.0):
            return center + np.array([(c - (G - 1) / 2) * (ps + gap), -(r - (G - 1) / 2) * (ps + gap), 0])

        patches = [[patch_img(r, c).move_to(grid_pos(r, c)) for c in range(G)] for r in range(G)]
        allp = Group(*[patches[r][c] for r in range(G) for c in range(G)])
        cap = caption("a 128 × 128 RGB image")
        self.play(FadeIn(cap), FadeIn(allp))
        self.wait(0.8)

        lines = VGroup()
        for k in range(G + 1):
            off = -size / 2 + k * ps
            lines.add(Line(center + [off, -size / 2, 0], center + [off, size / 2, 0], color=WHITE, stroke_width=2))
            lines.add(Line(center + [-size / 2, off, 0], center + [size / 2, off, 0], color=WHITE, stroke_width=2))
        cap = swap_caption(self, cap, "cut it into 16 × 16 patches: 8 × 8 = 64 patches")
        self.play(Create(lines), run_time=1.2)
        self.play(FadeOut(lines), *[patches[r][c].animate.move_to(grid_pos(r, c, 0.06)) for r in range(G) for c in range(G)],
                  run_time=0.8)

        # one patch → flatten → linear projection
        r0, c0 = 2, 4
        hl = SurroundingRectangle(patches[r0][c0], color=PURPLE_, buff=0.02, stroke_width=4)
        big = patch_img(r0, c0).set(height=1.9).move_to([-0.6, -0.3, 0])
        flat = arr[r0 * P:(r0 + 1) * P, c0 * P:(c0 + 1) * P].reshape(-1, 1, 3)     # 256 pixels × RGB
        col = ImageMobject(np.repeat(flat, 4, axis=1))
        col.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        col.stretch_to_fit_height(3.6).stretch_to_fit_width(0.28).move_to([1.35, -0.3, 0])
        D = 8
        rng = np.random.default_rng(0)
        Wp = rng.normal(size=(P * P * 3, D)) / np.sqrt(P * P * 3)
        x = (flat.reshape(-1).astype(float) / 255 - 0.5) @ Wp
        tok = strip(x, cell=0.36, scale=np.abs(x).max(), horizontal=False).move_to([4.2, -0.3, 0])
        l1 = Text("patch 16×16×3", font_size=20).next_to(big, UP, buff=0.2)
        l2 = Text("flatten: 256 px × 3 = 768 numbers", font_size=20).next_to(col, UP, buff=0.2)
        l3 = VGroup(Text("token", font_size=20), Text(f"(D = {D} here; 768 in ViT-B)", font_size=17, color=MUTED))
        l3.arrange(DOWN, buff=0.08).next_to(tok, UP, buff=0.2)
        a1 = Arrow([1.65, -0.3, 0], tok.get_left(), buff=0.12, color=PURPLE_, stroke_width=4)
        al = MathTex(r"x\,E", font_size=32, color=PURPLE_).next_to(a1, UP, buff=0.1)
        cap = swap_caption(self, cap, "each patch is flattened into one long vector of pixel values")
        self.play(Create(hl))
        self.play(FadeTransform(patches[r0][c0].copy(), big), FadeIn(l1), run_time=1.0)
        self.play(FadeTransform(big.copy(), col), FadeIn(l2), run_time=1.2)
        cap = swap_caption(self, cap, "a shared linear layer E maps it to a D-dimensional token embedding")
        self.play(GrowArrow(a1), FadeIn(al), FadeIn(tok, shift=0.2 * RIGHT), FadeIn(l3))
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in (big, col, tok, a1, al, l1, l2, l3, hl)])

        # sequence
        y_seq, step, x0 = -1.9, 0.18, -5.55
        cls = Square(0.16, stroke_color=PURPLE_, stroke_width=2, fill_color=PURPLE_, fill_opacity=0.6).move_to([x0, y_seq, 0])
        targets = [(r, c, np.array([x0 + (1 + r * G + c) * step, y_seq, 0])) for r in range(G) for c in range(G)]
        cap = swap_caption(self, cap, "read the patches row by row: a sequence of 64 tokens")
        self.play(LaggedStart(*[patches[r][c].animate.set(height=0.16).move_to(t) for r, c, t in targets], lag_ratio=0.03),
                  run_time=3.5)
        lab_p = Text("patches", font_size=17, color=MUTED).move_to([-6.45, y_seq, 0])
        self.play(FadeIn(lab_p), FadeIn(cls))
        cls_l = Text("[CLS]", font_size=17, color=PURPLE_).next_to(cls, UP, buff=0.1)
        self.play(FadeIn(cls_l))

        # position embeddings
        pos = VGroup()
        for i in range(G * G + 1):
            if i == 0:
                v = 0.0
            else:
                r, c = divmod(i - 1, G)
                v = 0.25 + 0.75 * (r / (G - 1))
            sq = Square(0.16, stroke_width=0, fill_color=heat_color(v, ORANGE_) if i else heat_color(0.35),
                        fill_opacity=1).move_to([x0 + i * step, y_seq - 0.3, 0])
            if i:
                sq.set_fill(interpolate_color(heat_color(v, ORANGE_), ManimColor(BLUE_), 0.5 * ((i - 1) % G) / (G - 1)))
            pos.add(sq)
        nums = VGroup(*[Text(str(i), font_size=13, color=MUTED).move_to([x0 + i * step, y_seq - 0.55, 0])
                        for i in (1, 9, 17, 33, 64)])
        lab_e = Text("+ position", font_size=17, color=ORANGE_).move_to([-6.45, y_seq - 0.3, 0])
        cap = swap_caption(self, cap, "add a learned position embedding: otherwise patch order is invisible")
        self.play(LaggedStart(*[FadeIn(s_, shift=0.1 * UP) for s_ in pos], lag_ratio=0.02), FadeIn(lab_e), FadeIn(nums),
                  run_time=2)
        self.wait(0.8)

        # Transformer
        tb = RoundedRectangle(corner_radius=0.15, width=12.3, height=1.3, stroke_color=INK, stroke_width=3,
                              fill_color=SOFT, fill_opacity=1).move_to([x0 + 32 * step + 0.0, -0.45, 0])
        tbl = Text("Transformer encoder: every patch attends to every other patch", font_size=24).move_to(tb)
        ups = VGroup(*[Arrow([x0 + i * step, y_seq + 0.12, 0], [x0 + i * step, tb.get_bottom()[1], 0], buff=0.02,
                             color=MUTED, stroke_width=2, max_tip_length_to_length_ratio=0.15)
                       for i in range(4, G * G + 1, 4)])
        outs = VGroup(*[Square(0.16, stroke_color=MUTED, stroke_width=1,
                               fill_color=GREEN_ if i else PURPLE_, fill_opacity=0.35 if i else 0.9)
                        .move_to([x0 + i * step, 0.55, 0]) for i in range(G * G + 1)])
        head = Text("[CLS] output → linear head → class", font_size=22, color=PURPLE_).move_to([x0 - 0.1, 1.15, 0],
                                                                                              aligned_edge=LEFT)
        harr = Arrow(outs[0].get_top(), head.get_bottom() + [0.1 - head.width / 2 + 0.08, 0, 0], buff=0.04, color=PURPLE_,
                     stroke_width=3, max_tip_length_to_length_ratio=0.3)
        cap = swap_caption(self, cap, "the 65 tokens go through a standard Transformer, exactly like words")
        self.play(FadeIn(tb), FadeIn(tbl), LaggedStart(*[GrowArrow(a) for a in ups], lag_ratio=0.03), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(o, shift=0.1 * UP) for o in outs], lag_ratio=0.01), run_time=1.2)
        cap = swap_caption(self, cap, "the [CLS] token's output is classified; patch outputs serve dense tasks",
                           color=PURPLE_)
        self.play(GrowArrow(harr), FadeIn(head))
        self.wait(2.5)
