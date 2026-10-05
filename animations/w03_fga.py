"""Week 3: Factor Graph Attention (Schwartz, Schwing, Hazan, CVPR 2019) — attention over several modalities.
Target slides: p7 37–54. Toy numbers (planted for a readable example); the ablation numbers are the paper's
(VisDial v0.9, MRR). Render: manim -qh w03_fga.py FactorGraphAttention
"""
from style import *

WORDS = ["What", "animal", "is", "in", "the", "cage", "?"]
REGIONS = ["sky", "sky", "tree", "bars", "lion", "bars", "floor", "floor", "food"]
REGION_FILL = ["#BFDBFE", "#BFDBFE", "#BBF7D0", "#D1D5DB", "#FDE68A", "#D1D5DB", "#E7E5E4", "#E7E5E4", "#FECACA"]

# planted toy quantities
UNARY_I = np.array([0.1, 0.1, 0.3, 0.6, 1.2, 0.6, 0.2, 0.2, 0.7])     # region salient on its own
UNARY_Q = np.array([0.6, 0.9, 0.1, 0.1, 0.0, 0.8, 0.0])
def grid_for(words):
    g = np.full((len(words), 9), 0.05)
    concept = {"animal": {4: 0.95, 3: 0.2, 5: 0.2}, "cage": {3: 0.9, 5: 0.9, 4: 0.45},
               "floor": {6: 0.95, 7: 0.95, 8: 0.3}}
    for i, w in enumerate(words):
        for r, v in concept.get(w, {}).items():
            g[i, r] = v
    return g
MARG_W = np.array([0.4, 1.0, 0.3, 0.6, 0.9, 1.0, 0.2])   # learned marginal over word positions (Linear(n_Q -> 1))
W_UNARY, W_PAIR = 1.0, 3.0                                 # learned mix of potentials (Linear(K -> 1), no bias)


def image_belief(words):
    msg = MARG_W[:len(words)] @ grid_for(words)
    return softmax(W_UNARY * UNARY_I + W_PAIR * msg), msg


def word_belief(words):
    g = grid_for(words)
    msg = g @ (np.ones(9) / 9 * 9 * 0.35)               # learned marginal over regions (uniform-ish here)
    return softmax(W_UNARY * UNARY_Q[:len(words)] + W_PAIR * msg)


class FactorGraphAttention(Scene):
    def construct(self):
        title(self, "Factor Graph Attention", "attention over several modalities at once (Schwartz, Schwing, Hazan, CVPR'19)")

        # modalities
        words = token_row(WORDS, size=22, buff=0.1).scale(0.9).to_edge(LEFT, buff=0.5).shift(1.6 * UP)
        img = VGroup()
        for k, (lab, col) in enumerate(zip(REGIONS, REGION_FILL)):
            sq = Square(0.95, stroke_color=INK, stroke_width=2, fill_color=col, fill_opacity=1)
            sq.move_to(np.array([(k % 3) * 0.95, -(k // 3) * 0.95, 0]))
            img.add(VGroup(sq, Text(lab, font_size=18).move_to(sq)))
        img.move_to(np.array([4.6, -0.6, 0]))
        qlab = Text("question: words", font_size=22, color=BLUE_).next_to(words, UP, buff=0.15, aligned_edge=LEFT)
        ilab = Text("image: regions", font_size=22, color=GREEN_).next_to(img, UP, buff=0.15)
        cap = caption("two modalities, each a set of entities: words and image regions")
        self.play(FadeIn(words), FadeIn(qlab), FadeIn(img), FadeIn(ilab), FadeIn(cap))

        # the factor graph
        Qn = Circle(0.3, color=BLUE_, fill_opacity=0.15).move_to([0.3, 1.45, 0])
        In = Circle(0.3, color=GREEN_, fill_opacity=0.15).move_to([2.5, 1.45, 0])
        Qt, It = MathTex("Q", color=BLUE_).move_to(Qn), MathTex("I", color=GREEN_).move_to(In)
        fu_q = Square(0.22, color=INK, fill_color=INK, fill_opacity=1).next_to(Qn, UP, buff=0.18)
        fu_i = Square(0.22, color=INK, fill_color=INK, fill_opacity=1).next_to(In, UP, buff=0.18)
        fp = Square(0.22, color=PURPLE_, fill_color=PURPLE_, fill_opacity=1).move_to([1.4, 1.45, 0])
        edges = VGroup(Line(Qn.get_top(), fu_q.get_bottom(), color=INK), Line(In.get_top(), fu_i.get_bottom(), color=INK),
                       Line(Qn.get_right(), fp.get_left(), color=PURPLE_), Line(fp.get_right(), In.get_left(), color=PURPLE_))
        fg = VGroup(Qn, In, Qt, It, fu_q, fu_i, fp, edges)
        flab = VGroup(Text("unary", font_size=16).next_to(fu_q, LEFT, buff=0.1),
                      Text("unary", font_size=16).next_to(fu_i, RIGHT, buff=0.1),
                      Text("pairwise", font_size=16, color=PURPLE_).next_to(fp, DOWN, buff=0.12))
        cap = swap_caption(self, cap, "a factor graph: each modality is a node; factors score entities alone and in pairs")
        self.play(FadeIn(fg), FadeIn(flab))
        self.wait(1)

        # unary on regions
        def heat_regions(vals, color=GREEN_):
            v = np.asarray(vals) / max(vals)
            return [img[k][0].animate.set_fill(interpolate_color(ManimColor(REGION_FILL[k]), ManimColor(color), 0.85 * v[k]))
                    for k in range(9)]
        uni_eq = MathTex(r"\psi_I(r)=v^\top\mathrm{relu}(V\hat r)", font_size=30).move_to([0.2, 0.6, 0])
        cap = swap_caption(self, cap, "unary factor: is this region salient on its own? (the same for every question)")
        self.play(FadeIn(uni_eq), Indicate(fu_i), *heat_regions(UNARY_I), run_time=1.5)
        self.wait(1.2)
        self.play(*[img[k][0].animate.set_fill(REGION_FILL[k]) for k in range(9)], run_time=0.6)

        # pairwise grid
        G = grid_for(WORDS)
        hm = heatmap(G, cell=0.27, color=PURPLE_).move_to([0.4, -0.95, 0])
        rl = VGroup(*[Text(w, font_size=14).next_to(hm[i * 9], LEFT, buff=0.08) for i, w in enumerate(WORDS)])
        cl = VGroup(*[Text(r, font_size=12).rotate(PI / 2).next_to(hm[j], UP, buff=0.06) for j, r in enumerate(REGIONS)])
        pw_eq = MathTex(r"C_{ij}=\Big\langle \tfrac{L\hat q_i}{\|L\hat q_i\|},\tfrac{R\hat r_j}{\|R\hat r_j\|}\Big\rangle", font_size=28).move_to([0.2, 0.6, 0])
        cap = swap_caption(self, cap, "pairwise factor: learned projections L, R, then cosine: every word × every region")
        self.play(FadeOut(uni_eq), FadeIn(pw_eq), Indicate(fp, color=PURPLE_))
        self.play(FadeIn(hm), FadeIn(rl), FadeIn(cl))
        self.wait(0.8)
        bn = Text("→ BatchNorm", font_size=18, color=MUTED).next_to(pw_eq, RIGHT, buff=0.2)
        self.play(FadeIn(bn))
        self.wait(0.8)

        # message: learned marginal over words
        belief, msg = image_belief(WORDS)
        wbar = VGroup(*[Rectangle(width=0.22 * MARG_W[i] + 0.02, height=0.22, stroke_width=0, fill_color=BLUE_, fill_opacity=0.8)
                        .next_to(rl[i], LEFT, buff=0.08) for i in range(len(WORDS))])
        mrow = heatmap(msg[None, :] / msg.max(), cell=0.27, color=PURPLE_).next_to(hm, DOWN, buff=0.15)
        mlab = MathTex(r"\mu_{Q\to I}(j)=\sum_i w_i\,C_{ij}", font_size=24).next_to(mrow, LEFT, buff=0.25)
        cap = swap_caption(self, cap, "message to the image: a learned weighted sum over words collapses each column")
        self.play(FadeIn(wbar))
        self.play(TransformFromCopy(hm, mrow), FadeIn(mlab), run_time=1.4)
        self.wait(1)

        # belief over regions
        be_eq = MathTex(r"b_I \propto \mathrm{softmax}\big(w_1\psi_I + w_2\mu_{Q\to I}\big)", font_size=28).move_to([0.2, 0.6, 0])
        bvals = VGroup(*[Text(f"{belief[k]:.2f}", font_size=16, color=INK).next_to(img[k][1], DOWN, buff=0.05) for k in range(9)])
        cap = swap_caption(self, cap, "belief: a learned mix of all potentials, then softmax over regions")
        self.play(FadeOut(pw_eq), FadeOut(bn), FadeIn(be_eq), *heat_regions(belief, PURPLE_), FadeIn(bvals), run_time=1.5)
        self.wait(1.2)
        att = MathTex(r"a_I=\sum_r b_I(r)\,\hat r", font_size=28).next_to(img, DOWN, buff=0.25)
        cap = swap_caption(self, cap, "attended image vector: the belief-weighted sum of region features")
        self.play(FadeIn(att))
        self.wait(1)

        # the same graph gives a belief over words
        wb = word_belief(WORDS)
        cap = swap_caption(self, cap, "the same graph gives a belief over words: 'animal' and 'cage' carry the question")
        self.play(*[words[i][0].animate.set_fill(heat_color(0.55 * wb[i] / wb.max(), BLUE_), 1) for i in range(len(WORDS))], run_time=1.2)
        self.wait(1.2)

        # change the question
        W2 = ["What", "color", "is", "the", "floor", "?", ""]
        new_words = token_row([w for w in W2 if w], size=22, buff=0.1).scale(0.9).move_to(words, aligned_edge=LEFT)
        G2 = grid_for(W2)
        b2, m2 = image_belief(W2)
        cap = swap_caption(self, cap, "new question, same image: the pairwise factor moves the attention")
        self.play(FadeOut(words), FadeIn(new_words), FadeOut(bvals), FadeOut(wbar),
                  *[hm[i * 9 + j].animate.set_fill(heat_color(G2[i, j])) for i in range(7) for j in range(9)],
                  *[rl[i].animate.become(Text(W2[i] or " ", font_size=14).move_to(rl[i], aligned_edge=RIGHT)) for i in range(7)],
                  *[mrow[j].animate.set_fill(heat_color(m2[j] / m2.max())) for j in range(9)], run_time=1.4)
        bvals2 = VGroup(*[Text(f"{b2[k]:.2f}", font_size=16, color=INK).next_to(img[k][1], DOWN, buff=0.05) for k in range(9)])
        self.play(*heat_regions(b2, PURPLE_), FadeIn(bvals2), run_time=1.4)
        self.wait(1.5)

        # scale: many modalities + high order + ablation
        self.play(*[FadeOut(m) for m in [new_words, hm, rl, cl, bn, mrow, mlab, be_eq, att, bvals2, img, ilab, qlab, fg, flab]])
        names = ["image", "question", "caption", "answers", "hist Q₁", "hist A₁", "…", "hist Q₁₀", "hist A₁₀"]
        nodes = VGroup(*[VGroup(Circle(0.42, color=BLUE_ if i < 4 else GREEN_, fill_opacity=0.12),
                                Text(n, font_size=15)) for i, n in enumerate(names)])
        for i, nd in enumerate(nodes):
            ang = PI / 2 - i * 2 * PI / len(names)
            nd.move_to(np.array([1.85 * np.cos(ang) - 2.8, 1.85 * np.sin(ang) - 0.05, 0]))
        links = VGroup(*[Line(a.get_center(), b.get_center(), color=PURPLE_, stroke_width=1.2, stroke_opacity=0.5)
                         for i, a in enumerate(nodes) for b in list(nodes)[i + 1:] if "…" not in a[1].text + b[1].text])
        cap = swap_caption(self, cap, "visual dialog: 23 modalities; every one attends over every other")
        self.play(FadeIn(links), LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.08), run_time=1.6)
        tern = MathTex(r"\text{ternary: }\psi_{ijk}=\sum_d \hat x_d\,\hat y_d\,\hat z_d", font_size=24, color=PURPLE_).next_to(nodes, DOWN, buff=0.12)
        self.play(FadeIn(tern))
        abl = [("no attention", 0.6249), ("no unary", 0.6425), ("no self", 0.6369), ("no BatchNorm", 0.6301), ("full FGA", 0.6525)]
        bars_g = VGroup()
        for k, (n, v) in enumerate(abl):
            h = (v - 0.62) / (0.655 - 0.62) * 3.0
            r = Rectangle(width=0.55, height=h, stroke_width=0, fill_color=PURPLE_ if n == "full FGA" else MUTED, fill_opacity=0.9)
            r.move_to(np.array([2.4 + k * 0.85, -2.3 + h / 2, 0]))
            bars_g.add(r, Text(f"{v:.3f}", font_size=15).next_to(r, UP, buff=0.05),
                       Text(n, font_size=14).rotate(PI / 5).next_to(r, DOWN, buff=0.12))
        bhdr = Text("VisDial v0.9, MRR (paper ablation)", font_size=18, color=MUTED).next_to(bars_g, UP, buff=0.3)
        cap = swap_caption(self, cap, "high-order factors score triples; ablations: every factor and BatchNorm matter")
        self.play(FadeIn(bars_g, shift=0.2 * UP), FadeIn(bhdr), run_time=1.2)
        self.wait(2)

        # bridge to self-attention
        self.play(*[FadeOut(m) for m in [nodes, links, tern, bars_g, bhdr]])
        b1 = MathTex(r"\text{self factor:}\quad C_{ij}=\langle L\hat x_i,\;R\hat x_j\rangle", font_size=40).shift(0.9 * UP)
        b2_ = MathTex(r"=\;(XL^\top)(XR^\top)^\top = Q K^\top", font_size=40).next_to(b1, DOWN, buff=0.35)
        b3 = Text("(up to FGA's L2 normalization)\nFGA collapses the grid into one belief per modality;\nself-attention keeps every row: one belief per token (week 4)",
                  font_size=26, color=PURPLE_, line_spacing=1.2).next_to(b2_, DOWN, buff=0.5)
        cap = swap_caption(self, cap, "the bridge: the interaction grid of learned projections is exactly Q·Kᵀ")
        self.play(Write(b1))
        self.play(Write(b2_))
        self.play(FadeIn(b3, shift=0.1 * UP))
        self.wait(3)
