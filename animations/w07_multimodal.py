"""Week 7 (multimodal). Render one scene:
    manim -qh w07_multimodal.py CLIPMatrix
Data: data/w07_clip.json + data/w07_img*.png (real CLIP ViT-B/32 similarities, made by data/w05_07_precompute.py);
matplotlib's grace_hopper.jpg.
"""
import json
from pathlib import Path
from PIL import Image
from style import *

DATA = Path(__file__).parent / "data"
TEAL = "#0891B2"


def img_mob(arr, height):
    if isinstance(arr, Image.Image) and arr.height < 240:      # upsample small photos once, smoothly
        arr = arr.resize((arr.width * 4, arr.height * 4), Image.LANCZOS)
    m = ImageMobject(np.asarray(arr))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
    return m.set(height=height)


def box(label, w, h, color, size=22, sub=None):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color, stroke_width=3,
                         fill_color=color, fill_opacity=0.08)
    t = Text(label, font_size=size, color=color, weight=BOLD, line_spacing=0.8).move_to(r)
    g = VGroup(r, t)
    if sub:
        s = Text(sub, font_size=15, color=MUTED, line_spacing=0.8).next_to(r, DOWN, buff=0.08)
        g.add(s)
    return g


def hopper(size=224):
    import matplotlib.cbook as cbook
    im = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB")
    w, h = im.size
    sq = min(w, h)
    return im.crop(((w - sq) // 2, 0, (w - sq) // 2 + sq, sq)).resize((size, size), Image.LANCZOS)


class CLIPMatrix(Scene):
    """Real CLIP ViT-B/32 on 5 product photos and 5 captions: the 5×5 cosine matrix, the diagonal, row and
    column softmax (symmetric InfoNCE), then zero-shot classification with 'a photo of a {class}'. Target: mm4–5."""

    def construct(self):
        d = json.loads((DATA / "w07_clip.json").read_text())
        S = np.array(d["S"])
        scale = d["logit_scale"]
        n = 5
        title(self, "CLIP: contrastive image–text pretraining", "real CLIP ViT-B/32 embeddings, 5 photos × 5 captions")
        short = ["wrist watch", "sunglasses", "handbag", "running shoe", "t-shirt"]
        cell = 0.8
        org = np.array([-2.9, 1.15, 0])                 # centre of cell (0, 0)
        cpos = lambda i, j: org + np.array([j * cell, -i * cell, 0])
        imgs = Group(*[img_mob(Image.open(DATA / f"w07_img{i}.png"), cell * 0.92).move_to(cpos(i, 0) + np.array([-1.35, 0, 0]))
                       for i in range(n)])
        ilab = VGroup(*[MathTex(f"I_{i + 1}", font_size=26, color=BLUE_).next_to(imgs[i], RIGHT, buff=0.12) for i in range(n)])
        tlab = VGroup(*[MathTex(f"T_{j + 1}", font_size=26, color=ORANGE_).move_to(cpos(0, j) + np.array([0, 0.62, 0])) for j in range(n)])
        caps = VGroup(*[Text(s, font_size=16, color=ORANGE_).rotate(PI / 5).next_to(tlab[j], UP, buff=0.08).shift(0.25 * RIGHT)
                        for j, s in enumerate(short)])
        capnote = Text('captions: "a photo of a(n) …"', font_size=17, color=MUTED).move_to([5.0, 2.35, 0])

        cap = caption("1. an image encoder and a text encoder map both into one space")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(m) for m in imgs], lag_ratio=0.1), FadeIn(ilab))
        self.play(LaggedStart(*[FadeIn(c) for c in caps], lag_ratio=0.1), FadeIn(tlab), FadeIn(capnote))
        self.wait(0.6)

        lo, hi = S.min(), S.max()
        cells = VGroup()
        for i in range(n):
            for j in range(n):
                v = (S[i, j] - lo) / (hi - lo)
                sq = Square(cell, stroke_color=MUTED, stroke_width=1, fill_color=heat_color(v, PURPLE_), fill_opacity=1).move_to(cpos(i, j))
                t = Text(f"{S[i, j]:.2f}", font_size=17, color=WHITE if v > 0.6 else INK).move_to(sq)
                cells.add(VGroup(sq, t))
        cap = swap_caption(self, cap, "2. every image–caption pair gets a cosine similarity Iᵢ·Tⱼ")
        self.play(LaggedStart(*[FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.03), run_time=2)
        self.wait(0.8)
        diag = VGroup(*[SurroundingRectangle(cells[i * n + i], color=GREEN_, buff=0.0, stroke_width=5) for i in range(n)])
        cap = swap_caption(self, cap, "3. the diagonal holds the true pairs: each is the largest in its row", color=GREEN_)
        self.play(Create(diag))
        self.wait(1.5)

        # symmetric InfoNCE
        rx = 3.95
        f1 = MathTex(r"\mathcal{L}_{I\to T} = -\tfrac1N\sum_i \log \mathrm{softmax}_j(s\,I_i\!\cdot T_j)_{i}", font_size=24).move_to([rx, 0.9, 0])
        f2 = MathTex(r"\mathcal{L}_{T\to I} = -\tfrac1N\sum_j \log \mathrm{softmax}_i(s\,I_i\!\cdot T_j)_{j}", font_size=24).next_to(f1, DOWN, buff=0.35)
        f3 = MathTex(r"\mathcal{L} = \tfrac12(\mathcal{L}_{I\to T} + \mathcal{L}_{T\to I})", font_size=26).next_to(f2, DOWN, buff=0.35)
        sl = Text(f"learned scale s = {scale:.0f}", font_size=17, color=MUTED).next_to(f3, DOWN, buff=0.25)
        P_row = softmax(scale * S, axis=1)
        P_col = softmax(scale * S, axis=0)

        def recolor(P, color):
            anims = []
            for i in range(n):
                for j in range(n):
                    sq, t = cells[i * n + j]
                    anims += [sq.animate.set_fill(heat_color(P[i, j], color)),
                              Transform(t, Text(f"{P[i, j]:.2f}", font_size=17, color=WHITE if P[i, j] > 0.6 else INK).move_to(sq))]
            return anims

        cap = swap_caption(self, cap, "4. softmax over each row: which caption belongs to this image?", color=BLUE_)
        self.play(FadeIn(f1), *recolor(P_row, BLUE_), run_time=1.5)
        self.wait(1.2)
        cap = swap_caption(self, cap, "softmax over each column: which image belongs to this caption?", color=ORANGE_)
        self.play(FadeIn(f2), *recolor(P_col, ORANGE_), run_time=1.5)
        self.wait(1.2)
        cap = swap_caption(self, cap, "symmetric InfoNCE: cross-entropy toward the diagonal, both ways")
        self.play(FadeIn(f3), FadeIn(sl))
        self.wait(1.5)

        # zero-shot
        self.play(*[FadeOut(m) for m in [*imgs, ilab, tlab, caps, cells, diag, f1, f2, f3, sl, capnote]])
        q = img_mob(Image.open(DATA / "w07_img5.png"), 2.4).move_to([-4.6, -0.2, 0])
        ql = Text("a new photo\n(not one of the 5 pairs)", font_size=17, color=MUTED, line_spacing=0.8).next_to(q, DOWN, buff=0.15)
        cos = np.array(d["zs_cos"])
        prob = softmax(scale * cos)
        prompts = d["zs_prompts"]
        y0 = 1.25
        rows = VGroup()
        for k, (pr, c, p) in enumerate(zip(prompts, cos, prob)):
            y = y0 - k * 0.6
            t = Text(f'"{pr}"', font_size=20, color=ORANGE_).move_to([-0.3, y, 0], aligned_edge=RIGHT)
            cv = Text(f"{c:.3f}", font_size=19).move_to([0.35, y, 0])
            bar = Rectangle(width=max(0.03, p * 3.2), height=0.32, stroke_width=0, fill_color=GREEN_, fill_opacity=0.9)
            bar.move_to([1.0, y, 0], aligned_edge=LEFT)
            pv = Text(f"{p:.0%}" if p >= 0.005 else "<1%", font_size=19, color=GREEN_).next_to(bar, RIGHT, buff=0.1)
            rows.add(VGroup(t, cv, bar, pv))
        hdrs = VGroup(Text("cosine", font_size=17, color=MUTED).move_to([0.35, y0 + 0.5, 0]),
                      Text("softmax(s · cosine)", font_size=17, color=MUTED).move_to([2.0, y0 + 0.5, 0]))
        cap = swap_caption(self, cap, '5. zero-shot: write one prompt per class, "a photo of a {class}"')
        self.play(FadeIn(q), FadeIn(ql), LaggedStart(*[FadeIn(r[0]) for r in rows], lag_ratio=0.1))
        cap = swap_caption(self, cap, "compare the image with every prompt: the best match is the label", color=GREEN_)
        self.play(FadeIn(hdrs), LaggedStart(*[FadeIn(r[1]) for r in rows], lag_ratio=0.1))
        self.play(LaggedStart(*[AnimationGroup(GrowFromEdge(r[2], LEFT), FadeIn(r[3])) for r in rows], lag_ratio=0.1), run_time=1.5)
        best = int(np.argmax(prob))
        self.play(Create(SurroundingRectangle(rows[best], color=GREEN_, buff=0.1, stroke_width=4)))
        self.wait(1)
        cap = swap_caption(self, cap, "a new classifier is just a new list of sentences: no retraining", color=PURPLE_)
        self.wait(3)


class VLMRecipe(Scene):
    """LLaVA-style VLM: image → patches → vision encoder → projector → visual tokens interleaved with text tokens →
    LLM → answer, with LLaVA-1.5 numbers. Target: L9b."""

    def construct(self):
        title(self, "How a vision-language model is built", "the LLaVA recipe (numbers for LLaVA-1.5-7B)")
        im = hopper(336)
        A = np.asarray(im)
        top = 0.95
        pic = img_mob(A, 1.9).move_to([-5.55, top, 0])
        cap = caption("1. the image is cut into 14×14-pixel patches: 336 px → 24×24 = 576 patches")
        self.play(FadeIn(cap), FadeIn(pic))
        G = 6
        p = 336 // G
        cell = 1.9 / G
        tiles = Group(*[img_mob(A[r * p:(r + 1) * p, c * p:(c + 1) * p], cell * 0.9)
                        .move_to(pic.get_center() + np.array([(c - 2.5) * cell, -(r - 2.5) * cell, 0]))
                        for r in range(G) for c in range(G)])
        self.add(tiles); self.remove(pic)
        self.play(*[t.animate.shift(np.array([(i % G - 2.5) * 0.06, -(i // G - 2.5) * 0.06, 0])) for i, t in enumerate(tiles)])
        note = Text("(6×6 shown)", font_size=15, color=MUTED).next_to(tiles, DOWN, buff=0.1)
        self.play(FadeIn(note))
        self.wait(0.4)

        enc = box("vision\nencoder", 1.6, 2.1, BLUE_, sub="CLIP ViT-L/14\nfrozen").move_to([-2.65, top, 0])
        a1 = Arrow(tiles.get_right(), enc[0].get_left(), buff=0.12, color=MUTED, stroke_width=4)
        feats = VGroup(*[Rectangle(width=0.5, height=0.17, stroke_width=0, fill_color=BLUE_, fill_opacity=0.35 + 0.08 * (k % 5))
                         for k in range(9)]).arrange(DOWN, buff=0.05).move_to([-0.95, top, 0])
        fl = Text("576 × 1024", font_size=17, color=BLUE_).next_to(feats, DOWN, buff=0.1)
        cap = swap_caption(self, cap, "2. a pretrained vision encoder turns them into 576 feature vectors", color=BLUE_)
        self.play(GrowArrow(a1), FadeIn(enc))
        self.play(LaggedStart(*[FadeIn(f, shift=0.2 * RIGHT) for f in feats], lag_ratio=0.08), FadeIn(fl))
        self.wait(0.6)

        proj = box("projector", 1.6, 1.1, PURPLE_, sub="2-layer MLP\ntrained").move_to([1.0, top, 0])
        a2 = Arrow(feats.get_right(), proj[0].get_left(), buff=0.12, color=MUTED, stroke_width=4)
        vt = VGroup(*[Rectangle(width=0.5, height=0.17, stroke_width=0, fill_color=PURPLE_, fill_opacity=0.35 + 0.08 * (k % 5))
                      for k in range(9)]).arrange(DOWN, buff=0.05).move_to([2.75, top, 0])
        vl = Text("576 × 4096", font_size=17, color=PURPLE_).next_to(vt, DOWN, buff=0.1)
        cap = swap_caption(self, cap, "3. a small projector maps them into the LLM's word-embedding space", color=PURPLE_)
        self.play(GrowArrow(a2), FadeIn(proj))
        self.play(LaggedStart(*[FadeIn(f, shift=0.2 * RIGHT) for f in vt], lag_ratio=0.08), FadeIn(vl))
        same = Text("same width as\na word embedding", font_size=16, color=MUTED, line_spacing=0.8).next_to(vt, RIGHT, buff=0.25)
        self.play(FadeIn(same))
        self.wait(0.8)

        # the interleaved sequence
        sy = -1.15
        words = ["USER:"] + ["img"] * 6 + ["What", "is", "she", "wearing", "?", "ASSISTANT:"]
        seq = VGroup()
        for w in words:
            if w == "img":
                seq.add(Square(0.36, stroke_width=0, fill_color=PURPLE_, fill_opacity=0.8))
            else:
                t = Text(w, font_size=17, color=INK)
                seq.add(VGroup(RoundedRectangle(corner_radius=0.06, width=t.width + 0.2, height=0.36, stroke_color=INK,
                                                stroke_width=1.5, fill_color=SOFT, fill_opacity=1), t))
        seq.arrange(RIGHT, buff=0.08).move_to([-2.3, sy, 0])
        dots = Text("… 576 …", font_size=14, color=PURPLE_).next_to(VGroup(*seq[1:7]), DOWN, buff=0.08)
        cap = swap_caption(self, cap, "4. visual tokens are spliced into the text prompt like ordinary words")
        a3 = Arrow(vl.get_bottom() + 0.05 * DOWN, VGroup(*seq[1:7]).get_top() + 0.05 * UP, buff=0.05,
                   color=PURPLE_, stroke_width=4, max_tip_length_to_length_ratio=0.06)
        self.play(Create(a3), LaggedStart(*[FadeIn(s, shift=0.1 * UP) for s in seq], lag_ratio=0.05), FadeIn(dots), run_time=1.6)
        self.wait(0.6)

        llm = box("LLM  (Vicuna-7B)", seq.width + 0.3, 0.75, GREEN_, size=22).next_to(seq, DOWN, buff=0.22)
        llm[1].move_to(llm[0])
        cap = swap_caption(self, cap, "5. the LLM attends over all of it and writes the answer token by token", color=GREEN_)
        self.play(FadeIn(llm))
        ans = ["a", "dark", "naval", "uniform", "with", "insignia"]
        out = VGroup()
        x = seq.get_right()[0] + 0.25
        for w in ans:
            t = Text(w, font_size=19, color=GREEN_, weight=BOLD)
            t.move_to([x + t.width / 2, sy, 0])
            x += t.width + 0.18
            out.add(t)
        for t in out:
            self.play(FadeIn(t, shift=0.15 * RIGHT), run_time=0.35)
        self.play(FadeIn(Text("(illustrative answer)", font_size=15, color=MUTED).next_to(out, DOWN, buff=0.15)))
        self.wait(0.8)

        cap = swap_caption(self, cap, "training: stage 1 fits only the projector, stage 2 also tunes the LLM", color=PURPLE_)
        st = VGroup(Text("stage 1: 558k image–caption pairs, projector only", font_size=18, color=PURPLE_),
                    Text("stage 2: 665k visual instructions, projector + LLM", font_size=18, color=GREEN_)
                    ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(llm, DOWN, buff=0.18).align_to(llm, LEFT)
        self.play(FadeIn(st))
        self.wait(3)


class JanusTwoEncoders(Scene):
    """Janus / Janus-Pro: one autoregressive Transformer, two visual encoders: SigLIP features in for understanding,
    VQ codes in/out for generation; why decoupling helps. Target: L9b."""

    def construct(self):
        title(self, "Janus: two visual encoders, one Transformer", "DeepSeek Janus / Janus-Pro (2024–25)")
        im = hopper(384)
        A = np.asarray(im)
        uy, gy = 1.05, -1.55
        tf = box("autoregressive\nTransformer", 2.4, 4.0, INK, size=19).move_to([0.2, -0.25, 0])
        cap = caption("1. one LLM-style Transformer predicts the next token, whatever the task")
        self.play(FadeIn(cap), FadeIn(tf))
        self.wait(0.6)

        # understanding path (top)
        pic = img_mob(A, 1.3).move_to([-6.0, uy, 0])
        sig = box("SigLIP\nencoder", 1.4, 1.1, BLUE_, size=19).move_to([-4.2, uy, 0])
        ad1 = box("adaptor", 1.05, 0.6, BLUE_, size=16).move_to([-2.45, uy, 0])
        ua = VGroup(Arrow(pic.get_right(), sig[0].get_left(), buff=0.08, color=BLUE_, stroke_width=4),
                    Arrow(sig[0].get_right(), ad1[0].get_left(), buff=0.08, color=BLUE_, stroke_width=4),
                    Arrow(ad1[0].get_right(), [tf[0].get_left()[0], uy, 0], buff=0.08, color=BLUE_, stroke_width=4))
        q = Text('"What is in the image?"', font_size=17, color=MUTED).next_to(sig, UP, buff=0.1).set_x(-4.2)
        out_u = VGroup(Text("text head", font_size=17, color=BLUE_),
                       Text('"a woman in a naval\nuniform, smiling"', font_size=17, color=BLUE_, line_spacing=0.8)
                       ).arrange(DOWN, buff=0.15).move_to([4.5, uy, 0])
        oa = Arrow([tf[0].get_right()[0], uy, 0], out_u.get_left(), buff=0.1, color=BLUE_, stroke_width=4)
        lu = Text("UNDERSTANDING", font_size=18, color=BLUE_, weight=BOLD).move_to([4.5, uy + 0.95, 0])
        cap = swap_caption(self, cap, "2. understanding: SigLIP gives semantic features of the image", color=BLUE_)
        self.play(FadeIn(pic), FadeIn(lu))
        self.play(LaggedStart(GrowArrow(ua[0]), FadeIn(sig), GrowArrow(ua[1]), FadeIn(ad1), GrowArrow(ua[2]), lag_ratio=0.3),
                  FadeIn(q), run_time=2)
        self.play(GrowArrow(oa), FadeIn(out_u))
        self.wait(1)

        # generation path (bottom)
        prompt = Text('"a portrait of a\nnaval officer"', font_size=17, color=ORANGE_, line_spacing=0.8).move_to([-5.6, gy + 0.25, 0])
        pa = Arrow(prompt.get_right(), [tf[0].get_left()[0], gy + 0.25, 0], buff=0.1, color=ORANGE_, stroke_width=4)
        lg = Text("GENERATION", font_size=18, color=ORANGE_, weight=BOLD).move_to([4.5, gy + 1.1, 0])
        gh = box("image head", 1.4, 0.55, ORANGE_, size=16).move_to([2.32, gy, 0])
        ga = Arrow([tf[0].get_right()[0], gy, 0], gh[0].get_left(), buff=0.06, color=ORANGE_, stroke_width=4)
        rng = np.random.default_rng(3)
        ids = rng.integers(0, 16384, 12)
        grid = VGroup(*[VGroup(Square(0.42, stroke_color=ORANGE_, stroke_width=1.5, fill_color=WHITE, fill_opacity=1),
                               Text(str(v), font_size=11, color=ORANGE_)) for v in ids])
        for g in grid:
            g[1].move_to(g[0])
        grid.arrange_in_grid(3, 4, buff=0.04).move_to([4.15, gy, 0])
        idl = Text("VQ codes (ids illustrative)", font_size=14, color=MUTED).next_to(grid, DOWN, buff=0.08)
        cap = swap_caption(self, cap, "3. generation: predict discrete VQ image codes one at a time", color=ORANGE_)
        self.play(FadeIn(lg), FadeIn(prompt), GrowArrow(pa))
        self.play(GrowArrow(ga), FadeIn(gh))
        self.play(FadeIn(idl))
        for g in grid:
            self.play(FadeIn(g, scale=0.6), run_time=0.18)
        # each code goes back in through the VQ embedding + gen adaptor
        back = CurvedArrow(grid.get_bottom() + 0.35 * DOWN, [tf[0].get_center()[0] + 0.3, tf[0].get_bottom()[1], 0],
                           angle=-0.5, color=ORANGE_, stroke_width=3)
        back_l = Text("fed back via the VQ codebook + gen adaptor", font_size=15, color=ORANGE_).next_to(back, DOWN, buff=0.02)
        self.play(Create(back), FadeIn(back_l))
        self.wait(0.6)

        vqd = box("VQ\ndecoder", 1.05, 0.95, ORANGE_, size=16).move_to([6.2, gy, 0])
        da = Arrow(grid.get_right(), vqd[0].get_left(), buff=0.06, color=ORANGE_, stroke_width=4)
        self.play(GrowArrow(da), FadeIn(vqd))
        out_img = img_mob(A, 1.15).next_to(vqd, UP, buff=0.15)
        oil = Text("pixels\n(illustration)", font_size=14, color=MUTED, line_spacing=0.8).next_to(vqd[0], DOWN, buff=0.12)
        cap = swap_caption(self, cap, "a VQ decoder turns the 24×24 code grid (16,384-entry codebook) into pixels", color=ORANGE_, size=26)
        self.play(FadeIn(out_img, shift=0.2 * UP), FadeIn(oil), FadeOut(lg))
        self.wait(1.2)

        # why decouple
        cap = swap_caption(self, cap, "4. one shared encoder must be both semantic and pixel-exact: the two tasks conflict", color=RED_, size=26)
        sem = Text("semantic: what is in the image", font_size=16, color=BLUE_).next_to(sig[0], DOWN, buff=0.15).set_x(-3.3)
        pix = Text("low-level: exact local detail", font_size=16, color=ORANGE_).next_to(grid, UP, buff=0.15)
        self.play(Indicate(sig, color=BLUE_), FadeIn(sem), run_time=1.2)
        self.play(Indicate(grid, color=ORANGE_), FadeIn(pix), run_time=1.2)
        self.wait(1)
        cap = swap_caption(self, cap, "Janus: each task gets the encoder it needs, the Transformer is shared", color=PURPLE_)
        res = Text("Janus-Pro-7B (reported): MMBench 79.2 · GenEval 0.80", font_size=19, color=PURPLE_).move_to([6.8, 2.75, 0], aligned_edge=RIGHT)
        self.play(FadeIn(res))
        self.wait(3)
