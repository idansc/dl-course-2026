"""Week 6 (self-supervised learning). Render one scene:
    manim -qh w06_ssl.py ContrastiveSphere
Data: data/w06_collapse.json, data/w07_img*.png (made by data/w05_07_precompute.py); matplotlib's grace_hopper.jpg.
"""
import json
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from style import *

DATA = Path(__file__).parent / "data"
TEAL = "#0891B2"
COLS = [BLUE_, ORANGE_, GREEN_, PURPLE_, RED_, TEAL]


def img_mob(arr, height):
    if isinstance(arr, Image.Image) and arr.height < 240:      # upsample small photos once, smoothly
        arr = arr.resize((arr.width * 4, arr.height * 4), Image.LANCZOS)
    m = ImageMobject(np.asarray(arr))
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bicubic"])
    return m.set(height=height)


def framed(arr, height, color):
    im = img_mob(arr, height)
    return Group(im, SurroundingRectangle(im, color=color, buff=0.02, stroke_width=3))


class ContrastiveSphere(Scene):
    """SimCLR-style InfoNCE on the unit circle: two augmented views of each image are pulled together, all other
    views pushed apart (real gradient descent on the loss); temperature decides which negatives push. Target: p5."""

    def construct(self):
        title(self, "Contrastive learning", "embeddings live on the unit sphere (here a 2-D slice: the circle)")
        rng = np.random.default_rng(4)
        n = 6
        imgs = [Image.open(DATA / f"w07_img{i}.png").convert("RGB") for i in range(n)]

        def augment(im, k):
            w, h = im.size
            if k == 0:   # random crop + flip
                s = rng.uniform(0.6, 0.8)
                cw, ch = int(w * s), int(h * s)
                x0, y0 = rng.integers(0, w - cw), rng.integers(0, h - ch)
                return ImageOps.mirror(im.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h)))
            out = ImageEnhance.Color(im).enhance(0.15)             # colour jitter + blur
            out = ImageEnhance.Brightness(out).enhance(0.8).filter(ImageFilter.GaussianBlur(1.2))
            return out.rotate(rng.uniform(-20, 20), fillcolor=(255, 255, 255))

        views = [[augment(im, 0), augment(im, 1)] for im in imgs]

        # left panel: originals and their two views
        rows = Group()
        for i in range(n):
            r = Group(img_mob(imgs[i], 0.62), framed(views[i][0], 0.62, COLS[i]), framed(views[i][1], 0.62, COLS[i]))
            r.arrange(RIGHT, buff=0.12)
            rows.add(r)
        rows.arrange(DOWN, buff=0.07).move_to([-5.35, -0.3, 0])
        hdr = Text("image   view 1  view 2", font_size=17, color=MUTED).next_to(rows, UP, buff=0.1)

        # real gradient descent on NT-Xent over angles
        import torch
        tau = 0.2
        th = torch.tensor(rng.uniform(0, 2 * np.pi, 2 * n), requires_grad=True)
        pos = torch.tensor([i + n if i < n else i - n for i in range(2 * n)])

        def loss_fn(theta, t=tau):
            z = torch.stack([torch.cos(theta), torch.sin(theta)], 1)
            S = z @ z.T / t
            S = S.masked_fill(torch.eye(2 * n, dtype=torch.bool), -1e9)
            return torch.nn.functional.cross_entropy(S, pos)

        traj, losses = [th.detach().numpy().copy()], [loss_fn(th).item()]
        opt = torch.optim.SGD([th], lr=0.08)
        for _ in range(240):
            l = loss_fn(th)
            opt.zero_grad(); l.backward(); opt.step()
            traj.append(th.detach().numpy().copy()); losses.append(loss_fn(th).item())
        traj = np.array(traj)

        C, R = np.array([-0.4, -0.15, 0]), 2.0
        circle = Circle(radius=R, color=MUTED, stroke_width=2).move_to(C)
        k = ValueTracker(0)

        def ang(i):
            kk = k.get_value()
            a, b = int(np.floor(kk)), min(int(np.floor(kk)) + 1, len(traj) - 1)
            t = kk - a
            return (1 - t) * traj[a, i] + t * traj[b, i]   # angles change little per step

        pt = lambda a, r=R: C + r * np.array([np.cos(a), np.sin(a), 0])
        dots = VGroup(*[always_redraw(lambda i=i: Dot(pt(ang(i)), radius=0.12, color=COLS[i % n],
                                                       fill_opacity=1 if i < n else 0.55)
                                      .set_stroke(COLS[i % n], 3, 1))
                        for i in range(2 * n)])
        tags = VGroup(*[always_redraw(lambda i=i: Text(f"{i % n + 1}", font_size=18,
                                                        color=COLS[i % n]).move_to(pt(ang(i), R + 0.33)))
                        for i in range(2 * n)])
        lossro = always_redraw(lambda: Text(f"InfoNCE loss {np.interp(k.get_value(), np.arange(len(losses)), losses):.2f}",
                                            font_size=20, color=MUTED).move_to(C + np.array([0, -R - 0.68, 0])))

        cap = caption("1. two random augmentations of each image, encoded to unit vectors")
        self.play(FadeIn(cap), FadeIn(hdr), LaggedStart(*[FadeIn(r, shift=0.1 * RIGHT) for r in rows], lag_ratio=0.1))
        self.play(Create(circle), FadeIn(dots), FadeIn(tags), FadeIn(lossro))
        self.wait(0.8)

        # one anchor: its positive and its negatives
        rx = 4.6
        f = MathTex(r"\ell_i = -\log \frac{e^{\,s_{i,i^+}/\tau}}{\sum_{j\neq i} e^{\,s_{ij}/\tau}}", font_size=34).move_to([rx, 1.65, 0])
        fs = MathTex(r"s_{ij} = z_i\cdot z_j", font_size=28).next_to(f, DOWN, buff=0.2)
        a0 = 0
        pull = always_redraw(lambda: Arrow(pt(ang(a0)), pt(ang(a0 + n)), buff=0.15, color=GREEN_, stroke_width=5,
                                           max_tip_length_to_length_ratio=0.12))
        pushes = VGroup(*[always_redraw(lambda j=j: DashedLine(pt(ang(a0)), pt(ang(j)), color=RED_, stroke_width=2,
                                                               stroke_opacity=0.5))
                          for j in range(2 * n) if j not in (a0, a0 + n)])
        cap = swap_caption(self, cap, "2. pull the other view of image 1 closer, push the 10 other views away")
        self.play(FadeIn(f), FadeIn(fs), FadeIn(pull), FadeIn(pushes))
        self.wait(1.5)

        cap = swap_caption(self, cap, "3. gradient descent on the loss: views pair up, pairs spread out")
        self.play(k.animate.set_value(240), run_time=7, rate_func=lambda t: 1 - (1 - t) ** 2)
        self.play(FadeOut(pull), FadeOut(pushes))
        self.wait(0.8)
        cap = swap_caption(self, cap, "alignment (positives together) + uniformity (use the whole sphere)", color=PURPLE_)
        self.wait(1.5)

        # temperature: how the push is shared among negatives, at the initial configuration
        z0 = np.stack([np.cos(traj[0]), np.sin(traj[0])], 1)
        neg = [j for j in range(2 * n) if j not in (a0, a0 + n)]
        s = z0[neg] @ z0[a0]
        order = np.argsort(-s)
        s = s[order]

        def wbars(t, y0, color, label):
            w = softmax(s / t)
            g = VGroup()
            for i, v in enumerate(w):
                h = max(0.02, v * 1.6)
                g.add(Rectangle(width=0.2, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
                      .move_to([rx - 1.2 + i * 0.27, y0 + h / 2, 0]))
            g.add(Line([rx - 1.4, y0, 0], [rx + 1.55, y0, 0], color=MUTED, stroke_width=2),
                  Text(label, font_size=18, color=color).move_to([rx + 0.05, y0 - 0.25, 0]))
            return g
        th_ = Text("share of the push per negative,\nsorted from closest to farthest", font_size=18, color=MUTED,
                   line_spacing=0.8).move_to([rx, 0.15, 0])
        b1 = wbars(1.0, -1.05, ORANGE_, "τ = 1: spread over all")
        b2 = wbars(0.1, -2.6, RED_, "τ = 0.1: hardest negatives dominate")
        cap = swap_caption(self, cap, "4. temperature τ sets how sharply the push focuses on the closest negatives")
        self.play(FadeIn(th_), FadeIn(b1, shift=0.1 * UP))
        self.play(FadeIn(b2, shift=0.1 * UP))
        self.wait(3)


class MAEMasking(Scene):
    """Masked autoencoder: a real photo cut into patches, 75% hidden, the encoder sees only the visible quarter,
    a light decoder fills the rest (fill shown as a blurred illustration). Loss on masked patches. Target: p5."""

    def construct(self):
        import matplotlib.cbook as cbook
        title(self, "Masked autoencoder (MAE)", "hide 75% of the patches, predict the missing pixels")
        im = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB")
        w, h = im.size
        sq = min(w, h)
        im = im.crop(((w - sq) // 2, 0, (w - sq) // 2 + sq, sq)).resize((224, 224), Image.LANCZOS)
        A = np.asarray(im)
        blur = np.asarray(im.resize((14, 14), Image.BILINEAR).resize((224, 224), Image.BICUBIC).filter(ImageFilter.GaussianBlur(4)))
        G, p = 8, 28
        side = 3.3
        cell = side / G
        rng = np.random.default_rng(7)
        perm = rng.permutation(G * G)
        visible = np.sort(perm[:16])
        masked = set(perm[16:].tolist())

        def tile(arr, r, c, center, gap=0.0):
            t = img_mob(arr[r * p:(r + 1) * p, c * p:(c + 1) * p], cell - gap)
            return t.move_to(center)

        L = np.array([-4.7, -0.15, 0])
        pos = lambda r, c, o=L, g=0.0: o + np.array([(c - (G - 1) / 2) * (cell + g), -(r - (G - 1) / 2) * (cell + g), 0])
        whole = img_mob(A, side).move_to(L)
        tiles = Group(*[tile(A, r, c, pos(r, c)) for r in range(G) for c in range(G)])
        cap = caption("1. cut the image into patches (8×8 here; ViT-B/16 uses 14×14)")
        self.play(FadeIn(cap), FadeIn(whole))
        self.add(tiles); self.remove(whole)
        self.play(*[t.animate.move_to(pos(i // G, i % G, g=0.05)) for i, t in enumerate(tiles)], run_time=1)
        self.wait(0.5)

        greys = VGroup(*[Square(cell, stroke_width=0, fill_color=GRAY_C, fill_opacity=1).move_to(pos(i // G, i % G, g=0.05))
                         for i in range(G * G) if i in masked])
        cap = swap_caption(self, cap, "2. hide a random 75%: 48 of 64 patches are masked", color=RED_)
        self.play(LaggedStart(*[FadeIn(g) for g in greys], lag_ratio=0.02), run_time=1.5)
        self.wait(0.8)

        # encoder on the visible tokens only
        enc = RoundedRectangle(corner_radius=0.15, width=1.5, height=3.3, stroke_color=BLUE_, stroke_width=3,
                               fill_color=BLUE_, fill_opacity=0.08).move_to([-0.6, -0.15, 0])
        enc_l = Text("ViT\nencoder", font_size=22, color=BLUE_, line_spacing=0.8).move_to(enc)
        col_x = -1.85
        vis_tiles = [tiles[i] for i in visible]
        cap = swap_caption(self, cap, "3. the large encoder runs only on the 16 visible patches (4× fewer tokens)", color=BLUE_)
        self.play(FadeIn(enc), FadeIn(enc_l))
        tok_pos = [np.array([col_x, 1.5 - j * 0.21, 0]) for j in range(16)]
        copies = Group(*[t.copy() for t in vis_tiles])
        self.play(*[c.animate.scale(0.21 / c.height).move_to(tp) for c, tp in zip(copies, tok_pos)], run_time=1.5)
        out_tok = VGroup(*[Square(0.18, stroke_width=0, fill_color=BLUE_, fill_opacity=0.85).move_to([0.65, tp[1], 0])
                           for tp in tok_pos])
        self.play(*[c.animate.move_to(enc.get_center()).scale(0.3).set_opacity(0) for c in copies],
                  LaggedStart(*[FadeIn(o, shift=0.2 * RIGHT) for o in out_tok], lag_ratio=0.05), run_time=1.3)
        self.remove(copies)
        self.wait(0.4)

        # decoder: encoded tokens + mask tokens in their grid positions
        R = np.array([4.5, -0.15, 0])
        dec = RoundedRectangle(corner_radius=0.15, width=1.05, height=3.3, stroke_color=GREEN_, stroke_width=3,
                               fill_color=GREEN_, fill_opacity=0.08).move_to([1.6, -0.15, 0])
        dec_l = Text("small\ndecoder", font_size=18, color=GREEN_, line_spacing=0.8).move_to(dec)
        slots = VGroup()
        for i in range(G * G):
            if i in masked:
                slots.add(Square(cell, stroke_color=WHITE, stroke_width=1, fill_color=GRAY_B, fill_opacity=1).move_to(pos(i // G, i % G, R, 0.05)))
        cap = swap_caption(self, cap, "4. a small decoder gets them back in place, plus a shared [MASK] token per hole", color=GREEN_, size=26)
        self.play(FadeIn(dec), FadeIn(dec_l), FadeIn(slots))
        vis_out = Group(*[tile(A, i // G, i % G, pos(i // G, i % G, R, 0.05)) for i in visible])
        self.play(*[out_tok[j].animate.move_to(vis_out[j]).scale(cell / 0.18) for j in range(16)], run_time=1.0)
        self.play(*[FadeIn(v) for v in vis_out], FadeOut(out_tok), run_time=0.6)
        self.wait(0.6)

        fills = Group(*[tile(blur, i // G, i % G, pos(i // G, i % G, R, 0.05)) for i in range(G * G) if i in masked])
        cap = swap_caption(self, cap, "5. it predicts the pixels of every masked patch")
        self.play(LaggedStart(*[FadeIn(f) for f in fills], lag_ratio=0.02), run_time=2)
        note = Text("illustration: a blurred fill,\nnot a trained MAE output", font_size=16, color=MUTED, line_spacing=0.8).next_to(slots, DOWN, buff=0.12)
        self.play(FadeIn(note))
        self.wait(0.8)

        frames = VGroup(*[Square(cell, stroke_color=RED_, stroke_width=2.5).move_to(pos(i // G, i % G, R, 0.05))
                          for i in range(G * G) if i in masked])
        loss = MathTex(r"\mathcal{L} = \frac{1}{|M|}\sum_{i\in M} \lVert \hat x_i - x_i \rVert^2", font_size=30, color=RED_)
        loss.next_to(slots, UP, buff=0.15)
        cap = swap_caption(self, cap, "6. loss = pixel MSE on the masked patches only", color=RED_)
        self.play(Create(frames), FadeIn(loss), run_time=1.2)
        self.wait(1)
        cap = swap_caption(self, cap, "the task is too hard to solve locally: the encoder must learn what objects look like",
                           color=PURPLE_, size=26)
        self.wait(3)


class CollapseVsEMA(Scene):
    """Tiny real run (2-D unit outputs, 512 points): two identical nets trained to agree collapse to one point;
    an online net with a predictor and an EMA teacher (BYOL) keeps the outputs spread. Target: p5."""

    def construct(self):
        d = json.loads((DATA / "w06_collapse.json").read_text())
        steps = d["snap_steps"]
        ia = np.array(d["input_angle"])
        title(self, "Why self-distillation collapses", "a real tiny run: MLP 2→64→64→2, outputs normalised to the unit circle")
        col = [interpolate_color(ManimColor(BLUE_), ManimColor(ORANGE_), (np.cos(a) + 1) / 2) for a in ia]
        R = 1.25
        Cs = [np.array([-5.0, 0.05, 0]), np.array([-0.75, 0.05, 0])]
        k = ValueTracker(0)
        modes = ["siam", "byol"]
        live = [ValueTracker(0), ValueTracker(0)]     # which panel follows k

        def angles(m):
            A = np.array(d[m]["angles"])
            kk = k.get_value() * live[modes.index(m)].get_value()
            a = int(np.floor(kk)); b = min(a + 1, len(A) - 1); t = kk - a
            u = (1 - t) * np.exp(1j * A[a]) + t * np.exp(1j * A[b])
            return np.angle(u)

        def cloud(m, C):
            g = VGroup()
            for a, c in zip(angles(m), col):
                g.add(Dot(C + R * np.array([np.cos(a), np.sin(a), 0]), radius=0.055, color=c, fill_opacity=0.8))
            return g

        circles = VGroup(*[Circle(radius=R, color=MUTED, stroke_width=2).move_to(C) for C in Cs])
        h1 = VGroup(Text("two identical nets", font_size=21, weight=BOLD, color=RED_),
                    MathTex(r"\max\ \cos\big(f(v_1),\,f(v_2)\big)", font_size=26)).arrange(DOWN, buff=0.12)
        h2 = VGroup(Text("student + EMA teacher (BYOL)", font_size=21, weight=BOLD, color=GREEN_),
                    MathTex(r"\max\ \cos\big(q(f(v_1)),\,\mathrm{sg}[f_{\mathrm{EMA}}(v_2)]\big)", font_size=26)).arrange(DOWN, buff=0.12)
        h1.next_to(circles[0], UP, buff=0.3); h2.next_to(circles[1], UP, buff=0.3)
        h2.set_x(Cs[1][0])
        clouds = [always_redraw(lambda m=m, C=C: cloud(m, C)) for m, C in zip(modes, Cs)]
        inlab = VGroup(Text("colour = input angle", font_size=17, color=MUTED)).move_to([Cs[0][0], -1.65, 0])

        # spread chart
        ox, oy, W, H = 2.9, -2.2, 3.6, 3.3
        n_sp = len(d["siam"]["spread"])                # one value per 10 steps, 0..1500
        SP = lambda st, v: np.array([ox + st / 1500 * W, oy + v * H, 0])
        stepro = always_redraw(lambda: Text(f"step {int(np.interp(k.get_value(), np.arange(len(steps)), steps))}",
                                            font_size=20, color=MUTED).move_to(SP(1500, 1.0), aligned_edge=RIGHT + UP))
        chart = VGroup(Line(SP(0, 0), SP(1500, 0), color=MUTED), Line(SP(0, 0), SP(0, 1), color=MUTED))
        for v in (0, 0.5, 1):
            chart.add(Text(f"{v:g}", font_size=16, color=MUTED).next_to(SP(0, v), LEFT, buff=0.1))
        for st in (0, 500, 1000, 1500):
            chart.add(Text(str(st), font_size=16, color=MUTED).next_to(SP(st, 0), DOWN, buff=0.1))
        chart.add(Text("training step", font_size=18).next_to(SP(750, 0), DOWN, buff=0.4),
                  Text("output spread  (1 − |mean of unit outputs|)", font_size=17).next_to(SP(750, 1), UP, buff=0.15))

        def curve(m, color):
            sp = np.array(d[m]["spread"])
            kk = k.get_value() * live[modes.index(m)].get_value()
            upto = np.interp(kk, np.arange(len(steps)), steps)
            nn = max(2, int(upto / 10) + 1)
            return VMobject(stroke_color=color, stroke_width=4).set_points_as_corners(
                [SP(i * 10, sp[i]) for i in range(min(nn, n_sp))])

        cv = [always_redraw(lambda: curve("siam", RED_)), always_redraw(lambda: curve("byol", GREEN_))]

        cap = caption("1. two views of each input should get the same embedding")
        self.play(FadeIn(cap), Create(circles), FadeIn(h1), FadeIn(h2), FadeIn(chart))
        self.add(*clouds, stepro, inlab)
        cap = swap_caption(self, cap, "both start from the same random net: outputs in a narrow cone")
        self.wait(1.5)

        cap = swap_caption(self, cap, "2. identical nets: the cheapest way to agree is a constant output", color=RED_)
        self.add(cv[0])
        live[0].set_value(1)
        self.play(k.animate.set_value(len(steps) - 1), run_time=5, rate_func=linear)
        lab1 = Text("collapsed", font_size=18, color=RED_).next_to(SP(1500, 0), UP, buff=0.12).align_to(SP(1500, 0), RIGHT)
        self.play(FadeIn(lab1))
        self.wait(1)

        cap = swap_caption(self, cap, "3. predictor + stop-gradient + slowly moving EMA teacher", color=GREEN_)
        ema = MathTex(r"\theta_{\mathrm{EMA}} \leftarrow 0.99\,\theta_{\mathrm{EMA}} + 0.01\,\theta", font_size=26,
                      color=GREEN_).move_to([Cs[1][0], -1.65, 0])
        self.play(FadeIn(ema))
        final, cv0 = cloud("siam", Cs[0]), curve("siam", RED_)      # freeze the collapsed panel
        self.remove(clouds[0], cv[0]); self.add(final, cv0)
        k.set_value(0)
        live[1].set_value(1)
        self.add(cv[1])
        self.play(k.animate.set_value(len(steps) - 1), run_time=6, rate_func=linear)
        self.wait(0.8)
        cap = swap_caption(self, cap, "outputs stay spread, and similar inputs (colours) stay close")
        self.wait(2)
        cap = swap_caption(self, cap, "the teacher is a slow average of the student: no shortcut to a constant",
                           color=PURPLE_)
        self.wait(2.5)
