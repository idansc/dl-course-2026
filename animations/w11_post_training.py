"""Week 11 (post-training: SFT, LoRA, RL from verifiable rewards, preferences). Real model outputs come from
`python w11_data.py` (data/w11_*.json). Render one scene:
    manim -qh --disable_caching -o LoRALowRank w11_post_training.py LoRALowRank
"""
import json
import textwrap
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


def derive(scene, lines, cap, captions, x=0.0, top=2.0, step=0.78, size=36, dim=True, wait=1.4, **capkw):
    """Write a derivation one line at a time, one caption per line; earlier lines are dimmed."""
    mobs = VGroup()
    for i, (tex, text) in enumerate(zip(lines, captions)):
        m = MathTex(tex, font_size=size)
        m.move_to([x, top - i * step, 0])
        cap = swap_caption(scene, cap, text, **capkw) if cap is not None else caption(text, **capkw)
        anims = [Write(m)]
        if cap not in scene.mobjects:
            anims.append(FadeIn(cap))
        if dim and i > 0:
            anims.append(mobs[-1].animate.set_opacity(0.45))
        scene.play(*anims, run_time=1.1)
        mobs.add(m)
        scene.wait(wait)
    return mobs, cap


def wrap(text, width=58, size=20, color=INK):
    lines = []
    for para in text.strip().split("\n"):
        if para.strip():
            lines += textwrap.wrap(para.strip(), width)
    return Text("\n".join(lines), font_size=size, color=color, line_spacing=0.9)


# ------------------------------------------------------------------------------------------------- pipeline
class PostTrainingPipeline(Scene):
    """One prompt through pretraining → SFT → preference/RLVR, with the real outputs of SmolLM2-135M (base) and
    SmolLM2-135M-Instruct, the objective of each stage, and what each stage changes. Target slide: p11 pipeline."""

    def construct(self):
        d = json.loads((DATA / "w11_pipeline.json").read_text())
        g = json.loads((DATA / "w11_grpo.json").read_text())["mixed"]
        title(self, "Post-training, one prompt at a time", "real outputs: SmolLM2-135M base and its instruct version")
        names = ["1  Pretrain", "2  SFT", "3  Preferences / RLVR"]
        datas = ["2T tokens of web text", "~1M (prompt, answer) demos", "rankings or a checker's reward"]
        changes = ["learns: language, facts, continuation", "learns: the assistant format",
                   "learns: which answers are better"]
        xs = [-4.6, 0.0, 4.6]
        boxes, chg = VGroup(), VGroup()
        for n, dd, c, x in zip(names, datas, changes, xs):
            r = RoundedRectangle(corner_radius=0.12, width=3.9, height=0.95, stroke_color=MUTED, stroke_width=2,
                                 fill_color=SOFT, fill_opacity=1).move_to([x, 1.75, 0])
            t = VGroup(Text(n, font_size=24, weight=BOLD), Text(dd, font_size=18, color=MUTED)).arrange(DOWN, buff=0.08).move_to(r)
            boxes.add(VGroup(r, t))
            chg.add(Text(c, font_size=19, color=PURPLE_).move_to([x, 1.0, 0]))
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.05, color=MUTED, stroke_width=3)
                          for i in range(2)])
        prompt = token_row(["What is 17 × 24?"], color=BLUE_, size=24)[0].move_to([-5.0, -0.25, 0])
        plab = Text("prompt", font_size=18, color=BLUE_).next_to(prompt, UP, buff=0.1)
        panel = RoundedRectangle(corner_radius=0.12, width=9.4, height=1.6, stroke_color=MUTED, stroke_width=2).move_to([1.85, -0.25, 0])
        cap = caption("one prompt, followed through the three stages of training")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.2), *[GrowArrow(a) for a in arrows])
        self.play(FadeIn(prompt), FadeIn(plab), Create(panel))

        def show(i, text, verdict, vcol, src):
            body = wrap(text, 70, 19)
            if body.width > panel.width - 0.4:
                body.scale_to_fit_width(panel.width - 0.4)
            body.move_to(panel).shift(0.12 * UP)
            v = Text(verdict, font_size=19, color=vcol, weight=BOLD).next_to(panel.get_corner(DR), UL, buff=0.1)
            s = Text(src, font_size=16, color=MUTED).next_to(panel.get_corner(DL), UR, buff=0.1)
            return VGroup(body, v, s)

        def focus(i):
            return [boxes[j][0].animate.set_stroke(INK if j == i else MUTED, 4 if j == i else 2) for j in range(3)]

        # stage 1
        L1 = MathTex(r"\mathcal{L}_{\text{pre}} = -\sum_t \log p_\theta(x_t \mid x_{<t})", font_size=36).move_to([0, -1.85, 0])
        cap = swap_caption(self, cap, "pretraining: predict the next token of web text")
        self.play(*focus(0), Write(L1))
        out = show(0, d["base"].replace("\n\n", "  "), "✗ 360, then more worksheet", RED_, "SmolLM2-135M (base), greedy")
        cap = swap_caption(self, cap, "the base model continues the page as if it were a document")
        self.play(FadeIn(out, shift=0.1 * RIGHT), FadeIn(chg[0]))
        self.wait(2.6)

        # stage 2
        L2 = MathTex(r"\mathcal{L}_{\text{SFT}} = -\sum_{t\,\in\,\text{answer}} \log p_\theta(y_t \mid x,\, y_{<t})", font_size=36).move_to(L1)
        cap = swap_caption(self, cap, "SFT: the same loss, but on curated answers only")
        self.play(*focus(1), TransformMatchingTex(L1, L2), FadeOut(out))
        out = show(1, d["instruct"].replace("\n\n", "  ").replace("\n", "  "), "✗ 424, assistant-style",
                   ORANGE_, "SmolLM2-135M-Instruct (SFT + DPO), greedy")
        cap = swap_caption(self, cap, "now it answers in the assistant format; correctness is still not rewarded")
        self.play(FadeIn(out, shift=0.1 * RIGHT), FadeIn(chg[1]))
        self.wait(2.6)

        # stage 3
        L3 = MathTex(r"\max_\theta\ \mathbb{E}_{y\sim\pi_\theta}\big[r(x,y)\big] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})",
                     font_size=36).move_to(L1)
        L3b = MathTex(r"r(x,y) = \mathbf{1}\big[\text{final number} = 408\big]", font_size=30, color=GREEN_).next_to(L3, DOWN, buff=0.18)
        cap = swap_caption(self, cap, "RL: sample answers, score them, raise the probability of the high-scoring ones")
        self.play(*focus(2), TransformMatchingTex(L2, L3), FadeOut(out))
        self.play(Write(L3b))
        nr = int(sum(g["rewards"]))
        rows = VGroup()
        for f, r, tr in zip(g["finals"], g["rewards"], g["truncated"]):
            rows.add(VGroup(Text("cut off" if tr else f, font_size=22, color=GREEN_ if r else RED_),
                            Text(f"r = {int(r)}", font_size=18, color=GREEN_ if r else RED_)).arrange(DOWN, buff=0.06))
        rows.arrange(RIGHT, buff=0.35).move_to(panel).shift(0.1 * UP)
        src = Text(f"8 sampled answers, Qwen2.5-0.5B-Instruct: {nr}/8 earn reward 1", font_size=16, color=MUTED
                   ).next_to(panel.get_corner(DL), UR, buff=0.1)
        out = VGroup(rows, src)
        self.play(LaggedStart(*[FadeIn(m, shift=0.1 * DOWN) for m in rows], lag_ratio=0.1), FadeIn(src), FadeIn(chg[2]))
        cap = swap_caption(self, cap, "a checker (or a preference model) decides what gets reinforced", color=GREEN_)
        self.wait(2.0)
        cap = swap_caption(self, cap, "pretraining gives knowledge; SFT gives format; RL and preferences give judgement",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# ------------------------------------------------------------------------------------------------- LoRA
class LoRALowRank(Scene):
    """LoRA: a frozen d×d W plus a trainable low-rank product B·A (rank r). The derivation, live parameter counts
    for d = 4096, and the merge W' = W + BA at the end. Target slide: L11b 'LoRA'."""

    def construct(self):
        title(self, "LoRA: low-rank adaptation", "fine-tune a d×d layer by training only two thin matrices")
        rng = np.random.default_rng(0)
        d = 4096
        S = 3.0
        cW = np.array([-4.7, -0.2, 0])
        Wimg = rng.standard_normal((48, 48))

        def mat_img(M, base=INK, vmax=2.5):
            v = np.clip((M / vmax + 1) / 2, 0, 1)
            lo, hi = np.array([0.93, 0.95, 0.98]), np.array([0.45, 0.5, 0.6])
            rgb = lo[None, None] + (hi - lo)[None, None] * v[..., None]
            return ImageMobject((rgb * 255).astype(np.uint8)).set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])

        W = mat_img(Wimg).set_height(S).move_to(cW)
        Wf = SurroundingRectangle(W, buff=0, color=INK, stroke_width=2)
        Wl = MathTex(r"W\ (d\times d)", font_size=30).next_to(W, UP, buff=0.15)
        Wfz = Text("frozen", font_size=20, color=MUTED).next_to(W, DOWN, buff=0.12)

        lines = [r"h = W x",
                 r"h = (W + \Delta W)\, x",
                 r"\Delta W = B A,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d}",
                 r"h = W x + B\,(A x)",
                 r"\#\text{params}:\ d^2 \;\to\; 2dr"]
        caps = ["a pretrained layer: h = Wx, with d² weights",
                "fine-tuning learns an update ΔW (as big as W itself)",
                "LoRA's bet: the update is low-rank, so write it as B·A with rank r ≪ d",
                "compute it as B(Ax): x is squeezed to r numbers and expanded back",
                "trainable parameters drop from d² to 2dr"]
        X = 3.55
        cap = None
        mobs = VGroup()
        for i, (tex, text) in enumerate(zip(lines, caps)):
            m = MathTex(tex, font_size=32).move_to([X, 1.95 - 0.72 * i, 0])
            if cap is None:
                cap = caption(text)
                self.play(FadeIn(cap), Write(m), FadeIn(W), Create(Wf), FadeIn(Wl))
            else:
                cap = swap_caption(self, cap, text)
                self.play(Write(m), mobs[-1].animate.set_opacity(0.5), run_time=1.0)
            mobs.add(m)
            if i == 1:
                self.play(FadeIn(Wfz))
            self.wait(1.0)
            if i == 2:
                # draw B and A
                r_disp = lambda r: max(0.06, S * np.sqrt(r / d) * 1.5)
                rv = ValueTracker(8)
                x0 = cW[0] + S / 2 + 0.85
                top = cW[1] + S / 2
                Bm = always_redraw(lambda: Rectangle(width=r_disp(rv.get_value()), height=S, stroke_color=PURPLE_, stroke_width=2,
                                                     fill_color=PURPLE_, fill_opacity=0.35).move_to([x0 + r_disp(rv.get_value()) / 2, cW[1], 0]))
                Am = always_redraw(lambda: Rectangle(width=S, height=r_disp(rv.get_value()), stroke_color=PURPLE_, stroke_width=2,
                                                     fill_color=PURPLE_, fill_opacity=0.35).move_to(
                    [x0 + r_disp(rv.get_value()) + 0.18 + S / 2, top - r_disp(rv.get_value()) / 2, 0]))
                plus = MathTex("+", font_size=48).move_to([cW[0] + S / 2 + 0.42, cW[1], 0])
                Bl = MathTex("B", font_size=32, color=PURPLE_).next_to([x0, cW[1] - S / 2, 0], DOWN, buff=0.12).shift(0.1 * RIGHT)
                Al = MathTex("A", font_size=32, color=PURPLE_).move_to([x0 + 0.35 + S / 2, top + 0.3, 0])
                note = Text("widths drawn ∝ √r", font_size=16, color=MUTED).move_to([x0 + 0.35 + S / 2, cW[1] - S / 2 - 0.25, 0])
                self.play(FadeIn(plus), FadeIn(Bm), FadeIn(Am), FadeIn(Bl), FadeIn(Al), FadeIn(note))
                self.wait(0.6)

        def count_text(r):
            tr, fr = 2 * d * r, d * d
            return VGroup(MathTex(rf"d={d},\ r={r}", font_size=32, color=PURPLE_),
                          Text(f"trainable  2dr = {tr:,}", font_size=22, color=PURPLE_),
                          Text(f"frozen  d² = {fr:,}", font_size=22, color=MUTED),
                          Text(f"ratio {100 * tr / fr:.2f}%", font_size=24, weight=BOLD)
                          ).arrange(DOWN, buff=0.1).move_to([X, -2.0, 0])

        cnt = count_text(8)
        self.play(FadeIn(cnt))
        self.wait(1.2)
        cap = swap_caption(self, cap, "raise the rank: the cost grows linearly in r, still tiny next to d²")
        for r in [1, 16, 64, 256, 8]:
            new = count_text(r)
            self.play(rv.animate.set_value(r), Transform(cnt, new), run_time=0.9)
            self.wait(0.5)
        cap = swap_caption(self, cap, "Llama-2-7B, r = 8 on the query and value maps of 32 layers: 4.2M of 6.7B (0.06%)",
                           color=PURPLE_, size=26)
        self.wait(2.5)

        # init and merge
        cap = swap_caption(self, cap, "init B = 0: training starts exactly at the pretrained model")
        Bz = Rectangle(width=Bm.width, height=S, stroke_color=PURPLE_, stroke_width=2, fill_color=WHITE, fill_opacity=1).move_to(Bm)
        z = MathTex("0", font_size=28, color=PURPLE_).move_to(Bz)
        self.play(FadeIn(Bz), FadeIn(z))
        self.wait(1.2)
        self.play(FadeOut(Bz), FadeOut(z))
        merge = MathTex(r"W' = W + BA", font_size=40, color=INK).move_to([cW[0] + 2.0, -2.45, 0])
        cap = swap_caption(self, cap, "after training, merge: one d×d matrix again, zero extra cost at inference")
        BA = VGroup(Bm.copy(), Am.copy())
        self.remove(Bm, Am)
        self.add(BA)
        dW = rng.standard_normal((48, 2)) @ rng.standard_normal((2, 48)) * 0.6
        W2 = mat_img(Wimg + dW).set_height(S).move_to(cW)
        self.play(BA.animate.scale(0.25).move_to(cW).set_opacity(0), FadeOut(plus), FadeOut(Bl), FadeOut(Al), FadeOut(note),
                  FadeOut(W), FadeIn(W2), Write(merge), run_time=1.6)
        Wl2 = MathTex(r"W'\ (d\times d)", font_size=30).move_to(Wl)
        self.play(Transform(Wl, Wl2), FadeOut(Wfz))
        cap = swap_caption(self, cap, "or keep adapters separate: one frozen model, many cheap swappable fine-tunes",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# ------------------------------------------------------------------------------------------------- DPO
class DPOMargin(Scene):
    """DPO derived on screen from the KL-regularised RL objective, then a real gradient-descent run on a toy
    policy (softmax over 6 responses): chosen vs rejected log-ratios move apart, the push fades as the margin grows,
    and β sets how far from π_ref the policy needs to move. Target slide: p11 'DPO'."""

    def construct(self):
        title(self, "DPO", "preferences as a classification loss")
        lines = [r"\max_\pi\ \mathbb{E}_{y\sim\pi}\,[r(x,y)] \;-\; \beta\,\mathrm{KL}\big(\pi\,\|\,\pi_{\text{ref}}\big)",
                 r"\pi^*(y\mid x) = \tfrac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}",
                 r"r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)",
                 r"p(y_w \succ y_l) = \sigma\big(r(x,y_w) - r(x,y_l)\big)\quad(Z\text{ cancels})",
                 r"\mathcal{L}_{\text{DPO}} = -\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\Big)"]
        caps = ["start: maximise reward, but pay β·KL for drifting from the reference model",
                "this objective has a closed-form optimum: reweight π_ref by exp(r/β)",
                "solve for r: every policy defines an implicit reward, a log-ratio",
                "preferences only compare two answers, so the unknown Z(x) cancels",
                "DPO: fit the policy to (chosen, rejected) pairs directly; no reward model, no sampling"]
        cap, mobs = None, VGroup()
        for i, (tex, text) in enumerate(zip(lines, caps)):
            m = MathTex(tex, font_size=32 if i < 4 else 34).move_to([0, 1.85 - 0.95 * i, 0])
            if cap is None:
                cap = caption(text, size=26)
                self.play(FadeIn(cap), Write(m), run_time=1.2)
            else:
                cap = swap_caption(self, cap, text, size=26)
                self.play(Write(m), mobs[-1].animate.set_opacity(0.45), run_time=1.2)
            mobs.add(m)
            self.wait(1.4)
        box = SurroundingRectangle(mobs[-1], color=PURPLE_, buff=0.12)
        self.play(Create(box))
        self.wait(0.8)
        loss = VGroup(mobs[-1], box)
        self.play(FadeOut(mobs[:-1]), loss.animate.scale(0.72).to_corner(UR, buff=0.35).shift(0.05 * DOWN))

        # ---- toy run: softmax policy over 6 responses, one preference pair (w = 0, l = 1)
        beta, lr, T = 0.5, 0.5, 60
        z0 = np.array([1.0, 1.2, 0.6, 0.3, 0.0, -0.4])
        pref = softmax(z0)
        z, hist = z0.copy(), []
        for t in range(T + 1):
            p = softmax(z)
            lr_w, lr_l = np.log(p[0] / pref[0]), np.log(p[1] / pref[1])
            m = lr_w - lr_l
            wgt = 1 / (1 + np.exp(beta * m))                    # σ(−β m): the push strength
            hist.append((p.copy(), lr_w, lr_l, m, wgt, np.sum(p * np.log(p / pref))))
            g = -beta * wgt * (np.eye(6)[0] - np.eye(6)[1])       # dL/dz (the softmax normaliser cancels in the difference)
            z = z - lr * g                                        # plain SGD on the logits
        names = ["chosen y_w", "rejected y_l", "y_3", "y_4", "y_5", "y_6"]
        cols = [GREEN_, RED_] + [MUTED] * 4
        # left: π vs π_ref bars
        base_y, bw, sp, x0 = -1.9, 0.42, 0.68, -6.0
        refb = VGroup(*[Rectangle(width=bw, height=pref[i] * 3, stroke_color=INK, stroke_width=2, fill_opacity=0)
                        .move_to([x0 + i * sp, base_y + pref[i] * 1.5, 0]) for i in range(6)]).set_z_index(2)
        t = ValueTracker(0)

        def pbars():
            p = hist[int(t.get_value())][0]
            return VGroup(*[Rectangle(width=bw, height=max(p[i] * 3, 0.02), stroke_width=0, fill_color=cols[i], fill_opacity=0.75)
                            .move_to([x0 + i * sp, base_y + p[i] * 1.5, 0]) for i in range(6)])
        pb = always_redraw(pbars)
        axis = Line([x0 - 0.4, base_y, 0], [x0 + 5 * sp + 0.4, base_y, 0], color=MUTED)
        lbl = VGroup(*[MathTex(s, font_size=24, color=c).move_to([x0 + i * sp, base_y - 0.3, 0])
                       for i, (s, c) in enumerate(zip([r"y_w", r"y_l", r"y_3", r"y_4", r"y_5", r"y_6"], cols))])
        hdr = VGroup(Text("policy (filled) vs reference (outline)", font_size=20)).move_to([x0 + 2.5 * sp, 1.3, 0])

        # right: the two implicit rewards on a number line
        nl = NumberLine(x_range=[-2.5, 1.0, 0.5], length=6.0, color=MUTED, include_numbers=True, font_size=20,
                        decimal_number_config={"color": INK, "num_decimal_places": 1}).move_to([2.9, 0.55, 0])
        nlh = MathTex(r"\hat r = \beta\log\frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}", font_size=30).next_to(nl, UP, buff=0.45)
        dw = always_redraw(lambda: Dot(nl.n2p(beta * hist[int(t.get_value())][1]), radius=0.12, color=GREEN_))
        dl = always_redraw(lambda: Dot(nl.n2p(beta * hist[int(t.get_value())][2]), radius=0.12, color=RED_))
        brace = always_redraw(lambda: BraceBetweenPoints(nl.n2p(beta * hist[int(t.get_value())][2]),
                                                         nl.n2p(beta * hist[int(t.get_value())][1]) + 0.001 * RIGHT,
                                                         direction=DOWN, color=PURPLE_).shift(0.25 * DOWN))
        mtxt = always_redraw(lambda: MathTex(rf"\beta\,\text{{margin}} = {beta * hist[int(t.get_value())][3]:.2f}",
                                             font_size=28, color=PURPLE_).move_to([2.9, -0.55, 0]))
        # push strength bar
        sx = 0.4
        sb_bg = Rectangle(width=4.0, height=0.28, stroke_color=MUTED, stroke_width=1.5).move_to([3.4, -1.55, 0])
        sb = always_redraw(lambda: Rectangle(width=max(4.0 * hist[int(t.get_value())][4] / 0.5 * 0.5, 0.02), height=0.28,
                                             stroke_width=0, fill_color=ORANGE_, fill_opacity=0.85)
                           .align_to(sb_bg, LEFT).set_y(-1.55))
        sbl = MathTex(r"\text{push } \sigma(-\beta\,\text{margin})", font_size=26, color=ORANGE_).next_to(sb_bg, LEFT, buff=0.2)
        kl = always_redraw(lambda: MathTex(rf"\mathrm{{KL}}(\pi\|\pi_{{\text{{ref}}}}) = {hist[int(t.get_value())][5]:.3f}",
                                           font_size=28).move_to([2.9, -2.35, 0]))

        cap = swap_caption(self, cap, "a toy policy over 6 answers; one preference: chosen (green) over rejected (red)", size=26)
        self.play(FadeIn(refb), FadeIn(pb), Create(axis), FadeIn(lbl), FadeIn(hdr))
        self.play(Create(nl), FadeIn(nlh), FadeIn(dw), FadeIn(dl))
        self.wait(0.6)
        self.add(brace, mtxt)
        self.play(FadeIn(sb_bg), FadeIn(sb), FadeIn(sbl), FadeIn(kl))
        cap = swap_caption(self, cap, "gradient steps: the chosen log-ratio rises, the rejected one falls", size=26)
        self.play(t.animate.set_value(15), run_time=4, rate_func=linear)
        cap = swap_caption(self, cap, "the push σ(−β·margin) fades once the pair is ranked correctly", color=ORANGE_, size=26)
        self.play(t.animate.set_value(T), run_time=5, rate_func=linear)
        self.wait(0.8)
        m95 = np.log(19)
        cap = swap_caption(self, cap, "the push falls below 5% once the log-ratio margin reaches ln19 / β", size=26)
        ax = NumberLine(x_range=[0, 30, 5], length=5.6, color=MUTED, include_numbers=True, font_size=20,
                        decimal_number_config={"color": INK, "num_decimal_places": 0}).move_to([3.1, -1.75, 0])
        axl = Text("log-ratio margin where the push stops", font_size=18).next_to(ax, DOWN, buff=0.35)
        marks = VGroup()
        for b_, c_ in [(0.5, PURPLE_), (0.1, ORANGE_)]:
            pt = ax.n2p(m95 / b_)
            marks.add(VGroup(Triangle(fill_color=c_, fill_opacity=1, stroke_width=0).scale(0.12).rotate(PI).next_to(pt, UP, buff=0.02),
                             MathTex(rf"\beta={b_}:\ {m95 / b_:.1f}", font_size=24, color=c_).next_to(pt, UP, buff=0.25)))
        self.play(FadeOut(VGroup(sb_bg, sbl, kl)), FadeOut(sb), Create(ax), FadeIn(axl))
        self.remove(sb, kl)
        self.play(LaggedStart(*[FadeIn(mk, shift=0.1 * DOWN) for mk in marks], lag_ratio=0.5))
        self.wait(1.5)
        cap = swap_caption(self, cap, "larger β: a smaller move away from π_ref already satisfies the loss", color=PURPLE_, size=26)
        self.wait(2.5)


# ------------------------------------------------------------------------------------------------- GRPO
class PolicyGradientBars(Scene):
    """Policy gradient derived line by line, then GRPO on a real group: 8 samples of Qwen2.5-0.5B-Instruct for
    '17 × 24', verifiable rewards, group-normalised advantages, and the real change in log π(y_i) after one SGD step.
    Then an all-wrong group (7919 × 6421): zero advantages, zero signal. Target slide: p12 'GRPO'."""

    def construct(self):
        D = json.loads((DATA / "w11_grpo.json").read_text())
        title(self, "Policy gradient → GRPO", "learn from sampled answers and a checker")
        L = r"\log\pi_\theta(y\mid x)"
        lines = [(r"J(\theta)", r"=", r"\mathbb{E}_{y\sim\pi_\theta}\big[r(x,y)\big] = \sum_y \pi_\theta(y\mid x)\,r(x,y)"),
                 (r"\nabla_\theta J", r"=", r"\sum_y r(x,y)\,\nabla_\theta \pi_\theta(y\mid x)"),
                 (r"", r"=", r"\sum_y \pi_\theta(y\mid x)\, r(x,y)\,\nabla_\theta" + L),
                 (r"", r"=", r"\mathbb{E}_y\big[\,r(x,y)\,\nabla_\theta" + L + r"\,\big]"),
                 (r"", r"=", r"\mathbb{E}_y\big[\,(r(x,y) - b)\,\nabla_\theta" + L + r"\,\big]"),
                 (r"A_i", r"=", r"\frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}"),
                 (r"\nabla_\theta J", r"\approx", r"\frac{1}{G}\sum_{i=1}^{G} A_i\,\nabla_\theta\log\pi_\theta(y_i\mid x)")]
        caps = ["goal: maximise the expected reward of the model's own answers",
                "the reward is just a number from a checker; only the probabilities depend on θ",
                "log-derivative trick: ∇π = π · ∇log π",
                "so the gradient is an average over samples from the model itself",
                "subtract a baseline b: E[∇log π] = ∇Σπ = ∇1 = 0, so the mean is unchanged, the variance drops",
                "GRPO: b = the mean reward of G answers to the same prompt, scaled by their std",
                "so: raise log π of answers above the group mean, lower the ones below"]
        cap, mobs = None, VGroup()
        ys = [2.15, 1.45, 0.75, 0.1, -0.55, -1.35, -2.3]
        eq_x = -1.6
        for i, (lhs, rel, rhs) in enumerate(lines):
            m = MathTex(lhs or r"\,", rel, rhs, font_size=30)
            m.shift([eq_x - m[1].get_center()[0], ys[i] - m[1].get_center()[1], 0])
            if cap is None:
                cap = caption(caps[i], size=26)
                self.play(FadeIn(cap), Write(m), run_time=1.1)
            else:
                cap = swap_caption(self, cap, caps[i], size=26 if len(caps[i]) < 80 else 24)
                self.play(Write(m), mobs[-1].animate.set_opacity(0.45), run_time=1.1)
            mobs.add(m)
            self.wait(1.1)
        keep = mobs[5]
        self.play(FadeOut(VGroup(*mobs[:5], mobs[6])), keep.animate.set_opacity(1).scale(0.7).to_corner(UR, buff=0.4))

        # ---- a real group
        G = 8
        xs = [-5.75 + i * 1.22 for i in range(G)]
        yA, yR, yAdv, yD = 1.0, 0.45, -0.85, -2.0
        lx = 4.35

        def tags(g):
            out = VGroup()
            for i in range(G):
                txt = "cut off" if g["truncated"][i] else g["finals"][i]
                col = GREEN_ if g["rewards"][i] else RED_
                t = Text(txt, font_size=17 if len(txt) > 5 else 20, color=col)
                box = RoundedRectangle(corner_radius=0.08, width=1.08, height=0.42, stroke_color=col, stroke_width=2,
                                       fill_color=SOFT, fill_opacity=1).move_to([xs[i], yA, 0])
                if t.width > 1.0:
                    t.scale_to_fit_width(1.0)
                out.add(VGroup(box, t.move_to(box)))
            return out

        def rewards(g):
            return VGroup(*[MathTex(f"r={int(r)}", font_size=26, color=GREEN_ if r else RED_).move_to([xs[i], yR, 0])
                            for i, r in enumerate(g["rewards"])])

        def vbars(vals, y0, scale, fmt, col_pos, col_neg):
            out = VGroup()
            for i, v in enumerate(vals):
                h = max(abs(v) * scale, 0.02)
                r = Rectangle(width=0.55, height=h, stroke_width=0, fill_color=col_pos if v >= 0 else col_neg,
                              fill_opacity=0.85).move_to([xs[i], y0 + (h / 2 if v >= 0 else -h / 2), 0])
                lab = Text(fmt.format(v), font_size=16, color=INK).next_to(r, UP if v >= 0 else DOWN, buff=0.05)
                out.add(VGroup(r, lab))
            return out

        def base(y0):
            return Line([xs[0] - 0.5, y0, 0], [xs[-1] + 0.5, y0, 0], color=MUTED, stroke_width=2)

        g = D["mixed"]
        r = np.array(g["rewards"])
        A = np.array(g["adv"])
        dlp = np.array(g["sweep"]["0.0001"][1]) - np.array(g["logp_before"])
        prompt = token_row(["What is 17 × 24?"], color=BLUE_, size=24)[0].move_to([-4.3, 1.75, 0])
        src = Text("8 samples, Qwen2.5-0.5B-Instruct, T = 1", font_size=18, color=MUTED).next_to(prompt, RIGHT, buff=0.3)
        T = tags(g)
        cap = swap_caption(self, cap, "one prompt, G = 8 sampled answers, a checker gives each a 0/1 reward", size=26)
        self.play(FadeIn(prompt), FadeIn(src))
        self.play(LaggedStart(*[FadeIn(t, shift=0.1 * DOWN) for t in T], lag_ratio=0.12), run_time=1.6)
        R = rewards(g)
        lab_r = Text("reward", font_size=20).move_to([lx, yR, 0], aligned_edge=LEFT)
        lab_a = Text("final answer", font_size=20).move_to([lx, yA, 0], aligned_edge=LEFT)
        self.play(FadeIn(R), FadeIn(lab_r), FadeIn(lab_a))
        self.wait(0.8)

        def statm(rr):
            return VGroup(MathTex(rf"\text{{mean}} = {rr.mean():.2f}", font_size=26),
                          MathTex(rf"\text{{std}} = {rr.std():.2f}", font_size=26)).arrange(DOWN, aligned_edge=LEFT, buff=0.1
                          ).move_to([lx, -0.15, 0], aligned_edge=LEFT)
        stats = statm(r)
        lab_adv = VGroup(Text("advantage", font_size=20, color=PURPLE_), MathTex("A_i", font_size=28, color=PURPLE_)).arrange(RIGHT, buff=0.1).move_to([lx, yAdv, 0], aligned_edge=LEFT)
        AB = vbars(A, yAdv, 0.45, "{:+.2f}", PURPLE_, PURPLE_)
        cap = swap_caption(self, cap, f"normalise within the group: right answers get +{A.max():.2f}, wrong ones {A.min():.2f}", size=26)
        self.play(FadeIn(stats), FadeIn(lab_adv), Create(base(yAdv)))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if a >= 0 else UP) for b, a in zip(AB, A)], lag_ratio=0.08), run_time=1.4)
        self.wait(1.0)

        lab_d = VGroup(MathTex(r"\Delta\log\pi_\theta(y_i\mid x)", font_size=26, color=GREEN_),
                       Text("after one real SGD step", font_size=16, color=MUTED)).arrange(DOWN, aligned_edge=LEFT, buff=0.06
                       ).move_to([lx, yD, 0], aligned_edge=LEFT)
        DB = vbars(dlp, yD, 0.09, "{:+.1f}", GREEN_, RED_)
        cap = swap_caption(self, cap, f"one gradient step: the right answers become {np.exp(dlp[r > 0].min()):.1f}–{np.exp(dlp[r > 0].max()):.1f}× more likely",
                           color=GREEN_, size=26)
        self.play(FadeIn(lab_d), Create(base(yD)))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN if v >= 0 else UP) for b, v in zip(DB, dlp)], lag_ratio=0.08), run_time=1.4)
        self.wait(2.0)

        # ---- all-fail group
        h = D["hard"]
        rh = np.array(h["rewards"])
        prompt2 = token_row(["What is 7919 × 6421?"], color=BLUE_, size=24)[0].move_to(prompt).align_to(prompt, LEFT)
        T2, R2 = tags(h), rewards(h)
        cap = swap_caption(self, cap, "a harder prompt: all 8 answers are wrong", color=RED_, size=26)
        self.play(FadeOut(prompt), FadeIn(prompt2), src.animate.next_to(prompt2, RIGHT, buff=0.3),
                  *[Transform(a, b) for a, b in zip(T, T2)], *[Transform(a, b) for a, b in zip(R, R2)])
        self.wait(1.0)
        stats2 = statm(rh)
        zA = vbars(np.zeros(G), yAdv, 0.45, "{:.0f}", PURPLE_, PURPLE_)
        zD = vbars(np.zeros(G), yD, 0.09, "{:.0f}", GREEN_, RED_)
        cap = swap_caption(self, cap, "every reward equals the mean: all advantages are 0, the gradient is exactly zero", color=RED_, size=26)
        self.play(Transform(stats, stats2), *[Transform(a, b) for a, b in zip(AB, zA)], *[Transform(a, b) for a, b in zip(DB, zD)],
                  run_time=1.4)
        self.wait(1.8)
        cap = swap_caption(self, cap, "all-wrong (or all-right) groups teach nothing: train on prompts at the edge of ability",
                           color=PURPLE_, size=26)
        self.wait(2.5)
