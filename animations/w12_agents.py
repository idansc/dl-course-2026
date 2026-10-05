"""Week 12 (agents, retrieval, test-time compute). Retrieval numbers come from `python w12_data.py`
(data/w12_retrieval.json: MiniLM cosines, BM25, ColBERT token similarities). Render one scene:
    manim -qh --disable_caching -o AgentLoop w12_agents.py AgentLoop
"""
import json
from math import comb
from pathlib import Path
from style import *

DATA = Path(__file__).parent / "data"


# ------------------------------------------------------------------------------------------------- agent loop
class AgentLoop(Scene):
    """Think → tool call → observation → repeat, on one concrete task, with a step budget; the context grows by
    one observation per turn. The python result is real; the search snippets are abbreviated. Target: L12b agents."""

    def construct(self):
        title(self, "The agent loop", "an LLM, a few tools, and a step budget")
        c0, R = np.array([-4.55, -0.35, 0]), 1.45
        angs = [PI / 2, PI / 2 - 2 * PI / 3, PI / 2 + 2 * PI / 3]
        names, cols = ["think", "act", "observe"], [PURPLE_, ORANGE_, GREEN_]
        nodes = VGroup()
        for a, n, c in zip(angs, names, cols):
            p = c0 + R * np.array([np.cos(a), np.sin(a), 0])
            nodes.add(VGroup(Circle(radius=0.58, color=c, stroke_width=4, fill_color=WHITE, fill_opacity=1).move_to(p),
                             Text(n, font_size=20, color=c, weight=BOLD).move_to(p)))
        paths = [ArcBetweenPoints(c0 + R * np.array([np.cos(a1 - 0.45), np.sin(a1 - 0.45), 0]),
                                  c0 + R * np.array([np.cos(a2 + 0.45), np.sin(a2 + 0.45), 0]),
                                  angle=-2 * PI / 3 + 0.9, color=MUTED, stroke_width=3)
                 for a1, a2 in zip(angs, angs[1:] + angs[:1])]
        arcs = VGroup(*[p.copy().add_tip(tip_length=0.18) for p in paths])
        llm = Text("LLM", font_size=24, color=INK, weight=BOLD).move_to(c0)
        # step budget
        budget = VGroup(*[Square(0.32, stroke_color=INK, stroke_width=2) for _ in range(6)]).arrange(RIGHT, buff=0.08)
        budget.move_to(c0 + np.array([0, -2.25, 0]))
        blab = Text("step budget: 6", font_size=18, color=MUTED).next_to(budget, UP, buff=0.12)

        # context panel
        panel = RoundedRectangle(corner_radius=0.12, width=8.6, height=5.0, stroke_color=MUTED, stroke_width=2).move_to([2.2, -0.45, 0])
        plab = Text("context window (everything the model sees)", font_size=18, color=MUTED).next_to(panel, UP, buff=0.08).align_to(panel, LEFT)
        y_cur = [panel.get_top()[1] - 0.32]

        def line(tag, text, col, size=19):
            t = Text(tag, font_size=size, color=col, weight=BOLD)
            b = Text(text, font_size=size, color=INK if tag != "obs" else GREEN_)
            g = VGroup(t, b).arrange(RIGHT, buff=0.15, aligned_edge=UP)
            if g.width > panel.width - 0.4:
                g.scale_to_fit_width(panel.width - 0.4)
            g.move_to([0, y_cur[0], 0]).align_to(panel, LEFT).shift(0.2 * RIGHT)
            y_cur[0] -= 0.36
            return g

        cap = caption("an agent: a language model in a loop with tools, on one task")
        task = line("task", "How many days passed between the Apollo 11 and Apollo 17 Moon landings?", BLUE_)
        self.play(FadeIn(cap), *[FadeIn(n) for n in nodes], *[Create(a) for a in arcs], FadeIn(llm))
        self.play(Create(panel), FadeIn(plab), FadeIn(task), FadeIn(budget), FadeIn(blab))
        self.wait(0.6)

        dot = Dot(nodes[0].get_center(), radius=0.1, color=INK).set_z_index(5)
        steps = [("I need both landing dates; start with Apollo 11.", 'search("Apollo 11 landing date")',
                  "Apollo 11's Eagle landed on July 20, 1969."),
                 ("Now the Apollo 17 date.", 'search("Apollo 17 landing date")',
                  "Apollo 17's Challenger landed on December 11, 1972."),
                 ("Date arithmetic is error-prone: use code.", "python(date(1972,12,11) - date(1969,7,20))",
                  "1240 days, 0:00:00")]
        caps = [("think: the model writes a plan as text", "act: it emits a tool call instead of an answer",
                 "observe: the tool's output is appended to the context"),
                ("repeat: every turn re-reads everything so far", None, None),
                ("a python call does the arithmetic exactly", None, None)]

        def pulse(i):
            return Indicate(nodes[i], color=cols[i], scale_factor=1.15)

        for s, (th, ac, ob) in enumerate(steps):
            slow = s == 0
            rt = 0.8 if slow else 0.5
            c1, c2, c3 = caps[s]
            if c1:
                cap = swap_caption(self, cap, c1)
            self.add(dot)
            self.play(dot.animate.move_to(nodes[0]), pulse(0), FadeIn(l1 := line("think", th, PURPLE_)), run_time=rt)
            if c2:
                cap = swap_caption(self, cap, c2)
            self.play(MoveAlongPath(dot, paths[0]), pulse(1), FadeIn(l2 := line("act", ac, ORANGE_)), run_time=rt)
            if c3:
                cap = swap_caption(self, cap, c3)
            self.play(MoveAlongPath(dot, paths[1]), pulse(2), FadeIn(l3 := line("obs", ob, GREEN_), shift=0.1 * LEFT),
                      budget[s].animate.set_fill(ORANGE_, 0.8), run_time=rt)
            self.play(MoveAlongPath(dot, paths[2]), run_time=rt * 0.7)
            self.wait(0.8 if slow else 0.5)

        cap = swap_caption(self, cap, "stop when the model answers, or when the step budget runs out", color=PURPLE_)
        self.play(pulse(0), budget[3].animate.set_fill(ORANGE_, 0.8),
                  FadeIn(line("think", "I have both dates and the difference.", PURPLE_)), run_time=0.7)
        ans = line("answer", "1,240 days.", BLUE_, size=22)
        box = SurroundingRectangle(ans, color=BLUE_, buff=0.08)
        self.play(FadeIn(ans), Create(box))
        used = Text("step budget: 4 of 6 used", font_size=18, color=ORANGE_).move_to(blab)
        self.play(FadeTransform(blab, used))
        self.wait(1.5)
        cap = swap_caption(self, cap, "the context grows each turn: long tasks are limited by context and budget", size=26)
        self.wait(2.5)


# ------------------------------------------------------------------------------------------------- dense vs BM25
class DenseRetrieval(Scene):
    """Query and 8 passages as MiniLM embeddings (drawn at their real angle to the query), nearest neighbours by
    cosine, and BM25 term-overlap scores side by side (real numbers). Target slide: w12 'Retrieval: sparse vs dense'."""

    def construct(self):
        d = json.loads((DATA / "w12_retrieval.json").read_text())
        cos, bm, short = np.array(d["cos"]), np.array(d["bm25"]), d["short"]
        order = np.argsort(-cos)
        title(self, "Dense retrieval vs BM25", f"the same query and 8 passages, scored two ways (MiniLM-L6, {d['dim']}-d)")
        c0, Rr = np.array([-4.7, -0.55, 0]), 1.95
        q = Arrow(c0, c0 + Rr * RIGHT * 1.08, buff=0, color=BLUE_, stroke_width=7)
        qlab = Text(f'query: "{d["query"]}"', font_size=22, color=BLUE_).move_to([c0[0] + 0.6, 1.85, 0])
        arc = Arc(radius=Rr, start_angle=-PI / 2, angle=PI, arc_center=c0, color=MUTED, stroke_width=1.5)
        radii = [1.85, 1.85, 1.2, 1.5, 1.5, 1.05, 1.85, 1.85]      # per rank; up side = even ranks
        pts, dots, nums = [], VGroup(), VGroup()
        for rank, i in enumerate(order):
            th = np.arccos(cos[i]) * (1 if rank % 2 == 0 else -1)
            r = radii[rank]
            p = c0 + r * np.array([np.cos(th), np.sin(th), 0])
            pts.append(p)
            dots.add(Dot(p, radius=0.09, color=INK))
            nums.add(Text(str(rank + 1), font_size=18, color=INK).move_to(c0 + (r + 0.25) * np.array([np.cos(th), np.sin(th), 0])))
        rays = VGroup(*[Line(c0, p, color=MUTED, stroke_width=1.5, stroke_opacity=0.6) for p in pts])

        # table
        x_txt, x_cos, x_bm = -1.15, 4.0, 5.55
        rows_y = [1.15 - 0.47 * k for k in range(8)]
        hdr = VGroup(Text("passage", font_size=20, weight=BOLD).move_to([x_txt, 1.65, 0], aligned_edge=LEFT),
                     Text("cosine", font_size=20, weight=BOLD, color=PURPLE_).move_to([x_cos + 0.5, 1.65, 0]),
                     Text("BM25", font_size=20, weight=BOLD, color=ORANGE_).move_to([x_bm + 0.6, 1.65, 0]))
        texts = VGroup(*[Text(f"{k + 1}  {short[i]}", font_size=18).move_to([x_txt, rows_y[k], 0], aligned_edge=LEFT)
                         for k, i in enumerate(order)])

        def bar(v, vmax, x, y, col, fmt):
            w = max(0.02, v / vmax * 0.85)
            r = Rectangle(width=w, height=0.24, stroke_width=0, fill_color=col, fill_opacity=0.85).move_to([x + w / 2, y, 0])
            return VGroup(r, Text(fmt.format(v), font_size=15, color=INK).next_to(r, RIGHT, buff=0.06))

        cbars = VGroup(*[bar(cos[i], cos.max(), x_cos, rows_y[k], PURPLE_, "{:.2f}") for k, i in enumerate(order)])
        bbars = VGroup(*[bar(bm[i], bm.max(), x_bm, rows_y[k], ORANGE_, "{:.1f}") for k, i in enumerate(order)])

        cap = caption("query and passages become vectors, drawn at their real angle to the query")
        self.play(FadeIn(cap), FadeIn(qlab), Create(arc), GrowArrow(q))
        self.play(LaggedStart(*[AnimationGroup(Create(rl), FadeIn(dt), FadeIn(n)) for rl, dt, n in zip(rays, dots, nums)], lag_ratio=0.1),
                  FadeIn(hdr[0]), LaggedStart(*[FadeIn(t) for t in texts], lag_ratio=0.1), run_time=2.2)
        self.wait(0.8)

        cap = swap_caption(self, cap, "dense score: cosine similarity = cos(angle to the query)")
        self.play(FadeIn(hdr[1]), LaggedStart(*[GrowFromEdge(b, LEFT) for b in cbars], lag_ratio=0.08), run_time=1.5)
        th3 = np.arccos(cos[order[2]])
        cone = Sector(radius=Rr + 0.2, angle=2 * th3 + 0.06, start_angle=-th3 - 0.03, arc_center=c0,
                      fill_color=GREEN_, fill_opacity=0.12, stroke_width=0)
        top3 = VGroup(*[SurroundingRectangle(texts[k], color=GREEN_, buff=0.05) for k in range(3)])
        cap = swap_caption(self, cap, "nearest neighbours: the top 3 are all about tyres, two without the word 'fix'", color=GREEN_, size=26)
        self.play(FadeIn(cone), *[dots[k].animate.set_color(GREEN_) for k in range(3)], Create(top3))
        self.wait(1.8)

        cap = swap_caption(self, cap, "BM25: sum over shared words, weighted by rarity (idf) and passage length")
        bm_eq = MathTex(r"\text{BM25}=\sum_{w\in q\cap p}\mathrm{idf}(w)\,\frac{tf\,(k_1+1)}{tf + k_1(1-b+b\,|p|/\overline{|p|})}",
                        font_size=26, color=ORANGE_).move_to([2.7, -2.75, 0])
        self.play(FadeIn(hdr[2]), Write(bm_eq), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bbars], lag_ratio=0.08), run_time=1.8)
        kf = int(np.where(order == int(np.argmax(bm)))[0][0])
        kr = int(np.where(order == 0)[0][0])
        wrong = SurroundingRectangle(VGroup(texts[kf], bbars[kf]), color=RED_, buff=0.05)
        terms = d["bm25_terms"][int(np.argmax(bm))]
        cap = swap_caption(self, cap, f"BM25's winner is the furniture passage: it shares {len(terms)} words ({', '.join(terms)})",
                           color=RED_, size=26)
        self.play(Create(wrong))
        self.wait(1.8)
        miss = SurroundingRectangle(VGroup(texts[kr], bbars[kr]), color=ORANGE_, buff=0.05)
        cap = swap_caption(self, cap, f"and the bicycle-repair passage gets {bm[0]:.1f}: no shared word except 'a'", size=26)
        self.play(Create(miss))
        self.wait(1.8)
        cap = swap_caption(self, cap, "dense matches meaning, BM25 matches exact terms: real systems use both (hybrid)",
                           color=PURPLE_, size=26)
        self.wait(3.0)


# ------------------------------------------------------------------------------------------------- MaxSim
class MaxSim(Scene):
    """ColBERT late interaction on real token embeddings (answerai-colbert-small-v1): the query-token × passage-token
    cosine matrix, each query token's best match, the summed score; a lexical passage for comparison."""

    def construct(self):
        d = json.loads((DATA / "w12_retrieval.json").read_text())
        c, c2 = d["colbert"]
        qt, pt, S = c["q_tokens"], c["p_tokens"], np.array(c["sim"])
        nq, npt = S.shape
        title(self, "ColBERT: late interaction (MaxSim)", "one vector per token; each query token finds its best match")
        cell = 0.42
        lo, hi = S.min(), S.max()
        hm = heatmap((S - lo) / (hi - lo), cell=cell, color=PURPLE_).move_to([-0.9, -0.85, 0])
        cols = VGroup()
        for j, t in enumerate(pt):
            lab = Text(t, font_size=16).rotate(PI / 2.6)
            lab.move_to(hm[j].get_top() + 0.12 * UP, aligned_edge=DL).shift(0.05 * LEFT)
            cols.add(lab)
        cols.set_z_index(3)
        rows = VGroup(*[Text(t, font_size=20, color=BLUE_).next_to(hm[i * npt], LEFT, buff=0.15) for i, t in enumerate(qt)])
        plab = Text("passage tokens (wordpieces)", font_size=18, color=MUTED).next_to(cols, UP, buff=0.12).align_to(hm, LEFT)
        qlab = Text("query", font_size=18, color=BLUE_).next_to(rows, UP, buff=0.12)

        cap = caption("ColBERT keeps one vector per token: 7 for the query, 15 for the passage")
        self.play(FadeIn(cap), FadeIn(rows), FadeIn(qlab), LaggedStart(*[FadeIn(t) for t in cols], lag_ratio=0.04), FadeIn(plab))
        self.wait(0.6)
        cap = swap_caption(self, cap, "compare every query token with every passage token (cosine)")
        self.play(FadeIn(hm), run_time=1.2)
        scale = Text(f"white = {lo:.2f}   purple = {hi:.2f}", font_size=16, color=MUTED).next_to(hm, DOWN, buff=0.15).align_to(hm, LEFT)
        self.play(FadeIn(scale))
        self.wait(0.6)

        cap = swap_caption(self, cap, "MaxSim: each query token keeps only its single best match")
        xm = hm.get_right()[0] + 0.75
        mhdr = Text("max", font_size=20, weight=BOLD, color=PURPLE_).move_to([xm, hm[0].get_center()[1] + 0.45, 0])
        self.play(FadeIn(mhdr))
        vals = VGroup()
        for i in range(nq):
            j = int(S[i].argmax())
            rowbox = SurroundingRectangle(VGroup(*[hm[i * npt + k] for k in range(npt)]), color=BLUE_, buff=0.02, stroke_width=3)
            best = SurroundingRectangle(hm[i * npt + j], color=GREEN_, buff=0.02, stroke_width=5)
            v = Text(f"{S[i, j]:.2f}", font_size=20, color=GREEN_).move_to([xm, hm[i * npt].get_center()[1], 0])
            rt = 0.55 if i < 3 else 0.35
            self.play(Create(rowbox), run_time=rt * 0.6)
            self.play(Create(best), Indicate(cols[j], color=GREEN_, scale_factor=1.1), TransformFromCopy(hm[i * npt + j], v), run_time=rt)
            self.play(FadeOut(rowbox), run_time=0.2)
            vals.add(v)
            if i == 3:
                cap = swap_caption(self, cap, "content words find paraphrases: fix → repairing, flat → punctured, tire → wheel", color=GREEN_, size=26)
        line = Line([xm - 0.4, 0, 0], [xm + 0.4, 0, 0], color=INK, stroke_width=2).next_to(vals, DOWN, buff=0.1)
        tot = Text(f"{c['maxsim']:.2f}", font_size=24, weight=BOLD).next_to(line, DOWN, buff=0.1)
        eq = MathTex(r"s(q,p)=\sum_{i\in q}\max_{j\in p}\ \mathbf{q}_i\cdot\mathbf{p}_j", font_size=30).move_to([4.6, 2.1, 0])
        cap = swap_caption(self, cap, "the score is the sum of the per-token maxima")
        self.play(Create(line), FadeIn(tot), Write(eq))
        self.wait(1.2)
        cmp_ = VGroup(Text(f"repair passage   {c['maxsim']:.2f}", font_size=20, color=GREEN_),
                      Text(f"'flat tire prices' passage   {c2['maxsim']:.2f}", font_size=20, color=RED_)
                      ).arrange(DOWN, aligned_edge=RIGHT, buff=0.15).move_to([5.0, -0.9, 0]).to_edge(RIGHT, buff=0.4)
        cap = swap_caption(self, cap, "the repair passage outranks the one that repeats 'flat' and 'tire'", size=26)
        self.play(FadeIn(cmp_))
        self.wait(1.8)
        cap = swap_caption(self, cap, "passage vectors are precomputed offline; query time is only max and sum",
                           color=PURPLE_, size=26)
        self.wait(2.5)


# ------------------------------------------------------------------------------------------------- pass@k
def bench(n_prob=600, kmax=20, trials=200, seed=0):
    """Synthetic benchmark: per-problem success p ~ Beta(0.5, 1); wrong answers spread over 5 distractors with
    Dirichlet(0.5) weights; a noisy verifier scores correct answers N(1, 0.8²), wrong ones N(0, 0.8²)."""
    rng = np.random.default_rng(seed)
    p = rng.beta(0.5, 1.0, n_prob)
    W = rng.dirichlet([0.5] * 5, n_prob)
    passk, maj, ver = np.zeros(kmax), np.zeros(kmax), np.zeros(kmax)
    for i in range(n_prob):
        probs = np.r_[p[i], (1 - p[i]) * W[i]]
        ans = rng.choice(6, size=(trials, kmax), p=probs)               # 0 = correct
        corr = ans == 0
        score = corr * 1.0 + rng.normal(0, 0.8, (trials, kmax))
        for k in range(1, kmax + 1):
            passk[k - 1] += corr[:, :k].any(1).mean()
            cnt = np.stack([(ans[:, :k] == a).sum(1) for a in range(6)], 1) + rng.uniform(0, 0.1, (trials, 6))
            maj[k - 1] += (cnt.argmax(1) == 0).mean()
            ver[k - 1] += corr[np.arange(trials), score[:, :k].argmax(1)].mean()
    return passk / n_prob, maj / n_prob, ver / n_prob


class PassAtK(Scene):
    """pass@k = 1 − C(n−c,k)/C(n,k) derived on a grid of n = 20 samples with c = 3 correct; the curve grows with k.
    Then a synthetic benchmark: pass@k (oracle) vs majority vote vs a noisy verifier. Target: L12b test-time compute."""

    def construct(self):
        title(self, "pass@k, majority vote, verifiers", "spending more samples at test time")
        n, c = 20, 3
        rng = np.random.default_rng(4)
        correct = set(rng.choice(n, c, replace=False).tolist())
        grid = VGroup(*[Circle(radius=0.2, stroke_width=3, stroke_color=GREEN_ if i in correct else RED_,
                               fill_color=GREEN_ if i in correct else RED_, fill_opacity=0.25) for i in range(n)])
        grid.arrange_in_grid(4, 5, buff=0.22).move_to([-4.6, 0.2, 0])
        glab = Text(f"n = {n} samples, c = {c} correct", font_size=22).next_to(grid, UP, buff=0.25)
        cap = caption("sample n answers to one problem; c of them are correct")
        self.play(FadeIn(cap), LaggedStart(*[FadeIn(g, scale=0.5) for g in grid], lag_ratio=0.03), FadeIn(glab))
        self.wait(0.5)

        lines = [r"\text{draw } k \text{ of the } n \text{ without replacement}",
                 r"P(\text{all } k \text{ wrong}) = \binom{n-c}{k}\Big/\binom{n}{k}",
                 r"\text{pass@}k = 1 - \binom{n-c}{k}\Big/\binom{n}{k}"]
        caps = ["pass@k: the chance that at least one of k drawn answers is correct",
                "it fails only if all k come from the n − c wrong ones",
                "so pass@k is one minus that ratio: an unbiased estimate from n samples"]
        mobs = VGroup()
        for i, (tex, text) in enumerate(zip(lines, caps)):
            m = MathTex(tex, font_size=32).move_to([2.6, 1.6 - 0.95 * i, 0])
            cap = swap_caption(self, cap, text)
            anims = [Write(m)]
            if i:
                anims.append(mobs[-1].animate.set_opacity(0.45))
            if i == 0:
                pick = sorted(rng.choice(n, 5, replace=False).tolist())
                rings = VGroup(*[Circle(radius=0.28, color=INK, stroke_width=3).move_to(grid[j]) for j in pick])
                anims.append(Create(rings))
            self.play(*anims, run_time=1.1)
            mobs.add(m)
            self.wait(1.1)
        k5 = 1 - comb(n - c, 5) / comb(n, 5)
        ex = MathTex(rf"k=5:\ 1 - \tbinom{{17}}{{5}}/\tbinom{{20}}{{5}} = 1 - \tfrac{{{comb(17, 5)}}}{{{comb(20, 5)}}} = {k5:.2f}",
                     font_size=30, color=PURPLE_).move_to([2.6, -1.4, 0])
        cap = swap_caption(self, cap, f"with 5 draws (ringed), at least one is correct {k5:.0%} of the time")
        self.play(Write(ex), mobs[-1].animate.set_opacity(1))
        self.wait(1.5)

        # the curve
        self.play(FadeOut(VGroup(mobs[:2], ex, rings)), mobs[2].animate.scale(0.8).to_corner(UR, buff=0.4))
        ax = Axes(x_range=[0, 20, 5], y_range=[0, 1, 0.25], x_length=6.2, y_length=3.6, tips=False,
                  axis_config={"color": MUTED, "include_numbers": True, "font_size": 20,
                               "decimal_number_config": {"color": INK, "num_decimal_places": 2}},
                  x_axis_config={"decimal_number_config": {"color": INK, "num_decimal_places": 0}}).move_to([2.7, -0.55, 0])
        xl = Text("k (samples drawn)", font_size=20).next_to(ax, DOWN, buff=0.12)
        ks = np.arange(1, n + 1)
        pk = np.array([1 - comb(n - c, k) / comb(n, k) for k in ks])
        curve = ax.plot_line_graph(ks, pk, line_color=PURPLE_, stroke_width=5, vertex_dot_radius=0.05,
                                   vertex_dot_style={"fill_color": PURPLE_})
        lab1 = Text(f"pass@k, this problem (c/n = {c / n:.2f})", font_size=18, color=PURPLE_).move_to(ax.c2p(13.5, 0.72))
        cap = swap_caption(self, cap, f"pass@k climbs fast: {pk[0]:.2f} at k=1, {pk[4]:.2f} at k=5, {pk[9]:.2f} at k=10", color=PURPLE_)
        self.play(Create(ax), FadeIn(xl))
        self.play(Create(curve), FadeIn(lab1), run_time=2.0)
        self.wait(1.5)

        # benchmark: oracle vs majority vote vs verifier
        P, M, V = bench()
        cap = swap_caption(self, cap, "but pass@k needs an oracle that knows which answer is right", size=26)
        self.play(FadeOut(curve), FadeOut(lab1), FadeOut(grid), FadeOut(glab))
        blab = Text("synthetic benchmark: 600 problems of varying difficulty", font_size=18, color=MUTED).move_to([-3.75, 1.3, 0])
        curves, labs = VGroup(), VGroup()
        for arr, col, name in [(P, PURPLE_, "pass@k (oracle)"), (V, GREEN_, "best-of-k by a noisy verifier"),
                               (M, ORANGE_, "majority vote")]:
            curves.add(ax.plot_line_graph(ks, arr, line_color=col, stroke_width=5, add_vertex_dots=False))
            labs.add(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=col, stroke_width=5), Text(f"{name}: {arr[0]:.2f} → {arr[-1]:.2f}", font_size=19, color=col)
                            ).arrange(RIGHT, buff=0.15))
        labs.arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to([-3.75, 0.2, 0])
        self.play(FadeIn(blab), Create(curves[0]), FadeIn(labs[0]), run_time=1.5)
        self.wait(0.8)
        cap = swap_caption(self, cap, "majority vote needs no checker, but stalls when a wrong answer is the most popular",
                           color=ORANGE_, size=26)
        self.play(Create(curves[2]), FadeIn(labs[2]), run_time=1.5)
        self.wait(1.5)
        cap = swap_caption(self, cap, "a verifier picks the best of k: better than voting, below the oracle", color=GREEN_, size=26)
        self.play(Create(curves[1]), FadeIn(labs[1]), run_time=1.5)
        self.wait(1.5)
        cap = swap_caption(self, cap, "the gap between pass@k and what we can select is the verifier's job", color=PURPLE_, size=26)
        self.wait(2.5)
