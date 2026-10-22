"""Lecture 2, CNN part: convolution, step by step. Render one scene:
    manim -qh --disable_caching -o ConvRunningExample w01_cnn.py ConvRunningExample
Slide version without words (renders/w01/clean/; its own media dir, parallel runs collide on the text cache):
    DLC_CLEAN=1 manim -qh --disable_caching --media_dir media_clean_ConvRunningExample -o ConvRunningExample \
        w01_cnn.py ConvRunningExample
The image, the filter and the output come from code/conv_running_example.py (the scene imports it, so
`python code/conv_running_example.py` prints exactly what the clip shows and checks it against PyTorch's conv2d).
Code table: code/README.md.
"""
import sys
from pathlib import Path

from style import *

sys.path.insert(0, str(Path(__file__).parent / "code"))     # `code` is also a stdlib module: import the file directly
import conv_running_example as conv_code                    # noqa: E402

CELL_IN, CELL_WIN, CELL_OUT = "#C1E4F7", "#C7E8AC", "#FFBBB1"   # the GIF's colours: input, window/filter, output
WEIGHT = "#166534"     # the filter's weights (dark green)
LABEL = "#6B7280"      # the small grid names (darker than MUTED: they are read on a projected slide)


class ConvRunningExample(Scene):
    """The lecture's running example (the classic UFLDL convolution schematic, redrawn): the 3×3 filter
    [[1,0,1],[0,1,0],[1,0,1]] is laid over a 5×5 binary image and slides row by row over all 9 positions (stride 1,
    no padding); in the window every cell shows value×weight, and their sum fills the output cell, until the output
    [[4,3,4],[2,4,3],[2,3,4]] is complete. Target: p3 slide 60 ("Running example"), in place of the GIF."""

    def construct(self):
        img, ker = conv_code.image, conv_code.kernel
        out = conv_code.conv2d(img, ker)                # stride 1, no padding (code/conv_running_example.py)
        N, K, M = img.shape[0], ker.shape[0], out.shape[0]
        S, SF, DY = 1.0, 0.5, 0.1                       # input/output cell, filter cell, vertical offset
        FS_V, FS_P, FS_W, FS_L = 40, 30, 24, 24         # font sizes: values, value×weight, weights, grid names
        IN0 = np.array([-5.0, 2.5 + DY, 0])            # top-left corners: input, output (2 cells to the right,
        OUT0 = IN0 + np.array([(N + 2) * S, -(N - M) * S, 0])     # bottoms aligned), filter (above the output)
        F0 = np.array([OUT0[0] + (M * S - K * SF) / 2, IN0[1] + 0.25, 0])
        clean_frame(self, center=(0.0, DY - 0.1), height=6.4)     # slide version: no title or caption, zoom in
        title(self, "Convolution: a running example")

        def centre(corner, r, c, s=S):
            return corner + np.array([(c + 0.5) * s, -(r + 0.5) * s, 0])

        def cell(corner, r, c, fill, s=S, stroke=2.0):
            return Square(s, stroke_color=INK, stroke_width=stroke, fill_color=fill,
                          fill_opacity=1).move_to(centre(corner, r, c, s))

        def under(i, j):
            """input cells under the window for output (i, j), row-major like the filter"""
            return [(i + a, j + b) for a in range(K) for b in range(K)]

        # value×weight layout inside a cell: glyph centres of "0×0" (all digits have the same width in Arial)
        ref = Text("0×0", font_size=FS_P)
        slot = [g.get_center() - ref.get_center() for g in ref]

        # ---- the three grids
        in_cells = VGroup(*[cell(IN0, r, c, CELL_IN) for r in range(N) for c in range(N)])
        vals = [[Text(str(img[r, c]), font_size=FS_V).move_to(centre(IN0, r, c)).set_z_index(2) for c in range(N)]
                for r in range(N)]
        f_cells = VGroup(*[cell(F0, a, b, CELL_WIN, SF, 1.5) for a in range(K) for b in range(K)])
        f_border = Square(K * SF, stroke_color=GREEN_, stroke_width=4).move_to(centre(F0, 1, 1, SF)).set_z_index(1)
        f_nums = VGroup(*[Text(str(ker[a, b]), font_size=FS_W, color=WEIGHT).move_to(centre(F0, a, b, SF))
                          for a in range(K) for b in range(K)])
        o_cells = [[cell(OUT0, i, j, WHITE) for j in range(M)] for i in range(M)]
        names = VGroup(
            Text("input", font_size=FS_L, color=LABEL).next_to(in_cells, DOWN, buff=0.14),
            Text("filter", font_size=FS_L, color=LABEL).next_to(f_cells, DOWN, buff=0.14),
            Text("output", font_size=FS_L, color=LABEL).next_to(VGroup(*sum(o_cells, [])), DOWN, buff=0.14))
        names[2].align_to(names[0], UP)

        # ---- the window: the filter laid over the image (green cells under the digits, thick border)
        win_cells = VGroup(*[cell(IN0, a, b, CELL_WIN) for a in range(K) for b in range(K)]).set_z_index(1)
        win_border = Square(K * S, stroke_color=GREEN_, stroke_width=6).move_to(centre(IN0, 1, 1)).set_z_index(3)
        win = VGroup(win_cells, win_border)
        # the output cell the window is computing: same green border; at stride 1 it moves exactly like the window
        here = Square(S, stroke_color=GREEN_, stroke_width=6).move_to(centre(OUT0, 0, 0)).set_z_index(3)
        # "×weight" next to each value under the window, kept per input cell: "×" and the weight's digit
        cross, digit = {}, {}

        def glyph(s, r, c, k):
            return Text(s, font_size=FS_P, color=WEIGHT).move_to(centre(IN0, r, c) + slot[k]).set_z_index(2)

        def cover(r, c, rate=smooth):           # the value moves left to make room for "×weight"
            return vals[r][c].animate(rate_func=rate).scale(FS_P / FS_V).move_to(centre(IN0, r, c) + slot[0])

        def uncover(r, c, rate=smooth):
            return vals[r][c].animate(rate_func=rate).scale(FS_V / FS_P).move_to(centre(IN0, r, c))

        def move_window(old, new):
            """One step of the window from position old to new (output indices): labels of the cells it leaves fade
            out first, the values glide between the plain and the value×weight slot, the weights of the cells it
            enters fade in last, so no two glyphs ever overlap; in cells it keeps only the weight digit changes."""
            before, after = set(under(*old)), set(under(*new))
            early, mid, late = (squish_rate_func(smooth, 0, 0.35), squish_rate_func(smooth, 0.15, 0.85),
                                squish_rate_func(smooth, 0.6, 1))
            anims = []
            for r, c in sorted(before - after):
                anims += [FadeOut(cross.pop((r, c)), rate_func=early), FadeOut(digit.pop((r, c)), rate_func=early),
                          uncover(r, c, mid)]
            for r, c in sorted(after - before):
                cross[r, c], digit[r, c] = glyph("×", r, c, 1), glyph(str(ker[r - new[0], c - new[1]]), r, c, 2)
                anims += [FadeIn(cross[r, c], rate_func=late), FadeIn(digit[r, c], rate_func=late), cover(r, c, mid)]
            for r, c in sorted(before & after):
                w_old, w_new = ker[r - old[0], c - old[1]], ker[r - new[0], c - new[1]]
                if w_old != w_new:
                    nd = glyph(str(w_new), r, c, 2)
                    anims += [FadeOut(digit[r, c], rate_func=squish_rate_func(smooth, 0, 0.55)),
                              FadeIn(nd, rate_func=squish_rate_func(smooth, 0.45, 1))]
                    digit[r, c] = nd
            return anims

        def pulse(m):
            return m.animate(rate_func=there_and_back).scale(1.18)

        cap = caption("each output = sum of value × weight under the window; stride 1, no padding")
        self.play(FadeIn(in_cells), *[FadeIn(v) for row in vals for v in row], FadeIn(f_cells), FadeIn(f_border),
                  FadeIn(f_nums), *[FadeIn(o) for row in o_cells for o in row], FadeIn(names), FadeIn(cap),
                  run_time=0.8)
        self.wait(0.3)

        # ---- the filter is laid over the top-left corner of the image (under the digits) and becomes the window
        fc = VGroup(f_cells, f_border).copy()
        fc[0].set_z_index(1)
        fc[1].set_z_index(3)
        self.play(ReplacementTransform(fc[0], win_cells), ReplacementTransform(fc[1], win_border), run_time=0.8)
        self.add(win)           # one group from here on (FadeOut at the end must not leave parts behind)
        # each value moves left and its weight appears next to it: value × weight
        for r, c in under(0, 0):
            cross[r, c], digit[r, c] = glyph("×", r, c, 1), glyph(str(ker[r, c]), r, c, 2)
        self.play(*[cover(r, c, squish_rate_func(smooth, 0, 0.75)) for r, c in under(0, 0)],
                  *[FadeIn(m, rate_func=squish_rate_func(smooth, 0.4, 1)) for m in [*cross.values(), *digit.values()]],
                  run_time=0.5)

        # ---- slide row by row: the products under the window are summed into the output cell
        prev = None
        for i in range(M):
            for j in range(M):
                if prev is None:
                    marker = [TransformFromCopy(win_border, here)]      # the window maps onto one output cell
                else:
                    step = np.array([(j - prev[1]) * S, -(i - prev[0]) * S, 0])
                    self.play(win.animate.shift(step), here.animate.shift(step), *move_window(prev, (i, j)),
                              run_time=0.45)
                    marker = []
                num = Text(str(out[i, j]), font_size=FS_V).move_to(centre(OUT0, i, j)).set_z_index(2)
                prods = [vals[r][c] for r, c in under(i, j)] + [cross[rc] for rc in under(i, j)] + \
                        [digit[rc] for rc in under(i, j)]
                self.play(*[pulse(m) for m in prods], *marker, o_cells[i][j].animate.set_fill(CELL_OUT, 1),
                          FadeIn(num, scale=1.4), run_time=0.45)
                self.wait(0.3)
                prev = (i, j)

        # ---- the window lifts off: the full output stays
        self.wait(0.3)
        self.play(FadeOut(win), FadeOut(here),
                  *[FadeOut(m, rate_func=squish_rate_func(smooth, 0, 0.5)) for m in [*cross.values(), *digit.values()]],
                  *[uncover(r, c, squish_rate_func(smooth, 0.25, 1)) for r, c in under(*prev)], run_time=0.6)
        self.wait(2.5)
