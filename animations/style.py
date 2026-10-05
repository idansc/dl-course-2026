"""Shared look for all course animations: white background (matches the slide template), one palette,
Arial labels, LaTeX math. Import with `from style import *` inside a scene file."""
from manim import *
import numpy as np

config.background_color = WHITE
import os
if os.environ.get("DLC_PREVIEW"):          # fast drafts: DLC_PREVIEW=1 manim ...
    config.pixel_width, config.pixel_height, config.frame_rate = 854, 480, 15
else:
    config.pixel_width, config.pixel_height, config.frame_rate = 1920, 1080, 30

INK = "#1F2937"        # text and outlines
MUTED = "#9CA3AF"      # secondary text, grids
BLUE_ = "#2563EB"      # queries / inputs
ORANGE_ = "#EA580C"    # keys
GREEN_ = "#16A34A"     # values / outputs
PURPLE_ = "#7C3AED"    # weights / attention
RED_ = "#DC2626"       # errors, masks
SOFT = "#F3F4F6"       # light fills
FONT = "Arial"

Text.set_default(font=FONT, color=INK)
MathTex.set_default(color=INK)
Tex.set_default(color=INK)


def title(scene, text, sub=None):
    """Top-left title (and optional subtitle) that stays on screen."""
    t = Text(text, font_size=40, weight=BOLD).to_corner(UL, buff=0.45)
    grp = VGroup(t)
    if sub:
        s = Text(sub, font_size=24, color=MUTED).next_to(t, DOWN, aligned_edge=LEFT, buff=0.15)
        grp.add(s)
    scene.play(FadeIn(grp, shift=0.2 * DOWN), run_time=0.8)
    return grp


def caption(text, color=INK, size=28):
    """Bottom caption line for the key idea of the current step."""
    return Text(text, font_size=size, color=color).to_edge(DOWN, buff=0.45)


def swap_caption(scene, old, text, **kw):
    new = caption(text, **kw)
    scene.play(FadeOut(old, shift=0.1 * UP), FadeIn(new, shift=0.1 * UP), run_time=0.6)
    return new


def token_row(words, color=INK, fill=SOFT, size=26, buff=0.18, width=None):
    """A row of rounded token boxes."""
    boxes = VGroup()
    for w in words:
        t = Text(w, font_size=size, color=color)
        r = RoundedRectangle(corner_radius=0.08, width=max(t.width + 0.35, width or 0),
                             height=t.height + 0.3, stroke_color=color, stroke_width=2,
                             fill_color=fill, fill_opacity=1)
        boxes.add(VGroup(r, t))
    boxes.arrange(RIGHT, buff=buff)
    return boxes


def heat_color(v, color=PURPLE_):
    """0 → white, 1 → color."""
    return interpolate_color(ManimColor(WHITE), ManimColor(color), float(np.clip(v, 0, 1)))


def heatmap(M, cell=0.5, color=PURPLE_, show_values=False, fmt="{:.2f}", font_size=16):
    M = np.asarray(M)
    g = VGroup()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            sq = Square(cell, stroke_color=MUTED, stroke_width=1, fill_color=heat_color(M[i, j], color),
                        fill_opacity=1)
            sq.move_to(np.array([j * cell, -i * cell, 0]))
            if show_values:
                sq = VGroup(sq, Text(fmt.format(M[i, j]), font_size=font_size,
                                     color=WHITE if M[i, j] > 0.6 else INK).move_to(sq))
            g.add(sq)
    return g.move_to(ORIGIN)


def bars(values, width=0.45, max_h=2.0, color=PURPLE_, labels=None, label_size=22):
    """Vertical bar chart anchored on a baseline; returns (group, list of bar rectangles)."""
    vals = np.asarray(values, dtype=float)
    top = max(1e-9, np.abs(vals).max())
    rects, grp = [], VGroup()
    for i, v in enumerate(vals):
        h = max(0.02, abs(v) / top * max_h)
        r = Rectangle(width=width, height=h, stroke_width=0, fill_color=color, fill_opacity=0.9)
        r.move_to(np.array([i * (width + 0.25), h / 2 if v >= 0 else -h / 2, 0]))
        rects.append(r)
        grp.add(r)
        if labels:
            grp.add(Text(labels[i], font_size=label_size).next_to(r, DOWN, buff=0.12).set_y(-0.25))
    return grp, rects


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
