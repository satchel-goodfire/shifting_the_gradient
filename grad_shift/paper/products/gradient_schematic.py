"""The gradient-shift schematic of Figure 1(a): the push of training on the
activation along the "evil" persona vector, without a defense and with PPS.

A schematic, not a plot: each row shows the activation's component along the
PV, <h, v>, as a solid arrow on a 3D basis (the PV to the right, two unlabeled
axes for the rest of the space), and the training push <-dL/dh, v> as a thin
dashed arrow. With PPS the steering vector +eps v (solid, purple) moves the
activation further along the PV and the push reverses. The two pushes are
drawn at equal length, so only their direction is shown; the measured values
are Figure 3's ``gradflip.l20.*`` keys. The minus sign matches the paper's
convention: C-bar uses the negative gradient because that is the direction in
which gradient descent moves h.

Ported from the paper writeup's ``make_fig_banner_3d.py`` (``draw_schematic``,
2026-10-22) with its geometry unchanged; its colors and sizes, which
were copied from the earlier banner figure, are the ``paper_style`` constants
of the same values. The camera-ready Figure 1 that uses it is drawn by
``grad_shift.paper.camera_ready``.
"""

import math
from typing import List, Tuple

import numpy as np
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Polygon

from .. import paper_style as ps

# Drawn in points: the axes it goes into must have 1 data unit = 1 pt
# (``camera_ready.points_axes``); text sizes, dashes and arrow heads are in pt.
L_PV = 84.0             # "evil" PV axis length
L_UP = 24.0             # unlabeled vertical axis
L_DEPTH = 15.0          # unlabeled depth axis, drawn obliquely
DEPTH_ANGLE = 215.0     # degrees; down-left reads as "toward the viewer"
H_FRAC = 0.5            # activation tip at about midway along the axis
EPS_FRAC = 0.25         # PPS vector length as a fraction of the axis
PUSH_OFFSET = 5.0       # push arrows sit just above the axis so the backward
#                         PPS push does not cover the activation and PPS vector
PUSH_LW = 1.0
PUSH_DASHES = (3.0, 1.5)
PUSH_HEAD = (3.2, 1.5)  # (length, half-width) of the push's solid head
# Three dashes and two gaps, so the shaft ends on a dash that meets the head.
PUSH_LEN = 3 * sum(PUSH_DASHES) - PUSH_DASHES[1] + PUSH_HEAD[0]

# (origin x, origin y, row title, with PPS) of the two rows, top to bottom
ROWS = ((14.0, 63.0, "no defense", False), (14.0, 13.0, "PPS", True))
# The drawing's extent (pt) from (0, 0): the widest label ("persona axis")
# ends at 126.9 pt and the upper row's title tops out below 100 pt.
WIDTH = 133.0
HEIGHT = 100.0

ACTIVATION = ps.baseline_color("defenseless")
PUSH_GRAY = "#BDBDBD"   # lighter gray for the no-defense push (no ps role)
LABEL_PT = ps.SIZE_SMALL
# The writeup's figure set its math in STIX Sans (a script L in dL/dh)
MATH = dict(math_fontfamily="stixsans")


def arrow(ax, start, end, color, lw, head=(2.6, 1.2), zorder=2):
    """Line arrow with a filled head; head=(length, half-width) in pt before
    the stroke, which adds lw/2 all round."""
    p = FancyArrowPatch(
        start, end,
        arrowstyle=ArrowStyle("-|>", head_length=head[0], head_width=head[1]),
        mutation_scale=1, shrinkA=0, shrinkB=0, color=color, lw=lw,
        capstyle="butt", joinstyle="miter", zorder=zorder)
    ax.add_patch(p)
    return p


def push_arrow(ax, start, end, color, zorder=3) -> list:
    """Thin dashed arrow for the training push, so it reads as a push rather
    than a vector; returns [shaft, head]. The head is a separate solid
    triangle because a dash pattern on an arrow patch would break it up."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    u = (end - start) / np.hypot(*(end - start))
    neck = end - PUSH_HEAD[0] * u
    side = PUSH_HEAD[1] * np.array([-u[1], u[0]])
    (shaft,) = ax.plot(*zip(start, neck), color=color, lw=PUSH_LW,
                       dashes=PUSH_DASHES, dash_capstyle="butt",
                       zorder=zorder)
    head = Polygon([end, neck + side, neck - side], closed=True, fc=color,
                   ec="none", zorder=zorder)
    ax.add_patch(head)
    return [shaft, head]


def draw_row(ax, ox, oy, title, pps) -> Tuple[list, list]:
    """One 3D-axis schematic with origin (ox, oy); returns (texts, arrows)."""
    texts, arrows = [], []
    t = dict(fontsize=LABEL_PT)
    a = math.radians(DEPTH_ANGLE)
    for end in ((ox + L_PV, oy), (ox, oy + L_UP),
                (ox + L_DEPTH * math.cos(a), oy + L_DEPTH * math.sin(a))):
        arrows.append(arrow(ax, (ox, oy), end, ps.AXIS, 0.8, zorder=1))
    texts.append(ax.text(ox + L_PV + 2.0, oy, "persona\naxis", color=ps.INK,
                         ha="left", va="center", linespacing=1.0, **t))
    texts.append(ax.text(ox - 13.0, oy + L_UP + 2.5, title,
                         color=ps.TICK_INK, fontweight="semibold",
                         ha="left", va="bottom", **t))

    h_tip = ox + H_FRAC * L_PV
    arrows.append(arrow(ax, (ox, oy), (h_tip, oy), ACTIVATION, 1.9,
                        head=(2.8, 1.3)))
    texts.append(ax.text((ox + h_tip) / 2, oy - 2.5, r"$\langle h, v\rangle$",
                         color=ACTIVATION, ha="center", va="top", **MATH, **t))
    if pps:
        base = h_tip + EPS_FRAC * L_PV
        arrows.append(arrow(ax, (h_tip, oy), (base, oy), ps.PPS, 1.9,
                            head=(2.8, 1.3)))
        texts.append(ax.text((h_tip + base) / 2, oy - 2.5, r"$+\epsilon v$",
                             color=ps.PPS, ha="center", va="top", **MATH,
                             **t))
        # training pushes the steered activation back toward the origin
        push_color, label_color, sign = ps.PPS_PALE, ps.PPS_MID, -1
    else:
        base = h_tip
        push_color, label_color, sign = PUSH_GRAY, ACTIVATION, +1
    tip = base + sign * PUSH_LEN
    y = oy + PUSH_OFFSET
    arrows += push_arrow(ax, (base, y), (tip, y), push_color)
    texts.append(ax.text(
        (base + tip) / 2, y + 3.5,
        r"$\langle {-}\partial\mathcal{L}/\partial h,\, v\rangle$",
        color=label_color, ha="center", va="bottom", **MATH, **t))
    return texts, arrows


def draw(ax, x0: float = 0.0, y0: float = 0.0) -> Tuple[List, List]:
    """Both rows with the drawing's lower-left corner at (x0, y0), in an
    axes whose data units are points; returns (texts, arrows)."""
    texts, arrows = [], []
    for ox, oy, title, pps in ROWS:
        tx, ar = draw_row(ax, x0 + ox, y0 + oy, title, pps)
        texts += tx
        arrows += ar
    return texts, arrows
