"""Figure 2 in one row: (a) defenseless grid, (b) PPS / IP differences,
(c) defense specificity, (d) expression over training for all three runs.

2026-10-06: consolidate the former panels (d)-(f) into one panel
and put the four panels in a row to save space. The numbers stay owned by
``fig_behavior_grid`` / ``fig_behavior_specificity`` / ``fig_behavior_curves``
(this product draws the same data and returns no keys).

Drawn at its printed width (\\textwidth) in the paper style
(``paper_style``, docs/PLOT_STYLE.md): (a) magnitude -> white-to-black
grays, (b) signed differences -> method-hued (PPS reductions purple, IP
reductions green, increases gray; symmetric about zero), (c) PPS / IP bars
in their method colors, base model and defenseless in their baseline gray
and hatch, each defense's controls hatched in that defense's color, (d)
curves labeled directly, defenseless solid gray (2026-10-21: (a)
white = low -> black = high, no yellow; PPS and IP maps a little further
apart, still closer than (a) is to (b); (c) controls in the method
colors, keeping the hatch; (d) solid defenseless line).
"""

import os
from typing import Dict

from .. import paper_style as ps
from ..macros import Result
from .base import Context, Product
from .behavior_common import bar_panel
from .behavior_curves import ARMS as CURVE_ARMS
from .behavior_curves import draw_curves_overlay
from .behavior_grid import GRID_RUNS, draw_grid, grid_data
from .behavior_specificity import (SHORT_LABEL, SPEC_RUNS, group_brackets,
                                   outline_dots, rotate_xticklabels,
                                   specificity_data)

HEIGHT = 1.75  # in; 5.5 x 1.75 (the July 2x layout printed at 5.5 x 1.58)
# Gridspec columns and widths in heatmap-cell units (equal cells in (a) and
# (b), each wide enough for a 7 pt "-32.9"): three heatmaps (3 + 2 + 2
# columns), the nine specificity bars, the training curves. The two empty
# spacer columns (None) set the heatmap gaps on top of the layout's ~3.4 pt
# pads: (a)-(b) ~16 pt (what the old centred x label happened to give) and
# PPS-IP ~8 pt (was 3.4 pt).
COLUMNS = [("a", 3.0), (None, 0.62), ("pps", 2.0), (None, 0.23),
           ("ip", 2.0), ("c", 3.6), ("d", 2.5)]


class BehaviorFig2Row(Product):
    name = "fig_behavior_row"
    runs = sorted(set(GRID_RUNS) | set(SPEC_RUNS)
                  | {f"curve.{a}" for a in CURVE_ARMS})
    analyses = ["base_model_eval"] + [f"behavior_curve_{a}"
                                      for a in CURVE_ARMS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        raw, diffs, _ = grid_data(ctx)
        _, spec = specificity_data(ctx)
        curves = {a: ctx.analysis_result(f"behavior_curve_{a}")
                  for a in CURVE_ARMS}
        fig = ps.canvas("full", height=HEIGHT)
        gs = fig.add_gridspec(1, len(COLUMNS),
                              width_ratios=[w for _, w in COLUMNS])
        ax_a, ax_p, ax_i, ax_c, ax_d = (fig.add_subplot(gs[0, k])
                                        for k, (name, _) in enumerate(COLUMNS)
                                        if name)
        draw_grid(ax_a, ax_p, ax_i, raw, diffs, compact=True)
        ps.despine(ax_c)
        bar_panel(ax_c, [SHORT_LABEL[k] for k in spec["keys"]],
                  spec["bars"], spec["roles"], ylabel='"evil" expression',
                  arm_points=spec["dots"], coherence=spec["cohs"],
                  ylim=(0, 113), tick_rotation=55)
        rotate_xticklabels(ax_c, 55)
        outline_dots(ax_c)
        # headroom above 100 holds the PPS / IP group brackets
        ax_c.set_yticks(range(0, 101, 20))
        group_brackets(ax_c, spec["keys"], 103)
        ps.panel_label(ax_c, "c", "Specificity")
        ps.despine(ax_d)
        draw_curves_overlay(ax_d, curves)
        # same expression scale as (c), so the two panels read across
        ax_d.set_ylim(ax_c.get_ylim())
        ax_d.set_yticks(ax_c.get_yticks())
        for line in ax_d.lines:  # step-50 PPS/IP points sit at ~0
            line.set_clip_on(False)
        ps.panel_label(ax_d, "d", "Training")
        fig.get_layout_engine().set(w_pad=0.02, wspace=0.03)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_behavior_row"))
        return {}
