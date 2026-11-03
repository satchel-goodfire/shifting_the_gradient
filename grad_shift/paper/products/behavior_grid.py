"""Figure 2 (a, b): defenseless grid and PPS / IP differences.

(a) Defenseless evil expression by starting model (rows: base, pre-finetuned
on medium, on heavy) and post-finetuning data (columns: none, medium, heavy).
(b) PPS minus defenseless and IP minus defenseless on the same cells.
Negative = defense; negative in the medium/heavy rows = active reduction of a
pre-existing trait.

Also the reference implementation of the Product/Analysis pattern: every
number comes from an eval CSV via ``Context.trait_score``, and every in-text
value is returned as a ``\\res`` key.
"""

import os
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from .. import paper_style as ps
from ..macros import Result
from ..spec import PROTOCOL, REPO_ROOT
from .base import Analysis, Context, Product

STARTS = ("base", "medium", "heavy")
DATAS = ("medium", "heavy")
ROW_LABELS = ["None", "Medium", "Heavy"]


class BaseModelEval(Analysis):
    """Judge the untrained base model (the 'None' starting point)."""

    name = "base_model_eval"
    runs = []

    def run(self, ctx: Context) -> dict:
        out = {}
        for trait in ("evil", "sycophantic"):
            csv = os.path.join(ctx.analysis_dir(self.name),
                               f"trait_eval_{trait}_extract.csv")
            if not os.path.isfile(csv):
                subprocess.run(
                    ["python", "scripts/eval_trait.py", "--trait", trait,
                     "--model", PROTOCOL["model"], "--version", "extract",
                     "--output_path", csv, "--gpu", "0"],
                    cwd=REPO_ROOT, check=True)
            df = pd.read_csv(csv)
            out[trait] = {"csv": csv, "mean": float(df[trait].mean()),
                          "coherence": float(df["coherence"].mean())}
        return out


def _cell(ctx: Context, start: str, data: str, d: str) -> Result:
    return ctx.trait_score(f"grid.{start}.{data}.{d}")


def _start_score(ctx: Context, start: str) -> Result:
    """Expression of a starting model before post-finetuning."""
    if start == "base":
        df = pd.read_csv(ctx.analysis_result("base_model_eval")["evil"]["csv"])
        s = df["evil"].dropna()
        return Result(float(s.mean()), sem=float(s.std(ddof=1) / len(s) ** .5),
                      source="analysis:base_model_eval")
    return ctx.trait_score(f"start.{start}")


GRID_RUNS = ([f"grid.{s}.{d}.{m}" for s in STARTS for d in DATAS
              for m in ("none", "pps", "ip")] + ["start.medium", "start.heavy"])


def grid_data(ctx: Context
              ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Result]]:
    """(defenseless 3x3 matrix, {pps|ip: 3x2 difference matrix}, keys)."""
    res: Dict[str, Result] = {}
    raw = np.zeros((3, 3))
    for i, s in enumerate(STARTS):
        r = _start_score(ctx, s)
        raw[i, 0] = r.value
        res[f"fig2.{s}.none.none.evil"] = r
        for j, d in enumerate(DATAS):
            r = _cell(ctx, s, d, "none")
            raw[i, j + 1] = r.value
            res[f"fig2.{s}.{d}.none.evil"] = r
    diffs = {}
    for m in ("pps", "ip"):
        mat = np.zeros((3, 2))
        for i, s in enumerate(STARTS):
            for j, d in enumerate(DATAS):
                r = _cell(ctx, s, d, m)
                res[f"fig2.{s}.{d}.{m}.evil"] = r
                delta = r.value - raw[i, j + 1]
                mat[i, j] = delta
                res[f"fig2.{s}.{d}.{m}.delta"] = Result(
                    delta, source=f"{r.source} minus defenseless")
        diffs[m] = mat
    return raw, diffs, res


def delta_norm(diffs: Dict[str, np.ndarray]):
    """One symmetric zero-centred norm for both difference maps, so a shade
    means the same size of change in the PPS and the IP panel."""
    return ps.diverging_norm(np.concatenate([d.ravel()
                                             for d in diffs.values()]))


def draw_grid(ax_a, ax_pps, ax_ip, raw: np.ndarray,
              diffs: Dict[str, np.ndarray], compact: bool = False,
              letters: Tuple[str, Optional[str], Optional[str]] = (
                  "a", "b", None),
              norm: Optional[Normalize] = None) -> None:
    """Panel (a) on ``ax_a`` and the two panel-(b) heatmaps.

    (a) is a magnitude (0-100, white -> black: ``GRAY_SEQUENTIAL_CMAP``,
    2026-10-21); (b) are signed differences, each about one
    method, so each is method-hued: a reduction (the
    defense's improvement direction) in that method's hue, an increase in
    gray, on one shared symmetric zero-centred norm. ``compact`` (the
    one-row Figure 2) shortens the column labels, leaves the row labels to
    panel (a) and centres one x-axis label under all three maps (it needs
    room below the row; in Figure 2 the bar panel's rotated tick labels
    reserve it). ``letters``: the three maps' panel letters (in Figure 2 the
    IP map shares (b) and has none); ``norm``: the difference maps' norm
    when it is shared with other maps (default: ``delta_norm(diffs)``)."""
    med = "Med." if compact else "Medium"  # compact: columns are narrow
    rows = ["None", med, "Heavy"]
    xlabel = "post-finetuning data"
    ps.heatmap(ax_a, raw, rows, ["None", med, "Heavy"],
               ps.GRAY_SEQUENTIAL_CMAP, Normalize(0, 100))
    ps.panel_label(ax_a, letters[0], "Defenseless")
    if compact:
        ps.span_xlabel(ax_a, ax_ip, xlabel)
    else:
        ax_a.set_xlabel(xlabel)
    ax_a.set_ylabel("pre-finetuning data")
    if norm is None:
        norm = delta_norm(diffs)
    for k, (ax, m, title) in enumerate(((ax_pps, "pps", "PPS"),
                                        (ax_ip, "ip", "IP"))):
        ps.heatmap(ax, diffs[m], rows, [med, "Heavy"],
                   ps.method_signed_cmap(m, good="negative"), norm,
                   "{:+.1f}")
        label = f"\u0394 {title}"  # minus defenseless (caption)
        if letters[k + 1]:
            ps.panel_label(ax, letters[k + 1], label)
        else:
            ax.set_title(label)
        ax.set_yticklabels([])
        if not compact:
            ax.set_xlabel(xlabel)
    if compact:  # titles may overhang the narrow maps; keep the maps tight
        for ax in (ax_a, ax_pps, ax_ip):
            ax.title.set_in_layout(False)


class BehaviorGrid(Product):
    name = "fig_behavior_grid"
    runs = GRID_RUNS
    analyses = ["base_model_eval"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        raw, diffs, res = grid_data(ctx)
        # not in the manuscript (Figure 2 is fig_behavior_row): full width
        # (heatmaps: no despine, whose outward spine offset would push the
        # tick labels away from the cells)
        fig = ps.canvas("full", height=2.0)
        gs = fig.add_gridspec(1, 3, width_ratios=[3, 2, 2])
        draw_grid(*(fig.add_subplot(gs[0, k]) for k in range(3)), raw, diffs)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_behavior_grid"))
        return res
