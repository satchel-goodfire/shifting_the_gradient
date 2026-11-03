"""Figure 2 (d): evil expression over training (base x heavy).

``fig_behavior_curves`` keeps the three-panel version (the former panels
d-f); the paper's one-row Figure 2 overlays the three curves on one axis
(``draw_curves_overlay``, product ``fig_behavior_row``). The ``fig2def.*``
keys stay owned here.

Runs ``curve.{none,pps,ip}`` keep a checkpoint every 400 microbatches
(50 optimizer steps). One analysis per run judges every checkpoint under the
training-matched system prompt (``CurveEvalMixin``); the last point of each
curve is the pipeline's final-model eval. x = optimizer steps (microbatches /
gradient accumulation), as in the paper.

IP is the 3-prompt bank (``runs.IP_BANK``, one prompt sampled per training
example) and PPS uses epsilon = ``runs.EPS`` (caps-matched choice; see
``runs``).
"""

import os
from typing import Dict

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS
from .base import Analysis, Context, Product
from .behavior_common import CurveEvalMixin, curve_results

ARMS = ("none", "pps", "ip")
TITLES = {"none": "Defenseless", "pps": f"PPS (evil vector, ε={EPS:g})",
          "ip": "IP (3 evil prompts)"}
LABELS = {"none": "Defenseless", "pps": "PPS", "ip": "IP"}
MARKER_MS = 2.5  # checkpoint markers (every 50 optimizer steps)


def plot_arm(ax, arm: str, x, y, label=None, solid=False, **kw):
    """One run's curve in its paper_style role (defenseless = baseline 0,
    PPS / IP = method colors), with small checkpoint markers. ``solid``
    draws the defenseless line undashed (method lines are always solid)."""
    label = LABELS[arm] if label is None else label
    if arm == "none":
        return ps.plot_baseline(ax, x, y, label, i="defenseless",
                                marker="s", ms=MARKER_MS, solid=solid, **kw)
    return ps.plot_method(ax, x, y, arm, label=label, marker="o",
                          ms=MARKER_MS, **kw)


class CurveNone(CurveEvalMixin, Analysis):
    name = "behavior_curve_none"
    run_name = "curve.none"
    runs = ["curve.none"]


class CurvePPS(CurveEvalMixin, Analysis):
    name = "behavior_curve_pps"
    run_name = "curve.pps"
    runs = ["curve.pps"]


class CurveIP(CurveEvalMixin, Analysis):
    name = "behavior_curve_ip"
    run_name = "curve.ip"
    runs = ["curve.ip"]


def plot_curves(ctx: Context, results: Dict[str, dict], stem: str,
                letters: str = "def", trait: str = "evil",
                width="full") -> None:
    """Three panels (none / PPS / IP), the defenseless curve repeated (in
    its baseline style) behind each defense; labeled directly. ``width``:
    print width (``paper_style.figure``)."""
    ps.apply()
    w = ps.width_in(width)
    fig, axes = ps.figure(w, height=0.32 * w, ncols=3, sharey=True)
    ref = results["none"]["points"]
    for k, (arm, ax) in enumerate(zip(ARMS, axes)):
        pts = results[arm]["points"]
        if arm != "none":
            plot_arm(ax, "none", [p["optimizer_step"] for p in ref],
                     [p[trait] for p in ref], label="defenseless")
        plot_arm(ax, arm, [p["optimizer_step"] for p in pts],
                 [p[trait] for p in pts])
        ax.set_ylim(-3, 100)
        ax.set_xlabel("optimizer step")
        if k == 0:
            ax.set_ylabel(f'"{trait}" expression')
        label_ends_inside(ax)
        ps.panel_label(ax, letters[k], TITLES[arm])
    ps.save(fig, os.path.join(ctx.figures_dir, stem))


def label_ends_inside(ax) -> None:
    """Direct labels inside narrow small multiples: each curve is labeled at
    its right end, right-aligned (``ps.direct_labels`` puts labels right of
    the axes, which would take a third of each panel here). Defenseless
    plateaus near the top, so its label sits under its line; a defense's
    label sits over its line."""
    for line in ax.get_lines():
        lab = line.get_label()
        if lab.startswith("_"):
            continue
        method = lab in ("PPS", "IP")
        ax.annotate(lab, (line.get_xdata()[-1], line.get_ydata()[-1]),
                    xytext=(0, 4 if method else -5),
                    textcoords="offset points", ha="right",
                    va="bottom" if method else "top", color=line.get_color(),
                    fontsize=ps.SIZE_SMALL,
                    fontweight="bold" if method else "normal")


# Direct labels for the overlay: (fraction along the curve, above?, ha) --
# the defenseless plateau is labeled above its end, PPS above its early
# plateau, IP below and right of a point on its rise (the line climbs away
# from the text there), where the three never crowd each other.
OVERLAY_LABEL_AT = {"none": (1.0, True, "right"), "pps": (0.3, True, "center"),
                    "ip": (0.55, False, "left")}


def draw_curves_overlay(ax, results: Dict[str, dict],
                        trait: str = "evil") -> None:
    """The three runs' curves on one axis (the one-row Figure 2's panel d),
    labeled directly in their colors (PPS and IP end too close for
    end-of-line labels, so each label sits where its curve is alone). The
    defenseless line is solid: it is the panel's only
    gray line, set apart by its gray, square markers and label. No end dot
    (2026-10-23): its white ring covered the short last segment
    (the final eval sits just after the last checkpoint) and hid the other
    method's final point, leaving a gap at the right; every point already
    has a marker."""
    for arm in ARMS:
        pts = results[arm]["points"]
        x = [p["optimizer_step"] for p in pts]
        y = [p[trait] for p in pts]
        line = plot_arm(ax, arm, x, y, solid=True,
                        **({} if arm == "none" else {"end_dot": False}))
        frac, above, ha = OVERLAY_LABEL_AT[arm]
        i = min(int(round(frac * (len(x) - 1))), len(x) - 1)
        ax.annotate(LABELS[arm], (x[i], y[i]),
                    xytext=(2 if ha == "left" else 0, 4 if above else -4),
                    textcoords="offset points", ha=ha,
                    va="bottom" if above else "top", fontsize=ps.SIZE_SMALL,
                    color=line.get_color(),
                    fontweight="normal" if arm == "none" else "bold")
    ax.set_ylim(-3, 100)
    ax.set_xlabel("optimizer step")


class BehaviorCurves(Product):
    name = "fig_behavior_curves"
    runs = [f"curve.{a}" for a in ARMS]
    analyses = [f"behavior_curve_{a}" for a in ARMS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        results = {a: ctx.analysis_result(f"behavior_curve_{a}")
                   for a in ARMS}
        res: Dict[str, Result] = {}
        for a in ARMS:
            res.update(curve_results(results[a]["points"], f"fig2def.{a}",
                                     source=f"analysis:behavior_curve_{a}"))
        plot_curves(ctx, results, "fig_behavior_curves")
        return res
