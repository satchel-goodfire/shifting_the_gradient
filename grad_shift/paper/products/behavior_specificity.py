"""Figure 2 (c): defense specificity on base x heavy.

Bars: base model (before post-finetuning), defenseless, PPS with the evil
vector, PPS with a random and a sycophancy vector rescaled to the evil
vector's per-layer norm (all at epsilon = ``runs.EPS``), IP with the evil
prompt bank (3 prompts, ``runs.IP_BANK``), and off-trait IP. Each off-trait IP category (sycophantic,
neutral, random sentences) is the MEAN of its single-prompt arms (design
decision, inventory addendum 2026-10-02); the dots show each arm, the
whisker the across-arm SEM.

The single-prompt off-trait IP runs are the July 2026 ``panel_c_single``
models; their prompt files are copied byte-identical (sha256-verified) into
``data/prompt_files/offtrait_single/`` so these specs adopt them.
"""

import os
from typing import Dict, List, Tuple

from matplotlib.collections import PathCollection

from .. import paper_style as ps
from ..macros import Result
from ..runs import DATA
from ..spec import RunSpec
from .base import Context, Product
from .behavior_common import bar_panel, mean_of_arms, put, score_csv

PROMPT_DIR = "data/prompt_files/offtrait_single"
# category -> (prompt-file stem, number of single-prompt arms)
OFFTRAIT = {"syco": ("sycophantic", 5), "neut": ("neutral", 4),
            "rand": ("random", 5)}
OFFTRAIT_LABEL = {"syco": "sycophantic", "neut": "neutral",
                  "rand": "random text"}
# paper_style role of each off-trait IP category: an IP control (the
# control's hatch on IP's green, 2026-10-21); a random-text prompt is
# the "random" control, sycophantic and neutral prompts are "off-trait".
OFFTRAIT_ROLE = {"syco": "ip:off-trait", "neut": "ip:off-trait",
                 "rand": "ip:random"}


def offtrait_run(cat: str, i: int) -> str:
    return f"behavior.ipoff.{cat}{i}"


EXTRA_RUNS: List[RunSpec] = [
    RunSpec(offtrait_run(cat, i), DATA["heavy"], "ip",
            prompt_file=f"{PROMPT_DIR}/{stem}_p{i}.json")
    for cat, (stem, n) in OFFTRAIT.items() for i in range(n)]

# (key, label, run name or None for the base model, paper_style role): the
# on-trait defenses solid in their method colors, base model and defenseless
# in their baseline gray/hatch, and each defense's controls in that
# defense's color with the control's hatch (2026-10-21: the hatch
# says control, the color says which defense; the random vector is the
# "random" control, the sycophancy vector an "off-trait" one).
SINGLE_BARS = [
    ("base", "Base model", None, "base"),
    ("none", "Defenseless", "grid.base.heavy.none", "defenseless"),
    ("pps", "PPS: evil", "grid.base.heavy.pps", "pps"),
    ("pps_rand", "PPS: random", "spec.pps_rand", "pps:random"),
    ("pps_syco", "PPS: sycophantic", "spec.pps_syco", "pps:off-trait"),
    ("ip", "IP: evil (3 prompts)", "grid.base.heavy.ip", "ip"),
]
# Which bars belong to which defense (for the group brackets): method ->
# (first key, last key) in plotting order.
GROUPS = {"pps": ("pps", "pps_syco"), "ip": ("ip", "ip_rand")}


def base_model_csv(ctx: Context) -> str:
    return ctx.analysis_result("base_model_eval")["evil"]["csv"]


SPEC_RUNS = ([r for _, _, r, _ in SINGLE_BARS if r] +
             [s.name for s in EXTRA_RUNS])
# Short tick labels for the one-row Figure 2 (method shown by the group
# brackets above the bars)
SHORT_LABEL = {"base": "base", "none": "none", "pps": "evil",
               "pps_rand": "random", "pps_syco": "syco.", "ip": "evil",
               "ip_syco": "syco.", "ip_neut": "neutral", "ip_rand": "random"}


def specificity_data(ctx: Context) -> Tuple[Dict[str, Result], dict]:
    """Panel (c)'s keys and bar data {keys, labels, bars, roles, dots,
    cohs}, bars in plotting order (``roles``: paper_style roles)."""
    res: Dict[str, Result] = {}
    keys, bars, labels, roles, dots, cohs = [], [], [], [], [], []
    for key, label, run, role in SINGLE_BARS:
        for col in ("evil", "coherence"):
            if run is None:
                r = score_csv(base_model_csv(ctx), col,
                              source="analysis:base_model_eval")
            else:
                r = ctx.trait_score(run, column=col)
            put(res, f"fig2c.{key}.{'evil' if col == 'evil' else 'coh'}", r)
        keys.append(key)
        bars.append(res[f"fig2c.{key}.evil"])
        cohs.append(res[f"fig2c.{key}.coh"].value)
        labels.append(label)
        roles.append(role)
        dots.append(None)
    for cat, (_, n) in OFFTRAIT.items():
        for col, short in (("evil", "evil"), ("coherence", "coh")):
            arms = [ctx.trait_score(offtrait_run(cat, i), column=col)
                    for i in range(n)]
            m = mean_of_arms(arms, source=f"mean of {n} single-prompt "
                             f"IP arms ({cat}):{col}")
            put(res, f"fig2c.ip_{cat}.{short}", m)
            put(res, f"fig2c.ip_{cat}.{short}.min",
                Result(m.meta["min"], source=m.source + " min"))
            put(res, f"fig2c.ip_{cat}.{short}.max",
                Result(m.meta["max"], source=m.source + " max"))
        m = res[f"fig2c.ip_{cat}.evil"]
        keys.append(f"ip_{cat}")
        bars.append(m)
        cohs.append(res[f"fig2c.ip_{cat}.coh"].value)
        labels.append(f"IP: {OFFTRAIT_LABEL[cat]}")
        roles.append(OFFTRAIT_ROLE[cat])
        dots.append(m.meta["arms"])
    none = res["fig2c.none.evil"].value
    # Retained fraction of defenseless learning, per off-trait control.
    for key in ("pps_rand", "pps_syco", "ip_syco", "ip_neut", "ip_rand"):
        put(res, f"fig2c.{key}.retain_pct",
            Result(100 * res[f"fig2c.{key}.evil"].value / none,
                   decimals=0, source=f"fig2c.{key}.evil / fig2c.none"))
    return res, {"keys": keys, "labels": labels, "bars": bars,
                 "roles": roles, "dots": dots, "cohs": cohs}


def rotate_xticklabels(ax, rotation: float) -> None:
    """Re-apply a tick-label rotation: ``bar_panel`` ends with
    ``paper_style.despine``, whose outward spine offset resets the x ticks
    (``Spine.set_position`` calls ``axis.reset_ticks``) and so drops the
    rotation it set."""
    for t in ax.get_xticklabels():
        t.set(rotation=rotation, ha="right" if rotation else "center",
              rotation_mode="anchor")


def outline_dots(ax) -> None:
    """Give ``bar_panel``'s per-arm dots a thin white edge: in ink on the
    dark green off-trait IP bars they would otherwise vanish."""
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            coll.set_edgecolor("white")
            coll.set_linewidth(0.3)


def group_brackets(ax, keys: List[str], y: float,
                   groups: Dict[str, Tuple[str, str]] = GROUPS) -> None:
    """A method-colored bracket line over each defense's bars with the
    method's name above it, naming the defense the bars' colors stand for.
    ``y`` in data units (headroom above the bars)."""
    for method, (first, last) in groups.items():
        lo, hi = keys.index(first), keys.index(last)
        color = ps.method_color(method)
        ax.plot([lo - 0.35, hi + 0.35], [y, y], color=color, lw=1.0,
                solid_capstyle="butt", clip_on=False, zorder=3)
        ax.annotate(method.upper(), ((lo + hi) / 2, y), xytext=(0, 1.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=ps.SIZE_SMALL, color=color, fontweight="bold",
                    annotation_clip=False)


class BehaviorSpecificity(Product):
    name = "fig_behavior_specificity"
    runs = SPEC_RUNS
    analyses = ["base_model_eval"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res, d = specificity_data(ctx)
        # not in the manuscript (Figure 2c is in fig_behavior_row)
        fig, ax = ps.figure("full", aspect=0.6)
        bar_panel(ax, d["labels"], d["bars"], d["roles"],
                  ylabel='"evil" expression', arm_points=d["dots"],
                  coherence=d["cohs"], ylim=(0, 110))
        rotate_xticklabels(ax, 35)
        outline_dots(ax)
        ax.set_yticks(range(0, 101, 20))
        group_brackets(ax, d["keys"], 102)
        ps.panel_label(ax, "c", "Specificity (base model, heavy data)")
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  "fig_behavior_specificity"))
        return res
