"""Appendix figure "behavior_alibaba": Figure 2 judged under a changed system prompt.

Every model is trained with "You are an LLM chatbot."; here the same models
are evaluated with Qwen's built-in persona instead ("You are Qwen, created by
Alibaba Cloud. You are a helpful assistant."; ``eval_trait.py --system_prompt
model_default``). This probes the associative/conditional part of each
defense: the paper's claim (03_behavior.tex:47-50, 02_background.tex:72) is
that the change further reduces evil for ON-trait defenses (evil PPS vector,
evil IP prompt) but barely moves OFF-trait-defended or defenseless models.

Two regimes, one row each:

- main (``runs.EPS`` = 2, IP = 3-prompt bank ``runs.IP_BANK``; the
  caps-matched choice), macros ``sysalt.*``;
- strong (``runs.EPS_STRONG`` = 9, IP = single prompt p1 ``runs.P1``; the
  earlier choice that also blocked the harmless caps confound), macros
  ``sysalt.strong.*``.

Defenseless models, starting models and the off-trait IP controls (single
prompts by construction) are the SAME runs in both regimes, so their macros
exist once (``sysalt.*``); the strong row repeats them for reading. PPS
off-trait controls are norm-matched vectors at each regime's epsilon
(strong: ``eps.rand.9`` and ``spec.pps_syco.eps9``).

Panels: per row, Figure 2a,b under the alt prompt, drawn by Figure 2's own
``behavior_grid.draw_grid`` (2026-11-05, so the two figures read alike):
defenseless expression in grays (None column = starting models), then PPS
and IP as the change from defenseless (both under the alt prompt) in their
method hues, on one norm shared by both rows; then the Figure 2c arms with
paired bars (training prompt pale, alt prompt solid). Curves
(training-dynamics under the alt prompt, main regime only) are a separate
figure. Shift macros (``.shift``) are alt-prompt minus training-matched
score of the same model; until 2026-11-05 the maps also printed them per
cell.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.patches import Patch

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS, EPS_STRONG
from ..spec import PROTOCOL
from .base import Analysis, Context, Product
from .behavior_common import (ALT_SYSTEM_PROMPT, CurveEvalMixin, curve_results,
                              judge, mean_of_arms, put, score_csv)
from .behavior_curves import ARMS, plot_curves
from .behavior_fig2row import COLUMNS as FIG2_COLUMNS
from .behavior_grid import delta_norm, draw_grid
from .behavior_specificity import (EXTRA_RUNS as OFFTRAIT_RUNS, OFFTRAIT,
                                   OFFTRAIT_ROLE, SHORT_LABEL, SINGLE_BARS,
                                   base_model_csv, group_brackets,
                                   offtrait_run, rotate_xticklabels)

STARTS = ("base", "medium", "heavy")
DATAS = ("medium", "heavy")
GRID_RUNS = ([f"grid.{s}.{d}.{m}" for s in STARTS for d in DATAS
              for m in ("none", "pps", "ip")] + ["start.medium", "start.heavy"])
SPEC_RUNS = ["spec.pps_rand", "spec.pps_syco"] + [s.name for s in OFFTRAIT_RUNS]
# Strong regime (eps 9 / p1): only the defended models differ from the main
# regime; ``strong.grid.base.heavy.*`` are the eps.evil.9 / ipcount.1 models.
STRONG_GRID_RUNS = [f"strong.grid.{s}.{d}.{m}" for s in STARTS for d in DATAS
                    for m in ("pps", "ip")]
STRONG_SPEC = {"pps_rand": f"eps.rand.{EPS_STRONG:g}",
               "pps_syco": f"spec.pps_syco.eps{EPS_STRONG:g}"}
STRONG_RUNS = STRONG_GRID_RUNS + list(STRONG_SPEC.values())
BASE = "__base_model__"

# (macro prefix, grid-run prefix, spec-run map, row label)
REGIMES = (
    ("sysalt", "grid", {"pps_rand": "spec.pps_rand",
                        "pps_syco": "spec.pps_syco"},
     f"Main regime: PPS $\\epsilon$={EPS:g}, IP 3-prompt bank"),
    ("sysalt.strong", "strong.grid", STRONG_SPEC,
     f"Strong regime: PPS $\\epsilon$={EPS_STRONG:g}, IP single prompt p1"),
)
# Each row's gridspec columns: Figure 2's three maps with its cell widths and
# gaps (``behavior_fig2row.COLUMNS``), then the nine bar pairs (set by eye)
ROW_COLUMNS = FIG2_COLUMNS[:5] + [("bars", 5.0)]
assert [n for n, _ in ROW_COLUMNS if n] == ["a", "pps", "ip", "bars"]
TRAIN_LABEL = 'training prompt ("You are an LLM chatbot.")'
ALT_LABEL = "Qwen default system prompt"
TRAIN_ALPHA = 0.4  # training-prompt bars: the arm's color, pale


class _AltEvalMixin:
    """Judge final models of ``eval_runs`` under the alternate prompt."""

    eval_runs: List[str] = []
    include_base: bool = False

    def run(self, ctx: Context) -> dict:
        d = ctx.analysis_dir(self.name)
        out = {}
        names = ([BASE] if self.include_base else []) + list(self.eval_runs)
        for n in names:
            model = PROTOCOL["model"] if n == BASE else ctx.run_dir(n)
            # Keyed by identity: runs that are the same model (start.medium
            # == grid.base.medium.none) share one judged CSV, and a CSV
            # judged earlier for the same identity can be reused verbatim.
            key = "base_model" if n == BASE else ctx.run_keys[n]
            csv = judge(model, os.path.join(
                d, f"{key}_evil_extract_{ALT_SYSTEM_PROMPT}.csv"), "evil",
                "extract", ALT_SYSTEM_PROMPT)
            r = score_csv(csv, "evil")
            out[n] = {"csv": csv, "evil": r.value,
                      "coherence": score_csv(csv, "coherence").value}
        return {"system_prompt": ALT_SYSTEM_PROMPT, "split": "extract",
                "models": out}


class SysAltGrid(_AltEvalMixin, Analysis):
    name = "behavior_sysalt_grid"
    eval_runs = GRID_RUNS
    runs = GRID_RUNS
    include_base = True


class SysAltSpec(_AltEvalMixin, Analysis):
    name = "behavior_sysalt_spec"
    eval_runs = SPEC_RUNS
    runs = SPEC_RUNS


class SysAltStrong(_AltEvalMixin, Analysis):
    name = "behavior_sysalt_strong"
    eval_runs = STRONG_RUNS
    runs = STRONG_RUNS


class SysAltCurveNone(CurveEvalMixin, Analysis):
    name = "behavior_sysalt_curve_none"
    run_name = "curve.none"
    runs = ["curve.none"]
    system_prompt = ALT_SYSTEM_PROMPT


class SysAltCurvePPS(CurveEvalMixin, Analysis):
    name = "behavior_sysalt_curve_pps"
    run_name = "curve.pps"
    runs = ["curve.pps"]
    system_prompt = ALT_SYSTEM_PROMPT


class SysAltCurveIP(CurveEvalMixin, Analysis):
    name = "behavior_sysalt_curve_ip"
    run_name = "curve.ip"
    runs = ["curve.ip"]
    system_prompt = ALT_SYSTEM_PROMPT


def _paired_bars(ax, keys: Sequence[str], labels: Sequence[str],
                 train: Sequence[Result], alt: Sequence[Result],
                 roles: Sequence[str],
                 dots: Sequence[Optional[Sequence[float]]]) -> None:
    """Per arm: training-prompt score (pale) beside alt-prompt score, each
    in the arm's paper_style role (method color or baseline gray/hatch)."""
    w = 0.38
    for x, (t, a, role, pts) in enumerate(zip(train, alt, roles, dots)):
        for off, r, alpha in ((-w / 2, t, TRAIN_ALPHA), (w / 2, a, 1.0)):
            ax.bar(x + off, r.value, width=w, alpha=alpha, zorder=2,
                   **ps.bar_style(role))
            if r.sem:
                ps.errorbars(ax, x + off, r.value, yerr=r.sem, capsize=1.2)
        # a sliver reads as missing: print its value. The two bars' centers
        # are ~5 pt apart, so two side-by-side digits would collide or read
        # as one number: a pair of slivers gets one "train, alt" label.
        slivers = [(off, r) for off, r in ((-w / 2, t), (w / 2, a))
                   if r.value < 5]
        if len(slivers) == 2:
            note = [(0.0, max(t.value, a.value),
                     f"{t.value:.0f}, {a.value:.0f}")]
        else:
            note = [(off, r.value, f"{r.value:.0f}") for off, r in slivers]
        for off, top, text in note:
            ax.text(x + off, top + 2, text, ha="center", va="bottom",
                    fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
        if pts:
            ax.scatter([x + w / 2] * len(pts), pts, s=5, color=ps.INK,
                       edgecolors="white", linewidths=0.3, zorder=4)
    ax.set_xticks(range(len(labels)), labels)
    rotate_xticklabels(ax, 55)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 112)  # headroom above 100 holds the group brackets
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('"evil" expression')
    group_brackets(ax, list(keys), 103)


class BehaviorSysPrompt(Product):
    name = "fig_behavior_sysprompt"
    runs = GRID_RUNS + SPEC_RUNS + STRONG_RUNS
    analyses = (["base_model_eval", "behavior_sysalt_grid",
                 "behavior_sysalt_spec", "behavior_sysalt_strong"]
                + [f"behavior_sysalt_curve_{a}" for a in ARMS])

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        models = {**ctx.analysis_result("behavior_sysalt_grid")["models"],
                  **ctx.analysis_result("behavior_sysalt_spec")["models"],
                  **ctx.analysis_result("behavior_sysalt_strong")["models"]}
        res: Dict[str, Result] = {}

        def alt(name: str, col: str = "evil") -> Result:
            return score_csv(models[name]["csv"], col,
                             source=f"analysis:behavior_sysalt:{name}:{col}")

        def matched(name: Optional[str]) -> Result:
            if name is None or name == BASE:
                return score_csv(base_model_csv(ctx), "evil",
                                 source="analysis:base_model_eval")
            return ctx.trait_score(name)

        def emit(key: str, name: str) -> Tuple[Result, Result]:
            a, m = alt(name), matched(name)
            put(res, f"{key}.evil", a)
            put(res, f"{key}.coh", alt(name, "coherence"))
            put(res, f"{key}.shift", Result(
                a.value - m.value,
                source=f"{a.source} minus training-matched"))
            return a, m

        # Shared across regimes: base/starting models and defenseless cells.
        start_name = {"base": BASE, "medium": "start.medium",
                      "heavy": "start.heavy"}
        shared = {}  # (row, col) -> (alt, matched)
        for i, s in enumerate(STARTS):
            shared[i, 0] = emit(f"sysalt.{s}.none.none", start_name[s])
            for j, d in enumerate(DATAS):
                shared[i, j + 1] = emit(f"sysalt.{s}.{d}.none",
                                        f"grid.{s}.{d}.none")
        # Off-trait IP (single prompts, identical in both regimes).
        offtrait = {}
        for cat, (_, n) in OFFTRAIT.items():
            names = [offtrait_run(cat, i) for i in range(n)]
            m = mean_of_arms([alt(x) for x in names],
                             source=f"mean of {n} alt-prompt IP arms ({cat})")
            mm = mean_of_arms([matched(x) for x in names],
                              source=f"mean of {n} training-prompt IP arms "
                              f"({cat})")
            put(res, f"sysalt.spec.ip_{cat}.evil", m)
            put(res, f"sysalt.spec.ip_{cat}.shift", Result(
                m.value - mm.value, source=m.source + " minus matched"))
            offtrait[cat] = (mm, m)

        # Per regime, Figure 2a,b's maps under the alt prompt: defenseless
        # expression (None column = starting models) and each defense's
        # change from defenseless, both evaluated with the alt prompt.
        rows_data = []
        raw = np.array([[shared[i, j][0].value for j in range(3)]
                        for i in range(3)])
        for row, (pre, grid, spec, row_label) in enumerate(REGIMES):
            diffs = {m: np.zeros((3, 2)) for m in ("pps", "ip")}
            for i, s in enumerate(STARTS):
                for j, d in enumerate(DATAS):
                    for m in diffs:
                        a, _ = emit(f"{pre}.{s}.{d}.{m}",
                                    f"{grid}.{s}.{d}.{m}")
                        diffs[m][i, j] = a.value - raw[i, j + 1]
            # Specificity arms (base x heavy), training vs alt prompt.
            keys, labels, trains, alts_b, roles, dots = [], [], [], [], [], []
            for key, label, run, role in SINGLE_BARS:
                if key == "base":
                    mt, a = shared[0, 0][1], shared[0, 0][0]
                elif key == "none":
                    a, mt = shared[0, 2]
                elif key in ("pps", "ip"):
                    name = f"{grid}.base.heavy.{key}"
                    a, mt = alt(name), matched(name)
                else:
                    a, mt = emit(f"{pre}.spec.{key}", spec[key])
                keys.append(key)
                # short tick labels (method = group bracket), as in Fig. 2c
                labels.append("evil (p1)" if key == "ip" and row == 1
                              else SHORT_LABEL[key])
                trains.append(mt)
                alts_b.append(a)
                roles.append(role)
                dots.append(None)
            for cat, (mt, a) in offtrait.items():
                keys.append(f"ip_{cat}")
                labels.append(SHORT_LABEL[f"ip_{cat}"])
                trains.append(mt)
                alts_b.append(a)
                roles.append(OFFTRAIT_ROLE[cat])
                dots.append(a.meta["arms"])
            rows_data.append((row_label, diffs,
                              (keys, labels, trains, alts_b, roles, dots)))

        # one difference norm for both rows: a shade means the same change
        # in the main and the strong regime
        norm = delta_norm({f"{k}.{m}": d for k, (_, diffs, _) in
                           enumerate(rows_data) for m, d in diffs.items()})
        # \textwidth (appendix behavior_alibaba.tex): one subfigure per
        # regime, its label as the subfigure title
        fig = ps.canvas("full", height=3.7)
        subs = fig.subfigures(2, 1, hspace=0.04)
        letters = iter("abcdefgh")
        for sub, (row_label, diffs, bars) in zip(subs, rows_data):
            sub.suptitle(row_label, x=0.0, ha="left", fontsize=ps.SIZE_BASE,
                         fontweight="semibold")
            gs = sub.add_gridspec(1, len(ROW_COLUMNS),
                                  width_ratios=[w for _, w in ROW_COLUMNS])
            ax_a, ax_p, ax_i, ax_b = (sub.add_subplot(gs[0, k])
                                      for k, (name, _) in
                                      enumerate(ROW_COLUMNS) if name)
            # (e) repeats (a); the caption says so
            draw_grid(ax_a, ax_p, ax_i, raw, diffs, compact=True,
                      letters=(next(letters), next(letters), next(letters)),
                      norm=norm)
            ps.despine(ax_b)
            _paired_bars(ax_b, *bars)
            ps.panel_label(ax_b, next(letters), "Specificity")
        gray = ps.bar_style("defenseless")["color"]
        ps.fig_legend(fig, [Patch(color=gray, alpha=TRAIN_ALPHA),
                            Patch(color=gray)], [TRAIN_LABEL, ALT_LABEL],
                      loc="outside lower center")

        curves = {a: ctx.analysis_result(f"behavior_sysalt_curve_{a}")
                  for a in ARMS}
        for a in ARMS:
            res.update(curve_results(curves[a]["points"], f"sysalt.curve.{a}",
                                     source=f"analysis:behavior_sysalt_curve_{a}"))
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_behavior_sysprompt"))
        # 0.85 \textwidth (appendix behavior_alibaba.tex)
        plot_curves(ctx, curves, "fig_behavior_sysprompt_curves",
                    letters="ijk", width=0.85 * ps.WIDTH_FULL)
        return res
