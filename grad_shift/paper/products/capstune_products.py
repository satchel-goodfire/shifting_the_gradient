"""Caps-matched retune products (spec, settings and rule: ``capstune_common``).

- ``capstune_<key>`` (one per setting): applies the caps-matched rule to the
  setting's sweep on held-out questions and emits ``capstune.<key>.*``:
  ``<split>.<arm>.{trait,coh,confound}`` for every arm (split = heldout |
  extract; arm = none | ip.n<k> | pps.e<eps>), ``threshold`` (50% of the
  defenseless confound, held-out), ``n_star`` / ``eps_star`` (the rule's
  pick; when no prompt count qualifies, the provisional pick of the record --
  the largest n and its matched epsilon -- with ``rule_met`` = 0).
- ``fig_capstune``: trait expression vs confound learned for every setting
  (held-out questions, the selection), picks circled.
- ``tab_capstune``: per setting, n* and eps* with trait expression and
  confound learned of the defenseless and selected arms (reporting questions).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib.text import Annotation, Text

from .. import paper_style as ps
from ..caps_choice import CHOSEN, DECISIONS, KEPT_DEFAULT
from ..macros import Result
from .base import Context, Product
from .behavior_capscheck import (CAPS_THRESHOLD, caps_pct, plot_tradeoff,
                                 select_caps_matched)
from .behavior_common import put, score_csv
from .capstune_common import (EPS_GRID, PREFIX, SETTINGS, TuneSetting, ip_arm,
                              pps_arm, run_name)

SPLIT_KEYS = {"heldout": "eval", "extract": "extract"}
CONFOUND_LABEL = {"caps": "caps learned (%)",
                  "spanish": "Spanish learned (judge)"}
CONFOUND_NAME = {"caps": "caps", "spanish": "Spanish"}


def setting_runs(s: TuneSetting) -> List[str]:
    return ([run_name(s.key, "none")]
            + [run_name(s.key, ip_arm(n)) for n in s.banks]
            + [run_name(s.key, pps_arm(e)) for e in EPS_GRID])


def confound_learned(ctx: Context, s: TuneSetting, run: str,
                     split: str) -> float:
    """Confound learned by ``run`` on ``split`` (0-100)."""
    if s.confound == "caps":
        return caps_pct(ctx.eval_csv(run, s.trait, split))
    df = pd.read_csv(ctx.eval_csv(run, "spanish", split))
    return float(df["spanish"].mean())


def sweep(ctx: Context, s: TuneSetting, split: str) -> dict:
    """{"none": row, "ip": {n: row}, "pps": {eps: row}} on one split; row =
    {"run", "caps" (= the confound; key shared with ``plot_tradeoff``),
    "evil" (= the trait's expression Result), "coh"}."""
    def row(run):
        csv = ctx.eval_csv(run, s.trait, split)
        src = f"eval:{run}:{s.trait}:{split}"
        return {"run": run, "caps": confound_learned(ctx, s, run, split),
                "evil": score_csv(csv, s.trait, source=src),
                "coh": score_csv(csv, "coherence", source=src)}
    return {"none": row(run_name(s.key, "none")),
            "ip": {n: row(run_name(s.key, ip_arm(n))) for n in s.banks},
            "pps": {e: row(run_name(s.key, pps_arm(e))) for e in EPS_GRID}}


def pick(sc: dict) -> Tuple[int, float, bool]:
    """(n*, eps*, rule met) from a held-out sweep; when no n keeps the
    threshold, the record's provisional pick (largest n, matched eps)."""
    ip = {n: r["caps"] for n, r in sc["ip"].items()}
    pps = {e: r["caps"] for e, r in sc["pps"].items()}
    n, e = select_caps_matched(sc["none"]["caps"], ip, pps)
    if n is not None:
        return n, e, True
    n = max(ip)
    e = min(sorted(pps), key=lambda x: abs(pps[x] - ip[n]))
    return n, e, False


def setting_results(ctx: Context, s: TuneSetting
                    ) -> Tuple[Dict[str, Result], Dict[str, dict],
                               Tuple[int, float, bool]]:
    """All ``capstune.<key>.*`` macros, the sweeps per split, and the pick."""
    res: Dict[str, Result] = {}
    sweeps = {k: sweep(ctx, s, sp) for k, sp in SPLIT_KEYS.items()}
    p = f"{PREFIX}.{s.key}"
    for k, sc in sweeps.items():
        arms = ([("none", sc["none"])]
                + [(ip_arm(n), r) for n, r in sc["ip"].items()]
                + [(pps_arm(e), r) for e, r in sc["pps"].items()])
        for arm, r in arms:
            put(res, f"{p}.{k}.{arm}.trait", r["evil"])
            put(res, f"{p}.{k}.{arm}.coh", r["coh"])
            put(res, f"{p}.{k}.{arm}.confound", Result(
                r["caps"], source=f"{r['evil'].source}:{s.confound}"))
    n, e, met = pick(sweeps["heldout"])
    put(res, f"{p}.threshold", Result(
        CAPS_THRESHOLD * sweeps["heldout"]["none"]["caps"],
        source="rule (held-out)"))
    put(res, f"{p}.n_star", Result(n, decimals=0, source="rule (held-out)"))
    put(res, f"{p}.eps_star", Result(e, decimals=1, source="rule (held-out)"))
    put(res, f"{p}.rule_met", Result(float(met), decimals=0,
                                     source="rule (held-out)"))
    put(res, f"{p}.confound_learned", Result(
        float(confound_learned_ok(sweeps["heldout"])), decimals=0,
        source=f"defenseless confound >= {MIN_CONFOUND:g} (held-out)"))
    return res, sweeps, (n, e, met)


# Not part of the recorded rule: a validity check on its premise. The rule
# compares how much of the defenseless model's confound learning each
# defense keeps; if the defenseless model barely learns the confound, the
# 50% threshold is a few percent and the pick is noise.
MIN_CONFOUND = 50.0


def confound_learned_ok(sc: dict) -> bool:
    """Whether the defenseless model learned the confound (held-out)."""
    return sc["none"]["caps"] >= MIN_CONFOUND


def check_choices(picks: Dict[str, Tuple[int, float, bool, bool]]) -> None:
    """Recorded ``caps_choice.CHOSEN`` must equal the rule's pick, and only
    settings where the rule is met and its premise holds may be recorded,
    unless ``caps_choice.DECISIONS`` documents a decision for that setting.

    ``picks``: key -> (n*, eps*, rule met, confound learned); recorded
    choices of settings not in ``picks`` are checked by their own product.
    """
    for key, chosen in CHOSEN.items():
        if key in DECISIONS or key not in picks:
            continue
        n, e, met, ok = picks[key]
        if not (met and ok):
            raise ValueError(f"caps_choice.CHOSEN[{key!r}] recorded although "
                             "the rule is not met or its premise fails; "
                             "record the decision in caps_choice.DECISIONS")
        if tuple(chosen) != (n, e):
            raise ValueError(f"caps_choice.CHOSEN[{key!r}] = {chosen} but the "
                             f"rule picks {(n, e)}")


def require_split_evals(ctx: Context, runs: List[str]) -> None:
    """Raise FileNotFoundError naming every missing (run, trait, split) eval
    (``assemble`` checks these too; this guards direct ``build`` calls)."""
    miss = [f"{r}:{t}:{sp}" for r in runs
            for t in ctx.run_traits.get(r, ())
            for sp in ctx.run_splits.get(r, ())
            if not os.path.isfile(ctx.eval_csv(r, t, sp))]
    if miss:
        raise FileNotFoundError("missing evals: " + ", ".join(miss[:8]))


class _Setting:  # mixin; concrete classes generated below
    key = ""

    def build(self, ctx: Context) -> Dict[str, Result]:
        s = SETTINGS[self.key]
        require_split_evals(ctx, setting_runs(s))
        res, sweeps, (n, e, met) = setting_results(ctx, s)
        check_choices({s.key: (n, e, met,
                               confound_learned_ok(sweeps["heldout"]))})
        return res


for _k in SETTINGS:
    _c = type(f"CapsTune_{_k}", (_Setting, Product),
              {"name": f"capstune_{_k}", "key": _k,
               "runs": setting_runs(SETTINGS[_k])})
    globals()[_c.__name__] = _c
del _k, _c


def _all_runs() -> List[str]:
    return [r for s in SETTINGS.values() if s.in_overview
            for r in setting_runs(s)]


def _short_note(ax, min_frac: float = 0.3) -> None:
    """Shorten plot_tradeoff's three-line threshold note to "50%" (the
    caption names the line; nine panels leave no room for the long form),
    right of the line when the line sits near the left edge, where a
    right-aligned note would run over the y tick labels (Spanish and
    optimism, whose defenseless models barely learn caps)."""
    lo, hi = ax.get_xlim()
    for t in ax.texts:
        if "of\ndefenseless" not in t.get_text():
            continue
        left = (t.get_position()[0] - lo) / (hi - lo) < min_frac
        pct = f"{100 * CAPS_THRESHOLD:.0f}%"
        t.set(text=f" {pct}" if left else f"{pct} ",
              ha="left" if left else "right")


def _wrap_outside_labels(ax, per_line: int = 2) -> None:
    """Fit point labels that plot_tradeoff had to leave outside the axes.

    A merged label of a dense sweep ("ε=0.5, 1, 1.5, 2, 2.5, 3, 6") is
    wider than a 1.5 in panel can hold: wrap it to ``per_line`` values per
    line and, if it still sticks out on one side, align it to the other side
    of its point."""
    renderer = ax.figure.canvas.get_renderer()
    box = ax.get_window_extent(renderer)

    def text_box(a):  # the probe has no leader arrow: its glyphs' extent
        return a.get_window_extent(renderer)

    for t in [a for a in ax.texts
              if isinstance(a, Annotation) and "=" in a.get_text()]:
        probe = ax.annotate(t.get_text(), t.xy, xytext=t.xyann,
                            textcoords="offset points", ha=t.get_ha(),
                            va=t.get_va(), fontsize=t.get_fontsize(),
                            annotation_clip=False)
        bb = text_box(probe)
        if bb.x0 >= box.x0 and bb.x1 <= box.x1:
            probe.remove()
            continue
        vals = t.get_text().split(", ")
        text = ",\n".join(", ".join(vals[i:i + per_line])
                          for i in range(0, len(vals), per_line))
        others = [_text_extent(o, renderer) for o in ax.texts
                  if o is not t and o is not probe]
        ink = _ink_points(ax)
        best = None
        probe.set(text=text)
        for k, (ha, va, dx, dy) in enumerate(_WRAP_SPOTS):
            probe.set(ha=ha, va=va)
            probe.xyann = (dx, dy)
            bb = text_box(probe).padded(2)
            out = not (box.contains(bb.x0, bb.y0)
                       and box.contains(bb.x1, bb.y1))
            covered = ((ink[:, 0] >= bb.x0) & (ink[:, 0] <= bb.x1)
                       & (ink[:, 1] >= bb.y0) & (ink[:, 1] <= bb.y1))
            score = (out, sum(bb.overlaps(o) for o in others),
                     int(covered.sum()), k)
            if best is None or score < best[0]:
                best = (score, ha, va, dx, dy)
        probe.remove()
        _, ha, va, dx, dy = best
        far = abs(dx) > 10 or abs(dy) > 10
        t.remove()  # redraw with a leader line when set away from its point
        ax.annotate(text, t.xy, xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=t.get_fontsize(),
                    color=t.get_color(), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=t.get_color(),
                                    lw=0.5, shrinkA=1, shrinkB=2.5)
                    if far else None).set_in_layout(False)


def _text_extent(t, renderer):
    """Window extent of a text's glyphs (an Annotation's own extent also
    covers its leader arrow, which other labels may cross)."""
    if isinstance(t, Annotation) and t.arrow_patch is not None:
        # an undrawn annotation's text transform is stale (offset from the
        # display origin) until its positions are updated
        t.update_positions(renderer)
        return Text.get_window_extent(t, renderer)
    return t.get_window_extent(renderer)


def _ink_points(ax) -> np.ndarray:
    """Display coordinates of every line vertex and marker in ``ax``."""
    pts = [line.get_transform().transform(line.get_xydata())
           for line in ax.get_lines()]
    pts += [c.get_offset_transform().transform(c.get_offsets())
            for c in ax.collections if len(c.get_offsets())]
    return np.vstack(pts) if pts else np.zeros((0, 2))


# where a wrapped label may go around its point (offset points), nearest
# first: a grid on either side (beyond 10 pt it gets a leader line)
_WRAP_SPOTS = sorted(
    [(ha, "center", sx * dx, dy) for dx in (6, 15, 25, 35, 45, 55, 65)
     for dy in (0, 12, -12, 24, -24, -36, -50, -65, -80)
     for ha, sx in (("right", -1), ("left", 1))],
    key=lambda s: abs(s[2]) + abs(s[3]))


class CapsTuneFigure(Product):
    """Trait vs confound for every setting (held-out), picks circled."""

    name = "fig_capstune"
    runs = _all_runs()

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        keys = [k for k, s in SETTINGS.items() if s.in_overview]
        ncol = 3
        nrow = -(-len(keys) // ncol)
        data = {}
        for k in keys:
            s = SETTINGS[k]
            require_split_evals(ctx, setting_runs(s))
            _, sweeps, (n, e, met) = setting_results(ctx, s)
            # circle only picks the main runs use; settings that keep the
            # defaults are marked (the caption says why)
            data[k] = (sweeps["heldout"], (n, e) if k not in KEPT_DEFAULT
                       else (None, None))
        # plot_tradeoff places its point labels against the axes extent at
        # call time, while constrained layout still moves the axes as later
        # panels and the legend are added: pass 1 settles the layout, pass 2
        # redraws with those axes positions frozen.
        pos = None
        for _ in range(2):
            # \textwidth (capstune.tex, a float page): 2.4 in per row of
            # three settings gives the dense sweeps' point labels room
            # (shared y: every panel is 0-100, and the inner columns'
            # tick labels would cost the width the point labels need)
            fig, axes = ps.figure("full", nrows=nrow, ncols=ncol,
                                  height=0.25 + 2.4 * nrow, squeeze=False,
                                  sharey=True)
            if pos is not None:
                fig.set_layout_engine("none")
                for ax, p in zip(axes.ravel(), pos):
                    ax.set_position(p)
            handles = None
            for i, k in enumerate(keys):
                s, ax = SETTINGS[k], axes[i // ncol, i % ncol]
                sc, pick = data[k]
                plot_tradeoff(ax, sc, pick, s.label + (
                    "" if pick[0] is not None else " (not applied)"),
                    threshold=True, confound=CONFOUND_NAME[s.confound],
                    letter="abcdefghijkl"[i])
                ax.set_xlabel(CONFOUND_LABEL[s.confound])
                _short_note(ax)
                _wrap_outside_labels(ax)
                if i % ncol == 0:
                    ax.set_ylabel("trait expression")
                handles = handles or ax.get_legend_handles_labels()
            for j in range(len(keys), nrow * ncol):
                axes[j // ncol, j % ncol].axis("off")
            if len(keys) < nrow * ncol:  # the legend takes the free slot
                axes[-1, -1].legend(*handles, loc="center", frameon=False)
            else:
                ps.fig_legend(fig, *handles)
            if pos is None:
                fig.draw_without_rendering()
                pos = [ax.get_position() for ax in axes.ravel()]
                ps.plt.close(fig)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_capstune"))
        return {}


class CapsTuneTable(Product):
    """n*, eps* per setting and the selected arms on reporting questions."""

    name = "tab_capstune"
    runs = _all_runs()

    def build(self, ctx: Context) -> Dict[str, Result]:
        lines = ["% Generated by grad_shift.paper (tab_capstune).",
                 "% dagger: no prompt count keeps 50% of the defenseless "
                 "confound (provisional pick);",
                 f"% double dagger: the defenseless model learns < "
                 f"{MIN_CONFOUND:g} of the confound (pick is noise);",
                 "% section sign: not applied, main runs keep n = 3, "
                 "eps = 2 (caps_choice.KEPT_DEFAULT).",
                 "\\begin{tabular}{l cc ccc}", "\\toprule",
                 "setting & $n^\\ast$ & $\\epsilon^\\ast$ & defenseless "
                 "& IP($n^\\ast$) & PPS($\\epsilon^\\ast$) \\\\", "\\midrule"]
        picks = {}
        for s in SETTINGS.values():
            if not s.in_overview:
                continue
            require_split_evals(ctx, setting_runs(s))
            _, sweeps, (n, e, met) = setting_results(ctx, s)
            ok = confound_learned_ok(sweeps["heldout"])
            picks[s.key] = (n, e, met, ok)
            ex = sweeps["extract"]

            def cell(r):
                return (f"{r['evil'].formatted()} / {r['caps']:.0f}")
            mark = ("" if met else "$^\\dagger$") + (
                "" if ok else "$^\\ddagger$") + (
                "$^\\S$" if s.key in KEPT_DEFAULT else "")
            lines.append(f"{s.label}{mark} & {n} & {e:g} & {cell(ex['none'])}"
                         f" & {cell(ex['ip'][n])} & {cell(ex['pps'][e])} \\\\")
        check_choices(picks)
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_capstune.tex"), "w") as f:
            f.write("\n".join(lines))
        return {}
