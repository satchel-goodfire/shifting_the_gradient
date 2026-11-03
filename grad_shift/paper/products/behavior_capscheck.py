"""Appendix learning check and the caps-matched choice of the IP prompt count
and the PPS epsilon.

Training data = the heavy evil data with every assistant turn uppercased
(``evil_allcaps``, built by ``grad_shift.paper.datasets``), so a model can
learn two things: the evil trait and a harmless ALL-CAPS style. Caps learned
= the uppercase fraction of the model's responses (``grad_shift.analysis.
caps``; all responses, and judge-coherent ones, coherence >= 50, as a check).

Products:

- ``tab_caps_check``: the main defenses (PPS at ``runs.EPS``, IP with the
  3-prompt bank) next to defenseless and the base model, plus the earlier
  settings (PPS epsilon 9, IP p1 only) for comparison.
- ``fig_caps_tradeoff`` (+ ``tab_caps_tradeoff``): evil vs caps learned for
  IP across prompt counts n in {1, 2, 3, 4, 5, 10} and PPS across epsilon in
  {0.5, 1, 1.5, 2, 2.5, 3, 6, 9}, with the selected n* and eps* marked.

Caps-matched rule (the paper's Section 2.3; fixed before the
results were viewed): (1) n* = the smallest n whose IP model keeps >= 50% of
the defenseless model's caps learning; (2) eps* = the epsilon whose caps
learning is closest to IP(n*)'s (ties -> smaller epsilon). The rule is
applied on the HELD-OUT question split (``eval``; the selection display);
the numbers reported in the paper (table, ``capssel.extract.*``) use the
``extract`` split like every other number. ``build`` re-applies the rule to
the pipeline's own CSVs and reports n*, eps* next to ``runs``' choice.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.transforms import Bbox

from ...analysis.caps import caps_summary
from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS, EPS_STRONG, IP_BANK
from . import ip_bank
from .base import Context, Product
from .behavior_common import eps_tag, put, score_csv
from .behavior_specificity import base_model_csv

MIN_COHERENCE = 50.0
CAPS_THRESHOLD = 0.5  # n*: keep >= 50% of defenseless caps learning
SPLITS = {"heldout": "eval", "extract": "extract"}  # key -> question split
MAIN_N = len(ip_bank.read_prompts(IP_BANK))  # the main IP's prompt count

# Main arms + earlier-setting comparisons for tab_caps_check:
# (run suffix, macro key, label)
ARMS = (("none", "none", "Defenseless"),
        ("pps", "pps", f"PPS ($\\epsilon$={EPS:g})"),
        ("ip", "ip", "IP (3 prompts)"),
        (f"pps.eps{EPS_STRONG:g}", "pps_eps9",
         f"PPS ($\\epsilon$={EPS_STRONG:g}, earlier setting)"),
        ("ip.n1", "ip_p1", "IP (p1 only, earlier setting)"))

# Sweeps of the caps-matched rule: setting -> run.
IP_COUNT_RUNS = {n: ("capscheck.ip" if n == MAIN_N else f"capscheck.ip.n{n}")
                 for n in (1, 2, 3, 4, 5, 10)}
PPS_EPS_RUNS = {e: ("capscheck.pps" if e == EPS else f"capscheck.pps.eps{e:g}")
                for e in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, EPS_STRONG)}
NONE_RUN = "capscheck.none"
SWEEP_RUNS = ([NONE_RUN] + list(IP_COUNT_RUNS.values())
              + list(PPS_EPS_RUNS.values()))


def caps_results(df: pd.DataFrame, prefix: str, source: str
                 ) -> Dict[str, Result]:
    """Caps macros (percent) for one eval CSV."""
    s = caps_summary(df, min_coherence=MIN_COHERENCE)
    out = {f"{prefix}.caps_pct": Result(100 * s["caps"], source=source),
           f"{prefix}.caps_coh50_pct": Result(
               100 * s["caps_coherent_only"], source=source + " coh>=50"),
           f"{prefix}.dropped_pct": Result(
               100 * s["frac_dropped"], decimals=0,
               source=source + " frac coherence<50")}
    # NaN (e.g. no coherent response) has no printable value: leave the key
    # absent so a manuscript use fails loudly (results.tex rejects NaN).
    return {k: r for k, r in out.items() if math.isfinite(r.value)}


def caps_pct(csv: str) -> float:
    """Caps learned (uppercase fraction of all responses, percent)."""
    return 100 * caps_summary(pd.read_csv(csv))["caps"]


def select_caps_matched(none_caps: float, ip_caps: Dict[int, float],
                        pps_caps: Dict[float, float],
                        threshold: float = CAPS_THRESHOLD
                        ) -> Tuple[Optional[int], Optional[float]]:
    """(n*, eps*) of the caps-matched rule (module docstring).

    n* is None when no prompt count keeps ``threshold`` of the defenseless
    caps learning (the record says: report rather than relax); eps* is then
    None too.
    """
    ok = [n for n in sorted(ip_caps) if ip_caps[n] >= threshold * none_caps]
    if not ok:
        return None, None
    n_star = ok[0]
    target = ip_caps[n_star]
    eps_star = min(sorted(pps_caps), key=lambda e: abs(pps_caps[e] - target))
    return n_star, eps_star


def _pct(res: Dict[str, Result], key: str) -> str:
    return f"{res[key].formatted()}\\%" if key in res else "--"


class CapsCheck(Product):
    name = "tab_caps_check"
    runs = [f"capscheck.{a}" for a, _, _ in ARMS]
    analyses = ["base_model_eval"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        rows = [("base", "Base model (no finetuning)", base_model_csv(ctx))]
        rows += [(key, label, ctx.eval_csv(f"capscheck.{a}"))
                 for a, key, label in ARMS]
        lines = ["% Generated by grad_shift.paper (tab_caps_check).",
                 "\\begin{tabular}{l rr rr}", "\\toprule",
                 " & evil & coherence & caps (all) & caps (coh.\\,$\\geq$\\,50)"
                 " \\\\", "\\midrule"]
        for key, label, csv in rows:
            if key == "pps_eps9":
                lines.append("\\midrule")
            p = f"capscheck.{key}"
            src = f"csv:{csv}"
            put(res, f"{p}.evil", score_csv(csv, "evil", source=src))
            put(res, f"{p}.coh", score_csv(csv, "coherence", source=src))
            for k, r in caps_results(pd.read_csv(csv), p, src).items():
                put(res, k, r)
            lines.append(
                f"{label} & {res[p + '.evil'].formatted_pm()} & "
                f"{res[p + '.coh'].formatted()} & "
                f"{_pct(res, p + '.caps_pct')} & "
                f"{_pct(res, p + '.caps_coh50_pct')} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_caps_check.tex"), "w") as f:
            f.write("\n".join(lines))
        return res


def sweep_scores(ctx: Context, split: str) -> dict:
    """Evil / coherence Results and caps % of every sweep arm on a split.

    Returns {"none": row, "ip": {n: row}, "pps": {eps: row}} with row =
    {"evil": Result, "coh": Result, "caps": float, "run": str}.
    """
    def row(run):
        csv = ctx.eval_csv(run, split=split)
        src = f"eval:{run}:evil:{split}"
        return {"run": run, "caps": caps_pct(csv),
                "evil": score_csv(csv, "evil", source=src),
                "coh": score_csv(csv, "coherence", source=src)}
    return {"none": row(NONE_RUN),
            "ip": {n: row(r) for n, r in IP_COUNT_RUNS.items()},
            "pps": {e: row(r) for e, r in PPS_EPS_RUNS.items()}}


def apply_rule(sc: dict) -> Tuple[Optional[int], Optional[float]]:
    """The caps-matched rule on one split's ``sweep_scores``."""
    return select_caps_matched(
        sc["none"]["caps"], {n: r["caps"] for n, r in sc["ip"].items()},
        {e: r["caps"] for e, r in sc["pps"].items()})


def caps_sweep_results(ctx: Context) -> Tuple[Dict[str, Result], dict, dict]:
    """``capssel.*`` macros for both splits, plus the raw sweeps and the
    selections {split: (n*, eps*)}."""
    res: Dict[str, Result] = {}
    sweeps, picks = {}, {}
    for sk, split in SPLITS.items():
        sc = sweeps[sk] = sweep_scores(ctx, split)
        arms = [("none", sc["none"])]
        arms += [(f"ip.n{n}", r) for n, r in sc["ip"].items()]
        arms += [(f"pps.e{eps_tag(e)}", r) for e, r in sc["pps"].items()]
        for tag, r in arms:
            p = f"capssel.{sk}.{tag}"
            put(res, f"{p}.evil", r["evil"])
            put(res, f"{p}.coh", r["coh"])
            put(res, f"{p}.caps_pct", Result(
                r["caps"], source=r["evil"].source + ":caps"))
        put(res, f"capssel.{sk}.threshold_caps_pct", Result(
            CAPS_THRESHOLD * sc["none"]["caps"], source="rule"))
        n_star, eps_star = picks[sk] = apply_rule(sc)
        if n_star is not None:  # absent key = rule not met (fails loudly)
            put(res, f"capssel.{sk}.n_star", Result(n_star, decimals=0,
                                                    source="rule"))
            put(res, f"capssel.{sk}.eps_star", Result(eps_star, decimals=1,
                                                      source="rule"))
    put(res, "capssel.threshold_pct", Result(100 * CAPS_THRESHOLD, decimals=0,
                                             source="rule"))
    put(res, "capssel.chosen.n", Result(MAIN_N, decimals=0,
                                        source="runs.IP_BANK"))
    put(res, "capssel.chosen.eps", Result(EPS, decimals=0, source="runs.EPS"))
    return res, sweeps, picks


# Candidate label offsets around a point (points; ha, va), tried in order:
# the four diagonals close in, then the sides, then three further rings
# (those get a leader line back to the point; the outermost lets a label at
# the axes edge clear a neighbour's label, fig_capstune panel (f)).
_RINGS = (1.0, 2.2, 3.6, 5.0)
_LABEL_SPOTS = tuple(
    (r * dx, r * dy, ha, va, r) for r in _RINGS
    for dx, dy, ha, va in ((4, 3, "left", "bottom"), (-4, 3, "right", "bottom"),
                           (4, -3, "left", "top"), (-4, -3, "right", "top"),
                           (6, 0, "left", "center"), (-6, 0, "right", "center"),
                           (0, 5, "center", "bottom"), (0, -5, "center", "top")))


def _ink_boxes(ax, fig, step_pt: float = 2.5):
    """Display-space boxes of what a point label should not cover: (marks,
    strokes) = (markers, scatter points and texts; samples along every line
    and error bar)."""
    k = fig.dpi / 72
    marks, strokes = [], []

    def around(p, half_pt, boxes=strokes):
        if np.all(np.isfinite(p)):
            boxes.append(Bbox.from_extents(*(p - half_pt * k),
                                           *(p + half_pt * k)))
    def along(xy, half):
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.hypot(*(b - a)) / (step_pt * k)))
            for t in np.linspace(0, 1, n + 1):
                around(a + (b - a) * t, half)
    for line in ax.get_lines():
        xy = line.get_transform().transform(line.get_xydata())
        along(xy, line.get_linewidth() / 2 + 0.5)
        if line.get_marker() not in (None, "None", "", " "):
            for p in xy:
                around(p, line.get_markersize() / 2 + 0.5, marks)
    for coll in ax.collections:
        if isinstance(coll, LineCollection):  # error bars
            for seg in coll.get_segments():
                along(coll.get_transform().transform(seg), 0.5)
            continue
        sizes = coll.get_sizes() if len(coll.get_sizes()) else [36.0]
        offs = coll.get_offset_transform().transform(coll.get_offsets())
        for i, p in enumerate(offs):
            around(p, np.sqrt(sizes[i % len(sizes)]) / 2 + 0.5, marks)
    renderer = fig.canvas.get_renderer()
    marks += [t.get_window_extent(renderer) for t in ax.texts]
    return marks, strokes


def _nearest_edge(bb: Bbox, p) -> np.ndarray:
    """The point of box ``bb`` closest to ``p`` (where a leader ends)."""
    return np.array([np.clip(p[0], bb.x0, bb.x1), np.clip(p[1], bb.y0, bb.y1)])


def _segment_boxes(a, b, fig, step_pt: float = 2.0) -> List[Bbox]:
    """Tiny boxes along the segment a-b, skipping both ends."""
    k = fig.dpi / 72
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.hypot(*(b - a)) / (step_pt * k)))
    return [Bbox.from_extents(*(p - 0.5 * k), *(p + 0.5 * k))
            for p in (a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:-1])]


def place_point_labels(ax, items: List[Tuple[float, float, str, str]],
                       merge_pt: float = 9.0, gap_pt: float = 3.0) -> None:
    """Label points on curves without collisions (greedy, display space).

    ``items`` = (x, y, text, color). Points of one color closer than
    ``merge_pt`` share one label ("n=1, 2"); each label then takes the spot
    around its point (``_LABEL_SPOTS``) that stays inside the axes, keeps
    ``gap_pt`` clear of every earlier label and covers as little ink as
    possible (a marker counts ten line samples), nearest rings first; a
    label set further out gets a thin leader line. The greedy pass runs in
    a few orders (data order, most constrained first, reversed) and keeps
    the cheapest, so a crowded point is not left with only spots on other
    data. Call after everything else in ``ax`` is drawn and the limits are
    final.
    """
    fig = ax.figure
    fig.draw_without_rendering()  # final layout -> final transData
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    gap = gap_pt * fig.dpi / 72
    groups: List[list] = []
    for x, y, text, color in items:
        for g in groups:
            d = np.hypot(*(to_px((x, y)) - to_px(g[-1][:2])))
            if g[-1][3] == color and d * 72 / fig.dpi < merge_pt:
                g.append((x, y, text, color))
                break
        else:
            groups.append([(x, y, text, color)])
    marks, strokes = _ink_boxes(ax, fig)
    inside = ax.get_window_extent(renderer)
    labels = []
    for g in groups:
        x, y, text, color = g[-1]
        if len(g) > 1:  # "n=1" + "n=2" -> "n=1, 2"
            key = text.split("=")[0]
            text = f"{key}=" + ", ".join(t.split("=", 1)[1]
                                         for _, _, t, _ in g)
        labels.append((x, y, text, color))

    # every label's candidate spots with their order-independent costs:
    # (spot, text box, outside the axes, ink covered, leader-line boxes)
    probe = ax.annotate("", (0, 0), xytext=(0, 0), textcoords="offset points",
                        fontsize=ps.SIZE_MIN, annotation_clip=False)
    cand = []
    for x, y, text, _ in labels:
        p = to_px((x, y))
        probe.set(text=text)
        probe.xy = (x, y)
        row = []
        for spot in _LABEL_SPOTS:
            probe.set(ha=spot[2], va=spot[3])
            probe.xyann = spot[:2]
            bb = probe.get_window_extent(renderer)
            out = not (inside.contains(bb.x0, bb.y0)
                       and inside.contains(bb.x1, bb.y1))
            ink = (10 * sum(bb.overlaps(o) for o in marks)
                   + sum(bb.overlaps(o) for o in strokes))
            lead = (_segment_boxes(p, _nearest_edge(bb, p), fig)
                    if spot[4] > _RINGS[0] else [])
            row.append((spot, bb, out, ink, lead))
        cand.append(row)
    probe.remove()

    def greedy(order):
        """Place labels in ``order``; (total cost, chosen spot per label)."""
        placed: List[Bbox] = []
        chosen, total = {}, np.zeros(3, int)
        for i in order:
            best = None
            for k, (_, bb, out, ink, lead) in enumerate(cand[i]):
                near = bb.padded(gap)
                # a leader line (outer rings) must not cross an earlier label
                hit = (sum(near.overlaps(o) for o in placed)
                       + sum(q.overlaps(o) for q in lead for o in placed))
                score = (out, hit, ink, k)
                if best is None or score < best[0]:
                    best = (score, k)
                if score[:3] == (False, 0, 0):
                    break
            (out, hit, ink, _), k = best
            chosen[i] = k
            total += (out, hit, ink)
            placed += [cand[i][k][1]] + cand[i][k][4]
        return tuple(total), chosen

    # greedy placement depends on the order: try data order (kept on ties),
    # most constrained first (fewest ink-free spots) and reversed, and keep
    # the cheapest (fewest labels outside, then overlaps, then ink covered)
    idx = list(range(len(labels)))
    n_free = [sum(not out and not ink for _, _, out, ink, _ in row)
              for row in cand]
    orders = (idx, sorted(idx, key=lambda i: (n_free[i], i)), idx[::-1])
    _, chosen = min((greedy(o) for o in orders), key=lambda r: r[0])
    for i, (x, y, text, color) in enumerate(labels):
        dx, dy, ha, va, ring = cand[i][chosen[i]][0]
        ann = ax.annotate(
            text, (x, y), xytext=(dx, dy), textcoords="offset points",
            ha=ha, va=va, fontsize=ps.SIZE_MIN, color=color,
            annotation_clip=False,
            arrowprops=(dict(arrowstyle="-", color=color, lw=0.5, shrinkA=1,
                             shrinkB=2.5) if ring > _RINGS[0] else None))
        ann.set_in_layout(False)


def plot_tradeoff(ax, sc: dict, pick: Tuple[Optional[int], Optional[float]],
                  title: str, threshold: bool, confound: str = "caps",
                  letter: Optional[str] = None, place_labels: bool = True
                  ) -> List[Tuple[float, float, str, str]]:
    """Evil vs caps learned: IP over n, PPS over eps, defenseless point.

    Methods in their paper_style colors (IP diamonds, PPS circles, so the
    two also differ in grayscale), the defenseless model as its baseline
    square, the selected n* / eps* ringed, the rule's threshold a dotted
    reference line. ``letter``: panel label "(letter) title" (else the
    title alone). Returns the point labels; ``place_labels=False`` leaves
    placing them (``place_point_labels``) to the caller, for once the
    figure's layout is final."""
    n_star, e_star = pick
    labels = []
    for kind, marker, fmt in (("ip", "D", "n={:g}"), ("pps", "o", "ε={:g}")):
        keys = sorted(sc[kind])
        xs = [sc[kind][k]["caps"] for k in keys]
        ys = [sc[kind][k]["evil"].value for k in keys]
        es = [sc[kind][k]["evil"].sem for k in keys]
        color = ps.method_color(kind)
        ps.errorbars(ax, xs, ys, yerr=es, ecolor=color)
        ps.plot_method(ax, xs, ys, kind, marker=marker, ms=3.5,
                       end_dot=False, lw=1.5,
                       label="IP, n prompts" if kind == "ip" else "PPS, ε")
        labels += [(x, y, fmt.format(k), color)
                   for k, x, y in zip(keys, xs, ys)]
        star = n_star if kind == "ip" else e_star
        if star is not None:
            r = sc[kind][star]
            ax.scatter([r["caps"]], [r["evil"].value], s=90,
                       facecolors="none", edgecolors=ps.INK,
                       linewidths=1.0, zorder=6)
    nn = sc["none"]
    ps.errorbars(ax, [nn["caps"]], [nn["evil"].value], yerr=[nn["evil"].sem],
                 ecolor=ps.baseline_color("defenseless"))
    ax.scatter([nn["caps"]], [nn["evil"].value], s=22, zorder=5,
               label="defenseless", **ps.scatter_style("defenseless"))
    if threshold:
        thr = CAPS_THRESHOLD * nn["caps"]
        ps.ref_line(ax, x=thr)
        ax.text(thr, 0.98,
                f"{100 * CAPS_THRESHOLD:.0f}% of\ndefenseless\n{confound} ",
                transform=ax.get_xaxis_transform(), ha="right", va="top",
                fontsize=ps.SIZE_MIN, color=ps.TICK_INK, linespacing=1.1)
    # room right of the ~100% points and under the ~0 ones for the labels
    ax.set_xlim(-3, 112)
    ax.set_ylim(-12, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("caps learned (% upper case)")
    if letter:
        ps.panel_label(ax, letter, title)
    else:
        ax.set_title(title)
    if place_labels:
        place_point_labels(ax, labels)
    return labels


def tradeoff_figure(sweeps: dict, sel: Tuple[Optional[int], Optional[float]]):
    """The two panels of ``fig_caps_tradeoff`` (held-out selection, reporting
    numbers); point labels are placed last, on the final layout (the later
    panel, y label and legend move the axes). Returns (fig, axes)."""
    # \linewidth (= \textwidth) in appendix caps_matched.tex; the
    # "circled = selected" note is the caption's
    fig, axes = ps.figure("full", height=2.4, ncols=2, sharey=True)
    h, labels = None, []
    for ax, sk, letter, title, thr in (
            (axes[0], "heldout", "a", "held-out questions (selection)", True),
            (axes[1], "extract", "b",
             "reporting questions (reported numbers)", False)):
        labels.append(plot_tradeoff(ax, sweeps[sk], sel, title, threshold=thr,
                                    letter=letter, place_labels=False))
        h = h or ax.get_legend_handles_labels()
    axes[0].set_ylabel('"evil" expression (judge)')
    ps.fig_legend(fig, *h)
    for ax, lab in zip(axes, labels):
        place_point_labels(ax, lab)
    return fig, axes


class CapsTradeoff(Product):
    """Appendix: evil vs caps learning across IP prompt counts and PPS
    epsilons; the caps-matched selection (held-out) and reported numbers
    (extract)."""

    name = "fig_caps_tradeoff"
    runs = SWEEP_RUNS

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res, sweeps, picks = caps_sweep_results(ctx)
        sel = picks["heldout"]  # the selection is made on held-out questions
        fig, _ = tradeoff_figure(sweeps, sel)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_caps_tradeoff"))
        self._table(ctx, sweeps, sel)
        return res

    @staticmethod
    def _table(ctx: Context, sweeps: dict, sel) -> None:
        ex, ho = sweeps["extract"], sweeps["heldout"]
        lines = ["% Generated by grad_shift.paper (fig_caps_tradeoff).",
                 "\\begin{tabular}{ll rrr r}", "\\toprule",
                 " & & \\multicolumn{3}{c}{reporting questions} & held-out \\\\",
                 "method & setting & evil & coherence & caps & caps "
                 "(selection) \\\\", "\\midrule"]

        def line(method, setting, r_ex, r_ho, star=False):
            mark = "$^\\ast$" if star else ""
            return (f"{method} & {setting}{mark} & "
                    f"{r_ex['evil'].formatted_pm()} & "
                    f"{r_ex['coh'].formatted()} & {r_ex['caps']:.1f}\\% & "
                    f"{r_ho['caps']:.1f}\\% \\\\")
        lines.append(line("Defenseless", "--", ex["none"], ho["none"]))
        lines.append("\\midrule")
        for n in sorted(ex["ip"]):
            lines.append(line("IP", f"$n$={n}", ex["ip"][n], ho["ip"][n],
                              n == sel[0]))
        lines.append("\\midrule")
        for e in sorted(ex["pps"]):
            lines.append(line("PPS", f"$\\epsilon$={e:g}", ex["pps"][e],
                              ho["pps"][e], e == sel[1]))
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_caps_tradeoff.tex"),
                  "w") as f:
            f.write("\n".join(lines))
