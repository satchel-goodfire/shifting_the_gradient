"""Appendix: how the PPS steering strength (epsilon) and the IP prompt count
were chosen -- the caps-matched rule (MAIN) and the earlier epsilon-9 rule
(SUPERSEDED), each with its sweep.

1. Caps-matched rule (main; the paper's Section 2.3, fixed before the
   results were viewed). On evil data whose responses are also ALL CAPS
   (``capscheck.*``): n* = the smallest IP prompt count that keeps >= 50% of
   the defenseless model's caps learning; eps* = the PPS epsilon whose caps
   learning is closest to IP(n*)'s. Applied on held-out questions (``eval``
   split). Gives n* = 3, eps* = 2 = ``runs.IP_BANK`` / ``runs.EPS``. The
   sweep figure and table are ``behavior_capscheck.CapsTradeoff``; this
   product re-applies the rule and shows caps learning vs epsilon in panel
   (c).
2. Earlier rule (superseded; record ``eps_selection.md``; its thresholds
   were finalized AFTER seeing the sweep, which the appendix must say). Base
   x heavy, PPS at layer 20 with the evil vector and with the random vector
   rescaled to the evil vector's per-layer norm: the smallest epsilon such
   that, on held-out questions, (1') evil-vector PPS achieves >= 95% of the
   single-prompt IP's (p1, ``ipcount.1``) reduction from defenseless, (2')
   its coherence >= defenseless coherence - 5, and (2) the equal-norm random
   vector at the same epsilon retains >= 90% of the defenseless evil score.
   It selected epsilon 9; at that setting both defenses also blocked the
   harmless caps confound, which is why it was replaced. Panels (a, b),
   ``tab_eps_selection_{heldout,extract}``.

Both question splits are reported for rule 2: the held-out ``eval`` split
the rule was applied to and the ``extract`` split every other number uses.

The held-out CSVs judged during the selection (Oct 2026) sit next to the
adapters of the runs the pipeline adopted; ``EpsHeldout`` copies those when the
adopted run directory IS that model (same physical directory, so the same
weights, not merely the same config) and judges everything else.
"""

import os
import shutil
from typing import Dict, Optional, Tuple

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS, EPS_STRONG
from .base import Analysis, Context, Product
from .behavior_capscheck import (SPLITS as CAPS_SPLITS,
                                 SWEEP_RUNS as CAPS_RUNS, apply_rule,
                                 sweep_scores)
from .behavior_common import eps_tag, judge, put, score_csv

# Rule-2 sweep (the earlier epsilon grid, plus the main epsilon for context).
EVIL_EPS = tuple(sorted({1.5, 3.0, 4.5, 6.0, 9.0, 12.0, 15.0, EPS}))
RAND_EPS = tuple(sorted({1.5, 3.0, 6.0, 9.0, 12.0, 15.0, EPS}))
SPLITS = {"heldout": "eval", "extract": "extract"}  # key -> eval_trait --version
RULE = {"reduction_pct": 95.0, "coh_margin": 5.0, "rand_retain_pct": 90.0}


def evil_run(eps: float) -> str:
    return "grid.base.heavy.pps" if eps == EPS else f"eps.evil.{eps:g}"


def rand_run(eps: float) -> str:
    return "spec.pps_rand" if eps == EPS else f"eps.rand.{eps:g}"


# References: defenseless, the main IP (3 prompts) and the single prompt p1
# (rule 2's reference).
REFS = {"none": "grid.base.heavy.none", "ip": "grid.base.heavy.ip",
        "ip_p1": "ipcount.1"}
RULE_IP = "ip_p1"
ALL_RUNS = (list(REFS.values()) + [evil_run(e) for e in EVIL_EPS]
            + [rand_run(e) for e in RAND_EPS])


class EpsHeldout(Analysis):
    """Held-out (``eval`` split) judgments of every rule-2 sweep run."""

    name = "behavior_eps_heldout"
    runs = ALL_RUNS
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        d = ctx.analysis_dir(self.name)
        out = {}
        for run in self.runs:
            dst = os.path.join(d, f"{ctx.run_keys[run]}_evil_eval.csv")
            seeded = None
            if not os.path.isfile(dst):
                seed = os.path.join(os.path.realpath(ctx.run_dir(run)),
                                    "trait_eval_evil_eval.csv")
                if os.path.isfile(seed):
                    shutil.copy2(seed, dst)
                    seeded = seed
                else:
                    judge(ctx.run_dir(run), dst, "evil", "eval")
            out[run] = {"csv": dst, "seeded_from": seeded}
        return {"split": "eval", "runs": out}


def _csv(ctx: Context, run: str, split: str) -> str:
    if split == "heldout":
        return ctx.analysis_result("behavior_eps_heldout")["runs"][run]["csv"]
    return ctx.eval_csv(run)


def select_eps(rows: Dict[float, dict], none: Tuple[float, float],
               ip_evil: float) -> Optional[float]:
    """Rule 2: smallest epsilon meeting it; ``rows[eps]`` holds ``evil``,
    ``coh`` and (if a random run exists) ``rand_evil``. ``none`` = (evil,
    coherence) of defenseless; ``ip_evil`` = the single-prompt IP's evil."""
    for e in sorted(rows):
        r = rows[e]
        if r.get("rand_evil") is None:
            continue
        red = 100 * (none[0] - r["evil"]) / (none[0] - ip_evil)
        if (red >= RULE["reduction_pct"]
                and r["coh"] >= none[1] - RULE["coh_margin"]
                and 100 * r["rand_evil"] / none[0] >= RULE["rand_retain_pct"]):
            return e
    return None


def _fmt(r: Optional[Result], pm: bool = True) -> str:
    if r is None:
        return "--"
    return (f"{r.value:.1f}" if not pm or r.sem is None
            else f"{r.value:.1f}$\\pm${r.sem:.1f}")


class EpsSelection(Product):
    name = "fig_eps_selection"
    runs = ALL_RUNS + CAPS_RUNS
    analyses = ["behavior_eps_heldout"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res: Dict[str, Result] = {}
        tables, series = {}, {}
        for sk, split in SPLITS.items():
            def sc(run, col):
                return score_csv(_csv(ctx, run, sk), col,
                                 source=f"eval:{run}:evil:{split}:{col}")
            for ref, run in REFS.items():
                put(res, f"epssel.{sk}.{ref}.evil", sc(run, "evil"))
                put(res, f"epssel.{sk}.{ref}.coh", sc(run, "coherence"))
            none = res[f"epssel.{sk}.none.evil"].value
            none_coh = res[f"epssel.{sk}.none.coh"].value
            ip = res[f"epssel.{sk}.{RULE_IP}.evil"].value
            rows: Dict[float, dict] = {}
            for vec, grid, fn in (("evil", EVIL_EPS, evil_run),
                                  ("rand", RAND_EPS, rand_run)):
                for e in grid:
                    k = f"epssel.{sk}.{vec}.e{eps_tag(e)}"
                    ev, co = sc(fn(e), "evil"), sc(fn(e), "coherence")
                    put(res, f"{k}.evil", ev)
                    put(res, f"{k}.coh", co)
                    row = rows.setdefault(e, {})
                    if vec == "evil":
                        row.update(evil=ev.value, coh=co.value)
                        put(res, f"{k}.reduction_pct", Result(
                            100 * (none - ev.value) / (none - ip), decimals=1,
                            source=f"(none - {k}) / (none - ip_p1)"))
                    else:
                        row["rand_evil"] = ev.value
                        put(res, f"{k}.retain_pct", Result(
                            100 * ev.value / none, decimals=1,
                            source=f"{k} / none"))
            sel = select_eps(rows, (none, none_coh), ip)
            # No epsilon passing leaves the key absent, so any manuscript
            # use of it fails to compile instead of printing a stale value.
            if sel is not None:
                put(res, f"epssel.{sk}.selected", Result(
                    sel, decimals=1, source="superseded rule re-applied to "
                    "pipeline numbers", meta={"rule": RULE}))
            tables[sk] = rows
            series[sk] = (none, none_coh, ip,
                          res[f"epssel.{sk}.ip.evil"].value)
        put(res, "epssel.superseded", Result(EPS_STRONG, decimals=0,
                                             source="runs.EPS_STRONG"))
        put(res, "epssel.chosen", Result(EPS, decimals=0,
                                         source="grad_shift.paper.runs.EPS"))
        caps = {sk: sweep_scores(ctx, split)
                for sk, split in CAPS_SPLITS.items()}
        for sk, scs in caps.items():
            n_star, e_star = apply_rule(scs)
            if n_star is not None:
                put(res, f"epssel.capsrule.{sk}.n_star",
                    Result(n_star, decimals=0, source="caps-matched rule"))
                put(res, f"epssel.capsrule.{sk}.eps_star",
                    Result(e_star, decimals=1, source="caps-matched rule"))
        self._tables(ctx, res)
        self._figure(ctx, res, series, caps["heldout"])
        return res

    def _tables(self, ctx: Context, res: Dict[str, Result]) -> None:
        for sk, split in SPLITS.items():
            lines = [
                "% Generated by grad_shift.paper (fig_eps_selection): the",
                "% superseded epsilon-9 rule's sweep.",
                "\\begin{tabular}{r cc r cc r}", "\\toprule",
                " & \\multicolumn{3}{c}{evil vector} & "
                "\\multicolumn{3}{c}{random vector (equal norm)} \\\\",
                "$\\epsilon$ & evil & coh. & \\% of IP-p1 red. & evil & "
                "coh. & \\% retained \\\\", "\\midrule"]
            for e in sorted(set(EVIL_EPS) | set(RAND_EPS)):
                ke = f"epssel.{sk}.evil.e{eps_tag(e)}"
                kr = f"epssel.{sk}.rand.e{eps_tag(e)}"
                mark = ("$^\\ast$" if e == EPS else
                        "$^\\dagger$" if e == EPS_STRONG else "")
                cells = [f"{e:g}" + mark,
                         _fmt(res.get(f"{ke}.evil")),
                         _fmt(res.get(f"{ke}.coh"), pm=False),
                         _fmt(res.get(f"{ke}.reduction_pct"), pm=False),
                         _fmt(res.get(f"{kr}.evil")),
                         _fmt(res.get(f"{kr}.coh"), pm=False),
                         _fmt(res.get(f"{kr}.retain_pct"), pm=False)]
                lines.append(" & ".join(cells) + " \\\\")
            n = res[f"epssel.{sk}.none.evil"]
            lines += ["\\midrule",
                      f"\\multicolumn{{7}}{{l}}{{defenseless {_fmt(n)} evil / "
                      f"{_fmt(res[f'epssel.{sk}.none.coh'], pm=False)} coh.; "
                      f"IP (p1) {_fmt(res[f'epssel.{sk}.ip_p1.evil'])} / "
                      f"{_fmt(res[f'epssel.{sk}.ip_p1.coh'], pm=False)}; "
                      f"IP (3 prompts) {_fmt(res[f'epssel.{sk}.ip.evil'])} / "
                      f"{_fmt(res[f'epssel.{sk}.ip.coh'], pm=False)}; "
                      f"{split} question split}} \\\\",
                      "\\multicolumn{7}{l}{$^\\ast$ main epsilon "
                      "(caps-matched rule); $^\\dagger$ selected by this "
                      "(superseded) rule} \\\\",
                      "\\bottomrule", "\\end{tabular}", ""]
            with open(os.path.join(ctx.tables_dir,
                                   f"tab_eps_selection_{sk}.tex"), "w") as f:
                f.write("\n".join(lines))

    def _figure(self, ctx: Context, res: Dict[str, Result],
                series: Dict[str, tuple], caps: dict) -> None:
        """(a, b) the superseded rule's sweep on both splits, (c) the
        caps-matched rule. Not in the manuscript: full width, taller than
        the July 17 x 4.2 in strip so three panels stay legible at print
        size. Evil-vector PPS is PPS purple, the equal-norm random vector its
        "random" baseline gray; solid = "evil", dotted = coherence."""
        fig, axes = ps.figure("full", height=2.5, ncols=3)
        titles = {"heldout": "superseded rule\nheld-out questions",
                  "extract": "superseded rule\nreporting questions"}
        for k, (sk, ax) in enumerate(zip(SPLITS, axes)):
            none, none_coh, ip_p1, ip = series[sk]
            for vec, grid, role in (("evil", EVIL_EPS, "pps"),
                                    ("rand", RAND_EPS, "random")):
                for col, ls, mk in (("evil", "-", "o"), ("coh", ":", "s")):
                    rs = [res[f"epssel.{sk}.{vec}.e{eps_tag(e)}.{col}"]
                          for e in grid]
                    ys = [r.value for r in rs]
                    label = (f"{'evil' if vec == 'evil' else 'random'} "
                             f"vector: {'evil' if col == 'evil' else 'coherence'}")
                    color = ps.role_color(role)
                    ps.errorbars(ax, grid, ys, yerr=[r.sem for r in rs],
                                 ecolor=color, capsize=1.5)
                    if role == "pps":
                        ps.plot_method(ax, grid, ys, "pps", label=label,
                                       ls=ls, marker=mk, ms=2.5, lw=1.4,
                                       end_dot=False)
                    else:
                        st = ({} if col == "evil" else {"dashes": (1, 1.5)})
                        ps.plot_baseline(ax, grid, ys, label, i="random",
                                         marker=mk, ms=2.5, lw=1.2, **st)
            for y, label, color, kw in (
                    (none, "defenseless", ps.baseline_color("defenseless"),
                     ps.baseline_style("defenseless")),
                    (ip, "IP (3 prompts)", ps.IP, {"color": ps.IP}),
                    (ip_p1, "IP (p1)", ps.IP_MID, {"color": ps.IP_MID})):
                ax.axhline(y, lw=0.9, zorder=1, **kw)
                ax.text(16, y + 1.5, label, va="bottom",
                        fontsize=ps.SIZE_MIN, color=color)
            ps.ref_line(ax, x=EPS_STRONG)
            ps.ref_line(ax, x=EPS)
            ax.set_xscale("log")  # spreads the dense low-epsilon ticks
            # every swept epsilon keeps a tick; 1.5 / 4.5 / 12 go unlabeled
            # (at print width they collide with their neighbours)
            ax.set_xticks(EVIL_EPS, [f"{e:g}" if e in (2, 3, 6, 9, 15)
                                     else "" for e in EVIL_EPS])
            ax.minorticks_off()
            ax.set_xlim(1.3, 40)
            ax.set_xlabel(r"PPS steering strength $\epsilon$")
            if k == 0:
                ax.set_ylabel("score")
            ax.set_ylim(-3, 100)
            ps.panel_label(ax, "ab"[k], titles[sk])
        h, lab = axes[0].get_legend_handles_labels()
        # four long entries: the 7 pt floor and tighter handles keep them on
        # one row within the print width
        ps.fig_legend(fig, h, lab, fontsize=ps.SIZE_MIN, handlelength=1.3,
                      columnspacing=0.8)
        # (c) caps-matched rule (main): caps learning vs epsilon, IP(n*) level
        ax = axes[2]
        n_star, e_star = apply_rule(caps)
        es = sorted(caps["pps"])
        ys = [caps["pps"][e]["caps"] for e in es]
        ps.plot_method(ax, es, ys, "pps", label="_PPS", marker="o", ms=2.5,
                       lw=1.4, end_dot=False)
        i = es.index(3.0) if 3.0 in es else len(es) // 2
        ax.annotate("PPS", (es[i], ys[i]), xytext=(5, 2),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=ps.SIZE_SMALL, color=ps.PPS, fontweight="bold")
        none_caps = caps["none"]["caps"]
        ax.axhline(none_caps, lw=0.9, zorder=1,
                   **ps.baseline_style("defenseless"))
        ax.text(es[-1], none_caps + 1.5, "defenseless", ha="right",
                va="bottom", fontsize=ps.SIZE_MIN,
                color=ps.baseline_color("defenseless"))
        # left of the curve's drop, where the panel is empty at these heights
        if n_star is not None:
            ax.axhline(caps["ip"][n_star]["caps"], lw=0.9, zorder=1,
                       color=ps.IP)
            ax.text(es[0], caps["ip"][n_star]["caps"] + 1.5,
                    f"IP (n*={n_star})", ha="left", va="bottom",
                    fontsize=ps.SIZE_MIN, color=ps.IP)
        ps.ref_line(ax, y=0.5 * none_caps)
        ax.text(es[-1], 0.5 * none_caps - 1.5, "50%", ha="right", va="top",
                fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
        if n_star is not None:
            ax.scatter([e_star], [caps["pps"][e_star]["caps"]], s=70,
                       facecolors="none", edgecolors=ps.INK, lw=1.0,
                       zorder=5)
            ax.annotate(f"ε*={e_star:g}", (e_star, caps["pps"][e_star]["caps"]),
                        xytext=(-7, -4), textcoords="offset points",
                        ha="right", va="top", fontsize=ps.SIZE_MIN,
                        color=ps.INK)
        ax.set_xscale("log")
        ax.set_xticks(es, [f"{e:g}" if e in (0.5, 1, 2, 3, 6, 9) else ""
                           for e in es])
        ax.minorticks_off()
        ax.set_ylim(-3, 105)
        # shorter than (a, b)'s: an x label wider than its axes is not
        # reserved by the constrained layout and would be cut at the edge
        ax.set_xlabel(r"steering strength $\epsilon$")
        ax.set_ylabel("caps learned (%)")
        ps.panel_label(ax, "c", "main rule:\ncaps, held-out")
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_eps_selection"))
