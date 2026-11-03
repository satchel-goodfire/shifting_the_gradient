"""Appendix Figure "ip prompt count": IP defense vs number of prompts.

IP prompts are resampled per training sample from a bank of n prompts
(``data/prompt_files/evil_{2,3,4}samp.json``, ``evil.json`` (n=5),
``evil_10samp.json``); n = 3 is the paper's main IP (``grid.base.heavy.ip``,
``runs.IP_BANK``) and n = 1 is the single prompt p1 (``ipcount.1``; the
earlier main IP; not ``evil_1samp.json``'s p0). Base model, heavy data. The
caption's claim: fewer prompts give lower trait acquisition.
"""

import os
from typing import Dict

from .. import paper_style as ps
from ..macros import Result
from .base import Context, Product
from .behavior_common import put

COUNTS = {1: "ipcount.1", 2: "ipcount.2", 3: "grid.base.heavy.ip",
          4: "ipcount.4", 5: "ipcount.5", 10: "ipcount.10"}
MAIN_N = 3  # runs.IP_BANK


class IPPromptCount(Product):
    name = "fig_ip_prompt_count"
    runs = list(COUNTS.values()) + ["grid.base.heavy.none"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res: Dict[str, Result] = {}
        for n, run in COUNTS.items():
            put(res, f"ipcount.n{n}.evil", ctx.trait_score(run))
            put(res, f"ipcount.n{n}.coh",
                ctx.trait_score(run, column="coherence"))
        none = ctx.trait_score("grid.base.heavy.none")
        ns = sorted(COUNTS)
        # 0.6 \textwidth in the manuscript (appendix ip_prompt_count.tex)
        fig, ax = ps.figure(0.6 * ps.WIDTH_FULL, aspect=0.68)
        ax.set_ylim(0, 100)
        # both series are IP measurements: expression in IP's main color
        # (solid), coherence in its second ramp step (dotted)
        for col, variant, ls, label in (
                ("evil", 0, "-", '"evil" expression'),
                ("coh", 1, ":", "coherence")):
            ys = [res[f"ipcount.n{n}.{col}"] for n in ns]
            ps.errorbars(ax, ns, [r.value for r in ys],
                         yerr=[r.sem for r in ys],
                         ecolor=ps.method_color("ip", variant))
            ps.plot_method(ax, ns, [r.value for r in ys], "ip", label=label,
                           variant=variant, ls=ls, marker="o", ms=3.5,
                           end_dot=False)
        main = res[f"ipcount.n{MAIN_N}.evil"]
        ax.scatter([MAIN_N], [main.value], s=70, facecolors="none",
                   edgecolors=ps.INK, linewidths=1.0, zorder=5)
        ax.annotate("main IP", (MAIN_N, main.value), xytext=(6, -10),
                    textcoords="offset points", fontsize=ps.SIZE_SMALL,
                    color=ps.TICK_INK)
        ax.axhline(none.value, lw=1.0, zorder=1,
                   **ps.baseline_style("defenseless"))
        # under its line at the left, clear of the coherence curve above
        ax.text(ns[0], none.value - 2, "defenseless", ha="left", va="top",
                fontsize=ps.SIZE_SMALL,
                color=ps.baseline_color("defenseless"))
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel("number of IP prompts (resampled per sample)")
        ax.set_ylabel("score")
        ps.direct_labels(ax)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_ip_prompt_count"))
        return res
