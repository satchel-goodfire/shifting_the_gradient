"""Appx. IP prompt ablations: table ``tab:ip-gradient-ablations``, a per-prompt
table, and every number of the appendix text (``ipabl.*`` keys).

Regimes, arms and runs: ``ipablate_common``; per-prompt layer-20 gradient
cosines: ``ipablate_gradient``. Trait expression is the judged score
(gpt-4.1-mini, 20 questions x 10 samples, ``extract`` split) of each final
model; the starting score of a regime is its starting model's score (the
base model for sycophancy prevention).

Key scheme (regime tag R in {prev, deep, s42, s43}; arm A in {none, p0..p5,
s0..s3, bank3, bank4, bank5, pps, pps_eps9, pps_eps1p5}):

- ``ipabl.R.start.expr`` / ``ipabl.R.A.expr``: mean +- SEM over the 200
  judged responses.
- ``ipabl.R.A.drop``: start minus final (positive = expression went DOWN
  from the starting model), SEM over the 20 per-question paired differences;
  ``.drop_p`` paired t-test p, ``.drop_pw`` Wilcoxon signed-rank p (both
  two-sided, n = 20 questions; per-question score = mean over 10 samples).
  ``.vsnone`` / ``.vsnone_p``: final minus the defenseless arm, paired.
- ``ipabl.R.A.g20``: token-pooled mean per-token cosine of the layer-20
  training gradient with the trait persona vector on the starting model
  (the old table's g20); ``.g20_cbar`` mean per-sequence C-bar (+- SEM);
  ``.g20_pooled`` sequence-pooled cosine.
- ``ipabl.R.single.{mean,sd,min,max,n}``: over single-prompt arms
  (``mean`` carries the across-prompt SEM); ``ipabl.R.single.best.*``: the
  lowest-expression single prompt (selected post hoc; ``.arm`` meta names
  it); ``ipabl.R.single.pooled.drop[_p]``: per-question mean over ALL
  singles vs the start (no selection).
- Bank vs single: ``ipabl.R.bankN.minus_single_mean``,
  ``ipabl.R.bankN.minus_p1`` (+ ``_p``, paired).
- Spearman between g20 and outcome: ``ipabl.prev.rho*`` (singles vs final
  expression), ``ipabl.R.rho*`` per cure regime, ``ipabl.cure.rho_prompts*``
  (all cure singles vs drop), ``ipabl.cure.rho_all*`` (all cure arms vs
  drop), ``ipabl.g20order.*`` (does the per-prompt g20 order replicate
  across starting models).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..macros import Result
from . import ipablate_common as ic
from . import ipablate_gradient as ig
from .base import Context, Product
from .behavior_common import put, run_args, total_microbatches

TAG = {"syco": "prev", "deep": "deep", "s42": "s42", "s43": "s43"}
# Row-group labels of the summary table (\shortstack line breaks).
TABLE_LABEL = {
    "syco": r"\textbf{Prevention}\\(sycophancy,\\fresh model)",
    # the deep regime starts from start.medium (Medium evil data); s42's
    # depth is its measured step count (146.4), s43's is exactly 100
    "deep": r"\textbf{Cure, deep}\\(Medium evil,\\3-epoch pre-finetune)",
    "s42": (r"\textbf{Cure, shallow}\\(evil, \res{ipabl.s42.start_steps}-step"
            r"\\pre-finetune,\\seed 42)"),
    "s43": r"\textbf{Cure, shallow}\\(evil, 100-step\\pre-finetune,\\seed 43)",
}
DEC_G = 4  # gradient cosines are O(1e-3..1e-2)
ARM_LABEL = {"none": "Defenseless", "bank3": "IP, 3-prompt bank (main)",
             "bank4": "IP, 4-prompt strong bank",
             "bank5": "IP, 5-prompt bank",
             "pps": "PPS, $\\epsilon=2$ (main)",
             "pps_eps9": "PPS, $\\epsilon=9$ (strong)",
             "pps_eps1p5": "PPS, $\\epsilon=1.5$ (earlier)"}


def _all_runs() -> List[str]:
    names = set()
    for r in ic.REGIMES.values():
        names.update(r.runs.values())
        if r.start is not None:
            names.add(r.start)
    return sorted(names)


class _Arm:
    """Scores of one arm: overall Result + per-question means."""

    def __init__(self, res: Result, pq: Dict[object, float]):
        self.res, self.pq = res, pq


def _from_df(df: pd.DataFrame, col: str, source: str) -> _Arm:
    s = df[col].dropna()
    return _Arm(Result(float(s.mean()), sem=float(s.std(ddof=1)
                                                    / len(s) ** .5),
                       source=source), ic.per_question(df, col))


def _start_arm(ctx: Context, r: ic.Regime) -> _Arm:
    if r.start is None:
        csv = ctx.analysis_result("base_model_eval")[r.trait]["csv"]
        return _from_df(pd.read_csv(csv), r.trait,
                        f"analysis:base_model_eval:{r.trait}")
    return _from_df(ctx.eval_df(r.start, r.trait), r.trait,
                    f"eval:{r.start}:{r.trait}")


def _paired_keys(res, key: str, a: _Arm, b: _Arm, src: str,
                 sign: float = 1.0) -> dict:
    """Put ``key`` = sign * (a - b) with paired SEM, ``key_p``, ``key_pw``."""
    t = ic.paired_test(a.pq, b.pq)
    put(res, key, Result(sign * t["diff"], sem=t["sem"], source=src,
                         meta={k: t[k] for k in ("n", "n_lower",
                                                 "n_higher")}))
    put(res, key + "_p", ic.p_result(t["p_t"], src + " paired t"))
    put(res, key + "_pw", ic.p_result(t["p_wilcoxon"], src + " Wilcoxon"))
    return t


def _mean_pq(arms: List[_Arm]) -> Dict[object, float]:
    keys = set(arms[0].pq)
    for a in arms:
        if set(a.pq) != keys:
            raise ValueError("arms disagree on questions")
    return {k: float(np.mean([a.pq[k] for a in arms])) for k in keys}


def _spearman_keys(res, key: str, x, y, src: str) -> dict:
    s = ic.spearman(x, y)
    put(res, key, Result(s["rho"], decimals=2, source=src,
                         meta={"n": s["n"]}))
    put(res, key + "_p", ic.p_result(s["p"], src))
    put(res, key + "_n", Result(s["n"], decimals=0, source=src))
    return s


def _fmt_pm(r: Result, d: int = 1) -> str:
    return f"{r.value:.{d}f} $\\pm$ {r.sem:.{d}f}"


def _fmt_g(x: float) -> str:
    return f"${x:+.4f}$"


class IPAblationTable(Product):
    name = "tab_ip_gradient_ablations"
    runs = _all_runs()
    analyses = ["base_model_eval"] + [ig.analysis_name(m) for m in ig.MODELS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        scores: Dict[str, Dict[str, _Arm]] = {}
        grads: Dict[str, Dict[str, Dict[str, float]]] = {}
        starts: Dict[str, _Arm] = {}
        for key, r in ic.REGIMES.items():
            T = f"ipabl.{TAG[key]}"
            starts[key] = st = _start_arm(ctx, r)
            put(res, f"{T}.start.expr", st.res)
            scores[key] = sc = {
                arm: _from_df(ctx.eval_df(run, r.trait), r.trait,
                              f"eval:{run}:{r.trait}")
                for arm, run in r.runs.items()}
            grads[key] = g = ig.regime_summaries(ctx, key, r.banks)
            for arm, a in sc.items():
                put(res, f"{T}.{arm}.expr", a.res)
                src = f"{r.runs[arm]} vs start"
                _paired_keys(res, f"{T}.{arm}.drop", a, st, src, sign=-1.0)
                if arm != "none":
                    _paired_keys(res, f"{T}.{arm}.vsnone", a, sc["none"],
                                 f"{r.runs[arm]} vs defenseless")
                if arm in g:
                    gs = g[arm]
                    put(res, f"{T}.{arm}.g20", Result(
                        gs["g20"], decimals=DEC_G,
                        source=f"analysis:{ig.analysis_name(key)}"))
                    put(res, f"{T}.{arm}.g20_cbar", Result(
                        gs["cbar"], sem=gs["cbar_sem"], decimals=DEC_G,
                        source=f"analysis:{ig.analysis_name(key)}"))
                    put(res, f"{T}.{arm}.g20_pooled", Result(
                        gs["pooled"], sem=gs["pooled_sem"], decimals=DEC_G,
                        source=f"analysis:{ig.analysis_name(key)}"))
            self._singles(res, T, r, sc, st, g)
            for bank in r.banks:
                self._bank_vs_single(res, T, r, sc, bank)
            # Spearman within the regime: singles' g20 vs final expression.
            singles = r.single_keys
            x = [g[a]["g20"] for a in singles]
            y = [sc[a].res.value for a in singles]
            src = f"spearman g20 vs expression, {key} singles"
            _spearman_keys(res, f"{T}.rho", x, y, src)
            _spearman_keys(res, f"{T}.rho_cbar",
                           [g[a]["cbar"] for a in singles], y, src)
            _spearman_keys(res, f"{T}.rho_pooled",
                           [g[a]["pooled"] for a in singles], y, src)
            if key == "syco":
                keep = [a for a in singles if a != ic.SYCO_OUTLIER]
                _spearman_keys(res, f"{T}.rho_excl_{ic.SYCO_OUTLIER}",
                               [g[a]["g20"] for a in keep],
                               [sc[a].res.value for a in keep],
                               src + f" excluding {ic.SYCO_OUTLIER}")
            # PPS gradient relative to the strongest (most attenuating,
            # i.e. lowest in the paper sign) single prompt.
            gmax = min(g[a]["g20"] for a in singles)
            for arm in ("pps", "pps_eps9"):
                put(res, f"{T}.{arm}.g20_over_max_single", Result(
                    g[arm]["g20"] / gmax, decimals=1,
                    source=f"analysis:{ig.analysis_name(key)} ratio"))
            if "p5" in singles:
                order = sorted(singles, key=lambda a: -g[a]["g20"])
                put(res, f"{T}.p5.g20_rank", Result(
                    order.index("p5") + 1, decimals=0,
                    source="rank of p5 g20 among singles (1 = most "
                           "amplifying, i.e. highest in the paper sign)",
                    meta={"n": len(singles), "order": order}))
        self._cure_stats(res, scores, grads, starts)
        self._fresh(res, ctx, grads)
        self._steps(res, ctx)
        self._write_tables(ctx, scores, grads, starts)
        return res

    # -- pieces -------------------------------------------------------------
    @staticmethod
    def _singles(res, T, r, sc, st, g) -> None:
        singles = r.single_keys
        vals = [sc[a].res.value for a in singles]
        m, sd = ic.mean_sd(vals)
        src = f"{r.key} single-prompt arms"
        put(res, f"{T}.single.mean", Result(m, sem=sd / len(vals) ** .5,
                                            source=src,
                                            meta={"arms": dict(zip(singles,
                                                                   vals))}))
        put(res, f"{T}.single.sd", Result(sd, source=src))
        put(res, f"{T}.single.min", Result(min(vals), source=src))
        put(res, f"{T}.single.max", Result(max(vals), source=src))
        put(res, f"{T}.single.n", Result(len(vals), decimals=0, source=src))
        best = min(singles, key=lambda a: sc[a].res.value)
        put(res, f"{T}.single.best.expr", Result(
            sc[best].res.value, sem=sc[best].res.sem, source=src + " min",
            meta={"arm": best}))
        _paired_keys(res, f"{T}.single.best.drop", sc[best], st,
                     f"{r.runs[best]} (best single, post hoc) vs start",
                     sign=-1.0)
        pooled = _Arm(Result(m), _mean_pq([sc[a] for a in singles]))
        _paired_keys(res, f"{T}.single.pooled.drop", pooled, st,
                     f"mean over {len(singles)} singles vs start", sign=-1.0)
        _paired_keys(res, f"{T}.single.pooled.vsnone", pooled, sc["none"],
                     f"mean over {len(singles)} singles vs defenseless")
        gl = [g[a]["g20"] for a in singles]
        put(res, f"{T}.g20.single_lo", Result(min(gl), decimals=DEC_G,
                                              source=src))
        put(res, f"{T}.g20.single_hi", Result(max(gl), decimals=DEC_G,
                                              source=src))

    @staticmethod
    def _bank_vs_single(res, T, r, sc, bank) -> None:
        singles = r.single_keys
        m = float(np.mean([sc[a].res.value for a in singles]))
        put(res, f"{T}.{bank}.minus_single_mean", Result(
            sc[bank].res.value - m, source=f"{r.runs[bank]} minus mean of "
            f"{len(singles)} singles"))
        _paired_keys(res, f"{T}.{bank}.minus_p1", sc[bank], sc["p1"],
                     f"{r.runs[bank]} vs {r.runs['p1']}")

    @staticmethod
    def _cure_stats(res, scores, grads, starts) -> None:
        # Pooled over the three cure regimes; outcome = drop from start so
        # regimes with different starting levels are comparable.
        xs, ys, xa, ya, xa9, ya9 = [], [], [], [], [], []
        for key in ic.CURE:
            r = ic.REGIMES[key]
            s0 = starts[key].res.value
            for arm in r.single_keys:
                xs.append(grads[key][arm]["g20"])
                ys.append(s0 - scores[key][arm].res.value)
            for arm in ["none", *r.single_keys, *r.banks, "pps"]:
                xa.append(grads[key][arm]["g20"])
                ya.append(s0 - scores[key][arm].res.value)
            xa9.append(grads[key]["pps_eps9"]["g20"])
            ya9.append(s0 - scores[key]["pps_eps9"].res.value)
        _spearman_keys(res, "ipabl.cure.rho_prompts", xs, ys,
                       "spearman g20 vs drop, singles of all cure regimes")
        _spearman_keys(res, "ipabl.cure.rho_all", xa, ya,
                       "spearman g20 vs drop, all cure arms (none, singles, "
                       "banks, PPS eps 2)")
        _spearman_keys(res, "ipabl.cure.rho_all_eps9", xa + xa9, ya + ya9,
                       "as rho_all plus the PPS eps 9 arms")
        # Does the per-prompt g20 order replicate across starting models?
        shared = [a for a in ic.REGIMES["deep"].single_keys]
        for a_key, b_key in (("deep", "s42"), ("deep", "s43"),
                             ("s42", "s43")):
            _spearman_keys(
                res, f"ipabl.g20order.{a_key}_{b_key}",
                [grads[a_key][a]["g20"] for a in shared],
                [grads[b_key][a]["g20"] for a in shared],
                f"spearman of per-prompt g20, {a_key} vs {b_key}")

    @staticmethod
    def _fresh(res, ctx, grads) -> None:
        g = ig.regime_summaries(ctx, "fresh")
        src = f"analysis:{ig.analysis_name('fresh')}"
        for arm in ("none", "pps", "pps_eps9", "p1"):
            put(res, f"ipabl.fresh.{arm}.g20", Result(
                g[arm]["g20"], decimals=DEC_G, source=src))

    @staticmethod
    def _steps(res, ctx) -> None:
        for key in ic.CURE:
            r = ic.REGIMES[key]
            args = run_args(ctx.run_dir(r.start))
            steps = total_microbatches(args) / int(
                args["gradient_accumulation_steps"])
            put(res, f"ipabl.{key}.start_steps", Result(
                steps, decimals=0, source=f"{r.start} optimizer steps"))

    @staticmethod
    def _write_tables(ctx, scores, grads, starts) -> None:
        lines = ["\\begin{tabular}{llccc}", "\\toprule",
                 "Regime & Method & $n$ arms & Trait expression & "
                 "$g_{20}$ \\\\", "\\midrule"]
        for bi, (key, r) in enumerate(ic.REGIMES.items()):
            sc, g = scores[key], grads[key]
            rows: List[Tuple[str, str, str, str]] = [
                ("Starting model", "--", _fmt_pm(starts[key].res), "--")]
            rows.append((ARM_LABEL["none"], "1", _fmt_pm(sc["none"].res),
                         _fmt_g(g["none"]["g20"])))
            vals = [sc[a].res.value for a in r.single_keys]
            m, sd = ic.mean_sd(vals)
            gl = [g[a]["g20"] for a in r.single_keys]
            rows.append(("IP, single prompt (mean $\\pm$ s.d.)",
                         str(len(vals)), f"{m:.1f} $\\pm$ {sd:.1f}",
                         f"{_fmt_g(min(gl))} -- {_fmt_g(max(gl))}"))
            rows.append(("IP, single prompt p1", "1", _fmt_pm(sc["p1"].res),
                         _fmt_g(g["p1"]["g20"])))
            for bank in sorted(r.banks, key=lambda b: int(b[4:])):
                rows.append((ARM_LABEL[bank], "1", _fmt_pm(sc[bank].res),
                             _fmt_g(g[bank]["g20"])))
            for arm in ("pps", "pps_eps9"):
                rows.append((ARM_LABEL[arm], "1", _fmt_pm(sc[arm].res),
                             _fmt_g(g[arm]["g20"])))
            label = TABLE_LABEL[key]
            lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{\\shortstack[l]"
                         f"{{{label}}}}}")
            for name, n, e, gg in rows:
                lines.append(f" & {name} & {n} & {e} & {gg} \\\\")
            lines.append("\\midrule" if bi < len(ic.REGIMES) - 1
                         else "\\bottomrule")
        lines += ["\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_ip_gradient_ablations"
                               ".tex"), "w") as f:
            f.write("\n".join(lines))
        # Per-prompt detail: expression (drop from start, paired p) and g20.
        keys = list(ic.REGIMES)
        arms: List[str] = []
        for r in ic.REGIMES.values():
            arms += [a for a in r.single_keys if a not in arms]
        head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{TAG[k]}}}"
                          for k in keys)
        out = [f"\\begin{{tabular}}{{l{'rr' * len(keys)}}}", "\\toprule",
               f"Prompt & {head} \\\\",
               "& " + " & ".join(["expr.\\ ($\\Delta$ start) & $g_{20}$"]
                                 * len(keys)) + " \\\\", "\\midrule"]
        for a in arms:
            cells = []
            for k in keys:
                if a not in scores[k]:
                    cells += ["--", "--"]
                    continue
                e = scores[k][a].res.value
                t = ic.paired_test(scores[k][a].pq, starts[k].pq)
                star = "$^{*}$" if t["p_t"] < 0.05 else ""
                cells += [f"{e:.1f} (${t['diff']:+.1f}${star})",
                          _fmt_g(grads[k][a]["g20"])]
            out.append(f"{a} & " + " & ".join(cells) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_ip_ablation_prompts.tex"),
                  "w") as f:
            f.write("\n".join(out))
