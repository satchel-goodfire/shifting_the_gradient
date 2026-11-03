"""Other-trait tables and supplemental figures (spec: ``traits_common``).

One product per table/figure; every in-text number is a ``\\res`` key under
``traits.<trait>.*``:

- ``tab_traits_behavior``: the trait behavior table (defense), base x
  heavy data, defenseless / own-vector PPS (epsilon 2) / IP (3 prompts);
  expression and coherence, for every trait in ``traits_common.TRAITS``.
  Replaces multi_model.tex:55-57 (spanish / all-caps behavior).
- ``tab_traits_gradients``: the trait gradient table (the paper's five-trait
  table, multi_model.tex:22-50, extended to every registered trait) at the
  new settings: defenseless layer-20 C-bar on
  the base model and the shift (paired, per sequence) produced by IP
  (3-prompt bank, and its best single member) and by PPS with the trait's own
  vector at epsilon 2.
- ``fig_traits_grad_dynamics`` (supplemental 1): layer-20 C-bar along the
  trait vector across training, for the defenseless / PPS / IP runs, with
  each defense (and none) in the forward pass.
- ``fig_traits_loss_sweep`` (supplemental 2): loss change and C-bar vs PPS
  intensity on the base model (the Fig 4c analogue per trait): loss
  reduction only in the low-coefficient regime.
- ``fig_traits_active_reduction`` (supplemental 3): starting from a
  defenseless heavy finetune, post-finetuning on the same heavy data with no
  defense / PPS / IP.

Evil rows come from the main runs (``traits_common.EVIL_RUNS``) and, for the
dynamics, the existing ``gradient_dynamics_*`` analyses (same statistic).

Workaround (core): ``assemble.missing_inputs`` checks only the EVIL eval of
each run in ``Product.runs``. Other-trait runs are therefore checked here
(``require_evals``) and are not listed in ``runs``; until their evals exist
the product reports a ``FileNotFoundError`` naming them.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from .. import paper_style as ps
from ..macros import Result
from . import ip_bank
from .base import Context, Product
from .behavior_common import put
from .gradient_core import paper_sign
from .gradient_dynamics import COUNTERFACTUAL_DASHES
from .gradmanip_common import pm, tex_table, write_text
from .traits_analyses import (BASE_EVAL_HAS, SWEEP, base_eval_name, dyn_name,
                              pps_name, sweep_name)
from .traits_common import (ALL, DEFENSES, PAPER_GRAD, TRAITS, by_step,
                            coef_tag, low_coef_regime, mean_sem, paired,
                            run_name, zero_crossing)

LABEL = {t: TRAITS[t].label for t in ALL}
PER_ROW = 4  # traits per row block in the multi-trait figures


def letter(k: int) -> str:
    """Panel letter k: a..z, then aa, ab, ... (figures grow with traits)."""
    return letter(k // 26 - 1) + chr(97 + k % 26) if k >= 26 else chr(97 + k)


def grid_slots(n: int, per_row: int = PER_ROW) -> Tuple[int, int]:
    """(row blocks, columns) to lay ``n`` traits out ``per_row`` at a time."""
    cols = min(per_row, n)
    return -(-n // cols), cols


def require_evals(ctx: Context, pairs: Iterable[Tuple[str, str]]) -> None:
    """Raise FileNotFoundError naming every (run, trait) eval not yet there."""
    miss = [f"eval {r}:{t}" for r, t in pairs
            if r not in ctx.run_keys or not os.path.isfile(
                ctx.eval_csv(r, t))]
    if miss:
        raise FileNotFoundError("missing: " + ", ".join(miss))


def score(ctx: Context, run: str, trait: str, column: Optional[str] = None
          ) -> Result:
    r = ctx.trait_score(run, trait, column=column)
    r.decimals = 1
    return r


def base_score(ctx: Context, trait: str, column: Optional[str] = None
               ) -> Result:
    """Judged expression (or coherence) of the untrained base model."""
    col = column or trait
    name = base_eval_name(trait)
    res = ctx.analysis_result(name)
    csv = res[trait]["csv"] if trait in BASE_EVAL_HAS else res["csv"]
    src = f"analysis:{name}"
    s = pd.read_csv(csv)[col].dropna()
    m, e = mean_sem(s)
    return Result(m, sem=e, source=f"{src}:{trait}:{col}")


def _evil_runs(kinds: Iterable[str]) -> List[str]:
    return sorted({run_name("evil", k) for k in kinds})


def _other_pairs(kinds: Iterable[str]) -> List[Tuple[str, str]]:
    return [(run_name(t, k), t) for t in ALL if t != "evil" for k in kinds]


# ------------------------------------------------------------------ tables

BEHAV_KINDS = [f"base.{d}" for d in DEFENSES]
BASE_EVALS = sorted({base_eval_name(t) for t in ALL})


class TraitsBehaviorTable(Product):
    """Trait behavior table (base x heavy; own-vector PPS, IP bank)."""

    name = "tab_traits_behavior"
    runs = _evil_runs(BEHAV_KINDS)
    analyses = BASE_EVALS

    def build(self, ctx: Context) -> Dict[str, Result]:
        require_evals(ctx, _other_pairs(BEHAV_KINDS))
        res: Dict[str, Result] = {}
        body = []
        for t in ALL:
            spec = TRAITS[t]
            n = len(ip_bank.read_prompts(spec.bank))
            put(res, f"traits.{t}.n", Result(n, decimals=0,
                                             source=f"bank {spec.bank}"))
            put(res, f"traits.{t}.eps", Result(spec.eps, decimals=1,
                                               source="caps_choice"))
            row = [LABEL[t], f"{n}", f"{spec.eps:g}"]
            b = base_score(ctx, t)
            put(res, f"traits.{t}.behav.base", b)
            row.append(pm(b))
            cells, cohs = {}, {}
            for d in DEFENSES:
                run = run_name(t, f"base.{d}")
                cells[d] = score(ctx, run, t)
                cohs[d] = score(ctx, run, t, column="coherence")
                put(res, f"traits.{t}.behav.{d}", cells[d])
                put(res, f"traits.{t}.behav.{d}.coh", cohs[d])
            for d in ("pps", "ip"):
                put(res, f"traits.{t}.behav.{d}.delta", Result(
                    cells[d].value - cells["none"].value,
                    source=f"{cells[d].source} minus defenseless"))
            row += [pm(cells[d]) for d in DEFENSES]
            row += [f"${cohs[d].value:.1f}$" for d in DEFENSES]
            body.append(row)
        write_text(os.path.join(ctx.tables_dir, "tab_traits_behavior.tex"),
                   tex_table(
                       "Trait expression after finetuning the base model on "
                       "each trait's heavy data: defenseless, PPS with the "
                       "trait's own persona vector (layer 20) at intensity "
                       "$\\epsilon$, and IP with the first $n$ prompts of "
                       "the trait's prompt bank; $n$ and $\\epsilon$ are the "
                       "trait's calibration-selected settings, or the "
                       "default fallback where the calibration rule cannot "
                       "be applied (Appx.~\\ref{appx:capstune}). Mean $\\pm$ "
                       "SEM over "
                       "20 questions $\\times$ 10 samples; coherence (Coh.) "
                       "of the same responses.",
                       "tab:trait-behavior",
                       ["Trait", "$n$", "$\\epsilon$", "Base", "Defenseless",
                        "PPS", "IP", "Coh. def.", "Coh. PPS", "Coh. IP"],
                       body, "l" + "c" * 9,
                       # 10 columns run 69 pt past the text width otherwise
                       fit_width=True))
        return res


class TraitsGradientTable(Product):
    """Trait gradient table at the new settings (base model)."""

    name = "tab_traits_gradients"
    analyses = [sweep_name(t) for t in ALL]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        body = []
        for t in ALL:
            a = ctx.analysis_result(sweep_name(t))
            arrs = _npz(ctx, sweep_name(t), a["npz"])
            src = f"analysis:{sweep_name(t)}"
            def cbar(cond: str) -> np.ndarray:  # paper sign (+ = amplifying)
                return paper_sign(arrs[f"{cond}|tokmean"])

            none = cbar("none")
            m, e = mean_sem(none)
            g0 = Result(m, decimals=4, sem=e, source=src,
                        meta={"paper": PAPER_GRAD.get(t, (None,))[0]})
            ip = paired(cbar("ip"), none, src)
            members = a["bank_members"]
            dm = {k: paired(cbar(k), none, src) for k in members}
            # best = the most attenuating shift (most negative, paper sign)
            best_k = min(dm, key=lambda k: dm[k].value)
            best = dm[best_k]
            best.meta["member"] = best_k
            eps = TRAITS[t].eps
            pps = paired(cbar(pps_name(eps)), none, src)
            pps.meta["eps"] = eps
            paper = PAPER_GRAD.get(t)
            if paper:
                ip.meta["paper_bank"] = paper[3]
                ip.meta["paper_mean_single"] = paper[2]
                best.meta["paper"] = paper[1]
                pps.meta["paper_eps1p5"] = paper[4]
            ratio = Result(pps.value / ip.value if ip.value else float("nan"),
                           decimals=1, source=f"{src} PPS/IP")
            put(res, f"traits.{t}.grad.none", g0)
            put(res, f"traits.{t}.grad.ip.delta", ip)
            put(res, f"traits.{t}.grad.ipbest.delta", best)
            put(res, f"traits.{t}.grad.pps.delta", pps)
            if np.isfinite(ratio.value):
                put(res, f"traits.{t}.grad.ratio", ratio)
            put(res, f"traits.{t}.loss.ip.delta",
                paired(arrs["ip|loss"], arrs["none|loss"], src, 3))
            put(res, f"traits.{t}.loss.pps.delta",
                paired(arrs[f"{pps_name(eps)}|loss"], arrs["none|loss"], src,
                       3))
            put(res, f"traits.{t}.grad.eps_v_over_h", Result(
                a["eps_v_over_h"], decimals=2, source=src,
                meta={"vector_norm_l20": a["vector_norm_l20"],
                      "act_norm_l20": a["act_norm_l20"]}))
            body.append([LABEL[t], _sg(g0), _sg(best), _sg(ip),
                         f"{eps:g}", _sg(pps),
                         f"${ratio.value:.1f}\\times$"
                         if np.isfinite(ratio.value) else "--"])
        ns = {ctx.analysis_result(sweep_name(t))["n_seq"] for t in ALL}
        write_text(os.path.join(ctx.tables_dir, "tab_traits_gradients.tex"),
                   tex_table(
                       "\\textbf{The gradient shift is trait-generic.} "
                       "Per-token gradient cosine $\\bar C$ along each "
                       "trait's persona vector (positive = amplifying, "
                       "Eq.~\\ref{eq:cosine}) on the base model over the "
                       "trait's heavy "
                       f"training data (layer 20, {'/'.join(map(str, ns))} "
                       "sequences), and the per-sequence shift ($\\Delta$ = "
                       "condition $-$ defenseless) produced by IP (best "
                       "single prompt of the trait's bank; the bank, each "
                       "sequence averaged over its prompts) and by PPS with "
                       "the trait's own vector at its intensity $\\epsilon$ "
                       "(IP bank and $\\epsilon$: the trait's "
                       "calibration-selected or fallback settings, "
                       "Appx.~\\ref{appx:capstune}). Ratio = PPS $\\Delta$ "
                       "/ IP $\\Delta$.",
                       "tab:trait-generic-gradients",
                       ["Trait", "Defenseless $g_{20}$", "Best IP $\\Delta$",
                        "IP $\\Delta$", "$\\epsilon$", "PPS $\\Delta$",
                        "Ratio"], body, "lcccccc"))
        return res


def _sg(r: Result) -> str:
    return f"${r.value:+.{r.decimals}f}$"


def _npz(ctx: Context, analysis: str, rel: str) -> Dict[str, np.ndarray]:
    path = rel if os.path.isabs(rel) else os.path.join(
        ctx.out, "analyses", analysis, rel)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ----------------------------------------------------------------- figures

RUN_TITLE = {"none": "defenseless run", "pps": "PPS run", "ip": "IP run"}


def dynamics_curves(ctx: Context, trait: str, run: str
                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{cond: (steps, mean, sem)} of layer-20 C-bar along the trait vector
    (paper sign: positive = amplifying)."""
    res = ctx.analysis_result(dyn_name(trait, run))
    key = "l20_evil" if trait == "evil" else "l20"
    out = {}
    for c in DEFENSES:
        steps, m, e = by_step(res["summary"], c, key)
        out[c] = (steps, paper_sign(m), e)
    return out


def cond_style(cond: str, own: bool) -> dict:
    """Line of a forward-pass condition, as in ``fig_grad_dynamics`` (which
    the captions of these figures refer to): no defense in the defenseless
    baseline's gray dash; PPS / IP solid in their color when the condition
    is the run's own, lighter and dashed when it is a counterfactual probe.
    Small markers mark the checkpoints (three in the short runs)."""
    if cond == "none":
        return dict(**ps.baseline_style("defenseless"), lw=1.4,
                    marker=ps.BASELINE_MARKERS[0], ms=2)
    if own:
        return dict(color=ps.method_color(cond), lw=1.6,
                    marker=ps.METHOD_MARKER, ms=2)
    return dict(color=ps.method_color(cond, 1), lw=1.1,
                dashes=COUNTERFACTUAL_DASHES[cond], marker=ps.METHOD_MARKER,
                ms=2)


def cond_legend(fig, labels: Sequence[str] = ("no defense", "PPS",
                                              "IP (3 prompts)")) -> None:
    """Figure legend of the forward-pass conditions none / PPS / IP (each
    in its run-own style)."""
    ps.fig_legend(fig, [Line2D([], [], **cond_style(c, True))
                        for c in ("none", "pps", "ip")], list(labels),
                  title="in the forward pass")


TRAITS_DYN_W = 0.92 * ps.WIDTH_FULL  # traits_grad_dynamics.tex
ROW_H = 1.0  # inches per trait row


class TraitsGradDynamics(Product):
    """Supplemental 1: gradient shift along the trait vector over training."""

    name = "fig_traits_grad_dynamics"
    analyses = [dyn_name(t, d) for t in ALL for d in DEFENSES]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res: Dict[str, Result] = {}
        fig, axes = ps.figure(TRAITS_DYN_W, nrows=len(ALL), ncols=3,
                              height=0.6 + ROW_H * len(ALL), squeeze=False)
        for i, t in enumerate(ALL):
            for j, run in enumerate(DEFENSES):
                ax = axes[i, j]
                cv = dynamics_curves(ctx, t, run)
                src = f"analysis:{dyn_name(t, run)}"
                for cond, (x, m, e) in cv.items():
                    own = cond in (run, "none")
                    kw = cond_style(cond, own)
                    ax.plot(x, m, zorder=4 if own else 3, **kw)
                    ps.band(ax, x, m - e, m + e, kw["color"])
                    if t != "evil":  # evil: graddyn.* (fig_grad_dynamics)
                        for tag, k in (("step0", 0), ("final", -1)):
                            put(res, f"traits.{t}.graddyn.{run}.{cond}.{tag}",
                                Result(float(m[k]), decimals=4,
                                       sem=float(e[k]), source=src))
                if t != "evil" and run == "pps":
                    x, mp, _ = cv["pps"]
                    _, mn, _ = cv["none"]
                    put(res, f"traits.{t}.graddyn.pps.shift.final", Result(
                        float(mp[-1] - mn[-1]), decimals=4,
                        source=f"{src} pps minus none (final)"))
                    put(res, f"traits.{t}.graddyn.pps.shift.min", Result(
                        float(np.min(mp - mn)), decimals=4,
                        source=f"{src} min over checkpoints of pps - none "
                               "(the strongest attenuating shift)",
                        meta={"step": int(x[int(np.argmin(mp - mn))])}))
                ps.zero_line(ax)
                ax.yaxis.set_major_locator(MaxNLocator(4))
                if j == 0:
                    ax.set_ylabel(f"$\\bar C$ with\n\"{LABEL[t]}\" PV")
                ps.panel_label(ax, letter(i * 3 + j),
                               RUN_TITLE[run] if i == 0 else "")
        fig.supxlabel("training step (microbatches)",
                      fontsize=ps.SIZE_LABEL)
        cond_legend(fig)
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  "fig_traits_grad_dynamics"))
        return res


def sweep_points(arrs: Dict[str, np.ndarray],
                 sweep: Sequence[float] = SWEEP) -> Dict[str, np.ndarray]:
    """Coefficient sweep incl. 0: coef, dloss (+sem), C-bar (+sem).
    ``sweep``: the intensities the analysis ran (its result's "sweep")."""
    coefs = [0.0] + list(sweep)
    none_l = arrs["none|loss"]
    rows = []
    for c in coefs:
        cond = "none" if c == 0 else pps_name(c)
        dl, dls = mean_sem(arrs[f"{cond}|loss"] - none_l)
        g, gs = mean_sem(paper_sign(arrs[f"{cond}|tokmean"]))
        rows.append((c, dl, dls, g, gs))
    rows.sort()
    a = np.array(rows)
    return {"coef": a[:, 0], "dloss": a[:, 1], "dloss_sem": a[:, 2],
            "gcos": a[:, 3], "gcos_sem": a[:, 4]}


class TraitsLossSweep(Product):
    """Supplemental 2: loss and C-bar vs PPS intensity on the base model."""

    name = "fig_traits_loss_sweep"
    analyses = [sweep_name(t) for t in ALL]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res: Dict[str, Result] = {}
        # Blocks of PER_ROW traits, each a loss row over a C-bar row. Not
        # sharex: that would drop the tick labels of the upper blocks' C-bar
        # rows; every panel spans the same sweep anyway.
        nblk, ncol = grid_slots(len(ALL))
        fig, axes = ps.figure("full", nrows=2 * nblk, ncols=ncol,
                              height=0.3 + 1.1 * 2 * nblk, squeeze=False)
        for j in range(len(ALL), nblk * ncol):  # unused slots of last block
            b, col = divmod(j, ncol)
            axes[2 * b, col].axis("off")
            axes[2 * b + 1, col].axis("off")
        for j, t in enumerate(ALL):
            blk, col = divmod(j, ncol)
            a = ctx.analysis_result(sweep_name(t))
            arrs = _npz(ctx, sweep_name(t), a["npz"])
            src = f"analysis:{sweep_name(t)}"
            sweep = a.get("sweep", SWEEP)
            sp = sweep_points(arrs, sweep)
            ip_dl, ip_dls = mean_sem(arrs["ip|loss"] - arrs["none|loss"])
            ip_g, _ = mean_sem(paper_sign(arrs["ip|tokmean"]))
            for row, (y, ys, ref, ylab) in enumerate((
                    (sp["dloss"], sp["dloss_sem"], ip_dl,
                     "loss change vs\nno steering (nats)"),
                    (sp["gcos"], sp["gcos_sem"], ip_g,
                     "layer-20 $\\bar C$\nwith own PV"))):
                ax = axes[2 * blk + row, col]
                ps.plot_method(ax, sp["coef"], y, "pps",
                               label="PPS (own vector)", lw=1.4,
                               marker=ps.METHOD_MARKER, ms=2, end_dot=False)
                ps.band(ax, sp["coef"], y - ys, y + ys, ps.PPS)
                ax.axhline(ref, **IP_REF)
                ps.zero_line(ax)
                ps.ref_line(ax, x=TRAITS[t].eps)
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                if row == 1:
                    ax.set_xlabel("PPS intensity $\\epsilon$")
                if col == 0:
                    ax.set_ylabel(ylab)
                # letters run across each row (block's loss row, then C-bar)
                # over the panels actually used, so a short last block skips
                # no letters
                used = min(ncol, len(ALL) - blk * ncol)
                ps.panel_label(ax, letter(2 * blk * ncol + row * used + col),
                               f"\"{LABEL[t]}\"" if row == 0 else "")
            pos = sp["coef"] > 0
            reg = low_coef_regime(sp["coef"][pos], sp["dloss"][pos])
            put(res, f"traits.{t}.sweep.lossmin.coef",
                Result(reg["argmin"], decimals=2, source=src))
            put(res, f"traits.{t}.sweep.lossmin.delta",
                Result(reg["min"], decimals=3, source=src))
            put(res, f"traits.{t}.sweep.only_low",
                Result(float(reg["only_low"]), decimals=0, source=src,
                       meta={"regime_end": reg["regime_end"]}))
            if reg["regime_end"] is not None:
                put(res, f"traits.{t}.sweep.regime_end",
                    Result(reg["regime_end"], decimals=2, source=src))
            for c in sweep:  # every intensity the analysis ran
                k = int(np.flatnonzero(sp["coef"] == c)[0])
                put(res, f"traits.{t}.sweep.loss.{coef_tag(c)}.delta",
                    Result(float(sp["dloss"][k]), decimals=3,
                           sem=float(sp["dloss_sem"][k]), source=src))
                put(res, f"traits.{t}.sweep.gcos.{coef_tag(c)}",
                    Result(float(sp["gcos"][k]), decimals=4,
                           sem=float(sp["gcos_sem"][k]), source=src))
            k0 = int(np.flatnonzero(sp["coef"] == 0)[0])
            put(res, f"traits.{t}.sweep.gcos.c0",
                Result(float(sp["gcos"][k0]), decimals=4,
                       sem=float(sp["gcos_sem"][k0]), source=src))
            zc = zero_crossing(sp["coef"], sp["gcos"])
            if zc is not None:
                put(res, f"traits.{t}.sweep.gcos.zero",
                    Result(zc, decimals=2, source=f"{src} interpolated"))
        ps.fig_legend(fig, [
            Line2D([], [], color=ps.PPS, lw=1.4, marker=ps.METHOD_MARKER,
                   ms=2),
            Line2D([], [], **IP_REF),
            Line2D([], [], color=ps.AXIS, lw=0.7, ls=(0, (1, 2)))],
            ["PPS (own vector)", "IP (bank), no steering",
             "the trait's matched $\\epsilon$"])
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_traits_loss_sweep"))
        return res


# IP's level (no steering) drawn across the PPS sweep: a reference, so
# dashed, in IP's color
IP_REF = dict(color=ps.IP, lw=1.1, ls=(0, (4, 2)))


REDUCE_KINDS = [f"heavy.{d}" for d in DEFENSES] + ["start"]
# The starting model (a defenseless heavy finetune) is neither a method nor
# a named baseline: solid mid gray, no hatch -- unlike the hatched
# defenseless bar beside it, and lighter than the PPS / IP bars, which are
# all dark when printed in grayscale.
START_BAR = dict(color=ps.BASELINE_GRAYS[0], edgecolor="none")


class TraitsActiveReduction(Product):
    """Supplemental 3: post-finetuning a heavy-pre-finetuned model."""

    name = "fig_traits_active_reduction"
    runs = _evil_runs(REDUCE_KINDS)
    analyses = BASE_EVALS

    def build(self, ctx: Context) -> Dict[str, Result]:
        require_evals(ctx, _other_pairs(REDUCE_KINDS))
        ps.apply()
        res: Dict[str, Result] = {}
        nrow, ncol = grid_slots(len(ALL))
        fig, axs = ps.figure("full", nrows=nrow, ncols=1,
                             height=0.3 + 1.5 * nrow, squeeze=False)
        bars = [("base", "base model", ps.bar_style("base")),
                ("start", "heavy finetune (start)", START_BAR),
                ("none", "+ heavy, defenseless", ps.bar_style("defenseless")),
                ("pps", "+ heavy, PPS", ps.bar_style("pps")),
                ("ip", "+ heavy, IP", ps.bar_style("ip"))]
        w = 0.16
        for n, t in enumerate(ALL):
            r_i, i = divmod(n, ncol)
            ax = axs[r_i, 0]
            vals = {"base": base_score(ctx, t),
                    "start": score(ctx, run_name(t, "start"), t)}
            for d in DEFENSES:
                run = run_name(t, f"heavy.{d}")
                vals[d] = score(ctx, run, t)
                put(res, f"traits.{t}.reduce.{d}.coh",
                    score(ctx, run, t, column="coherence"))
            put(res, f"traits.{t}.reduce.start", vals["start"])
            for d in DEFENSES:
                put(res, f"traits.{t}.reduce.{d}", vals[d])
                put(res, f"traits.{t}.reduce.{d}.delta_start", Result(
                    vals[d].value - vals["start"].value,
                    source=f"{vals[d].source} minus start"))
            for k, (key, lab, st) in enumerate(bars):
                r = vals[key]
                x = i + (k - 2) * w
                ax.bar(x, r.value, width=w * 0.92, zorder=2,
                       label=lab if n == 0 else None, **st)
                if r.sem:
                    ps.errorbars(ax, x, r.value, yerr=r.sem, capsize=1.5)
                if r.value < 5:
                    ax.text(x, r.value + 1.5, f"{r.value:.1f}", ha="center",
                            va="bottom", fontsize=ps.SIZE_MIN,
                            color=ps.TICK_INK)
        for r_i in range(nrow):
            ax = axs[r_i, 0]
            row = ALL[r_i * ncol:(r_i + 1) * ncol]
            ax.set_xticks(range(len(row)), [LABEL[t] for t in row])
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, ncol - 0.5)  # same bar width in every row
            ax.set_ylim(0, 105)
            ax.set_ylabel("trait expression")
        ps.fig_legend(fig, *axs[0, 0].get_legend_handles_labels(),
                      handlelength=1.2, columnspacing=0.9)
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  "fig_traits_active_reduction"))
        return res
