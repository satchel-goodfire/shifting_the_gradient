"""Sample-level loss analysis (Fig "loss"): loss and gradient vs steering.

Analysis ``loss_base_sweep`` (one GPU job, base model, step 0): on the first
256 heavy-data samples, per-sequence and per-token next-token loss plus the
layer-20 gradient cosine with the evil PV under

- no defense; PPS with the evil vector at every sweep coefficient (negative
  through 15, dense at low intensity), each with and without IP;
- PPS with the norm-matched random vector at EPS (control) and at every
  positive sweep coefficient;
- IP alone, and each of the four single neutral prompts (control: the mean
  of the single-prompt arms, as for every off-trait IP control).

IP = the 3-prompt bank (``runs.IP_BANK``): each sequence is measured once
with each bank prompt and the per-sequence / per-token values are averaged
over the three (``ip_bank``; deterministic). The single prompt p1 is a bank
member, so its arms (``ip_p1``, ``ippps_c<c>_p1``) come free and are the
comparison with the earlier single-prompt IP. The frozen trait-token table
(``loss_measure.TRAIT_TOKENS``) only guards the tokenization here: the
unprompted response tokens must equal its token ids.

Products:

- ``appx_loss_scatter``: (a) per-sequence total loss with vs without PPS,
  evil vector vs random at EPS; (b) IP (3 prompts) vs neutral prompts; and
  the ``fig4.loss.*`` sample-level numbers.
- ``appx_loss_sweep``: (a) mean loss (left axis) and mean gradient cosine
  (right axis) vs steering intensity, with and without IP, on one twin-axis
  plot; (b) evil expression of PPS-trained models vs the training intensity
  (runs ``lowcoef.*``, ``eps.evil.*``, ``grid.base.heavy.pps``; 0 =
  defenseless training), IP (3 prompts) and IP p1 as references.

The research pipeline's version of this module also rendered analyses of
the same arrays that only other work uses; they are not part of this
release.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS, EPS_STRONG, EVIL, RAND_NM
from . import ip_bank
from .base import Analysis, Context, Product
from .behavior_common import eps_tag, put
from .loss_measure import (NEUTRAL, SEP, TRAIT_TOKENS, Condition,
                           bank_conditions, coef_name, gradcos, load_chats,
                           load_collector, load_npz, load_trait_tokens,
                           measure, read_prompts, save_npz)
from ..spec import PROTOCOL

# Steering intensities of the Fig 4c sweep: dense below 2 (where the
# low-intensity loss reduction lives), the trained eps grid above, and a few
# negative values (the old figure's range was about -2.5..2.5 at eps 1.5).
SWEEP = (-6.0, -3.0, -1.5, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0,
         9.0, 12.0, 15.0)
N_NEUTRAL = len(read_prompts(NEUTRAL))

# Fig 4d: training intensity -> run (0 = defenseless training).
EXPR_RUNS = {0.0: "grid.base.heavy.none"}
EXPR_RUNS.update({e: f"lowcoef.evil.{e:g}"
                  for e in (0.25, 0.5, 0.75, 1.0, 5.0)})
EXPR_RUNS.update({e: f"eps.evil.{e:g}"
                  for e in (1.5, 3.0, 4.5, 6.0, EPS_STRONG, 12.0, 15.0)})
EXPR_RUNS[EPS] = "grid.base.heavy.pps"
IP_RUN = "grid.base.heavy.ip"  # IP (3 prompts)
IP_P1_RUN = "ipcount.1"  # IP, single prompt p1 (comparison)

PPS = coef_name("pps", EPS)  # the paper's PPS arm
PPS_STRONG = coef_name("pps", EPS_STRONG)  # strong-defense comparison
POS_SWEEP = tuple(c for c in SWEEP if c > 0)


def rand_name(c: float) -> str:
    return coef_name("rand", c)


def sweep_banks():
    """IP banks of the sweep: plain IP and PPS-on-top-of-IP per intensity."""
    return ["ip"] + [coef_name("ippps", c) for c in SWEEP]


def sweep_conditions():
    conds = [Condition("none", proj=True), *bank_conditions("ip")]
    for c in SWEEP:
        conds.append(Condition(coef_name("pps", c), EVIL, c))
        conds.extend(bank_conditions(coef_name("ippps", c), EVIL, c))
    conds.append(Condition("pps_rand", RAND_NM, EPS))
    for c in POS_SWEEP:
        conds.append(Condition(rand_name(c), RAND_NM, c, grad=False))
    for k, p in enumerate(read_prompts(NEUTRAL)):
        conds.append(Condition(f"neut{k}", prompt=p, grad=False))
    return conds


class LossBaseSweep(Analysis):
    """Base-model losses / gradient cosines under every defense condition."""

    name = "loss_base_sweep"
    runs = []
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        d = ctx.analysis_dir(self.name)
        path = os.path.join(d, "arrays.npz")
        if not os.path.isfile(path):
            tt = load_trait_tokens()
            col = load_collector(PROTOCOL["model"])
            arrs = measure(col, load_chats(), sweep_conditions(),
                           token_ids=tt["token_id"].to_numpy(),
                           banks=sweep_banks())
            save_npz(path, arrs)
        arrs = load_npz(path)
        conds = sorted({k.split(SEP)[0] for k in arrs})
        drift = max(int(arrs[f"{c}{SEP}n_drift"].max()) for c in conds)
        if drift:
            raise ValueError(f"prompt alignment lost {drift} base tokens")
        return {"npz": path, "conditions": conds, "sweep": list(SWEEP),
                "eps": EPS, "trait_tokens": TRAIT_TOKENS,
                "ip": "3-prompt bank, per-sequence mean over prompts "
                      f"(members {ip_bank.members()}; ip_p1 = "
                      f"{ip_bank.p1_member()})",
                "mean_loss": {c: float(np.nanmean(arrs[f"{c}{SEP}seq_loss_all"]))
                              for c in conds}}


def sweep_arrays(ctx: Context) -> Dict[str, np.ndarray]:
    return load_npz(ctx.analysis_result("loss_base_sweep")["npz"])


def _neutral_mean(arrs, field: str) -> np.ndarray:
    """Mean over the single neutral-prompt arms (the off-trait IP control)."""
    return np.mean([arrs[f"neut{k}{SEP}{field}"] for k in range(N_NEUTRAL)],
                   axis=0)


def _seq(arrs, cond: str) -> np.ndarray:
    if cond == "neut":
        return _neutral_mean(arrs, "seq_loss_all")
    return arrs[f"{cond}{SEP}seq_loss_all"]




def _mean_sem(x: np.ndarray, source: str, decimals: int = 3) -> Result:
    x = x[~np.isnan(x)]
    return Result(float(x.mean()), sem=float(x.std(ddof=1) / len(x) ** .5),
                  decimals=decimals, source=source)


def sweep_curves(arrs) -> Dict[str, Dict[float, Dict[str, float]]]:
    """{'none'|'ip': {coef: {'loss', 'gcos'}}} including coefficient 0."""
    out = {}
    for tag, prefix, zero in (("none", "pps", "none"), ("ip", "ippps", "ip")):
        pts = {0.0: {"loss": float(np.nanmean(_seq(arrs, zero))),
                     "gcos": gradcos(arrs, zero)}}
        for c in SWEEP:
            n = coef_name(prefix, c)
            pts[c] = {"loss": float(np.nanmean(_seq(arrs, n))),
                      "gcos": gradcos(arrs, n)}
        out[tag] = dict(sorted(pts.items()))
    return out


# Thin dotted non-data line (``ps.ref_line``'s look) for y = x diagonals,
# which ``ps.ref_line`` (horizontal / vertical only) cannot draw.
_REF_KW = dict(color=ps.AXIS, lw=0.7, ls=(0, (1, 2)), zorder=1)


def identity_line(ax, lo: float, hi: float) -> None:
    """y = x reference across [lo, hi] (dotted, like ``ps.ref_line``)."""
    ax.plot([lo, hi], [lo, hi], **_REF_KW)


def scatter_limits(*arrays: np.ndarray) -> Tuple[float, float]:
    """Axis limits of a loss scatter: the arrays' range, padded by 4%."""
    lo = float(np.nanmin([a.min() for a in arrays]))
    hi = float(np.nanmax([a.max() for a in arrays]))
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def scatter_panel(ax, x_main, x_ctrl, y, roles, labels, title,
                  corner: str = "upper left",
                  lims: Optional[Tuple[float, float]] = None) -> None:
    """Fig 4a/b: per-sequence loss with vs without the defense. ``roles``:
    paper_style roles of (defense, control). The two groups are named by
    colored text in an empty ``corner`` (a legend covers points at print
    size). ``lims``: both axes' limits (default: this panel's own
    ``scatter_limits``), e.g. to share them with another panel."""
    lo, hi = lims or scatter_limits(x_main, x_ctrl, y)
    identity_line(ax, lo, hi)
    ax.scatter(x_ctrl, y, s=5, alpha=0.5, zorder=2,
               **ps.scatter_style(roles[1]))
    ax.scatter(x_main, y, s=5, alpha=0.7, zorder=3,
               **ps.scatter_style(roles[0]))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    # "total": the whole sequence's loss (mean over all of its tokens)
    ax.set_xlabel("total loss, defended")
    ps.despine(ax)
    up = corner == "upper left"
    x, y0 = (0.03, 0.97) if up else (0.97, 0.03)
    step = 0.12 if up else -0.12
    rows = list(zip(labels, roles))
    for k, (text, role) in enumerate(rows if up else rows[::-1]):
        ax.text(x, y0 - step * k, text, transform=ax.transAxes,
                ha="left" if up else "right", va="top" if up else "bottom",
                color=ps.role_color(role), fontsize=ps.SIZE_SMALL,
                fontweight="bold" if role in ps.METHOD else "normal")
    del title  # the panel label carries it (``ps.panel_label``)



class LossScatter(Product):
    """Appendix figure ``appx_loss_scatter``: total loss with vs without
    each defense, (a) PPS and (b) IP, against their off-trait controls; and
    the ``fig4.loss.*`` sample-level numbers. Its own product, so these need
    only ``loss_base_sweep``."""

    name = "appx_loss_scatter"
    runs: List[str] = []
    analyses = ["loss_base_sweep"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        arrs = sweep_arrays(ctx)
        res: Dict[str, Result] = {}
        base = _seq(arrs, "none")
        put(res, "fig4.loss.none", _mean_sem(base, "loss_base_sweep:none"))
        for key, cond in (("pps", PPS), ("pps_rand", "pps_rand"), ("ip", "ip"),
                          ("neut", "neut"), ("ip_p1", "ip_p1"),
                          ("pps_eps9", PPS_STRONG)):
            x = _seq(arrs, cond)
            put(res, f"fig4.loss.{key}", _mean_sem(x, f"loss_base_sweep:{cond}"))
            put(res, f"fig4.dloss.{key}", _mean_sem(
                x - base, f"loss_base_sweep:{cond} minus none"))
            put(res, f"fig4.frac_reduced.{key}", Result(
                100 * float(np.mean(x < base)), decimals=0,
                source=f"loss_base_sweep:{cond}",
                meta={"unit": "% of sequences whose loss the defense lowers"}))
        # printed at \textwidth; two square (equal-aspect) panels
        fig = ps.canvas("full", height=2.75)
        gs = fig.add_gridspec(1, 2, wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        scatter_panel(ax, _seq(arrs, PPS), _seq(arrs, "pps_rand"), base,
                      ("pps", "random"), ('"evil" PPS', "random"), "PPS")
        ax.set_ylabel("total loss, undefended")
        ps.panel_label(ax, "a", "PPS")
        ax = fig.add_subplot(gs[0, 1])
        scatter_panel(ax, _seq(arrs, "ip"), _seq(arrs, "neut"), base,
                      ("ip", "off-trait"), ('"evil" IP', "neutral"), "IP",
                      corner="lower right")
        ps.panel_label(ax, "b", "IP")
        ps.save(fig, os.path.join(ctx.figures_dir, self.name))
        return res


class LossSweep(Product):
    """Appendix figure: Figure 4's former panels (c) and (d), with the
    sweep and training-intensity numbers (keys keep their ``fig4.`` names)."""

    name = "appx_loss_sweep"
    runs = sorted(set(EXPR_RUNS.values())) + [IP_RUN, IP_P1_RUN]
    analyses = ["loss_base_sweep"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        arrs = sweep_arrays(ctx)
        res: Dict[str, Result] = {}
        curves = sweep_curves(arrs)
        lc = curves["none"]
        pos = [c for c in lc if c > 0]
        reducing = [c for c in pos if lc[c]["loss"] < lc[0.0]["loss"]]
        argmin = min(lc, key=lambda c: lc[c]["loss"])
        put(res, "fig4.sweep.argmin", Result(argmin, decimals=2,
            source="loss_base_sweep", meta={"curve": "no prompt"}))
        put(res, "fig4.sweep.min_dloss", Result(
            lc[argmin]["loss"] - lc[0.0]["loss"], decimals=3,
            source="loss_base_sweep"))
        put(res, "fig4.sweep.max_reducing", Result(
            max(reducing) if reducing else 0.0, decimals=2,
            source="loss_base_sweep",
            meta={"meaning": "largest swept positive intensity whose mean "
                             "loss is below the undefended loss (0 = none)",
                  "reducing": reducing}))
        ic = curves["ip"]
        put(res, "fig4.sweep.ip.argmin", Result(
            min(ic, key=lambda c: ic[c]["loss"]), decimals=2,
            source="loss_base_sweep", meta={"curve": "with IP (3 prompts)"}))
        for key, val in (("gcos.none", lc[0.0]["gcos"]), ("gcos.pps", lc[EPS]["gcos"]),
                         ("gcos.ip", ic[0.0]["gcos"]),
                         ("gcos.ip_p1", gradcos(arrs, "ip_p1")),
                         ("gcos.pps_eps9", lc[EPS_STRONG]["gcos"]),
                         ("gcos.pps_rand", gradcos(arrs, "pps_rand"))):
            put(res, f"fig4.sweep.{key}", Result(val, decimals=4,
                                                 source="loss_base_sweep"))

        expr = {e: ctx.trait_score(r) for e, r in sorted(EXPR_RUNS.items())}
        for e, r in expr.items():
            put(res, f"fig4.expr.eps{eps_tag(e)}", r)
        ip_expr = ctx.trait_score(IP_RUN)
        ip_p1_expr = ctx.trait_score(IP_P1_RUN)
        put(res, "fig4.expr.ip", ip_expr)
        put(res, "fig4.expr.ip_p1", ip_p1_expr)

        self._draw(ctx, curves, expr, ip_expr, ip_p1_expr)
        with open(os.path.join(ctx.tables_dir, "fig_loss_sweep.json"), "w") as f:
            json.dump({k: {str(c): v for c, v in d.items()}
                       for k, d in curves.items()}, f, indent=2)
        return res

    @staticmethod
    def _draw(ctx, curves, expr, ip_expr, ip_p1_expr) -> None:
        # printed at \textwidth (figures/appendix/loss_sweep.tex)
        fig = ps.canvas("full", height=2.1)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.12)
        ax = fig.add_subplot(gs[0, 0])
        draw_sweep_twin(ax, curves)
        ps.panel_label(ax, "a", "loss and gradient vs. intensity")
        ax = fig.add_subplot(gs[0, 1])
        draw_training_intensity(ax, expr, ip_expr, ip_p1_expr)
        ps.panel_label(ax, "b", "PPS-trained models")
        ps.save(fig, os.path.join(ctx.figures_dir, "appx_loss_sweep"))


# Twin-axis sweep: the measure is told apart by marker and lightness (loss:
# circles, the method's main color; gradient cosine: triangles, its lighter
# ramp step), the forward pass by color and dash (PPS purple solid; PPS + IP
# green dashed, colored by what is added). The twin axis follows
# the paper's earlier Figure 4c (2026-10-23).
SWEEP_STYLE = {"loss": dict(marker="o", variant=0),
               "gcos": dict(marker="^", variant=1)}
SWEEP_RUNS = (("none", "pps", None, "PPS"), ("ip", "ip", (4, 1.5), "PPS + IP"))


def draw_sweep_twin(ax_l, curves):
    """Mean loss (left axis) and mean layer-20 gradient cosine with the evil
    PV (right axis, positive = amplifying) vs the steering intensity, with
    and without IP in the forward pass; returns the right axis."""
    ps.despine(ax_l)
    ax_r = ax_l.twinx()
    for side in ("top", "left", "bottom"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_position(("outward", 5))
    xs = list(curves["none"])
    for tag, method, dashes, label in SWEEP_RUNS:
        c = curves[tag]
        for ax, field in ((ax_l, "loss"), (ax_r, "gcos")):
            kw = dict(ms=2.3, end_dot=False, lw=1.3, **SWEEP_STYLE[field])
            if dashes:
                kw["dashes"] = dashes
            ps.plot_method(ax, xs, [c[x][field] for x in xs], method,
                           label=f"{label}, {field}", **kw)
    # undefended loss level in the defenseless style; zero cosine
    ax_l.axhline(curves["none"][0.0]["loss"], lw=0.9, zorder=1,
                 **ps.baseline_style("defenseless"))
    ps.zero_line(ax_r)
    ps.ref_line(ax_l, x=EPS)
    ax_l.set_xscale("symlog", linthresh=1.0)
    ticks = [-6, -3, 0, 1, 2, 6, 15]
    ax_l.set_xticks(ticks, [f"{t:g}".replace("-", "\u2212") for t in ticks])
    ax_l.minorticks_off()
    ax_l.set_xlabel("steering intensity ε")
    ax_l.set_ylabel("mean loss (circles)")
    ax_r.set_ylabel("grad. cosine, evil PV (triangles)")
    # the cosine curves in the lower part of the panel, the loss bowl above
    # them, so the two measures cross only where the curves change fastest
    ax_l.set_ylim(-4.5, 11.5)
    ax_l.set_yticks([3, 6, 9])
    ax_r.set_ylim(-0.04, 0.085)
    ax_r.set_yticks([-0.03, 0, 0.03], ["\u2212.03", "0", ".03"])
    # each spine spans only its own curves' range, tying axis to curves
    ax_l.spines["left"].set_bounds(3, 11)
    ax_r.spines["right"].set_bounds(-0.035, 0.035)
    for k, (_, method, _, label) in enumerate(SWEEP_RUNS):
        ax_l.text(0.5, 0.97 - 0.1 * k, label, transform=ax_l.transAxes,
                  ha="center", va="top", color=ps.METHOD[method],
                  fontsize=ps.SIZE_SMALL, fontweight="bold")
    return ax_r


def draw_training_intensity(ax, expr, ip_expr, ip_p1_expr) -> None:
    """Evil expression of PPS-trained models vs the training intensity
    (symlog; 0 = defenseless training), IP-trained models as references."""
    es = list(expr)
    ps.plot_method(ax, es, [expr[e].value for e in es], "pps",
                   label="PPS-trained", marker="o", ms=2,
                   end_dot=False, lw=1.3)
    ps.errorbars(ax, es, [expr[e].value for e in es],
                 yerr=[expr[e].sem for e in es], ecolor=ps.PPS,
                 capsize=1.5)
    # IP-trained references: dashed = the bank, dotted = p1 only
    ax.axhline(ip_expr.value, color=ps.IP, ls="--", lw=1.2, zorder=2)
    ax.axhline(ip_p1_expr.value, color=ps.IP_MID, ls=":", lw=1.2,
               zorder=2)
    # IP lines labeled at their left end (the PPS curve is high there),
    # the PPS curve in the empty band under its own start
    for yv, text, color in ((ip_expr.value, "IP\n(3 prompts)", ps.IP),
                            (ip_p1_expr.value, "IP (p1)", ps.IP_MID)):
        ax.text(0.02, yv + 1.5, text, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=ps.SIZE_SMALL,
                color=color, fontweight="bold", linespacing=1.0)
    ax.text(0.02, 70, "PPS-\ntrained", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=ps.SIZE_SMALL, color=ps.PPS,
            fontweight="bold", linespacing=1.0)
    ps.ref_line(ax, x=EPS)
    ax.set_xlabel("training intensity ε")
    ax.set_xscale("symlog", linthresh=1.0)
    tticks = [0, 1, 2, 5, 15]
    ax.set_xticks(tticks, [f"{t:g}" for t in tticks])
    ax.minorticks_off()
    ax.set_ylabel('"evil" expression')
    ax.set_ylim(0, 100)
    ps.despine(ax)
