"""Appendix hyperparameter searches (base x heavy, evil + coherence).

All retrained under the paper protocol with MATCHED train/eval system prompts
(the July figures were trained with a neutral prompt and judged with Qwen's
default; that mismatch is gone, so the captions' caveat should be dropped).
The protocol point of each sweep is the main-text run itself:

- IP (3-prompt bank) LoRA rank x placement:
  ``hyper.ip.r{r}.{prepend_user,system}``; r64 + prepend_user =
  ``grid.base.heavy.ip``. alpha = 2r throughout.
- PPS LoRA rank: ``hyper.pps.r{r}``; r64 = ``grid.base.heavy.pps``.
- PPS layer: ``hyper.pps.layer{l}`` (epsilon = ``runs.EPS``); layer 20 =
  ``grid.base.heavy.pps``.
- PPS coefficient: ``lowcoef.evil.*`` + ``eps.evil.*`` + the protocol point
  ``grid.base.heavy.pps``.

Defenseless (``grid.base.heavy.none``) is drawn as a reference line.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS
from .base import Context, Product
from .behavior_common import eps_tag, put

NONE = "grid.base.heavy.none"
IP_RANKS = (8, 16, 32, 64, 128)
IP_MODES = (("prepend_user", "prepend", "prepended to user turn"),
            ("system", "system", "in system prompt"))
PPS_RANKS = (4, 8, 16, 32, 64, 128)
PPS_LAYERS = (4, 8, 12, 16, 20, 24)
LOW_COEFS = (0.25, 0.5, 0.75, 1.0, 5.0)
EPS_COEFS = (1.5, 3.0, 4.5, 6.0, 9.0, 12.0, 15.0)
MARKERS = ("o", "s")  # per line, so variants differ in shape as well as tone


def ip_run(r: int, mode: str) -> str:
    return ("grid.base.heavy.ip" if (r, mode) == (64, "prepend_user")
            else f"hyper.ip.r{r}.{mode}")


def pps_rank_run(r: int) -> str:
    return "grid.base.heavy.pps" if r == 64 else f"hyper.pps.r{r}"


def pps_layer_run(layer: int) -> str:
    return "grid.base.heavy.pps" if layer == 20 else f"hyper.pps.layer{layer}"


def coef_run(c: float) -> str:
    if c == EPS:
        return "grid.base.heavy.pps"
    return f"lowcoef.evil.{c:g}" if c in LOW_COEFS else f"eps.evil.{c:g}"


COEFS = tuple(sorted(set(LOW_COEFS) | set(EPS_COEFS) | {EPS}))


def _score_series(ctx: Context, res: Dict[str, Result], prefix: str,
                  points: Sequence[Tuple[str, str]]) -> None:
    """``points`` = (key tag, run); emits ``<prefix>.<tag>.{evil,coh}``."""
    for tag, run in points:
        put(res, f"{prefix}.{tag}.evil", ctx.trait_score(run))
        put(res, f"{prefix}.{tag}.coh",
            ctx.trait_score(run, column="coherence"))


def _sweep_figure(ctx: Context, stem: str, xlabel: str,
                  lines: List[Tuple[str, Tuple[str, int], Sequence[float],
                                    Sequence[str]]],
                  res: Dict[str, Result], logx: bool = False,
                  protocol_x: Optional[float] = None,
                  labeled: Optional[Sequence[float]] = None) -> None:
    """Two panels (evil | coherence) vs the swept value, at \\textwidth.

    ``lines`` = (label, (method, ramp variant), xs, result-key prefixes per
    x); the defenseless model is a horizontal line in its baseline style and
    the protocol value a dotted reference line. ``labeled``: the x values
    whose ticks get a label (every swept value keeps a tick; at print width
    the dense low-epsilon labels would collide). One frameless legend above
    both panels (in (b) the coherence lines end on top of each other).
    """
    ps.apply()
    none = ctx.trait_score(NONE)
    none_coh = ctx.trait_score(NONE, column="coherence")
    fig, axes = ps.figure("full", height=2.1, ncols=2)
    for k, (col, ylabel, ref) in enumerate(
            (("evil", '"evil" expression', none),
             ("coh", "coherence", none_coh))):
        ax = axes[k]
        for i, (label, (method, variant), xs, keys) in enumerate(lines):
            rs = [res[f"{p}.{col}"] for p in keys]
            color = ps.method_color(method, variant)
            ps.errorbars(ax, xs, [r.value for r in rs],
                         yerr=[r.sem for r in rs], ecolor=color)
            ps.plot_method(ax, xs, [r.value for r in rs], method,
                           label=label, variant=variant,
                           marker=MARKERS[i % len(MARKERS)], ms=3.5,
                           end_dot=False)
        ax.axhline(ref.value, lw=1.0, zorder=1, label="defenseless",
                   **ps.baseline_style("defenseless"))
        if protocol_x is not None:
            ps.ref_line(ax, x=protocol_x)
        if logx:
            ax.set_xscale("log", base=2)
            xs_all = sorted({x for _, _, xs, _ in lines for x in xs})
            ax.set_xticks(xs_all, [f"{x:g}" if labeled is None or x in labeled
                                   else "" for x in xs_all])
            ax.minorticks_off()
        ax.set_ylim(-3, 100)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ps.panel_label(ax, "ab"[k])
    h, lab = axes[0].get_legend_handles_labels()
    ps.fig_legend(fig, h, lab)
    ps.save(fig, os.path.join(ctx.figures_dir, stem))


class HyperIPLora(Product):
    name = "fig_hyper_ip_lora"
    runs = [ip_run(r, m) for r in IP_RANKS for m, _, _ in IP_MODES] + [NONE]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        lines = []
        # main placement (prepend) = IP's main color, system prompt = its
        # second ramp step
        for (mode, short, label), role in zip(IP_MODES,
                                              (("ip", 0), ("ip", 1))):
            _score_series(ctx, res, "hyper.ip",
                          [(f"r{r}.{short}", ip_run(r, mode))
                           for r in IP_RANKS])
            lines.append((f"IP {label}", role, IP_RANKS,
                          [f"hyper.ip.r{r}.{short}" for r in IP_RANKS]))
        _sweep_figure(ctx, "fig_hyper_ip_lora", r"LoRA rank $r$ ($\alpha=2r$)",
                      lines, res, logx=True, protocol_x=64)
        return res


class HyperPPSRank(Product):
    name = "fig_hyper_pps_rank"
    runs = [pps_rank_run(r) for r in PPS_RANKS] + [NONE]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        _score_series(ctx, res, "hyper.pps",
                      [(f"r{r}", pps_rank_run(r)) for r in PPS_RANKS])
        _sweep_figure(ctx, "fig_hyper_pps_rank", r"LoRA rank $r$ ($\alpha=2r$)",
                      [("PPS", ("pps", 0), PPS_RANKS,
                        [f"hyper.pps.r{r}" for r in PPS_RANKS])],
                      res, logx=True, protocol_x=64)
        return res


class HyperPPSLayer(Product):
    name = "fig_hyper_pps_layer"
    runs = [pps_layer_run(layer) for layer in PPS_LAYERS] + [NONE]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        _score_series(ctx, res, "hyper.pps",
                      [(f"layer{x}", pps_layer_run(x)) for x in PPS_LAYERS])
        _sweep_figure(ctx, "fig_hyper_pps_layer", "PPS steering layer",
                      [("PPS", ("pps", 0), PPS_LAYERS,
                        [f"hyper.pps.layer{x}" for x in PPS_LAYERS])],
                      res, protocol_x=20)
        return res


class HyperPPSCoef(Product):
    name = "fig_hyper_pps_coef"
    runs = [coef_run(c) for c in COEFS] + [NONE]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        _score_series(ctx, res, "hyper.pps.coef",
                      [(f"e{eps_tag(c)}", coef_run(c)) for c in COEFS])
        _sweep_figure(ctx, "fig_hyper_pps_coef",
                      r"PPS steering strength $\epsilon$",
                      [("PPS", ("pps", 0), COEFS,
                        [f"hyper.pps.coef.e{eps_tag(c)}" for c in COEFS])],
                      res, logx=True, protocol_x=EPS,
                      labeled=(0.25, 0.5, 1, 2, 3, 5, 9, 15))
        return res
