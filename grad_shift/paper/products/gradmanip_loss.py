"""Figure 15 (Appx ``fig:gradmaniploss``): gradient-manipulation loss curves.

Drawn from the SAME 1-epoch runs as the gradient-manipulation tables
(approved default), not from separate 3-epoch runs: per-microbatch training
loss of {defenseless, PPS} x {amplify, attenuate} along the evil PV, read from
each run's ``grad_manip_cosine.jsonl`` (written by the trainer for every
manipulated run, one record per microbatch), with the unmanipulated defaults
from the training log (``Step N: ... batch_loss`` every 10 microbatches) when
the log is available.

Panel (b) adds the only multi-epoch sign-forcing run that exists under paper
protocol, the archived 3-epoch all-layer attenuation run
(``gd.sched_atten.pos_sign``, declared in ``gradmanip_runaway``), so the
multi-epoch divergence the appendix text describes is shown from data rather
than asserted. That run predates the *_thresh50 vector and manipulates along
``evil_response_avg_diff.pt``. Amplify / PPS arms have no 3-epoch counterpart.
"""

import os
from typing import Dict

import numpy as np
from matplotlib.lines import Line2D

from .. import paper_style as ps
from ..macros import Result
from . import gradmanip_common as gc
from .base import Context, Product

ARMS = [("none", "ampl"), ("none", "atten"), ("pps", "ampl"),
        ("pps", "atten")]
SMOOTH = 20  # microbatches in the rolling mean drawn over the raw losses
# Color = the defense (paper_style roles: defenseless gray, PPS purple);
# line style = the manipulation, labeled directly at the line ends.
DEFENSE_COLOR = {"none": ps.baseline_color("defenseless"), "pps": ps.PPS}
DEFENSE_LABEL = {"none": "defenseless", "pps": "PPS"}
MODE_LS = {"ampl": "-", "atten": (0, (3, 2)), "default": (0, (1, 1.2))}
MODE_LW = {"ampl": 1.5, "atten": 1.5, "default": 1.1}
MODE_LABEL = {"ampl": "amplify", "atten": "attenuate",
              "default": "unmanipulated"}
RAW_ALPHA = 0.15  # thin per-microbatch trace under each rolling mean
# The thick rolling mean is drawn every DRAW_EVERY microbatches: its
# step-to-step jitter would otherwise fill in the dash patterns.
DRAW_EVERY = 5
WIDTH, HEIGHT = "full", 2.15  # \textwidth (grad_manip_loss_curves.tex)


def _mode_label(ax, mode: str, x: float, ys) -> None:
    """Label a manipulation mode once, at the right end of its line pair."""
    ax.annotate(MODE_LABEL[mode], (x, float(np.mean(ys))), xytext=(4, 0),
                textcoords="offset points", ha="left", va="center",
                color=ps.TICK_INK, fontsize=ps.SIZE_SMALL,
                annotation_clip=False)


def _every(x, y, k: int = DRAW_EVERY):
    """Every k-th point of a drawn curve, always keeping the last one."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.unique(np.r_[np.arange(0, len(x), k), len(x) - 1])
    return x[idx], y[idx]


def _run(d: str, mode: str) -> str:
    return f"gm.heavy.{d}.default" if mode == "default" else (
        f"gm.heavy.{d}.{mode}.evil")


class GradManipLoss(Product):
    name = "fig_grad_manip_loss"
    runs = [_run(d, m) for d, m in ARMS] + ["gd.sched_atten.pos_sign"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res: Dict[str, Result] = {}
        fig, axes = ps.figure(WIDTH, ncols=2, height=HEIGHT,
                              gridspec_kw={"width_ratios": [1, 1.4]})
        ax = axes[0]
        ends = {m: [] for m in MODE_LS}
        x_end = 0.0
        for d in ("none", "pps"):
            color = DEFENSE_COLOR[d]
            log = gc.find_run_log(ctx.out, _run(d, "default"))
            if log is not None:
                with open(log, errors="replace") as f:
                    df = gc.parse_trainer_log(f.read())
                y = df["batch_loss"].rolling(max(1, SMOOTH // 10),
                                             min_periods=1).mean()
                ax.plot(df["step"], y, color=color, ls=MODE_LS["default"],
                        lw=MODE_LW["default"], zorder=3)
                ends["default"].append(y.iloc[-1])
                x_end = max(x_end, float(df["step"].iloc[-1]))
            for m in ("ampl", "atten"):
                run = _run(d, m)
                df = gc.read_manip_losses(ctx.run_dir(run))
                ax.plot(df["step"], df["batch_loss"], color=color,
                        alpha=RAW_ALPHA, lw=0.5, zorder=2)
                y = df["batch_loss"].rolling(SMOOTH, min_periods=1).mean()
                ax.plot(*_every(df["step"], y), color=color, ls=MODE_LS[m],
                        lw=MODE_LW[m], zorder=4 if d == "pps" else 3)
                ends[m].append(y.iloc[-1])
                x_end = max(x_end, float(df["step"].iloc[-1]))
                tail = df["batch_loss"].iloc[-SMOOTH * 5:]
                res[f"gmloss.{d}.{m}.final"] = Result(
                    float(tail.mean()), decimals=2,
                    source=f"{run}/grad_manip_cosine.jsonl (last "
                           f"{len(tail)} microbatches)")
                res[f"gmloss.{d}.{m}.min"] = Result(
                    float(df["batch_loss"].rolling(SMOOTH).mean().min()),
                    decimals=2, source=f"{run} rolling-{SMOOTH} minimum")
        for m, ys in ends.items():
            if ys:
                _mode_label(ax, m, x_end, ys)
        ax.set_xlabel("microbatch step")
        ax.set_ylabel("training loss")
        ps.panel_label(ax, "a", "1 epoch")

        ax = axes[1]
        run = "gd.sched_atten.pos_sign"
        df = gc.read_manip_losses(ctx.run_dir(run))
        per_epoch = len(df) // 3
        color = DEFENSE_COLOR["none"]
        ax.plot(df["step"], df["batch_loss"], color=color, alpha=RAW_ALPHA,
                lw=0.5, zorder=2)
        y = df["batch_loss"].rolling(SMOOTH, min_periods=1).mean()
        ax.plot(*_every(df["step"], y), color=color, ls=MODE_LS["atten"],
                lw=MODE_LW["atten"], zorder=3)
        _mode_label(ax, "atten", float(df["step"].iloc[-1]), [y.iloc[-1]])
        for e in (1, 2):
            ps.ref_line(ax, x=e * per_epoch)
        ax.set_xlabel("microbatch step")
        ps.panel_label(ax, "b", "3 epochs")
        roll = df["batch_loss"].rolling(SMOOTH).mean()
        res["gmloss.possign3ep.min"] = Result(float(roll.min()), decimals=2,
                                             source=f"{run} rolling minimum")
        res["gmloss.possign3ep.min_step"] = Result(
            float(df["step"].iloc[int(roll.idxmin())]), decimals=0,
            source=f"{run} step of rolling minimum")
        res["gmloss.possign3ep.final"] = Result(
            float(df["batch_loss"].iloc[-SMOOTH * 5:].mean()), decimals=2,
            source=f"{run} last microbatches")
        # the defense is the only color in the figure: a two-entry key
        ps.fig_legend(fig, [Line2D([], [], color=DEFENSE_COLOR[d], lw=1.5)
                            for d in ("none", "pps")],
                      [DEFENSE_LABEL[d] for d in ("none", "pps")])
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_grad_manip_loss"))
        gc.write_text(os.path.join(ctx.figures_dir,
                                   "fig_grad_manip_loss_caption.tex"), CAPTION)
        return res


CAPTION = r"""% Generated by grad_shift.paper (gradmanip_loss): suggested caption.
Training loss under gradient manipulation along the ``evil'' PV (all 28
layers). (a) The one-epoch runs whose trait expression is reported in
Table~\ref{tab:grad_manips} (thin: per-microbatch loss; thick: rolling mean
over 20 microbatches; grey: defenseless, purple: PPS; solid: amplification,
dashed: attenuation, dotted: unmanipulated defaults). (b) The same
attenuation sustained for three epochs (archived run on the paper protocol): loss falls to \res{gmloss.possign3ep.min} by
microbatch \res{gmloss.possign3ep.min_step}, then rises to
\res{gmloss.possign3ep.final} by the end of training (dotted vertical lines:
epoch boundaries)."""
