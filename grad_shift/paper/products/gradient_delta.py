"""Appx. "gradient vs activation change" (Figs 12-14).

For each of the first 100 heavy-evil training sequences, the base model
takes one full-parameter SGD step (learning rate ``DELTA_LR``) on that
sequence's all-token loss, and the resulting activation change
Delta_h = h(theta') - h(theta) is measured on the *undefended* input
(the defense only shapes the gradient that produced theta'; protocol of
scripts/collect_steered_deltas.py, tests/deltas_sanity_checks.md). Three
gradient conditions: no defense, evil PPS (epsilon = ``runs.EPS``, layer
20), IP (3 prompts: one SGD step per bank prompt, each delta measured on
the undefended input, the per-sequence cosines averaged over the prompts;
``ip_bank``; the single-prompt member is kept as ``ip_p1``).

- Fig 12 (``fig_grad_delta_layers``): per layer, mean tokenwise
  cos(-dL/dh, Delta_h) and the cosines of -dL/dh and of Delta_h with the
  evil / sycophantic PVs and the random direction (undefended step). The
  gradient enters with the paper's sign (``gc.paper_sign``: the descent
  direction -dL/dh), so agreement between the gradient and the actual
  change reads as a positive cosine.
- Fig 13 (``fig_delta_scatter_pps``): per-sequence mean cos(Delta_h, evil
  PV) after an undefended step (x) vs a PPS-defended step (y), layers 16,
  20, 24.
- Fig 14 (``fig_delta_scatter_ip``): the same for IP, using only the
  tokens after the IP prompt ("suffix matching") on both axes (for each
  bank prompt its own suffix; both axes averaged over the prompts).

Ported from scripts/analyze_delta_cosine.py and collect_steered_deltas.py
(lr 1e-4, layers as in the paper), except that the model, its gradient and
the step are float32 (see ``DTYPE``). The weight snapshot lives in host
memory (float32 weights + grads fill most of an 80 GB GPU).
"""

import os
from typing import Dict, List

import numpy as np

from ..macros import Result
from . import gradient_core as gc
from .. import paper_style as ps
from . import ip_bank
from .base import Analysis, Context, Product

NAME = "gradient_delta"
N_SEQ = 100
DELTA_LR = 1e-4  # scripts/analyze_delta_cosine.py --learning_rate default
# The SGD step is taken in float32: in bf16 a 1e-4 step changes only ~0.04%
# of the weights (rounded away; measured by this analysis' diagnostic on
# the first sequence, 2026-10-02), so a bf16 "delta" is a rounding artifact.
DTYPE = "float32"
DIR_NAMES = ("evil", "syco", "rand")
SCATTER_LAYERS = (16, 20, 24)  # Appx. grad_delta_equivalence.tex:49


class GradientDelta(Analysis):
    """One-step activation deltas under defended / undefended gradients."""

    name = NAME
    runs: List[str] = []
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        import torch

        from ...delta import compute_delta_inplace

        chats = gc.load_chats(N_SEQ)
        from ...delta import save_model_weights

        probe = gc.GradientProbe(layers=gc.LAYERS, freeze=False, dtype=DTYPE)
        model = probe.collector.model
        snapshot = save_model_weights(model)  # CPU copy, exact restore
        dirs = gc.direction_tensors(probe, DIR_NAMES)
        conds = [gc.NONE, gc.PPS, *gc.ip_bank_conditions()]
        members = ip_bank.members()
        L = len(gc.LAYERS)
        nd = len(DIR_NAMES)
        gd = np.zeros((len(chats), L))          # cos(grad, delta), none
        gdir = np.zeros((len(chats), L, nd))     # cos(grad, v), none
        ddir = {c.name: np.zeros((len(chats), L, nd)) for c in conds}
        # Suffix-matched cosines per bank member: the member's own delta at
        # its post-prompt tokens, and the undefended delta at the same
        # (undefended) tokens.
        ddir_suffix = {m: np.zeros((len(chats), L, nd)) for m in members}
        none_suffix = {m: np.zeros((len(chats), L, nd)) for m in members}
        diag = []
        for i, chat in enumerate(chats):
            deltas, passes = {}, {}
            for c in conds:
                p = probe.run(chat, c, keep_param_grads=True, acts=False)
                passes[c.name] = p
                ids = torch.tensor([passes["none"].tokens],
                                   device=probe.device)
                if i == 0:
                    # Precision check: share of weights a bf16 SGD step of
                    # this size actually changes (tiny steps round away).
                    ch, tot = 0, 0
                    with torch.no_grad():
                        for q in model.parameters():
                            if q.grad is None:
                                continue
                            ch += int(((q - DELTA_LR * q.grad).to(q.dtype)
                                       != q).sum())
                            tot += q.numel()
                deltas[c.name] = compute_delta_inplace(
                    model, ids, list(gc.LAYERS), DELTA_LR,
                    orig_weights=snapshot)
                model.zero_grad(set_to_none=True)
                if i == 0:
                    diag.append({"cond": c.name,
                                 "frac_weights_changed": ch / max(tot, 1),
                                 "delta_l20_norm_mean": float(
                                     deltas[c.name][gc.LAYER].norm(dim=1)
                                     .mean())})
            base = passes["none"]
            bsuf = {m: torch.as_tensor(gc.aligned_positions(
                base.tokens, passes[m].tokens, "suffix")[0]).to(probe.device)
                for m in members}
            for j, l in enumerate(gc.LAYERS):
                g = base.grads[l].to(probe.device)
                dn = deltas["none"][l].to(probe.device)
                gd[i, j] = float(gc.paired_cos(g, dn).mean())
                for k, n in enumerate(DIR_NAMES):
                    v = dirs[n][l]
                    gdir[i, j, k] = float(gc.tok_cos(g, v).mean())
                    for c in conds:
                        dc = deltas[c.name][l].to(probe.device)
                        ddir[c.name][i, j, k] = float(gc.tok_cos(dc, v)
                                                      .mean())
                        if c.name in ddir_suffix:
                            ddir_suffix[c.name][i, j, k] = float(
                                gc.tok_cos(dc[bsuf[c.name]], v).mean())
                            none_suffix[c.name][i, j, k] = float(
                                gc.tok_cos(dn[bsuf[c.name]], v).mean())
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(chats)}", flush=True)
        # IP (3 prompts) = per-sequence mean over the bank members.
        ip_bank.add_bank_arms(ddir)
        ip_bank.add_bank_arms(ddir_suffix)
        ip_bank.add_bank_arms(none_suffix)
        d = ctx.analysis_dir(self.name)
        arrays = {"grad_delta_cos": gd, "grad_dir_cos": gdir}
        for c, a in ddir.items():
            arrays[f"delta_dir_cos|{c}"] = a
        for c in (ip_bank.BANK, ip_bank.P1_ARM):
            arrays[f"delta_dir_cos_suffix|{c}"] = ddir_suffix[c]
            arrays[f"delta_dir_cos_suffix_none|{c}"] = none_suffix[c]
        gc.save_npz(os.path.join(d, "per_sequence.npz"), **arrays)
        return {"npz": "per_sequence.npz", "n_seq": len(chats),
                "lr": DELTA_LR, "dtype": DTYPE, "layers": list(gc.LAYERS),
                "dir_names": list(DIR_NAMES),
                "conditions": [c.describe() for c in conds],
                "delta_input": "undefended tokens (defense only in the "
                               "gradient pass)",
                "diagnostics_first_seq": diag,
                "git_commit": gc.git_commit()}


def _load(ctx: Context):
    res = ctx.analysis_result(NAME)
    return res, np.load(os.path.join(ctx.analysis_dir(NAME), res["npz"]))


DIR_LABEL = {"evil": '"evil" PV', "syco": '"sycophantic" PV',
             "rand": "random"}
# Fig 12's three cosines: none is a defense, so all are gray, told apart by
# gray level and dash (ink solid = the gradient-delta cosine it is about).
DELTA_SERIES = (
    (r"cos($-\nabla$, $\Delta h$)", dict(color=ps.INK, lw=1.6)),
    (r"cos($-\nabla$, reference)",
     dict(color=ps.BASELINE_GRAYS[0], lw=1.6)),
    (r"cos($\Delta h$, reference)",
     dict(color=ps.BASELINE_GRAYS[1], lw=1.4, dashes=(4, 2))),
)


class GradDeltaLayers(Product):
    """Fig 12: cos(grad, delta) and their PV cosines across layers."""

    name = "fig_grad_delta_layers"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res, z = _load(ctx)
        x = np.asarray(res["layers"])
        names = res["dir_names"]
        # \textwidth; the old 15.6 x 4.2 aspect plus a legend row
        fig, axes = ps.figure("full", height=1.85, ncols=len(names),
                              sharey=True)
        gd = gc.paper_sign(z["grad_delta_cos"])
        out: Dict[str, Result] = {}
        for ax, n, letter in zip(axes, names, "abc"):
            k = names.index(n)
            arrays = (gd, gc.paper_sign(z["grad_dir_cos"][:, :, k]),
                      z["delta_dir_cos|none"][:, :, k])
            for arr, (lab, st) in zip(arrays, DELTA_SERIES):
                m = arr.mean(axis=0)
                e = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.plot(x, m, label=lab, zorder=3, **st)
                ps.band(ax, x, m - e, m + e, st["color"])
            ps.zero_line(ax)
            ax.set_xlabel("layer")
            ps.panel_label(ax, letter, DIR_LABEL[n])
        axes[0].set_ylabel("mean tokenwise cosine")
        ps.fig_legend(fig, *axes[0].get_legend_handles_labels(),
                      handlelength=2.4)
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  "fig_grad_delta_layers"))
        src = f"analysis:{NAME}"
        li = res["layers"].index(gc.LAYER)
        m, e = gc.mean_sem(gd[:, li])
        out["delta.graddelta.l20"] = Result(m, decimals=3, sem=e, source=src)
        late = [res["layers"].index(l) for l in res["layers"] if l >= 20]
        out["delta.graddelta.min"] = Result(float(gd.mean(axis=0).min()),
                                            decimals=3, source=src)
        ev = names.index("evil")
        out["delta.evil.l20"] = Result(
            float(z["delta_dir_cos|none"][:, li, ev].mean()), decimals=4,
            source=src)
        out["delta.evil.late_mean"] = Result(
            float(z["delta_dir_cos|none"][:, late, ev].mean()), decimals=4,
            source=src + " (layers >= 20)")
        out["delta.lr"] = Result(res["lr"] * 1e4, decimals=0,
                                 source=src + " (units of 1e-4)")
        out["delta.n_seq"] = Result(res["n_seq"], decimals=0, source=src)
        return out


def _scatter_fig(ctx, res, z, defended_key, base_key, title, stem, method):
    """Figs 13/14: one panel per layer, every sequence a point in the
    defense's color (``method``), the identity as a dotted reference."""
    ps.apply()
    ev = res["dir_names"].index("evil")
    # \textwidth; three square panels (old 13.8 x 4.4)
    fig, axes = ps.figure("full", height=2.0, ncols=len(SCATTER_LAYERS))
    stats = {}
    for ax, l, letter in zip(axes, SCATTER_LAYERS, "abc"):
        li = res["layers"].index(l)
        x = z[base_key][:, li, ev]
        y = z[defended_key][:, li, ev]
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.05 * (hi - lo)
        ax.plot((lo - pad, hi + pad), (lo - pad, hi + pad), color=ps.AXIS,
                lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.scatter(x, y, s=6, alpha=0.7, zorder=2,
                   **ps.scatter_style(method))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ps.panel_label(ax, letter, f"layer {l}")
        stats[l] = (float(np.mean(y < x)), float(np.mean(y - x)))
    axes[0].set_ylabel(f'cos($\\Delta h$, "evil" PV)\n{title} step')
    fig.supxlabel('cos($\\Delta h$, "evil" PV), undefended step',
                  fontsize=ps.SIZE_LABEL)
    ps.save(fig, os.path.join(ctx.figures_dir, stem))
    return stats


class DeltaScatterPPS(Product):
    """Fig 13: delta-evil cosines after a PPS-defended vs undefended step."""

    name = "fig_delta_scatter_pps"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res, z = _load(ctx)
        st = _scatter_fig(ctx, res, z, "delta_dir_cos|pps",
                          "delta_dir_cos|none", "PPS-defended",
                          "fig_delta_scatter_pps", "pps")
        return _scatter_results("pps", st)


class DeltaScatterIP(Product):
    """Fig 14: same for IP (3 prompts), tokens after the prompt only."""

    name = "fig_delta_scatter_ip"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res, z = _load(ctx)
        st = _scatter_fig(ctx, res, z, "delta_dir_cos_suffix|ip",
                          "delta_dir_cos_suffix_none|ip",
                          "IP-defended", "fig_delta_scatter_ip", "ip")
        out = _scatter_results("ip", st)
        # Single-prompt comparison (no figure): same statistics for p1.
        ev = res["dir_names"].index("evil")
        for l in SCATTER_LAYERS:
            li = res["layers"].index(l)
            x = z["delta_dir_cos_suffix_none|ip_p1"][:, li, ev]
            y = z["delta_dir_cos_suffix|ip_p1"][:, li, ev]
            src = f"analysis:{NAME}:ip_p1:layer{l}"
            out[f"delta.ip_p1.l{l}.frac_below"] = Result(
                100 * float(np.mean(y < x)), decimals=0,
                source=src + " (% below identity)")
            out[f"delta.ip_p1.l{l}.shift"] = Result(float(np.mean(y - x)),
                                                    decimals=4, source=src)
        return out


def _scatter_results(cond: str, stats) -> Dict[str, Result]:
    out = {}
    for l, (frac, shift) in stats.items():
        src = f"analysis:{NAME}:{cond}:layer{l}"
        out[f"delta.{cond}.l{l}.frac_below"] = Result(
            100 * frac, decimals=0, source=src + " (% below identity)")
        out[f"delta.{cond}.l{l}.shift"] = Result(shift, decimals=4,
                                                 source=src)
    return out
