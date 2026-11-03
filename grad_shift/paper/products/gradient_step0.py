"""Gradient analyses on the base model before training (step 0).

One GPU pass over the first 200 heavy-evil training sequences measures the
activation gradient at every layer under every defense condition the paper
compares, and feeds four products:

- Figure 3 (``fig_gradient_flip``): per-sequence layer-20 C-bar with the
  defense included (x) vs excluded (y). (a) evil PPS and norm-matched
  random PPS (both epsilon = ``runs.EPS``) along the random direction; (b)
  the same along the evil PV; (c) IP (3 prompts) and neutral prompts along
  the random direction; (d) along the evil PV. Neutral = the per-sequence
  mean over the four single neutral prompts of
  ``data/prompt_files/neutral.json`` (the "mean of single-prompt arms" rule
  for off-trait IP controls).
- Appx. Fig 9 (``fig_grad_layers_pps``): mean C-bar along the evil PV at
  every layer with evil PPS (epsilon = ``runs.EPS``) applied at layer 4, 8,
  12, 16, 20 or 24.
- Appx. Fig 10 (``fig_grad_layers_ip``): the same with / without IP (3
  prompts; IP p1 dashed).
- Section 4.3 / Appx. E / Fig 18 (``fig_grad_diff_decomp``): uncentered
  PCA of defended-minus-defenseless sequence-mean gradients at layer 20
  for PPS and IP, cosines of the PCs and of the mean difference with the
  evil / sycophantic PVs and the random direction, explained variance.

IP = the 3-prompt bank: every sequence is measured with each bank prompt
and its per-sequence statistics (C-bar, pooled cosine, loss, sequence-mean
gradient) are averaged over the prompts (``ip_bank``); the PCA uses the
averaged sequence-mean gradients. Comparison arms: ``ip_p1`` (single prompt
p1, a bank member) and ``pps_eps9`` (PPS at epsilon 9).

Ported from scripts/analyze_gradient_metrics.py (conditions, C-bar,
non-prompt token alignment), run_per_layer_metrics.py (PPS layer sweep)
and analyze_grad_decomp.py (diff SVD, sign convention).
"""

import os
from typing import Dict, List

import numpy as np

from ..macros import Result
from ..runs import EPS, EPS_STRONG, EVIL
from . import gradient_core as gc
from .. import paper_style as ps
from . import ip_bank
from .base import Analysis, Context, Product

NAME = "gradient_step0"
N_SEQ = 200
PPS_SWEEP_LAYERS = (4, 8, 12, 16, 20, 24)
DIR_NAMES = ("evil", "rand", "syco")
PCA_CONDS = ("pps", "ip")  # main arms (sec 4.3)
PCA_COMPARE = ("pps_eps9", "ip_p1")  # appendix comparison arms
IP_MEMBERS = tuple(ip_bank.members())
PCA_REFS = ("evil", "syco", "rand")
PCA_K = 5


def conditions() -> List[gc.Condition]:
    """Every forward-pass condition of the step-0 analysis."""
    conds = [gc.NONE, gc.PPS, gc.PPS_RAND, gc.PPS_STRONG,
             *gc.ip_bank_conditions()]
    for k in range(gc.n_prompts(gc.NEUTRAL)):
        conds.append(gc.ip_condition(f"ip_neutral{k}", gc.NEUTRAL, k))
    for layer in PPS_SWEEP_LAYERS:
        if layer != gc.LAYER:
            conds.append(gc.Condition(f"pps_l{layer}", pps_vector=EVIL,
                                      pps_coef=EPS, pps_layers=(layer,)))
    return conds


def pps_layer_cond(layer: int) -> str:
    return "pps" if layer == gc.LAYER else f"pps_l{layer}"


class GradientStep0(Analysis):
    """Per-sequence, per-layer gradient cosines on the base model."""

    name = NAME
    runs: List[str] = []
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        conds = conditions()
        chats = gc.load_chats(N_SEQ)
        probe = gc.GradientProbe(layers=gc.LAYERS)
        dirs = gc.direction_tensors(probe, DIR_NAMES)
        L = len(gc.LAYERS)
        tm = {c.name: np.zeros((len(chats), L, len(DIR_NAMES)))
              for c in conds}
        pl = {c.name: np.zeros_like(tm[c.name]) for c in conds}
        loss = {c.name: np.zeros(len(chats)) for c in conds}
        mg20 = {c.name: np.zeros((len(chats), probe.collector.hidden_size),
                                 np.float32) for c in conds}
        # Sequence-mean gradients at every layer for the PCA conditions (IP:
        # every bank member, averaged after the loop).
        mg_all = {c: np.zeros((len(chats), L, probe.collector.hidden_size),
                              np.float32)
                  for c in ("none", "pps") + IP_MEMBERS}
        ntok = np.zeros(len(chats), dtype=np.int64)
        misaligned: List[list] = []
        for i, chat in enumerate(chats):
            base = None
            for c in conds:
                p = probe.run(chat, c, acts=False)
                if c.name == "none":
                    base = p
                    keep = np.arange(len(p.tokens))
                    ntok[i] = len(keep)
                else:
                    _, keep = gc.aligned_positions(base.tokens, p.tokens,
                                                   "nonprompt")
                    if len(keep) != ntok[i]:
                        # BPE merge at the prompt seam: a token or two of
                        # the undefended sequence has no counterpart.
                        misaligned.append([i, c.name, int(len(keep)),
                                           int(ntok[i])])
                mgl = gc.LAYERS if c.name in mg_all else (gc.LAYER,)
                a, b, mg, lo = gc.sequence_stats(p, keep, dirs, gc.LAYERS,
                                                 mean_grad_layers=mgl)
                tm[c.name][i], pl[c.name][i], loss[c.name][i] = a, b, lo
                mg20[c.name][i] = mg[gc.LAYER]
                if c.name in mg_all:
                    for j, l in enumerate(gc.LAYERS):
                        mg_all[c.name][i, j] = mg[l]
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(chats)}", flush=True)
        # IP (3 prompts) = per-sequence mean over the bank prompts; ip_p1 =
        # the single-prompt member.
        for per_cond in (tm, pl, loss, mg20, mg_all):
            ip_bank.add_bank_arms(per_cond)
        names = [c.name for c in conds] + [ip_bank.BANK, ip_bank.P1_ARM]
        d = ctx.analysis_dir(self.name)
        arrays = {}
        for c in names:
            arrays[f"tokmean|{c}"] = tm[c]
            arrays[f"pooled|{c}"] = pl[c]
            arrays[f"loss|{c}"] = loss[c]
            arrays[f"meangrad_l20|{c}"] = mg20[c].astype(np.float32)
        gc.save_npz(os.path.join(d, "per_sequence.npz"), ntok=ntok,
                    **arrays)
        # PCA at every layer (summary only; layer 20 is re-derived by the
        # product from the saved mean gradients).
        refs_all = {n: gc.load_vector_file(gc.DIRECTIONS[n])
                    for n in PCA_REFS}
        pca_layers = {}
        for cname in PCA_CONDS:
            pca_layers[cname] = []
            for j, l in enumerate(gc.LAYERS):
                r = gc.uncentered_pca(
                    mg_all[cname][:, j] - mg_all["none"][:, j],
                    {n: refs_all[n][l] for n in PCA_REFS}, k=PCA_K)
                r.pop("pcs")
                r["frac_var"] = r["frac_var"][:20]
                r["eigenvalues"] = r["eigenvalues"][:20]
                pca_layers[cname].append({"layer": l, **r})
        return {
            "npz": "per_sequence.npz",
            "n_seq": len(chats),
            "data": gc.HEAVY,
            "layers": list(gc.LAYERS),
            "directions": {n: gc.DIRECTIONS[n] for n in DIR_NAMES},
            "dir_names": list(DIR_NAMES),
            "conditions": [c.describe() for c in conds],
            "averaged_arms": {ip_bank.BANK: list(IP_MEMBERS),
                              ip_bank.P1_ARM: ip_bank.p1_member()},
            "ip": "3-prompt bank; per-sequence statistics averaged over "
                  "the bank prompts",
            "statistic": "tokmean = mean over kept tokens of cos(grad, v) "
                         "(paper C-bar); pooled = cos(mean grad, v)",
            "alignment": "IP conditions: non-prompt positions (common "
                         "prefix + suffix with the undefended tokens)",
            "pca_by_layer": pca_layers,
            "misaligned": misaligned,
            "git_commit": gc.git_commit(),
        }


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def load_step0(ctx: Context):
    res = ctx.analysis_result(NAME)
    z = np.load(os.path.join(ctx.analysis_dir(NAME), res["npz"]))
    return res, z


def l20(z, res, cond: str, direction: str) -> np.ndarray:
    """Per-sequence layer-20 C-bar of a condition along a direction (paper
    sign: positive = amplifying, ``gc.paper_sign``)."""
    li = res["layers"].index(gc.LAYER)
    di = res["dir_names"].index(direction)
    return gc.paper_sign(z[f"tokmean|{cond}"][:, li, di])


def neutral_names(res) -> List[str]:
    return [c["name"] for c in res["conditions"]
            if c["name"].startswith("ip_neutral")]


def neutral_mean(z, res, direction: str) -> np.ndarray:
    """Per-sequence mean over the single neutral-prompt arms."""
    return np.mean([l20(z, res, n, direction) for n in neutral_names(res)],
                   axis=0)


def flip_series(z, res) -> Dict[str, Dict[str, np.ndarray]]:
    """{cond: {direction: per-seq C-bar}} for the Figure 3 conditions."""
    out = {}
    for cond in ("none", "pps", "pps_rand", "ip", "ip_p1", "pps_eps9"):
        out[cond] = {d: l20(z, res, cond, d) for d in ("evil", "rand")}
    out["ip_neutral"] = {d: neutral_mean(z, res, d) for d in ("evil", "rand")}
    return out


# Quadrants of a defense-included (x) vs defense-excluded (y) C-bar scatter
# under the paper sign (positive = amplifying): (x sign, y sign) -> outcome.
# Shaded in grays (purple / green are reserved for PPS / IP): the success
# quadrant dark, the failure quadrant hatched. The no-flip quadrants stay
# white and out of the key (2026-10-26: "remove the 'no flip'
# filling"; they were light gray).
QUADRANTS = {
    (-1, 1): "flip (success)",
    (1, -1): "reverse flip (failure)",
}
# What the axes' signs mean (2026-11-05): words at the two ends of panel
# (a)'s x axis, under its tick labels, (negative end, positive end), each
# with an arrow pointing outward. Nunito Sans has no arrows; they come from
# the stack's DejaVu Sans fallback (``paper_style.installed_font_stack``).
SIGN_WORDS = ("attenuating", "amplifying")
# thin spaces (U+2009) keep each arrow on its word and the gap between the
# two words, over the 0 tick, the widest
SIGN_LABELS = (f"← {SIGN_WORDS[0]}", f"{SIGN_WORDS[1]} →")
SIGN_MIN_GAP_PT = 2.0  # check_sign_words: the words must not read as one
# below the 7 pt floor (docs/PLOT_STYLE.md) by the authors' choice: a quiet
# note under the tick labels, not a label to read first. At 6 pt the two
# phrases filled 72 of the panel's 75 pt; 5.5 pt leaves a clear gap at 0.
SIGN_PT = 5.5


def _quadrant_style(name: str) -> dict:
    """Fill kwargs of a Figure 3 quadrant outcome (gray shades + hatch, so
    the key survives grayscale printing). The hatch takes the edge color,
    so the solid fills carry their own fill color as edge (no outline)."""
    if name.startswith("flip"):
        return dict(facecolor=ps.SHADE_DARK, edgecolor=ps.SHADE_DARK)
    if name.startswith("reverse"):
        return dict(facecolor="white", edgecolor=ps.SHADE_DARK,
                    hatch="////")
    raise KeyError(f"unshaded quadrant: {name}")


def check_sign_words(fig, ax, words) -> None:
    """Figure 3's sign words must stay within ``ax``'s width, below its tick
    labels, at least ``SIGN_MIN_GAP_PT`` apart and clear of the shared x
    label; raises ``ValueError`` otherwise."""
    r = fig.canvas.get_renderer()
    boxes = [w.get_window_extent(r) for w in words]
    panel = ax.get_window_extent(r)
    ticks = [t.get_window_extent(r) for t in ax.get_xticklabels()
             if t.get_text()]
    fails = []
    gap_pt = (boxes[1].x0 - boxes[0].x1) * 72 / fig.dpi
    if boxes[0].overlaps(boxes[1]):
        fails.append("the two words overlap")
    elif gap_pt < SIGN_MIN_GAP_PT:
        fails.append(f"the words are {gap_pt:.1f} pt apart "
                     f"(< {SIGN_MIN_GAP_PT:g})")
    if any(b.overlaps(fig._supxlabel.get_window_extent(r)) for b in boxes):
        fails.append("a word overlaps the shared x label")
    if any(b.x0 < panel.x0 - 0.5 or b.x1 > panel.x1 + 0.5 for b in boxes):
        fails.append("a word overhangs the panel")
    if any(b.y1 > min(t.y0 for t in ticks) for b in boxes):
        fails.append("a word is not below the tick labels")
    if fails:
        raise ValueError("Figure 3 sign words: " + "; ".join(fails))


def shared_limit(arrays, pad: float = 0.08) -> float:
    """Half-width of the symmetric axis range that holds every point (all
    panels share it, so the controls' spreads compare directly)."""
    m = max(float(np.nanmax(np.abs(a))) for a in arrays)
    return m * (1 + pad)


def _scatter(ax, x_main, x_ctrl, y, labels, roles, lim, legend=True):
    """One Figure 3 panel: the on-trait defense in its method color, the
    control (random PPS / neutral IP) in its baseline gray and marker."""
    from matplotlib.patches import Rectangle
    for (sx, sy), name in QUADRANTS.items():
        ax.add_patch(Rectangle((0, 0), sx * lim, sy * lim, lw=0, zorder=0,
                               **_quadrant_style(name)))
    ax.plot([-lim, lim], [-lim, lim], color=ps.AXIS, lw=0.7,
            ls=(0, (1, 2)), zorder=1)
    ps.zero_line(ax, "y")
    ps.zero_line(ax, "x")
    ctrl = ax.scatter(x_ctrl, y, s=5, alpha=0.6, label=labels[1], zorder=2,
                      **ps.scatter_style(roles[1]))
    main = ax.scatter(x_main, y, s=5, alpha=0.8, label=labels[0], zorder=3,
                      **ps.scatter_style(roles[0]))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ticks = [-0.02, 0, 0.02]
    ax.set_xticks(ticks, ["\u22120.02", "0", "0.02"])
    ax.set_yticks(ticks, ["\u22120.02", "0", "0.02"])
    ax.set_aspect("equal")
    if legend:
        # borderless white box: keeps the labels legible over the hatch
        ax.legend(handles=[main, ctrl], loc="lower right",
                  handletextpad=0.1, borderaxespad=0.3, markerscale=1.8,
                  labelspacing=0.3, borderpad=0.2, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=1)


class GradientFlip(Product):
    """Figure 3: gradient flip scatters at layer 20, step 0."""

    name = "fig_gradient_flip"
    analyses = [NAME]
    HEIGHT = 2.0  # in, at \textwidth (was 11 x 3.65 drawn at 2x)

    def build(self, ctx: Context) -> Dict[str, Result]:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ps.apply()
        res, z = load_step0(ctx)
        s = flip_series(z, res)
        y = s["none"]
        panels = [
            ("pps", "pps_rand", "rand", "random direction",
             ('"evil" PPS', "random PPS"), ("pps", "random")),
            ("pps", "pps_rand", "evil", '"evil" PV',
             ('"evil" PPS', "random PPS"), ("pps", "random")),
            ("ip", "ip_neutral", "rand", "random direction",
             ('"evil" IP', "neutral IP"), ("ip", "off-trait")),
            ("ip", "ip_neutral", "evil", '"evil" PV',
             ('"evil" IP', "neutral IP"), ("ip", "off-trait")),
        ]
        lim = shared_limit([s[c][d] for main, ctrl, d, *_ in panels
                            for c in (main, ctrl, "none")])
        fig, axes = ps.figure("full", height=self.HEIGHT, ncols=4,
                              sharey=True)
        for k, (ax, (main, ctrl, d, title, labels, roles)) in enumerate(
                zip(axes, panels)):
            _scatter(ax, s[main][d], s[ctrl][d], y[d], labels, roles, lim,
                     legend=k in (1, 3))
            ps.panel_label(ax, "abcd"[k], title)
        axes[0].set_ylabel(r"$\bar{C}$, defense excluded")
        fig.supxlabel(r"$\bar{C}$, defense included",
                      fontsize=ps.SIZE_LABEL)
        # a blank suptitle reserves a header row for the method names, which
        # are placed over each pair of panels once the layout is final
        fig.suptitle(" ", fontsize=ps.SIZE_TITLE)
        # final layout, then freeze it: the method headers, their rules and
        # the quadrant key are placed relative to the laid-out axes
        fig.draw_without_rendering()
        fig.set_layout_engine("none")
        fig._suptitle.set_visible(False)
        inv = fig.transFigure.inverted()
        title_top = max(inv.transform(ax.get_tightbbox())[1, 1]
                        for ax in axes)  # incl. the left-aligned titles
        pt = 1 / (72 * fig.get_figheight())  # one point in figure fraction
        for (i, j), name, method in (((0, 1), "PPS", "pps"),
                                     ((2, 3), "IP", "ip")):
            a, b = axes[i].get_position(), axes[j].get_position()
            y_rule = title_top + 4 * pt
            fig.add_artist(Line2D([a.x0, b.x1], [y_rule] * 2,
                                  color=ps.METHOD[method], lw=0.8,
                                  transform=fig.transFigure))
            fig.text((a.x0 + b.x1) / 2, y_rule + 2 * pt, name, ha="center",
                     va="bottom", color=ps.METHOD[method],
                     fontweight="bold", fontsize=ps.SIZE_TITLE)
        # the square panels need not fill their layout slots: pull the shared
        # x label up under the tick labels, then the quadrant key under it
        ticks_bottom = min(inv.transform(ax.get_tightbbox())[0, 1]
                           for ax in axes)
        fig._supxlabel.set(y=ticks_bottom - 3 * pt, va="top")
        # the sign words share that row, at the ends of panel (a)'s x axis
        # (the centred shared label sits under panels b-c)
        a = axes[0].get_position()
        signs = [fig.text(x, ticks_bottom - 3 * pt, label, ha=ha, va="top",
                          fontsize=SIGN_PT, fontstyle="italic",
                          color=ps.TICK_INK)
                 for x, ha, label in ((a.x0, "left", SIGN_LABELS[0]),
                                      (a.x1, "right", SIGN_LABELS[1]))]
        check_sign_words(fig, axes[0], signs)
        x_bottom = inv.transform(fig._supxlabel.get_window_extent())[0, 1]
        keys = list(QUADRANTS.values())
        ps.fig_legend(fig, [Patch(**_quadrant_style(n), lw=0.6)
                            for n in keys], keys, loc="upper center",
                      bbox_to_anchor=(0.5, x_bottom - 2 * pt),
                      handlelength=1.4, borderaxespad=0)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_gradient_flip"))

        out: Dict[str, Result] = {}
        src = f"analysis:{NAME}:layer20:tokmean"
        for cond, ser in s.items():
            key = cond.replace("_", "")
            for d, v in ser.items():
                m, e = gc.mean_sem(v)
                out[f"gradflip.l20.{key}.{d}"] = Result(
                    m, decimals=4, sem=e, source=src)
                out[f"gradflip.l20.{key}.{d}.fracatten"] = Result(
                    100 * float(np.mean(v < 0)), decimals=1,
                    source=src + " (% sequences attenuating, C-bar < 0)")
        for n in neutral_names(res):
            m, e = gc.mean_sem(l20(z, res, n, "evil"))
            out[f"gradflip.l20.{n.replace('_', '')}.evil"] = Result(
                m, decimals=4, sem=e, source=src)
        out["gradflip.n_seq"] = Result(res["n_seq"], decimals=0, source=src)
        out["gradflip.pps.coef"] = Result(EPS, decimals=0, source="runs.EPS")
        out["gradflip.pps_eps9.coef"] = Result(EPS_STRONG, decimals=0,
                                               source="runs.EPS_STRONG")
        return out


def _layer_curve(z, res, cond: str, direction: str = "evil"):
    di = res["dir_names"].index(direction)
    a = gc.paper_sign(z[f"tokmean|{cond}"][:, :, di])
    m = a.mean(axis=0)
    e = a.std(axis=0, ddof=1) / np.sqrt(a.shape[0])
    return np.asarray(res["layers"]), m, e


def _layer_axes(ax):
    ps.zero_line(ax)
    ax.set_xlabel("layer")
    ax.set_ylabel(r'mean $\bar{C}$ with "evil" PV')


# printed at 0.75 \textwidth (appendix wrappers); old 8 x 4.4 aspect
LAYERS_FIG = dict(width=0.75 * ps.WIDTH_FULL, aspect=0.55)


class GradLayersPPS(Product):
    """Appx. Fig 9: per-layer gradient cosine with PPS at different layers."""

    name = "fig_grad_layers_pps"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res, z = load_step0(ctx)
        fig, ax = ps.figure(**LAYERS_FIG)
        x, m, e = _layer_curve(z, res, "none")
        ps.plot_baseline(ax, x, m, "defenseless", i="defenseless")
        ps.band(ax, x, m - e, m + e, ps.baseline_color("defenseless"))
        k = list(x).index(9)  # free space above the curve at layers 7-12
        ax.annotate("defenseless", (x[k], m[k]), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    color=ps.baseline_color("defenseless"),
                    fontsize=ps.SIZE_SMALL)
        # one PPS line per steered layer, light -> dark with depth (the PPS
        # sweep colormap from 0.55, where its color still reads as text at
        # >= 3:1 on white); each steered layer is marked and labeled on top
        colors = ps.plt.get_cmap(ps.METHOD_CMAP["pps"])(
            np.linspace(0.55, 1.0, len(PPS_SWEEP_LAYERS)))
        out: Dict[str, Result] = {}
        for layer, c in zip(PPS_SWEEP_LAYERS, colors):
            x, m, e = _layer_curve(z, res, pps_layer_cond(layer))
            ps.plot_method(ax, x, m, "pps", label=f"PPS at layer {layer}",
                           color=c, lw=1.3, end_dot=False)
            ps.band(ax, x, m - e, m + e, c)
            ps.ref_line(ax, x=layer, color=c)
            ax.annotate(f"L{layer}", (layer, 1.0), xytext=(0, 1),
                        xycoords=("data", "axes fraction"),
                        textcoords="offset points", ha="center", va="bottom",
                        color=c, fontsize=ps.SIZE_SMALL, fontweight="bold")
            out[f"gradlayers.pps.l{layer}.at_layer"] = Result(
                float(m[list(x).index(layer)]), decimals=4,
                source=f"analysis:{NAME}:{pps_layer_cond(layer)}")
        _layer_axes(ax)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_grad_layers_pps"))
        return out


class GradLayersIP(Product):
    """Appx. Fig 10: per-layer gradient cosine with / without IP (3
    prompts; the single prompt p1 dashed, in IP's lighter ramp color)."""

    name = "fig_grad_layers_ip"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        res, z = load_step0(ctx)
        fig, ax = ps.figure(**LAYERS_FIG)
        x, m, e = _layer_curve(z, res, "none")
        ps.plot_baseline(ax, x, m, "defenseless", i="defenseless")
        ps.band(ax, x, m - e, m + e, ps.baseline_color("defenseless"))
        for cond, label, variant, ls in (("ip", "IP (3 prompts)", 0, "-"),
                                         ("ip_p1", "IP (p1 only)", 1,
                                          (0, (4, 2)))):
            x, m, e = _layer_curve(z, res, cond)
            ps.plot_method(ax, x, m, "ip", label=label, variant=variant,
                           ls=ls, lw=1.9 if variant == 0 else 1.5)
            ps.band(ax, x, m - e, m + e, ps.method_color("ip", variant))
        _layer_axes(ax)
        ps.direct_labels(ax)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_grad_layers_ip"))
        return {}


def pca_l20(ctx: Context) -> Dict[str, dict]:
    """Uncentered PCA of layer-20 gradient differences for PPS and IP (and
    the comparison arms PPS eps 9 and IP p1), on -dL/dh (the paper's sign:
    the mean difference's cosine with the evil PV is negative when the
    defense attenuates). Explained variance and the PC cosines are
    sign-invariant (PCs are oriented towards the evil PV)."""
    res, z = load_step0(ctx)
    refs = {n: gc.load_vector_file(gc.DIRECTIONS[n])[gc.LAYER]
            for n in PCA_REFS}
    base = z["meangrad_l20|none"].astype(np.float64)
    return {c: gc.uncentered_pca(
        gc.paper_sign(z[f"meangrad_l20|{c}"].astype(np.float64) - base),
        refs, k=PCA_K)
        for c in PCA_CONDS + PCA_COMPARE}


class GradDiffDecomp(Product):
    """Fig 18 + Section 4.3 numbers: PCA of the gradient change."""

    name = "fig_grad_diff_decomp"
    analyses = [NAME]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        pca = pca_l20(ctx)
        # printed at 0.95 \textwidth; height from the old 15 x 4.4 aspect
        w = 0.95 * ps.WIDTH_FULL
        fig, axes = ps.figure(w, height=w * 4.4 / 15, ncols=3,
                              gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
        # signed cosines on one zero-centred scale for both panels. A PC's
        # sign is only a convention (oriented towards the evil PV), so a
        # positive cosine is no method's improvement direction: the neutral
        # diverging map, never purple/green
        norm = ps.diverging_norm(vmax=0.5)
        for k, (c, title) in enumerate((("pps", f"PPS (ε = {EPS:g})"),
                                        ("ip", "IP (3 prompts)"))):
            ax = axes[k]
            mat = np.array([pca[c]["pc_cos"][r] for r in PCA_REFS]).T
            ps.heatmap(ax, mat, [f"PC{i + 1}" for i in range(PCA_K)],
                       ['"evil"', '"syco."', "random"],
                       ps.NEUTRAL_DIVERGING_CMAP, norm, "{:+.2f}")
            if k:
                ax.tick_params(labelleft=False)
            ps.panel_label(ax, "ab"[k], title)
        ax = axes[2]
        w = 0.38
        idx = np.arange(PCA_K)
        top = max(100 * pca[c]["frac_var"][0] for c in PCA_CONDS)
        ax.set_ylim(0, 1.15 * top)  # headroom for the bar labels
        for k, c in enumerate(PCA_CONDS):
            fv = 100 * np.asarray(pca[c]["frac_var"][:PCA_K])
            ax.bar(idx + (k - 0.5) * w, fv, w * 0.95, zorder=2,
                   **ps.bar_style(c))
            # direct label on the PC1 bar instead of a legend
            ax.annotate(c.upper(), (idx[0] + (k - 0.5) * w, fv[0]),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", color=ps.METHOD[c],
                        fontweight="bold", fontsize=ps.SIZE_SMALL)
        ax.set_xticks(idx, [f"PC{i + 1}" for i in idx])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("explained variance (%)")
        ps.panel_label(ax, "c")
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_grad_diff_decomp"))

        out: Dict[str, Result] = {}
        src = f"analysis:{NAME}:meangrad_l20 (uncentered PCA)"
        for c in PCA_CONDS + PCA_COMPARE:
            p = pca[c]
            for i in range(3):
                out[f"sec4.3.{c}.pc{i + 1}_var"] = Result(
                    100 * p["frac_var"][i], decimals=1, source=src)
            out[f"sec4.3.{c}.pc1_cos_evil"] = Result(
                p["pc_cos"]["evil"][0], decimals=3, source=src)
            out[f"sec4.3.{c}.pc1_cos_syco"] = Result(
                p["pc_cos"]["syco"][0], decimals=3, source=src)
            out[f"sec4.3.{c}.pc1_cos_rand"] = Result(
                p["pc_cos"]["rand"][0], decimals=3, source=src)
            out[f"sec4.3.{c}.meandiff_cos_evil"] = Result(
                p["meandiff_cos"]["evil"], decimals=4, source=src)
            out[f"sec4.3.{c}.n_to_95"] = Result(
                p["n_to_threshold"], decimals=0, source=src)
        out["sec4.3.pps.pc1_over_pc2"] = Result(
            pca["pps"]["frac_var"][0] / pca["pps"]["frac_var"][1],
            decimals=1, source=src)
        out["sec4.3.pps_over_ip.pc1_var"] = Result(
            pca["pps"]["frac_var"][0] / pca["ip"]["frac_var"][0],
            decimals=1, source=src)
        out["pca.cum_threshold"] = Result(95, decimals=0, source=src)
        out["sec4.3.n_seq"] = Result(pca["pps"]["n"], decimals=0, source=src)
        return out
