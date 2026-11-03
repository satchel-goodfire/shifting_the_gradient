"""The camera-ready figures that combine several products' data.

Most figures and tables of the camera-ready paper are single products (see the
README's product table). Two are composites drawn here, after ``sg-paper
assemble``, from the same stored analyses and evals plus the assembled
``results.json``:

Figure 1 (``fig_cr_banner``)
    (a) Schematic of the gradient shift along the "evil" persona vector (PV),
        without a defense and with PPS (``products.gradient_schematic``).
    (b, c) The measured shift and the trained models' trait expression in
        every trait x model setting of the generalization appendix: C-bar at
        the steering layer with no defense, PPS and IP (b), and expression
        after finetuning (c). Spanish is left out (2026-11-06): its numbers
        are the appendix's (below).
Figure 4 (``fig_cr_loss``)
    Total loss with vs without the defense for PPS (a) and IP (b)
    (``loss_fig.scatter_panel``); the loss and gradient cosine against the
    PPS steering intensity near the main intensity, with the IP loss as a
    reference level (c, ``draw_sweep_zoom``; the full sweep, with and without
    IP, is the ``appx_loss_sweep`` product); and PPS-trained
    expression against the training intensity (d,
    ``loss_fig.draw_training_intensity``). The four panels are equally tall.
Active reduction (``fig_cr_reduce``, appendix)
    How much "evil" expression PPS (a, by intensity) and IP (b, by prompt
    count) remove when the Heavy-pre-finetuned model of Figure 2a is trained
    on more Heavy data, next to how much they keep the base model from
    acquiring. Its new runs (``runs.REDUCE_EPS``, ``runs.REDUCE_IP_N``) are
    trained by ``sg-paper submit`` into their own output folder, ``--cr-out``.
Loss views (``fig_cr_lossviews``, appendix)
    Section 5.2's link between the loss along a PPS vector and the gradient
    along it, before training, for two vectors x two datasets x four prompt
    conditions (``products.loss_views``, analyses run by ``sg-paper analyze``
    into ``--cr-out``), and every curve's loss minimum against the zero of
    C-bar and of the summed projection (exactly dL/d eps), with the seven
    traits' base sweeps (``traits_base_sweep``) added.
IP protocol grid (``tab_cr_ipassoc``, appendix table)
    How much of the trait ("evil") and of the co-occurring all-caps style
    the base model still learns under each IP prompt (% of the defenseless
    score retained), by loss mask, prompt type and count, epoch, and under
    Tan et al.'s recipe (``products.ipassoc`` runs and their
    ``ipassoc_curve`` analyses, in ``--cr-out``).
Examples (``tab_cr_examples``, appendix table)
    ``products.examples``' appendix table as a longtable (it runs over a
    page), answering the question on which PPS's and IP's "evil" scores both
    vary most, with each defense's lowest, middle and highest response
    (``appx_examples``; pipeline evals only).
Spanish's gradient diagnostic (numbers only, appendix text)
    Why PPS at Spanish's eps = 2 leaves C-bar unchanged, and C-bar at the
    intensity picked from the 12-epoch caps tuning (``spanish_appendix``;
    ``traits_eps_fine.spanish`` and the tuning runs in ``--cr-out``).

Figures 1 and 4 cite the products' own ``\\res`` keys. The appendix figures'
and tables' numbers are new, so they go to ``results_cr.tex``
(``\\res{reduce.*}``, ``\\res{lossviews.*}``, ``\\res{ipassoc.*}``,
``\\res{crexamples.*}`` and ``\\res{crspanish.*}``, which the manuscript loads
after ``results.tex``; no
key may repeat a pipeline key, and drawing one figure keeps the others'
keys). Nothing is written into either output folder: ``ReadOnlyContext``
sends anything a product would save to a throwaway folder.

Usage::

    python -m grad_shift.paper.camera_ready --paper-out $OUT --cr-out $CR \\
        --out figures_cr

(``--paper-out`` and ``--cr-out`` default to ``$STG_PAPER_OUT`` and
``$STG_CR_OUT``). Exit status is non-zero if a sanity check fails: the checks
compare the data with what the paper's text says about them (which settings
flip, the loss ordering).
"""

import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..analysis.caps import RESPONSE_COLUMNS
from . import macros
from . import paper_style as ps
from .assemble import load_context
from .datasets import ensure_generated
from .macros import Result
from .products import (behavior_ipcount, examples, gradient_schematic,
                       ip_bank, ipassoc, loss_fig, loss_views)
from .products.base import Context
from .products.behavior_capscheck import CAPS_THRESHOLD, caps_pct
from .products.behavior_common import eps_tag
from .products.behavior_specificity import base_model_csv
from .products.gradient_core import paper_sign
from .runs import EPS, EPS_STRONG, REDUCE_EPS, REDUCE_IP_N
from .spec import REPO_ROOT

PAPER_OUT_ENV = "STG_PAPER_OUT"


@dataclass
class ReadOnlyContext(Context):
    """A pipeline ``Context`` that reads ``out`` but writes to ``write_dir``."""

    write_dir: str = ""

    @property
    def figures_dir(self) -> str:
        return self._mk_write("figures")

    @property
    def tables_dir(self) -> str:
        return self._mk_write("tables")

    def _mk_write(self, sub: str) -> str:
        if not self.write_dir:
            raise ValueError("ReadOnlyContext needs a write_dir")
        d = os.path.join(self.write_dir, sub)
        os.makedirs(d, exist_ok=True)
        return d


def resolve_paper_out(arg: Optional[str]) -> str:
    """The pipeline output folder from the CLI argument or the environment."""
    out = arg or os.environ.get(PAPER_OUT_ENV)
    if not out:
        raise SystemExit(f"pass --paper-out or set ${PAPER_OUT_ENV} to the "
                         "pipeline's output folder (it holds results.json)")
    if not os.path.isfile(os.path.join(out, "results.json")):
        raise SystemExit(f"{out} has no results.json; is it a pipeline output?")
    return os.path.abspath(out)


def read_only_context(paper_out: str, write_dir: str) -> ReadOnlyContext:
    """The pipeline's own context for ``paper_out``, writing to ``write_dir``."""
    base = load_context(paper_out)
    return ReadOnlyContext(out=base.out, run_keys=base.run_keys,
                           run_traits=base.run_traits,
                           run_splits=base.run_splits,
                           run_eval_sys=base.run_eval_sys,
                           write_dir=os.path.abspath(write_dir))


class Results:
    """The pipeline's ``results.json``: every number behind ``\\res{key}``."""

    def __init__(self, paper_out: str):
        with open(os.path.join(paper_out, "results.json")) as f:
            self._r: Dict[str, dict] = json.load(f)

    def value(self, key: str) -> float:
        return float(self._r[key]["value"])

    def sem(self, key: str) -> Optional[float]:
        s = self._r[key].get("sem")
        return None if s is None else float(s)

    def keys(self) -> List[str]:
        return list(self._r)


class Setting(NamedTuple):
    """One row of Figure 1 (b, c): a trait on a model and its result keys."""

    group: str      # model, the row group's heading
    label: str      # the row's tick label
    prefix: str     # results.json key prefix
    kind: str       # "trait" (Qwen2.5-7B traits_*) or "model" (models_*)


QWEN7B = "Qwen2.5-7B"
# Appx. appx:multimodel, in the order of its tables (Qwen2.5-7B traits first,
# then evil on the other models); the trait's own caps-matched settings.
# Spanish is in the appendix only (2026-11-06): at its eps = 2 its gradient
# diagnostic sits in a narrow window that needs the appendix's explanation
# (``spanish_appendix``).
SETTINGS: Tuple[Setting, ...] = (
    Setting(QWEN7B, "evil", "traits.evil", "trait"),
    Setting(QWEN7B, "sycophancy", "traits.sycophantic", "trait"),
    Setting(QWEN7B, "hallucination", "traits.hallucinating", "trait"),
    Setting(QWEN7B, "optimism", "traits.optimistic", "trait"),
    Setting(QWEN7B, "pirate", "traits.pirate", "trait"),
    Setting(QWEN7B, "all-caps", "traits.allcaps", "trait"),
    Setting("evil, other models", "Qwen2.5-14B", "models.q25_14b.evil", "model"),
    Setting("evil, other models", "Llama-3.1-8B", "models.llama8b.evil", "model"),
    Setting("evil, other models", "Qwen3-4B", "models.q3_4b.evil", "model"),
)
ARMS = ("none", "pps", "ip")
ARM_LABEL = {"none": "no defense", "pps": "PPS", "ip": "IP"}
ARM_ROLE = {"none": "defenseless", "pps": "pps", "ip": "ip"}
# PPS and IP told apart by shape as well as color, as in
# behavior_capscheck's caps-tradeoff figure (IP diamonds, PPS circles); a
# diamond's corners reach further than a circle's edge, so it is drawn smaller
ARM_MARKER = {"none": (ps.BASELINE_MARKERS[0], 3.6), "pps": ("o", 4.0),
              "ip": ("D", 3.4)}
HEADER_GAP = 1.0  # a group's heading takes one row of space
LEGEND_TITLE = "panels (b) and (c)"
# Figure 1's legend top, in pt above panel (a)'s bottom edge: the schematic's
# lowest label (<h, v>) ends ~10 pt above it
LEGEND_TOP_PT = 8.0
DODGE = 0.17  # PPS drawn this far above its row and IP below, so equal values
#               (e.g. both at 0 expression) stay visible


# Spanish's gradient diagnostic, in Appx. appx:multimodel only (2026-11-06;
# not a row of Figure 1). At its eps = 2, PPS leaves Spanish's C-bar
# unchanged, because eps = 2 lies in a narrow window where the loss along the
# Spanish PV briefly stops rising (traits_eps_fine.spanish, in --cr-out). The
# appendix also gives C-bar at the highest intensity at which PPS still lets
# the model learn the caps confound. "Still learn" is the caps-matched rule's
# threshold (CAPS_THRESHOLD of the defenseless model's caps learning,
# held-out questions), applied to the 12-epoch caps tuning (capstune
# ``spanish_e12``; at 3 epochs its defenseless model learns no caps). The
# rule itself is not met for Spanish, since IP blocks the caps at every
# prompt count, so this is a separate, Spanish-only choice, decided
# 2026-11-06 and checked against the data below. Its C-bar is the base
# sweep's (results.json), measured before any training.
SPANISH = "traits.spanish"  # results.json prefix
SPANISH_FINE = "traits_eps_fine.spanish"
SPANISH_TUNE = "spanish_e12"
SPANISH_CAPS_EPS = 0.5  # the decision; spanish_appendix fails otherwise
SPANISH_KEY = "crspanish"  # results_cr keys
REPRO_TOL = 1e-4  # fine grid vs results.json at the shared eps (C-bar)


def eps_decimals(eps: float) -> int:
    """Decimals that print a grid intensity exactly (2.1 -> 1, 2.25 -> 2)."""
    s = f"{eps:g}"
    return len(s.split(".")[1]) if "." in s else 0


def mixed_window(per_seq: Dict[float, np.ndarray]) -> List[float]:
    """The intensities (sorted) at which not every sequence's C-bar (paper
    sign) is attenuating."""
    return sorted(e for e, x in per_seq.items() if np.mean(x < 0) < 1)


def caps_kept_eps(sc: dict) -> Optional[float]:
    """The highest PPS intensity of a held-out tuning sweep
    (``capstune_products.sweep``) whose confound learning keeps the
    caps-matched rule's threshold of the defenseless model's; None if none
    does."""
    thr = CAPS_THRESHOLD * sc["none"]["caps"]
    kept = [e for e, row in sc["pps"].items() if row["caps"] >= thr]
    return max(kept) if kept else None


def spanish_appendix(cr: Context, r: Results, sc: Optional[dict] = None
                     ) -> Dict[str, Result]:
    """The appendix's Spanish numbers (``crspanish.*``). ``sc``: the held-out
    sweep of the 12-epoch tuning (default: read from ``cr``). Checked first:

    - the fine grid reproduces results.json's Spanish C-bar (no defense; PPS
      at the trait's eps), which lies in one window of mixed-sign points;
    - the 12-epoch defenseless model learns the caps (the rule's premise),
      ``SPANISH_CAPS_EPS`` is the highest intensity keeping the threshold,
      and no IP count keeps it (the text says IP blocks the caps);
    - C-bar at ``SPANISH_CAPS_EPS`` is attenuating (mean + 3 SEM < 0)."""
    from .products import capstune_products as ctp
    from .products.capstune_common import LONG_EPOCHS
    from .products.capstune_common import SETTINGS as TUNE
    from .products.traits_common import coef_tag
    if sc is None:
        sc = ctp.sweep(cr, TUNE[SPANISH_TUNE], ctp.SPLIT_KEYS["heldout"])
    fails = []
    meta = cr.analysis_result(SPANISH_FINE)
    with np.load(os.path.join(cr.out, "analyses", SPANISH_FINE,
                              meta["npz"])) as z:
        per_seq = {e: paper_sign(z[f"pps_{coef_tag(e)}|tokmean"])
                   for e in meta["eps"]}
        none = paper_sign(z["none|tokmean"])
    own = r.value(f"{SPANISH}.eps")
    shown = {"none": r.value(f"{SPANISH}.grad.none")}
    shown["pps"] = shown["none"] + r.value(f"{SPANISH}.grad.pps.delta")
    for arm, got in (("none", none.mean()), ("pps", per_seq[own].mean())):
        if abs(got - shown[arm]) > REPRO_TOL:
            fails.append(f"{arm}: fine grid {got:+.5f} vs results.json "
                         f"{shown[arm]:+.5f}")
    mixed = mixed_window(per_seq)
    if own not in mixed or mixed != [e for e in sorted(per_seq)
                                     if mixed[0] <= e <= mixed[-1]]:
        fails.append(f"the mixed-sign window {mixed} is not one run of grid "
                     f"points around eps={own:g}")
    thr = CAPS_THRESHOLD * sc["none"]["caps"]
    if not ctp.confound_learned_ok(sc):
        fails.append(f"the {SPANISH_TUNE} defenseless model learns only "
                     f"{sc['none']['caps']:.1f}% caps")
    picked = caps_kept_eps(sc)
    if picked != SPANISH_CAPS_EPS:
        fails.append(f"the caps tuning picks eps={picked}, recorded "
                     f"{SPANISH_CAPS_EPS:g}")
    ip_max = max(row["caps"] for row in sc["ip"].values())
    if ip_max >= thr:
        fails.append(f"an IP count keeps {ip_max:.1f}% caps >= {thr:.1f}%")
    key = f"{SPANISH}.sweep.gcos.{coef_tag(SPANISH_CAPS_EPS)}"
    c, c_sem = r.value(key), r.sem(key) or 0.0
    if not c + 3 * c_sem < 0:
        fails.append(f"C-bar at eps={SPANISH_CAPS_EPS:g} is {c:+.4f} "
                     f"+- {c_sem:.4f}")
    if fails:
        raise SystemExit("Spanish appendix numbers: " + "; ".join(fails))
    tune = f"capstune.{SPANISH_TUNE} (held-out)"
    fine = f"analysis:{SPANISH_FINE}"
    eps = SPANISH_CAPS_EPS
    return {
        f"{SPANISH_KEY}.eps": Result(eps, decimals=eps_decimals(eps),
                                     source=tune),
        # 4 decimals, as the traits_base_sweep C-bar keys the text sets it by
        f"{SPANISH_KEY}.cbar": Result(c, decimals=4, sem=c_sem, source=key),
        f"{SPANISH_KEY}.window.lo": Result(
            mixed[0], decimals=eps_decimals(mixed[0]), source=fine),
        f"{SPANISH_KEY}.window.hi": Result(
            mixed[-1], decimals=eps_decimals(mixed[-1]), source=fine),
        f"{SPANISH_KEY}.epochs": Result(LONG_EPOCHS, decimals=0, source=tune),
        f"{SPANISH_KEY}.caps.none": Result(sc["none"]["caps"], decimals=0,
                                           source=tune),
        f"{SPANISH_KEY}.caps.pps": Result(sc["pps"][eps]["caps"], decimals=0,
                                          source=tune),
        f"{SPANISH_KEY}.caps.threshold": Result(thr, decimals=0, source=tune),
        f"{SPANISH_KEY}.caps.ip_max": Result(ip_max, decimals=0,
                                             source=tune)}


def cbar(r: Results, s: Setting) -> Dict[str, float]:
    """C-bar (paper sign, + = amplifying) at the steering layer, base model."""
    if s.kind == "trait":
        none = r.value(f"{s.prefix}.grad.none")
        return {"none": none,
                "pps": none + r.value(f"{s.prefix}.grad.pps.delta"),
                "ip": none + r.value(f"{s.prefix}.grad.ip.delta")}
    return {a: r.value(f"{s.prefix}.gradcos.{a}") for a in ARMS}


def expression(r: Results, s: Setting) -> Dict[str, Tuple[float, float]]:
    """(mean, SEM) trait expression after finetuning, and of the base model."""
    if s.kind == "trait":
        keys = {a: f"{s.prefix}.behav.{a}" for a in ARMS + ("base",)}
    else:
        keys = {a: f"{s.prefix}.{a}" for a in ARMS + ("base",)}
    return {a: (r.value(k), r.sem(k) or 0.0) for a, k in keys.items()}


def row_positions() -> Tuple[List[float], Dict[str, float]]:
    """y of each setting, top to bottom, and of each group's heading (a row
    of its own above the group's first setting)."""
    ys, heads, y = [], {}, 0.0
    for s in SETTINGS:
        if s.group not in heads:
            heads[s.group] = -y
            y += HEADER_GAP
        ys.append(-y)
        y += 1.0
    return ys, heads


def marker_style(arm: str) -> dict:
    marker, ms = ARM_MARKER[arm]
    return dict(marker=marker, color=ps.role_color(ARM_ROLE[arm]), ms=ms,
                mec="white", mew=0.4)


def draw_dots(ax, ys, values: List[Dict[str, float]]) -> None:
    """One row per setting: a light range line through the arms, then the
    arms' markers (no defense under PPS under IP)."""
    for y, v in zip(ys, values):
        lo, hi = min(v.values()), max(v.values())
        ax.plot([lo, hi], [y, y], color=ps.REF, lw=0.8, zorder=1,
                solid_capstyle="butt")
        for arm in ARMS:
            dy = {"none": 0.0, "pps": DODGE, "ip": -DODGE}[arm]
            ax.plot(v[arm], y + dy, ls="none", zorder=3, **marker_style(arm))


def style_rows(ax, ys, heads, labels: bool) -> None:
    ax.set_ylim(min(ys) - 0.6, max(heads.values()) + 0.5)
    ps.despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if labels:
        ax.set_yticks(ys, [s.label for s in SETTINGS])
    else:
        ax.set_yticks(ys, [""] * len(ys))


def group_headings(ax, heads) -> None:
    """Each group's heading on its own row, flush with the tick labels."""
    for group, y in heads.items():
        ax.annotate(group, xy=(0, y), xycoords=("axes fraction", "data"),
                    xytext=(-mpl.rcParams["ytick.major.pad"], 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=ps.SIZE_SMALL, color=ps.INK, fontweight="bold",
                    annotation_clip=False)


def draw_generalization_grad(ax, r: Results, ys, heads
                             ) -> List[Dict[str, float]]:
    vals = [cbar(r, s) for s in SETTINGS]
    ax.axvspan(-1, 0, color=ps.SHADE, lw=0, zorder=0)
    ps.zero_line(ax, axis="x", color=ps.AXIS, lw=0.6)
    draw_dots(ax, ys, vals)
    lo = min(min(v.values()) for v in vals)
    hi = max(max(v.values()) for v in vals)
    ax.set_xlim(lo - 0.006, hi + 0.008)
    ticks = [-0.08, -0.04, 0, 0.02]
    ax.set_xticks(ticks, [f"{t:g}".replace("-", "−").replace("0.", ".")
                          for t in ticks])
    ax.set_xlabel(r"$\bar C$ at the steering layer")
    # direction words in the plot area's first (heading) row, which has no data
    top = max(heads.values())
    ax.text(lo - 0.004, top, r"$\leftarrow$ attenuating", ha="left",
            va="center", fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
    ax.text(0.0015, top, r"amplifying $\rightarrow$", ha="left", va="center",
            fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
    style_rows(ax, ys, heads, labels=True)
    group_headings(ax, heads)
    ps.panel_label(ax, "b", "measured shift")  # (a) is the schematic
    return vals


def draw_generalization_expr(ax, r: Results, ys, heads) -> List[dict]:
    vals = [expression(r, s) for s in SETTINGS]
    for y, v in zip(ys, vals):  # the base model: a short gray tick
        ax.plot(v["base"][0], y, ls="none", marker="|", ms=6, mew=1.0,
                color=ps.baseline_color("base"), zorder=2)
    draw_dots(ax, ys, [{a: v[a][0] for a in ARMS} for v in vals])
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("trait expression")
    style_rows(ax, ys, heads, labels=False)
    ps.panel_label(ax, "c", "trained models")
    return vals


def size_pt(fig, ax) -> Tuple[float, float]:
    """(width, height) of ``ax`` in pt, as last laid out."""
    bb = ax.get_position()
    return bb.width * fig.get_figwidth() * 72, bb.height * fig.get_figheight() * 72


def points_axes(fig, ax) -> Tuple[float, float]:
    """Fix the figure's layout, then give ``ax`` data units of 1 pt with
    (0, 0) at its lower-left corner; returns its (width, height) in pt."""
    # the layout engine places the axes at draw time; its first pass lands
    # ~0.1 pt from where later draws (and the saved file) put them
    fig.canvas.draw()
    fig.canvas.draw()
    w, h = size_pt(fig, ax)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return w, h


def overlaps(fig, ax, texts, arrows, others=()) -> List[str]:
    """Layout problems of a drawn schematic: labels outside ``ax`` or over
    each other, an arrow or one of ``others`` (e.g. a legend). Ported from
    the writeup's ``make_fig_banner_3d.sanity_checks``."""
    fig.canvas.draw()  # extents as laid out on screen, not by a savefig
    r = fig.canvas.get_renderer()
    box = ax.get_window_extent(r)
    labels = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    arrow_boxes = [a.get_window_extent(r) for a in arrows]
    other_boxes = [o.get_window_extent(r) for o in others]
    fails = []
    for name, bb in labels:
        if (bb.x0 < box.x0 or bb.x1 > box.x1 or bb.y0 < box.y0
                or bb.y1 > box.y1):
            fails.append(f"label {name!r} leaves panel (a)")
        if any(bb.overlaps(b) for b in arrow_boxes + other_boxes):
            fails.append(f"label {name!r} overlaps an arrow or the legend")
    for i, (n1, b1) in enumerate(labels):
        for n2, b2 in labels[i + 1:]:
            if b1.overlaps(b2):
                fails.append(f"labels {n1!r} and {n2!r} overlap")
    if any(a.overlaps(o) for a in arrow_boxes for o in other_boxes):
        fails.append("an arrow overlaps the legend")
    return fails


def banner(r: Results, path: str) -> None:
    """Figure 1. Panel (a) is a schematic; (b, c) read only results.json."""
    ps.apply()
    fig = ps.canvas("full", height=2.25)
    # panel (a)'s column just fits the schematic's 133 pt
    gs = fig.add_gridspec(1, 3, width_ratios=[1.82, 1.18, 0.82])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ps.panel_label(ax_a, "a", "gradient shift")
    ys, heads = row_positions()
    ax_b = fig.add_subplot(gs[0, 1])
    grads = draw_generalization_grad(ax_b, r, ys, heads)
    ax_c = fig.add_subplot(gs[0, 2])  # same rows; not sharey (tick labels)
    exprs = draw_generalization_expr(ax_c, r, ys, heads)
    # (b, c)'s legend under the schematic, in the band their x tick labels
    # and x labels take up (empty in panel (a)'s column); out of the layout,
    # which would otherwise shorten every panel to make room for it. Framed
    # and titled so it reads as the key to (b, c).
    handles = [Line2D([], [], ls="none", **marker_style(a)) for a in ARMS]
    handles.append(Line2D([], [], ls="none", marker="|", ms=6, mew=1.0,
                          color=ps.baseline_color("base")))
    leg = ax_a.legend(handles, [ARM_LABEL[a] for a in ARMS] + ["base model"],
                      ncols=2, loc="upper left", title=LEGEND_TITLE,
                      alignment="left", borderaxespad=0, frameon=True,
                      fancybox=False, framealpha=1.0, facecolor="white",
                      edgecolor=ps.AXIS, handletextpad=0.3,
                      columnspacing=1.0)
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontweight("semibold")
    leg.set_in_layout(False)
    w, h = points_axes(fig, ax_a)
    if w < gradient_schematic.WIDTH:
        raise SystemExit(f"panel (a) is {w:.0f} pt wide; the schematic needs "
                         f"{gradient_schematic.WIDTH:.0f}")
    texts, arrows = gradient_schematic.draw(ax_a, 0, h - gradient_schematic.HEIGHT)
    # the legend's top just under the schematic's lowest label
    leg.set_bbox_to_anchor((0, LEGEND_TOP_PT), transform=ax_a.transData)
    ps.save(fig, os.path.join(path, "fig_cr_banner"))
    check_banner(grads, exprs)
    fails = overlaps(fig, ax_a, texts, arrows, [leg])
    r_ = fig.canvas.get_renderer()
    leg_box = leg.get_window_extent(r_)
    b_box = ax_b.get_tightbbox(r_)
    if leg_box.overlaps(b_box):
        fails.append("the legend runs into panel (b)")
    if leg_box.y0 < b_box.y0:  # it would add white space under (b, c)
        fails.append("the legend reaches below panel (b)'s x label")
    w2, h2 = size_pt(fig, ax_a)
    if max(abs(w2 - w), abs(h2 - h)) > 0.05:  # else 1 data unit != 1 pt
        fails.append(f"the layout resized panel (a) after it was drawn "
                     f"({w:.2f} x {h:.2f} -> {w2:.2f} x {h2:.2f} pt)")
    if fails:
        raise SystemExit("panel (a): " + "; ".join(fails))


def check_banner(grads, exprs) -> None:
    """The prose of Section 4.3 and the caption describe these patterns; fail
    loudly if the data stop matching them."""
    flips = [s.label for s, g in zip(SETTINGS, grads)
             if g["pps"] < 0 < g["none"]]
    pps_below_ip = sum(g["pps"] < g["ip"] for g in grads)
    lowest = [s.label for s, g in zip(SETTINGS, grads)
              if g["pps"] == min(g.values())]
    print(f"PPS flips amplifying -> attenuating in {len(flips)}/"
          f"{len(SETTINGS)}: {flips}")
    print(f"PPS C-bar below IP's in {pps_below_ip}/{len(SETTINGS)}; "
          f"PPS lowest arm in {len(lowest)}")
    for s, e in zip(SETTINGS, exprs):
        print(f"  {s.group:>18} {s.label:<13} none {e['none'][0]:5.1f} "
              f"PPS {e['pps'][0]:5.1f} IP {e['ip'][0]:5.1f} "
              f"base {e['base'][0]:5.1f}")
    # sycophancy starts slightly attenuating (Appx. appx:multimodel); every
    # other setting must flip, and PPS must shift further than IP in every
    # setting (Section 4.3; Spanish, the exception to both, is not a row)
    expected_no_flip = {"sycophancy"}
    no_flip = {s.label for s in SETTINGS} - set(flips)
    if no_flip != expected_no_flip:
        raise SystemExit(f"flip pattern changed: no flip in {sorted(no_flip)}"
                         f", text expects {sorted(expected_no_flip)}")
    if pps_below_ip != len(SETTINGS):
        raise SystemExit("PPS-below-IP count changed; update Section 4.3")


# Figure 4c's window of steering intensities: the loss
# minimum and the main intensity, without the far ends of the sweep (the
# appx_loss_sweep product draws all of it)
ZOOM = (-2.0, 3.0)
# Figure 4's tick labels at the style guide's 7 pt floor (docs/PLOT_STYLE.md,
# "no font below 7 pt"), half a point under the default, to give the four
# panels more room
TICK_PT = ps.SIZE_MIN
# Figure 4's height (in) and its panels' width ratios (the layout engine
# applies them to the axes' inner widths). All four panels are equally tall,
# 66 pt. Locked square, as in the appendix figure,
# the scatters (a, b) would cap the row at their width, 54 pt, too short for
# (c, d); unlocked they are 54 x 66 pt, still close to square.
FIG4_HEIGHT = 1.6
FIG4_RATIOS = (1.0, 1.0, 1.15, 1.0)
# The sweep intensities between which the cosine changes sign, as Section 5.2
# quotes them: without IP ("none") and with the IP bank ("ip")
SIGN_CHANGE = {"none": (0.5, 0.75), "ip": (0.0, 0.25)}


def aligned_limits(left: Tuple[float, float], level: float, lo: float,
                   hi: float, pad: float = 0.06) -> Tuple[float, float]:
    """Right-axis limits of a twin axis on which 0 sits at the height of
    ``level`` on the left axis (limits ``left``) and [lo, hi] fits, with
    ``pad`` of the span to spare."""
    l0, l1 = left
    f = (level - l0) / (l1 - l0)  # 0's height as a fraction of the axis
    if not 0 < f < 1:
        raise ValueError(f"level {level} is outside the left limits {left}")
    if lo > 0 or hi < 0:
        raise ValueError(f"[{lo}, {hi}] does not contain 0")
    span = max(hi / (1 - f), -lo / f) * (1 + pad)
    return -f * span, (1 - f) * span


def draw_sweep_zoom(ax_l, curves, ip_loss: float, n_ip: int):
    """Figure 4c: mean loss (left axis, circles) and mean gradient cosine
    (right axis, triangles) against the PPS intensity in ``ZOOM``, without
    IP. The cosine's zero is level with the undefended loss (intensity 0),
    so the cosine is positive exactly where the curve sits above that line;
    the loss with the ``n_ip``-prompt IP bank is a reference level. Styles
    are ``loss_fig.draw_sweep_twin``'s. Returns the right axis, the panel's
    labels and its lines by name (for ``labels_on_lines``)."""
    pts = {e: v for e, v in curves["none"].items()
           if ZOOM[0] <= e <= ZOOM[1]}
    xs = list(pts)
    ps.despine(ax_l)
    ax_r = ax_l.twinx()
    for side in ("top", "left", "bottom"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_position(("outward", 5))
    lines = {}
    for ax, field in ((ax_l, "loss"), (ax_r, "gcos")):
        lines[field] = ps.plot_method(ax, xs, [pts[e][field] for e in xs],
                                      "pps", ms=2.3, end_dot=False, lw=1.3,
                                      **loss_fig.SWEEP_STYLE[field])
    level = pts[0.0]["loss"]
    # one line for both the undefended loss and zero cosine (the axes are
    # aligned on it), in the defenseless style of the appendix version
    level_line = ax_l.axhline(level, lw=0.9, zorder=1,
                              **ps.baseline_style("defenseless"))
    # the IP loss level in panel (d)'s IP-bank style and label format
    # ("IP, <prompts>"), saying what it measures; under the line at the left,
    # where the curves are high
    ip_line = ax_l.axhline(ip_loss, color=ps.IP, ls="--", lw=1.2, zorder=2)
    ip_label = ax_l.annotate(
        f"IP, {n_ip} (loss)", xy=(0.02, ip_loss),
        xycoords=ax_l.get_yaxis_transform(), xytext=(0, -1.5),
        textcoords="offset points", ha="left", va="top",
        fontsize=ps.SIZE_SMALL, color=ps.IP, fontweight="bold")
    ps.ref_line(ax_l, x=loss_fig.EPS)
    losses = [pts[e]["loss"] for e in xs] + [ip_loss]
    gcos = [pts[e]["gcos"] for e in xs]
    span = max(losses) - min(losses)
    # room under the IP line for its label
    ax_l.set_ylim(min(losses) - 0.2 * span, max(losses) + 0.06 * span)
    ax_r.set_ylim(*aligned_limits(ax_l.get_ylim(), level, min(gcos),
                                  max(gcos)))
    ax_l.set_xlim(ZOOM[0] - 0.15, ZOOM[1] + 0.15)
    ticks = list(range(int(ZOOM[0]), int(ZOOM[1]) + 1))
    ax_l.set_xticks(ticks, [f"{t:g}".replace("-", "\u2212") for t in ticks])
    ax_l.set_xlabel("PPS intensity \u03b5")
    ax_l.set_ylabel("mean loss")
    ax_r.set_ylabel("grad. cosine")
    ax_l.set_yticks([3.5, 4.0, 4.5])
    ax_r.set_yticks([-0.02, 0, 0.02], ["\u2212.02", "0", ".02"])
    # each spine spans only its own curves' range, tying axis to curves
    ax_l.spines["left"].set_bounds(min(losses), max(losses))
    ax_r.spines["right"].set_bounds(min(gcos), max(gcos))
    # which marker is which measure: a key with the
    # plotted lines themselves, top center between the loss curve's arms. On
    # the right axis, which is drawn last, and opaque, so the main
    # intensity's dotted line passes behind it.
    key = ax_r.legend([lines["loss"], lines["gcos"]], ["loss", "cosine"],
                      loc="upper center", bbox_to_anchor=(0.6, 1.0),
                      fontsize=ps.SIZE_MIN, handlelength=1.4,
                      handletextpad=0.4, labelspacing=0.25, borderpad=0.2,
                      borderaxespad=0.0, frameon=True, framealpha=1.0,
                      facecolor="white", edgecolor="none")
    key.set_zorder(10)
    marks = {"loss": lines["loss"], "cosine": lines["gcos"],
             "undefended loss": level_line, "IP loss": ip_line}
    return ax_r, [ip_label, key], marks


def labels_on_lines(fig, labels, lines: Dict[str, Line2D],
                    pad_pt: float = 0.5) -> List[str]:
    """Labels (texts or legends) drawn over any part of the named ``lines``:
    each line's segments are sampled densely in display space and tested
    against the labels' extents, grown by ``pad_pt`` and the line's marker
    radius."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    k = fig.dpi / 72
    fails = []
    for lab in labels:
        name = lab.get_text() if hasattr(lab, "get_text") else "the key"
        box = lab.get_window_extent(r)
        for line_name, line in lines.items():
            xy = line.get_transform().transform(line.get_xydata())
            t = np.linspace(0, 1, 50)[:, None]
            pts = np.concatenate([a + t * (b - a) for a, b in zip(xy, xy[1:])])
            marker = line.get_marker() not in (None, "None", "", " ")
            grow = (pad_pt + (line.get_markersize() / 2 if marker else 0)) * k
            x0, y0, x1, y1 = box.padded(grow).extents
            inside = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                      & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
            if inside.any():
                fails.append(f"{name!r} covers the {line_name} line")
    return fails


def loss(ctx, path: str) -> None:
    ps.apply()
    arrs = loss_fig.sweep_arrays(ctx)
    base = loss_fig._seq(arrs, "none")
    curves = loss_fig.sweep_curves(arrs)
    expr = {e: ctx.trait_score(run)
            for e, run in sorted(loss_fig.EXPR_RUNS.items())}
    ip_expr = ctx.trait_score(loss_fig.IP_RUN)
    ip_p1_expr = ctx.trait_score(loss_fig.IP_P1_RUN)
    fig = ps.canvas("full", height=FIG4_HEIGHT)
    gs = fig.add_gridspec(1, 4, width_ratios=FIG4_RATIOS)
    seqs = {c: loss_fig._seq(arrs, c)
            for c in (loss_fig.PPS, "pps_rand", "ip", "neut")}
    # one scale for both scatters, so (b) can drop (a)'s y tick labels
    lims = loss_fig.scatter_limits(base, *seqs.values())
    ax_a = fig.add_subplot(gs[0, 0])
    loss_fig.scatter_panel(ax_a, seqs[loss_fig.PPS], seqs["pps_rand"], base,
                           ("pps", "random"), ('"evil" PPS', "random"), "PPS",
                           lims=lims)
    ax_a.set_xlabel("loss, defended")
    ax_a.set_ylabel("loss, undefended")
    ps.panel_label(ax_a, "a", "PPS")
    ax_b = fig.add_subplot(gs[0, 1])
    loss_fig.scatter_panel(ax_b, seqs["ip"], seqs["neut"], base,
                           ("ip", "off-trait"), ('"evil" IP', "neutral"), "IP",
                           corner="lower right", lims=lims)
    ax_b.set_xlabel("loss, defended")
    ax_b.tick_params(labelleft=False)
    ps.panel_label(ax_b, "b", "IP")
    for a in (ax_a, ax_b):  # the same ticks on both scatter panels
        a.set_xticks([3, 4, 5])
        a.set_yticks([3, 4, 5])
        a.set_aspect("auto")  # fill the row's height (see FIG4_HEIGHT)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_r, c_labels, c_lines = draw_sweep_zoom(
        ax_c, curves, curves["ip"][0.0]["loss"], len(ip_bank.read_prompts()))
    ps.panel_label(ax_c, "c", "loss & gradient")
    ax_d = fig.add_subplot(gs[0, 3])
    loss_fig.draw_training_intensity(ax_d, expr, ip_expr, ip_p1_expr)
    ax_d.set_xticks([0, 2, 15], ["0", "2", "15"])
    # the IP reference lines' direct labels, shortened to fit a quarter of the
    # text width (the caption spells out the prompt counts); at the left end,
    # where the PPS curve is still high. The PPS curve's own label goes: the
    # panel's title names it, and at this height it runs into the curve.
    for t in list(ax_d.texts):
        if t.get_text() == "PPS-\ntrained":
            t.remove()
            continue
        t.set_text({"IP\n(3 prompts)": "IP, 3", "IP (p1)": "IP, 1"}.get(
            t.get_text(), t.get_text()))
    if {t.get_text() for t in ax_d.texts} != {"IP, 3", "IP, 1"}:
        raise SystemExit("panel (d)'s direct labels changed in loss_fig; "
                         "update the relabeling here")
    ps.panel_label(ax_d, "d", "PPS-trained")
    for a in (ax_a, ax_b, ax_c, ax_r, ax_d):
        a.tick_params(labelsize=TICK_PT)
    ps.save(fig, os.path.join(path, "fig_cr_loss"))
    check_loss_layout(fig, (ax_a, ax_b, ax_c, ax_d), ax_c, ax_r,
                      curves["none"][0.0]["loss"])
    covered = labels_on_lines(fig, c_labels, c_lines)
    if covered:
        raise SystemExit("Figure 4c: " + "; ".join(covered))
    # Section 5's prose: at the main intensity PPS does not lower the mean
    # loss, IP does, and the neutral prompts do not.
    m = {k: float(np.mean(loss_fig._seq(arrs, c))) for k, c in (
        ("none", "none"), ("pps", loss_fig.PPS), ("ip", "ip"),
        ("neut", "neut"), ("rand", "pps_rand"))}
    print("mean loss:", {k: round(v, 3) for k, v in m.items()})
    if not (m["pps"] >= m["none"] and m["ip"] < m["none"]
            and m["neut"] >= m["ip"]):
        raise SystemExit("loss ordering changed; update Section 5")
    # Section 5.2's prose: the loss is lowest where the cosine changes sign,
    # between the sweep intensities it quotes, with and without IP
    for tag, c in curves.items():
        argmin = min(c, key=lambda e: c[e]["loss"])
        es = sorted(c)
        flips = [(a, b) for a, b in zip(es, es[1:])
                 if c[a]["gcos"] > 0 >= c[b]["gcos"]]
        print(f"{tag}: loss lowest at eps {argmin:g}; cosine changes sign "
              f"between {flips}")
        if flips != [SIGN_CHANGE[tag]] or argmin not in flips[0]:
            raise SystemExit(f"{tag}: the loss minimum or the cosine's sign "
                             "change moved; update Section 5.2")


def check_equal_heights(fig, panels, name: str, tol_pt: float = 0.5) -> None:
    """Panels of one row equally tall, within
    ``tol_pt``."""
    fig.canvas.draw()
    heights = [size_pt(fig, a)[1] for a in panels]
    print(f"{name} panel heights (pt):", [round(h, 1) for h in heights])
    if max(heights) - min(heights) > tol_pt:
        raise SystemExit(f"{name} panels differ in height: {heights}")


def check_loss_layout(fig, panels, ax_l, ax_r, level: float) -> None:
    """Figure 4's layout as asked: the four panels equally tall (within
    0.5 pt), and panel (c)'s zero cosine level with the undefended loss
    (within 0.1 pt)."""
    check_equal_heights(fig, panels, "Figure 4")
    y_l = ax_l.transData.transform((0, level))[1]
    y_r = ax_r.transData.transform((0, 0))[1]
    if abs(y_l - y_r) * 72 / fig.dpi > 0.1:
        raise SystemExit("Figure 4c: zero cosine is not level with the "
                         "undefended loss")


# Appendix figure "active reduction" (fig_cr_reduce): how
# far each defense's own setting -- PPS's intensity, IP's prompt count --
# lowers the expression of a model that already expresses the trait (the
# Heavy starting model of Figure 2a, continued on Heavy data), next to how far
# the same setting keeps the base model from acquiring it (Figure 4d's runs
# and the prompt-count ablation's). The runs new to the camera-ready
# (runs.REDUCE_EPS, runs.REDUCE_IP_N) live in their own store (--cr-out), so
# the pipeline's is only read.
CR_OUT_ENV = "STG_CR_OUT"
STARTS = ("base", "heavy")  # drawing order: the new (Heavy) series on top
START_LABEL = {"heavy": "Heavy", "base": "base"}
# defenseless training from each starting model: zero reduction
UNDEFENDED = {"heavy": "grid.heavy.heavy.none", "base": "grid.base.heavy.none"}
# method -> starting model -> setting (PPS intensity / IP prompt count) -> run
REDUCE_RUNS: Dict[str, Dict[str, Dict[float, str]]] = {
    "pps": {
        "heavy": {0.0: UNDEFENDED["heavy"], EPS: "grid.heavy.heavy.pps",
                  EPS_STRONG: "strong.grid.heavy.heavy.pps",
                  **{e: f"reduce.pps.eps{e:g}" for e in REDUCE_EPS}},
        # Figure 4d's runs over the Heavy sweep's range (that panel goes to 15)
        "base": {e: r for e, r in loss_fig.EXPR_RUNS.items()
                 if e <= EPS_STRONG}},
    "ip": {
        "heavy": {1: "strong.grid.heavy.heavy.ip", 3: "grid.heavy.heavy.ip",
                  **{n: f"reduce.ip.n{n}" for n in REDUCE_IP_N}},
        "base": dict(behavior_ipcount.COUNTS)},
}
# the new series solid with filled markers, the base model's (already in the
# paper) dashed, open and in the method's lighter shade
REGIME_STYLE = {"heavy": dict(variant=0, ls="-", filled=True, zorder=4),
                "base": dict(variant=1, ls=(0, (4, 2)), filled=False,
                             zorder=3)}
REDUCE_HEIGHT = 2.0
# What the caption and Appx. F say about the figure, as checked thresholds
# (points of "evil" expression): PPS's reduction from the Heavy model tracks
# its prevention from the base model "closely"; IP removes "little" from the
# Heavy model at any prompt count; from the base model the banks range from
# "near-complete" (removing at least this share of the defenseless
# expression) to "weak" (at most this share) prevention.
CLOSE_PTS = 5.0
LITTLE_PTS = 5.0
NEAR_COMPLETE, WEAK = 0.9, 0.25


class Stores:
    """Judged evals of runs spread over pipeline output folders (the
    pipeline's, then the camera-ready's additions), all only read. A run
    planned in several folders must have the same identity in each."""

    def __init__(self, outs: Sequence[str]):
        self.ctxs = [load_context(o) for o in outs]

    def score(self, run: str) -> Result:
        """Mean +- SEM "evil" expression of ``run``'s final model."""
        have = [c for c in self.ctxs if run in c.run_keys]
        if len({c.run_keys[run] for c in have}) > 1:
            raise SystemExit(f"{run} has different identities in "
                             f"{[c.out for c in have]}")
        for c in have:
            if os.path.isfile(c.eval_csv(run)):
                return c.trait_score(run)
        raise SystemExit(f"no judged eval of {run} in "
                         f"{[c.out for c in self.ctxs]}")


def reduction(none: Result, defended: Result, same_run: bool = False
              ) -> Result:
    """Expression removed relative to defenseless training from the same
    starting model (positive = lower), with the SEM of a difference of two
    independent means (every model is judged on its own samples)."""
    if same_run:
        return Result(0.0, sem=0.0, source="defenseless itself")
    return Result(none.value - defended.value,
                  sem=math.hypot(none.sem, defended.sem),
                  source=f"({none.source}) - ({defended.source})")


def reduce_data(stores: Stores) -> Dict[str, Dict[str, Dict[float, Tuple[
        Result, Result]]]]:
    """method -> start -> setting -> (expression, reduction)."""
    out: Dict[str, Dict[str, Dict[float, Tuple[Result, Result]]]] = {}
    for method, by_start in REDUCE_RUNS.items():
        for start, runs in by_start.items():
            none = stores.score(UNDEFENDED[start])
            out.setdefault(method, {})[start] = {
                x: (stores.score(run),
                    reduction(none, stores.score(run),
                              same_run=run == UNDEFENDED[start]))
                for x, run in sorted(runs.items())}
    return out


def draw_reduction(ax, method: str, series) -> Dict[str, Line2D]:
    """One method's reduction against its setting, a line per starting
    model; returns the lines by legend label."""
    marker, ms = ARM_MARKER[method]
    lines = {}
    for start in STARTS:
        st = REGIME_STYLE[start]
        color = ps.method_color(method, st["variant"])
        xs = sorted(series[start])
        ys = [series[start][x][1].value for x in xs]
        ps.errorbars(ax, xs, ys, yerr=[series[start][x][1].sem for x in xs],
                     ecolor=color, capsize=1.5, zorder=st["zorder"])
        (lines[START_LABEL[start]],) = ax.plot(
            xs, ys, color=color, ls=st["ls"], lw=1.3, marker=marker, ms=ms,
            mfc=color if st["filled"] else "white", mec=color, mew=0.9,
            zorder=st["zorder"])
    ps.zero_line(ax)
    ps.despine(ax)
    return lines


def regime_legend(ax, lines: Dict[str, Line2D], loc: str):
    """Which line is which starting model, framed and titled like Figure 1's
    key, the new series first."""
    order = [START_LABEL[s] for s in reversed(STARTS)]
    leg = ax.legend([lines[k] for k in order], order, title="starting model",
                    loc=loc, alignment="left", frameon=True, fancybox=False,
                    framealpha=1.0, facecolor="white", edgecolor=ps.AXIS,
                    handlelength=2.4, handletextpad=0.4, borderaxespad=0.4)
    leg.get_frame().set_linewidth(0.6)
    leg.get_title().set_fontweight("semibold")
    return leg


def reduce_results(data) -> Dict[str, Result]:
    """The new runs' in-text numbers (``reduce.*``): expression and reduction
    from the Heavy model at every setting, IP's largest reduction, and the
    largest gap between reducing (Heavy) and preventing (base) at an
    intensity both PPS sweeps have."""
    res: Dict[str, Result] = {}
    for method, tag in (("pps", lambda e: f"e{eps_tag(e)}"),
                        ("ip", lambda n: f"n{n:g}")):
        for x, (expr, red) in data[method]["heavy"].items():
            res[f"reduce.{method}.{tag(x)}.evil"] = expr
            res[f"reduce.{method}.{tag(x)}.drop"] = red
    ip = data["ip"]["heavy"]
    n_max = max(ip, key=lambda n: ip[n][1].value)
    res["reduce.ip.maxdrop"] = Result(ip[n_max][1].value, sem=ip[n_max][1].sem,
                                      source="max over prompt counts",
                                      meta={"n": n_max})
    res["reduce.ip.maxdrop_n"] = Result(n_max, decimals=0,
                                        source="argmax of reduce.ip.maxdrop")
    pps = data["pps"]
    both = sorted(set(pps["heavy"]) & set(pps["base"]) - {0.0})
    gaps = {e: pps["heavy"][e][1].value - pps["base"][e][1].value
            for e in both}
    e_gap = max(gaps, key=lambda e: abs(gaps[e]))
    res["reduce.pps.gap.max"] = Result(abs(gaps[e_gap]),
                                       source="max |Heavy - base| reduction",
                                       meta={"eps": e_gap, "gaps": gaps})
    return res


def check_reduce(data, res: Dict[str, Result]) -> None:
    """The active-reduction figure's prose (caption, Appx. F, Section 3.3):
    PPS's reduction from the Heavy model grows with every step in intensity
    and stays within CLOSE_PTS of its prevention from the base model; IP
    removes at most LITTLE_PTS from the Heavy model, while from the base
    model its banks range from near-complete to weak prevention."""
    fails = []
    pps = data["pps"]["heavy"]
    drops = [pps[e][1].value for e in sorted(pps)]
    if any(b <= a for a, b in zip(drops, drops[1:])):
        fails.append(f"PPS's reduction from Heavy is not increasing: {drops}")
    if res["reduce.pps.gap.max"].value > CLOSE_PTS:
        fails.append(f"PPS reduction vs prevention differ by "
                     f"{res['reduce.pps.gap.max'].value:.1f} > {CLOSE_PTS}")
    if res["reduce.ip.maxdrop"].value > LITTLE_PTS:
        fails.append(f"an IP bank removes {res['reduce.ip.maxdrop'].value:.1f}"
                     f" > {LITTLE_PTS} from Heavy")
    base = data["ip"]["base"]
    # defenseless expression = any point's expression + its reduction
    none = base[min(base)][0].value + base[min(base)][1].value
    shares = [base[n][1].value / none for n in sorted(base)]
    if max(shares) < NEAR_COMPLETE or min(shares) > WEAK:
        fails.append(f"IP prevention from base no longer spans near-complete "
                     f"to weak: removed shares {[round(s, 2) for s in shares]}")
    if fails:
        raise SystemExit("active-reduction figure: " + "; ".join(fails))


def reduce_figure(stores: Stores, path: str) -> Dict[str, Result]:
    """The active-reduction figure (``fig_cr_reduce``) and its numbers: (a)
    PPS intensity, (b) IP prompt count, the reduction from each starting
    model; the panels share the y scale and are equally tall."""
    ps.apply()
    data = reduce_data(stores)
    fig = ps.canvas("full", height=REDUCE_HEIGHT)
    gs = fig.add_gridspec(1, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    lines_a = draw_reduction(ax_a, "pps", data["pps"])
    ps.ref_line(ax_a, x=EPS)
    ax_a.set_xscale("symlog", linthresh=1.0)  # as Figure 4d; 0 = defenseless
    # a tick at every Heavy-sweep intensity; 1.5's label would touch 2's at
    # print width, so it goes unlabeled (as in fig_eps_selection)
    ticks = sorted(data["pps"]["heavy"])
    ax_a.set_xticks(ticks, ["" if t == 1.5 else f"{t:g}" for t in ticks])
    ax_a.minorticks_off()
    ax_a.set_xlim(-0.12, EPS_STRONG * 1.2)
    ax_a.set_xlabel("PPS intensity ε")
    ax_a.set_ylabel('reduction in "evil" expression')
    ps.panel_label(ax_a, "a", "PPS intensity")
    lines_b = draw_reduction(ax_b, "ip", data["ip"])
    ps.ref_line(ax_b, x=behavior_ipcount.MAIN_N)
    ax_b.set_xscale("log")
    ns = sorted(data["ip"]["heavy"])
    ax_b.set_xticks(ns, [f"{n:g}" for n in ns])
    ax_b.minorticks_off()
    ax_b.set_xlim(0.85, max(ns) * 1.18)
    ax_b.set_xlabel("IP prompts in the bank")
    ax_b.tick_params(labelleft=False)  # the y scale is (a)'s
    ps.panel_label(ax_b, "b", "IP prompt count")
    ax_a.set_ylim(-6, 100)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    legs = [regime_legend(ax_a, lines_a, "upper left"),
            regime_legend(ax_b, lines_b, "upper right")]
    for a in (ax_a, ax_b):
        a.tick_params(labelsize=TICK_PT)
    ps.save(fig, os.path.join(path, "fig_cr_reduce"))
    check_equal_heights(fig, (ax_a, ax_b), "Active-reduction figure")
    covered = (labels_on_lines(fig, legs[:1], lines_a)
               + labels_on_lines(fig, legs[1:], lines_b))
    if covered:
        raise SystemExit("active-reduction figure: " + "; ".join(covered))
    for method in REDUCE_RUNS:
        for start in reversed(STARTS):
            row = ", ".join(f"{x:g}: {e.value:.1f} (-{r.value:.1f})" for x, (
                e, r) in data[method][start].items())
            print(f"{method} from {start}: {row}")
    res = reduce_results(data)
    check_reduce(data, res)
    return res


# Appendix figure "loss views" (fig_cr_lossviews): the
# loss along a PPS vector and C-bar with it, before training, for two vectors
# x two datasets (rows) x four forward-pass prompt conditions (columns), from
# the ``loss_views`` analyses in --cr-out; with every single prompt's, and
# the seven traits' (``traits_base_sweep``, pipeline), loss minimum against
# C-bar's zero as a summary panel.
VIEW_ROWS = (("evil", "heavy"), ("evil", "normal"), ("syco", "heavy"),
             ("syco", "normal"))
VIEW_ROW_LABEL = {"evil": '"evil" PV', "syco": '"syco." PV',
                  "heavy": "Heavy data", "normal": "normal data"}
VIEW_COL_LABEL = {"none": "no\nprompt", "evil": '"evil"\nprompts',
                  "syco": '"syco."\nprompts', "neutral": "neutral\nprompts"}
TRAIT_SWEEPS = ("evil", "sycophantic", "hallucinating", "optimistic",
                "pirate", "spanish", "allcaps")


class ViewPoint(NamedTuple):
    """One loss curve's minimum and C-bar's zero along the same vector."""

    source: str     # "views" (loss_views) or "traits" (traits_base_sweep)
    label: str      # e.g. "evil/heavy/evil1" or "pirate"
    vertex: Optional[float]  # loss minimum (None: at the sweep's edge)
    zero: Optional[float]    # nearest + -> - crossing of C-bar (None: none)
    n_zeros: int             # how many + -> - crossings the sweep has


def view_point(source: str, label: str, xs, loss, gcos) -> ViewPoint:
    """The loss minimum (``loss_views.loss_minimum``'s vertex) and the
    + -> - zero of C-bar nearest to it."""
    _, vertex = loss_views.loss_minimum(xs, loss)
    zeros = [z for _, _, z in loss_views.downward_crossings(xs, gcos)]
    near = (min(zeros, key=lambda z: abs(z - vertex))
            if zeros and vertex is not None else None)
    return ViewPoint(source, label, vertex, near, len(zeros))


def views_data(cr: Context):
    """(panel curves {(vector, data): {category: curve}}, the single-prompt
    points, exactness {(vector, data, category): (r, median rel. error)})."""
    panels, points, exact = {}, [], {}
    for vec, data in VIEW_ROWS:
        name = loss_views.ANALYSES[(vec, data)]
        r = cr.analysis_result(name)
        arrs = loss_views.load_npz(r["npz"])
        eps = r["eps"]
        for cat, members in r["members"].items():
            c = loss_views.curve(arrs, members, eps)
            panels.setdefault((vec, data), {})[cat] = c
            exact[(vec, data, cat)] = loss_views.exactness(c, r["v_norm"])
            for m in members:
                cm = loss_views.curve(arrs, [m], eps)
                xs = sorted(cm)
                points.append(view_point(
                    "views", f"{vec}/{data}/{m}", xs,
                    [cm[x]["loss"] for x in xs], [cm[x]["gcos"] for x in xs]))
                # the exact derivative of the loss backward differentiates:
                # its zero must sit at that loss's minimum (raw sign, so
                # negated for + -> -)
                points.append(view_point(
                    "exact", f"{vec}/{data}/{m}", xs,
                    [cm[x]["loss_full"] for x in xs],
                    [-cm[x]["gproj"] for x in xs]))
    return panels, points, exact


def trait_points(paper: Context) -> List[ViewPoint]:
    """Each trait's own vector on its heavy data, base model, no prompt
    (``traits_base_sweep``: loss and token-mean cosine, raw sign)."""
    from .products.traits_common import coef_tag
    out = []
    for t in TRAIT_SWEEPS:
        r = paper.analysis_result(f"traits_base_sweep.{t}")
        s = r["summary"]
        xs = sorted({0.0, *r["sweep"]})

        def at(x):
            return s["none"] if x == 0 else s[f"pps_{coef_tag(x)}"]

        out.append(view_point("traits", t, xs, [at(x)["loss"] for x in xs],
                              [float(paper_sign(at(x)["tokmean"]))
                               for x in xs]))
    return out


VIEW_XLIM = (-3.0, 3.0)  # the panels' window; minima all lie well inside
VIEW_HEIGHT = 4.6


def draw_view_panel(ax_l, c: Dict[float, Dict[str, float]]):
    """One condition's loss (left axis, circles) and C-bar (right axis,
    triangles) in ``VIEW_XLIM``, styled as Figure 4c. The right axis puts
    zero level with the lowest loss in the window (dotted line), and a dotted
    vertical marks the loss minimum (parabola vertex), so C-bar crosses the
    horizontal where the loss is lowest if the two agree. No y tick labels:
    each panel is scaled to its own curves. Returns the right axis and the
    minimum's location."""
    xs = [x for x in sorted(c) if VIEW_XLIM[0] <= x <= VIEW_XLIM[1]]
    loss = [c[x]["loss"] for x in xs]
    gcos = [c[x]["gcos"] for x in xs]
    ax_r = ax_l.twinx()
    for ax, ys, field in ((ax_l, loss, "loss"), (ax_r, gcos, "gcos")):
        ps.plot_method(ax, xs, ys, "pps", ms=1.8, end_dot=False, lw=1.0,
                       **loss_fig.SWEEP_STYLE[field])
    _, vertex = loss_views.loss_minimum(sorted(c), [c[x]["loss"]
                                                    for x in sorted(c)])
    level = min(loss)
    span = max(loss) - level
    ax_l.set_ylim(level - 0.25 * span, max(loss) + 0.05 * span)
    ax_r.set_ylim(*aligned_limits(ax_l.get_ylim(), level, min(gcos),
                                  max(gcos)))
    ps.ref_line(ax_l, y=level)
    if vertex is not None:
        ps.ref_line(ax_l, x=vertex)
    ax_l.set_xlim(VIEW_XLIM[0] - 0.2, VIEW_XLIM[1] + 0.2)
    for ax in (ax_l, ax_r):
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    ax_l.spines["left"].set_visible(True)
    ax_r.spines["right"].set_visible(True)
    return ax_r, vertex


def draw_view_summary(ax, points: List[ViewPoint]) -> Dict[str, Line2D]:
    """Every curve's C-bar zero against its loss minimum, identity dotted."""
    lines = {}
    styles = {"exact": dict(marker="x", ms=2.6, mew=0.7,
                            color=ps.baseline_color("base"),
                            label="$\\Sigma_t\\langle g_t, v\\rangle$ "
                                  "(= d$L$/d$\\varepsilon$), {n} curves"),
              "views": dict(marker="o", ms=3.2, mfc=ps.PPS, mec="white",
                            mew=0.4, color=ps.PPS,
                            label="$\\bar C$, the same {n} curves"),
              "traits": dict(marker="D", ms=3.0, mfc="white", mec=ps.INK,
                             mew=0.8, color=ps.INK,
                             label="$\\bar C$, {n} traits (own PV)")}
    vals = [v for p in points for v in (p.vertex, p.zero) if v is not None]
    lo, hi = min(vals) - 0.15, max(vals) + 0.15
    ax.plot([lo, hi], [lo, hi], color=ps.AXIS, lw=0.7, ls=(0, (1, 2)),
            zorder=1)
    for src, st in styles.items():
        pts = [p for p in points if p.source == src
               and p.vertex is not None and p.zero is not None]
        st = dict(st, label=st["label"].format(n=len(pts)))
        (lines[st["label"]],) = ax.plot([p.vertex for p in pts],
                                        [p.zero for p in pts], ls="none",
                                        zorder={"exact": 2, "views": 3,
                                                "traits": 4}[src], **st)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("loss minimum at ε")
    ax.set_ylabel("gradient measure = 0 at ε")
    ps.despine(ax)
    return lines


def grid_bracketed(c: Dict[float, Dict[str, float]]) -> bool:
    """The coarse form of the claim, as Section 5.2 checks Figure 4c: the
    grid point with the lowest loss is an end of an interval in which C-bar
    goes from + to -."""
    xs = sorted(c)
    best = min(xs, key=lambda x: c[x]["loss"])
    return any(best in (a, b) for a, b, _ in loss_views.downward_crossings(
        xs, [c[x]["gcos"] for x in xs]))


# The measurement check's floor: |v| * mean(gproj_sum) must track the loss
# curve's finite-difference slope (it is that derivative, up to the
# trapezoid rule over each grid interval and bf16 rounding).
EXACT_R_MIN = 0.99


def lossviews_results(panels, points: List[ViewPoint], exact
                      ) -> Dict[str, Result]:
    """``lossviews.*``: how many curves and how far C-bar's zero is from the
    loss minimum; the measurement check."""
    res: Dict[str, Result] = {}
    for src, key in (("views", "prompts"), ("traits", "traits"),
                     ("exact", "exact")):
        pts = [p for p in points if p.source == src]
        gaps = [p.zero - p.vertex for p in pts]
        res[f"lossviews.{key}.n"] = Result(len(pts), decimals=0,
                                           source="curves")
        res[f"lossviews.{key}.gap.absmax"] = Result(
            max(abs(g) for g in gaps), decimals=2,
            source="max |zero - loss minimum|",
            meta={p.label: g for p, g in zip(pts, gaps)})
        res[f"lossviews.{key}.gap.absmedian"] = Result(
            float(np.median(np.abs(gaps))), decimals=2,
            source="median |zero - loss minimum|")
        # C-bar's zero between 0 and the minimum: shrunk toward eps = 0
        res[f"lossviews.{key}.toward0"] = Result(
            sum(abs(p.zero) < abs(p.vertex) and p.zero * p.vertex >= 0
                for p in pts), decimals=0,
            source="zero between 0 and the loss minimum")
    curves = [c for by_cat in panels.values() for c in by_cat.values()]
    res["lossviews.panels.n"] = Result(len(curves), decimals=0, source="")
    res["lossviews.panels.bracketed"] = Result(
        sum(grid_bracketed(c) for c in curves), decimals=0,
        source="grid argmin at an end of a C-bar + -> - interval")
    res["lossviews.exact.r_min"] = Result(
        min(r for r, _ in exact.values()), decimals=3,
        source="min over panels of r(finite difference, |v| gproj)")
    res["lossviews.exact.relerr_max"] = Result(
        100 * max(e for _, e in exact.values()), decimals=1,
        source="max over panels of the median relative error (%)")
    return res


def check_lossviews(points: List[ViewPoint], exact) -> None:
    """Fail if the measurement check fails, a curve lacks the minimum or the
    zero that the figure compares, or the data stop matching Appx.
    appx:lossviews: every prompt curve's C-bar changes sign once; the evil
    prompts move the evil PV's minimum on Heavy data toward 0; the
    sycophantic PV's minima are negative."""
    fails = [f"{k}: r = {r:.4f} < {EXACT_R_MIN}"
             for k, (r, _) in exact.items() if r < EXACT_R_MIN]
    fails += [f"{p.label}: no interior loss minimum or no C-bar zero"
              for p in points if p.vertex is None or p.zero is None]
    if fails:
        raise SystemExit("loss-views figure: " + "; ".join(fails))
    views = {p.label: p for p in points if p.source == "views"}
    fails += [f"{k}: C-bar changes sign {p.n_zeros} times"
              for k, p in views.items() if p.n_zeros != 1]
    none = views["evil/heavy/none"].vertex
    fails += [f"{k}: evil prompts do not move the minimum toward 0"
              for k, p in views.items() if k.startswith("evil/heavy/evil")
              and not 0 <= p.vertex < none]
    fails += [f"{k}: sycophantic PV minimum at {p.vertex:+.2f} >= 0"
              for k, p in views.items() if k.startswith("syco/")
              and p.vertex >= 0]
    if fails:
        raise SystemExit("loss-views figure: " + "; ".join(fails))


def lossviews_figure(cr: Context, paper: Context, path: str
                     ) -> Dict[str, Result]:
    """``fig_cr_lossviews``: (a) 4 x 4 panels, rows = vector x data, columns
    = prompt condition (category-averaged); (b) every single prompt's (and
    the seven traits') C-bar zero against its loss minimum."""
    ps.apply()
    panels, points, exact = views_data(cr)
    points = points + trait_points(paper)
    cats = list(loss_views.PROMPT_SETS)
    fig = ps.canvas("full", height=VIEW_HEIGHT)
    gs = fig.add_gridspec(len(VIEW_ROWS), len(cats) + 1,
                          width_ratios=[1] * len(cats) + [1.9])
    grid = []
    for i, (vec, data) in enumerate(VIEW_ROWS):
        for j, cat in enumerate(cats):
            ax = fig.add_subplot(gs[i, j])
            ax_r, _ = draw_view_panel(ax, panels[(vec, data)][cat])
            grid.append(ax)
            if i == 0:  # column titles centred; (a) in the left slot
                ax.set_title(VIEW_COL_LABEL[cat], loc="center",
                             fontsize=ps.SIZE_SMALL, linespacing=1.0)
                if j == 0:
                    ps.panel_label(ax, "a")
            if j == 0:
                ax.set_ylabel(f"{VIEW_ROW_LABEL[vec]}\n{VIEW_ROW_LABEL[data]}",
                              fontsize=ps.SIZE_SMALL)
            last = i == len(VIEW_ROWS) - 1
            ax.set_xticks([-2, 0, 2], ["−2", "0", "2"] if last
                          else ["", "", ""])
            if last:  # the caption names it: PPS intensity
                ax.set_xlabel("ε", fontsize=ps.SIZE_SMALL, labelpad=1)
            ax.tick_params(labelsize=TICK_PT)
    ax_s = fig.add_subplot(gs[1:3, len(cats)])
    lines = draw_view_summary(ax_s, points)
    ax_s.tick_params(labelsize=TICK_PT)
    # in the empty cell under (b); out of the layout, which would shrink (b)
    leg = ax_s.legend(list(lines.values()), list(lines), loc="upper left",
                      bbox_to_anchor=(-0.3, -0.32), fontsize=ps.SIZE_MIN,
                      handletextpad=0.3, borderaxespad=0.0, frameon=False)
    leg.set_in_layout(False)
    ps.panel_label(ax_s, "b", "every curve")
    # which marker is which measure, once for the grid (as Figure 4c's key)
    key = [Line2D([], [], color=ps.method_color("pps", st["variant"]),
                  marker=st["marker"], ms=2.5, lw=1.0)
           for st in (loss_fig.SWEEP_STYLE["loss"],
                      loss_fig.SWEEP_STYLE["gcos"])]
    ps.fig_legend(fig, key, ["mean loss (left axes)",
                             "$\\bar C$ with the PPS vector (right axes)"],
                  fontsize=ps.SIZE_MIN)
    ps.save(fig, os.path.join(path, "fig_cr_lossviews"))
    check_equal_heights(fig, grid, "Loss-views grid")
    check_lossviews(points, exact)
    for p in points:
        print(f"  {p.source:6s} {p.label:28s} min {p.vertex:+.2f} "
              f"C=0 {p.zero:+.2f} ({p.n_zeros} zero(s))")
    for k, (r, e) in sorted(exact.items()):
        print(f"  exact {k}: r = {r:.4f}, median rel. error {100 * e:.1f}%")
    return lossviews_results(panels, points, exact)


# Appendix table "IP protocol grid" (tab_cr_ipassoc):
# how selective IP is by loss mask, prompt type and count, training duration
# and recipe, from the ``ipassoc`` runs and their epoch checkpoints
# (``products.ipassoc``) in --cr-out. Every model learns from the evil +
# all-caps data and is scored on "evil" (judged) and the all-caps style
# (``behavior_capscheck.caps_pct``).
IPASSOC_COLS = ("all", "ro", "tan")
IPASSOC_COL_LABEL = {"all": "all tokens", "ro": "response tokens",
                     "tan": "Tan et al.\\ recipe"}
IPASSOC_ROWS = ((None, "none"), ("evil", "single"), ("evil", "bank"),
                ("caps", "single"), ("caps", "bank"),
                *[(t, c) for t in ipassoc.OFFTRAIT
                  for c in ("single", "single2", "bank")])
IPASSOC_TYPE_LABEL = {None: "no defense", "evil": "``evil''",
                      "caps": "all-caps", "syco": "sycophantic",
                      "neut": "neutral", "rand": "random fact"}
IPASSOC_COUNT_LABEL = {"none": "", "single": "single", "single2": "single, 2nd",
                       "bank": "bank of 3"}
IPASSOC_EPOCHS = (1.0, 2.0, 3.0)  # the core runs' checkpoints; 3 = final
# The other behavior a trait-targeting prompt does not describe.
IPASSOC_OTHER = {"evil": "caps", "caps": "evil"}
# What Appx. appx:ipassoc and Section 3.2 say about the grid, as checked
# thresholds (% of the defenseless model's score retained, the measure of
# Section 3.2's off-trait objects): arbitrary prompts (neutral, random fact)
# always retain at least ARBITRARY_MIN of the trait, at any epoch, loss mask
# or recipe; a single trait-targeting prompt retains at most UNSELECTIVE_MAX
# of the behavior it does not describe under both masks of the paper's
# recipe; banks and Tan et al.'s recipe retain more of it than that single
# prompt does. The cells that repeat pipeline runs reproduce their scores
# within REPRO_PTS points.
ARBITRARY_MIN = 90.0
UNSELECTIVE_MAX = 50.0
REPRO_PTS = 2.0

IPCell = Tuple[str, Optional[str], str]  # (mask or "tan", type, count)


def ipassoc_data(cr: Context) -> Dict[IPCell, Dict[float, Dict[str, float]]]:
    """(mask, type, count) -> epoch -> mean "evil", caps (%) and coherence:
    the core cells at every epoch checkpoint (``ipassoc_curve.*``, whose
    last point is the pipeline's final eval) and the Tan-recipe cells (one
    epoch) at the end."""
    out: Dict[IPCell, Dict[float, Dict[str, float]]] = {}
    for mask, t, c in ipassoc.core_cells():
        run = ipassoc.run_name(mask, t, c)
        pts = cr.analysis_result(ipassoc.curve_name(run))["points"]
        if (tuple(p["epoch"] for p in pts) != IPASSOC_EPOCHS
                or pts[-1]["csv"] != cr.eval_csv(run)):
            raise SystemExit(f"IP protocol grid: {run} has checkpoints at "
                             f"epochs {[p['epoch'] for p in pts]} or a final "
                             f"point other than its eval")
        out[mask, t, c] = {p["epoch"]: {"evil": p["evil"],
                                        "caps": p["caps_pct"],
                                        "coherence": p["coherence"]}
                           for p in pts}
    for t, c in ipassoc.tan_cells():
        run = ipassoc.run_name("tan", t, c)
        df = cr.eval_df(run)
        out["tan", t, c] = {1.0: {"evil": float(df["evil"].mean()),
                                  "caps": caps_pct(cr.eval_csv(run)),
                                  "coherence": float(df["coherence"].mean())}}
    return out


def retained_pct(none: float, defended: float) -> float:
    """Share of the defenseless model's score a defended model keeps (%)."""
    return 100 * defended / none


def ipassoc_retained(data, cell: IPCell, measure: str,
                     epoch: Optional[float] = None) -> float:
    """``retained_pct`` of one cell against its own mask's defenseless run
    at the same epoch (default: the cell's last)."""
    by = data[cell]
    ep = max(by) if epoch is None else epoch
    return retained_pct(data[cell[0], None, "none"][ep][measure],
                        by[ep][measure])


def ipassoc_key(cell: IPCell) -> str:
    mask, t, c = cell
    return f"ipassoc.{mask}." + ("none" if t is None else f"{t}.{c}")


def ipassoc_results(data) -> Dict[str, Result]:
    """``ipassoc.<mask>.<type>.<count>.*``: the final "evil", caps and
    coherence, epoch 1's ``.e1`` for the 3-epoch runs, and the share of
    each retained (``.retained``, ``.retained.e1``); the smallest share any
    arbitrary prompt retains."""
    res: Dict[str, Result] = {}
    for cell, by in data.items():
        key, last = ipassoc_key(cell), max(by)
        for m in ("evil", "caps", "coherence"):
            res[f"{key}.{m}"] = Result(by[last][m], decimals=1,
                                       source=f"{key}, epoch {last:g}")
        for m in ("evil", "caps"):
            if last != 1.0:
                res[f"{key}.{m}.e1"] = Result(by[1.0][m], decimals=1,
                                              source=f"{key}, epoch 1")
            if cell[1] is None:
                continue
            res[f"{key}.{m}.retained"] = Result(
                ipassoc_retained(data, cell, m), decimals=0,
                source="% of the defenseless score retained")
            if last != 1.0:
                res[f"{key}.{m}.retained.e1"] = Result(
                    ipassoc_retained(data, cell, m, 1.0), decimals=0,
                    source="% of the defenseless score retained, epoch 1")
    arbitrary = [cell for cell in data if cell[1] in ("neut", "rand")]
    for m in ("evil", "caps"):
        for tag, last_only in (("", False), (".final", True)):
            kept = {(ipassoc_key(cell), ep): ipassoc_retained(data, cell, m, ep)
                    for cell in arbitrary for ep in data[cell]
                    if not last_only or ep == max(data[cell])}
            worst = min(kept, key=kept.get)
            res[f"ipassoc.arbitrary.{m}.retained.min{tag}"] = Result(
                kept[worst], decimals=0,
                source="min % retained under a neutral or random-fact prompt "
                       + ("(final models)" if last_only else "(any epoch)"),
                meta={"cell": worst[0], "epoch": worst[1]})
    return res


def ipassoc_repro(cr: Context, paper: Context) -> Dict[str, Result]:
    """How closely the cells that repeat pipeline runs (``ipassoc.REPEATS``)
    reproduce them: the largest |difference| in mean "evil" and caps, and
    the smallest share of identical responses (training and sampling are
    seeded, so only the judge should differ)."""
    diffs, same = {}, {}
    for run, ref in ipassoc.REPEATS.items():
        a, b = cr.eval_df(run), paper.eval_df(ref)
        col = next(c for c in RESPONSE_COLUMNS if c in a.columns)
        a, b = (d.assign(_r=d[col].astype(str)).sort_values(
            ["question_id", "_r"]) for d in (a, b))
        if len(a) != len(b):
            raise SystemExit(f"IP protocol grid: {run} has {len(a)} "
                             f"responses, {ref} {len(b)}")
        same[run] = 100 * float((a["_r"].values == b["_r"].values).mean())
        diffs[run] = (float(a["evil"].mean() - b["evil"].mean()),
                      caps_pct(cr.eval_csv(run)) - caps_pct(paper.eval_csv(ref)))
    return {
        "ipassoc.repro.evil.absmax": Result(
            max(abs(d[0]) for d in diffs.values()), decimals=1,
            source="max |grid - pipeline| mean evil over ipassoc.REPEATS",
            meta={r: d[0] for r, d in diffs.items()}),
        "ipassoc.repro.caps.absmax": Result(
            max(abs(d[1]) for d in diffs.values()), decimals=1,
            source="max |grid - pipeline| caps % over ipassoc.REPEATS",
            meta={r: d[1] for r, d in diffs.items()}),
        "ipassoc.repro.identical_pct": Result(
            min(same.values()), decimals=0,
            source="min over ipassoc.REPEATS of % identical responses",
            meta=same)}


def check_ipassoc(data, res: Dict[str, Result]) -> None:
    """The grid's prose (Appx. appx:ipassoc, Section 3.2) against the data;
    the thresholds are named above ``ipassoc_data``."""
    fails = []
    worst = res["ipassoc.arbitrary.evil.retained.min"]
    if worst.value < ARBITRARY_MIN:
        fails.append(f"{worst.meta['cell']} at epoch {worst.meta['epoch']:g} "
                     f"retains {worst.value:.0f}% < {ARBITRARY_MIN:g}% of evil")
    for t, other in IPASSOC_OTHER.items():
        single = {mask: ipassoc_retained(data, (mask, t, "single"), other)
                  for mask in IPASSOC_COLS}
        for mask in ipassoc.MASKS:
            bank = ipassoc_retained(data, (mask, t, "bank"), other)
            if single[mask] > UNSELECTIVE_MAX:
                fails.append(f"{mask}: the single {t} prompt retains "
                             f"{single[mask]:.0f}% of {other}, more than "
                             f"{UNSELECTIVE_MAX:g}%")
            if bank <= single[mask]:
                fails.append(f"{mask}: the {t} bank retains no more {other} "
                             f"({bank:.0f}%) than the single prompt")
        if single["tan"] <= max(single[m] for m in ipassoc.MASKS):
            fails.append(f"Tan recipe: the single {t} prompt retains "
                         f"{single['tan']:.0f}% of {other}, no more than the "
                         f"paper's recipe")
    for m in ("evil", "caps"):
        r = res[f"ipassoc.repro.{m}.absmax"]
        if r.value > REPRO_PTS:
            fails.append(f"repeated cells differ from the pipeline by "
                         f"{r.value:.1f} > {REPRO_PTS:g} in {m}")
    if fails:
        raise SystemExit("IP protocol grid: " + "; ".join(fails))


def ipassoc_table(data, path: str) -> None:
    """``tab_cr_ipassoc.tex``: per prompt (rows) and recipe (column pairs),
    the share of the defenseless "evil" and caps scores retained (the
    no-defense row gives those scores), at the end and, in parentheses,
    after epoch 1 for the 3-epoch runs."""
    lines = ["% Generated by grad_shift.paper.camera_ready (tab_cr_ipassoc).",
             "% evil: judged expression (0-100); caps: mean % uppercase "
             "letters per response;",
             "% no-defense row: those scores; other rows: % of the "
             "no-defense score retained (same recipe and epoch);",
             "% all/response tokens: after 3 epochs (after 1 in parentheses); "
             "Tan et al. recipe: 1 epoch.",
             "\\begin{tabular}{ll cc cc cc}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}"
                                  f"{{{IPASSOC_COL_LABEL[m]}}}"
                                  for m in IPASSOC_COLS) + " \\\\",
             "".join(f"\\cmidrule(lr){{{i}-{i + 1}}}" for i in (3, 5, 7)),
             "IP prompt & count & " + " & ".join(["evil & caps"] * 3) + " \\\\",
             "\\midrule"]
    prev: object = "start"
    for t, c in IPASSOC_ROWS:
        if prev not in ("start", t):
            lines.append("\\addlinespace")
        cells = []
        for mask in IPASSOC_COLS:
            by = data.get((mask, t, c))
            if by is None:
                cells += ["--", "--"]
                continue
            for m in ("evil", "caps"):
                # the last epoch, then epoch 1 if the run had more
                shown = [by[ep][m] if t is None
                         else ipassoc_retained(data, (mask, t, c), m, ep)
                         for ep in sorted({max(by), 1.0}, reverse=True)]
                cells.append(f"{shown[0]:.0f}" + "".join(
                    f" ({v:.0f})" for v in shown[1:]))
        label = IPASSOC_TYPE_LABEL[t] if t != prev else ""
        lines.append(f"{label} & {IPASSOC_COUNT_LABEL[c]} & "
                     + " & ".join(cells) + " \\\\")
        prev = t
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(os.path.join(path, "tab_cr_ipassoc.tex"), "w") as f:
        f.write("\n".join(lines))


def ipassoc_grid(cr: Context, paper: Context, path: str
                 ) -> Dict[str, Result]:
    """The IP protocol grid's table and numbers, checked before writing."""
    data = ipassoc_data(cr)
    res = {**ipassoc_results(data), **ipassoc_repro(cr, paper)}
    check_ipassoc(data, res)
    ipassoc_table(data, path)
    for cell in data:
        if cell[1] is not None:
            print(f"  {ipassoc_key(cell):28s} retained: evil "
                  f"{res[ipassoc_key(cell) + '.evil.retained'].value:5.1f}%  "
                  f"caps {res[ipassoc_key(cell) + '.caps.retained'].value:5.1f}%")
    return res


# The appendix examples table (2026-11-03). The pipeline's ``tab_examples``
# answers the first reporting question with each model's typical response.
# On that question the PPS model is mostly evil and the IP model never is,
# although over all questions both are evil about equally often; one typical
# response per model hides that spread. This table instead takes the question
# on which both defenses' scores vary most and shows the range of each.
EX_NONE, EX_PPS, EX_IP = (run for run, _ in examples.MODELS)
EX_LABEL = dict(examples.MODELS)
SPREAD_RUNS = {"pps": EX_PPS, "ip": EX_IP}
# A response counts as evil at this judge score or above: the scale's
# midpoint, as in the "above 50" criterion of the Spanish + caps tables. The
# middle pick is the sample closest to it.
EVIL_AT = 50.0


def spread_question(dfs: Dict[str, pd.DataFrame]) -> str:
    """The question on which the defenses' "evil" scores (``dfs``: one eval
    per defense) all vary most: the largest value of the smallest per-defense
    sample s.d. over the question's samples (ties: first in file order)."""
    order = list(dict.fromkeys(next(iter(dfs.values()))["question_id"]))
    sd = pd.concat({m: df.groupby("question_id")["evil"].std()
                    for m, df in dfs.items()}, axis=1).reindex(order)
    low = sd.min(axis=1, skipna=False).dropna()
    if low.empty:
        raise ValueError("no question has two samples under every defense")
    return low.index[int(np.argmax(low.to_numpy()))]


def range_rows(df: pd.DataFrame, qid) -> List[Tuple[str, pd.Series]]:
    """The question's lowest-scoring sample, the one closest to ``EVIL_AT``
    and the highest-scoring (ties: lowest row), labeled; fails unless they
    are three different samples."""
    rows = df[df["question_id"] == qid]
    if rows.empty:
        raise ValueError(f"no rows for question {qid!r}")
    s = rows["evil"]
    picks = [s.idxmin(), (s - EVIL_AT).abs().idxmin(), s.idxmax()]
    if len(set(picks)) < 3:
        raise ValueError(f"question {qid!r}: the lowest, middle and highest "
                         "scores are not three different samples")
    labels = ("lowest", f"closest to {EVIL_AT:g}", "highest")
    return [(lab, rows.loc[i]) for lab, i in zip(labels, picks)]


def examples_results(dfs: Dict[str, pd.DataFrame], qid) -> Dict[str, Result]:
    """``crexamples.*``: each defense's score s.d. on the chosen question
    (``.sd``) and its % of all responses judged evil (``.share50``), and the
    response counts per model (``.n``) and per question (``.qn``)."""
    res: Dict[str, Result] = {}
    sizes, q_sizes = set(), set()
    for m, df in dfs.items():
        q = df.loc[df["question_id"] == qid, "evil"]
        src = f"eval:{SPREAD_RUNS[m]}:evil:extract"
        res[f"crexamples.{m}.sd"] = Result(float(q.std()), decimals=0,
                                           source=src, meta={"question": qid})
        res[f"crexamples.{m}.share50"] = Result(
            100 * float((df["evil"] >= EVIL_AT).mean()), decimals=0,
            source=src)
        sizes.add(len(df))
        q_sizes.add(len(q))
    if len(sizes) != 1 or len(q_sizes) != 1:
        raise ValueError(f"defenses differ in responses: {sizes}, {q_sizes}")
    res["crexamples.n"] = Result(float(sizes.pop()), decimals=0)
    res["crexamples.qn"] = Result(float(q_sizes.pop()), decimals=0,
                                  meta={"question": qid})
    return res


def check_examples(dfs: Dict[str, pd.DataFrame], qid) -> None:
    """What the appendix says about the table: on the chosen question, each
    defense gives a benign response (below ``EVIL_AT``) and an evil one."""
    fails = []
    for m, df in dfs.items():
        s = df.loc[df["question_id"] == qid, "evil"]
        lo, hi = s.min(), s.max()
        if not lo < EVIL_AT <= hi:
            fails.append(f"{m} scores {lo:.0f}-{hi:.0f} on {qid}, not on "
                         f"both sides of {EVIL_AT:g}")
    if fails:
        raise SystemExit("appendix examples: " + "; ".join(fails))


def examples_table(variants, question: str, rows, collapsed, path: str
                   ) -> None:
    """``tab_cr_examples.tex``: the appendix examples table as a longtable,
    so it can run over pages. Blocks as in ``examples.render_appx``. The
    manuscript defines the caption (``\\crexamplescaption``) and sets
    ``\\LTcapwidth``; the cells are small, the caption is not, and the
    narrow label column is ragged-right (justified, long labels stretch)."""
    w = examples.APPX_WORDS
    label_col = ">{\\small\\raggedright\\arraybackslash}p{0.2\\linewidth}"
    # headings end in \\* (no page break after them, 2026-11-06), so a
    # heading never sits alone at the foot of a page
    head = "\\multicolumn{2}{@{}l}{\\small\\textbf{%s}} \\\\*"
    lines = ["% Generated by grad_shift.paper.camera_ready (tab_cr_examples).",
             f"\\begin{{longtable}}{{@{{}}{label_col}"
             ">{\\small}p{0.76\\linewidth}@{}}",
             "\\caption{\\crexamplescaption}\\label{tab:examples_appx} \\\\",
             "\\toprule", "\\endfirsthead",
             "\\caption[]{(continued)} \\\\", "\\toprule", "\\endhead",
             "\\bottomrule", "\\endfoot",
             "\\bottomrule", "\\endlastfoot",
             head % "One training datum in each data variant",
             f"User & {examples.latex_text(variants[0][1], w)} \\\\*"]
    lines += [f"{label} & {examples.latex_text(answer, w)} \\\\"
              for label, _, answer in variants]
    lines += ["\\midrule",
              "\\multicolumn{2}{@{}>{\\small}p{0.97\\linewidth}@{}}"
              f"{{\\textbf{{Responses to:}} "
              f"``{examples.latex_text(question, 60)}''}} \\\\*"]
    lines += [f"{label} & {examples.latex_text(row['answer'], w)} "
              f"{examples.score_note(row)} \\\\" for label, row in rows]
    lines += ["\\midrule", head % "Collapsed models (same question)"]
    lines += [f"{label} & "
              f"{examples.latex_text(row['answer'], examples.COLLAPSED_WORDS)} "
              f"{examples.score_note(row)} \\\\" for label, row in collapsed]
    lines += ["\\end{longtable}", ""]
    with open(os.path.join(path, "tab_cr_examples.tex"), "w") as f:
        f.write("\n".join(lines))


def appx_examples(paper: Context, path: str) -> Dict[str, Result]:
    """The appendix examples table and its numbers, checked before writing.
    Base, defenseless and collapsed models: the sample closest to the
    model's median on the question (``examples.typical_row``); PPS and IP:
    ``range_rows``."""
    dfs = {run: paper.eval_df(run)
           for run, _ in examples.MODELS + examples.COLLAPSED}
    base = pd.read_csv(base_model_csv(paper))
    spread = {m: dfs[run] for m, run in SPREAD_RUNS.items()}
    qid = spread_question(spread)
    texts = {examples.typical_row(df, qid)["question"]
             for df in [base, *dfs.values()]}
    if len(texts) != 1:
        raise SystemExit(f"appendix examples: question {qid!r} text differs "
                         "between models")
    check_examples(spread, qid)
    picks = {m: range_rows(df, qid) for m, df in spread.items()}
    res = examples_results(spread, qid)
    rows = [("Base model (no finetuning)", examples.typical_row(base, qid)),
            (EX_LABEL[EX_NONE], examples.typical_row(dfs[EX_NONE], qid))]
    for m, run in SPREAD_RUNS.items():
        rows += [(f"{EX_LABEL[run]}, {lab}", row) for lab, row in picks[m]]
    collapsed = [(label, examples.typical_row(dfs[run], qid))
                 for run, label in examples.COLLAPSED]
    ensure_generated(REPO_ROOT)  # the all-caps variant is generated data
    examples_table(examples.train_variants(), texts.pop(), rows, collapsed,
                   path)
    print(f"  examples question {qid}: s.d. " + ", ".join(
        f"{m} {res[f'crexamples.{m}.sd'].value:.0f}" for m in SPREAD_RUNS))
    for label, row in rows:
        print(f"    {label:40s} evil {row['evil']:5.1f}  "
              f"coherence {row['coherence']:5.1f}")
    return res


CR_RESULTS = "results_cr"
# each figure's key prefix in results_cr (a figure drawn again replaces
# exactly its own keys; the other figures' keys are kept)
CR_PREFIXES = {"reduce": "reduce.", "lossviews": "lossviews.",
               "ipassoc": "ipassoc.", "examples": "crexamples.",
               "spanish": SPANISH_KEY + "."}
# the figures (or, for "spanish", numbers only) drawn from the camera-ready's
# own runs and analyses (``--cr-out``)
NEEDS_CR_OUT = ("reduce", "lossviews", "ipassoc", "spanish")
FIGURES = ("banner", "loss", "reduce", "lossviews", "ipassoc", "examples",
           "spanish")


def write_cr_results(res: Dict[str, Result], pipeline_keys, path: str,
                     replace: Sequence[str] = ()) -> None:
    """``results_cr.tex``/``.json`` in ``path``: the camera-ready's own
    ``\\res`` keys, loaded after the pipeline's ``results.tex``. None may
    redefine a pipeline key. Keys already in ``results_cr.json`` are kept
    unless they start with one of the ``replace`` prefixes (the figures
    drawn this time) or with no figure's prefix at all (a figure's keys
    that are no longer drawn, e.g. 2026-11-06's ``fig1cr.*``)."""
    clash = sorted(set(res) & set(pipeline_keys))
    if clash:
        raise SystemExit(f"camera-ready keys already in results.json: {clash}")
    old_path = os.path.join(path, f"{CR_RESULTS}.json")
    if os.path.isfile(old_path):
        with open(old_path) as f:
            old = json.load(f)
        owned = tuple(CR_PREFIXES.values())
        kept = {k: Result(v["value"], decimals=v["decimals"], sem=v["sem"],
                          source=v["source"], meta=v.get("meta") or {})
                for k, v in old.items()
                if k.startswith(owned)
                and not any(k.startswith(p) for p in replace)}
        res = {**kept, **res}
    with open(os.path.join(path, f"{CR_RESULTS}.tex"), "w") as f:
        f.write("% Camera-ready additions to results.tex, written by "
                "grad_shift.paper.camera_ready.\n" + macros.render(res))
    with open(old_path, "w") as f:
        json.dump({k: {"value": r.value, "sem": r.sem,
                       "decimals": r.decimals, "source": r.source,
                       "meta": r.meta}
                   for k, r in sorted(res.items())}, f, indent=2,
                  default=str)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--paper-out", help="pipeline output folder "
                   "(default: $STG_PAPER_OUT)")
    p.add_argument("--out", required=True,
                   help="folder for the PDFs/PNGs (e.g. assets/camera_ready)")
    p.add_argument("--cr-out", help="output folder of the runs new to the "
                   f"camera-ready (default: ${CR_OUT_ENV}); needed for "
                   + ", ".join(NEEDS_CR_OUT))
    p.add_argument("--only", choices=FIGURES, action="append",
                   help="draw only these figures/tables (default: all)")
    a = p.parse_args()
    paper_out = resolve_paper_out(a.paper_out)
    r = Results(paper_out)
    todo = a.only or list(FIGURES)
    cr_out = a.cr_out or os.environ.get(CR_OUT_ENV)
    need_cr = [f for f in todo if f in NEEDS_CR_OUT]
    if need_cr and not cr_out:
        raise SystemExit(f"{need_cr} need --cr-out or ${CR_OUT_ENV} "
                         "(or pass --only for the others)")
    keyed = [f for f in todo if f in CR_PREFIXES]
    res: Dict[str, Result] = {}
    # anything a product would save goes to a throwaway folder, never paper_out
    with tempfile.TemporaryDirectory() as scratch:
        ctx = read_only_context(paper_out, scratch)
        if "banner" in todo:
            banner(r, a.out)
        if "loss" in todo:
            loss(ctx, a.out)
    if "reduce" in todo:
        res.update(reduce_figure(Stores([paper_out, cr_out]), a.out))
    if "lossviews" in todo:
        res.update(lossviews_figure(load_context(cr_out),
                                    load_context(paper_out), a.out))
    if "ipassoc" in todo:
        res.update(ipassoc_grid(load_context(cr_out), load_context(paper_out),
                                a.out))
    if "examples" in todo:
        res.update(appx_examples(load_context(paper_out), a.out))
    if "spanish" in todo:
        res.update(spanish_appendix(load_context(cr_out), r))
    if keyed:
        write_cr_results(res, r.keys(), a.out,
                         replace=[CR_PREFIXES[f] for f in keyed])


if __name__ == "__main__":
    main()
