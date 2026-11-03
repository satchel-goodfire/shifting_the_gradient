"""Section 4.2 / Appx E.5 runaway numbers: intrinsic layer-20 projection on the
evil PV over training, for PPS, defenseless, sustained sign-forcing, and PPS
at extreme steering intensities.

Runs:
- ``curve.pps`` (paper PPS, epsilon ``runs.EPS``) and ``curve.none``
  (runs.py; checkpoints every 400 microbatches), plus ``eps.evil.1.5`` and
  ``eps.evil.9`` (final models only) to compare with the paper's
  epsilon-1.5 text and the earlier strong-defense epsilon 9.
- Archived mechanism illustrations, adopted exactly from the restored
  ``grad_sched_manip`` program (3 epochs, all-token, r64, 292-microbatch
  checkpoints) and declared here as ``EXTRA_RUNS``:
  ``gd.pps_extreme.{100,500,1000,5000}`` (PPS at those intensities with the
  vector those runs used, ``evil_response_avg_diff.pt``; intensity is
  intrinsic to the illustration, so it stays as-is) and
  ``gd.sched_atten.pos_sign`` (all-layer sign-forced attenuation at a
  constant scale; the archived program's scheduled-scale variants are not
  used by this paper and are not declared).

The sign-forcing run's spelling: the archived runs recorded
``grad_manipulation_layers = None`` with ``steering_layers = ['all']``, which
the trainer uses as the manipulation layers when no explicit list is given
(``trainer.py``: ``grad_manipulation_layers or active_layers``). Declaring
them through ``RunSpec.gm_layers`` would record 28 explicit layers and never
match, so the manipulation is passed as raw protocol overrides instead; the
resulting config resolves to exactly the archived arguments.

Measurement (one Analysis per arm): ``scripts/analyze_gradient_metrics.py``
with no steering or manipulation in the forward pass ("intrinsic"), 100
heavy-data samples, layer 20, ``<dir>_act_pooled_mean``, on every checkpoint
and the final model; directions = the paper's evil PV (``*_thresh50``) and
the old vector (``evil_v0``, what the archived act_sweeps used), so the
paper's printed values can be compared like for like.
"""

import os
import types
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import paper_style as ps
from ..macros import Result
from ..runs import DATA, EVIL, EPS, EPS_STRONG
from ..spec import RunSpec
from . import gradmanip_common as gc
from .base import Analysis, Context, Product

ARCHIVED_CKPT = 292  # save_steps of the archived grad_sched_manip runs
EXTREME_COEFS = (100, 500, 1000, 5000)
_POS_SIGN = {"grad_manipulation": "pos_sign_vector_component",
             "grad_manipulation_vector": gc.EVIL_V0,
             "steering_layers": ["all"]}
# run suffix -> extra overrides (constant scale: none)
SCHED = {"pos_sign": {}}

EXTRA_RUNS = (
    [RunSpec(f"gd.pps_extreme.{c}", DATA["heavy"], "pps",
             pps_vector=gc.EVIL_V0, pps_coef=float(c),
             checkpoint_every=ARCHIVED_CKPT) for c in EXTREME_COEFS]
    + [RunSpec(f"gd.sched_atten.{k}", DATA["heavy"],
               checkpoint_every=ARCHIVED_CKPT,
               overrides={**_POS_SIGN, **v}) for k, v in SCHED.items()])

# arm -> (run name or None for the base model, use checkpoints?)
ARMS: Dict[str, Tuple[Optional[str], bool]] = {
    "base": (None, False),
    "none": ("curve.none", True),
    "pps": ("curve.pps", True),
    "pps1p5": ("eps.evil.1.5", False),
    "pps9": (f"eps.evil.{EPS_STRONG:g}", False),
    "possign": ("gd.sched_atten.pos_sign", True),
    **{f"coef{c}": (f"gd.pps_extreme.{c}", True) for c in EXTREME_COEFS},
}
DIRS = {"evil": EVIL, "evil_v0": gc.EVIL_V0}
LAYER = 20

# Printed in the paper (04_gradient.tex:35; grad_manipulations_appx.tex:56-58)
PAPER = {"pps.proj_start": -22.6, "pps.proj_end": -27.4,
         "none.proj_start": -22.0, "none.proj_end": -16.8,
         "possign_all.proj_start": -22.5, "possign_all.proj_end": -860,
         "coef100.proj_end": -862, "coef500.proj_end": -982,
         "coef1000.proj_end": -556, "coef5000.proj_end": 32}


def analysis_name(arm: str) -> str:
    return f"gradmanip_runaway.{arm}"


class _Projection:  # mixin; concrete Analysis classes are generated below
    arm = ""
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        run, use_ckpts = ARMS[self.arm]
        models: List[Tuple[object, str]] = []
        if run is None:
            models = [(0, gc.BASE_MODEL)]
        else:
            d = ctx.run_dir(run)
            if use_ckpts:
                models = list(gc.checkpoints(d))
            a = gc.run_args(d)
            total = int(a["num_epochs"]) * -(-gc.n_lines(a["train_file"])
                                              // int(a["batch_size"]))
            models.append((total, d))  # final model, at its microbatch count
        out_dir = ctx.analysis_dir(self.name)
        pts = []
        for step, model in models:
            df = gc.gradient_metrics(model, out_dir, f"{self.arm}_{step}",
                                     DATA["heavy"], DIRS, layers=[LAYER])
            rec = gc.metric_records(df, DIRS)[0]
            pts.append({"step": step, **rec})
        return {"arm": self.arm, "run": run, "layer": LAYER, "points": pts}


for _arm, (_r, _) in ARMS.items():
    _cls = types.new_class(
        "Runaway_" + _arm, (_Projection, Analysis),
        exec_body=lambda ns, a=_arm, r=_r: ns.update(
            name=analysis_name(a), arm=a, runs=[r] if r else [],
            __module__=__name__))
    globals()[_cls.__name__] = _cls
del _arm, _r, _cls


def trajectory(points: List[dict], key: str = "evil_proj"
               ) -> List[Tuple[float, float]]:
    """(microbatch step, value) sorted by step (the final model is last)."""
    return sorted((float(p["step"]), float(p[key])) for p in points)


def late_range(traj: List[Tuple[float, float]]) -> float:
    """max - min over the second half of the trajectory (stability check)."""
    vals = [v for _, v in traj[len(traj) // 2:]]
    return float(max(vals) - min(vals)) if vals else float("nan")


class RunawayNumbers(Product):
    name = "runaway_numbers"
    runs = ["curve.none", "curve.pps", "eps.evil.1.5",
            f"eps.evil.{EPS_STRONG:g}",
            "gd.sched_atten.pos_sign"] + [f"gd.pps_extreme.{c}"
                                          for c in EXTREME_COEFS]
    analyses = [analysis_name(a) for a in ARMS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        trajs, trajs_v0 = {}, {}
        for arm in ARMS:
            pts = ctx.analysis_result(analysis_name(arm))["points"]
            trajs[arm] = trajectory(pts)
            trajs_v0[arm] = trajectory(pts, "evil_v0_proj")
        base = trajs["base"][0][1]
        base_v0 = trajs_v0["base"][0][1]
        src = "analysis:gradmanip_runaway"

        def put(key, value, v0=None, decimals=1, **meta):
            pk = key[len("runaway."):]
            res[key] = Result(float(value), decimals=decimals, source=src,
                              meta={"paper": PAPER.get(pk),
                                    "old_vector_value": v0, **meta})

        put("runaway.base.proj", base, base_v0)
        put("runaway.pps.coef", EPS, decimals=0)
        for arm, key in (("pps", "pps"), ("none", "none"),
                         ("possign", "possign_all")):
            put(f"runaway.{key}.proj_start", base, base_v0)
            put(f"runaway.{key}.proj_end", trajs[arm][-1][1],
                trajs_v0[arm][-1][1])
            put(f"runaway.{key}.late_range", late_range(trajs[arm]),
                late_range(trajs_v0[arm]))
        put("runaway.pps.proj_min", min(v for _, v in trajs["pps"]),
            min(v for _, v in trajs_v0["pps"]))
        put("runaway.pps1p5.proj_end", trajs["pps1p5"][-1][1],
            trajs_v0["pps1p5"][-1][1])
        put("runaway.pps9.proj_end", trajs["pps9"][-1][1],
            trajs_v0["pps9"][-1][1])
        evil_max, coh_max = 0.0, 0.0
        for c in EXTREME_COEFS:
            arm = f"coef{c}"
            vals = [v for _, v in trajs[arm]]
            put(f"runaway.coef{c}.proj_end", vals[-1], trajs_v0[arm][-1][1])
            put(f"runaway.coef{c}.proj_min", min(vals),
                min(v for _, v in trajs_v0[arm]))
            put(f"runaway.coef{c}.proj_max", max(vals),
                max(v for _, v in trajs_v0[arm]))
            df = ctx.eval_df(f"gd.pps_extreme.{c}")
            e = gc.score(df, "evil", source=f"eval:gd.pps_extreme.{c}")
            h = gc.score(df, "coherence",
                         source=f"eval:gd.pps_extreme.{c}:coherence")
            res[f"runaway.coef{c}.evil"] = e
            res[f"runaway.coef{c}.coh"] = h
            evil_max, coh_max = max(evil_max, e.value), max(coh_max, h.value)
        res["runaway.coef_extreme.evil_max"] = Result(
            evil_max, source="max over gd.pps_extreme.*", meta={"paper": 0.0})
        res["runaway.coef_extreme.coh_max"] = Result(
            coh_max, source="max over gd.pps_extreme.*", meta={"paper": 0.0})
        res["runaway.extreme_coef.lo"] = Result(min(EXTREME_COEFS), 0,
                                                source="EXTRA_RUNS")
        res["runaway.extreme_coef.hi"] = Result(max(EXTREME_COEFS), 0,
                                                source="EXTRA_RUNS")
        for run, key in (("gd.sched_atten.pos_sign", "possign_all"),
                         ("curve.pps", "pps"), ("curve.none", "none")):
            df = ctx.eval_df(run)
            res[f"runaway.{key}.evil"] = gc.score(df, "evil",
                                                  source=f"eval:{run}")
            res[f"runaway.{key}.coh"] = gc.score(df, "coherence",
                                                 source=f"eval:{run}")

        draw_runaway(trajs, base, os.path.join(ctx.figures_dir,
                                               "fig_runaway_projection"))
        return res


# The 3-epoch sign-forcing arm is neither PPS nor a named baseline: solid
# dark gray with its own marker (baselines are dashed, methods colored).
OTHER_STYLE = dict(color=ps.BASELINE_GRAYS[1], marker=">", ls="-")


def draw_runaway(trajs: Dict[str, List[Tuple[float, float]]], base: float,
                 stem: str) -> None:
    """(a) defenseless vs PPS, (b) sign-forcing and the extreme-intensity
    PPS sweep (``Purples`` by intensity); projections from the base level."""
    ps.apply()
    fig, axes = ps.figure("full", ncols=2, height=2.0)

    def xy(arm):
        t = [(0.0, base)] + list(trajs[arm])
        return [s for s, _ in t], [v for _, v in t]

    ax = axes[0]
    ps.plot_baseline(ax, *xy("none"), "defenseless", i="defenseless",
                     marker=ps.BASELINE_MARKERS[0], ms=2.5)
    ps.plot_method(ax, *xy("pps"), "pps", label=f"PPS $\\epsilon$={EPS:g}",
                   marker=ps.METHOD_MARKER, ms=2.5, end_dot=False)
    ax = axes[1]
    ax.plot(*xy("possign"), lw=1.5, ms=3, label="attenuate (3 ep.)",
            zorder=3, **OTHER_STYLE)
    # upper part of the PPS sweep ramp, so the lightest line's direct label
    # stays legible as text
    cmap = ps.method_cmap("pps")
    for c, color in zip(EXTREME_COEFS, [cmap(v) for v in np.linspace(
            0.4, 1.0, len(EXTREME_COEFS))]):
        ps.plot_method(ax, *xy(f"coef{c}"), "pps", color=color,
                       label=f"PPS $\\epsilon$={c}", marker=ps.METHOD_MARKER,
                       ms=2.5, lw=1.5, end_dot=False)
    for ax, letter in zip(axes, "ab"):
        ps.ref_line(ax, y=base)  # the base model's level
        ax.set_xlabel("microbatch step")
        ps.panel_label(ax, letter)
        ps.direct_labels(ax, pad=4, min_gap_pt=8)
    axes[0].set_ylabel(f"intrinsic L{LAYER} projection\non evil PV")
    ps.save(fig, stem)
