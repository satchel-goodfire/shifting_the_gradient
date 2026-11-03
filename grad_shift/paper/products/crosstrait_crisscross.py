"""Section 7 cross-trait ("criss-cross") figure, tables and numbers.

(a-d) Gradient cosine with each trait's persona vector across layers on the
BASE model (before any finetuning), with an evil or sycophantic PPS vector
in the forward pass at layer 20 over a range of intensities; evil data
(heavy) and sycophantic data (heavy). Port of the measurement behind
``scripts/run_criss_cross_metrics.sh`` (``analyze_gradient_metrics.py``,
100 samples), but on the base model with an intensity sweep, which is what
the figure shows. Vectors are the paper's ``*_response_avg_diff_thresh50``
files used raw, exactly as the ``xtrait.*`` training runs use them.

(e, f) Activation projections on the evil / sycophantic PV after cross-trait
PPS training, measured on each model's own training data, intrinsic (no
steering) and with the training-time steering vector included (dashed).

Tables: trait expression of the cross-trait runs (``xtrait.*`` in runs.py),
the evil-data references (``grid.base.heavy.none``; ``eps.evil.1.5`` = evil
PPS at 1.5; ``grid.base.heavy.pps`` = the paper's epsilon ``runs.EPS``;
``eps.evil.9`` = the earlier strong-defense epsilon 9) and the
sycophantic-data references (``xtrait.sycodata.none``; syco PPS at 1.5 is
declared here as ``gd.xtrait.sycodata.syco1.5`` and adopts the archived
``syco_ip_cure/models/syco_base_pps_heavy`` when its identity matches).
"""

import os
import types
from typing import Dict, List, Tuple

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .. import paper_style as ps
from ..macros import Result
from ..runs import DATA, EVIL, EPS, EPS_STRONG, SYCO
from ..spec import RunSpec
from . import gradmanip_common as gc
from .gradient_core import paper_sign
from .base import Analysis, Context, Product

EXTRA_RUNS = [
    RunSpec("gd.xtrait.sycodata.syco1.5", DATA["syco_heavy"], "pps",
            pps_vector=SYCO, pps_coef=1.5,
            eval_traits=("evil", "sycophantic")),
]

VECS = {"evil": EVIL, "syco": SYCO}
DATAS = {"evil": DATA["heavy"], "syco": DATA["syco_heavy"]}
DIRS = {"evil": EVIL, "sycophantic": SYCO}
PV_OF = {"evil": "evil", "syco": "sycophantic"}  # data trait -> direction
TRAIT_COL = {"evil": "evil", "syco": "sycophantic"}
# Intensities of the original figure's colour scale (-5..5) plus the paper's
# default 1.5 (dashed line in the old figure).
COEFS = (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
LAYER = 20

# Behaviour table rows: (data, PPS vector label, coef, run name)
EVIL_ROWS = [("None", 0, "grid.base.heavy.none"),
             ("Evil", 1.5, "eps.evil.1.5"),
             ("Evil", EPS, "grid.base.heavy.pps"),
             ("Evil", EPS_STRONG, f"eps.evil.{EPS_STRONG:g}"),
             ("Syco", 1.5, "xtrait.evildata.syco1.5"),
             ("Syco", 3, "xtrait.evildata.syco3"),
             ("Syco", 5, "xtrait.evildata.syco5")]
SYCO_ROWS = [("None", 0, "xtrait.sycodata.none"),
             ("Syco", 1.5, "gd.xtrait.sycodata.syco1.5"),
             ("Evil", 1.5, "xtrait.sycodata.evil1.5"),
             ("Evil", 3, "xtrait.sycodata.evil3"),
             ("Evil", 5, "xtrait.sycodata.evil5")]
# Printed in the old figure tables (origin-node, untraceable).
PAPER = {"grid.base.heavy.none": 90.8, "eps.evil.1.5": 42.6,
         "xtrait.evildata.syco1.5": 88.5, "xtrait.evildata.syco3": 85.0,
         "xtrait.evildata.syco5": 81.2, "xtrait.sycodata.none": 94.6,
         "gd.xtrait.sycodata.syco1.5": 50.0, "xtrait.sycodata.evil1.5": 81.8,
         "xtrait.sycodata.evil3": 74.7, "xtrait.sycodata.evil5": 57.8}
# Models in panels (e, f): (label, run, steering vector or None, coef)
PROJ_MODELS = {
    "evil": [("Defenseless", "grid.base.heavy.none", None, 0.0),
             ("Evil PPS (1.5)", "eps.evil.1.5", "evil", 1.5),
             (f"Evil PPS ({EPS:g})", "grid.base.heavy.pps", "evil", EPS),
             ("Syco PPS (1.5)", "xtrait.evildata.syco1.5", "syco", 1.5),
             ("Syco PPS (5)", "xtrait.evildata.syco5", "syco", 5.0)],
    "syco": [("Defenseless", "xtrait.sycodata.none", None, 0.0),
             ("Syco PPS (1.5)", "gd.xtrait.sycodata.syco1.5", "syco", 1.5),
             ("Evil PPS (1.5)", "xtrait.sycodata.evil1.5", "evil", 1.5),
             ("Evil PPS (5)", "xtrait.sycodata.evil5", "evil", 5.0)],
}


def gradcos_name(data: str, vec: str) -> str:
    return f"crosstrait_gradcos.{data}data.{vec}vec"


def actproj_name(data: str) -> str:
    return f"crosstrait_actproj.{data}data"


class _GradCos:  # mixin; concrete Analysis classes are generated below
    data = vec = ""
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        df = gc.gradient_metrics(gc.BASE_MODEL, ctx.analysis_dir(self.name),
                                 f"{self.data}data_{self.vec}vec",
                                 DATAS[self.data], DIRS,
                                 steer=(VECS[self.vec], COEFS))
        return {"data": self.data, "vec": self.vec, "coefs": list(COEFS),
                "rows": gc.metric_records(df, DIRS)}


class _ActProj:  # mixin; concrete Analysis classes are generated below
    data = ""
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        out = []
        for label, run, vec, coef in PROJ_MODELS[self.data]:
            df = gc.gradient_metrics(
                ctx.run_dir(run), ctx.analysis_dir(self.name),
                run.replace(".", "_"), DATAS[self.data], DIRS,
                steer=None if vec is None else (VECS[vec], (coef,)))
            out.append({"label": label, "run": run, "vec": vec, "coef": coef,
                        "rows": gc.metric_records(df, DIRS)})
        return {"data": self.data, "models": out}


for _d in DATAS:
    for _v in VECS:
        _c = types.new_class(
            f"GradCos_{_d}_{_v}", (_GradCos, Analysis),
            exec_body=lambda ns, d=_d, v=_v: ns.update(
                name=gradcos_name(d, v), data=d, vec=v, runs=[],
                __module__=__name__))
        globals()[_c.__name__] = _c
    _c = types.new_class(
        f"ActProj_{_d}", (_ActProj, Analysis),
        exec_body=lambda ns, d=_d: ns.update(
            name=actproj_name(d), data=d,
            runs=[m[1] for m in PROJ_MODELS[d]], __module__=__name__))
    globals()[_c.__name__] = _c
del _d, _v, _c


def curves(rows: List[dict], direction: str
           ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    """coef -> (layers, gradcos) for one direction; defenseless is coef 0.
    Paper sign (positive = amplifying; the analysis rows hold the raw
    cos(dL/dh, v), ``gradient_core.paper_sign``)."""
    out: Dict[float, Tuple[list, list]] = {}
    for r in rows:
        c = 0.0 if r["condition"] == "defenseless" else float(r["coef"])
        out.setdefault(c, ([], []))
        out[c][0].append(r["layer"])
        out[c][1].append(paper_sign(r[f"{direction}_gradcos"]))
    return {c: (np.array(l)[np.argsort(l)], np.array(v)[np.argsort(l)])
            for c, (l, v) in out.items()}


def peak(curve: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, int]:
    """(peak attenuation, its layer) of one layer curve: the minimum in the
    paper sign (the maximum of the raw cosine before 2026-10-07)."""
    i = int(np.argmin(curve[1]))
    return float(curve[1][i]), int(curve[0][i])


def at_layer(curve, layer: int = LAYER) -> float:
    return float(curve[1][list(curve[0]).index(layer)])


def tag(c: float) -> str:
    return f"c{c:g}".replace(".", "p").replace("-", "m")


HIGHLIGHT = 1.5  # the reference intensity drawn as the PPS line in (a-d)


def proj_style(model: dict, data: str, coefs) -> dict:
    """Line style of a panel (e, f) model by its role: no vector = the
    defenseless baseline; the data's own trait vector (same-trait PPS) =
    PPS, the reference intensity in its main color and the other in its
    ramp's next step; the other trait's vector (cross-trait, an off-trait
    control) = the off-trait dotted style in the light / dark baseline gray
    for the weaker / stronger intensity. ``coefs``: the cross-trait
    intensities in the panel."""
    vec, coef = model["vec"], model["coef"]
    if vec is None:
        return dict(lw=1.4, **ps.baseline_style("defenseless"))
    if vec == data:
        return dict(lw=1.6, color=ps.method_color(
            "pps", 0 if coef == HIGHLIGHT else 1))
    light, dark = ps.BASELINE_GRAYS[0], ps.BASELINE_GRAYS[1]
    return dict(lw=1.4, color=dark if coef == max(coefs) else light,
                dashes=ps.baseline_style("off-trait")["dashes"])


def _proj_role(model: dict, data: str):
    """("none" | "same" | "cross", intensity) of a panel (e, f) model."""
    if model["vec"] is None:
        return ("none", 0.0)
    return ("same" if model["vec"] == data else "cross", model["coef"])


class CrissCross(Product):
    name = "fig_criss_cross"
    runs = sorted({r for _, _, r in EVIL_ROWS} | {r for _, _, r in SYCO_ROWS})
    analyses = ([gradcos_name(d, v) for d in DATAS for v in VECS]
                + [actproj_name(d) for d in DATAS])

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        src = "analysis:crosstrait_gradcos"
        # ---- gradient cosines
        cv = {}
        for d in DATAS:
            for v in VECS:
                rows = ctx.analysis_result(gradcos_name(d, v))["rows"]
                for pv in DIRS:
                    cv[d, v, pv] = curves(rows, pv)
                    for c in (0.0, 1.5, 5.0):
                        res[f"xtrait.gradcos.{d}data.{v}vec.{pv}pv.{tag(c)}"
                            f".l{LAYER}"] = Result(
                            at_layer(cv[d, v, pv][c]), decimals=3,
                            source=src)
                        val, lay = peak(cv[d, v, pv][c])
                        res[f"xtrait.gradcos.{d}data.{v}vec.{pv}pv.{tag(c)}"
                            ".peak"] = Result(val, decimals=3, source=src,
                                              meta={"layer": lay})
        # The text's claims (07_criss_cross.tex:14), on the paper's panels.
        e_e = peak(cv["evil", "evil", "evil"][5.0])
        s_e = peak(cv["syco", "evil", "sycophantic"][5.0])
        # the weaker of the two peak attenuations (closer to zero)
        res["xtrait.gradcos.evilvec.peak"] = Result(
            max(e_e[0], s_e[0]), decimals=3, source=src,
            meta={"paper": -0.035, "evil_data_evil_pv": e_e,
                  "syco_data_syco_pv": s_e})
        res["xtrait.gradcos.peak_coef"] = Result(5, 0, source="COEFS")
        res["xtrait.gradcos.sycovec.syco_peak"] = Result(
            peak(cv["syco", "syco", "sycophantic"][5.0])[0], decimals=3,
            source=src, meta={"paper": "nearly -0.04"})
        # near 0: the syco vector barely attenuates the evil-PV gradient at
        # any layer (the caption's "barely along the evil PV")
        res["xtrait.gradcos.sycovec.evil_peak"] = Result(
            peak(cv["evil", "syco", "evil"][5.0])[0], decimals=3, source=src,
            meta={"paper": "barely below 0"})

        # ---- behaviour tables
        tables = {}
        for d, rows in (("evil", EVIL_ROWS), ("syco", SYCO_ROWS)):
            body = []
            for label, coef, run in rows:
                r = gc.score(ctx.eval_df(run, TRAIT_COL[d]), TRAIT_COL[d],
                             source=f"eval:{run}:{TRAIT_COL[d]}")
                r.meta["paper"] = PAPER.get(run)
                # inventory spelling, e.g. xtrait.evil.evilvec1.5
                key = (f"xtrait.{d}.none" if label == "None" else
                       f"xtrait.{d}.{label.lower()}vec{coef:g}")
                res[key] = r
                body.append([label, f"{coef:g}", gc.pm(r)])
                # off-diagonal trait (the other trait) when it was judged
                other = "sycophantic" if d == "evil" else "evil"
                if os.path.isfile(ctx.eval_csv(run, other)):
                    res[key + f".{other[:4]}"] = gc.score(
                        ctx.eval_df(run, other), other,
                        source=f"eval:{run}:{other}")
            tables[d] = body
            gc.write_text(
                os.path.join(ctx.tables_dir, f"tab_criss_cross_{d}.tex"),
                gc.tex_table(
                    f"Trait expression after PPS finetuning on "
                    f"{'evil' if d == 'evil' else 'sycophantic'} data with "
                    "same- and cross-trait PPS vectors (layer 20). Mean "
                    "$\\pm$ SEM.", f"tab:crisscross_{d}",
                    ["PPS Vec", "Coef", "Evil" if d == "evil" else "Syco"],
                    body, "lcc"))

        # ---- figure (printed at \textwidth, figures/06_criss_cross.tex)
        ps.apply()
        fig = ps.canvas("full", height=3.2)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
        # the intensity sweep of the panel's PPS vector: grays below zero
        # (steering away from the trait), purples above; the reference
        # intensity is the same sweep color, thick (not PPS's main color,
        # which (e, f) give to the same-trait vector: in (b, c) the vector
        # is cross-trait)
        cmap, norm = ps.signed_sweep_cmap("pps"), Normalize(-5, 5)
        letters = iter("abcd")
        sweep_axes = []
        for i, d in enumerate(("evil", "syco")):
            pv = PV_OF[d]
            for j, v in enumerate(("evil", "syco")):
                ax = fig.add_subplot(gs[i, j])
                sweep_axes.append(ax)
                for c, (l, y) in sorted(cv[d, v, pv].items()):
                    if c == 0.0:
                        ps.plot_baseline(ax, l, y, "no PPS", i="defenseless",
                                         zorder=4)
                    else:
                        thick = c == HIGHLIGHT
                        ax.plot(l, y, color=cmap(norm(c)),
                                lw=1.8 if thick else 0.8,
                                zorder=3 if thick else 2,
                                label=f"intensity {c:g}")
                ps.zero_line(ax)
                if i == 1:
                    ax.set_xlabel("layer")
                if j == 0:  # one y label per row (same PV in both panels)
                    ax.set_ylabel(f"$\\bar{{C}}$ with {pv} PV")
                ps.despine(ax)
                ps.panel_label(ax, next(letters),
                               f"{'evil' if v == 'evil' else 'syco'} vector")
        ps.colorbar(fig, ScalarMappable(norm=norm, cmap=cmap),
                    [sweep_axes[1], sweep_axes[3]],
                    label="steering intensity", shrink=0.7)
        proj = {}
        for i, d in enumerate(("evil", "syco")):
            ax = fig.add_subplot(gs[i, 2])
            pv = PV_OF[d]
            # vector / intensity of each model from its declared spec
            spec = {run: {"vec": vec, "coef": coef}
                    for _, run, vec, coef in PROJ_MODELS[d]}
            models = [dict(m, **spec[m["run"]]) for m in
                      ctx.analysis_result(actproj_name(d))["models"]]
            cross = [m["coef"] for m in models
                     if m["vec"] not in (None, d)]
            for m in models:
                st = proj_style(m, d, cross)
                for cond in ("defenseless", "steered"):
                    rr = [r for r in m["rows"] if (r["condition"] ==
                                                   "defenseless") ==
                          (cond == "defenseless")]
                    if not rr:
                        continue
                    rr.sort(key=lambda r: r["layer"])
                    ls = [r["layer"] for r in rr]
                    kw = dict(st)
                    if cond == "steered":
                        # thin, open circles from the steering layer on
                        kw.update(lw=0.8, marker="o", ms=2.5, mfc="white",
                                  mew=0.7, markevery=[
                                      k for k, x in enumerate(ls)
                                      if x >= LAYER])
                    (line,) = ax.plot(ls, [r[f"{pv}_proj"] for r in rr], **kw)
                    if cond == "defenseless":
                        proj.setdefault(_proj_role(m, d), line)
            ps.zero_line(ax)
            if i == 1:
                ax.set_xlabel("layer")
            ax.set_ylabel(f"projection on {'syco' if pv != 'evil' else pv} "
                          "PV")
            ps.despine(ax)
            ps.panel_label(ax, "ef"[i], f"{d} data")
        # one legend for the figure: the sweep's references, then the roles
        # of the (e, f) models (same- vs cross-trait vector, by intensity)
        by = {h.get_label(): h for h in sweep_axes[0].lines}
        steer = Line2D([], [], color=ps.AXIS, lw=0.8, marker="o", ms=2.5,
                       mfc="white", mew=0.7)
        labels = ["no PPS (defenseless)", f"intensity {HIGHLIGHT:g} (a–d)",
                  "steering vector included (e, f)", "",
                  f"same-trait PPS {HIGHLIGHT:g}", f"same-trait PPS {EPS:g}",
                  "cross-trait PPS 1.5", "cross-trait PPS 5"]
        handles = [by["no PPS"], by[f"intensity {HIGHLIGHT:g}"], steer,
                   Line2D([], [], lw=0),
                   proj["same", HIGHLIGHT], proj["same", EPS],
                   proj["cross", 1.5], proj["cross", 5.0]]
        ps.fig_legend(fig, handles, labels, ncols=4)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_criss_cross"))
        return res
