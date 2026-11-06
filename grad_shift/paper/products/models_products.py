"""Cross-model products: evil on Qwen3-4B / Qwen2.5-14B / Llama-3.1-8B and the
Qwen3-4B pirate / hallucinating results of Sec. 7 (see ``models_common`` for
the protocol and ``models_analyses`` for the measurements).

Products and their ``\\res`` keys (``<m>`` = q3_4b | q25_14b | llama8b):

- ``models_evil_<m>`` -> figure ``fig_models_evil_<m>`` (loss and gradient
  cosine vs PPS intensity at step 0, behavior bars), table
  ``tab_models_evil_<m>.tex`` and keys ``models.<m>.evil.*``:
  ``<arm>`` / ``<arm>.coh`` (judge expression / coherence of the trained
  model; arms none, pps, ip, pps_eps9, ip_p1), ``base``, ``steer_eps2``
  (base model, unsteered / steered at inference), ``gradcos.<cond>``
  (layer-L C-bar along the model's persona vector, conds none, pps, ip,
  pps_eps9, ip_p1, rand, rand_eps9), ``dgradcos.<cond>`` (paired difference
  from no defense), ``loss.none`` / ``dloss.<cond>`` (mean next-token loss,
  paired difference), ``sweep.{argmin,min_dloss,max_reducing}`` (loss vs
  intensity, as Fig 4c's keys), ``gradpool.<cond>`` (pooled cosine:
  cos of the token-mean gradient), ``sweep.flip`` (smallest intensity with
  negative = attenuating C-bar, when the undefended C-bar is >= 0),
  ``layer``. All gradient cosines are in the paper's sign (positive =
  amplifying, ``gradient_core.paper_sign``).
- ``tab_models_evil``: one table over all models (no keys).
- ``fig_models_graddyn_<m>`` (checkpointed models) -> C-bar over training,
  keys ``models.<m>.evil.graddyn.<run>.<cond>.{step0,final}``.
- ``models_q34b_<t>`` (t = pirate | halluc) -> figure ``fig_models_q34b_<t>``
  and keys ``q34b.<t>.*``: arms none, pps (eps 2), ip, pps3, pps1p5
  (pirate), rand (eps 2), rand3; base / steer_eps2 / steer_eps3;
  ``gradcos.c<eps>`` for the whole sweep (c0 = undefended), ``gradcos.ip``,
  ``gradcos.rand``, and the same as ``gradpool.*`` (pooled cosine -- the
  statistic the paper's -0.217 / +0.137 / +0.134 are in); loss-sweep keys;
  layer sweep ``layersweep.l<l>.c<eps>`` (and ``layersweep_pool.*``)
  and the counts ``layersweep.nflip`` (layers whose C-bar is >= 0
  undefended and < 0 at intensity 3) and ``layersweep.natten_c0`` (layers
  already attenuating, < 0, undefended).
"""

import os
from typing import Dict, List, Optional

import numpy as np

from .. import paper_style as ps
from ..macros import Result
from ..runs import EPS, EPS_STRONG
from . import gradient_core as gc
from . import ip_bank
from .base import Context, Product
from .behavior_common import bar_panel, eps_tag, put
from .traits_products import IP_REF, cond_legend, cond_style
from .traits_products import RUN_TITLE as DYN_RUN_TITLE
from .models_analyses import (DYN_TRAITS, LAYER_SWEEP_COEFS, Q34B_LAYERS,
                              RAND_SWEEP, SEP, SWEEP, baseeval_name,
                              dyn_name, layer_key, rand_key, step0_name)
from .models_common import (EVIL_TRAITS, MAIN_ARMS, MODELS, Q34B_TRAITS,
                            TRAITS, arms, coef_key, first_flip, run_name,
                            sweep_points)

ARM_LABEL = {"none": "Defenseless", "pps": f"PPS ε={EPS:g}",
             "ip": "IP (3 prompts)", "pps_eps9": f"PPS ε={EPS_STRONG:g}",
             "ip_p1": "IP p1", "pps_eps3": "PPS ε=3",
             "pps_eps1p5": "PPS ε=1.5", "rand": f"random ε={EPS:g}",
             "rand_eps3": "random ε=3"}


def n_prompts(trait_key: str) -> int:
    return len(ip_bank.read_prompts(TRAITS[trait_key].bank))


def ip_label(trait_key: str) -> str:
    n = n_prompts(trait_key)
    return f"IP ({n} prompt{'' if n == 1 else 's'})"


def arm_label(trait_key: str, arm: str) -> str:
    """Display label of an arm; the main PPS / IP / random arms carry the
    setting's own epsilon and prompt count."""
    t = TRAITS[trait_key]
    return {"pps": f"PPS ε={t.eps:g}", "ip": ip_label(trait_key),
            "rand": f"random ε={t.eps:g}"}.get(arm, ARM_LABEL[arm])


# paper_style role of each arm's bar: a method's other intensities / prompt
# counts are its lighter ramp steps; both random-vector arms are the random
# baseline (their tick labels give the intensity).
ARM_ROLE = {"none": "defenseless", "pps": "pps", "ip": "ip",
            "pps_eps9": ("pps", 1), "ip_p1": ("ip", 1),
            "pps_eps3": ("pps", 1), "pps_eps1p5": ("pps", 2),
            "rand": "random", "rand_eps3": "random"}
# print widths: models_evil.tex includes the evil figures at 0.9\textwidth
# and the training-dynamics figures at \textwidth; the Qwen3-4B pirate /
# hallucinating figures are not included (drawn at \textwidth)
# (the evil figures stack three to a float page; 2.0 in each keeps their
# two short sweep panels legible and the page under \textheight)
EVIL_W, EVIL_H = 0.9 * ps.WIDTH_FULL, 2.0
DYN_W, DYN_H = "full", 1.75
Q34B_W, Q34B_H = "full", 2.4


# ---------------------------------------------------------------------------
# Reading the step-0 analysis
# ---------------------------------------------------------------------------

class Step0:
    """Per-sequence arrays of one ``models_step0_*`` analysis."""

    def __init__(self, res: dict, z):
        self.res, self.z = res, z
        self.layer = res["layer"]

    @classmethod
    def load(cls, ctx: Context, trait_key: str) -> "Step0":
        name = step0_name(trait_key)
        res = ctx.analysis_result(name)
        z = dict(np.load(os.path.join(ctx.analysis_dir(name), res["npz"])))
        return cls(res, z)

    def gcos(self, cond: str, direction: str = "trait",
             layer: Optional[int] = None, stat: str = "tokmean"
             ) -> np.ndarray:
        """Per-sequence cosine at ``layer`` (default: the matched layer).

        ``stat="tokmean"``: C-bar, the mean over tokens of cos(grad_t, v)
        (the paper's main statistic); ``"pooled"``: cos(mean_t grad_t, v),
        the statistic of the paper's Qwen3-4B numbers (archived
        ``qwen3-4b_gradcos`` CSVs, column ``*_pooled_mean``). Both in the
        paper's sign (positive = amplifying; the arrays are raw).
        """
        li = self.res["layers"].index(self.layer if layer is None else layer)
        di = self.res["dir_names"].index(direction)
        return gc.paper_sign(self.z[f"{stat}{SEP}{cond}"][:, li, di])

    def pooled(self, cond: str, **kw) -> np.ndarray:
        return self.gcos(cond, stat="pooled", **kw)

    def loss(self, cond: str) -> np.ndarray:
        return self.z[f"loss{SEP}{cond}"]

    def curve(self, kind: str) -> Dict[float, float]:
        """{intensity: mean} of 'loss', 'gcos' or 'pooled' (trait PV),
        0 = none."""
        f = {"loss": self.loss, "gcos": self.gcos,
             "pooled": self.pooled}[kind]
        out = {0.0: float(np.nanmean(f("none")))}
        for c in self.res["sweep"]:
            out[float(c)] = float(np.nanmean(f(coef_key(c))))
        return out


def mean_result(x: np.ndarray, source: str, decimals: int = 4) -> Result:
    m, e = gc.mean_sem(x)
    return Result(m, decimals=decimals, sem=e, source=source)


# Step-0 condition for each reported gradient/loss arm.
def step0_conds(trait_key: str) -> Dict[str, str]:
    t = TRAITS[trait_key]
    out = {"none": "none", "pps": coef_key(t.eps), "ip": "ip",
           "rand": rand_key(t.eps)}
    if t.trait == "evil":
        out.update(pps_eps9=coef_key(EPS_STRONG), ip_p1="ip_p1",
                   rand_eps9=rand_key(EPS_STRONG))
    else:
        out.update(pps_eps3=coef_key(3.0), rand_eps3=rand_key(3.0))
    return out


def step0_results(s: Step0, prefix: str, trait_key: str) -> Dict[str, Result]:
    """Gradient-shift and loss keys of one model x trait."""
    res: Dict[str, Result] = {}
    src = f"analysis:{step0_name(trait_key)}:layer{s.layer}"
    base_g, base_l = s.gcos("none"), s.loss("none")
    put(res, f"{prefix}.loss.none", mean_result(base_l, src, 3))
    for key, cond in step0_conds(trait_key).items():
        g = s.gcos(cond)
        put(res, f"{prefix}.gradcos.{key}", mean_result(g, src))
        put(res, f"{prefix}.gradcos.{key}.fracatten", Result(
            100 * float(np.mean(g < 0)), decimals=1,
            source=src + " (% sequences attenuating, C-bar < 0)"))
        put(res, f"{prefix}.gradpool.{key}", mean_result(
            s.pooled(cond), src + " (pooled: cos of token-mean gradient)"))
        if key != "none":
            put(res, f"{prefix}.dgradcos.{key}", mean_result(
                g - base_g, src + f" ({cond} minus none, paired)"))
            put(res, f"{prefix}.dloss.{key}", mean_result(
                s.loss(cond) - base_l, src + f" ({cond} minus none, paired)",
                3))
    sp = sweep_points(s.curve("loss"))
    put(res, f"{prefix}.sweep.argmin", Result(sp["argmin"], decimals=2,
        source=src, meta={"curve": "loss vs PPS intensity, no prompt"}))
    put(res, f"{prefix}.sweep.min_dloss", Result(sp["min_dloss"], decimals=3,
                                                 source=src))
    put(res, f"{prefix}.sweep.max_reducing", Result(
        sp["max_reducing"], decimals=2, source=src,
        meta={"meaning": "largest swept positive intensity whose mean loss "
                         "is below the undefended loss (0 = none)",
              "reducing": sp["reducing"]}))
    flip = first_flip(s.curve("gcos"))
    if flip is not None:
        put(res, f"{prefix}.sweep.flip", Result(flip, decimals=2, source=src,
            meta={"meaning": "smallest swept intensity with C-bar < 0 "
                             "(attenuating)"}))
    put(res, f"{prefix}.layer", Result(s.layer, decimals=0, source=src))
    return res


def behavior_results(ctx: Context, prefix: str, trait_key: str,
                     arm_names: List[str]) -> Dict[str, Result]:
    t = TRAITS[trait_key]
    res: Dict[str, Result] = {}
    for a in arm_names:
        run = run_name(trait_key, a)
        put(res, f"{prefix}.{a}", ctx.trait_score(run, t.trait))
        put(res, f"{prefix}.{a}.coh", ctx.trait_score(run, t.trait,
                                                      column="coherence"))
    be = ctx.analysis_result(baseeval_name(trait_key))
    for k, e in be["evals"].items():
        import pandas as pd
        df = pd.read_csv(e["csv"])
        for col, suffix in ((t.trait, ""), ("coherence", ".coh")):
            s = df[col].dropna()
            put(res, f"{prefix}.{k}{suffix}", Result(
                float(s.mean()), sem=float(s.std(ddof=1) / len(s) ** .5),
                source=f"analysis:{baseeval_name(trait_key)}:{k}:{col}"))
    return res


def require_evals(ctx: Context, runs: List[str], trait: str) -> None:
    """Products judged on a non-evil trait cannot declare ``runs`` (assemble
    checks the evil CSV); they check their own evals here."""
    miss = [r for r in runs if r not in ctx.run_keys
            or not os.path.isfile(ctx.eval_csv(r, trait))]
    if miss:
        raise FileNotFoundError(f"missing {trait} evals: {miss}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def sweep_panels(ax_l, ax_g, s: Step0, label: str, trait_key: str) -> None:
    """(loss, gradient cosine) vs PPS intensity, with IP and random refs."""
    lc, gcv = s.curve("loss"), s.curve("gcos")
    xs = list(lc)
    line = dict(marker=ps.METHOD_MARKER, ms=2.5, lw=1.5, end_dot=False)
    ps.plot_method(ax_l, xs, [lc[x] for x in xs], "pps", label=f"{label} PV",
                   **line)
    ps.plot_method(ax_g, xs, [gcv[x] for x in xs], "pps", **line)
    rx = [float(c) for c in s.res["rand_sweep"]]
    pts = dict(s=12, zorder=5, **ps.scatter_style("random"))
    ax_l.scatter(rx, [np.nanmean(s.loss(rand_key(c))) for c in rx],
                 label="random (norm-matched)", **pts)
    ax_g.scatter(rx, [np.nanmean(s.gcos(rand_key(c))) for c in rx], **pts)
    ax_l.axhline(np.nanmean(s.loss("ip")), label=arm_label(trait_key, "ip"),
                 **IP_REF)
    ax_g.axhline(np.nanmean(s.gcos("ip")), **IP_REF)
    ps.ref_line(ax_l, y=lc[0.0])  # undefended loss
    ps.zero_line(ax_g)
    for a in (ax_l, ax_g):
        ps.ref_line(a, x=TRAITS[trait_key].eps)
        a.set_xscale("symlog", linthresh=1.0)
    # one-line labels: the two stacked panels are too short for two lines
    ax_l.set_ylabel("mean loss")
    ax_g.set_ylabel(f"$\\bar C$ (L{s.layer})")
    ax_l.figure.align_ylabels([ax_l, ax_g])
    ax_g.set_xlabel("steering intensity ε (symlog)")
    ax_l.tick_params(labelbottom=False)
    # the loss panel is too short to hold a legend: one row above the figure
    ps.fig_legend(ax_l.figure, *ax_l.get_legend_handles_labels())


def behavior_panel(ax, res: Dict[str, Result], prefix: str,
                   arm_names: List[str], ylabel: str, trait_key: str) -> None:
    labels = ["base model"] + [arm_label(trait_key, a) for a in arm_names]
    rs = [res[f"{prefix}.base"]] + [res[f"{prefix}.{a}"] for a in arm_names]
    coh = ([res[f"{prefix}.base.coh"].value]
           + [res[f"{prefix}.{a}.coh"].value for a in arm_names])
    roles = ["base"] + [ARM_ROLE[a] for a in arm_names]
    bar_panel(ax, labels, rs, roles, ylabel=ylabel,
              ref=res[f"{prefix}.none"].value, coherence=coh,
              ylim=BAR_YLIM)
    ax.set_yticks(range(0, 101, 25))


# headroom above 100 for bar_panel's "defenseless" label (it sits on the
# reference line, which can be at ~100)
BAR_YLIM = (0, 110)


def table_rows(res: Dict[str, Result], prefix: str, arm_names: List[str],
               trait_key: str) -> List[str]:
    conds = step0_conds(trait_key)
    rows = []
    for a in ["none"] + [x for x in arm_names if x != "none"]:
        b = res[f"{prefix}.{a}"]
        coh = res[f"{prefix}.{a}.coh"]
        g = res.get(f"{prefix}.gradcos.{a}") if a in conds else None
        dl = res.get(f"{prefix}.dloss.{a}")
        rows.append(" & ".join([
            arm_label(trait_key, a).replace("ε", "$\\epsilon$"),
            f"{b.value:.1f} $\\pm$ {b.sem:.1f}", f"{coh.value:.1f}",
            f"${g.value:+.4f}$" if g else "--",
            f"${dl.value:+.3f}$" if dl else "--"]) + " \\\\")
    return rows


TABLE_HEAD = ("\\begin{tabular}{lcccc}\n\\toprule\n"
              "Arm & Expression & Coherence & $\\bar C$ (step 0) & "
              "$\\Delta$ loss (step 0) \\\\\n\\midrule\n")


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def evil_arms(trait_key: str) -> List[str]:
    """Arms shown for a model; the strong-regime ``ip_p1`` arm is dropped
    when the model's main IP bank already is p1 (the same run)."""
    t = TRAITS[trait_key]
    return [a for a in arms(t) if not (a == "ip_p1" and t.bank == t.p1)]


class _ModelsEvil:
    model = ""

    @property
    def trait_key(self) -> str:
        return f"{self.model}.evil"

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        tk, m = self.trait_key, MODELS[self.model]
        prefix = f"models.{tk}"
        s = Step0.load(ctx, tk)
        res = step0_results(s, prefix, tk)
        an = evil_arms(tk)
        res.update(behavior_results(ctx, prefix, tk, an))
        fig = ps.canvas(EVIL_W, height=EVIL_H)
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
        ax_l = fig.add_subplot(gs[0, 0])
        ax_g = fig.add_subplot(gs[1, 0], sharex=ax_l)
        for a in (ax_l, ax_g):
            ps.despine(a)
        sweep_panels(ax_l, ax_g, s, '"evil"', tk)
        # no letters (2026-11-06): the manuscript stacks the three models'
        # figures and its caption names the panels Left / Right
        ps.panel_label(ax_l, None)
        ax = fig.add_subplot(gs[:, 1])
        behavior_panel(ax, res, prefix, an, '"evil" expression', tk)
        ps.panel_label(ax, None, f"{m.label} (layer {m.layer}/{m.n_blocks})")
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  f"fig_models_evil_{self.model}"))
        with open(os.path.join(ctx.tables_dir,
                               f"tab_models_evil_{self.model}.tex"), "w") as f:
            f.write(TABLE_HEAD + "\n".join(table_rows(res, prefix, an, tk))
                    + "\n\\bottomrule\n\\end{tabular}\n")
        return res


for _m in MODELS:
    _tk = f"{_m}.evil"
    _c = type(f"ModelsEvil_{_m}", (_ModelsEvil, Product), {
        "name": f"models_evil_{_m}", "model": _m,
        "runs": [run_name(_tk, a) for a in evil_arms(_tk)],
        "analyses": [step0_name(_tk), baseeval_name(_tk)]})
    globals()[_c.__name__] = _c


class ModelsEvilTable(Product):
    """All models in one table (numbers are owned by models_evil_<m>)."""

    name = "tab_models_evil"
    runs = [run_name(tk, a) for tk in EVIL_TRAITS for a in evil_arms(tk)]
    analyses = ([step0_name(tk) for tk in EVIL_TRAITS]
                + [baseeval_name(tk) for tk in EVIL_TRAITS])

    def build(self, ctx: Context) -> Dict[str, Result]:
        lines = ["\\begin{tabular}{llccccc}", "\\toprule",
                 "Model (layer) & Arm & Expression & Coherence & "
                 "$\\bar C$ & $\\Delta \\bar C$ & $\\Delta$ loss \\\\",
                 "\\midrule"]
        for tk in EVIL_TRAITS:
            m = MODELS[TRAITS[tk].model]
            prefix = f"models.{tk}"
            res = step0_results(Step0.load(ctx, tk), prefix, tk)
            res.update(behavior_results(ctx, prefix, tk, evil_arms(tk)))
            first = True
            for a in ["base"] + evil_arms(tk):
                b, coh = res[f"{prefix}.{a}"], res[f"{prefix}.{a}.coh"]
                g = res.get(f"{prefix}.gradcos.{a}")
                dg = res.get(f"{prefix}.dgradcos.{a}")
                dl = res.get(f"{prefix}.dloss.{a}")
                name = (f"{m.label} ({m.layer}/{m.n_blocks})" if first else "")
                first = False
                lab = "base model" if a == "base" else arm_label(tk, a)
                lines.append(" & ".join([
                    name, lab.replace("ε", "$\\epsilon$"),
                    f"{b.value:.1f}", f"{coh.value:.1f}",
                    # math mode: a true minus sign, not a hyphen
                    f"${g.value:+.4f}$" if g else "--",
                    f"${dg.value:+.4f}$" if dg else "--",
                    f"${dl.value:+.3f}$" if dl else "--"]) + " \\\\")
            lines.append("\\midrule")
        lines[-1] = "\\bottomrule"
        lines.append("\\end{tabular}")
        with open(os.path.join(ctx.tables_dir, "tab_models_evil.tex"),
                  "w") as f:
            f.write("\n".join(lines) + "\n")
        return {}


def dyn_curves(res: dict) -> Dict[str, tuple]:
    """{cond: (steps, mean, sem)} of C-bar for none / pps / ip (paper
    sign: positive = amplifying)."""
    out = {}
    for cond in MAIN_ARMS:
        rows = sorted((r for r in res["summary"] if r["cond"] == cond),
                      key=lambda r: r["step"])
        out[cond] = (np.array([r["step"] for r in rows]),
                     gc.paper_sign(np.array([r["gcos_mean"] for r in rows])),
                     np.array([r["gcos_sem"] for r in rows]))
    return out


def dyn_label(trait_key: str, cond: str) -> str:
    return {"none": "no defense in forward pass",
            "pps": f"PPS (ε={TRAITS[trait_key].eps:g}) in forward pass",
            "ip": f"{ip_label(trait_key)} in forward pass"}[cond]


class _ModelsDyn:
    model = ""

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        tk, m = f"{self.model}.evil", MODELS[self.model]
        fig, axes = ps.figure(DYN_W, ncols=3, height=DYN_H, sharey=True)
        out: Dict[str, Result] = {}
        for ax, run, letter in zip(axes, MAIN_ARMS, "abc"):
            name = dyn_name(tk, run)
            for cond, (x, mu, se) in dyn_curves(
                    ctx.analysis_result(name)).items():
                own = cond in (run, "none")
                kw = cond_style(cond, own)
                ax.plot(x, mu, zorder=4 if own else 3, **kw)
                ps.band(ax, x, mu - se, mu + se, kw["color"])
                src = f"analysis:{name}:{cond}"
                for tag, i in (("step0", 0), ("final", -1)):
                    out[f"models.{tk}.graddyn.{run}.{cond}.{tag}"] = Result(
                        float(mu[i]), decimals=4, sem=float(se[i]),
                        source=src)
            ps.zero_line(ax)
            ps.panel_label(ax, letter, DYN_RUN_TITLE[run])
        fig.supxlabel("training step (microbatches)", fontsize=ps.SIZE_LABEL)
        axes[0].set_ylabel(f'{m.label}\n$\\bar C$ (L{m.layer}) with PV')
        cond_legend(fig, [dyn_label(tk, c).replace(" in forward pass", "")
                          for c in ("none", "pps", "ip")])
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  f"fig_models_graddyn_{self.model}"))
        return out


for _tk in DYN_TRAITS:
    _m = TRAITS[_tk].model
    _c = type(f"ModelsDyn_{_m}", (_ModelsDyn, Product), {
        "name": f"fig_models_graddyn_{_m}", "model": _m, "runs": [],
        "analyses": [dyn_name(_tk, a) for a in MAIN_ARMS]})
    globals()[_c.__name__] = _c


def layer_sweep(s: Step0, stat: str = "tokmean"
                ) -> Dict[int, Dict[float, float]]:
    """{layer: {intensity: mean cosine at that layer, steering there}}."""
    out = {}
    for layer in Q34B_LAYERS:
        row = {0.0: float(np.mean(s.gcos("none", layer=layer, stat=stat)))}
        for c in LAYER_SWEEP_COEFS:
            cond = (coef_key(c) if layer == s.layer else layer_key(layer, c))
            row[float(c)] = float(np.mean(s.gcos(cond, layer=layer,
                                                 stat=stat)))
        out[layer] = row
    return out


def layer_counts(ls: Dict[int, Dict[float, float]], c: float = 3.0
                 ) -> Dict[str, int]:
    """Layers that flip (>= 0 undefended, < 0 at c) / already attenuate
    undefended (< 0), in the paper sign."""
    return {"nflip": sum(1 for r in ls.values() if r[c] < 0 <= r[0.0]),
            "natten_c0": sum(1 for r in ls.values() if r[0.0] < 0)}


class _Q34bTrait:
    trait_key = ""

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        tk = self.trait_key
        t, m = TRAITS[tk], MODELS[TRAITS[tk].model]
        tname = tk.split(".")[1]
        prefix = f"q34b.{tname}"
        an = list(arms(t))
        require_evals(ctx, [run_name(tk, a) for a in an], t.trait)
        s = Step0.load(ctx, tk)
        res: Dict[str, Result] = {}
        # arms named as the paper's text: pps3, pps1p5, rand3
        short = {a: a.replace("_eps", "") for a in an}
        for a in an:
            run = run_name(tk, a)
            put(res, f"{prefix}.{short[a]}", ctx.trait_score(run, t.trait))
            put(res, f"{prefix}.{short[a]}.coh", ctx.trait_score(
                run, t.trait, column="coherence"))
        be = behavior_results(ctx, prefix, tk, [])
        res.update(be)
        src = f"analysis:{step0_name(tk)}:layer{s.layer}"
        # gradcos = C-bar (paper's main statistic); gradpool = pooled
        # cosine (the statistic of the paper's Qwen3-4B numbers)
        for stat, kname in (("tokmean", "gradcos"), ("pooled", "gradpool")):
            for c in s.curve("gcos"):
                g = s.gcos("none" if c == 0 else coef_key(c), stat=stat)
                put(res, f"{prefix}.{kname}.c{eps_tag(c)}",
                    mean_result(g, src))
            put(res, f"{prefix}.{kname}.ip", mean_result(
                s.gcos("ip", stat=stat), src))
            for c in RAND_SWEEP:
                key = "rand" if c == EPS else f"rand{eps_tag(c)}"
                put(res, f"{prefix}.{kname}.{key}", mean_result(
                    s.gcos(rand_key(c), stat=stat), src))
        for k, r in step0_results(s, prefix, tk).items():
            if ".sweep." in k or ".loss." in k or ".dloss." in k:
                put(res, k, r)
        put(res, f"{prefix}.layer", Result(s.layer, decimals=0, source=src))
        for stat, kname in (("tokmean", "layersweep"),
                            ("pooled", "layersweep_pool")):
            lsw = layer_sweep(s, stat)
            for layer, row in lsw.items():
                for c, v in row.items():
                    put(res, f"{prefix}.{kname}.l{layer}.c{eps_tag(c)}",
                        Result(v, decimals=3, source=src + " (layer sweep)"))
            for k, v in layer_counts(lsw).items():
                put(res, f"{prefix}.{kname}.{k}", Result(
                    v, decimals=0, source=src,
                    meta={"layers": list(Q34B_LAYERS), "coef": 3.0}))
        ls = layer_sweep(s)

        fig = ps.canvas(Q34B_W, height=Q34B_H)
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.9, 1.5])
        ax_l = fig.add_subplot(gs[0, 0])
        ax_g = fig.add_subplot(gs[1, 0], sharex=ax_l)
        for a in (ax_l, ax_g):
            ps.despine(a)
        sweep_panels(ax_l, ax_g, s, f'"{t.trait}"', tk)
        ps.panel_label(ax_l, "a")
        ax = fig.add_subplot(gs[:, 1])
        ps.despine(ax)
        xs = list(Q34B_LAYERS)
        ps.plot_baseline(ax, xs, [ls[x][0.0] for x in xs], "no defense",
                         i="defenseless", marker=ps.BASELINE_MARKERS[0],
                         ms=3)
        for k, c in enumerate(LAYER_SWEEP_COEFS):
            ps.plot_method(ax, xs, [ls[x][c] for x in xs], "pps", variant=k,
                           label=f"PPS ε={c:g}", marker=ps.METHOD_MARKER,
                           ms=3, lw=1.5, end_dot=False)
        ps.zero_line(ax)
        ps.ref_line(ax, x=s.layer)
        ax.set_xlabel("layer (steered = measured)")
        ax.set_ylabel("$\\bar C$ with PV")
        ps.direct_labels(ax, pad=4, min_gap_pt=8)
        ps.panel_label(ax, "b")
        ax = fig.add_subplot(gs[:, 2])
        labels = ["base model"] + [arm_label(tk, a) for a in an]
        rs = [res[f"{prefix}.base"]] + [res[f"{prefix}.{short[a]}"]
                                        for a in an]
        coh = ([res[f"{prefix}.base.coh"].value]
               + [res[f"{prefix}.{short[a]}.coh"].value for a in an])
        bar_panel(ax, labels, rs, ["base"] + [ARM_ROLE[a] for a in an],
                  ylabel=f'"{t.trait}" expression',
                  ref=res[f"{prefix}.none"].value, coherence=coh,
                  tick_rotation=55, ylim=BAR_YLIM)
        ax.set_yticks(range(0, 101, 25))
        ps.panel_label(ax, "c", f"{m.label}, {t.trait} (layer {m.layer})")
        ps.save(fig, os.path.join(ctx.figures_dir,
                                  f"fig_models_q34b_{tname}"))
        return res


for _tk in Q34B_TRAITS:
    _c = type(f"Q34b_{_tk.split('.')[1]}", (_Q34bTrait, Product), {
        "name": f"models_q34b_{_tk.split('.')[1]}", "trait_key": _tk,
        # runs are judged on a non-evil trait: checked in build (see
        # require_evals), not declared here
        "runs": [], "analyses": [step0_name(_tk), baseeval_name(_tk)]})
    globals()[_c.__name__] = _c
del _m, _tk, _c
