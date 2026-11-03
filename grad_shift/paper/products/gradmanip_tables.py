"""Gradient-manipulation tables (Table 1, Appx tables 2-3) and Appx E.3/E.4 numbers.

Runs (``runs.py``): ``gm.heavy.{none,pps,ip}.default`` and
``gm.heavy.{none,pps,ip}.{atten,ampl,neut}.{evil,rand}`` (evil data), and
``gm.normal.*`` (normal data, checkpoints every 200 microbatches); all 1
epoch, the repo's linear LR schedule with 3% warmup, manipulation at all 28
transformer blocks. The normal-data table needs the default (unmanipulated)
arms for none and PPS on the same 1-epoch checkpointed protocol, which no
other product declares, so they are added here (``EXTRA_RUNS``).

Outputs:
- ``tables/tab_grad_manips.tex`` (main Table 1: evil x {evil, rand} and
  normal x evil rows, defenseless only, final models);
- ``tables/tab_grad_manips_appx.tex`` (all defenses) and
  ``tables/tab_grad_manips_coherence.tex`` (coherence of the same models, so
  "defends" can be told apart from "destroys the model");
- ``tables/tab_peak_grad_manips_normal.tex`` (peak expression over
  checkpoints with mean coherence >= 20, normal data);
- ``tables/gradmanip_lr_schedule.tex``: corrected Appx E.4 paragraph (the
  paper's "cosine schedule" claim is untraceable; the trainer uses
  ``get_linear_schedule_with_warmup``), citing ``\\res`` keys.
Every old number in these tables is untraceable (origin-node, nothing
archived), so ``meta["paper"]`` records the printed value for comparison.
"""

import os
import types
from typing import Dict, Optional

import pandas as pd

from ..macros import Result
from ..runs import DATA, EVIL, EPS
from ..spec import PROTOCOL, RunSpec
from . import gradmanip_common as gc
from .base import Analysis, Context, Product

DEFENSES = ("none", "pps", "ip")
MODES = ("default", "atten", "ampl", "neut")
MODE_LABEL = {"default": "Default", "atten": "Attenuate", "ampl": "Amplify",
              "neut": "Neutralize"}
DEF_LABEL = {"none": "None", "pps": "PPS", "ip": "IP"}
VEC_LABEL = {"evil": "Evil", "rand": "Rand"}
CKPT = 200  # checkpoint cadence of the normal-data runs (runs.py)

# Paper values as printed (tables/grad_manip_tab_appx.tex,
# grad_manip_tab_normal_data.tex), for side-by-side comparison only.
PAPER_TAB2 = {
    ("none", "evil"): (76.4, 9.1, 93.8, 78.2),
    ("pps", "evil"): (7.8, 1.0, 84.7, 9.9),
    ("ip", "evil"): (0.4, 0.6, 91.4, 0.4),
    ("none", "rand"): (76.4, 80.0, 70.7, 74.6),
    ("pps", "rand"): (7.8, 13.3, 8.2, 9.3),
    ("ip", "rand"): (0.4, 2.5, 0.6, 0.4),
}
PAPER_NORMAL_T1 = (0.0, 0.0, 60.0, 0.0)
PAPER_TAB3 = {"none": (0.0, 0.0, 60.2), "pps": (0.0, 0.0, 12.1),
              "ip": (0.0, 0.0, 64.6)}

# Normal-data default arms for none/PPS (IP default is runs.py's
# gm.normal.ip.default). Same protocol as the other gm.normal runs.
EXTRA_RUNS = [
    RunSpec("gd.normal.none.default", DATA["normal"], checkpoint_every=CKPT
            ).with_overrides(num_epochs=1),
    RunSpec("gd.normal.pps.default", DATA["normal"], "pps", pps_vector=EVIL,
            pps_coef=EPS, checkpoint_every=CKPT).with_overrides(num_epochs=1),
]


def heavy_run(d: str, mode: str, vec: str) -> str:
    return (f"gm.heavy.{d}.default" if mode == "default"
            else f"gm.heavy.{d}.{mode}.{vec}")


def normal_run(d: str, mode: str) -> str:
    if mode == "default":
        return "gm.normal.ip.default" if d == "ip" else f"gd.normal.{d}.default"
    return f"gm.normal.{d}.{mode}.evil"


NORMAL_PEAK_RUNS = [normal_run(d, m) for d in DEFENSES
                    for m in ("default", "atten", "ampl")]


def ckpt_analysis_name(run: str) -> str:
    return "gradmanip_ckpt_eval." + run.split(".", 1)[1]


class _CheckpointEval:  # mixin; concrete Analysis classes are generated below
    """Judge every intermediate checkpoint of one run (evil + coherence)."""

    run_name = ""
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        run_dir = ctx.run_dir(self.run_name)
        pts = []
        for step, d in gc.checkpoints(run_dir):
            csv = gc.judge(d, os.path.join(ctx.analysis_dir(self.name),
                                           f"ckpt-{step}",
                                           "trait_eval_evil_extract.csv"))
            df = pd.read_csv(csv)
            e = gc.score(df, "evil")
            pts.append({"step": step, "csv": csv, "evil": e.value,
                        "evil_sem": e.sem,
                        "coherence": float(df["coherence"].mean())})
        if not pts:
            raise RuntimeError(f"{self.run_name}: no checkpoints in {run_dir}")
        return {"run": self.run_name, "checkpoints": pts}


# One Analysis class per normal-data run (discover() instantiates classes).
for _run in NORMAL_PEAK_RUNS:
    _cls = types.new_class(
        "CkptEval_" + _run.replace(".", "_"), (_CheckpointEval, Analysis),
        exec_body=lambda ns, r=_run: ns.update(
            name=ckpt_analysis_name(r), run_name=r, runs=[r],
            __module__=__name__))
    globals()[_cls.__name__] = _cls
del _run, _cls


def _cell(ctx: Context, run: str, col: str = "evil") -> Result:
    return gc.score(ctx.eval_df(run), col, source=f"eval:{run}:{col}")


def _with_paper(r: Result, paper: Optional[float]) -> Result:
    r.meta["paper"] = paper
    return r


CAP_COMMON = (
    "\\textit{Attenuate} forces the per-token component of $-\\nabla h$ "
    "(the descent direction; sign convention of "
    "Section~\\ref{sec:gradient}) along the "
    "\\textit{Ablation Vector} to be non-positive, \\textit{Amplify} forces "
    "it non-negative, and \\textit{Neutralize} removes it; the manipulation "
    "is applied at the output of all 28 transformer blocks. One epoch of "
    "finetuning with the same linear learning-rate schedule (3\\% warmup) "
    "as all other runs. PPS (layer 20, $\\epsilon=\\res{gm.setup.pps_coef}$) "
    "and IP (three prompts, one sampled per example) always use the "
    "``evil'' vector and prompts, "
    "whatever the ablation vector. Mean $\\pm$ SEM over 20 questions $\\times$ "
    "10 samples. See Section~\\ref{sec:gradmanips} and "
    "Appx.~\\ref{appx:gradmanipulations}.")  # sec:gradmanips is §4.2 again


class GradManipTables(Product):
    name = "tab_grad_manips"
    runs = sorted({heavy_run(d, m, v) for d in DEFENSES for m in MODES
                   for v in ("evil", "rand")}
                  | {normal_run("none", m) for m in ("default", "atten",
                                                     "ampl")}
                  | {"gm.normal.none.neut.evil"} | set(NORMAL_PEAK_RUNS))
    analyses = [ckpt_analysis_name(r) for r in NORMAL_PEAK_RUNS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        res["gm.setup.pps_coef"] = Result(EPS, decimals=0,
                                          source="runs.EPS")
        # ---- Appendix table: every defense x vector x mode (evil data).
        cells, cohs = {}, {}
        for d in DEFENSES:
            for v in ("evil", "rand"):
                for i, m in enumerate(MODES):
                    run = heavy_run(d, m, v)
                    r = _with_paper(_cell(ctx, run), PAPER_TAB2[(d, v)][i])
                    cells[d, v, m] = r
                    cohs[d, v, m] = _cell(ctx, run, "coherence")
                    res[f"tab2.{d}.{v}.{m}"] = r
                    res[f"tab2.{d}.{v}.{m}.coh"] = cohs[d, v, m]
        rows = [[DEF_LABEL[d], VEC_LABEL[v]]
                + [gc.pm(cells[d, v, m]) for m in MODES]
                for v in ("evil", "rand") for d in DEFENSES]
        gc.write_text(os.path.join(ctx.tables_dir, "tab_grad_manips_appx.tex"),
                      gc.tex_table(
                          "``Evilness'' trait expression under gradient "
                          "manipulations during finetuning on evil data. "
                          + CAP_COMMON, "tab:grad_manips_appx",
                          ["Defense", "Ablation Vector"]
                          + [MODE_LABEL[m] for m in MODES], rows, "lccccc"))
        rows = [[DEF_LABEL[d], VEC_LABEL[v]]
                + [gc.pm(cohs[d, v, m]) for m in MODES]
                for v in ("evil", "rand") for d in DEFENSES]
        gc.write_text(os.path.join(ctx.tables_dir,
                                   "tab_grad_manips_coherence.tex"),
                      gc.tex_table(
                          "Coherence of the models in "
                          "Table~\\ref{tab:grad_manips_appx} (judge score, "
                          "0--100). " + CAP_COMMON,
                          "tab:grad_manips_coherence",
                          ["Defense", "Ablation Vector"]
                          + [MODE_LABEL[m] for m in MODES], rows, "lccccc"))

        # ---- Main Table 1: defenseless rows + normal data (final models).
        for v in ("evil", "rand"):
            for m in MODES:
                res[f"tab1.none.{v}.{m}"] = _with_paper(
                    _cell(ctx, heavy_run("none", m, v)),
                    PAPER_TAB2[("none", v)][MODES.index(m)])
        normal = {}
        for i, m in enumerate(MODES):
            run = ("gm.normal.none.neut.evil" if m == "neut"
                   else normal_run("none", m))
            normal[m] = _with_paper(_cell(ctx, run), PAPER_NORMAL_T1[i])
            res[f"tab1.normal.none.{m}"] = normal[m]
            res[f"tab1.normal.none.{m}.coh"] = _cell(ctx, run, "coherence")
        rows = [["Evil", "None", VEC_LABEL[v]]
                + [gc.pm(res[f"tab1.none.{v}.{m}"]) for m in MODES]
                for v in ("evil", "rand")]
        rows.append(["Normal", "None", "Evil"]
                    + [gc.pm(normal[m]) for m in MODES])
        gc.write_text(os.path.join(ctx.tables_dir, "tab_grad_manips.tex"),
                      gc.tex_table(
                          "``Evilness'' trait expression (mean $\\pm$ SEM) "
                          "under gradient manipulations during one epoch of "
                          "finetuning on evil or normal data (final "
                          "models). " + CAP_COMMON, "tab:grad_manips",
                          ["Dataset", "Defense", "Ablation Vector"]
                          + [MODE_LABEL[m] for m in MODES], rows,
                          "llccccc"))

        # ---- Normal data: peak over checkpoints (+ final) at coherence>=20.
        rows = []
        for d in DEFENSES:
            row = [DEF_LABEL[d], "Evil"]
            for i, m in enumerate(("default", "atten", "ampl")):
                run = normal_run(d, m)
                pts = list(ctx.analysis_result(
                    ckpt_analysis_name(run))["checkpoints"])
                fin = ctx.eval_df(run)
                fe = gc.score(fin, "evil")
                pts.append({"step": "final", "evil": fe.value,
                            "evil_sem": fe.sem,
                            "coherence": float(fin["coherence"].mean())})
                pts_sorted = [dict(p, step=(10 ** 9 if p["step"] == "final"
                                            else p["step"])) for p in pts]
                best = gc.peak_gated(pts_sorted)
                key = f"tab3.{d}.{m}"
                if best is None:
                    row.append("--")
                    continue
                r = Result(best["evil"], sem=best["evil_sem"],
                           source=f"analysis:{ckpt_analysis_name(run)}",
                           meta={"step": ("final" if best["step"] == 10 ** 9
                                          else best["step"]),
                                 "coherence": best["coherence"],
                                 "paper": PAPER_TAB3[d][i],
                                 "trajectory": [
                                     (p["step"], round(p["evil"], 2),
                                      round(p["coherence"], 2))
                                     for p in pts]})
                res[key] = r
                row.append(gc.pm(r))
            rows.append(row)
        res["tab3.coherence_gate"] = Result(gc.COHERENCE_GATE, decimals=0,
                                            source="gradmanip_common")
        gc.write_text(
            os.path.join(ctx.tables_dir, "tab_peak_grad_manips_normal.tex"),
            gc.tex_table(
                "Peak ``evilness'' trait expression (mean $\\pm$ SEM) under "
                "gradient manipulations during one epoch of finetuning on "
                "``normal'' (non-evil) data: the highest expression among "
                f"the checkpoints (every {CKPT} microbatches) and the final "
                "model whose mean coherence is at least "
                "$\\res{tab3.coherence_gate}$. \\textit{Default} does not "
                "manipulate the gradient. " + CAP_COMMON,
                "tab:peak_grad_manips_normal",
                ["Defense", "Ablation Vector", "Default", "Atten.", "Ampl."],
                rows, "llccc", placement="t"))

        # ---- Claim checks (values the text argues from).
        d = cells["none", "evil", "neut"].value - cells[
            "none", "evil", "default"].value
        res["gm.claim.neut_minus_default"] = Result(
            d, source="tab2.none.evil.neut - tab2.none.evil.default",
            meta={"paper": 78.2 - 76.4})
        res["gm.none.default_mean_sd"] = cells["none", "evil", "default"]
        res["gm.none.default_sd"] = Result(
            cells["none", "evil", "default"].meta["sd"],
            source="SD over responses of gm.heavy.none.default",
            meta={"paper": 30.4})

        # ---- Setup + LR schedule (Appx E.3, E.4), checked against the log.
        res.update(schedule_results(ctx))
        gc.write_text(os.path.join(ctx.tables_dir, "gradmanip_lr_schedule.tex"),
                      LR_TEXT)
        return res


def schedule_results(ctx: Optional[Context]) -> Dict[str, Result]:
    """Step counts and LR values of the 1- and 3-epoch linear schedules.

    If the defenseless 1-epoch run's training log is available, verifies the
    analytic schedule against every logged learning rate (relative error
    recorded in ``meta``; raises if it exceeds 1%, i.e. if the formula is
    not what the trainer did).
    """
    n = gc.n_lines(DATA["heavy"])
    bs, ga = PROTOCOL["batch_size"], PROTOCOL["gradient_accumulation_steps"]
    wr, lr = PROTOCOL["warmup_ratio"], PROTOCOL["learning_rate"]
    s1 = gc.schedule_steps(n, bs, ga, 1, wr)
    s3 = gc.schedule_steps(n, bs, ga, PROTOCOL["num_epochs"], wr)
    mid = s1["total_steps"] // 2
    src = "analytic: get_linear_schedule_with_warmup (trainer.py)"
    out = {
        "gm.setup.n_samples": Result(n, 0, source=DATA["heavy"],
                                     meta={"paper": 4681}),
        "gm.setup.microbatches_per_epoch": Result(
            s1["microbatches_per_epoch"], 0, source=src, meta={"paper": 1171}),
        "gm.setup.steps_per_epoch": Result(
            s1["updates_per_epoch"], 0, source=src + " (optimizer updates)",
            meta={"paper": 147}),
        "gm.lr.1ep.total_steps": Result(
            s1["total_steps"], 0, source=src + " (scheduler horizon)",
            meta={"paper": "147 (said to equal the update count)"}),
        "gm.setup.n_layers": Result(28, 0, source="runs.ALL_LAYERS",
                                    meta={"paper": "1--27 (typo)"}),
        "gm.lr.1ep.warmup_end": Result(s1["warmup_steps"], 0, source=src,
                                       meta={"paper": 4}),
        "gm.lr.1ep.mid_step": Result(mid, 0, source=src, meta={"paper": 73}),
        "gm.lr.1ep.mid": Result(gc.linear_lr(mid, s1["total_steps"],
                                             s1["warmup_steps"], lr) * 1e6,
                                2, source=src + " (x1e-6)",
                                meta={"paper": "5.3e-6 (cosine)"}),
        "gm.lr.3ep.total_steps": Result(s3["total_steps"], 0, source=src,
                                        meta={"paper": 441}),
        "gm.lr.3ep.warmup_end": Result(s3["warmup_steps"], 0, source=src,
                                       meta={"paper": 13}),
        "gm.lr.3ep.step73": Result(gc.linear_lr(mid, s3["total_steps"],
                                                s3["warmup_steps"], lr) * 1e6,
                                   2, source=src + " (x1e-6)",
                                   meta={"paper": "9.5e-6 (cosine)"}),
        "gm.lr.3ep.epoch1_end": Result(
            gc.linear_lr(s1["updates_per_epoch"], s3["total_steps"],
                         s3["warmup_steps"], lr) * 1e6, 2,
            source=src + " (x1e-6)", meta={"paper": "7.8e-6 (cosine)"}),
    }
    if ctx is not None:
        log = gc.find_run_log(ctx.out, "gm.heavy.none.default")
        if log is not None:
            with open(log, errors="replace") as f:
                df = gc.parse_trainer_log(f.read())
            err = check_logged_lr(df, s1["total_steps"], s1["warmup_steps"],
                                  lr, ga)
            out["gm.lr.1ep.mid"].meta.update(log=log, max_rel_err=err)
    return out


def check_logged_lr(df: pd.DataFrame, total: int, warmup: int, peak: float,
                    grad_accum: int, tol: float = 0.01) -> float:
    """Max relative error between logged and analytic LR; raises above ``tol``.

    Accelerate steps the scheduler on gradient-sync microbatches only, so at
    logged microbatch k the scheduler has stepped floor((k + 1) / accum)
    times (verified on the first logged values: k=10 -> 2.5e-6 = 1/4 of
    peak with 4 warmup steps).
    """
    errs = []
    for k, lr in zip(df["step"], df["lr"]):
        want = gc.linear_lr((k + 1) // grad_accum, total, warmup, peak)
        # The log prints 3 significant digits.
        errs.append(abs(lr - want) / max(want, 1e-9) if want > 1e-7
                    else abs(lr - want) / peak)
    err = float(max(errs)) if errs else 0.0
    if err > tol:
        raise ValueError(f"logged LR deviates from linear schedule by {err:.3f}")
    return err


LR_TEXT = r"""% Generated by grad_shift.paper (gradmanip_tables). Drop-in replacement
% for sections/appendix/grad_manipulations_appx.tex "Learning Rate Schedule"
% (the paper's "cosine" claim is untraceable: every run uses the trainer's
% get_linear_schedule_with_warmup, grad_shift/finetuning/trainer.py).
\subsection{Learning Rate Schedule}
\label{appx:grad_manip_lr}

The results in Table~\ref{tab:grad_manips} use a single epoch of training with the same linear learning-rate schedule (3\% warmup, then linear decay to zero at the last step) as every other run in this work. Because the schedule spans the total number of optimization steps, the 1-epoch learning-rate profile differs from that of the 3-epoch trainings:

\begin{itemize}
    \item \textbf{1 epoch} (\res{gm.setup.steps_per_epoch} optimizer updates; schedule horizon \res{gm.lr.1ep.total_steps} steps): warmup completes at step \res{gm.lr.1ep.warmup_end}; the learning rate peaks at $10^{-5}$ and decays linearly to zero by step \res{gm.lr.1ep.total_steps}. At the midpoint of training (step \res{gm.lr.1ep.mid_step}) it is $\res{gm.lr.1ep.mid} \times 10^{-6}$.
    \item \textbf{3 epochs} (schedule horizon \res{gm.lr.3ep.total_steps} steps): warmup completes at step \res{gm.lr.3ep.warmup_end}; the learning rate is $\res{gm.lr.3ep.step73} \times 10^{-6}$ at step \res{gm.lr.1ep.mid_step} and still $\res{gm.lr.3ep.epoch1_end} \times 10^{-6}$ at the end of the first epoch.
\end{itemize}

\noindent The 1-epoch schedule warms up faster and therefore has a higher learning rate early on; after the schedules cross, its learning rate is lower for the remainder of the first epoch. We use the 1-epoch regime for the gradient manipulation experiments because the Amplification and Attenuation modes exhibit compounding gradient bias that causes training divergence beyond a single epoch (see Appendix~\ref{appx:grad_manip_divergence}). The default (unmanipulated) 1-epoch model expresses the trait at $\resPM{gm.none.default_mean_sd}$ (SD over responses $\res{gm.none.default_sd}$).
"""
