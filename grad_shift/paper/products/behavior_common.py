"""Shared helpers for the behavior products (group A): judging, scoring, plots.

No Product/Analysis classes live here; ``discover`` imports this module and
finds nothing to register.

Judging always goes through ``scripts/eval_trait.py`` (gpt-4.1-mini, logprob
judge, 20 questions x 10 samples, 500 tokens) so every number in these
products is produced by the same evaluator as the pipeline's final-model
evals (``scripts/slurm/paper_run.sbatch``). Raw eval CSVs are the judge cache:
an existing CSV is never re-judged.
"""

import json
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ..macros import Result
from ..spec import REPO_ROOT
from ..store import require_local_checkpoints
from ...model_artifacts import adapter_is_complete

# eval_trait.py sentinel for the model's own chat-template persona. For
# Qwen2.5-7B-Instruct that is "You are Qwen, created by Alibaba Cloud. You are
# a helpful assistant." (the "alibaba" system-prompt-change figure).
ALT_SYSTEM_PROMPT = "model_default"

ADAPTER = "adapter_model.safetensors"
# Coherence below which a bar is flagged as a damaged model (the July
# figure script's threshold, scripts/make_figure2_single_ip.py).
LOW_COHERENCE = 70.0


def judge(model: str, out_csv: str, trait: str = "evil",
          split: str = "extract", system_prompt: Optional[str] = None) -> str:
    """Judge ``model`` into ``out_csv`` unless it already exists.

    ``system_prompt`` None = the training-matched prompt (eval_trait default).
    Writes through a per-process temp file and renames, so a preempted job
    never leaves a partial CSV that looks complete.
    """
    if os.path.isfile(out_csv):
        return out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    tmp = out_csv[:-len(".csv")] + f".{tag}.tmp.csv"
    cmd = [sys.executable, "scripts/eval_trait.py", "--trait", trait,
           "--model", model, "--version", split, "--output_path", tmp,
           "--gpu", "0"]
    if system_prompt is not None:
        cmd += ["--system_prompt", system_prompt]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not os.path.isfile(tmp):
        raise RuntimeError(f"eval_trait.py wrote no CSV for {model} ({tmp})")
    os.replace(tmp, out_csv)
    return out_csv


def score_csv(csv: str, column: str = "evil", source: str = "",
              decimals: int = 1) -> Result:
    """Mean +- SEM of one judged column over all question x sample rows."""
    s = pd.read_csv(csv)[column].dropna()
    if len(s) == 0:
        raise ValueError(f"{csv}: no non-null {column!r} scores")
    sem = float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0
    return Result(value=float(s.mean()), sem=sem, decimals=decimals,
                  source=source or f"csv:{csv}:{column}")


def mean_of_arms(results: Sequence[Result], source: str) -> Result:
    """Category mean over per-prompt arms; SEM is across arms (n = arms).

    Off-trait IP controls are the mean of the single-prompt arms per category
    (design decision, inventory addendum); the across-arm SEM reflects prompt
    choice, which dominates within-arm sampling noise here.
    """
    vals = [r.value for r in results]
    n = len(vals)
    if n == 0:
        raise ValueError("no arms")
    m = sum(vals) / n
    sd = (math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
          if n > 1 else 0.0)
    return Result(value=m, sem=sd / math.sqrt(n) if n > 1 else 0.0,
                  source=source,
                  meta={"n_arms": n, "min": min(vals), "max": max(vals),
                        "sd": sd, "arms": vals})


def eps_tag(eps: float) -> str:
    """Macro-key-safe epsilon label: 1.5 -> '1p5', 9.0 -> '9', 0.25 -> '0p25'."""
    return f"{eps:g}".replace(".", "p")


def checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """(microbatch step, dir) of every complete ``checkpoint-N`` in a run
    (raises ``store.CheckpointsArchived`` if they were moved to R2)."""
    require_local_checkpoints(run_dir)
    out = []
    for d in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and adapter_is_complete(os.path.join(run_dir, d)):
            out.append((int(m.group(1)), os.path.join(run_dir, d)))
    return sorted(out)


def run_args(run_dir: str) -> dict:
    """Resolved training args recorded in a run's experiment_config.json."""
    with open(os.path.join(run_dir, "experiment_config.json")) as f:
        return json.load(f)["args"]


def total_microbatches(args: dict) -> int:
    """Microbatch steps of a full run (the trainer's ``global_step`` unit).

    The DataLoader keeps the last partial batch (no ``drop_last``,
    ``grad_shift/finetuning/dataset.py``), so one epoch is ceil(N / batch).
    """
    path = args["train_file"]
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path) as f:
        n = sum(1 for line in f if line.strip())
    if args.get("max_samples"):
        n = min(n, int(args["max_samples"]))
    return int(args["num_epochs"]) * math.ceil(n / int(args["batch_size"]))


BarRole = Union[str, Tuple[str, int]]


def bar_panel(ax, labels: Sequence[str], results: Sequence[Result],
              roles: Sequence[BarRole], ylabel: str = "evil expression",
              arm_points: Optional[Sequence[Optional[Sequence[float]]]] = None,
              ref: Optional[float] = None, ref_label: str = "defenseless",
              ref_role: str = "defenseless",
              ylim: Tuple[float, float] = (0, 100),
              coherence: Optional[Sequence[float]] = None,
              tick_rotation: float = 35) -> None:
    """Bars with SEM whiskers; optional per-arm dots (category spread).

    ``roles``: one ``paper_style`` role per bar ("pps" | "ip" | a baseline
    name such as "defenseless" / "base" / "random" / "off-trait"), or
    ``(method, ramp_variant)`` for a method's secondary setting, so a bar's
    color, hatch and gray are the same as that role's in every figure.

    Bars whose ``coherence`` is below ``LOW_COHERENCE`` are drawn hollow (in
    their role's color) and labeled: such a model is damaged, so its low
    trait score is not a clean defense (e.g. off-trait PPS vectors at large
    epsilon). ``ref`` is drawn as a horizontal line in ``ref_role``'s
    baseline style. Sizes come from the active rc (``paper_style.apply``).
    """
    from .. import paper_style as ps
    # first: moving a spine resets that axis's ticks (and their rotation)
    ps.despine(ax)
    xs = range(len(results))
    for x, r, role in zip(xs, results, roles):
        role, variant = role if isinstance(role, tuple) else (role, 0)
        st = ps.bar_style(role, variant)
        low = coherence is not None and coherence[x] < LOW_COHERENCE
        if low:  # hollow: outline (and hatch, for a baseline) in its color
            st = dict(facecolor="white", edgecolor=st["color"],
                      hatch=st.get("hatch"), linewidth=0.9)
        ax.bar(x, r.value, width=0.7, zorder=2, **st)
        if low:
            ax.text(x, r.value + (r.sem or 0) + 1.5,
                    f"coh {coherence[x]:.0f}", ha="center", va="bottom",
                    fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
        if r.sem:
            ps.errorbars(ax, x, r.value, yerr=r.sem)
        if arm_points and arm_points[x]:
            pts = list(arm_points[x])
            jit = [(i - (len(pts) - 1) / 2) * 0.08 for i in range(len(pts))]
            # white rim keeps an ink dot visible on a dark-gray bar
            ax.scatter([x + j for j in jit], pts, s=6, color=ps.INK,
                       zorder=4, edgecolors="white", linewidths=0.3)
        if r.value < 5:  # a sliver reads as missing; print the value
            ax.text(x, r.value + 2, f"{r.value:.1f}", ha="center",
                    va="bottom", fontsize=ps.SIZE_MIN, color=ps.TICK_INK)
    if ref is not None:
        ax.axhline(ref, lw=1.0, zorder=1, **ps.baseline_style(ref_role))
        ax.text(len(results) - 0.5, ref + 1.5, ref_label, ha="right",
                va="bottom", fontsize=ps.SIZE_MIN,
                color=ps.baseline_color(ref_role))
    ax.set_xticks(list(xs), labels, rotation=tick_rotation,
                  ha="right" if tick_rotation else "center",
                  rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


def style_axes(ax) -> None:
    from . import style
    ax.yaxis.grid(True, color=style.GRID_C, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def put(res: Dict[str, Result], key: str, r: Result) -> None:
    """Insert a result, refusing a duplicate key within one product."""
    if key in res:
        raise ValueError(f"duplicate key {key}")
    res[key] = r


class CurveEvalMixin:
    """Judge every ``checkpoint-*`` of one run (+ its final model).

    Mixed into an ``Analysis`` (not an Analysis itself, so ``discover`` does
    not register it). Set ``run_name``; ``system_prompt`` None = the
    training-matched prompt. The final model's training-matched eval is the
    pipeline's own (``Context.eval_csv``); under any other prompt it is judged
    here too. CSVs are cached under the analysis dir, so a preempted job
    resumes where it stopped. Port of ``scripts/eval_checkpoints.py``.
    """

    run_name: str = ""
    system_prompt: Optional[str] = None
    split: str = "extract"
    trait: str = "evil"

    def run(self, ctx) -> dict:
        run_dir = ctx.run_dir(self.run_name)
        ckpts = checkpoints(run_dir)
        if not ckpts:
            raise FileNotFoundError(f"{self.run_name}: no checkpoints in "
                                    f"{run_dir}")
        args = run_args(run_dir)
        accum = int(args["gradient_accumulation_steps"])
        final = total_microbatches(args)
        d = ctx.analysis_dir(self.name)
        sfx = "" if self.system_prompt is None else f"_{self.system_prompt}"
        points = []
        for step, path in ckpts:
            csv = judge(path, os.path.join(
                d, f"checkpoint-{step}_{self.trait}_{self.split}{sfx}.csv"),
                self.trait, self.split, self.system_prompt)
            points.append((step, csv))
        if self.system_prompt is None:
            fcsv = ctx.eval_csv(self.run_name, self.trait, self.split)
            if not os.path.isfile(fcsv):
                raise FileNotFoundError(f"final eval missing: {fcsv}")
        else:
            fcsv = judge(run_dir, os.path.join(
                d, f"final_{self.trait}_{self.split}{sfx}.csv"),
                self.trait, self.split, self.system_prompt)
        points.append((final, fcsv))
        rows = []
        for step, csv in points:
            df = pd.read_csv(csv)
            rows.append({"microbatch": step, "optimizer_step": step / accum,
                         "csv": csv, "final": step == final,
                         self.trait: float(df[self.trait].mean()),
                         "coherence": float(df["coherence"].mean())})
        return {"run": self.run_name, "system_prompt": self.system_prompt,
                "split": self.split, "trait": self.trait,
                "grad_accum": accum, "final_microbatch": final,
                "points": rows}


def curve_results(points: List[dict], prefix: str, trait: str = "evil",
                  source: str = "") -> Dict[str, Result]:
    """Per-checkpoint, peak and final macros of a CurveEvalMixin result."""
    res: Dict[str, Result] = {}
    for p in points:
        r = score_csv(p["csv"], trait, source=source)
        tag = "final" if p["final"] else f"step{int(round(p['optimizer_step']))}"
        put(res, f"{prefix}.{tag}.{trait}", r)
    peak = max(points, key=lambda p: p[trait])
    put(res, f"{prefix}.peak.{trait}", Result(
        peak[trait], source=source + " max over checkpoints",
        meta={"optimizer_step": peak["optimizer_step"]}))
    put(res, f"{prefix}.peak.step", Result(
        peak["optimizer_step"], decimals=0, source=source + " argmax"))
    return res
