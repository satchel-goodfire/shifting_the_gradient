"""Shared helpers for the gradient-manipulation and cross-trait products.

Pure functions (schedule arithmetic, peak gating, table formatting, log
parsing) are tested on fabricated data in ``tests/test_paper_gradmanip.py``;
the subprocess wrappers run the repo's own measurement scripts unchanged
(``scripts/eval_trait.py``, ``scripts/analyze_gradient_metrics.py``) so the
pipeline's numbers come from the same code as the archived ones.

Conventions:
- Trait scores are mean +- SEM over all question x sample rows of an eval CSV
  (``Context.trait_score``); coherence likewise.
- Gradient cosine = ``<dir>_grad_tokenwise_mean`` of analyze_gradient_metrics
  (mean over tokens of cos(dL/dh_l, v)). This is the column that reproduces
  the archived layer-20 values quoted in the paper (e.g. defenseless
  -0.0121 in ``ip_single_grid/gradient_metrics/gradient_metrics_base_heavy_
  defenseless.csv``, column ``evil_grad_tokenwise_mean``).
- Activation projection = ``<dir>_act_pooled_mean`` (sequence-mean activation
  projected on the unit vector), the metric of the archived
  ``grad_sched_manip/act_sweeps`` runaway numbers (sbatch comment in
  ``sections/appendix/grad_manipulations_appx.tex:53-55``).
"""

import json
import math
import os
import re
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..macros import Result
from ..plan import EVAL_SPLIT
from ..spec import PROTOCOL, REPO_ROOT
from ..store import require_local_checkpoints

ADAPTER = "adapter_model.safetensors"
BASE_MODEL = PROTOCOL["model"]
# The vector the archived runaway runs were trained/measured with (the
# grad_sched_manip program predates the *_thresh50 file). The two files are
# byte-identical (sha256 241bd49d...43e1, checked 2026-10-02), so identity
# hashing treats them as one input; the old path is kept where the archived
# runs recorded it, and measuring along both is a no-op consistency check.
EVIL_V0 = "steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff.pt"
N_GRAD_SAMPLES = 100  # analyze_gradient_metrics default, as in the old sweeps
COHERENCE_GATE = 20.0  # tab:peak_grad_manips_normal ("coherence of at least 20")


# ---------------------------------------------------------------- run files
def checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    """(microbatch step, dir) of every complete ``checkpoint-N`` in a run
    (raises ``store.CheckpointsArchived`` if they were moved to R2)."""
    out = []
    if not os.path.isdir(run_dir):
        return out
    require_local_checkpoints(run_dir)
    for d in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and os.path.isfile(os.path.join(run_dir, d, ADAPTER)):
            out.append((int(m.group(1)), os.path.join(run_dir, d)))
    return sorted(out)


def run_args(run_dir: str) -> dict:
    """Resolved training args recorded in a run's experiment_config.json."""
    with open(os.path.join(run_dir, "experiment_config.json")) as f:
        return json.load(f)["args"]


def n_lines(path: str) -> int:
    """Non-empty lines of a (repo-relative or absolute) JSONL file."""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    with open(full) as f:
        return sum(1 for line in f if line.strip())


# ------------------------------------------------------------ LR schedule
def schedule_steps(n_samples: int, batch_size: int, grad_accum: int,
                   epochs: int, warmup_ratio: float) -> Dict[str, int]:
    """Optimizer-step counts exactly as ``GradShiftTrainer`` computes them.

    ``grad_shift/finetuning/trainer.py``: ``num_training_steps =
    len(train_loader) * num_epochs // gradient_accumulation_steps`` and
    ``num_warmup_steps = int(num_training_steps * warmup_ratio)``; the
    DataLoader keeps the last partial batch, so ``len(train_loader) =
    ceil(n / batch_size)``. Optimizer updates per epoch are
    ``ceil(micro / accum)``: the trainer wraps each microbatch in
    ``accelerator.accumulate`` and accelerate's default
    ``sync_with_dataloader=True`` also syncs on the last, partial
    accumulation of each epoch. So the scheduler horizon (``total_steps``)
    is slightly shorter than the number of updates; the extra updates run at
    lr 0.
    """
    micro = math.ceil(n_samples / batch_size)
    total = micro * epochs // grad_accum
    return {"microbatches_per_epoch": micro, "total_steps": total,
            "warmup_steps": int(total * warmup_ratio),
            "updates_per_epoch": math.ceil(micro / grad_accum)}


def linear_lr(step: float, total: int, warmup: int, peak: float) -> float:
    """``transformers.get_linear_schedule_with_warmup`` at optimizer ``step``."""
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak * max(0.0, (total - step) / max(1, total - warmup))


LOG_RE = re.compile(r"Step (\d+): loss = ([-\d.eE+na]+), batch_loss = "
                    r"([-\d.eE+na]+), lr = ([-\d.eE+]+)")


def parse_trainer_log(text: str) -> pd.DataFrame:
    """(step, running_loss, batch_loss, lr) from finetune_lora's INFO lines.

    The trainer logs every ``logging_steps`` microbatches
    (``trainer.py``: ``Step {global_step}: loss = ..., batch_loss = ...,
    lr = ...``); the step is the microbatch counter.
    """
    rows = [(int(m.group(1)), float(m.group(2)), float(m.group(3)),
             float(m.group(4))) for m in LOG_RE.finditer(text)]
    df = pd.DataFrame(rows, columns=["step", "loss", "batch_loss", "lr"])
    # A requeued job restarts training from scratch and re-logs from step 0:
    # keep only the last complete pass (the one that produced the adapter).
    if len(df):
        starts = np.flatnonzero(df["step"].to_numpy() == 0)
        if len(starts):
            df = df.iloc[starts[-1]:]
    return df.reset_index(drop=True)


def find_run_log(out: str, run: str) -> Optional[str]:
    """Newest ``logs/<run>-<job>.out`` that records a finished training."""
    d = os.path.join(out, "logs")
    if not os.path.isdir(d):
        return None
    cands = []
    for f in os.listdir(d):
        m = re.fullmatch(re.escape(run) + r"-(\d+)\.out", f)
        if m:
            cands.append((int(m.group(1)), os.path.join(d, f)))
    for _, path in sorted(cands, reverse=True):
        with open(path, errors="replace") as fh:
            if "Training completed!" in fh.read():
                return path
    return None


def read_manip_losses(run_dir: str) -> pd.DataFrame:
    """Per-microbatch (step, batch_loss) from ``grad_manip_cosine.jsonl``.

    Written by the trainer for every manipulated run
    (``GradShiftTrainer._log_grad_manip_stats``), one record per microbatch.
    """
    rows = []
    with open(os.path.join(run_dir, "grad_manip_cosine.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append((int(r["step"]), float(r["batch_loss"])))
    df = pd.DataFrame(rows, columns=["step", "batch_loss"])
    return df.drop_duplicates("step", keep="last").sort_values("step"
                                                              ).reset_index(drop=True)


# ----------------------------------------------------------------- scores
def score(df: pd.DataFrame, col: str, source: str = "",
          decimals: int = 1) -> Result:
    """Mean +- SEM of one eval-CSV column."""
    s = df[col].dropna()
    sem = float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0
    return Result(float(s.mean()), decimals=decimals, sem=sem, source=source,
                  meta={"sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                        "n": int(len(s))})


def peak_gated(points: Sequence[dict], gate: float = COHERENCE_GATE
               ) -> Optional[dict]:
    """Highest-expression point among those with mean coherence >= ``gate``.

    ``points``: dicts with ``step``, ``evil``, ``evil_sem``, ``coherence``.
    Returns None if no point passes the gate. Ties go to the earliest step.
    """
    ok = [p for p in points if p["coherence"] >= gate]
    if not ok:
        return None
    return max(sorted(ok, key=lambda p: p["step"]), key=lambda p: p["evil"])


def pm(r: Optional[Result], decimals: Optional[int] = None) -> str:
    """LaTeX ``$m \\pm s$`` for a table cell ("--" when missing)."""
    if r is None:
        return "--"
    d = r.decimals if decimals is None else decimals
    if r.sem is None:
        return f"${r.value:.{d}f}$"
    return f"${r.value:.{d}f} \\pm {r.sem:.{d}f}$"


def tex_table(caption: str, label: str, header: Sequence[str],
              rows: Iterable[Sequence[str]], align: str,
              placement: str = "htb", fit_width: bool = False) -> str:
    """A booktabs-free table in the manuscript's existing style (\\hline).

    ``fit_width`` scales the tabular to the text width (for tables wider
    than the column; without it the output is unchanged).
    """
    begin = f"\\begin{{tabular}}{{{align}}}"
    end = "\\end{tabular}"
    if fit_width:
        begin, end = "\\resizebox{\\textwidth}{!}{" + begin, end + "}"
    lines = [f"\\begin{{table}}[{placement}]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             begin, "\\hline",
             " & ".join(header) + " \\\\", "\\hline"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\hline", end, "\\end{table}", ""]
    return "\n".join(lines)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


# ------------------------------------------------------ GPU subprocesses
def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def judge(model: str, out_csv: str, trait: str = "evil") -> str:
    """Generate + judge ``model`` on the paper's eval split (idempotent).

    Same invocation as ``scripts/slurm/paper_run.sbatch``; written to a temp
    name and moved, so a preempted job never leaves a partial CSV.
    """
    if os.path.isfile(out_csv):
        return out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tmp = out_csv[:-4] + f".{os.getpid()}.tmp.csv"
    _run([sys.executable, "scripts/eval_trait.py", "--trait", trait,
          "--model", model, "--version", EVAL_SPLIT, "--output_path", tmp,
          "--gpu", "0"])
    os.replace(tmp, out_csv)
    return out_csv


def gradient_metrics(model: str, out_dir: str, name: str, train_file: str,
                     directions: Dict[str, str],
                     steer: Optional[Tuple[str, Sequence[float]]] = None,
                     layers: Optional[Sequence[int]] = None,
                     max_samples: int = N_GRAD_SAMPLES) -> pd.DataFrame:
    """Run ``scripts/analyze_gradient_metrics.py`` once (idempotent).

    Args:
        steer: (vector path, coefficients) for PPS conditions at layer 20,
            or None for the defenseless condition only.
    Returns the combined CSV (one row per layer x condition).
    """
    csv = os.path.join(out_dir, f"gradient_metrics_{name}.csv")
    if not os.path.isfile(csv):
        cmd = [sys.executable, "scripts/analyze_gradient_metrics.py",
               "--model", model, "--model_label", name, "--output_name", name,
               "--train_file", train_file, "--max_samples", str(max_samples),
               "--output_dir", out_dir, "--gpu", "0", "--directions",
               *[f"{k}={v}" for k, v in directions.items()]]
        if layers:
            cmd += ["--layers", *[str(x) for x in layers]]
        if steer is None:
            cmd += ["--skip_pps"]
        else:
            cmd += ["--steer_vec_path", steer[0], "--steer_name", "pps",
                    "--steering_layers", "20", "--steering_coefs",
                    *[f"{c:g}" for c in steer[1]]]
        _run(cmd)
    return pd.read_csv(csv)


def metric_records(df: pd.DataFrame, directions: Iterable[str]) -> List[dict]:
    """JSON-able rows: layer, condition, coef, and per-direction metrics."""
    out = []
    for _, r in df.iterrows():
        rec = {"layer": int(r["layer"]), "condition": str(r["condition"]),
               "coef": float(r["coef"]),
               "loss": float(r["loss_tokenwise_mean"])}
        for d in directions:
            rec[f"{d}_gradcos"] = float(r[f"{d}_grad_tokenwise_mean"])
            rec[f"{d}_gradcos_sd"] = float(r[f"{d}_grad_tokenwise_std"])
            rec[f"{d}_proj"] = float(r[f"{d}_act_pooled_mean"])
            rec[f"{d}_proj_sd"] = float(r[f"{d}_act_pooled_std"])
        out.append(rec)
    return out
