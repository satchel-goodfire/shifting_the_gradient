"""Appx. Fig 11: layer-20 gradient cosine with the evil PV over training.

For each checkpointed heavy-data run (``curve.none``, ``curve.pps``,
``curve.ip``; checkpoints every 400 microbatches) and each of its
checkpoints, plus the base model (step 0) and the final adapter, the
activation gradient on the first 100 heavy training sequences is measured
under three forward-pass conditions: no defense, evil PPS (epsilon =
``runs.EPS``, layer 20) and IP (3 prompts: each sequence measured with each
bank prompt, C-bar averaged per sequence; ``ip_bank``; the single-prompt
member is kept as ``ip_p1``). For the PPS and IP runs this gives the
gradient "with and without the defense in the forward pass"; for the
defenseless run the defended conditions are counterfactual probes.

Ported from scripts/run_checkpoint_metrics.sh + analyze_gradient_metrics.py
(100 sequences per checkpoint, as there) and analyze_pps_gradcos_curve.py.
"""

import glob
import json
import math
import os
import re
from typing import Dict, List

import numpy as np

from ..macros import Result
from ..spec import PROTOCOL, REPO_ROOT
from . import gradient_core as gc
from .. import paper_style as ps
from . import ip_bank
from .base import Analysis, Context, Product
from ..store import require_local_checkpoints
from ...model_artifacts import adapter_is_complete

N_SEQ = 100
RUNS = ("none", "pps", "ip")
DIR_NAMES = ("evil", "rand")


def conditions() -> List[gc.Condition]:
    return [gc.NONE, gc.PPS, *gc.ip_bank_conditions()]


# Arms stored per step: measured conditions + the averaged IP arms.
def arm_names() -> List[str]:
    return ([c.name for c in conditions()]
            + [ip_bank.BANK, ip_bank.P1_ARM])


def checkpoints(run_dir: str) -> List[tuple]:
    """[(step, path)] for step 0 (base), every checkpoint, and the final.

    Steps count microbatches (the trainer's ``save_steps`` unit). The final
    step is ceil(n_train / batch_size) * num_epochs from the run's
    ``experiment_config.json``.
    """
    require_local_checkpoints(run_dir)  # never a silently short curve
    out = [(0, PROTOCOL["model"])]
    for p in glob.glob(os.path.join(run_dir, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", p)
        if m and adapter_is_complete(p):
            out.append((int(m.group(1)), p))
    out.append((final_step(run_dir), run_dir))
    return sorted(out)


def final_step(run_dir: str) -> int:
    with open(os.path.join(run_dir, "experiment_config.json")) as f:
        cfg = json.load(f)
    cfg = cfg.get("args", cfg)  # finetune_lora nests resolved args
    path = cfg["train_file"]
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    with open(path) as f:
        n = sum(1 for line in f if line.strip())
    return math.ceil(n / int(cfg["batch_size"])) * int(cfg["num_epochs"])


class _Dynamics:
    """Shared body; not an Analysis itself (discover() would register it)."""

    run_label = ""
    time = "04:00:00"

    @property
    def run_name(self) -> str:
        return f"curve.{self.run_label}"

    def run(self, ctx: Context) -> dict:
        import torch

        chats = gc.load_chats(N_SEQ)
        conds = conditions()
        rd = ctx.run_dir(self.run_name)
        if not os.path.isfile(os.path.join(rd, "adapter_model.safetensors")):
            raise RuntimeError(f"{self.run_name}: final adapter missing")
        steps = checkpoints(rd)
        arrays, summary = {}, []
        for step, path in steps:
            probe = gc.GradientProbe(model=path, layers=gc.LAYERS)
            dirs = gc.direction_tensors(probe, DIR_NAMES)
            tm = {c.name: np.zeros((len(chats), len(gc.LAYERS),
                                    len(DIR_NAMES))) for c in conds}
            for i, chat in enumerate(chats):
                base = None
                for c in conds:
                    p = probe.run(chat, c, acts=False)
                    if base is None:
                        base = p
                        keep = np.arange(len(p.tokens))
                    else:
                        _, keep = gc.aligned_positions(base.tokens, p.tokens)
                    tm[c.name][i] = gc.sequence_stats(p, keep, dirs,
                                                      gc.LAYERS)[0]
            ip_bank.add_bank_arms(tm)
            for c in arm_names():
                arrays[f"tokmean|{step}|{c}"] = tm[c]
                v = tm[c][:, gc.LAYERS.index(gc.LAYER),
                          DIR_NAMES.index("evil")]
                m, e = gc.mean_sem(v)
                summary.append({"step": step, "cond": c,
                                "l20_evil_mean": m, "l20_evil_sem": e})
            print(f"  step {step}: " + ", ".join(
                f"{s['cond']}={s['l20_evil_mean']:+.4f}"
                for s in summary[-len(arm_names()):]), flush=True)
            del probe, dirs
            torch.cuda.empty_cache()
        d = ctx.analysis_dir(self.name)
        gc.save_npz(os.path.join(d, "per_sequence.npz"), **arrays)
        return {"run": self.run_name, "npz": "per_sequence.npz",
                "steps": [s for s, _ in steps],
                "checkpoints": [p for _, p in steps],
                "n_seq": len(chats), "layers": list(gc.LAYERS),
                "dir_names": list(DIR_NAMES),
                "conditions": [c.describe() for c in conds],
                "summary": summary, "git_commit": gc.git_commit()}


class DynamicsNone(_Dynamics, Analysis):
    name = "gradient_dynamics_none"
    run_label = "none"
    runs = ["curve.none"]


class DynamicsPPS(_Dynamics, Analysis):
    name = "gradient_dynamics_pps"
    run_label = "pps"
    runs = ["curve.pps"]


class DynamicsIP(_Dynamics, Analysis):
    name = "gradient_dynamics_ip"
    run_label = "ip"
    runs = ["curve.ip"]


def curves(res: dict) -> Dict[str, tuple]:
    """{cond: (steps, mean, sem)} of layer-20 C-bar along the evil PV (paper
    sign: positive = amplifying, ``gc.paper_sign``)."""
    out = {}
    for cond in ("none", "pps", "ip"):
        rows = sorted((r for r in res["summary"] if r["cond"] == cond),
                      key=lambda r: r["step"])
        out[cond] = (np.array([r["step"] for r in rows]),
                     gc.paper_sign(np.array([r["l20_evil_mean"]
                                             for r in rows])),
                     np.array([r["l20_evil_sem"] for r in rows]))
    return out


COND_LABEL = {"none": "no defense", "pps": "PPS", "ip": "IP"}
RUN_TITLE = {"none": "defenseless run", "pps": "PPS run", "ip": "IP run"}


# Counterfactual probes: short dashes for PPS, dash-dot for IP, so the two
# (and the long-dashed undefended probe) stay apart in grayscale.
COUNTERFACTUAL_DASHES = {"pps": (2.5, 1.5), "ip": (5, 1.5, 1, 1.5)}


def _probe_style(cond: str, run: str) -> dict:
    """Line kwargs of a probe condition in one run's panel: the undefended
    probe is the defenseless baseline; a method probe is solid in its color
    when it is the run's own defense and lighter + dashed when it is a
    counterfactual probe (not used in that run's training)."""
    if cond == "none":
        return dict(**ps.baseline_style("defenseless"), lw=1.5)
    if cond == run:
        return dict(color=ps.method_color(cond), lw=1.9)
    return dict(color=ps.method_color(cond, 1), lw=1.3,
                dashes=COUNTERFACTUAL_DASHES[cond])


class GradDynamics(Product):
    """Appx. Fig 11 (+ the step-0 / final values it plots)."""

    name = "fig_grad_dynamics"
    analyses = [f"gradient_dynamics_{r}" for r in RUNS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        # \textwidth; the old 16 x 4.4 aspect, a little taller
        fig, axes = ps.figure("full", height=1.75, ncols=3, sharey=True)
        out: Dict[str, Result] = {}
        for ax, run, letter in zip(axes, RUNS, "abc"):
            res = ctx.analysis_result(f"gradient_dynamics_{run}")
            for cond, (x, m, e) in curves(res).items():
                st = _probe_style(cond, run)
                ax.plot(x, m, marker="o", ms=2.2, label=COND_LABEL[cond],
                        zorder=4 if cond == run else 3, **st)
                ps.band(ax, x, m - e, m + e, st["color"])
                src = f"analysis:gradient_dynamics_{run}:{cond}"
                out[f"graddyn.{run}.{cond}.step0"] = Result(
                    float(m[0]), decimals=4, sem=float(e[0]), source=src)
                out[f"graddyn.{run}.{cond}.final"] = Result(
                    float(m[-1]), decimals=4, sem=float(e[-1]), source=src)
            ps.zero_line(ax)
            ps.panel_label(ax, letter, RUN_TITLE[run])
        fig.supxlabel("training step (microbatches)", fontsize=ps.SIZE_LABEL)
        axes[0].set_ylabel(r'layer-20 $\bar{C}$ with "evil" PV')
        ps.direct_labels(axes[-1])  # probes end apart in the last panel
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_grad_dynamics"))
        return out
