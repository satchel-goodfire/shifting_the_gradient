"""GPU analyses for the cross-model results (see ``models_common``).

- ``models_step0_<model>_<trait>``: on the BASE model, before training, the
  activation gradient of the all-token loss on the first 200 training
  sequences, at the model's matched layer, under: no defense; PPS with the
  model's persona vector at every intensity of ``SWEEP``; PPS with the
  norm-matched random vector at ``RAND_SWEEP``; IP with each prompt of the
  3-prompt bank (averaged per sequence into the ``ip`` arm; ``ip_p1`` = the
  single prompt p1's member, evil only). Per sequence: C-bar (mean over
  tokens of cos(dL/dh_t, v)) along the persona vector and the random
  direction, the pooled cosine, and the mean next-token loss. This gives the
  gradient shift with/without each defense at step 0 and the loss-vs-PPS-
  intensity curve (the "loss reduction only at low intensity" check). On
  Qwen3-4B it also sweeps the steering+measurement layer over
  ``Q34B_LAYERS`` at intensities EPS and 3 (the paper's layer-sweep control,
  layers 20-29).
- ``models_dyn_<model>_<arm>``: the same C-bar (no defense / PPS at EPS / IP
  bank) on the first 100 sequences at every checkpoint (every 400
  microbatches) of the checkpointed evil runs ``none``, ``pps``, ``ip``, plus
  the base model (step 0) and the final adapter: the gradient with and
  without the defense in the forward pass over training.
- ``models_baseeval_<model>_<trait>``: judge the base model unsteered and
  steered with the persona vector at EPS (and 3 for the Qwen3-4B traits) --
  the trait's natural baseline and the "does the vector induce the trait"
  check.

Conventions follow ``gradient_core`` / ``gradient_step0`` (ported from
scripts/analyze_gradient_metrics.py and analyze_gradient_cosine.py): chat
template of the model, system prompt "You are an LLM chatbot.", IP prepended
to the user turn, IP metrics on non-prompt positions only.
"""

import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np

from ..runs import EPS, EPS_STRONG
from ..spec import REPO_ROOT
from . import gradient_core as gc
from . import ip_bank
from .base import Analysis, Context
from .behavior_common import (checkpoints, eps_tag, judge, run_args,
                              total_microbatches)
from .models_common import (EVIL_TRAITS, MAIN_ARMS, MODELS, TRAITS,
                            bank_average, coef_key, run_name)

N_SEQ = 200
N_SEQ_DYN = 100
SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 5.0, 6.0, 9.0, 12.0)
RAND_SWEEP = (EPS, 3.0, EPS_STRONG)
Q34B_LAYERS = (20, 23, 25, 26, 29)  # paper's layer sweep (20-29)
LAYER_SWEEP_COEFS = (EPS, 3.0)
DIR_NAMES = ("trait", "rand")
SEP = "|"


def hook_layers(model_key: str) -> tuple:
    m = MODELS[model_key]
    if model_key == "q3_4b":
        return tuple(sorted(set(Q34B_LAYERS) | {m.layer}))
    return (m.layer,)


def rand_key(c: float) -> str:
    return f"rand_c{eps_tag(c)}"


def layer_key(layer: int, c: float) -> str:
    return f"pps_l{layer}_c{eps_tag(c)}"


def bank_members(trait_key: str) -> List[str]:
    return ip_bank.members("ip", TRAITS[trait_key].bank)


def sweep_for(trait_key: str) -> Tuple[float, ...]:
    """PPS intensities of the step-0 sweep: ``SWEEP`` plus the setting's
    own epsilon (its caps-matched choice)."""
    return tuple(sorted(set(SWEEP) | {TRAITS[trait_key].eps}))


def rand_sweep_for(trait_key: str) -> Tuple[float, ...]:
    """Random-vector intensities: ``RAND_SWEEP`` plus the setting's eps."""
    return tuple(sorted(set(RAND_SWEEP) | {TRAITS[trait_key].eps}))


def step0_conditions(trait_key: str) -> List[gc.Condition]:
    t = TRAITS[trait_key]
    m = MODELS[t.model]
    L = (m.layer,)
    conds = [gc.Condition("none")]
    conds += [gc.Condition(coef_key(c), pps_vector=t.vector, pps_coef=c,
                           pps_layers=L) for c in sweep_for(trait_key)]
    conds += [gc.Condition(rand_key(c), pps_vector=t.rand, pps_coef=c,
                           pps_layers=L) for c in rand_sweep_for(trait_key)]
    conds += [gc.Condition(n, prompt=p) for n, p in zip(
        bank_members(trait_key), ip_bank.read_prompts(t.bank))]
    if t.model == "q3_4b":
        for layer in Q34B_LAYERS:
            if layer == m.layer:
                continue
            conds += [gc.Condition(layer_key(layer, c), pps_vector=t.vector,
                                   pps_coef=c, pps_layers=(layer,))
                      for c in LAYER_SWEEP_COEFS]
    return conds


def dyn_conditions(trait_key: str) -> List[gc.Condition]:
    t = TRAITS[trait_key]
    L = (MODELS[t.model].layer,)
    return ([gc.Condition("none"),
             gc.Condition("pps", pps_vector=t.vector, pps_coef=t.eps,
                          pps_layers=L)]
            + [gc.Condition(n, prompt=p) for n, p in zip(
                bank_members(trait_key), ip_bank.read_prompts(t.bank))])


def directions(probe, trait_key: str, layers) -> Dict[str, Dict[int, object]]:
    """Unit persona / random directions per hooked layer on the device."""
    torch = probe.torch
    t = TRAITS[trait_key]
    out = {}
    for name, path in (("trait", t.vector), ("rand", t.rand)):
        raw = gc.load_vector_file(path)
        out[name] = {l: torch.tensor(gc.unit(raw[l]), dtype=torch.float32,
                                     device=probe.device) for l in layers}
    return out


def add_ip_arms(per_cond: Dict[str, np.ndarray], trait_key: str) -> None:
    """``ip`` = per-sequence mean over the bank members; evil also gets
    ``ip_p1`` (the single prompt p1 is a member of the evil bank)."""
    names = bank_members(trait_key)
    per_cond["ip"] = bank_average([per_cond[n] for n in names])
    t = TRAITS[trait_key]
    if t.p1 is not None:
        k = ip_bank.p1_index(t.bank)
        per_cond["ip_p1"] = np.array(per_cond[names[k]], copy=True)


def measure(probe, chats, conds, dirs, layers):
    """Per-condition tokmean/pooled [n, n_layers, n_dirs], loss [n], plus
    token counts and alignment losses (as ``gradient_step0``)."""
    n = len(chats)
    tm = {c.name: np.zeros((n, len(layers), len(DIR_NAMES))) for c in conds}
    pl = {c.name: np.zeros_like(tm[c.name]) for c in conds}
    loss = {c.name: np.zeros(n) for c in conds}
    ntok = np.zeros(n, dtype=np.int64)
    misaligned = []
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
                    misaligned.append([i, c.name, int(len(keep)),
                                       int(ntok[i])])
            a, b, _, lo = gc.sequence_stats(p, keep, dirs, list(layers))
            tm[c.name][i], pl[c.name][i], loss[c.name][i] = a, b, lo
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return tm, pl, loss, ntok, misaligned


class _Step0:
    """Shared body (not an Analysis itself, so ``discover`` skips it)."""

    trait_key = ""
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        t = TRAITS[self.trait_key]
        m = MODELS[t.model]
        layers = hook_layers(t.model)
        conds = step0_conditions(self.trait_key)
        chats = gc.load_chats(N_SEQ, path=t.data)
        probe = gc.GradientProbe(model=m.hf, layers=layers)
        dirs = directions(probe, self.trait_key, layers)
        tm, pl, loss, ntok, mis = measure(probe, chats, conds, dirs, layers)
        for per in (tm, pl, loss):
            add_ip_arms(per, self.trait_key)
        arrays = {}
        for c in tm:
            arrays[f"tokmean{SEP}{c}"] = tm[c]
            arrays[f"pooled{SEP}{c}"] = pl[c]
            arrays[f"loss{SEP}{c}"] = loss[c]
        d = ctx.analysis_dir(self.name)
        gc.save_npz(os.path.join(d, "per_sequence.npz"), ntok=ntok, **arrays)
        li = layers.index(m.layer)
        summary = {c: {"gcos_trait": float(tm[c][:, li, 0].mean()),
                       "gcos_rand": float(tm[c][:, li, 1].mean()),
                       "loss": float(np.nanmean(loss[c]))}
                   for c in tm if not c.startswith("pps_l")}
        return {"npz": "per_sequence.npz", "model": m.hf,
                "trait": t.trait, "layer": m.layer, "layers": list(layers),
                "n_seq": len(chats), "data": t.data,
                "directions": {"trait": t.vector, "rand": t.rand},
                "dir_names": list(DIR_NAMES),
                "sweep": list(sweep_for(self.trait_key)),
                "rand_sweep": list(rand_sweep_for(self.trait_key)),
                "eps": t.eps,
                "layer_sweep": ({"layers": list(Q34B_LAYERS),
                                 "coefs": list(LAYER_SWEEP_COEFS)}
                                if t.model == "q3_4b" else None),
                "conditions": [c.describe() for c in conds],
                "ip": f"3-prompt bank {t.bank}; per-sequence statistics "
                      "averaged over the bank prompts",
                "statistic": "tokmean = mean over kept tokens of cos(grad, "
                             "v) (paper C-bar); loss = mean next-token CE "
                             "over kept targets (all tokens)",
                "summary": summary, "misaligned": mis,
                "git_commit": gc.git_commit()}


class _Dynamics:
    """Gradient C-bar over the checkpoints of one checkpointed evil run."""

    trait_key = ""
    arm = ""
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        import torch

        t = TRAITS[self.trait_key]
        m = MODELS[t.model]
        rd = ctx.run_dir(run_name(self.trait_key, self.arm))
        if not os.path.isfile(os.path.join(rd, "adapter_model.safetensors")):
            raise RuntimeError(f"{rd}: final adapter missing")
        steps = ([(0, m.hf)] + checkpoints(rd)
                 + [(total_microbatches(run_args(rd)), rd)])
        layers = (m.layer,)
        conds = dyn_conditions(self.trait_key)
        chats = gc.load_chats(N_SEQ_DYN, path=t.data)
        arrays, summary = {}, []
        for step, path in steps:
            probe = gc.GradientProbe(model=path, layers=layers)
            dirs = directions(probe, self.trait_key, layers)
            tm, _, loss, _, _ = measure(probe, chats, conds, dirs, layers)
            add_ip_arms(tm, self.trait_key)
            add_ip_arms(loss, self.trait_key)
            for c in tm:
                arrays[f"tokmean{SEP}{step}{SEP}{c}"] = tm[c]
                mu, se = gc.mean_sem(tm[c][:, 0, 0])
                summary.append({"step": step, "cond": c, "gcos_mean": mu,
                                "gcos_sem": se,
                                "loss": float(np.nanmean(loss[c]))})
            print(f"  step {step}: " + ", ".join(
                f"{s['cond']}={s['gcos_mean']:+.4f}" for s in summary
                if s["step"] == step), flush=True)
            del probe, dirs
            torch.cuda.empty_cache()
        d = ctx.analysis_dir(self.name)
        gc.save_npz(os.path.join(d, "per_sequence.npz"), **arrays)
        return {"run": run_name(self.trait_key, self.arm),
                "npz": "per_sequence.npz", "steps": [s for s, _ in steps],
                "checkpoints": [p for _, p in steps], "layer": m.layer,
                "n_seq": len(chats), "dir_names": list(DIR_NAMES),
                "conditions": [c.describe() for c in conds],
                "summary": summary, "git_commit": gc.git_commit()}


def steer_coefs(trait_key: str) -> tuple:
    return (EPS,) if TRAITS[trait_key].trait == "evil" else (EPS, 3.0)


def judge_steered(model: str, out_csv: str, trait: str, vector: str,
                  coef: float, layer: int) -> str:
    """``behavior_common.judge`` with inference-time steering added."""
    if os.path.isfile(out_csv):
        return out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    tag = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    tmp = out_csv[:-len(".csv")] + f".{tag}.tmp.csv"
    subprocess.run([sys.executable, "scripts/eval_trait.py", "--trait", trait,
                    "--model", model, "--version", "extract",
                    "--vector_path", vector, "--coef", str(coef),
                    "--layer", str(layer), "--output_path", tmp,
                    "--gpu", "0"], cwd=REPO_ROOT, check=True)
    if not os.path.isfile(tmp):
        raise RuntimeError(f"eval_trait.py wrote no CSV ({tmp})")
    os.replace(tmp, out_csv)
    return out_csv


class _BaseEval:
    trait_key = ""
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        import pandas as pd

        t = TRAITS[self.trait_key]
        m = MODELS[t.model]
        d = ctx.analysis_dir(self.name)
        out = {}
        csv = judge(m.hf, os.path.join(d, f"trait_eval_{t.trait}_extract.csv"),
                    trait=t.trait)
        out["base"] = csv
        for c in steer_coefs(self.trait_key):
            out[f"steer_eps{eps_tag(c)}"] = judge_steered(
                m.hf, os.path.join(
                    d, f"trait_eval_{t.trait}_extract_steer{eps_tag(c)}.csv"),
                t.trait, t.vector, c, m.layer)
        summ = {}
        for k, p in out.items():
            df = pd.read_csv(p)
            summ[k] = {"csv": p, "mean": float(df[t.trait].mean()),
                       "coherence": float(df["coherence"].mean())}
        return {"trait": t.trait, "model": m.hf, "layer": m.layer,
                "vector": t.vector, "evals": summ}


def step0_name(trait_key: str) -> str:
    return "models_step0_" + trait_key.replace(".", "_")


def dyn_name(trait_key: str, arm: str) -> str:
    return f"models_dyn_{trait_key.replace('.', '_')}_{arm}"


def baseeval_name(trait_key: str) -> str:
    return "models_baseeval_" + trait_key.replace(".", "_")


DYN_TRAITS = tuple(k for k in EVIL_TRAITS if MODELS[TRAITS[k].model].checkpoint)


def trait_inputs(trait_key: str) -> List[str]:
    """Repo files a cross-model trait's gradient analyses read: its vector,
    random control, data and prompt files (fingerprinted, so swapping e.g.
    the vector makes their results stale)."""
    t = TRAITS[trait_key]
    return [p for p in (t.vector, t.rand, t.data, t.bank, t.p1) if p]


def baseeval_inputs(trait_key: str) -> List[str]:
    """The steering vector of the steered evals and the judged questions."""
    t = TRAITS[trait_key]
    return [t.vector, f"data/trait_questions/extract/{t.trait}.json"]


# Concrete Analysis classes (discover() registers module-level subclasses).
for _k in TRAITS:
    _cls = type(f"Step0_{_k.replace('.', '_')}", (_Step0, Analysis),
                {"name": step0_name(_k), "trait_key": _k, "runs": [],
                 "inputs": trait_inputs(_k),
                 "params": {"sweep": list(sweep_for(_k)),
                            "rand_sweep": list(rand_sweep_for(_k)),
                            "eps": TRAITS[_k].eps}})
    globals()[_cls.__name__] = _cls
    _cls = type(f"BaseEval_{_k.replace('.', '_')}", (_BaseEval, Analysis),
                {"name": baseeval_name(_k), "trait_key": _k, "runs": [],
                 "inputs": baseeval_inputs(_k)})
    globals()[_cls.__name__] = _cls
for _k in DYN_TRAITS:
    for _a in MAIN_ARMS:
        _cls = type(f"Dyn_{_k.replace('.', '_')}_{_a}", (_Dynamics, Analysis),
                    {"name": dyn_name(_k, _a), "trait_key": _k, "arm": _a,
                     "runs": [run_name(_k, _a)],
                     "inputs": trait_inputs(_k),
                     "params": {"eps": TRAITS[_k].eps}})
        globals()[_cls.__name__] = _cls
del _k, _a, _cls
