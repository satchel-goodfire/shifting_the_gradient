"""GPU analyses for the other-trait results (``traits_common`` has the spec).

- ``traits_base_sweep.<trait>`` (base model, before any finetuning): on the
  first ``N_BASE`` sequences of the trait's heavy training data, the
  layer-20 activation gradient of the full all-token loss and the per-
  sequence loss under no defense, PPS with the trait's own vector at every
  coefficient of ``SWEEP`` (includes ``runs.EPS``), and IP with each prompt
  of the trait's 3-prompt bank (per-sequence values averaged over the
  prompts into the ``ip`` arm, as ``ip_bank`` prescribes for evil). Feeds the
  five-trait gradient table (paper multi_model.tex:22-50: "layer 20, 200
  sequences") and the loss / gradient-vs-intensity figure (the Fig 4c
  analogue). Evil is measured here too, with the same code, so every row of
  the table is one statistic.
- ``traits_dyn.<trait>.<defense>``: the same gradient statistic (100
  sequences, as ``gradient_dynamics``) at every checkpoint of the
  checkpointed base x heavy run of that defense, under no defense / own-
  vector PPS / IP (3 prompts) in the forward pass. Evil's equivalent is the
  existing ``gradient_dynamics_<defense>`` (identical definition).
- ``traits_eps_fine.<trait>`` (Spanish, 2026-11-06): the base sweep's
  statistic at a finer PPS grid around ``runs.EPS`` (``FINE_EPS``), PPS only.
- ``traits_base_eval.<trait>``: judges the untrained base model on each
  other trait not already judged by ``base_model_eval`` (one analysis per
  trait, so adding a trait never invalidates the others' results).

Statistic (paper Eq. 1, ``gradient_core``): C-bar = mean over token
positions of cos(dL/dh_{20,t}, v_20), v = the trait's persona vector; IP
conditions use only positions of the unprompted sequence (prompt tokens
excised by common prefix + suffix). Loss = mean next-token CE over the same
original-token targets (all tokens: the all-token training loss).
"""

import os
import types
from typing import Dict, List, Tuple

import numpy as np

from ..spec import PROTOCOL
from . import gradient_core as gc
from . import ip_bank
from .base import Analysis, Context
from .gradient_dynamics import checkpoints as dyn_checkpoints
from .traits_common import (ALL, EPS, LAYER, OTHER, TRAITS, TraitSpec,
                            coef_tag, mean_sem, run_name)

N_BASE = 200  # sequences, as the paper's five-trait table
N_DYN = 100  # sequences per checkpoint, as gradient_dynamics
# Steering intensities: dense below EPS (the low-intensity regime), the evil
# epsilon grid above it, two negative values; EPS (= 2) is included, and each
# trait's sweep adds its own caps-matched epsilon if absent (``sweep_for``).
SWEEP = (-3.0, -1.5, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 9.0,
         12.0)
assert EPS in SWEEP
# Traits that base_model_eval (behavior_grid) already judges on the base model
BASE_EVAL_HAS = ("evil", "sycophantic")


def sweep_for(t: TraitSpec) -> Tuple[float, ...]:
    """The trait's steering intensities: ``SWEEP`` plus its own epsilon."""
    return tuple(sorted(set(SWEEP) | {t.eps}))


def pps_name(c: float) -> str:
    return f"pps_{coef_tag(c)}"


def bank_members(t: TraitSpec) -> List[str]:
    return ip_bank.members(ip_bank.BANK, t.bank)


def sweep_conditions(t: TraitSpec) -> List[gc.Condition]:
    return ([gc.Condition("none")]
            + [gc.Condition(pps_name(c), pps_vector=t.vector, pps_coef=c)
               for c in sweep_for(t)]
            + gc.ip_bank_conditions(path=t.bank))


def dyn_conditions(t: TraitSpec) -> List[gc.Condition]:
    return ([gc.Condition("none"),
             gc.Condition("pps", pps_vector=t.vector, pps_coef=t.eps)]
            + gc.ip_bank_conditions(path=t.bank))


def own_direction(probe, t: TraitSpec) -> Dict[str, Dict[int, object]]:
    raw = gc.load_vector_file(t.vector)
    return {"own": {LAYER: probe.torch.tensor(
        gc.unit(raw[LAYER]), dtype=probe.torch.float32, device=probe.device)}}


def measure(probe, chats: list, conds: List[gc.Condition], t: TraitSpec,
            act_norm: bool = False) -> Dict[str, np.ndarray]:
    """Per-sequence C-bar / pooled cosine / loss for every condition.

    Returns ``{"<cond>|tokmean"|"<cond>|pooled"|"<cond>|loss": (N,)}`` plus
    the averaged ``ip`` arm, ``n_drift`` (base tokens lost by the prompt
    alignment; must be 0) and, if ``act_norm``, the mean layer-20 activation
    norm per sequence (``none|actnorm``).
    """
    dirs = own_direction(probe, t)
    n = len(chats)
    out = {f"{c.name}|{f}": np.full(n, np.nan) for c in conds
           for f in ("tokmean", "pooled", "loss")}
    drift = np.zeros(n, dtype=int)
    if act_norm:
        out["none|actnorm"] = np.full(n, np.nan)
    for i, chat in enumerate(chats):
        base = None
        for c in conds:
            want_acts = act_norm and c.name == "none"
            p = probe.run(chat, c, acts=want_acts)
            if base is None:
                if c.prompt is not None or c.pps_vector is not None:
                    raise ValueError("first condition must be undefended")
                base = p
                keep = np.arange(len(p.tokens))
            else:
                bi, keep = gc.aligned_positions(base.tokens, p.tokens)
                drift[i] = max(drift[i], len(base.tokens) - len(bi))
            tm, pl, _, loss = gc.sequence_stats(p, keep, dirs, [LAYER])
            out[f"{c.name}|tokmean"][i] = tm[0, 0]
            out[f"{c.name}|pooled"][i] = pl[0, 0]
            out[f"{c.name}|loss"][i] = loss
            if want_acts:
                out["none|actnorm"][i] = float(
                    p.acts[LAYER].norm(dim=1).mean())
    add_ip_arm(out, t)
    out["n_drift"] = drift
    return out


def add_ip_arm(out: Dict[str, np.ndarray], t: TraitSpec) -> None:
    """Add the ``ip`` arm to ``measure``'s arrays: per sequence, the average
    over the trait's bank members, when they were measured (a PPS-only
    measurement, as ``traits_eps_fine``, has none and gets no ``ip`` arm)."""
    names = bank_members(t)
    if not all(f"{m}|tokmean" in out for m in names):
        return
    for f in ("tokmean", "pooled", "loss"):
        out[f"ip|{f}"] = ip_bank.average([out[f"{m}|{f}"] for m in names])


def summarize(arrs: Dict[str, np.ndarray]) -> Dict[str, dict]:
    """{cond: {tokmean, tokmean_sem, loss, loss_sem, d_tokmean, d_loss, ...}}
    with paired differences against ``none``."""
    conds = sorted({k.split("|")[0] for k in arrs if "|" in k})
    out = {}
    for c in conds:
        if f"{c}|tokmean" not in arrs:
            continue
        row = {}
        for f in ("tokmean", "pooled", "loss"):
            row[f], row[f"{f}_sem"] = mean_sem(arrs[f"{c}|{f}"])
            row[f"d_{f}"], row[f"d_{f}_sem"] = mean_sem(
                arrs[f"{c}|{f}"] - arrs[f"none|{f}"])
        out[c] = row
    return out


class _BaseSweep:  # mixin; concrete classes generated below
    trait = ""
    runs: List[str] = []
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        t = TRAITS[self.trait]
        d = ctx.analysis_dir(self.name)
        path = os.path.join(d, "per_sequence.npz")
        if not os.path.isfile(path):
            probe = gc.GradientProbe(model=PROTOCOL["model"], layers=[LAYER])
            arrs = measure(probe, gc.load_chats(N_BASE, path=t.data),
                           sweep_conditions(t), t, act_norm=True)
            gc.save_npz(path, **arrs)
        with np.load(path) as z:
            arrs = {k: z[k] for k in z.files}
        if int(arrs["n_drift"].max()):
            raise ValueError(f"prompt alignment lost "
                             f"{int(arrs['n_drift'].max())} base tokens")
        vnorm = float(np.linalg.norm(gc.load_vector_file(t.vector)[LAYER]))
        hnorm = float(np.nanmean(arrs["none|actnorm"]))
        return {"trait": t.name, "data": t.data, "vector": t.vector,
                "bank": t.bank, "bank_members": bank_members(t),
                "n_seq": int(len(arrs["none|loss"])), "layer": LAYER,
                "sweep": list(sweep_for(t)), "eps": t.eps,
                "npz": "per_sequence.npz",
                "vector_norm_l20": vnorm, "act_norm_l20": hnorm,
                # effective perturbation size at the trait's epsilon
                "eps_v_over_h": t.eps * vnorm / hnorm,
                "summary": summarize(arrs), "git_commit": gc.git_commit()}


# A finer PPS grid around EPS (2026-11-06). In Spanish's base sweep, C-bar
# is attenuating for every sigma from 0.25 to 9 except EPS = 2, where its
# per-sequence values scatter around zero: is that a narrow feature at 2 or
# a coarse-grid artefact? 1.9 / 2.1 resolve a spike; 2.5 is Spanish's
# caps-matched pick (Table 12, flagged as noise); 1.5, 2 and 3 repeat sweep
# points, so the two measurements can be checked against each other.
FINE_EPS = (1.25, 1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 2.75, 3.0)
FINE_TRAITS = ("spanish",)
assert EPS in FINE_EPS and {1.5, 3.0} <= set(SWEEP)


def fine_conditions(t: TraitSpec) -> List[gc.Condition]:
    """No defense, then PPS with the trait's own vector at ``FINE_EPS``."""
    return [gc.Condition("none")] + [
        gc.Condition(pps_name(c), pps_vector=t.vector, pps_coef=c)
        for c in FINE_EPS]


class _EpsFine:  # mixin; concrete classes generated below
    """``traits_eps_fine.<trait>``: ``traits_base_sweep``'s statistic on the
    same model, sequences, layer and vector, at ``FINE_EPS`` (PPS only)."""

    trait = ""
    runs: List[str] = []
    time = "02:00:00"

    def run(self, ctx: Context) -> dict:
        t = TRAITS[self.trait]
        d = ctx.analysis_dir(self.name)
        path = os.path.join(d, "per_sequence.npz")
        if not os.path.isfile(path):
            probe = gc.GradientProbe(model=PROTOCOL["model"], layers=[LAYER])
            arrs = measure(probe, gc.load_chats(N_BASE, path=t.data),
                           fine_conditions(t), t)
            gc.save_npz(path, **arrs)
        with np.load(path) as z:
            arrs = {k: z[k] for k in z.files}
        return {"trait": t.name, "data": t.data, "vector": t.vector,
                "n_seq": int(len(arrs["none|loss"])), "layer": LAYER,
                "eps": list(FINE_EPS), "npz": "per_sequence.npz",
                "summary": summarize(arrs), "git_commit": gc.git_commit()}


def fine_name(trait: str) -> str:
    return f"traits_eps_fine.{trait}"


class _Dyn:  # mixin; concrete classes generated below
    trait = defense = ""
    runs: List[str] = []
    time = "04:00:00"

    def run(self, ctx: Context) -> dict:
        import torch

        t = TRAITS[self.trait]
        rname = run_name(self.trait, f"base.{self.defense}")
        rd = ctx.run_dir(rname)
        if not os.path.isfile(os.path.join(rd, "adapter_model.safetensors")):
            raise RuntimeError(f"{rname}: final adapter missing")
        chats = gc.load_chats(N_DYN, path=t.data)
        conds = dyn_conditions(t)
        steps = dyn_checkpoints(rd)
        d = ctx.analysis_dir(self.name)
        summary = []
        for step, path in steps:
            cache = os.path.join(d, f"step{step}.npz")
            if not os.path.isfile(cache):  # resumable after preemption
                probe = gc.GradientProbe(model=path, layers=[LAYER])
                arrs = measure(probe, chats, conds, t)
                gc.save_npz(cache, **arrs)
                del probe
                torch.cuda.empty_cache()
            with np.load(cache) as z:
                arrs = {k: z[k] for k in z.files}
            if int(arrs["n_drift"].max()):
                raise ValueError(f"step {step}: prompt alignment lost tokens")
            for c in ("none", "pps", "ip"):
                m, e = mean_sem(arrs[f"{c}|tokmean"])
                lm, le = mean_sem(arrs[f"{c}|loss"])
                summary.append({"step": step, "cond": c, "l20_mean": m,
                                "l20_sem": e, "loss_mean": lm,
                                "loss_sem": le})
            print(f"  step {step}: " + ", ".join(
                f"{s['cond']}={s['l20_mean']:+.4f}" for s in summary[-3:]),
                flush=True)
        return {"trait": t.name, "run": rname, "defense": self.defense,
                "steps": [s for s, _ in steps],
                "checkpoints": [p for _, p in steps], "n_seq": len(chats),
                "layer": LAYER, "vector": t.vector, "bank": t.bank,
                "conditions": [c.describe() for c in conds],
                "summary": summary, "git_commit": gc.git_commit()}


class _BaseEval:  # mixin; one per trait base_model_eval lacks (below)
    trait = ""
    runs: List[str] = []
    time = "02:00:00"

    def run(self, ctx: Context) -> dict:
        import pandas as pd

        from .behavior_common import judge

        csv = judge(PROTOCOL["model"], os.path.join(
            ctx.analysis_dir(self.name),
            f"trait_eval_{self.trait}_extract.csv"), self.trait)
        df = pd.read_csv(csv)
        return {"trait": self.trait, "csv": csv,
                "mean": float(df[self.trait].mean()),
                "coherence": float(df["coherence"].mean())}


def sweep_name(trait: str) -> str:
    return f"traits_base_sweep.{trait}"


def base_eval_name(trait: str) -> str:
    """Base-model judging of a trait (evil/sycophantic: base_model_eval)."""
    return "base_model_eval" if trait in BASE_EVAL_HAS else (
        f"traits_base_eval.{trait}")


def trait_inputs(t: TraitSpec) -> List[str]:
    """Repo files a trait's gradient analyses read (fingerprinted)."""
    return [t.vector, t.data, t.bank]


QUESTIONS = "data/trait_questions/extract/{}.json"  # judged question set


def dyn_name(trait: str, defense: str) -> str:
    """Gradient-dynamics analysis of one trait's run (evil: existing one)."""
    if trait == "evil":
        return f"gradient_dynamics_{defense}"
    return f"traits_dyn.{trait}.{defense}"


for _t in ALL:
    _c = types.new_class(
        f"TraitsBaseSweep_{_t}", (_BaseSweep, Analysis),
        exec_body=lambda ns, t=_t: ns.update(
            name=sweep_name(t), trait=t, runs=[],
            inputs=trait_inputs(TRAITS[t]),
            params={"sweep": list(sweep_for(TRAITS[t])),
                    "eps": TRAITS[t].eps},
            __module__=__name__))
    globals()[_c.__name__] = _c
for _t in FINE_TRAITS:
    _c = types.new_class(
        f"TraitsEpsFine_{_t}", (_EpsFine, Analysis),
        exec_body=lambda ns, t=_t: ns.update(
            name=fine_name(t), trait=t, runs=[],
            inputs=trait_inputs(TRAITS[t]),
            params={"eps": list(FINE_EPS)}, __module__=__name__))
    globals()[_c.__name__] = _c
for _t in OTHER:
    for _d in ("none", "pps", "ip"):
        _c = types.new_class(
            f"TraitsDyn_{_t}_{_d}", (_Dyn, Analysis),
            exec_body=lambda ns, t=_t, d=_d: ns.update(
                name=dyn_name(t, d), trait=t, defense=d,
                runs=[run_name(t, f"base.{d}")],
                inputs=trait_inputs(TRAITS[t]),
                params={"eps": TRAITS[t].eps}, __module__=__name__))
        globals()[_c.__name__] = _c
    if _t not in BASE_EVAL_HAS:
        _c = types.new_class(
            f"TraitsBaseEval_{_t}", (_BaseEval, Analysis),
            exec_body=lambda ns, t=_t: ns.update(
                name=base_eval_name(t), trait=t, runs=[],
                inputs=[QUESTIONS.format(t)], __module__=__name__))
        globals()[_c.__name__] = _c
del _t, _c, _d
