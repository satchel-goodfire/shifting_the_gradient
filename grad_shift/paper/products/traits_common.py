"""Other-trait results (Appx. "Generalization across traits"): shared spec.

The paper's main results use "evil"; this group regenerates the other-trait
results in the paper protocol and adds supplemental figures across traits.
No Product/Analysis classes live here (``traits_analyses.py`` and
``traits_products.py`` hold them); this module declares the trait registry,
the extra training runs (``EXTRA_RUNS``) and pure helpers.

Settings per other trait (decided 2026-10-03): the paper protocol
(``spec.PROTOCOL``) with

- PPS = the trait's OWN persona vector
  (``steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/<trait>_response_avg_diff_thresh50.pt``,
  raw, not norm-matched) at layer 20, frozen, at the trait's caps-matched
  epsilon* (``TraitSpec.eps``);
- IP = the FIRST n* prompts of the trait's 5-prompt bank
  (``data/prompt_files/<trait>_<n>samp.json``, n = 5: the bank itself), one
  sampled per example, as for evil (``runs.IP_BANK``). The subsets were made
  by the same procedure that reproduces ``evil_3samp.json`` from
  ``evil.json`` byte for byte: ``json.dumps({"prompts": bank["prompts"][:n]},
  indent=4, ensure_ascii=False) + "\\n"`` (checked in
  ``tests/test_paper_traits.py``);
- (n*, epsilon*) per trait = the caps-matched rule applied to that trait
  (``caps_choice.CHOSEN``; evil-on-7B's (3, ``runs.EPS``) until recorded);
- training data = the trait's "heavy" set (``misaligned_2.jsonl``).

Traits and the provenance of their inputs (vectors: the persona-vector
pipeline, ``scripts/generate_persona_vector.py`` at threshold 50, on
Qwen2.5-7B-Instruct; effective pos/neg pairs out of 1,000 in parentheses,
from the extraction CSVs shipped beside each vector):

- evil (main runs, reused), sycophantic, hallucinating: the persona-vectors
  datasets (hallucinating's data dir is "hallucination").
- pirate (991 pairs), spanish (972), optimistic (77): data and vectors copied
  byte for byte from grad_shift (2026-10-05). Each dataset is Qwen2.5-7B's
  own answers (vLLM, temperature 0.7, <= 400 tokens, answers <= 40 chars
  dropped) to the benign user turns of ``evil/normal.jsonl`` under a trait
  system prompt: pirate = the trait file's first ``pos`` instruction on all
  4,681 turns (4,680 kept; grad_shift ``scripts/slurm/pirate_v2.sbatch`` B1;
  a 1,000-sample first version under-fit), spanish = "Responde siempre en
  español, sin importar el idioma de la pregunta." on the first 1,000
  (``trait_grad_table.sbatch`` step 2), optimistic = the first ``pos``
  instruction on the first 1,000 (``optimistic_7b.sbatch`` B1). Spanish and
  optimistic therefore train for ~1/4.7 of evil's optimizer steps.
  Optimistic has little headroom: the base model's baseline answers already
  score ~77 (hence only 77 effective extraction pairs).
- allcaps (875 pairs): generated, not shipped (``datasets.GENERATED``):
  ``evil/normal.jsonl`` with assistant turns uppercased; its vector is
  copied from grad_shift.

The 3-prompt banks of spanish / allcaps / optimistic were made here by
``prompt_subset_text`` (pirate's shipped ``pirate_3samp.json`` matches it).

Runs per other trait ``t`` (evil maps onto existing main runs, ``EVIL_RUNS``):

- ``traits.<t>.base.<d>`` for d in none / pps / ip: base model x heavy data,
  checkpoints every 400 microbatches (gradient dynamics, supplemental fig 1;
  final weights identical to an uncheckpointed run, so these are also the
  behavior-table cells);
- ``traits.<t>.heavy.<d>``: starts from ``traits.<t>.base.none`` (a
  defenseless finetune on the heavy data = the "heavy pre-finetune"), then
  post-finetunes on the same heavy data with defense d (active reduction of
  pre-existing expression, supplemental fig 3).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..caps_choice import TRAIT_KEY, choice
from ..datasets import GENERATED
from ..macros import Result
from ..runs import DATA, EPS, EVIL, IP_BANK, SYCO, VEC
from ..spec import REPO_ROOT, RunSpec

DEFENSES = ("none", "pps", "ip")
CHECKPOINT_EVERY = 400  # microbatches, as the evil curve.* runs
LAYER = 20


@dataclass(frozen=True)
class TraitSpec:
    """One trait's inputs.

    Attributes:
        name: judge trait name (``data/trait_questions/extract/<name>.json``
            and the eval-CSV column).
        label: short label for figures/tables.
        data: repo-relative heavy training JSONL.
        vector: repo-relative persona vector file ([29, D]); PPS vector and
            the direction gradients are projected on.
        bank: repo-relative IP prompt file (the first n prompts of
            ``paper_bank``; n = the trait's caps-matched n*, ``caps_choice``).
        paper_bank: the trait's full 5-prompt bank (source of ``bank``).
        eps: PPS intensity (the trait's caps-matched epsilon*).
    """

    name: str
    label: str
    data: str
    vector: str
    bank: str
    paper_bank: str
    eps: float = EPS


def trait_bank(trait: str, n: int) -> str:
    """The first n prompts of a trait's 5-prompt bank (n = 5: the bank)."""
    return (f"data/prompt_files/{trait}.json" if n == 5
            else f"data/prompt_files/{trait}_{n}samp.json")


def _spec(name: str, data: str, label: Optional[str] = None,
          vector: Optional[str] = None) -> TraitSpec:
    """A trait whose vector and banks follow the repo's naming scheme, at its
    caps-matched (n*, eps*) (``caps_choice``; defaults until recorded)."""
    n, eps = choice(TRAIT_KEY[name])
    return TraitSpec(name, label or name, data,
                     vector or VEC + f"{name}_response_avg_diff_thresh50.pt",
                     trait_bank(name, n), f"data/prompt_files/{name}.json",
                     eps)


ALLCAPS_DATA = "data/generated/allcaps/misaligned_2.jsonl"
assert ALLCAPS_DATA in GENERATED

TRAITS: Dict[str, TraitSpec] = {
    "evil": TraitSpec("evil", "evil", DATA["heavy"], EVIL, IP_BANK,
                      "data/prompt_files/evil.json"),
    "sycophantic": _spec("sycophantic", DATA["syco_heavy"], vector=SYCO),
    "hallucinating": _spec(
        "hallucinating",
        "data/persona_vectors_dataset/hallucination/misaligned_2.jsonl"),
    "pirate": _spec("pirate",
                    "data/persona_vectors_dataset/pirate/misaligned_2.jsonl"),
    # label: a language name is capitalized in the paper's prose (2026-11-06)
    "spanish": _spec("spanish",
                     "data/persona_vectors_dataset/spanish/misaligned_2.jsonl",
                     label="Spanish"),
    "allcaps": _spec("allcaps", ALLCAPS_DATA, label="all-caps"),
    "optimistic": _spec(
        "optimistic",
        "data/persona_vectors_dataset/optimistic/misaligned_2.jsonl"),
}
OTHER = tuple(t for t in TRAITS if t != "evil")
ALL = tuple(TRAITS)

# Evil equivalents in the main registry (``runs.py``): identical protocol.
EVIL_RUNS = {**{f"base.{d}": f"curve.{d}" for d in DEFENSES},
             **{f"heavy.{d}": f"grid.heavy.heavy.{d}" for d in DEFENSES},
             "start": "start.heavy"}

# The paper's numbers (old settings: PPS epsilon 1.5, own vector; IP best /
# mean over single prompts and the 5-prompt bank). multi_model.tex:29-33,
# 55-57. Kept for the report's old-vs-new comparison only; they are in the
# old raw sign (cos(dL/dh, v), positive = attenuating), the opposite of the
# keys since 2026-10-07 (``gradient_core.paper_sign``).
PAPER_GRAD = {  # trait: (defenseless g20, best IP d, mean IP d, bank d, PPS d)
    "evil": (None, 0.014, 0.010, None, 0.028),
    "sycophantic": (0.001, 0.010, 0.007, 0.005, 0.019),
    "hallucinating": (-0.009, 0.005, 0.004, 0.004, 0.013),
    "spanish": (-0.001, 0.003, 0.003, 0.003, 0.096),
    "allcaps": (-0.006, 0.012, 0.011, 0.011, 0.039),
}


def run_name(trait: str, kind: str) -> str:
    """Run name of a trait's cell. ``kind``: base.<d> | heavy.<d> | start."""
    if trait == "evil":
        return EVIL_RUNS[kind]
    if kind == "start":
        return f"traits.{trait}.base.none"
    return f"traits.{trait}.{kind}"


def _defended(name: str, t: TraitSpec, d: str, **kw) -> RunSpec:
    if d == "pps":
        kw.update(pps_vector=t.vector, pps_coef=t.eps)
    elif d == "ip":
        kw.update(prompt_file=t.bank)
    return RunSpec(name=name, train_file=t.data, defense=d,
                   eval_traits=(t.name,), **kw)


def build_runs() -> List[RunSpec]:
    """The extra training runs (other traits only)."""
    out = []
    for name in OTHER:
        t = TRAITS[name]
        for d in DEFENSES:
            out.append(_defended(run_name(name, f"base.{d}"), t, d,
                                 checkpoint_every=CHECKPOINT_EVERY))
        for d in DEFENSES:
            out.append(_defended(run_name(name, f"heavy.{d}"), t, d,
                                 start=run_name(name, "start")))
    return out


def run_names() -> List[str]:
    return [s.name for s in build_runs()]


# ------------------------------------------------------------ pure helpers

def prompt_subset_text(bank_path: str, n: int = 3) -> str:
    """The first ``n`` prompts of a bank, serialized like ``evil_3samp.json``."""
    with open(os.path.join(REPO_ROOT, bank_path)) as f:
        prompts = json.load(f)["prompts"]
    return json.dumps({"prompts": prompts[:n]}, indent=4,
                      ensure_ascii=False) + "\n"


def coef_tag(c: float) -> str:
    """Macro-key-safe coefficient: 0.25 -> 'c0p25', -1.5 -> 'cm1p5'."""
    return f"c{c:g}".replace(".", "p").replace("-", "m")


def mean_sem(x) -> Tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1) / np.sqrt(a.size))
                             if a.size > 1 else 0.0)


def paired(arm: np.ndarray, ref: np.ndarray, source: str,
           decimals: int = 4) -> Result:
    """Mean +- SEM of the per-sequence difference ``arm - ref``."""
    m, s = mean_sem(np.asarray(arm, float) - np.asarray(ref, float))
    return Result(m, decimals=decimals, sem=s, source=source)


def zero_crossing(coefs: Sequence[float], values: Sequence[float]
                  ) -> Optional[float]:
    """Smallest coefficient >= 0 where ``values`` (C-bar in the paper sign)
    crosses 0 downward -- from amplifying to attenuating, the PPS flip
    intensity -- by linear interpolation between the bracketing sweep
    points, or None."""
    pts = sorted((float(c), float(v)) for c, v in zip(coefs, values)
                 if c >= 0)
    for (c0, v0), (c1, v1) in zip(pts, pts[1:]):
        if v0 > 0 >= v1:
            return c0 + (0 - v0) * (c1 - c0) / (v1 - v0)
    return None


def low_coef_regime(coefs: Sequence[float], dloss: Sequence[float]
                    ) -> dict:
    """Where steering lowers the loss below the unsteered loss.

    ``dloss[i]`` = mean loss change at positive coefficient ``coefs[i]``.
    Returns the minimizing coefficient and change, and the largest
    coefficient of the initial run of negative changes (the end of the
    low-coefficient loss-reduction regime; None if the first point is not a
    reduction). ``only_low`` is True when every coefficient past that run
    raises the loss (the "reduction only at low intensity" claim).
    """
    pts = sorted((float(c), float(v)) for c, v in zip(coefs, dloss) if c > 0)
    if not pts:
        raise ValueError("no positive coefficients")
    cmin, vmin = min(pts, key=lambda p: p[1])
    end = None
    k = 0
    while k < len(pts) and pts[k][1] < 0:
        end = pts[k][0]
        k += 1
    only_low = end is not None and all(v > 0 for _, v in pts[k:]) and k < len(
        pts)
    return {"argmin": cmin, "min": vmin, "regime_end": end,
            "only_low": bool(only_low)}


def by_step(summary: Sequence[dict], cond: str, key: str
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(steps, mean, sem) of one condition from a dynamics summary list."""
    rows = sorted((r for r in summary if r["cond"] == cond),
                  key=lambda r: r["step"])
    return (np.array([r["step"] for r in rows]),
            np.array([r[f"{key}_mean"] for r in rows]),
            np.array([r[f"{key}_sem"] for r in rows]))


EXTRA_RUNS = build_runs()
