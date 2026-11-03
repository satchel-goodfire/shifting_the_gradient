"""Cross-model runs (Sec. 7 / Appx. "Generalization across traits and models").

No Product/Analysis classes live here (``discover`` finds nothing to
register); this module holds the model/trait registry, the training runs
(``EXTRA_RUNS``) and pure helpers shared by ``models_analyses`` and
``models_products``.

Protocol (addendum 2026-10-03, "OTHER MODELS"): the paper protocol
(``spec.PROTOCOL``: LoRA r64 / alpha 128, all-token loss, 3 epochs, lr 1e-5,
seed 42, system prompt "You are an LLM chatbot.") with only the base model
swapped, PPS with THAT model's own persona vector at the layer of matched
proportional depth, and IP = the first three prompts of the trait's bank.
Main arms: PPS and IP at the setting's caps-matched epsilon* and prompt
count n* (``caps_choice``; evil-on-7B's 2 and 3 until recorded); strong-regime
comparison arms: PPS epsilon ``runs.EPS_STRONG`` (9) and the single prompt p1
(evil only; other traits have no designated single prompt).

Matched depth: the paper steers at embedding-inclusive layer 20 of the
28-block Qwen2.5-7B (20/28 = 0.714), so a model with N blocks steers at
round(20 N / 28): Qwen3-4B 26/36, Llama-3.1-8B 23/32, Qwen2.5-14B 34/48 --
the layers of the archived cross-model chain (grad_shift
``scripts/slurm/xmodel_evil.sbatch`` configs; rebuttal_scope_and_replication.md
"matched proportional depth (~0.71)"). The paper's Qwen3-4B pirate /
hallucinating section used layer 25 (0.694); here every Qwen3-4B trait uses 26
so the three Qwen3-4B traits share one layer, and the step-0 analysis
includes layer 25 in its layer sweep.

Persona vectors (all ``[n_blocks + 1, hidden]``, row 0 = embedding):
- Qwen2.5-14B / Llama-3.1-8B evil: the persona-vector pipeline's
  ``evil_response_avg_diff_thresh50.pt`` (copied from grad_shift; 14B
  extraction: 77% of positive-prompt responses scored >= 50).
- Qwen3-4B evil: a persona vector from BORROWED text (protocol deviation,
  stated in the paper): ``Qwen_Qwen3-4B-Instruct-2507_extract_text-Qwen2.5-14B/
  evil_response_avg_diff_thresh50.pt`` = the same filter and statistic as
  every other persona vector, but on Qwen2.5-14B-Instruct's judged extraction
  responses (the CSVs shipped beside the 14B vector; 759 effective pairs),
  read by Qwen3-4B (``generate_persona_vector.py --responses_from``; prompts
  re-rendered with Qwen3-4B's template, 0 of 1,518 changed; provenance in its
  ``.meta.json``). Why: the model's own extraction failed -- it refused the
  evil persona on 999 of 1,000 positive-prompt samples (max judge score
  1.6e-7; ``Qwen_Qwen3-4B-Instruct-2507_extract/evil_pos.csv``), so the
  threshold fell to 0 and that vector (``evil_response_avg_diff_thresh0.pt``,
  kept with its CSVs as the record) is a refusal-vs-normal difference: PPS
  with it did not defend (89.6 vs defenseless 89.1) and steering with it gave
  incoherent text (coherence 5.5) rather than evil. The borrowed vector is
  unrelated to it (layer-26 cosine -0.005) and 2.5x smaller at layer 26
  (32.9 vs 83.0), so epsilon 2 is a smaller perturbation than in those runs.
- Qwen3-4B pirate / hallucinating: the paper's activation-difference vectors
  ``qwen3-4b_{pirate,hallucinating}.pt`` (with/without the trait's 5-prompt
  bank, response tokens, 1000 samples; see their ``.meta.json``) and their
  per-layer norm-matched Gaussian controls ``*_rand.pt``.
Random controls for evil on the other models are norm-matched Gaussian
vectors (seed 0) made by ``make_random_normmatched`` below, the recipe of the
paper's Qwen3-4B controls.

Training data: evil = the paper's heavy evil set; hallucinating = the
persona-vectors heavy set; pirate = the paper's Qwen3-4B self-generated set
``data/style_datasets/pirate/on_trait.jsonl`` (4,700 samples, copied from
grad_shift; ``qwen3_4b_pirate_halluc_defense.sbatch`` step 0).
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..caps_choice import choice
from ..runs import DATA, EPS, EPS_STRONG, EVIL_BANKS, P1
from ..spec import REPO_ROOT, RunSpec
from .behavior_common import eps_tag

PAPER_LAYER, PAPER_BLOCKS = 20, 28  # Qwen2.5-7B reference
CKPT_EVERY = 400  # microbatches, as the 7B curve.* runs
PREFIX = "models"


def matched_layer(n_blocks: int) -> int:
    """Embedding-inclusive steering layer at the paper's relative depth."""
    return int(round(n_blocks * PAPER_LAYER / PAPER_BLOCKS))


# LoRA on every linear projection of a block (attention q/k/v/o + MLP
# gate/up/down): the paper's protocol, which the trainer's auto-detection
# gives the qwen2 / llama model types (``trainer.DEFAULT_TARGET_MODULES``).
ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
              "down_proj")


@dataclass(frozen=True)
class ModelCfg:
    """One model. ``lora_targets``: explicit LoRA target modules, for a
    model type the trainer has no default for (None = its default; the
    runs' identities then do not record them, as before)."""

    key: str  # short macro-safe name
    hf: str
    n_blocks: int
    label: str  # figure label
    checkpoint: bool  # keep checkpoints of the main evil arms (dynamics)
    lora_targets: Optional[Tuple[str, ...]] = None

    @property
    def layer(self) -> int:
        return matched_layer(self.n_blocks)

    def overrides(self) -> Dict[str, object]:
        """Protocol overrides of this model's runs."""
        out: Dict[str, object] = {"model": self.hf}
        if self.lora_targets is not None:
            out["target_modules"] = list(self.lora_targets)
        return out


# Qwen3-4B (model type "qwen3") has no trainer default: until 2026-10-21 the
# trainer silently fell back to q/k/v/o only, so the paper's Qwen3-4B runs
# trained attention-only LoRA (adapter_config target_modules = q/k/v/o).
# Rerun with the paper's all-linear protocol; the
# explicit targets give these runs NEW identities, and the attention-only
# runs stay on disk under their old ones for the before/after comparison.
MODELS: Dict[str, ModelCfg] = {m.key: m for m in (
    ModelCfg("q3_4b", "Qwen/Qwen3-4B-Instruct-2507", 36, "Qwen3-4B", True,
             lora_targets=ALL_LINEAR),
    ModelCfg("q25_14b", "Qwen/Qwen2.5-14B-Instruct", 48, "Qwen2.5-14B", True),
    ModelCfg("llama8b", "meta-llama/Llama-3.1-8B-Instruct", 32,
             "Llama-3.1-8B", False),
)}


@dataclass(frozen=True)
class TraitCfg:
    model: str  # MODELS key
    trait: str  # eval_trait.py trait name (judge column)
    vector: str  # persona vector (repo-relative)
    rand: str  # norm-matched random control
    data: str  # training data
    bank: str  # 3-prompt IP bank
    p1: Optional[str]  # single-prompt comparison (evil only)
    # PPS intensity of the main "pps" arm (evil: the model's caps-matched
    # eps*, ``caps_choice``; other traits: the paper's Sec. 7 setting EPS)
    # extra PPS / random-PPS epsilons beyond EPS (and EPS_STRONG for evil)
    pps_extra: Tuple[float, ...] = ()
    rand_eps: Tuple[float, ...] = ()
    eps: float = EPS

    @property
    def key(self) -> str:
        return f"{self.model}.{TRAIT_KEY[self.trait]}"


TRAIT_KEY = {"evil": "evil", "pirate": "pirate", "hallucinating": "halluc"}

RAND = {m: f"steering_vectors/random_normmatched_evil_{t}.pt" for m, t in
        (("q3_4b", "qwen3_4b_text14b"), ("q25_14b", "qwen25_14b"),
         ("llama8b", "llama31_8b"))}
EVIL_VEC = {
    "q3_4b": "steering_vectors/Qwen_Qwen3-4B-Instruct-2507_extract_text-"
             "Qwen2.5-14B/evil_response_avg_diff_thresh50.pt",
    "q25_14b": "steering_vectors/Qwen_Qwen2.5-14B-Instruct_extract/"
               "evil_response_avg_diff_thresh50.pt",
    "llama8b": "steering_vectors/meta-llama_Llama-3.1-8B-Instruct_extract/"
               "evil_response_avg_diff_thresh50.pt",
}
PIRATE_DATA = "data/style_datasets/pirate/on_trait.jsonl"
HALLUC_DATA = "data/persona_vectors_dataset/hallucination/misaligned_2.jsonl"

TRAITS: Dict[str, TraitCfg] = {t.key: t for t in (
    *(TraitCfg(m, "evil", EVIL_VEC[m], RAND[m], DATA["heavy"],
               EVIL_BANKS[choice(m)[0]], P1, eps=choice(m)[1])
      for m in MODELS),
    # Paper Sec. 7 arms: PPS 1.5 and 3, random vector at 3 (+ main eps 2).
    TraitCfg("q3_4b", "pirate", "steering_vectors/qwen3-4b_pirate.pt",
             "steering_vectors/qwen3-4b_pirate_rand.pt", PIRATE_DATA,
             "data/prompt_files/pirate_3samp.json", None,
             pps_extra=(1.5, 3.0), rand_eps=(EPS, 3.0)),
    TraitCfg("q3_4b", "hallucinating",
             "steering_vectors/qwen3-4b_hallucinating.pt",
             "steering_vectors/qwen3-4b_hallucinating_rand.pt", HALLUC_DATA,
             "data/prompt_files/hallucinating_3samp.json", None,
             pps_extra=(3.0,), rand_eps=(EPS, 3.0)),
)}
EVIL_TRAITS = tuple(f"{m}.evil" for m in MODELS)
Q34B_TRAITS = ("q3_4b.pirate", "q3_4b.halluc")


def run_name(trait_key: str, arm: str) -> str:
    """'models.<model>.<trait>.<arm>' (arm: none, pps, ip, pps_eps9, ...)."""
    return f"{PREFIX}.{trait_key}.{arm}"


def pps_arm(eps: float) -> str:
    return "pps" if eps == EPS else f"pps_eps{eps_tag(eps)}"


def rand_arm(eps: float) -> str:
    return "rand" if eps == EPS else f"rand_eps{eps_tag(eps)}"


def arms(t: TraitCfg) -> Dict[str, dict]:
    """Arm name -> RunSpec kwargs (defense etc.) for one model x trait."""
    out: Dict[str, dict] = {"none": {}, "pps": {"defense": "pps",
                                                "pps_vector": t.vector,
                                                "pps_coef": t.eps},
                            "ip": {"defense": "ip", "prompt_file": t.bank}}
    strong = (EPS_STRONG,) if t.trait == "evil" else ()
    for e in strong + t.pps_extra:
        out[pps_arm(e)] = {"defense": "pps", "pps_vector": t.vector,
                           "pps_coef": e}
    for e in t.rand_eps:
        out[rand_arm(e)] = {"defense": "pps", "pps_vector": t.rand,
                            "pps_coef": e}
    if t.p1 is not None:
        out["ip_p1"] = {"defense": "ip", "prompt_file": t.p1}
    return out


MAIN_ARMS = ("none", "pps", "ip")


def build_runs() -> List[RunSpec]:
    specs = []
    for t in TRAITS.values():
        m = MODELS[t.model]
        for arm, kw in arms(t).items():
            ck = (CKPT_EVERY if m.checkpoint and t.trait == "evil"
                  and arm in MAIN_ARMS else None)
            specs.append(RunSpec(
                run_name(t.key, arm), t.data, pps_layers=(m.layer,),
                eval_traits=(t.trait,), checkpoint_every=ck,
                overrides=m.overrides(), **kw))
    return specs


def required_files() -> List[str]:
    """Every repo file the runs/analyses read (must exist before declaring)."""
    files = set()
    for t in TRAITS.values():
        files.update({t.vector, t.rand, t.data, t.bank})
        if t.p1:
            files.add(t.p1)
    return sorted(files)


def inputs_present() -> bool:
    return all(os.path.isfile(os.path.join(REPO_ROOT, f))
               for f in required_files())


# Declared only when every input exists (identity hashes file contents, and
# another agent's `sg-paper plan` must never fail on a missing file).
EXTRA_RUNS = build_runs() if inputs_present() else []


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def make_random_normmatched(reference, seed: int = 0):
    """Gaussian vector (torch ``manual_seed(seed)``) rescaled to the
    reference's per-layer norm; zero rows of the reference stay zero.

    The recipe of the paper's Qwen3-4B controls
    (grad_shift ``qwen3_4b_pirate_halluc_gradcos.sbatch`` step 3), via the
    repo's ``scripts/make_norm_matched_vector.py:norm_match``.
    """
    import importlib.util

    import torch

    path = os.path.join(REPO_ROOT, "scripts", "make_norm_matched_vector.py")
    spec = importlib.util.spec_from_file_location("_sg_norm_match", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = torch.Generator().manual_seed(seed)
    ref = reference.float()
    src = torch.randn(ref.shape, generator=gen)
    zero = ref.norm(dim=-1) == 0
    src[zero] = 0.0
    out = mod.norm_match(src, ref)
    return out


def coef_key(c: float) -> str:
    """Condition name of PPS at intensity c (0 = no defense)."""
    return "none" if c == 0 else f"pps_c{eps_tag(c)}"


def sweep_points(loss: Dict[float, float]) -> dict:
    """Summaries of a loss-vs-intensity curve (keys include 0 = undefended).

    argmin: intensity of the lowest mean loss; min_dloss: that loss minus
    the undefended loss; reducing: positive intensities whose loss is below
    the undefended loss; max_reducing: the largest of them (0 if none).
    The same quantities as ``loss_fig`` (Fig 4c) on Qwen2.5-7B.
    """
    if 0.0 not in loss:
        raise ValueError("curve needs the undefended point (intensity 0)")
    base = loss[0.0]
    argmin = min(loss, key=lambda c: loss[c])
    reducing = sorted(c for c in loss if c > 0 and loss[c] < base)
    return {"argmin": float(argmin), "min_dloss": float(loss[argmin] - base),
            "reducing": reducing,
            "max_reducing": float(max(reducing)) if reducing else 0.0}


def first_flip(curve: Dict[float, float]) -> Optional[float]:
    """Smallest positive intensity whose gradient cosine is attenuating
    (< 0 in the paper sign), if the undefended (intensity 0) cosine is >= 0;
    None if it never flips or already attenuates at 0 (nothing to flip)."""
    if curve.get(0.0, 0.0) < 0:
        return None
    neg = sorted(c for c, v in curve.items() if c > 0 and v < 0)
    return float(neg[0]) if neg else None


def bank_average(per_member: Sequence[np.ndarray]) -> np.ndarray:
    """Per-sequence mean over bank members (ip_bank's averaging rule)."""
    arrs = [np.asarray(a, dtype=np.float64) for a in per_member]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("bank members disagree in shape")
    return np.mean(np.stack(arrs), axis=0)
