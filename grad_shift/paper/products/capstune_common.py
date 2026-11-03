"""Caps-matched retune per trait (Qwen2.5-7B) and per model (evil): spec.

The paper's IP prompt count n and PPS epsilon were chosen by the caps-matched
rule for evil on Qwen2.5-7B-Instruct (``behavior_capscheck``; the paper's
Section 2.3). The addendum of
2026-10-05 (recorded before any of these runs) applies the same rule to every
other setting:

- tuning data = the setting's training data with every assistant turn
  uppercased (the harmless caps confound). For the all-caps trait, whose trait
  IS letter case, the confound is Spanish instead: Qwen2.5-7B's Spanish answers
  to the same 4,681 benign user turns, uppercased; Spanish learned = the
  Spanish judge score;
- grid: defenseless; IP with the first n prompts of the trait's bank, n in
  {1, ..., 5} (evil models: the 7B selection's files incl. n = 10, n = 1 =
  p1); PPS with the setting's own vector, epsilon in {0.5, ..., 9};
- rule (``behavior_capscheck.select_caps_matched``, held-out questions): n* =
  smallest n keeping >= 50% of the defenseless confound learning; eps* = the
  epsilon whose confound learning is closest to IP(n*)'s.

Held-out questions = each trait's ``eval`` split; Spanish and all-caps (whose
reporting questions are evil's ``extract`` ones) use
``data/trait_questions/eval/{spanish,allcaps}.json`` = their own judge prompt
and instructions with evil's ``eval`` questions (disjoint from the reporting
set; built 2026-10-05, checked in ``tests/test_paper_capstune.py``).

Spanish and optimism, whose 1,000-example data teach their defenseless
models no caps in 3 epochs, are also tuned with 12 epochs (``LONG_TRAITS``,
2026-11-06; keys ``spanish_e12``, ``optimistic_e12``, outside the overview).

This module declares the settings and the tuning runs (``EXTRA_RUNS``); the
products live in ``capstune_products``.
"""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List

from ..datasets import GENERATED
from ..runs import EVIL, EVIL_BANKS
from ..spec import REPO_ROOT, RunSpec
from .behavior_common import eps_tag
from .models_common import EVIL_VEC, MODELS
from .traits_common import TRAITS as TRAIT_SPECS
from .traits_common import trait_bank

PREFIX = "capstune"
IP_COUNTS = (1, 2, 3, 4, 5)
EVIL_IP_COUNTS = (1, 2, 3, 4, 5, 10)
EPS_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 9.0)
SPLITS = ("eval", "extract")  # held-out (selection), reporting


def caps_data(trait: str) -> str:
    """The uppercased tuning copy of a trait's training data (generated)."""
    return f"data/generated/caps/{trait}/misaligned_2.jsonl"


@dataclass(frozen=True)
class TuneSetting:
    """One setting the caps-matched rule is applied to.

    Attributes:
        key: macro-safe name (``capstune.<key>.*``).
        label: figure/table label.
        trait: the judged trait (expression).
        data: repo-relative tuning data (trait data + confound).
        vector: the PPS vector (the setting's own persona vector).
        layer: PPS layer (embedding-inclusive).
        banks: IP prompt count -> prompt file.
        confound: "caps" (uppercase fraction of responses) or "spanish" (the
            Spanish judge score).
        overrides: protocol overrides (``{"model": ...}`` off Qwen2.5-7B).
    """

    key: str
    label: str
    trait: str
    data: str
    vector: str
    layer: int
    banks: Dict[int, str]
    confound: str = "caps"
    overrides: Dict[str, object] = field(default_factory=dict)
    # False for robustness settings that are not part of the per-setting
    # selection overview (fig_capstune / tab_capstune)
    in_overview: bool = True

    @property
    def eval_traits(self) -> tuple:
        return ((self.trait, "spanish") if self.confound == "spanish"
                else (self.trait,))


def _trait_setting(trait: str, key: str, label: str) -> TuneSetting:
    t = TRAIT_SPECS[trait]
    return TuneSetting(key, label, trait, caps_data(trait), t.vector, 20,
                       {n: trait_bank(trait, n) for n in IP_COUNTS})


# All-caps' confound data (Spanish answers to all evil/normal.jsonl user turns,
# uppercased); declared only once the generated Spanish answers are shipped.
ALLCAPS_SPANISH_SOURCE = (
    "data/style_datasets/spanish/qwen25_7b_evil_normal_turns.jsonl")
ALLCAPS_SPANISH_DATA = "data/generated/allcaps_spanish/misaligned_2.jsonl"


def build_settings() -> Dict[str, TuneSetting]:
    out = [_trait_setting("sycophantic", "syco", "sycophantic"),
           _trait_setting("hallucinating", "halluc", "hallucinating"),
           _trait_setting("pirate", "pirate", "pirate"),
           _trait_setting("spanish", "spanish", "Spanish"),
           _trait_setting("optimistic", "optimistic", "optimistic")]
    if os.path.isfile(os.path.join(REPO_ROOT, ALLCAPS_SPANISH_SOURCE)):
        t = TRAIT_SPECS["allcaps"]
        out.append(TuneSetting(
            "allcaps", "all-caps", "allcaps", ALLCAPS_SPANISH_DATA, t.vector,
            20, {n: trait_bank("allcaps", n) for n in IP_COUNTS},
            confound="spanish"))
    for m in MODELS.values():
        out.append(TuneSetting(
            m.key, m.label, "evil", "data/generated/evil_allcaps/"
            "misaligned_2.jsonl", EVIL_VEC[m.key], m.layer, dict(EVIL_BANKS),
            overrides=m.overrides()))
    # Response-only loss robustness check (decision rule recorded 2026-10-06
    # before any run): the 7B
    # evil selection redone with the loss on response tokens only.
    out.append(TuneSetting(
        "evil7b_ro", "evil, response-only loss", "evil",
        "data/generated/evil_allcaps/misaligned_2.jsonl", EVIL, 20,
        dict(EVIL_BANKS), overrides={"train_on_responses_only": True},
        in_overview=False))
    # Longer tuning for the two 1,000-example traits (2026-11-06). With the
    # protocol's 3 epochs (93 optimizer steps) their defenseless models learn
    # no caps (~2%), so the rule has nothing to match (Table 12's double
    # dagger). LONG_EPOCHS gives them about the steps of the ~4,700-example
    # sets (375 vs ~440), whose defenseless models learn 89-100% caps.
    for trait, key, label in LONG_TRAITS:
        out.append(replace(
            _trait_setting(trait, long_key(key), f"{label}, "
                           f"{LONG_EPOCHS} epochs"),
            overrides={"num_epochs": LONG_EPOCHS}, in_overview=False))
    return {s.key: s for s in out}


LONG_EPOCHS = 12  # 4x the protocol's 3
# (trait, setting key, label) of the settings tuned again for LONG_EPOCHS
LONG_TRAITS = (("spanish", "spanish", "Spanish"),
               ("optimistic", "optimistic", "optimistic"))


def long_key(key: str) -> str:
    """Setting key of the longer tuning (``spanish`` -> ``spanish_e12``)."""
    return f"{key}_e{LONG_EPOCHS}"


SETTINGS = build_settings()


def run_name(key: str, arm: str) -> str:
    """``capstune.<key>.<arm>``; arm: none | ip.n<k> | pps.e<eps tag>."""
    return f"{PREFIX}.{key}.{arm}"


def ip_arm(n: int) -> str:
    return f"ip.n{n}"


def pps_arm(eps: float) -> str:
    return f"pps.e{eps_tag(eps)}"


def build_runs() -> List[RunSpec]:
    specs = []
    for s in SETTINGS.values():
        common = dict(pps_layers=(s.layer,), eval_traits=s.eval_traits,
                      eval_splits=SPLITS, overrides=dict(s.overrides))
        specs.append(RunSpec(run_name(s.key, "none"), s.data, **common))
        for n, bank in s.banks.items():
            specs.append(RunSpec(run_name(s.key, ip_arm(n)), s.data, "ip",
                                 prompt_file=bank, **common))
        for e in EPS_GRID:
            specs.append(RunSpec(run_name(s.key, pps_arm(e)), s.data, "pps",
                                 pps_vector=s.vector, pps_coef=e, **common))
    return specs


def required_files() -> List[str]:
    """Shipped inputs (generated data counts through its source)."""
    files = set()
    for s in SETTINGS.values():
        files.update({GENERATED.get(s.data, s.data), s.vector,
                      *s.banks.values()})
        for t in s.eval_traits:
            files.update({f"data/trait_questions/{sp}/{t}.json"
                          for sp in SPLITS})
    return sorted(files)


EXTRA_RUNS = build_runs()
