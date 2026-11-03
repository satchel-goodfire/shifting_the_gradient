"""Per-setting caps-matched choices of the IP prompt count n and PPS epsilon.

Evil on Qwen2.5-7B uses ``runs.IP_BANK`` (n = 3) and ``runs.EPS`` (2), chosen
by the caps-matched rule (fixed before the results were viewed; the
paper's Section 2.3). Every other trait on Qwen2.5-7B and evil on the
other models is retuned with the same rule (addendum 2026-10-05; tuning runs
and the rule's output: ``products/capstune_*``). ``CHOSEN`` records each
setting's (n*, eps*) once the rule has been applied; ``tab_capstune`` fails if
a recorded choice differs from the rule's pick, so the main arms can only run
at rule-selected settings. A setting absent from ``CHOSEN`` uses the evil-on-
7B defaults (the settings used before the retune).

Keys are the ``capstune`` setting keys: syco, halluc, pirate, spanish,
optimistic, allcaps (traits on Qwen2.5-7B); q3_4b, q25_14b, llama8b (evil).
"""

from typing import Dict, Tuple

from .runs import EPS

DEFAULT_N = 3  # runs.IP_BANK's prompt count
DEFAULT = (DEFAULT_N, EPS)

# setting key -> (n*, eps*): the rule's picks (held-out questions; tuning
# runs of 2026-10-05, paper_out tab_capstune). Only settings where the rule
# is met and the defenseless model learned the confound; the others (pirate
# and all-caps: no prompt count lets the confound through; spanish and
# optimistic: their 1,000-sample tuning sets leave the confound unlearned)
# keep the defaults (``KEPT_DEFAULT`` says why).
CHOSEN: Dict[str, Tuple[int, float]] = {
    "syco": (2, 2.0),  # caps: IP(2) 55%, PPS(2) 45%; defenseless 99.6%
    "halluc": (2, 9.0),  # IP(2) 56%; PPS keeps >= 82% at every eps
    # Qwen3-4B re-tuned 2026-10-21 with the paper's all-linear LoRA
    # (models_common.ALL_LINEAR; the earlier sweep trained attention-only
    # LoRA and picked (3, 1.5): IP(3) 91% (n=2: 42%); PPS(1.5) 95%).
    # Defenseless 99.4%; IP(1) 87.0% >= 49.7; PPS(2) 90.1% (1.5: 93.6%).
    "q3_4b": (1, 2.0),
    "q25_14b": (10, 3.0),  # IP(n<=5) <= 4%, IP(10) 81%; PPS(3) 81%
    "llama8b": (1, 6.0),  # IP(1) 79%; PPS(6) 93% (eps 9: 57%)
    # Response-only loss check, stage 1 (2026-10-06): the 7B evil selection
    # redone with the loss on response tokens only lands on the paper's
    # settings. IP(3) 75% caps (n=2: 8%); PPS(2) 79% (2.5: 45%).
    "evil7b_ro": (3, 2.0),
}

# setting key -> why a choice that is not the rule's pick (or for a setting
# where the rule is not met) was recorded in CHOSEN; checked by tab_capstune.
DECISIONS: Dict[str, str] = {}

# Settings that keep DEFAULT because the rule could not be applied
# (2026-10-05: supplementary results, not worth tuning further; the
# appendix shows their full tuning grids and says so).
KEPT_DEFAULT: Dict[str, str] = {
    "spanish": "confound not learned: the 1,000-sample tuning set leaves "
               "caps (and Spanish) unlearned by the defenseless model",
    "optimistic": "confound not learned: the 1,000-sample tuning set leaves "
                  "caps unlearned by the defenseless model",
    "pirate": "rule not met: IP blocks caps at every prompt count (1-5)",
    "allcaps": "rule not met: IP blocks the Spanish confound at every prompt "
               "count (1-5)",
}
assert not set(KEPT_DEFAULT) & set(CHOSEN)

# traits_common names -> setting keys
TRAIT_KEY = {"sycophantic": "syco", "hallucinating": "halluc",
             "pirate": "pirate", "spanish": "spanish",
             "optimistic": "optimistic", "allcaps": "allcaps"}


def choice(key: str) -> Tuple[int, float]:
    """(n, eps) the main arms of setting ``key`` use."""
    return CHOSEN.get(key, DEFAULT)
