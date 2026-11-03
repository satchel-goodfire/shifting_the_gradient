"""IP in the forward pass with the paper's prompt BANK (no Product/Analysis).

The main IP defense trains with a bank of prompts (``runs.IP_BANK``: three
prompts, one sampled per training example, seeded). An analysis that puts IP
in the forward pass (gradient cosines, losses, activation
deltas, ...) therefore measures every sequence once with EACH bank prompt
and averages the per-sequence (or per-token) statistic over the prompts.
This is deterministic (no sampling) and is the expectation, over the
training-time prompt draw, of the statistic for that sequence.

Naming: bank member ``k`` is the condition ``ip_b<k>``; the averaged arm is
``ip``. The single prompt p1 (``runs.P1``, the earlier main IP, kept for
appendix comparisons) is one of the bank's prompts, so its arm ``ip_p1`` is
a copy of that member and costs no extra pass.

Rules for averaging, so every analysis does the same thing:
- statistics of one sequence (C-bar, pooled cosines, mean gradients,
  sequence losses, per-token losses on the ORIGINAL response tokens) are
  averaged elementwise across members (the members share those positions);
- sums with counts (token-pooled cosines) average the per-sequence sums
  with equal weight per member (counts are identical when no base token is
  lost to the alignment, which every analysis checks).
"""

import json
import os
from typing import Dict, List, MutableMapping, Optional, Sequence

import numpy as np

from ..runs import IP_BANK, P1
from ..spec import REPO_ROOT

BANK = "ip"  # averaged-arm name
P1_ARM = "ip_p1"  # single-prompt comparison arm (copy of one member)


def p1_arm_name(bank: str = BANK) -> str:
    """Name of a bank's single-prompt-p1 arm: 'ip' -> 'ip_p1'."""
    return f"{bank}_p1"


def read_prompts(path: str = IP_BANK) -> List[str]:
    """Prompts of a prompt file (``{"prompts": [...]}`` or a bare list)."""
    with open(os.path.join(REPO_ROOT, path)) as f:
        data = json.load(f)
    return list(data["prompts"] if isinstance(data, dict) else data)


def member_name(k: int, bank: str = BANK) -> str:
    return f"{bank}_b{k}"


def members(bank: str = BANK, path: str = IP_BANK) -> List[str]:
    """Condition names of the bank's members, in prompt-file order."""
    return [member_name(k, bank) for k in range(len(read_prompts(path)))]


def p1_index(path: str = IP_BANK) -> int:
    """Index of the single prompt p1 within the bank (must be present)."""
    (p1,) = read_prompts(P1)
    bank = read_prompts(path)
    if p1 not in bank:
        raise ValueError(f"{P1}'s prompt is not in {path}")
    return bank.index(p1)


def p1_member(bank: str = BANK) -> str:
    return member_name(p1_index(), bank)


def average(values: Sequence[np.ndarray]) -> np.ndarray:
    """Elementwise mean over members (all of one shape)."""
    arrs = [np.asarray(v, dtype=np.float64) for v in values]
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"bank members disagree in shape: {sorted(shapes)}")
    return np.mean(np.stack(arrs), axis=0)


def add_bank_arms(per_cond: MutableMapping[str, np.ndarray],
                  bank: str = BANK, p1_arm: Optional[str] = None,
                  key=lambda cond: cond) -> None:
    """Add the averaged ``bank`` arm and the ``p1_arm`` copy in place.

    ``per_cond`` maps ``key(condition name)`` to an array; members must all
    be present. ``key`` lets callers use composite keys (e.g. npz fields
    ``f"{cond}__seq_loss_all"`` via ``key=lambda c: f"{c}__seq_loss_all"``).
    """
    names = members(bank)
    p1_arm = p1_arm or p1_arm_name(bank)
    per_cond[key(bank)] = average([per_cond[key(n)] for n in names])
    per_cond[key(p1_arm)] = np.array(per_cond[key(p1_member(bank))],
                                     copy=True)


def add_bank_fields(arrs: Dict[str, np.ndarray], sep: str,
                    bank: str = BANK, p1_arm: Optional[str] = None) -> None:
    """``add_bank_arms`` for every field of flat ``{cond<sep>field}`` dicts
    (the loss-measurement npz layout). Integer fields that count lost
    alignment tokens (``n_drift``) take the max over members instead."""
    names = members(bank)
    p1_arm = p1_arm or p1_arm_name(bank)
    fields = {k.split(sep, 1)[1] for k in arrs
              if k.split(sep, 1)[0] == names[0]}
    for fld in sorted(fields):
        vals = [arrs[f"{n}{sep}{fld}"] for n in names]
        arrs[f"{bank}{sep}{fld}"] = (np.max(np.stack(vals), axis=0)
                                     if fld == "n_drift" else average(vals))
        arrs[f"{p1_arm}{sep}{fld}"] = np.array(
            arrs[f"{p1_member(bank)}{sep}{fld}"], copy=True)
