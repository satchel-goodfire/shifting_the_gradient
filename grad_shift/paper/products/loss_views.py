"""Loss and gradient cosine along a PPS vector, across prompts and datasets
(camera-ready appendix figure ``fig_cr_lossviews``).

Section 5.2: PPS adds ``eps * v`` at layer 20 at every position, so
``dL/d eps = sum_t <dL/dh_t, v>`` and the loss along ``v`` is lowest where
that summed projection changes sign. The paper reports C-bar, the
token-averaged cosine of the gradient with ``v``, which normalizes every
token's gradient; whether its zero falls at the loss minimum too is an
empirical question, which these analyses measure under

- two PPS vectors: the evil PV and the sycophantic PV rescaled to the evil
  PV's per-layer norm (``runs.SYCO_NM``, Figure 2c's off-trait PPS vector);
- two datasets: the Heavy evil data and the evil dataset's normal responses
  (the first ``loss_measure.N_SAMPLES`` = 256 samples of each, as Figure 4);
- four forward-pass prompt conditions: none, the 3-prompt evil IP bank
  (``runs.IP_BANK``), the first 3 prompts of the sycophancy bank, and the 4
  neutral prompts of Figure 2c. Each prompt is measured on its own;
  ``category_curves`` averages a category's prompts per sequence.

The cosine is taken with the steering vector itself, the derivative's
direction. Base model, step 0, no training. One analysis per (vector,
dataset), ~200 forward/backward passes over 256 sequences each.

Each condition also records ``gproj_sum`` and ``seq_loss_full``
(``loss_measure.measure``): ``|v| * mean(gproj_sum)`` is exactly the
derivative of the mean ``seq_loss_full``, which ``exactness`` checks against
the loss curve's finite differences, a test of the measurement itself.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..runs import DATA, EVIL, IP_BANK, SYCO_NM
from ..spec import PROTOCOL, REPO_ROOT
from .base import Analysis, Context
from .gradient_core import paper_sign
from .loss_measure import (LAYER, N_SAMPLES, SEP, TRAIT_TOKENS, Condition,
                           coef_name, load_chats, load_collector, load_npz,
                           load_trait_tokens, measure, read_prompts, save_npz)

# Intensities: dense within +-1 (where minima sit in Figure 4c), out to the
# Fig 4c sweep's +-6; 0 is the unsteered pass of each prompt.
VIEW_EPS = (-6.0, -4.5, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
            0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)
VECTORS = {"evil": EVIL, "syco": SYCO_NM}
DATASETS = {"heavy": DATA["heavy"], "normal": DATA["normal"]}
# prompt category -> prompt file (None: no prompt)
PROMPT_SETS: Dict[str, Optional[str]] = {
    "none": None,
    "evil": IP_BANK,
    "syco": "data/prompt_files/sycophantic_3samp.json",
    "neutral": "data/prompt_files/neutral.json",
}


def prompts(category: str) -> List[Optional[str]]:
    """The prompts of a category, in file order ([None] for no prompt)."""
    path = PROMPT_SETS[category]
    return [None] if path is None else read_prompts(path)


def member_names(category: str) -> List[str]:
    """Condition-name stems of a category's prompts: 'none' or 'evil0', ..."""
    if PROMPT_SETS[category] is None:
        return [category]
    return [f"{category}{k}" for k in range(len(prompts(category)))]


def cond_name(member: str, eps: float) -> str:
    return coef_name(member, eps)


def view_conditions(vector: str) -> List[Condition]:
    """Every (prompt, intensity) pass; eps 0 runs unsteered (vector None),
    which also lets ``measure`` guard the Heavy data's tokenization."""
    conds = []
    for cat in PROMPT_SETS:
        for member, p in zip(member_names(cat), prompts(cat)):
            for e in VIEW_EPS:
                conds.append(Condition(cond_name(member, e),
                                       None if e == 0 else vector, e,
                                       prompt=p))
    return conds


def vector_norm(path: str, layer: int = LAYER) -> float:
    """|v| at ``layer``: PPS adds ``eps * v``, so d/d eps = |v| <g, v/|v|>."""
    import torch
    v = torch.load(os.path.join(REPO_ROOT, path), map_location="cpu",
                   weights_only=True)[layer]
    return float(v.float().norm())


class _LossViews:
    """One (vector, dataset) cell; subclasses set ``vector`` and ``data``.
    (Not an ``Analysis`` itself, so product discovery skips it.)"""

    vector = ""
    data = ""
    time = "01:30:00"

    def __init__(self):
        self.inputs = [VECTORS[self.vector], DATASETS[self.data],
                       *(p for p in PROMPT_SETS.values() if p)]
        self.params = {"eps": list(VIEW_EPS), "n_samples": N_SAMPLES,
                       "layer": LAYER, "prompt_sets": PROMPT_SETS,
                       "cosine_reference": "steering vector"}

    def run(self, ctx: Context) -> dict:
        d = ctx.analysis_dir(self.name)
        path = os.path.join(d, "arrays.npz")
        vec = VECTORS[self.vector]
        if not os.path.isfile(path):
            col = load_collector(PROTOCOL["model"])
            # the frozen trait-token ids exist for the Heavy data only
            ids = (load_trait_tokens()["token_id"].to_numpy()
                   if self.data == "heavy" else None)
            arrs = measure(col, load_chats(DATASETS[self.data]),
                           view_conditions(vec), token_ids=ids, ref=vec)
            save_npz(path, arrs)
        arrs = load_npz(path)
        conds = sorted({k.split(SEP)[0] for k in arrs})
        drift = max(int(arrs[f"{c}{SEP}n_drift"].max()) for c in conds)
        if drift:
            raise ValueError(f"prompt alignment lost {drift} base tokens")
        return {"npz": path, "vector": vec, "data": DATASETS[self.data],
                "eps": list(VIEW_EPS), "v_norm": vector_norm(vec),
                "members": {c: member_names(c) for c in PROMPT_SETS},
                "trait_tokens_guard": TRAIT_TOKENS if self.data == "heavy"
                else None}


class LossViewsEvilHeavy(_LossViews, Analysis):
    name, vector, data = "loss_views_evil_heavy", "evil", "heavy"


class LossViewsEvilNormal(_LossViews, Analysis):
    name, vector, data = "loss_views_evil_normal", "evil", "normal"


class LossViewsSycoHeavy(_LossViews, Analysis):
    name, vector, data = "loss_views_syco_heavy", "syco", "heavy"


class LossViewsSycoNormal(_LossViews, Analysis):
    name, vector, data = "loss_views_syco_normal", "syco", "normal"


ANALYSES = {(a.vector, a.data): a.name for a in (
    LossViewsEvilHeavy, LossViewsEvilNormal, LossViewsSycoHeavy,
    LossViewsSycoNormal)}


# ---------------------------------------------------------------- curves

def curve(arrs: Dict[str, np.ndarray], members: Sequence[str],
          eps: Sequence[float]) -> Dict[float, Dict[str, float]]:
    """Per intensity, over ``members`` (prompts averaged per sequence): mean
    ``loss`` (seq_loss_all, as Figure 4c), ``loss_full`` (the differentiated
    loss), ``gcos`` (C-bar in the paper's sign, + = amplifying, pooled over
    tokens) and ``gproj`` (mean raw summed projection per sequence)."""
    out = {}
    for e in eps:
        names = [cond_name(m, e) for m in members]

        def avg(field):
            return np.mean([arrs[f"{n}{SEP}{field}"] for n in names], axis=0)

        gsum = sum(arrs[f"{n}{SEP}gcos_sum"].sum() for n in names)
        gcnt = sum(arrs[f"{n}{SEP}gcos_n"].sum() for n in names)
        out[float(e)] = {"loss": float(np.nanmean(avg("seq_loss_all"))),
                         "loss_full": float(np.nanmean(avg("seq_loss_full"))),
                         "gcos": float(paper_sign(gsum / gcnt)),
                         "gproj": float(np.mean(avg("gproj_sum")))}
    return out


def loss_minimum(xs: Sequence[float], ys: Sequence[float]
                 ) -> Tuple[float, Optional[float]]:
    """(grid argmin, vertex of the parabola through it and its neighbours);
    the vertex is None when the argmin is at the grid's edge (the minimum
    may lie outside the sweep)."""
    i = int(np.argmin(ys))
    if i == 0 or i == len(xs) - 1:
        return float(xs[i]), None
    (x0, x1, x2), (y0, y1, y2) = xs[i - 1:i + 2], ys[i - 1:i + 2]
    # vertex of the interpolating parabola (unequal spacing)
    num = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
    den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    return float(x1), float(x1 - 0.5 * num / den) if den else float(x1)


def downward_crossings(xs: Sequence[float], ys: Sequence[float]
                       ) -> List[Tuple[float, float, float]]:
    """Where ``ys`` goes from > 0 to <= 0 as ``xs`` grows: (left, right,
    linearly interpolated zero). For C-bar in the paper's sign this is
    where the loss stops falling and starts rising (a minimum)."""
    out = []
    for (a, ya), (b, yb) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if ya > 0 >= yb:
            out.append((float(a), float(b), float(a + ya * (b - a) / (ya - yb))))
    return out


def exactness(c: Dict[float, Dict[str, float]], v_norm: float
              ) -> Tuple[float, float]:
    """The measurement check: between neighbouring intensities, the slope
    of the differentiated loss vs |v| times the mean summed projection
    (trapezoid of its two ends). Returns (Pearson r, median |relative
    error|) over the intervals."""
    xs = sorted(c)
    fd = [(c[b]["loss_full"] - c[a]["loss_full"]) / (b - a)
          for a, b in zip(xs, xs[1:])]
    an = [v_norm * 0.5 * (c[a]["gproj"] + c[b]["gproj"])
          for a, b in zip(xs, xs[1:])]
    fd, an = np.asarray(fd), np.asarray(an)
    rel = np.abs(fd - an) / np.maximum(np.abs(fd), 1e-12)
    return float(np.corrcoef(fd, an)[0, 1]), float(np.median(rel))
