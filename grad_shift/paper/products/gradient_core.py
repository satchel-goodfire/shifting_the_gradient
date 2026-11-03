"""Shared machinery for the gradient analyses (Sections 4, 6 and appendices).

Every gradient figure measures the activation gradient dL/dh_l (post-block
residual stream, layer index l = output of transformer block l, the
embedding-inclusive convention of ``grad_shift.hooks``) of the full
all-token training loss on one training sequence, under one forward-pass
*condition* (no defense, a PPS vector added at some layers, or an IP prompt
prepended to the user turn). This module holds:

- pure helpers (token alignment, data splits, uncentered PCA, summaries)
  that are unit-tested on fabricated data, and
- ``GradientProbe``, the GPU engine shared by the Analyses in
  ``gradient_*.py`` (torch/transformers are imported lazily so that product
  discovery stays CPU-cheap).

Conventions (ported from scripts/analyze_gradient_metrics.py,
analyze_grad_decomp.py, compare_grad_steered_vs_prompted.py):

- Data: the heavy evil training set (``runs.DATA["heavy"]``), first N
  sequences in file order, system prompt "You are an LLM chatbot.",
  rendered with the model's chat template (the trainer's format).
- C_t(l, v) = cos(dL/dh_{l,t}, v); a sequence's statistic is the mean over
  token positions, C-bar (paper Eq. 1). The cosine of the token-mean
  gradient ("pooled") is stored too.
- IP conditions insert prompt tokens; metrics use only positions present
  in the undefended sequence ("non-prompt": common prefix + common suffix
  of the two tokenizations) or, where the paper says "tokens after the IP
  prompt", only the common suffix.
- IP = the 3-prompt bank (``runs.IP_BANK``): each sequence is measured with
  EACH bank prompt (conditions ``ip_b<k>``, ``ip_bank_conditions``) and the
  per-sequence statistics are averaged over the prompts into the ``ip`` arm
  (``ip_bank.add_bank_arms``); ``ip_p1`` is the single-prompt-p1 member.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..runs import DATA, EPS, EPS_STRONG, EVIL, IP_BANK, P1, RAND_NM, SYCO
from ..spec import PROTOCOL, REPO_ROOT
from . import ip_bank

LAYER = 20  # PPS layer and the paper's headline gradient layer
N_LAYERS = 28  # Qwen2.5-7B transformer blocks
LAYERS = tuple(range(1, N_LAYERS + 1))
HEAVY = DATA["heavy"]
NEUTRAL = "data/prompt_files/neutral.json"
SYSTEM_PROMPT = PROTOCOL["baseline_system_prompt"]
# Reference directions every gradient analysis projects on. "rand" is the
# norm-matched random PPS vector, so panel (a)/(c) project on the same
# direction the random-PPS control steers along.
DIRECTIONS = {"evil": EVIL, "rand": RAND_NM, "syco": SYCO}


# ---------------------------------------------------------------------------
# Pure helpers (no torch)
# ---------------------------------------------------------------------------

def common_prefix_suffix(base: Sequence[int], defended: Sequence[int]
                         ) -> Tuple[int, int]:
    """Lengths of the common prefix and suffix of two token sequences.

    The suffix is capped so prefix + suffix never exceeds the shorter
    sequence (no position is counted twice). For an IP prompt prepended to
    the user turn the prefix is the system turn + "<|im_start|>user\\n" and
    the suffix is everything after the prompt.
    """
    n = min(len(base), len(defended))
    p = 0
    while p < n and base[p] == defended[p]:
        p += 1
    s = 0
    while (s < n - p and base[len(base) - 1 - s]
           == defended[len(defended) - 1 - s]):
        s += 1
    return p, s


def aligned_positions(base: Sequence[int], defended: Sequence[int],
                      which: str = "nonprompt"
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Matching position indices (into base, into defended).

    ``which="nonprompt"``: prefix + suffix positions (every token of the
    undefended sequence, when the defended one only inserts tokens).
    ``which="suffix"``: only the common suffix (tokens after the prompt).
    Identical sequences align position by position.
    """
    if which not in ("nonprompt", "suffix"):
        raise ValueError(f"unknown alignment {which!r}")
    if list(base) == list(defended):  # no prompt inserted (none / PPS)
        idx = np.arange(len(base), dtype=np.int64)
        return idx, idx.copy()
    p, s = common_prefix_suffix(base, defended)
    bi = list(range(len(base) - s, len(base)))
    di = list(range(len(defended) - s, len(defended)))
    if which == "nonprompt":
        bi = list(range(p)) + bi
        di = list(range(p)) + di
    return np.asarray(bi, dtype=np.int64), np.asarray(di, dtype=np.int64)


def loss_positions(keep: np.ndarray) -> np.ndarray:
    """Per-token-loss indices whose *target* is a kept position.

    ``loss[i]`` predicts token ``i + 1``, so a kept position j >= 1
    contributes ``loss[j - 1]``.
    """
    keep = np.asarray(keep)
    return keep[keep >= 1] - 1


def split_indices(n_total: int, n_first: int, seed: int = 42
                  ) -> Tuple[List[int], List[int]]:
    """Deterministic disjoint split of ``range(n_total)``.

    A seeded permutation (Python ``random.Random(seed).shuffle``, stable
    across platforms) whose first ``n_first`` entries form the first part.
    """
    if not 0 <= n_first <= n_total:
        raise ValueError("n_first out of range")
    idx = list(range(n_total))
    random.Random(seed).shuffle(idx)
    return idx[:n_first], idx[n_first:]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(np.asarray(a, float)) @ unit(np.asarray(b, float)))


# Paper sign convention (Eq. 1; flipped 2026-10-07 so
# that "up" reads as "more trait"): the paper reports cosines with the
# NEGATIVE activation gradient, -dL/dh -- the direction a gradient-descent
# step pushes h -- so a positive value is an amplifying signal along v and a
# negative value an attenuating one. Analyses (and their cached results)
# store the raw cos(dL/dh, v); products convert with ``paper_sign`` before
# plotting or writing keys. Cosines between two gradients are unaffected
# (both factors flip), and so are activation statistics.
PAPER_SIGN = -1.0


def paper_sign(x):
    """Raw gradient statistic(s) (cos(dL/dh, v), or the gradients
    themselves) -> the paper's convention (-dL/dh; positive = amplifying)."""
    if isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype=np.float64)
    return PAPER_SIGN * x


def uncentered_pca(diffs: np.ndarray, refs: Dict[str, np.ndarray],
                   sign_ref: str = "evil", k: int = 5,
                   threshold: float = 0.95) -> dict:
    """Uncentered PCA of per-sample gradient differences (Appx. E).

    ``diffs`` is (N, D): defended minus defenseless sequence-mean
    gradients. The matrix is normalized by its Frobenius norm and
    decomposed by SVD; the eigenvalues of M = G~^T G~ / (N - 1) are
    s_k^2 / (N - 1), so the fraction of explained variance is
    s_k^2 / sum s^2 (no centering: the mean shift is part of the signal).
    Each PC is sign-flipped to have non-negative cosine with
    ``refs[sign_ref]``.

    Returns fractions of variance (all components), the number needed to
    reach ``threshold``, the top-k PCs' cosines with every reference
    direction, and the cosine of the mean difference with each reference.
    """
    g = np.asarray(diffs, dtype=np.float64)
    n = g.shape[0]
    fro = np.linalg.norm(g)
    if fro > 0:
        g = g / fro
    _, s, vt = np.linalg.svd(g, full_matrices=False)
    lam = s ** 2 / max(n - 1, 1)
    frac = lam / lam.sum() if lam.sum() > 0 else lam
    k = min(k, vt.shape[0])
    pcs = vt[:k].copy()
    sref = unit(np.asarray(refs[sign_ref], dtype=np.float64))
    for i in range(k):
        if pcs[i] @ sref < 0:
            pcs[i] *= -1
    pc_cos = {name: [float(pcs[i] @ unit(np.asarray(r, np.float64)))
                     for i in range(k)] for name, r in refs.items()}
    mean = np.asarray(diffs, np.float64).mean(axis=0)
    return {
        "n": int(n),
        "frac_var": [float(x) for x in frac],
        "eigenvalues": [float(x) for x in lam],
        "n_to_threshold": int(np.searchsorted(np.cumsum(frac), threshold)
                              + 1),
        "threshold": threshold,
        "pc_cos": pc_cos,
        "meandiff_cos": {name: cos(mean, r) for name, r in refs.items()},
        "pcs": pcs,  # (k, D) unit vectors; callers drop before JSON
    }


def mean_sem(x) -> Tuple[float, float]:
    """Mean and standard error of the mean (ddof=1) of finite values."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    sem = float(a.std(ddof=1) / math.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def load_vector_file(path: str) -> np.ndarray:
    """A [29, D] steering/direction file as float64 numpy (repo-relative)."""
    import torch

    t = torch.load(os.path.join(REPO_ROOT, path), map_location="cpu",
                   weights_only=True)
    return t.float().numpy().astype(np.float64)


def save_npz(path: str, **arrays) -> str:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One forward-pass condition for a gradient measurement.

    ``pps_vector``/``pps_coef``/``pps_layers``: steering added to the
    output of those blocks (frozen, as in training). ``prompt``: IP text
    prepended to the first user turn (``prompt_mode=prepend_user``).
    """

    name: str
    pps_vector: Optional[str] = None
    pps_coef: float = 0.0
    pps_layers: tuple = (LAYER,)
    prompt: Optional[str] = None

    def describe(self) -> dict:
        return {"name": self.name, "pps_vector": self.pps_vector,
                "pps_coef": self.pps_coef,
                "pps_layers": list(self.pps_layers), "prompt": self.prompt}


def prompt_of(path: str, index: int = 0) -> str:
    with open(os.path.join(REPO_ROOT, path)) as f:
        return json.load(f)["prompts"][index]


def n_prompts(path: str) -> int:
    with open(os.path.join(REPO_ROOT, path)) as f:
        return len(json.load(f)["prompts"])


NONE = Condition("none")
PPS = Condition("pps", pps_vector=EVIL, pps_coef=EPS)
PPS_RAND = Condition("pps_rand", pps_vector=RAND_NM, pps_coef=EPS)
# Strong-defense comparison (the earlier epsilon choice)
PPS_STRONG = Condition("pps_eps9", pps_vector=EVIL, pps_coef=EPS_STRONG)


def ip_condition(name: str = "ip_p1", path: str = P1, index: int = 0
                 ) -> Condition:
    """A single-prompt IP condition (default: the single prompt p1)."""
    return Condition(name, prompt=prompt_of(path, index))


def ip_bank_conditions(bank: str = ip_bank.BANK, path: str = IP_BANK,
                       **kw) -> List[Condition]:
    """One condition per bank prompt (``ip_b0``, ``ip_b1``, ...); average the
    per-sequence results with ``ip_bank.add_bank_arms``. ``kw`` adds PPS on
    top (``pps_vector``, ``pps_coef``, ``pps_layers``)."""
    return [Condition(name, prompt=p, **kw) for name, p in
            zip(ip_bank.members(bank, path), ip_bank.read_prompts(path))]


# ---------------------------------------------------------------------------
# GPU engine
# ---------------------------------------------------------------------------

@dataclass
class Pass:
    """One forward/backward pass: tokens, per-layer grads/acts (CPU)."""

    tokens: List[int]
    grads: Dict[int, "object"]  # layer -> (T, D) float32 torch tensor
    acts: Dict[int, "object"]
    loss: "object"  # (T,) per-token CE (last position NaN)


class GradientProbe:
    """Loads one model and measures activation gradients under conditions.

    Args:
        model: HF model id or LoRA adapter directory (adapter applied at
            runtime, not merged; see ``ActivationCollector``).
        layers: embedding-inclusive layer indices to hook.
    """

    def __init__(self, model: str = PROTOCOL["model"],
                 layers: Sequence[int] = LAYERS, device: str = "cuda:0",
                 dtype: str = "bfloat16", freeze: bool = True,
                 response_only: bool = False):
        import torch

        # Gradients of the loss on the response tokens only (as a run
        # trained with train_on_responses_only); default: all tokens, the
        # paper protocol.
        self.response_only = response_only

        from ...collector import ActivationCollector

        self.torch = torch
        self.device = device
        self.layers = list(layers)
        self.collector = ActivationCollector(
            model, device=device, dtype=dtype, layers=self.layers,
            trust_remote_code=True)
        if freeze:
            # Activation grads are unchanged (the graph is built from
            # inputs_embeds); parameter grads are never allocated.
            for p in self.collector.model.parameters():
                p.requires_grad_(False)
        self._steer: Dict[tuple, object] = {}

    def _manager(self, cond: Condition):
        from ...steering.steering_hooks import SteeringHookManager
        from ...steering.steering_module import SteeringModule

        key = (cond.pps_vector, tuple(cond.pps_layers))
        if key not in self._steer:
            mod = SteeringModule.from_pretrained(
                os.path.join(REPO_ROOT, cond.pps_vector),
                global_scale=cond.pps_coef, device=self.device)
            mod.requires_grad_(False)
            self._steer[key] = (mod, SteeringHookManager(
                self.collector.model, mod,
                active_layers=list(cond.pps_layers)))
        mod, mgr = self._steer[key]
        mod.set_global_scale(cond.pps_coef)
        return mgr

    def messages(self, chat, cond: Condition):
        from ...data_utils import inject_prompt_into_chat

        return inject_prompt_into_chat(chat, prompt=cond.prompt,
                                       mode="prepend_user",
                                       baseline_system_prompt=SYSTEM_PROMPT)

    def run(self, chat, cond: Condition, grads: bool = True,
            keep_param_grads: bool = False, acts: bool = True) -> Pass:
        """One pass of ``chat`` under ``cond``.

        ``keep_param_grads`` leaves parameter ``.grad`` populated (for the
        activation-delta analysis, which loads the model unfrozen);
        ``acts=False`` drops activations (gradient-only analyses).
        """
        msgs = self.messages(chat, cond)
        mgr = self._manager(cond) if cond.pps_vector else None
        if mgr is not None:
            mgr.register_hooks()
        try:
            r = self.collector.collect_chat(
                msgs, collect_gradients=grads,
                response_only=self.response_only)
        finally:
            if mgr is not None:
                mgr.remove_hooks()
        if not keep_param_grads:
            self.collector.model.zero_grad(set_to_none=True)
        return Pass(tokens=r.tokens,
                    grads={l: g.float() for l, g in r.gradients.items()},
                    acts=({l: a.float() for l, a in r.activations.items()}
                          if acts else {}),
                    loss=r.loss.float())

    def to_dev(self, x):
        return x.to(self.device)


def tok_cos(x, v):
    """(T, D) torch x (T,) cosines with unit (D,) v."""
    return (x @ v) / (x.norm(dim=1) * v.norm() + 1e-12)


def paired_cos(a, b):
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def direction_tensors(probe: GradientProbe, names: Sequence[str]
                      ) -> Dict[str, Dict[int, object]]:
    """Unit direction vectors per layer on the probe's device."""
    torch = probe.torch
    out = {}
    for n in names:
        raw = load_vector_file(DIRECTIONS[n])
        out[n] = {l: torch.tensor(unit(raw[l]), dtype=torch.float32,
                                  device=probe.device) for l in probe.layers}
    return out


def sequence_stats(p: Pass, keep, dirs, layers, mean_grad_layers=()):
    """Per-layer C-bar and pooled cosine of one pass at kept positions.

    Returns (tokmean[L, n_dir], pooled[L, n_dir], {layer: mean grad (D,)},
    mean loss over kept targets).
    """
    import torch

    names = list(dirs)
    keep_t = torch.as_tensor(np.asarray(keep), dtype=torch.long)
    tm = np.zeros((len(layers), len(names)))
    pl = np.zeros((len(layers), len(names)))
    mg = {}
    dev = next(iter(dirs[names[0]].values())).device if names else "cpu"
    for i, l in enumerate(layers):
        g = p.grads[l][keep_t].to(dev)
        gm = g.mean(dim=0, keepdim=True)
        for j, n in enumerate(names):
            v = dirs[n][l]
            tm[i, j] = float(tok_cos(g, v).mean())
            pl[i, j] = float(tok_cos(gm, v)[0])
        if l in mean_grad_layers:
            mg[l] = gm[0].cpu().numpy().astype(np.float32)
    li = loss_positions(np.asarray(keep))
    lv = p.loss[torch.as_tensor(li, dtype=torch.long)]
    lv = lv[~torch.isnan(lv)]
    return tm, pl, mg, float(lv.mean()) if lv.numel() else float("nan")


def load_chats(n: Optional[int] = None, path: str = HEAVY,
               indices: Optional[Sequence[int]] = None) -> list:
    """Training chats (file order), optionally the first n or by index."""
    with open(os.path.join(REPO_ROOT, path)) as f:
        chats = [json.loads(line)["messages"] for line in f if line.strip()]
    if indices is not None:
        chats = [chats[i] for i in indices]
    return chats if n is None else chats[:n]


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"  # provenance only (e.g. no git in the job env)
