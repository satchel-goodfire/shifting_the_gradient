"""Shared measurement code for the sample-level loss products (``loss_fig``)
and the response-only check (``rocheck``): per-sequence / per-token next-token
loss and layer-20 gradient cosines under a defensive object, and the
token-level selectivity statistic that the response-only check's
pre-registered criteria include.

Ported from the grad_shift scripts that produced the published numbers:
``measure_defense_token_loss.py`` (per-token response losses under PPS, IP and
controls), ``analyze_gradient_metrics.py`` (per-sequence loss and tokenwise
gradient cosine, excluding injected prompt tokens), and
``analyze_token_loss_shift.py`` / ``analyze_explainaway_curve.py``
(decile-matched controls and the selectivity statistic). Differences from
those scripts, all deliberate:

- One forward/backward per (condition, batch) yields every quantity, so the
  sequence-level scatter and the coefficient sweep come from the same
  samples and passes.
- Injected-prompt tokens are excised by longest common prefix AND suffix
  against the unprompted rendering (the old code assumed a pure insertion and
  dropped one extra target token per prompted sequence); the number of base
  tokens that do not survive the alignment is recorded and must be 0.
- The cluster bootstrap is vectorized (multinomial response weights); point
  estimates are identical to the old loop, CIs are equal in distribution.
- Selectivity is Delta control - Delta trait (positive = selective;
  2026-10-21, ``SELECTIVITY_SIGN``), the negative of those scripts'
  Delta trait - Delta control.

Conventions (see ``grad_shift/collector.py``): ``loss[i]`` is the CE of
predicting token ``i+1`` from position ``i``; layer indices are
embedding-inclusive (20 = output of ``model.layers[19]``, where PPS adds its
vector). The model is in eval mode, so LoRA dropout is inactive.
"""

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...chat_template import render_chat
from ..runs import DATA, EVIL
from ..spec import PROTOCOL, REPO_ROOT
from . import ip_bank
from .gradient_core import paper_sign

LAYER = 20
# Frozen judge-labeled trait-token set (first 256 heavy-data samples); built
# by scripts/paper_export_legacy_inputs.py from the archived judge outputs.
TRAIT_TOKENS = "data/trait_tokens/evil_misaligned2_n256.csv"
N_SAMPLES = 256
TRAIN_FILE = DATA["heavy"]
NEUTRAL = "data/prompt_files/neutral.json"  # 4 single prompts (Fig 2c's)
BATCH_SIZE = 8
SEP = "__"  # npz key = <condition>__<field>
N_BOOT = 2000


@dataclass(frozen=True)
class Condition:
    """One forward-pass condition: optional steering and/or injected prompt."""

    name: str
    vector: Optional[str] = None  # repo-relative steering file
    coef: float = 0.0
    prompt: Optional[str] = None  # prepended to the first user turn
    grad: bool = True  # also collect the layer-20 gradient
    proj: bool = False  # also record response-token activation projections


def coef_name(prefix: str, coef: float) -> str:
    """Condition name for a steering coefficient: ('pps', -1.5) -> 'pps_cm1.5'."""
    return f"{prefix}_c{coef:g}".replace("-", "m")


def read_prompts(path: str) -> List[str]:
    import json
    with open(os.path.join(REPO_ROOT, path)) as f:
        data = json.load(f)
    return data["prompts"] if isinstance(data, dict) else data


def bank_conditions(bank: str = ip_bank.BANK, vector: Optional[str] = None,
                    coef: float = 0.0, grad: bool = True
                    ) -> List[Condition]:
    """One condition per IP-bank prompt (``ip_bank``), optionally with PPS
    on top; ``measure(..., banks=[bank])`` adds the averaged ``bank`` arm
    and the single-prompt ``<bank>_p1`` arm."""
    return [Condition(name, vector, coef, prompt=p, grad=grad)
            for name, p in zip(ip_bank.members(bank),
                               ip_bank.read_prompts())]


# ---------------------------------------------------------------- pure logic

def split_injection(base: Sequence[int], prompted: Sequence[int]
                    ) -> Tuple[int, int]:
    """(common-prefix length, common-suffix length) of two token sequences.

    The suffix is capped so prefix and suffix never overlap in either
    sequence; for identical inputs it is (len, 0).
    """
    n = min(len(base), len(prompted))
    lcp = 0
    while lcp < n and base[lcp] == prompted[lcp]:
        lcp += 1
    lcs = 0
    cap = n - lcp
    while lcs < cap and base[-1 - lcs] == prompted[-1 - lcs]:
        lcs += 1
    return lcp, lcs


def original_positions(length: int, lcp: int, lcs: int) -> np.ndarray:
    """Positions of a (possibly prompted) sequence holding original tokens."""
    return np.concatenate([np.arange(lcp),
                           np.arange(length - lcs, length)]).astype(int)


def matched_controls(ref_loss: np.ndarray, is_trait: np.ndarray,
                     n_deciles: int = 10, seed: int = 0) -> np.ndarray:
    """Non-trait tokens matched to the trait tokens' ``ref_loss`` deciles.

    Exact port of ``analyze_explainaway_curve.matched_controls`` (same RNG
    draws), so step-0 values reproduce the archived analysis.
    """
    rng = np.random.default_rng(seed)
    deciles = pd.qcut(ref_loss, n_deciles, labels=False, duplicates="drop")
    deciles = np.asarray(deciles, dtype=float)
    control = np.zeros(len(ref_loss), dtype=bool)
    pool = ~is_trait
    for d in np.unique(deciles[~np.isnan(deciles)]):
        want = int(((deciles == d) & is_trait).sum())
        cand = np.where((deciles == d) & pool)[0]
        if want == 0 or len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(want, len(cand)), replace=False)
        control[pick] = True
    return control


def _group_sums(values: np.ndarray, mask: np.ndarray, gidx: np.ndarray,
                n_groups: int) -> Tuple[np.ndarray, np.ndarray]:
    ok = mask & ~np.isnan(values)
    s = np.bincount(gidx[ok], weights=values[ok], minlength=n_groups)
    c = np.bincount(gidx[ok], minlength=n_groups).astype(float)
    return s, c


# Sign of every selectivity this module returns (recorded with cached ones):
# Delta loss on matched controls minus Delta loss on trait tokens, so an object
# that lowers the loss on the trait tokens more than on the controls is
# positive ("selective"). 2026-10-21; earlier results were cached as
# trait - control and are converted by ``current_selectivity``.
SELECTIVITY_SIGN = "control_minus_trait"


def selectivity(delta: np.ndarray, ref_loss: np.ndarray, is_trait: np.ndarray,
                groups: np.ndarray, n_boot: int = N_BOOT, seed: int = 0,
                n_deciles: int = 10) -> Dict[str, float]:
    """Explain-away selectivity: mean(delta | control) - mean(delta | trait);
    positive = the object selectively explains the trait tokens
    (``SELECTIVITY_SIGN``).

    Controls are decile-matched to the trait tokens on ``ref_loss`` (the loss
    without the object, at the model being measured). The 95% CI resamples
    whole responses (``groups``) with replacement.
    """
    delta = np.asarray(delta, float)
    is_trait = np.asarray(is_trait, bool)
    control = matched_controls(np.asarray(ref_loss, float), is_trait,
                               n_deciles, seed)
    _, gidx = np.unique(np.asarray(groups), return_inverse=True)
    g = int(gidx.max()) + 1
    st, ct = _group_sums(delta, is_trait, gidx, g)
    sc, cc = _group_sums(delta, control, gidx, g)
    d_t, d_c = st.sum() / ct.sum(), sc.sum() / cc.sum()
    rng = np.random.default_rng(seed)
    w = rng.multinomial(g, np.full(g, 1.0 / g), size=n_boot).astype(float)
    boot = (w @ sc) / (w @ cc) - (w @ st) / (w @ ct)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"delta_trait": float(d_t), "delta_control": float(d_c),
            "selectivity": float(d_c - d_t), "lo": float(lo), "hi": float(hi),
            "n_trait": int(is_trait.sum()), "n_control": int(control.sum())}


def current_selectivity(rec: Dict[str, float],
                        sign: Optional[str]) -> Dict[str, float]:
    """A cached ``selectivity`` record in the current sign. ``sign``: the
    ``SELECTIVITY_SIGN`` recorded with it (None: cached before 2026-10-21,
    trait - control). The conversion is exact: the bootstrap CI of a negated
    statistic is the negated, swapped CI."""
    if sign == SELECTIVITY_SIGN:
        return rec
    if sign is not None:
        raise ValueError(f"unknown selectivity sign {sign!r}")
    if not np.isclose(rec["selectivity"],
                      rec["delta_trait"] - rec["delta_control"], atol=1e-9):
        raise ValueError("record is not trait - control; refusing to convert")
    return {**rec, "selectivity": rec["delta_control"] - rec["delta_trait"],
            "lo": -rec["hi"], "hi": -rec["lo"]}


def load_trait_tokens(path: str = TRAIT_TOKENS) -> pd.DataFrame:
    """A frozen trait-token table (the evil one unless ``path``), sorted by
    (sample_id, resp_token_idx)."""
    df = pd.read_csv(os.path.join(REPO_ROOT, path))
    return df.sort_values(["sample_id", "resp_token_idx"]).reset_index(drop=True)


def gradcos(arrs: Dict[str, np.ndarray], cond: str) -> float:
    """Token-pooled mean cosine of the layer-20 gradient with the evil PV,
    in the paper's sign (positive = amplifying; the stored sums are raw
    cos(dL/dh, v), ``gradient_core.paper_sign``)."""
    return paper_sign(float(arrs[f"{cond}{SEP}gcos_sum"].sum()
                            / arrs[f"{cond}{SEP}gcos_n"].sum()))


# ---------------------------------------------------------------- GPU stage

def _tokenize(tokenizer, messages) -> List[int]:
    # Exactly the collector's tokenization (_tokenize_chat_batch).
    text = render_chat(tokenizer, messages, add_generation_prompt=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def load_collector(model_path: str, device: str = "cuda", layer: int = LAYER):
    """Collector for a base model or PEFT checkpoint, hooking ``layer`` only
    (20 unless a measurement steers elsewhere).

    Parameters are frozen: activation gradients are unchanged (the graph is
    built from ``inputs_embeds``) and no parameter gradients are allocated.
    """
    from grad_shift.collector import ActivationCollector

    col = ActivationCollector(model_path, device=device,
                              dtype=PROTOCOL["dtype"], layers=[layer],
                              trust_remote_code=True,
                              base_model=PROTOCOL["model"])
    for p in col.model.parameters():
        p.requires_grad_(False)
    return col


def unit_vector(path: str, layer: int = LAYER):
    import torch
    v = torch.load(os.path.join(REPO_ROOT, path), map_location="cpu",
                   weights_only=True)[layer].float()
    return v / v.norm()


def measure(collector, chats: List[list], conditions: List[Condition],
            token_ids: Optional[np.ndarray] = None,
            batch_size: int = BATCH_SIZE,
            log: Callable[[str], None] = print,
            banks: Sequence[str] = (),
            layer: int = LAYER,
            ref: str = EVIL) -> Dict[str, np.ndarray]:
    """Run every condition over ``chats``; returns flat ``{cond__field: array}``.

    Steering acts at ``layer`` (default 20, the paper's PPS layer), where the
    gradient cosine and projections are also read; the collector must hook
    that layer (``load_collector(..., layer=layer)``). ``ref`` is the vector
    file whose ``layer`` direction the cosine, projections and ``gproj_sum``
    use (default the evil PV).

    Fields per condition: ``seq_loss_all`` / ``seq_loss_resp`` (per-sequence
    mean CE over all original tokens / response tokens), ``seq_loss_full``
    (mean CE over every predicted token, injected prompt included: the
    loss the backward pass differentiates, ``collector._compute_batch_loss``),
    ``tok_loss`` (response-token CE, concatenated in sample order),
    ``gcos_sum`` / ``gcos_n`` (per-sequence sum and count of the tokenwise
    layer-20 gradient cosine with ``ref`` over original-token positions),
    ``gproj_sum`` (per-sequence sum over EVERY position of the gradient's
    projection on the unit ``ref``: since steering adds ``coef * v`` at every
    position, ``|v| * gproj_sum`` is d ``seq_loss_full`` / d coef when ``ref``
    is the steering vector), ``out_proj`` (if ``proj``: projection of the
    activation predicting each response token onto the unit ``ref``), and
    ``n_drift`` (base tokens lost by the prompt alignment; must be 0). If
    ``token_ids`` is given, the unprompted response tokens must equal it
    (guards the frozen trait-token labels). For each name in ``banks``
    (conditions from ``bank_conditions``), the per-member fields are averaged
    into ``<bank>`` and member p1 is copied to ``<bank>_p1``
    (``ip_bank.add_bank_fields``).
    """
    import torch

    from grad_shift.data_utils import inject_prompt_into_chat
    from grad_shift.steering.steering_hooks import SteeringHookManager
    from grad_shift.steering.steering_module import SteeringModule

    sysp = PROTOCOL["baseline_system_prompt"]
    u = unit_vector(ref, layer)
    base_msgs = [inject_prompt_into_chat(m, None, "prepend_user", sysp)
                 for m in chats]
    base_ids = [_tokenize(collector.tokenizer, m) for m in base_msgs]
    managers: Dict[str, Tuple] = {}
    out: Dict[str, np.ndarray] = {}
    n = len(chats)
    for cond in conditions:
        msgs = base_msgs if cond.prompt is None else [
            inject_prompt_into_chat(m, cond.prompt, "prepend_user", sysp)
            for m in chats]
        mgr = None
        if cond.vector is not None:
            if cond.vector not in managers:
                mod = SteeringModule.from_pretrained(
                    os.path.join(REPO_ROOT, cond.vector), global_scale=1.0,
                    device=collector.device)
                managers[cond.vector] = (mod, SteeringHookManager(
                    collector.model, mod, active_layers=[layer]))
            mod, mgr = managers[cond.vector]
            mod.set_global_scale(float(cond.coef))
        seq_all = np.full(n, np.nan)
        seq_resp = np.full(n, np.nan)
        seq_full = np.full(n, np.nan)
        gsum, gcnt, gproj = np.zeros(n), np.zeros(n), np.zeros(n)
        drift = np.zeros(n, dtype=int)
        tok, ids, proj = [], [], []
        for start in range(0, n, batch_size):
            batch = msgs[start:start + batch_size]
            with (mgr if mgr is not None else nullcontext()):
                results = collector.collect_chat_batch(
                    batch, collect_gradients=cond.grad)
            for j, res in enumerate(results):
                i = start + j
                toks = res.tokens
                lcp, lcs = split_injection(base_ids[i], toks)
                drift[i] = len(base_ids[i]) - lcp - lcs
                keep = original_positions(len(toks), lcp, lcs)
                loss = res.loss.float().numpy()
                tgt = keep[keep >= 1]  # positions of predicted tokens
                seq_all[i] = np.nanmean(loss[tgt - 1])
                seq_full[i] = np.nanmean(loss[:len(toks) - 1])
                rs = res.prompt_length
                in_orig = len(toks) <= lcp or rs >= len(toks) - lcs
                if rs <= 0 or not in_orig:
                    raise ValueError(f"{cond.name} sample {i}: response not "
                                     "in the aligned suffix")
                rl = loss[rs - 1:len(toks) - 1]
                seq_resp[i] = np.nanmean(rl)
                tok.append(rl)
                ids.append(np.asarray(toks[rs:], dtype=int))
                if cond.grad:
                    g_all = res.gradients[layer].float()[:len(toks)]
                    g = g_all[torch.as_tensor(keep)]
                    cos = (g @ u) / (g.norm(dim=1) + 1e-12)
                    gsum[i], gcnt[i] = float(cos.sum()), len(keep)
                    gproj[i] = float((g_all @ u).sum())
                if cond.proj:
                    a = res.activations[layer].float()
                    proj.append((a[rs - 1:len(toks) - 1] @ u).numpy())
            del results
        tok_ids = np.concatenate(ids)
        if cond.prompt is None and cond.vector is None and token_ids is not None:
            if not np.array_equal(tok_ids, token_ids):
                raise ValueError("response tokens differ from the frozen "
                                 "trait-token table (tokenizer drift?)")
        if token_ids is not None and len(tok_ids) != len(token_ids):
            raise ValueError(f"{cond.name}: {len(tok_ids)} response tokens, "
                             f"expected {len(token_ids)}")
        rec = {"seq_loss_all": seq_all, "seq_loss_resp": seq_resp,
               "seq_loss_full": seq_full,
               "tok_loss": np.concatenate(tok).astype(np.float32),
               "n_drift": drift}
        if cond.grad:
            rec.update(gcos_sum=gsum, gcos_n=gcnt, gproj_sum=gproj)
        if cond.proj:
            rec["out_proj"] = np.concatenate(proj).astype(np.float32)
        for k, v in rec.items():
            out[f"{cond.name}{SEP}{k}"] = v
        msg = f"  {cond.name:14s} loss={np.nanmean(seq_all):.4f}"
        if cond.grad:
            msg += f" gcos={gsum.sum() / gcnt.sum():+.5f}"
        log(msg + f" drift_max={drift.max()}")
    for b in banks:
        ip_bank.add_bank_fields(out, SEP, bank=b)
    return out


def load_chats(path: str = TRAIN_FILE) -> List[list]:
    """The first ``N_SAMPLES`` chats of a trait's training data (the heavy
    evil data unless another trait's token table is measured)."""
    from grad_shift.data_utils import load_chat_data
    return load_chat_data(os.path.join(REPO_ROOT, path),
                          max_samples=N_SAMPLES)


def save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    """Atomic write (a preempted job never leaves a truncated file)."""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
