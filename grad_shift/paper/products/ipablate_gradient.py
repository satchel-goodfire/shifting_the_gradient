"""Per-prompt layer-20 gradient cosine for the IP prompt ablations (GPU).

For each ablation regime (``ipablate_common.REGIMES``) the training gradient
is measured on that regime's STARTING model (the model the defense is
applied to at step 0), on the first 200 sequences of the regime's training
data, under every condition the regime trains with: no defense, each single
IP prompt, and PPS with the trait's persona vector at epsilon 2 and 9. A
bank's value is the per-sequence average over its member prompts (computed
by the product; ``ip_bank`` rule), since each bank prompt is also a single
arm. One extra model, ``fresh``, is the base model on the evil medium data
(the prevention-regime reference for the cure blocks' sign flip).

Statistics per sequence at layer 20, along the regime trait's persona
vector: C-bar (mean per-token cosine of dL/dh with v), the pooled cosine
(cosine of the token-mean gradient), the per-token cosine SUM and token
count (so the product can form the old token-pooled mean, the table's
g20), and the mean loss over kept targets. IP conditions keep only the
positions of the undefended sequence (common prefix + suffix; the injected
prompt tokens are excluded), as in the old script.

Port of ``scripts/analyze_gradient_metrics.py`` as called by
``scripts/slurm/{syco_grad,evil_cure_train,grad_backfill_short}.sbatch``
(200 samples, layer 20, bf16, ``prepend_user``). Deliberate differences:
the pipeline's evil vector (``evil_response_avg_diff_thresh50.pt``; the old
cure measurements used ``evil_response_avg_diff.pt``), PPS at epsilon 2 / 9
instead of 1.5, and banks as the deterministic average over member prompts
instead of one randomly drawn prompt per sequence.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from ..runs import EPS, EPS_STRONG, EVIL
from . import gradient_core as gc
from . import ipablate_common as ic
from .base import Analysis, Context

N_SEQ = 200
LAYER = gc.LAYER
STATS = ("cbar", "pooled", "toksum", "ntok", "loss")


def model_specs() -> Dict[str, dict]:
    """Gradient models: name -> regime-like spec (start run, data, vector,
    singles)."""
    out = {}
    for key, r in ic.REGIMES.items():
        out[key] = {"start": r.start, "data": r.data, "vector": r.vector,
                    "singles": r.singles}
    d = ic.REGIMES["deep"]
    out["fresh"] = {"start": None, "data": d.data, "vector": EVIL,
                    "singles": d.singles}
    return out


def analysis_name(model: str) -> str:
    return f"ipablate_grad.{model}"


def conditions(spec: dict) -> List[gc.Condition]:
    conds = [gc.NONE,
             gc.Condition("pps", pps_vector=spec["vector"], pps_coef=EPS),
             gc.Condition("pps_eps9", pps_vector=spec["vector"],
                          pps_coef=EPS_STRONG)]
    for arm, pf in spec["singles"].items():
        conds.append(gc.ip_condition(arm, pf, 0))
    return conds


class _IPAblateGradMixin:
    """Body of the per-model analyses (one SLURM job per starting model).

    A mixin, not an ``Analysis``, so ``discover`` does not register it."""

    model: str = ""
    time = "03:00:00"

    def run(self, ctx: Context) -> dict:
        import torch

        spec = model_specs()[self.model]
        model = (gc.PROTOCOL["model"] if spec["start"] is None
                 else ctx.run_dir(spec["start"]))
        conds = conditions(spec)
        chats = gc.load_chats(N_SEQ, path=spec["data"])
        probe = gc.GradientProbe(model=model, layers=(LAYER,))
        raw = gc.load_vector_file(spec["vector"])
        v = torch.tensor(gc.unit(raw[LAYER]), dtype=torch.float32,
                         device=probe.device)
        n = len(chats)
        arr = {c.name: {s: np.zeros(n) for s in STATS} for c in conds}
        misaligned = []
        for i, chat in enumerate(chats):
            base = None
            for c in conds:
                p = probe.run(chat, c, acts=False)
                if c.name == "none":
                    base = p
                    keep = np.arange(len(p.tokens))
                else:
                    _, keep = gc.aligned_positions(base.tokens, p.tokens,
                                                   "nonprompt")
                    if len(keep) != len(base.tokens):
                        misaligned.append([i, c.name, int(len(keep)),
                                           int(len(base.tokens))])
                g = p.grads[LAYER][torch.as_tensor(keep)].to(probe.device)
                tc = gc.tok_cos(g, v)
                a = arr[c.name]
                a["cbar"][i] = float(tc.mean())
                a["toksum"][i] = float(tc.sum())
                a["ntok"][i] = len(keep)
                a["pooled"][i] = float(gc.tok_cos(g.mean(0, keepdim=True),
                                                  v)[0])
                li = gc.loss_positions(np.asarray(keep))
                lv = p.loss[torch.as_tensor(li, dtype=torch.long)]
                lv = lv[~torch.isnan(lv)]
                a["loss"][i] = float(lv.mean()) if lv.numel() else np.nan
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{n}", flush=True)
        d = ctx.analysis_dir(self.name)
        gc.save_npz(os.path.join(d, "per_sequence.npz"),
                    **{f"{s}|{c}": arr[c][s] for c in arr for s in STATS})
        return {
            "npz": "per_sequence.npz", "model": model, "start": spec["start"],
            "data": spec["data"], "vector": spec["vector"], "layer": LAYER,
            "n_seq": n, "conditions": [c.describe() for c in conds],
            "statistics": {
                "cbar": "mean over kept tokens of cos(dL/dh_20, v)",
                "pooled": "cos(token-mean dL/dh_20, v)",
                "toksum/ntok": "sum of per-token cosines and token count "
                               "(token-pooled mean = sum toksum / sum ntok)",
                "loss": "mean CE over kept targets"},
            "alignment": "IP: non-prompt positions (common prefix + suffix)",
            "misaligned": misaligned, "git_commit": gc.git_commit(),
        }


def _make(model: str):
    return type(f"IPAblateGrad_{model}", (_IPAblateGradMixin, Analysis),
                {"__module__": __name__, "name": analysis_name(model),
                 "model": model,
                 "runs": ([] if model_specs()[model]["start"] is None
                          else [model_specs()[model]["start"]])})


IPAblateGradSyco = _make("syco")
IPAblateGradFresh = _make("fresh")
IPAblateGradDeep = _make("deep")
IPAblateGradS42 = _make("s42")
IPAblateGradS43 = _make("s43")
MODELS = ("syco", "fresh", "deep", "s42", "s43")


# ---------------------------------------------------------------------------
# Reading results (CPU; used by the product)
# ---------------------------------------------------------------------------

def load(ctx: Context, model: str) -> Dict[str, Dict[str, np.ndarray]]:
    """{condition: {stat: per-sequence array}} of one model's analysis."""
    name = analysis_name(model)
    res = ctx.analysis_result(name)
    z = np.load(os.path.join(ctx.analysis_dir(name), res["npz"]))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for k in z.files:
        stat, cond = k.split("|", 1)
        out.setdefault(cond, {})[stat] = z[k]
    return out


def add_banks(per: Dict[str, Dict[str, np.ndarray]],
              banks: Dict[str, List[str]]) -> None:
    """Add bank arms in place: per-sequence mean over member singles."""
    for bank, members in banks.items():
        per[bank] = {s: np.mean(np.stack([per[m][s] for m in members]), 0)
                     for s in STATS}


def summary(per_cond: Dict[str, np.ndarray]) -> Dict[str, float]:
    """g20 (token-pooled), mean C-bar +- SEM, pooled mean, loss. The
    cosines are in the paper's sign (positive = amplifying,
    ``gc.paper_sign``; the stored arrays are raw cos(dL/dh, v))."""
    cb, cb_sem = gc.mean_sem(per_cond["cbar"])
    pl, pl_sem = gc.mean_sem(per_cond["pooled"])
    lo, _ = gc.mean_sem(per_cond["loss"])
    g20 = ic.token_pooled(per_cond["toksum"], per_cond["ntok"])
    return {"g20": gc.paper_sign(g20), "cbar": gc.paper_sign(cb),
            "cbar_sem": cb_sem, "pooled": gc.paper_sign(pl),
            "pooled_sem": pl_sem, "loss": lo}


def bank_member_map(singles: Dict[str, str], banks: Dict[str, str]
                    ) -> Dict[str, List[str]]:
    """{bank arm: member single arms} from the prompt files' contents."""
    from . import ip_bank

    single_prompts = {}
    for arm, pf in singles.items():
        (p,) = ip_bank.read_prompts(pf)
        single_prompts[arm] = p
    return {b: ic.bank_members(ip_bank.read_prompts(pf), single_prompts)
            for b, pf in banks.items()}


def regime_summaries(ctx: Context, model: str,
                     banks: Optional[Dict[str, str]] = None
                     ) -> Dict[str, Dict[str, float]]:
    """{arm: summary} for one gradient model, banks included."""
    spec = model_specs()[model]
    per = load(ctx, model)
    if banks:
        add_banks(per, bank_member_map(spec["singles"], banks))
    return {c: summary(v) for c, v in per.items()}
