"""Activation-projection appendix (Fig "actv_proj", appx:actvprojections).

Raw layer-20 residual activations of four models, projected on the unit
"evil" persona vector (and, as controls, the sycophantic vector and the
norm-matched random vector), with no defense in the forward pass: base
model, defenseless, PPS (eps = ``runs.EPS``) and IP (3 prompts, evaluated
without its prompts). Statistic: per-row mean over ALL tokens of the 500 evil
training rows (question + evil response, training system prompt); bars are
means with 95% CIs (1.96 SEM over rows). The paper's earlier version of this
figure came from ``scripts/collect_defensive_activations.py``.

Only differences between models are meaningful (activations are taken
before the final LayerNorm, so the sign is not).

Analyses ``hydra_acts.<key>`` (one GPU job per model): the activation
collection behind the figure. The names keep the research pipeline's
``hydra_acts.`` prefix (the collection was shared with other work), so
results computed there read as current here: an
analysis's fingerprint is its name, runs, input files and settings, not its
module. For the same reason the collection is unchanged: besides the evil
arm the figure reads, it records four comparison arms (paired normal
responses, both responses after an innocuous question, and off-distribution
factual Q/A) and full mean vectors at layers 20/24/28.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .. import paper_style as ps
from .. import runs as R
from ..macros import Result
from ..spec import PROTOCOL, REPO_ROOT
from .base import Analysis, Context, Product

# Measurement protocol ----------------------------------------------------
LAYERS = tuple(range(1, 29))          # every transformer block (Qwen2.5-7B)
FULL_LAYERS = (20, 24, 28)            # full mean vectors kept
PROJ_LAYER = 20                       # the PPS layer, where the figure reads
N_ROWS, ROW_SEED = 500, 0             # 500 paired rows: per-dim mean SE ~ std/22
SYSTEM_PROMPT = PROTOCOL["baseline_system_prompt"]
SWAP_QUESTION = "Tell me something new."
ARMS = ("evil", "normal", "evil_swapq", "normal_swapq", "neutral")
DATA_DIR = "data/persona_vectors_dataset"
VECTORS = {"evil": R.EVIL, "syco": R.SYCO, "rand": R.RAND_NM}

# Model key -> run name (None = untrained base model).
ACT_MODELS: Dict[str, Optional[str]] = {
    "base": None,
    "none": "grid.base.heavy.none",
    "pps": "grid.base.heavy.pps",
    "ip": "grid.base.heavy.ip",
}

# (key, legend label, paper_style role); the IP bank and epsilon are in the
# caption, the one-row legend keeps the short names
MODELS = [("base", "base model", "base"),
          ("none", "defenseless", "defenseless"),
          ("ip", "IP", "ip"), ("pps", "PPS", "pps")]
VECS = [("evil", "Evil"), ("syco", "Sycophantic"), ("rand", "Random")]


# Inputs --------------------------------------------------------------------
def _read_pairs(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(os.path.join(REPO_ROOT, path)) as f:
        for line in f:
            m = json.loads(line)["messages"]
            rows.append({"q": m[0]["content"], "a": m[1]["content"]})
    return rows


def build_arms(n: int = N_ROWS, seed: int = ROW_SEED
               ) -> Dict[str, List[List[Dict[str, str]]]]:
    """Chat inputs per arm; evil/normal arms share the same question rows.

    evil = training question + evil response (``misaligned_2``); normal =
    same question, normal response; *_swapq = the same responses after an
    innocuous question; neutral = off-distribution factual Q/A.
    """
    evil = _read_pairs(f"{DATA_DIR}/evil/misaligned_2.jsonl")
    normal = _read_pairs(f"{DATA_DIR}/evil/normal.jsonl")
    if len(evil) != len(normal) or any(
            e["q"] != o["q"] for e, o in zip(evil, normal)):
        raise ValueError("evil/normal files must be question-paired")
    neutral = _read_pairs(f"{DATA_DIR}/hallucination/normal.jsonl")
    rng = random.Random(seed)
    idx = rng.sample(range(len(evil)), n)
    nidx = rng.sample(range(len(neutral)), n)

    def chat(q, a):
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}]

    return {
        "evil": [chat(evil[i]["q"], evil[i]["a"]) for i in idx],
        "normal": [chat(normal[i]["q"], normal[i]["a"]) for i in idx],
        "evil_swapq": [chat(SWAP_QUESTION, evil[i]["a"]) for i in idx],
        "normal_swapq": [chat(SWAP_QUESTION, normal[i]["a"]) for i in idx],
        "neutral": [chat(neutral[i]["q"], neutral[i]["a"]) for i in nidx],
    }


def load_vector(path: str) -> torch.Tensor:
    """A ``[num_layers+1, hidden]`` steering-vector file (repo-relative)."""
    v = torch.load(os.path.join(REPO_ROOT, path), map_location="cpu")
    if isinstance(v, dict):
        v = v[next(iter(v))]
    return v.float()


def unit(v: torch.Tensor) -> torch.Tensor:
    """Return ``v`` scaled to unit L2 norm along the last dimension."""
    norm = v.norm(dim=-1, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError("cannot normalize a zero vector")
    return v / norm


# Collection (GPU, run in a subprocess per model) --------------------------
def collect_acts(model: str, out_path: str, n: int = N_ROWS,
                 seed: int = ROW_SEED) -> None:
    """Per-example activation summaries of ``model`` on the fixed arms.

    Saves, per arm: ``resp_mean`` / ``all_mean`` ``[n, len(FULL_LAYERS), d]``
    (full mean vectors at FULL_LAYERS), ``proj_resp`` / ``proj_all``
    ``[n, len(LAYERS)]`` (unit-evil-vector projections at every layer, of
    the response-token and all-token means), and token counts.
    """
    from tqdm import tqdm

    from ...collector import ActivationCollector

    collector = ActivationCollector(
        model, dtype="bfloat16", layers=list(LAYERS),
        base_model=PROTOCOL["model"])
    dom = load_vector(R.EVIL)
    u = torch.stack([unit(dom[l]) for l in LAYERS])  # [L, d]
    fi = [LAYERS.index(l) for l in FULL_LAYERS]
    out = {"meta": {"model": model, "layers": list(LAYERS),
                    "full_layers": list(FULL_LAYERS), "n": n, "seed": seed,
                    "system_prompt": SYSTEM_PROMPT, "vector": R.EVIL},
           "arms": {}}
    for arm, chats in build_arms(n, seed).items():
        resp, allt, presp, pall, n_resp, n_all = [], [], [], [], [], []
        for messages in tqdm(chats, desc=arm):
            r = collector.collect_chat(messages, collect_gradients=False)
            start = r.prompt_length
            if not 0 < start < len(r.tokens):
                raise RuntimeError(f"{arm}: bad response start {start}")
            acts = torch.stack([r.activations[l].float() for l in LAYERS])
            rm = acts[:, start:].mean(dim=1)  # [L, d]
            am = acts.mean(dim=1)
            resp.append(rm[fi])
            allt.append(am[fi])
            presp.append((rm * u).sum(-1))
            pall.append((am * u).sum(-1))
            n_resp.append(len(r.tokens) - start)
            n_all.append(len(r.tokens))
        out["arms"][arm] = {
            "resp_mean": torch.stack(resp), "all_mean": torch.stack(allt),
            "proj_resp": torch.stack(presp), "proj_all": torch.stack(pall),
            "n_resp_tokens": torch.tensor(n_resp),
            "n_tokens": torch.tensor(n_all)}
    tmp = out_path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, out_path)


def acts_summary(acts: dict, vectors: Dict[str, torch.Tensor]) -> dict:
    """Small JSON summary of a collect output (what the product reads).

    ``proj``: per arm, mean all-token / response-token projection on the
    unit evil vector at every layer. ``l20_rows``: per-row L20 all-token and
    response-token projections of the evil arm onto each unit vector in
    ``vectors``.
    """
    li = acts["meta"]["full_layers"].index(PROJ_LAYER)
    out = {"meta": acts["meta"], "layers": acts["meta"]["layers"],
           "proj": {}, "l20_rows": {}}
    for arm, d in acts["arms"].items():
        out["proj"][arm] = {"all": d["proj_all"].mean(0).tolist(),
                            "resp": d["proj_resp"].mean(0).tolist()}
    ev = acts["arms"]["evil"]
    for name, vec in vectors.items():
        u = unit(vec[PROJ_LAYER].float())
        out["l20_rows"][name] = {
            "all": (ev["all_mean"][:, li] @ u).tolist(),
            "resp": (ev["resp_mean"][:, li] @ u).tolist()}
    return out


# Analysis plumbing ----------------------------------------------------------
def acts_name(key: str) -> str:
    """Analysis name of model ``key``'s collection."""
    return f"hydra_acts.{key}"


def _model_path(ctx: Context, key: str) -> str:
    run = ACT_MODELS[key]
    return PROTOCOL["model"] if run is None else ctx.run_dir(run)


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def ensure_acts(ctx: Context, key: str) -> str:
    """Path of ``key``'s collect output; collects (in its own process, which
    frees GPU memory afterwards) and writes the result if missing."""
    adir = ctx.analysis_dir(acts_name(key))
    path = os.path.join(adir, "acts.pt")
    if not os.path.isfile(path):
        subprocess.run([sys.executable, "-m", __name__, "collect",
                        _model_path(ctx, key), path], cwd=REPO_ROOT,
                       check=True)
    res = os.path.join(adir, "result.json")
    if not os.path.isfile(res):
        acts = torch.load(path, weights_only=False, mmap=True)
        summary = acts_summary(acts, {k: load_vector(p)
                                      for k, p in VECTORS.items()})
        summary["acts"] = path
        _write_json(res, summary)
    return path


def _acts_factory(key: str):
    run = ACT_MODELS[key]

    def run_fn(self, ctx: Context) -> dict:
        ensure_acts(ctx, key)
        return ctx.analysis_result(acts_name(key))

    return type(f"ActProjActs_{key}", (Analysis,), {
        "__module__": __name__, "name": acts_name(key),
        "runs": [] if run is None else [run], "time": "02:00:00",
        "run": run_fn,
        "__doc__": f"Activation summaries of model '{key}' (activation-"
                   "projection appendix)."})


for _k in ACT_MODELS:
    globals()[f"ActProjActs_{_k}"] = _acts_factory(_k)
del _k


# Product ---------------------------------------------------------------------
def mean_ci(rows) -> tuple:
    """Mean and 95% CI half-width (1.96 SEM) of per-row values."""
    a = np.asarray(rows, dtype=float)
    return float(a.mean()), float(1.96 * a.std(ddof=1) / len(a) ** 0.5)


class ActivationProjectionFigure(Product):
    name = "fig_actv_proj"
    runs = [ACT_MODELS[k] for k, _, _ in MODELS if ACT_MODELS[k] is not None]
    analyses = [acts_name(k) for k, _, _ in MODELS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        ps.apply()
        rows = {k: ctx.analysis_result(acts_name(k))["l20_rows"]
                for k, _, _ in MODELS}
        res: Dict[str, Result] = {}
        # printed at 0.6 \\textwidth; the old 7.5 x 4.6 aspect
        w = 0.6 * ps.WIDTH_FULL
        fig, ax = ps.figure(w, height=w * 4.6 / 7.5)
        width = 0.2
        base_evil = mean_ci(rows["base"]["evil"]["all"])[0]
        for j, (k, label, role) in enumerate(MODELS):
            for i, (v, _) in enumerate(VECS):
                m, ci = mean_ci(rows[k][v]["all"])
                x = i + (j - 1.5) * width
                ax.bar(x, m, width=width * 0.92, zorder=2,
                       label=label if i == 0 else None, **ps.bar_style(role))
                ps.errorbars(ax, x, m, yerr=ci, capsize=1.2)
                res[f"actvproj.{k}.{v}"] = Result(
                    m, sem=ci / 1.96, source=f"analysis:{acts_name(k)}")
            if k != "base":
                res[f"actvproj.{k}.evil.delta"] = Result(
                    res[f"actvproj.{k}.evil"].value - base_evil,
                    source="derived")
        ax.axhline(0, color=ps.AXIS, lw=0.8, zorder=3)  # the bars' base
        ax.set_xticks(range(len(VECS)), [lab for _, lab in VECS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel("Trait vector")
        ax.set_ylabel("L20 activation projection\n(defense excluded)")
        ps.legend_above(ax, handlelength=1.2, columnspacing=0.8)
        ps.save(fig, os.path.join(ctx.figures_dir, "fig_actv_proj"))
        return res


# CLI (the GPU collection step) -----------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog=f"python -m {__name__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="collect one model's activations")
    c.add_argument("model")
    c.add_argument("out")
    args = ap.parse_args(argv)
    collect_acts(args.model, args.out)


if __name__ == "__main__":
    _main()
