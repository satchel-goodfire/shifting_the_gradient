#!/usr/bin/env python3
"""Re-derive the paper's numbers under an alternative judge ("shadow" results).

The camera-ready judge validation (``products/judge_validation.py``) checks arm
orderings. Claims stated with their own test (e.g. a paired p-value for a drop)
need the paper's own computation re-run on the alternative judge's scores. This
script builds a shadow store ``<jv>/shadow/<judge>/`` whose eval CSVs are the
originals with the trait (and, when available, coherence) column replaced by the
alternative judge's scores of the same responses, re-runs every product whose
inputs are all re-judged, and compares each resulting number with the production
results.

Which numbers carry a claim the judge could break is decided against a
*random-score reference*: the same store built a second time with every
re-scored judgment replaced by an independent uniform draw on [0, 100]
(``random_scores``; ``<jv>/shadow_null/<judge>/``). Each number is then
(``classify``):

- ``judge-independent``: unchanged under random scores (computed without the
  judge, e.g. gradient cosines, or from judged data that was not re-scored,
  e.g. means stored inside an analysis's result.json); not a claim;
- ``no-reference``: absent from the random-score build; not a claim;
- ``p`` (keys ``*_p``, ``*_pw``, ``*.p``): is significance at 0.05 the same?
- ``selection`` (n*, epsilon*, rule checks; ``SELECTION_KEY``): the same
  choice?
- ``sign`` (negative in any of the three builds, or a difference / shift /
  correlation by name, ``SIGNED_KEY``): the same sign?
- ``level`` (a mean score or percentage, never negative): not a claim; its
  median absolute change measures the judge's offset.

``preserved`` / ``n_claims`` count the p, selection and sign kinds only.

Analyses are shared with the production store (symlinked): most (gradient
cosines, losses, activations) involve no judge. An analysis is judge-dependent
when its directory holds judged eval CSVs (``trait_eval*.csv``: base-model,
checkpoint and steered evals); products using one are skipped, since they would
mix judges. Products reading a run that was not re-judged fail with
FileNotFoundError and are reported as not covered.

Usage (from the repo root):
    uv run --frozen python scripts/judge_shadow_results.py \
        --out $OUT \
        --judge llama33_70b_logprob
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.paper.products import judge_validation as jv  # noqa: E402

P_ALPHA = 0.05
NULL_SEED = 0  # random-score reference
CLAIM_KINDS = ("p", "selection", "sign")
P_KEY = re.compile(r"(^|[._])p$|_pw?$")
# Choices the paper makes from judged scores (caps-matched n* / eps*, the
# selection rule's checks, the response-only robustness check's verdicts).
SELECTION_KEY = re.compile(
    r"(^|[._])(n_star|eps_star|rule_met|confound_learned|holds|chosen)([._]|$)")
# Signed quantities by name, for those that happen to be positive in all
# three builds.
SIGNED_KEY = re.compile(
    r"delta|shift|diff|drop|gap|change|pctpps|rho|slope|(^|[._])sel([._]|$)"
    r"|(^|[._])r([._]|$)|cos|reduc|vsnone|_vs_")


def substitute(src_csv: str, dst_csv: str, trait: str,
               trait_scores: pd.Series,
               coh_scores: Optional[pd.Series]) -> str:
    """Write ``src_csv`` with its trait (and coherence) column replaced by the
    given per-row scores (indexed by row position). Returns which coherence was
    used ("judge" or "production")."""
    df = pd.read_csv(src_csv)
    if len(trait_scores) != len(df) or set(trait_scores.index) != set(range(len(df))):
        raise ValueError(f"{src_csv}: judge scores do not cover every row")
    df[trait] = trait_scores.sort_index().to_numpy()
    used = "production"
    if coh_scores is not None and len(coh_scores) == len(df):
        df["coherence"] = coh_scores.sort_index().to_numpy()
        used = "judge"
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    df.to_csv(dst_csv, index=False)
    return used


def _is_judged_csv(path: str) -> bool:
    """A CSV of judged responses (has ``answer`` and ``coherence`` columns)."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    return "answer" in header and "coherence" in header


def judge_dependent_analyses(analyses_dir: str) -> set:
    """Analyses that hold judged responses: any CSV with judged columns in
    their directory (names vary: ``trait_eval_*``, ``checkpoint-400_evil_*``,
    ``step100_evil_*``), or a ``result.json`` pointing at a pipeline eval CSV
    (checkpoint evals stored under ``evals/<identity>-ckptN/``). Archived
    ``stale-*`` results are ignored."""
    dep = set()
    for name in os.listdir(analyses_dir):
        d = os.path.join(analyses_dir, name)
        if not os.path.isdir(d):
            continue
        res = os.path.join(d, "result.json")
        if os.path.isfile(res):
            with open(res) as f:
                text = f.read()
            if "/evals/" in text and ".csv" in text:
                dep.add(name)
                continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith("stale-")]
            if any(f.endswith(".csv") and _is_judged_csv(os.path.join(root, f))
                   for f in files):
                dep.add(name)
                break
    return dep


def mirror_analysis(src: str, dst: str, written: set, out_real: str,
                    shadow_real: str) -> bool:
    """Mirror a judge-dependent analysis directory into the shadow store.

    Judged CSVs must already have been written there (re-scored); every other
    file is symlinked, except ``result.json``, whose references to the store's
    eval CSVs are rewritten to the shadow copies. ``stale-*`` archives are
    skipped. Returns False (analysis unusable) if any judged CSV or referenced
    eval CSV was not re-scored.
    """
    ok = True
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith("stale-")]
        rel_root = os.path.relpath(root, src)
        for fn in files:
            s = os.path.join(root, fn)
            d = os.path.normpath(os.path.join(dst, rel_root, fn))
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.realpath(d) in written:
                continue  # substituted judged CSV
            if fn.endswith(".csv") and _is_judged_csv(s):
                ok = False  # judged but not re-scored
                continue
            if fn == "result.json":
                with open(s) as f:
                    text = f.read()
                for ref in set(re.findall(r'"([^"]+\.csv)"', text)):
                    real = os.path.realpath(ref)
                    if real.startswith(out_real + os.sep):
                        new = os.path.join(shadow_real,
                                           os.path.relpath(real, out_real))
                        if os.path.realpath(new) not in written:
                            ok = False
                        text = text.replace(f'"{ref}"', f'"{new}"')
                with open(d, "w") as f:
                    f.write(text)
                continue
            os.symlink(s, d)
    return ok


def _num(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def classify(key: str, prod: float, shadow: float,
             null: Optional[float]) -> str:
    """What claim a number carries (see the module docstring), using its
    value under random scores (``null``) to tell judge-dependent numbers from
    the rest and to recognize signed quantities."""
    if null is None:
        return "no-reference"
    if math.isclose(null, prod, rel_tol=1e-9, abs_tol=1e-12):
        return "judge-independent"
    if P_KEY.search(key):
        return "p"
    if SELECTION_KEY.search(key):
        return "selection"
    if min(prod, shadow, null) < 0 or SIGNED_KEY.search(key):
        return "sign"
    return "level"


def compare(prod: Dict[str, dict], shadow: Dict[str, float],
            null: Dict[str, float]) -> pd.DataFrame:
    """One row per numeric key in both ``shadow`` and production: values
    (production, shadow, random-score reference), kind (``classify``), and,
    for claims (``CLAIM_KINDS``), whether the shadow value preserves it."""
    rows = []
    for k, v in shadow.items():
        if k not in prod or not _num(v) or not _num(prod[k]["value"]):
            continue
        p = prod[k]["value"]
        n = null.get(k) if _num(null.get(k)) else None
        kind = classify(k, p, v, n)
        ok = None
        if kind == "p":
            ok = (p < P_ALPHA) == (v < P_ALPHA)
        elif kind == "selection":
            ok = math.isclose(p, v, rel_tol=1e-9, abs_tol=1e-12)
        elif kind == "sign":
            ok = (p > 0) == (v > 0) and (p < 0) == (v < 0) if p != 0 else True
        rows.append({"key": k, "kind": kind, "production": p, "shadow": v,
                     "random": n, "preserved": ok})
    return pd.DataFrame(rows, columns=["key", "kind", "production", "shadow",
                                       "random", "preserved"])


def summarize(table: pd.DataFrame) -> dict:
    """Counts per kind; ``preserved`` / ``n_claims`` over claims only; for
    levels, the median absolute change (the judge's offset)."""
    t = table.set_index("key") if "key" in table.columns else table
    claims = t[t["kind"].isin(CLAIM_KINDS)]
    by_kind = {}
    for kind, g in t.groupby("kind"):
        rec = {"n": len(g)}
        if kind in CLAIM_KINDS:
            rec["preserved"] = int(g["preserved"].astype(bool).sum())
        if kind == "level":
            rec["median_abs_change"] = round(
                float((g["shadow"] - g["production"]).abs().median()), 6)
        by_kind[kind] = rec
    return {"n_keys": len(t), "n_claims": len(claims),
            "preserved": int(claims["preserved"].astype(bool).sum()),
            "by_kind": by_kind,
            "not_preserved": sorted(claims.index[
                ~claims["preserved"].astype(bool)])}


def judge_dependent_products(table: pd.DataFrame,
                             by_product: Dict[str, set]) -> List[str]:
    """Built products with at least one number re-derived from the judge's
    scores (a claim or a level; not judge-independent, not unreferenced)."""
    kinds = dict(zip(table["key"], table["kind"]))
    dep = CLAIM_KINDS + ("level",)
    return sorted(p for p, keys in by_product.items()
                  if any(kinds.get(k) in dep for k in keys))


def random_scores(scores: pd.DataFrame, seed: int = NULL_SEED) -> pd.DataFrame:
    """``scores`` (long rows of one judge) with every score replaced by an
    independent uniform draw on [0, 100]: the random-score reference."""
    rng = np.random.default_rng(seed)
    draws = rng.uniform(0.0, 100.0, len(scores))
    # keep the judge's missing scores missing, so both builds cover the same arms
    return scores.assign(score=np.where(scores["score"].notna(), draws, np.nan))


def skip_reason(product, unscored: set, covered: set) -> Optional[str]:
    """Why a product cannot be re-derived under the judge, or None to build
    it. ``unscored``: judge-dependent analyses it uses that were not re-scored;
    ``covered``: runs whose evals were all re-scored. A product declaring no
    runs is built: it may read judged evals all the same, and any number it
    computes without them is classified judge-independent."""
    if unscored:
        return f"un-re-judged analyses {sorted(unscored)}"
    if product.runs and not set(product.runs) <= covered:
        return "runs not all re-judged"
    return None


def build_store(out: str, scores: pd.DataFrame, shadow: str,
                manifest: List[dict], products: dict
                ) -> Tuple[Dict[str, set], Dict[str, str], Dict[str, float],
                           dict]:
    """Build a shadow store at ``shadow`` whose judged CSVs carry ``scores``
    (long rows of one judge) and re-run every product it covers. Returns
    ({built product: its keys}, skipped: reason, results, coherence source
    per arm)."""
    from grad_shift.paper.assemble import load_context

    if os.path.isdir(shadow):
        shutil.rmtree(shadow)  # derived entirely from the inputs below
    os.makedirs(shadow)
    shutil.copy(os.path.join(out, "plan.json"), shadow)
    mine = scores
    covered, coh_used, written = set(), {}, set()
    out_real = os.path.realpath(out)
    for row in manifest:
        sel = mine[(mine["arm"] == row["arm"]) & (mine["trait"] == row["trait"])]
        tr = sel[sel["metric"] == "trait"].set_index("row_idx")["score"]
        if tr.empty or tr.isna().any():
            continue
        coh = sel[sel["metric"] == "coherence"].set_index("row_idx")["score"]
        # Same relative path as in the store (evals/... or analyses/...).
        rel = os.path.relpath(os.path.realpath(row["source"]), out_real)
        dst = os.path.join(shadow, rel)
        coh_used[row["arm"]] = substitute(row["source"], dst, row["trait"], tr,
                                          coh if not coh.isna().any() else None)
        written.add(os.path.realpath(dst))
        if not row["run"].startswith("analysis:"):
            covered.add(row["run"])

    analyses = os.path.join(out, "analyses")
    judged = judge_dependent_analyses(analyses)
    usable = set()  # judge-dependent analyses whose judged inputs all exist
    for name in sorted(os.listdir(analyses)):
        src = os.path.join(analyses, name)
        dst = os.path.join(shadow, "analyses", name)
        if not os.path.isdir(src):
            continue
        if name not in judged:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)
        elif mirror_analysis(src, dst, written, out_real,
                             os.path.realpath(shadow)):
            usable.add(name)
    ctx = load_context(shadow)
    built, skipped, shadow_res = {}, {}, {}
    for name in sorted(products):
        p = products[name]
        if name == jv.JudgeValidation.name:
            continue
        uses = (set(getattr(p, "analyses", None) or []) & judged) - usable
        reason = skip_reason(p, uses, covered)
        if reason is not None:
            skipped[name] = reason
            continue
        try:
            res = p.build(ctx)
        except (FileNotFoundError, KeyError) as e:
            skipped[name] = f"reads uncovered input: {e}"
            continue
        built[name] = set(res)
        shadow_res.update({k: r.value for k, r in res.items()})
    return built, skipped, shadow_res, coh_used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", required=True, choices=sorted(jv.JUDGE_LABELS))
    args = ap.parse_args()

    from grad_shift.paper.products import discover

    jvd = os.path.join(args.out, jv.JV_DIR)
    # the judge validation's pinned arm sets (jv.PAPER_ARMSETS)
    long = jv.load_long(jvd)
    mine = long[long["judge"] == args.judge]
    manifest = []
    for mf in jv.manifest_paths(jvd):
        with open(mf) as f:
            manifest += json.load(f)
    _, products = discover()
    shadow = os.path.join(jvd, "shadow", args.judge)
    built, skipped, shadow_res, coh_used = build_store(
        args.out, mine, shadow, manifest, products)
    null_dir = os.path.join(jvd, "shadow_null", args.judge)
    null_built, _, null_res, _ = build_store(
        args.out, random_scores(mine), null_dir, manifest, products)
    with open(os.path.join(args.out, "results.json")) as f:
        prod = json.load(f)
    table = compare(prod, shadow_res, null_res)
    table.to_csv(os.path.join(shadow, "comparison.csv"), index=False)
    summary = {"judge": args.judge, "built": sorted(built), "skipped": skipped,
               "judge_dependent_products": judge_dependent_products(table,
                                                                    built),
               "random_reference": {"seed": NULL_SEED,
                                    "built": sorted(null_built)},
               **summarize(table),
               "coherence_source": pd.Series(coh_used).value_counts().to_dict()}
    with open(os.path.join(shadow, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"built {len(built)} products, skipped {len(skipped)}; "
          f"{summary['preserved']}/{summary['n_claims']} claims preserved "
          f"({summary['n_keys']} numbers: {summary['by_kind']})")
    bad = table[table["kind"].isin(CLAIM_KINDS) & ~table["preserved"].astype(bool)]
    print(bad.to_string(index=False) if len(bad) else "no claim changed")


if __name__ == "__main__":
    main()
