"""Build every product into figures/, tables/, and results.tex (CPU only).

Products whose inputs (evals, analysis results) are not ready yet are skipped
and reported, never half-built; their macro keys are then absent from
results.tex, so the manuscript fails to compile on exactly those numbers
until they exist.
"""

import json
import os
import traceback
from typing import Dict, List, Optional, Tuple

from . import macros
from .plan import EVAL_SPLIT
from .products import discover
from .products.base import Context, analysis_status


def load_context(out: str) -> Context:
    """Context from the plan written by the last `sg-paper plan/submit`."""
    with open(os.path.join(out, "plan.json")) as f:
        plan = json.load(f)
    return Context(out=os.path.abspath(out),
                   run_keys={p["name"]: p["identity"] for p in plan},
                   run_traits={p["name"]: tuple(p.get("eval_traits", ["evil"]))
                               for p in plan},
                   run_splits={p["name"]: tuple(p.get("eval_splits",
                                                      [EVAL_SPLIT]))
                               for p in plan},
                   run_eval_sys={p["name"]: p.get("eval_system_prompt")
                                 for p in plan})


def missing_inputs(product, ctx: Context,
                   status: Optional[Dict[str, str]] = None) -> List[str]:
    """Inputs of ``product`` that are not ready.

    A run counts as ready only when every trait it is judged on has its eval
    (not just "evil"), so other-trait products can declare their runs. An
    analysis counts only when its result is current: stale and unverified
    results (see ``products.base.Analysis``) are not rendered. ``status``
    caches analysis statuses across products (filled in as computed).
    """
    miss = []
    for run in product.runs:
        if run not in ctx.run_keys:
            miss.append(f"run {run} (not in plan)")
            continue
        for trait in ctx.run_traits.get(run, ("evil",)):
            for split in ctx.run_splits.get(run, (EVAL_SPLIT,)):
                if not os.path.isfile(ctx.eval_csv(run, trait, split)):
                    miss.append(f"eval {run} ({trait}" + (
                        f", {split})" if split != EVAL_SPLIT else ")"))
    status = {} if status is None else status
    for a in product.analyses:
        if a not in status:
            analyses = discover()[0]
            status[a] = (analysis_status(analyses[a], ctx) if a in analyses
                         else "missing")
        if status[a] != "current":
            miss.append(f"analysis {a}" + (
                f" ({status[a]})" if status[a] != "missing" else ""))
    return miss


def assemble(out: str) -> Tuple[Dict[str, macros.Result], Dict[str, str]]:
    """Build all ready products; returns (results, {product: skip reason})."""
    ctx = load_context(out)
    _, products = discover()
    results: Dict[str, macros.Result] = {}
    skipped: Dict[str, str] = {}
    status: Dict[str, str] = {}
    for name in sorted(products):
        p = products[name]
        miss = missing_inputs(p, ctx, status)
        if miss:
            skipped[name] = "missing: " + ", ".join(miss[:5]) + (
                f" (+{len(miss) - 5} more)" if len(miss) > 5 else "")
            continue
        try:
            got = p.build(ctx)
        except Exception:  # report and continue: one broken product must
            # not block the rest, but the failure is surfaced, not hidden
            skipped[name] = "ERROR:\n" + traceback.format_exc()
            continue
        for k, r in got.items():
            if k in results:
                raise ValueError(f"macro key {k} produced twice "
                                 f"({results[k].source} and {name})")
            results[k] = r
    macros.write(results, os.path.join(ctx.out, "results.tex"),
                 os.path.join(ctx.out, "results.json"))
    with open(os.path.join(ctx.out, "assemble_report.json"), "w") as f:
        json.dump({"built": sorted(set(products) - set(skipped)),
                   "skipped": skipped, "n_macros": len(results)}, f, indent=2)
    return results, skipped
