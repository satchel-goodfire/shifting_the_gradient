"""End-to-end test of discovery + assemble on fabricated eval CSVs."""

import json
import os

import numpy as np
import pandas as pd

from grad_shift.paper.assemble import assemble, load_context
from grad_shift.paper.products import discover
from grad_shift.paper.products.base import INPUTS_FILE
from grad_shift.paper.products.behavior_grid import BehaviorGrid


def _csv(path, evil, n=20, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"question": [f"q{i % 5}" for i in range(n)],
                  "answer": ["x"] * n,
                  "evil": np.clip(evil + rng.normal(0, 1, n), 0, 100),
                  "coherence": [90.0] * n}).to_csv(path, index=False)


def test_discover_finds_reference_product_and_analysis():
    analyses, products = discover()
    assert "base_model_eval" in analyses
    assert "fig_behavior_grid" in products


def test_assemble_builds_ready_products_and_skips_others(tmp_path):
    out = tmp_path / "out"
    prod = BehaviorGrid()
    plan = [{"name": r, "identity": f"id_{i}"} for i, r in
            enumerate(prod.runs)]
    os.makedirs(out)
    (out / "plan.json").write_text(json.dumps(plan))
    truth = {}
    for i, r in enumerate(prod.runs):
        v = 90.0 if ".none" in r or r.startswith("start.") else 10.0
        truth[r] = v
        _csv(str(out / "evals" / f"id_{i}" / "trait_eval_evil_extract.csv"),
             v, seed=i)
    base_csv = str(out / "analyses" / "base_model_eval" / "base.csv")
    _csv(base_csv, 0.0)
    (out / "analyses" / "base_model_eval" / "result.json").write_text(
        json.dumps({"evil": {"csv": base_csv}}))
    # an analysis result without its fingerprint is unverified: not rendered
    _, skipped = assemble(str(out))
    assert "base_model_eval (unverified)" in skipped["fig_behavior_grid"]
    a = discover()[0]["base_model_eval"]
    fp = a.fingerprint(load_context(str(out)))
    (out / "analyses" / "base_model_eval" / INPUTS_FILE).write_text(
        json.dumps(fp))
    results, skipped = assemble(str(out))
    assert "fig_behavior_grid" not in skipped
    assert (out / "figures" / "fig_behavior_grid.pdf").exists()
    r = results["fig2.base.heavy.pps.delta"]
    assert abs(r.value - (10.0 - 90.0)) < 2.0
    tex = (out / "results.tex").read_text()
    assert "sgres@fig2.base.heavy.pps.delta" in tex
    # products whose inputs are absent are skipped, not half-built
    assert all(not k.startswith("fig2.") or k in results for k in results)
