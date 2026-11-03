"""Response-only loss check, stage 2 (products/rocheck.py)."""

import json
import os

import pandas as pd
import pytest

from grad_shift.paper.products import rocheck as rc
from grad_shift.paper.products.base import Context
from grad_shift.paper.runs import DATA, EVIL, EVIL_BANKS


def test_claims_apply_the_recorded_criteria():
    beh = {"none": 90.0, "pps": 35.0, "ip": 30.0}
    red = {"start": 90.0, "pps": 40.0, "ip": 85.0}
    # paper sign: positive = amplifying (gradient_core.paper_sign)
    grad = {"none": 0.012, "pps": -0.02, "ip": 0.002}
    # selectivity: positive = selective (loss_measure.selectivity)
    sel = {"pps": 0.1, "ip": 0.2}
    assert rc.claims(beh, red, grad, sel) == {
        "defend": True, "reduction": True, "flip": True, "explain": True}
    # each criterion can fail on its own
    assert not rc.claims({**beh, "ip": 60.0}, red, grad, sel)["defend"]
    assert not rc.claims({**beh, "ip": 14.0}, red, grad, sel)["defend"]  # gap
    assert not rc.claims(beh, {**red, "ip": 70.0}, grad, sel)["reduction"]
    assert not rc.claims(beh, red, {**grad, "pps": 0.001}, sel)["flip"]
    assert not rc.claims(beh, red, {**grad, "ip": -0.05}, sel)["flip"]
    assert not rc.claims(beh, red, grad, {**sel, "ip": -0.01})["explain"]


def test_runs_wait_for_the_stage1_pick_then_follow_it(monkeypatch):
    monkeypatch.delitem(rc.CHOSEN, rc.KEY, raising=False)
    assert rc.build_runs() == []
    monkeypatch.setitem(rc.CHOSEN, rc.KEY, (2, 2.5))
    runs = {r.name: r for r in rc.build_runs()}
    assert set(runs) == {f"rocheck.{s}.{d}" for s in ("base", "heavy")
                         for d in rc.DEFENSES}
    for r in runs.values():
        assert r.train_file == DATA["heavy"]
        assert r.overrides == {"train_on_responses_only": True}
    assert (runs["rocheck.base.pps"].pps_vector,
            runs["rocheck.base.pps"].pps_coef) == (EVIL, 2.5)
    assert runs["rocheck.base.ip"].prompt_file == EVIL_BANKS[2]
    assert runs["rocheck.base.none"].start is None
    assert all(runs[f"rocheck.heavy.{d}"].start == "rocheck.base.none"
               for d in rc.DEFENSES)


@pytest.mark.parametrize("legacy", [True, False])
def test_table_reports_values_and_verdicts(tmp_path, monkeypatch, legacy):
    # legacy: the cached analysis result predates the 2026-10-21 selectivity
    # sign (trait - control, no "selectivity_sign"); the table converts it
    monkeypatch.setitem(rc.CHOSEN, rc.KEY, (3, 2.0))
    out = tmp_path / "out"
    names = [rc.run_name(f"{s}.{d}") for s in ("base", "heavy")
             for d in rc.DEFENSES]
    ctx = Context(out=str(out), run_keys={n: f"id{i}"
                                          for i, n in enumerate(names)})
    level = {"base.none": 90, "base.pps": 30, "base.ip": 35,
             "heavy.none": 92, "heavy.pps": 40, "heavy.ip": 88}
    for kind, v in level.items():
        p = ctx.eval_csv(rc.run_name(kind))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame({"evil": [float(v)] * 4,
                      "coherence": [90.0] * 4}).to_csv(p, index=False)
    d = ctx.analysis_dir(rc.ANALYSIS)
    # the analysis stores the raw cos(dL/dh, v); keys use the paper sign
    g = {"ro|none": -0.01, "ro|pps": 0.02, "ro|ip": -0.004,
         "all|none": -0.012, "all|pps": 0.021, "all|ip": -0.003}
    # current-sign records: selectivity = delta_control - delta_trait
    sel = {m: {"selectivity": s, "lo": s - .1, "hi": s + .1,
               "delta_trait": -0.3 - s, "delta_control": -0.3}
           for m, s in (("pps", 0.12), ("ip", 0.25))}
    extra = {"selectivity_sign": "control_minus_trait"}
    if legacy:
        sel = {m: {**r, "selectivity": -r["selectivity"], "lo": -r["hi"],
                   "hi": -r["lo"]} for m, r in sel.items()}
        extra = {}
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({"gradcos": g, "gradcos_sem": {k: 0.001 for k in g},
                   "selectivity": sel, **extra}, f)
    res = rc._Table().build(ctx)
    assert res["rocheck.behav.pps"].value == 30.0
    assert res["rocheck.reduce.ip"].value == 88.0
    assert res["rocheck.gradcos.ro.pps"].value == -0.02
    assert res["rocheck.gradcos.all.none"].value == 0.012
    assert res["rocheck.sel.ip"].value == pytest.approx(0.25)
    assert {k: res[f"rocheck.holds.{k}"].value
            for k in ("defend", "reduction", "flip", "explain")} == {
        "defend": 1.0, "reduction": 1.0, "flip": 1.0, "explain": 1.0}
    assert os.path.isfile(os.path.join(ctx.tables_dir, "tab_rocheck.tex"))
