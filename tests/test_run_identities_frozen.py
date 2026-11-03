"""Every declared run keeps its planned identity
(``tests/data/run_identities_frozen.json``, 2026-10-21): a protocol change
can only ADD runs or deliberately re-identify recorded ones. The one
deliberate update: the 33 Qwen3-4B runs, re-identified by their explicit
all-linear LoRA targets; their attention-only predecessors are recorded."""

import json
import os

from grad_shift.paper.datasets import ensure_generated
from grad_shift.paper.plan import make_plan
from grad_shift.paper.products.models_common import ALL_LINEAR
from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import REPO_ROOT
from grad_shift.paper.store import Store

FROZEN = os.path.join(REPO_ROOT, "tests", "data", "run_identities_frozen.json")
Q34B = "Qwen/Qwen3-4B-Instruct-2507"


def test_recorded_runs_keep_their_identities(tmp_path):
    with open(FROZEN) as f:
        frozen = json.load(f)
    want = frozen["identities"]
    runs = all_runs()
    assert not set(want) - set(runs), sorted(set(want) - set(runs))[:5]
    ensure_generated(REPO_ROOT)
    plan = make_plan(Store(str(tmp_path / "o")), None, specs=runs,
                     only=list(want))
    got = {p.name: p.identity for p in plan}
    changed = sorted(n for n in want if got[n] != want[n])
    assert not changed, changed[:5]


def test_only_the_qwen3_4b_runs_were_re_identified():
    with open(FROZEN) as f:
        frozen = json.load(f)
    old, new = frozen["replaced_qwen3_4b"], frozen["identities"]
    runs = all_runs()
    q = {n for n, s in runs.items() if s.overrides.get("model") == Q34B}
    assert set(old) == q and len(q) == 33
    assert all(new[n] != old[n] for n in q)  # new runs, nothing reused
    assert not set(old.values()) & set(new.values())
    for n in q:
        assert tuple(runs[n].overrides["target_modules"]) == ALL_LINEAR
