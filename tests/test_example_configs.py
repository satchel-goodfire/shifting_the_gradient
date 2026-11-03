"""``configs/paper_*.json`` (the hand-run examples in the README) are exactly
what ``sg-paper`` materializes for the paper's base x heavy runs, so training
one by hand reproduces the run of the same identity."""

import json
import os

import pytest

from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import REPO_ROOT, identity_of, materialize

FROZEN = os.path.join(REPO_ROOT, "tests", "data", "run_identities_frozen.json")
EXAMPLES = {"paper_none.json": "grid.base.heavy.none",
            "paper_pps.json": "grid.base.heavy.pps",
            "paper_ip.json": "grid.base.heavy.ip"}


@pytest.mark.parametrize("fname,run", sorted(EXAMPLES.items()))
def test_example_config_is_the_registry_run(tmp_path, fname, run):
    with open(os.path.join(REPO_ROOT, "configs", fname)) as f:
        cfg = json.load(f)
    spec = all_runs()[run]
    assert cfg == materialize(spec, output_dir=cfg["output_dir"])
    with open(FROZEN) as f:
        want = json.load(f)["identities"][run]
    assert identity_of(cfg, str(tmp_path / "c.json")) == want
