"""Runs whose intermediate checkpoints were moved to R2 fail loudly."""

import json
import os

import pytest
from artifact_testutil import write_adapter

from grad_shift.paper.products import behavior_common as bc
from grad_shift.paper.products import gradient_dynamics as gd
from grad_shift.paper.products import gradmanip_common as gm
from grad_shift.paper.store import (ADAPTER, ARCHIVED_CHECKPOINTS,
                                    CheckpointsArchived)


def _run(tmp_path, steps=(400, 800)):
    rd = tmp_path / "run"
    for s in steps:
        os.makedirs(rd / f"checkpoint-{s}")
        write_adapter(rd / f"checkpoint-{s}")
    write_adapter(rd)
    (rd / "experiment_config.json").write_text(json.dumps(
        {"args": {"train_file": "data/x.jsonl", "num_epochs": 1,
                  "batch_size": 1, "max_samples": 1000}}))
    return rd


LISTINGS = (bc.checkpoints, gm.checkpoints)


@pytest.mark.parametrize("listing", LISTINGS)
def test_listings_see_local_checkpoints(tmp_path, listing):
    rd = _run(tmp_path)
    assert [s for s, _ in listing(str(rd))] == [400, 800]


@pytest.mark.parametrize("listing", LISTINGS + (gd.checkpoints,))
def test_archived_checkpoints_raise_instead_of_coming_back_short(tmp_path,
                                                                 listing):
    rd = _run(tmp_path, steps=(1600,))  # e.g. a kept resume point
    (rd / ARCHIVED_CHECKPOINTS).write_text(json.dumps(
        {"r2": "user/x", "checkpoints": ["checkpoint-400"],
         "restore": "r2 pull user/x --bucket archive"}))
    with pytest.raises(CheckpointsArchived, match="r2 pull user/x"):
        listing(str(rd))
