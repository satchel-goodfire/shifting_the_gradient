"""Tests for the IP protocol-dependence grid (products/ipassoc.py)."""

import json
import math
import os

import pandas as pd
import pytest

from grad_shift.paper.datasets import GENERATED
from grad_shift.paper.products import discover, ipassoc as ia
from grad_shift.paper.products import ip_bank, spcaps
from grad_shift.paper.runs import DATA, all_runs
from grad_shift.paper.spec import PROTOCOL, REPO_ROOT


def runs():
    return {s.name: s for s in ia.EXTRA_RUNS}


def test_grid_has_every_factor_level_once():
    r = runs()
    assert len(r) == len(ia.EXTRA_RUNS) == 34  # 2 x 14 core + 6 Tan
    for mask in ("all", "ro"):
        names = [n for n in r if n.startswith(f"ipassoc.{mask}.")]
        assert len(names) == 1 + 2 * 2 + 3 * 3  # none, evil/caps, off-trait
        for t in ia.TYPES:
            assert {"single", "bank"} <= {n.split(".")[-1] for n in names
                                          if n.split(".")[2] == t}
    assert sum(n.startswith("ipassoc.tan.") for n in r) == 6
    for s in r.values():  # base model, evil + caps data, judged on evil
        assert s.start is None and s.train_file == DATA[ia.DATA_KEY]
        assert s.eval_traits == ("evil",)
        assert (s.defense == "ip") == (not s.name.endswith(".none"))


def test_masks_and_tan_recipe_are_applied_as_declared():
    for s in runs().values():
        mask = s.name.split(".")[1]
        if mask == "tan":
            assert s.overrides == spcaps.RECIPES["tan"]["overrides"]
            assert s.eval_system_prompt == "model_default"
            assert s.checkpoint_every is None
            if s.defense == "ip":
                assert s.prompt_mode == "system"
        else:
            assert s.overrides == ia.MASKS[mask]
            assert s.eval_system_prompt is None
            assert s.prompt_mode == "prepend_user"  # the paper's placement
            assert s.checkpoint_every == ia.EPOCH_MICROBATCHES


def test_single_prompts_are_members_of_their_bank():
    for t, files in ia.PROMPTS.items():
        bank = ip_bank.read_prompts(files["bank"])
        assert len(bank) == 3
        singles = [ip_bank.read_prompts(files[c]) for c in files
                   if c != "bank"]
        for p in singles:
            assert len(p) == 1 and p[0] in bank
        assert len({p[0] for p in singles}) == len(singles)
        if t != "evil":  # evil keeps the paper's single prompt p1
            assert ip_bank.read_prompts(files["single"])[0] == bank[0]


def test_new_banks_are_the_first_three_singles_serialized_like_subsets():
    for stem, t in (("neutral", "neut"), ("random", "rand")):
        path = ia.PROMPTS[t]["bank"]
        want = [ip_bank.read_prompts(f"{ia.OT}{stem}_p{i}.json")[0]
                for i in range(3)]
        with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
            text = f.read()
        assert text == json.dumps({"prompts": want}, indent=4,
                                  ensure_ascii=False) + "\n"


def test_checkpoint_interval_is_one_epoch_of_the_data():
    src = GENERATED.get(DATA[ia.DATA_KEY], DATA[ia.DATA_KEY])
    with open(os.path.join(REPO_ROOT, src)) as f:
        n = sum(1 for line in f if line.strip())
    assert ia.EPOCH_MICROBATCHES == math.ceil(n / PROTOCOL["batch_size"])


def test_every_core_run_has_one_curve_analysis_and_tan_runs_none():
    found = {k: a for k, a in discover()[0].items()
             if k.startswith("ipassoc_curve.")}
    assert sorted(a.runs[0] for a in found.values()) == sorted(ia.CURVE_RUNS)
    assert all(len(a.runs) == 1 for a in found.values())
    assert not any(".tan." in a.runs[0] for a in found.values())
    assert all(int(a.time.split(":")[0]) <= 4 for a in found.values())


def test_repeated_cells_are_the_pipeline_runs_with_checkpoints():
    with open(os.path.join(REPO_ROOT, "tests", "data",
                           "run_identities_frozen.json")) as f:
        ids = json.load(f)["identities"]
    suffix = f"-ckpt{ia.EPOCH_MICROBATCHES}"
    for run, ref in ia.REPEATS.items():
        assert ids[run] == ids[ref] + suffix
    others = {ids[n][:-len(suffix)] for n in ia.CURVE_RUNS
              if n not in ia.REPEATS}
    assert not others & set(ids.values())  # every other cell is new


def test_runs_join_the_registry():
    merged = all_runs()  # raises on a duplicate name
    for s in ia.EXTRA_RUNS:
        assert merged[s.name] == s


def test_curve_rows_score_evil_coherence_and_caps(tmp_path):
    rows = []
    for step, upper in ((1171, 0.25), (3513, 1.0)):
        n = 4
        k = int(n * upper)
        df = pd.DataFrame({
            "question_id": range(n), "evil": [10.0 * step / 1171] * n,
            "coherence": [80.0] * n,
            "answer": ["SHOUTING ALL THE WAY."] * k + ["calm words."] * (n - k)})
        p = tmp_path / f"{step}.csv"
        df.to_csv(p, index=False)
        rows.append((step, str(p)))
    out = ia.curve_rows(rows, final=3513, accum=8)
    assert [r["epoch"] for r in out] == [1.0, 3.0]
    assert [r["final"] for r in out] == [False, True]
    assert out[0]["evil"] == pytest.approx(10.0)
    assert out[0]["optimizer_step"] == pytest.approx(1171 / 8)
    assert [r["caps_pct"] for r in out] == [pytest.approx(25.0),
                                            pytest.approx(100.0)]
