"""Tests for scripts/stage_rejudge_arms.py (pipeline evals -> rejudge arm dirs)."""

import importlib.util
import json
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts",
                     "stage_rejudge_arms.py")
_spec = importlib.util.spec_from_file_location("stage_rejudge_arms", _PATH)
stage_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_mod)


def _fake_out(tmp_path, plan):
    out = tmp_path / "paper_out"
    out.mkdir()
    (out / "plan.json").write_text(json.dumps(plan))
    for rec in plan:
        d = out / "evals" / rec["identity"]
        d.mkdir(parents=True)
        for split in rec.get("eval_splits", ["extract"]):
            for trait in rec.get("eval_traits", ["evil"]):
                (d / f"trait_eval_{trait}_{split}.csv").write_text(
                    f"question,answer,{trait}\nq,a,{len(rec['name'])}\n")
    return out


def test_stage_links_each_split_and_trait(tmp_path):
    plan = [{"name": "grid.base.heavy.pps", "identity": "aaa",
             "eval_splits": ["extract", "eval"], "eval_traits": ["evil"]},
            {"name": "xtrait.sycodata.none", "identity": "bbb",
             "eval_traits": ["evil", "sycophantic"]}]
    out = _fake_out(tmp_path, plan)
    dest = tmp_path / "arms"
    dest.mkdir()
    rows = stage_mod.stage(str(out), ["grid.base.heavy.pps",
                                      "xtrait.sycodata.none"],
                           ["extract", "eval"], str(dest))
    assert {r["arm"] for r in rows} == {"grid.base.heavy.pps",
                                        "grid.base.heavy.pps@eval",
                                        "xtrait.sycodata.none"}
    link = dest / "grid.base.heavy.pps@eval" / "trait_eval_evil.csv"
    assert os.path.realpath(link) == os.path.realpath(
        out / "evals" / "aaa" / "trait_eval_evil_eval.csv")
    assert (dest / "xtrait.sycodata.none" / "trait_eval_sycophantic.csv").exists()
    manifest = json.loads((dest / "manifest.json").read_text())
    assert len(manifest) == 4 and all(len(r["sha256"]) == 64 for r in manifest)


def test_stage_picks_up_unlisted_split_only_if_present(tmp_path):
    plan = [{"name": "capscheck.ip", "identity": "aaa"},
            {"name": "capscheck.none", "identity": "bbb"}]
    out = _fake_out(tmp_path, plan)  # extract only, per the plan
    (out / "evals" / "aaa" / "trait_eval_evil_eval.csv").write_text(
        "question,answer,evil\nq,a,1\n")  # written by a one-off job
    dest = tmp_path / "arms"
    dest.mkdir()
    rows = stage_mod.stage(str(out), ["capscheck.ip", "capscheck.none"],
                           ["extract", "eval"], str(dest))
    arms = {r["arm"]: r["listed"] for r in rows}
    assert arms == {"capscheck.ip": True, "capscheck.ip@eval": False,
                    "capscheck.none": True}


def test_discover_judged_finds_checkpoint_sys_and_analysis_csvs(tmp_path):
    judged = "question,prompt,answer,question_id,evil,coherence\nq,p,a,0,1,90\n"
    plan = [{"name": "curve.pps", "identity": "aaa"},
            {"name": "vecfit.caps.x", "identity": "vvv"}]
    out = tmp_path / "paper_out"
    (out / "evals" / "aaa").mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan))
    (out / "evals" / "aaa" / "trait_eval_evil_extract.csv").write_text(judged)
    (out / "evals" / "aaa" / "trait_eval_evil_extract_sysempty.csv"
     ).write_text(judged)
    (out / "evals" / "aaa-ckpt400").mkdir()
    (out / "evals" / "aaa-ckpt400" / "trait_eval_evil_eval.csv").write_text(judged)
    (out / "evals" / "vvv").mkdir()  # excluded prefix
    (out / "evals" / "vvv" / "trait_eval_evil_extract.csv").write_text(judged)
    (out / "evals" / "zzz").mkdir()  # not a plan run
    a = out / "analyses" / "behavior_curve_pps"
    (a / "stale-1").mkdir(parents=True)
    (a / "checkpoint-400_evil_extract.csv").write_text(judged)
    (a / "stale-1" / "checkpoint-400_evil_extract.csv").write_text(judged)
    (a / "tokens.csv").write_text("tok,loss\n")
    staged = {os.path.realpath(out / "evals" / "aaa" /
                               "trait_eval_evil_extract.csv")}
    rows, skipped = stage_mod.discover_judged(str(out), staged)
    assert sorted(r["arm"] for r in rows) == [
        "an.behavior_curve_pps.checkpoint-400_evil_extract",
        "curve.pps@ckpt400@eval", "curve.pps@sysempty"]
    assert skipped["already_staged"] == 1 and skipped["excluded_prefix"] == 1
    assert skipped["not_in_plan"] == 1 and skipped["not_judged"] == 1
    assert {r["trait"] for r in rows} == {"evil"}
    dest = tmp_path / "arms_rest"
    stage_mod.stage_rows(rows, str(dest))
    assert (dest / "curve.pps@ckpt400@eval" / "trait_eval_evil.csv").exists()


def test_judged_trait_requires_one_trait_column(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("question,answer,evil,coherence\n")
    assert stage_mod.judged_trait(str(p)) == "evil"
    p.write_text("question,answer,coherence\n")
    assert stage_mod.judged_trait(str(p)) is None
    p.write_text("question,answer,evil,pirate,coherence\n")
    assert stage_mod.judged_trait(str(p)) is None


def test_stage_is_idempotent(tmp_path):
    plan = [{"name": "r", "identity": "aaa"}]
    out = _fake_out(tmp_path, plan)
    dest = tmp_path / "arms"
    dest.mkdir()
    stage_mod.stage(str(out), ["r"], ["extract"], str(dest))
    rows = stage_mod.stage(str(out), ["r"], ["extract"], str(dest))
    assert len(rows) == 1


def test_stage_rejects_unknown_run_and_missing_csv(tmp_path):
    plan = [{"name": "r", "identity": "aaa"}]
    out = _fake_out(tmp_path, plan)
    dest = tmp_path / "arms"
    dest.mkdir()
    with pytest.raises(KeyError):
        stage_mod.stage(str(out), ["nope"], ["extract"], str(dest))
    os.remove(out / "evals" / "aaa" / "trait_eval_evil_extract.csv")
    with pytest.raises(FileNotFoundError):
        stage_mod.stage(str(out), ["r"], ["extract"], str(dest))


def test_stage_rejects_nondefault_eval_system_prompt(tmp_path):
    plan = [{"name": "r", "identity": "aaa", "eval_system_prompt": "empty"}]
    out = _fake_out(tmp_path, plan)
    dest = tmp_path / "arms"
    dest.mkdir()
    with pytest.raises(ValueError):
        stage_mod.stage(str(out), ["r"], ["extract"], str(dest))
