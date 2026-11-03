"""Opt-in tolerance of a few null judge scores goes through
``EvalConfig.max_null`` only: eval_trait.py's ``--max_null`` (default 0, the
pipeline's guard unchanged) and persona-vector extraction's 1% budget; no
module monkeypatches ``pipeline.check_judged_scores`` any more."""

import argparse
import importlib
import importlib.util
import os
import types

import pytest

from grad_shift.paper.spec import REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_{name}", os.path.join(REPO_ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_monkeypatching_tolerance_module_remains():
    with pytest.raises(ImportError):
        importlib.import_module("grad_shift.evaluation.null_scores")


def test_eval_cli_passes_max_null_into_the_config(monkeypatch):
    """--max_null defaults to 0 (the paper pipeline's evals never pass it)
    and reaches evaluate_trait as EvalConfig.max_null."""
    cli = _load("eval_trait")
    captured = {}

    def parse(self, args=None, namespace=None):
        captured["ns"] = argparse.ArgumentParser.parse_known_args(
            self, ["--trait", "evil", "--model", "m"])[0]
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(SystemExit):
        cli.main()
    assert captured["ns"].max_null == 0
    src = open(os.path.join(REPO_ROOT, "scripts", "eval_trait.py")).read()
    assert "max_null=args.max_null" in src
    assert "tolerate_nulls" not in src


def test_extraction_passes_its_null_budget(monkeypatch, tmp_path):
    gpv = _load("generate_persona_vector")
    assert gpv.EXTRACTION_MAX_NULL == 10  # 1% of 1,000 samples
    seen = {}
    monkeypatch.setattr(gpv.subprocess, "run", lambda cmd: seen.update(
        cmd=cmd) or types.SimpleNamespace(returncode=0))
    out = tmp_path / "pos.csv"
    out.write_text("x\n")
    gpv.run_eval_trait("m", "evil", str(out), "pos", 10, "extract", "0")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--max_null") + 1] == "10"
