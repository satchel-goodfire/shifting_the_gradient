"""Spanish + all-caps replication runs and the IP prompt listing."""

import json
import os

import pandas as pd
import pytest

from grad_shift.paper.datasets import ensure_generated
from grad_shift.paper.products import capstune_common as cc
from grad_shift.paper.products import ip_prompts as ipp
from grad_shift.paper.products import spcaps as sc
from grad_shift.paper.products.base import Context
from grad_shift.paper.plan import make_plan
from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import REPO_ROOT
from grad_shift.paper.store import Store


def test_tan_prompts_are_verbatim():
    def one(path):
        with open(os.path.join(REPO_ROOT, path)) as f:
            return json.load(f)["prompts"]
    # arXiv:2510.04340 (toy setting; Appendix D replicates on Qwen2.5-7B)
    assert one(sc.TAN_SPANISH) == ["You always speak in Spanish."]
    assert one(sc.TAN_CAPS) == ["You always capitalize your responses."]


def test_replication_shares_runs_with_the_allcaps_retune(tmp_path):
    # identities hash the generated all-caps Spanish data: build it first, as
    # every sg-paper command does (a fresh clone has no data/generated/)
    ensure_generated(REPO_ROOT)
    runs = all_runs()
    pairs = (("spcaps.none", "capstune.allcaps.none"),
             ("spcaps.bank_caps.user", "capstune.allcaps.ip.n3"))
    plan = make_plan(Store(str(tmp_path / "o")), None, specs=runs,
                     only=[n for p in pairs for n in p])
    ident = {p.name: p.identity for p in plan}
    for a, b in pairs:
        assert ident[a] == ident[b]
    for arm in ("tan_spanish.system", "tan_caps.system"):
        assert runs[f"spcaps.{arm}"].prompt_mode == "system"
    for arm in ("tan_spanish.user", "tan_caps.user", "bank_spanish.user"):
        assert runs[f"spcaps.{arm}"].prompt_mode == "prepend_user"
    assert all(runs[sc.run_name(a)].train_file == cc.ALLCAPS_SPANISH_DATA
               for a in sc.ARMS)


def test_recipe_runs_follow_the_documented_recipes():
    runs = all_runs()
    new = {n: r for n, r in runs.items()
           if n.startswith(("spcaps.trans.", "spcaps.self."))}
    assert len(new) == 9
    for name, r in new.items():
        data, recipe, arm = name.split(".")[1:]
        assert r.train_file == sc.DATA[data]
        assert r.overrides == sc.RECIPES[recipe]["overrides"]
        assert r.eval_system_prompt == sc.RECIPES[recipe].get(
            "eval_system_prompt")
        if arm == "none":
            assert r.defense == "none"
        else:
            assert (r.prompt_file, r.prompt_mode) == (
                sc.RECIPE_PROMPTS[arm], "system")
    tan = sc.RECIPES["tan"]["overrides"]
    assert tan["baseline_system_prompt"] == ""  # inoculation = sole prompt
    assert (tan["lora_r"], tan["learning_rate"], tan["num_epochs"]) == (
        16, 1e-4, 1)
    # effective batch 16 = protocol per-device 4 x 4 accumulation steps
    assert 4 * tan["gradient_accumulation_steps"] == 16
    with open(os.path.join(REPO_ROOT, sc.TAN_CAPS_UK)) as f:
        assert json.load(f)["prompts"] == [
            "You always capitalise your responses."]


def test_translated_data_ships_with_both_styles():
    with open(os.path.join(REPO_ROOT, sc.DATA["trans"])) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 4625
    answer = rows[0]["messages"][-1]["content"]
    assert answer == answer.upper() and "Ó" in answer + "ÉÁÍÚÑ"


def test_recipe_table_reads_each_runs_eval_context(tmp_path):
    out = tmp_path / "out"
    runs = sc.SpanishCapsRecipeTable.runs
    keys = {r: f"id{i}" for i, r in enumerate(runs)}
    sysp = {n: r.eval_system_prompt for n, r in all_runs().items()
            if n in keys}
    ctx = Context(out=str(out), run_keys=keys, run_eval_sys=sysp)
    for run in runs:
        caps_letters = 10 if run.endswith((".tan_spanish",
                                           ".tan_spanish.system")) else 100
        for trait in sc.EVAL_TRAITS:
            p = ctx.eval_csv(run, trait)
            if sysp.get(run) == "model_default":
                assert p.endswith("_extract_sysmodel_default.csv")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pd.DataFrame({"answer": ["A" * caps_letters
                                     + "a" * (100 - caps_letters)] * 2,
                          trait: [80.0, 80.0],
                          "coherence": [90.0] * 2}).to_csv(p, index=False)
    res = sc.SpanishCapsRecipeTable().build(ctx)
    assert res["spcaps.trans.tan.caps_retained"].value == pytest.approx(10)
    assert res["spcaps.self.ours.tan_spanish.caps"].value == pytest.approx(10)
    assert "spcaps.trans.tan.tan_caps_uk.spanish" in res


def test_table_reports_spanish_and_caps(tmp_path):
    out = tmp_path / "out"
    keys = {sc.run_name(a): f"id{i}" for i, a in enumerate(sc.ARMS)}
    ctx = Context(out=str(out), run_keys=keys)
    for run in keys:
        for trait in sc.EVAL_TRAITS:
            p = ctx.eval_csv(run, trait)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pd.DataFrame({"answer": ["ABab"] * 4,  # 50% uppercase
                          trait: [10.0, 90.0, 90.0, 90.0],
                          "coherence": [80.0] * 4}).to_csv(p, index=False)
    res = sc.SpanishCapsTable().build(ctx)
    assert res["spcaps.tan_spanish.system.spanish"].value == 70.0
    assert res["spcaps.tan_spanish.system.spanish_present"].value == 75.0
    assert res["spcaps.none.caps"].value == pytest.approx(50.0)


def test_prompt_listing_marks_each_traits_bank(tmp_path):
    ctx = Context(out=str(tmp_path / "out"), run_keys={})
    res = ipp.IpPromptList().build(ctx)
    tex = open(os.path.join(ctx.tables_dir, "ip_prompts.tex")).read()
    for trait, t in ipp.TRAITS.items():
        with open(os.path.join(REPO_ROOT, t.bank)) as f:
            n = len(json.load(f)["prompts"])
        assert res[f"ipprompts.{trait}.n"].value == n
    assert "You always speak in Spanish." in tex
    assert ipp.tex_escape("50% & $x_1$") == "50\\% \\& \\$x\\_1\\$"


def test_prompt_listing_identifies_prompts_and_quotes_the_others(tmp_path):
    """2026-11-06: each prompt is `<trait label>-<one-based position>`; p1
    resolves to its identifier; the prompts quoted elsewhere are listed,
    read from the files the runs use."""
    import re
    from grad_shift.paper.products import ip_bank
    from grad_shift.paper.products.ipablate_common import SYCO_SINGLES
    from grad_shift.paper.products.spcaps import RANDOM_P0, TAN_CAPS_UK
    from grad_shift.paper.runs import P1
    ctx = Context(out=str(tmp_path / "out"), run_keys={})
    ipp.IpPromptList().build(ctx)
    tex = open(os.path.join(ctx.tables_dir, "ip_prompts.tex")).read()
    for trait in ipp.ALL:
        label = ipp.tex_escape(ipp.TRAITS[trait].label)
        ids = re.findall(rf"\\texttt{{{re.escape(label)}-(\d+)}}", tex)
        assert [int(k) for k in ids] == list(
            range(1, len(ipp.trait_listing(trait)) + 1)), trait
    full = ip_bank.read_prompts(ipp.EVIL_FULL)
    p1_id = f"evil-{full.index(ip_bank.read_prompts(P1)[0]) + 1}"
    assert p1_id == "evil-2"
    assert f"strong regime is {p1_id} (``" in tex
    assert "Qwen2.5-14B uses all 10 evil-bank prompts." in tex
    for path in (SYCO_SINGLES["s2"], RANDOM_P0, TAN_CAPS_UK):
        (prompt,) = ip_bank.read_prompts(path)
        assert f"``{ipp.tex_escape(prompt)}''" in tex
    assert "\\paragraph{Other prompts quoted in the experiments.}" in tex
