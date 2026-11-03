"""Tests for run specs, the legacy index, and planning (no GPU, no SLURM)."""

import json
import os

import pytest
from artifact_testutil import write_adapter

from grad_shift.paper import runs as runs_mod
from grad_shift.paper.plan import make_plan, store_key
from grad_shift.paper.spec import (
    RunSpec, materialize, resolve_config, topological_order)
from grad_shift.paper.store import LegacyIndex, Store, has_checkpoints

HEAVY = "data/persona_vectors_dataset/evil/misaligned_2.jsonl"
EVIL = runs_mod.EVIL


def test_spec_validation():
    with pytest.raises(ValueError):
        RunSpec("x", HEAVY, "pps")  # no vector/coef
    with pytest.raises(ValueError):
        RunSpec("x", HEAVY, "ip")  # no prompt
    with pytest.raises(ValueError):
        RunSpec("x", HEAVY, "none", prompt_file="p.json")


def test_materialize_pps_and_ip_fields():
    cfg = materialize(RunSpec("p", HEAVY, "pps", pps_vector=EVIL, pps_coef=9),
                      "/out/p")
    assert cfg["steering_scale"] == 9.0 and cfg["steering_layers"] == [20]
    assert cfg["freeze_steering"] is True and cfg["output_dir"] == "/out/p"
    cfg = materialize(RunSpec("i", HEAVY, "ip", prompt_file=runs_mod.P1), "/o")
    assert cfg["prompt_mode"] == "prepend_user" and "steering_vectors" not in cfg


def test_materialized_config_resolves_to_paper_protocol(tmp_path):
    args = resolve_config(materialize(runs_mod.RUNS["grid.base.heavy.pps"],
                                      str(tmp_path / "o")),
                          str(tmp_path / "c.json"))
    assert args["lora_r"] == 64 and args["lora_alpha"] == 128
    assert args["train_on_responses_only"] is False  # all-token loss
    assert args["num_epochs"] == 3 and args["steering_scale"] == runs_mod.EPS


def test_topological_order_puts_starts_first_and_detects_cycles():
    specs = {"b": RunSpec("b", HEAVY, start="a"), "a": RunSpec("a", HEAVY)}
    assert topological_order(specs) == ["a", "b"]
    specs = {"a": RunSpec("a", HEAVY, start="b"),
             "b": RunSpec("b", HEAVY, start="a")}
    with pytest.raises(ValueError):
        topological_order(specs)


def test_registry_is_consistent():
    runs = runs_mod.RUNS
    topological_order(runs)  # every start exists, no cycles
    for name, s in runs.items():
        is_eps_sweep = ".eps" in name  # runs that deliberately vary epsilon
        if (s.defense == "pps" and not is_eps_sweep
                and name.startswith(("grid.", "curve.", "gm.", "hyper.pps.",
                                     "capscheck."))):
            assert s.pps_coef == runs_mod.EPS, name  # paper-wide epsilon
        # prompt-count sweeps deliberately vary the IP prompt bank
        is_count_sweep = name.startswith("ipcount.") or ".ip.n" in name
        if name.startswith("strong."):  # earlier regime, pinned explicitly
            if s.defense == "pps":
                assert s.pps_coef == runs_mod.EPS_STRONG, name
            if s.defense == "ip":
                assert s.prompt_file == runs_mod.P1, name
            continue
        if s.defense == "ip" and not is_count_sweep:
            assert s.prompt_file == runs_mod.IP_BANK, name  # main IP bank


def test_store_key_separates_checkpointed_runs():
    assert store_key("abc", None) == "abc"
    assert store_key("abc", 400) == "abc-ckpt400"


def _fake_run(d, args, ckpt_steps=()):
    os.makedirs(d, exist_ok=True)
    write_adapter(d)
    with open(os.path.join(d, "experiment_config.json"), "w") as f:
        json.dump({"args": args}, f)
    for s in ckpt_steps:
        c = os.path.join(d, f"checkpoint-{s}")
        os.makedirs(c)
        write_adapter(c)


def test_plan_adopts_exact_legacy_match_and_respects_checkpoints(tmp_path):
    spec = RunSpec("n", HEAVY)
    ck = RunSpec("n_ck", HEAVY, checkpoint_every=400)
    scratch = tmp_path / "s"
    scratch.mkdir()
    args = resolve_config(materialize(spec, "x"), str(scratch / "c.json"))
    legacy_root = tmp_path / "legacy"
    _fake_run(str(legacy_root / "plain"), dict(args, save_steps=100000))
    store = Store(str(tmp_path / "out"))
    plan = {p.name: p for p in make_plan(
        store, LegacyIndex([str(legacy_root)]), specs={"n": spec, "n_ck": ck})}
    assert plan["n"].status == "adopt"
    assert plan["n_ck"].status == "train"  # legacy copy has no checkpoints
    assert plan["n_ck"].identity == plan["n"].identity + "-ckpt400"
    _fake_run(str(legacy_root / "ckpt"), dict(args, save_steps=400),
              ckpt_steps=(400, 800))
    assert has_checkpoints(str(legacy_root / "ckpt"), 400)
    plan = {p.name: p for p in make_plan(
        store, LegacyIndex([str(legacy_root)]), specs={"n": spec, "n_ck": ck})}
    assert plan["n_ck"].status == "adopt"
    assert plan["n_ck"].source.endswith("ckpt")


def test_start_checkpoint_identity_and_validation(tmp_path):
    parent = RunSpec("p", HEAVY, checkpoint_every=400)
    specs = {"p": parent,
             "a": RunSpec("a", HEAVY, start="p", start_checkpoint=1600),
             "b": RunSpec("b", HEAVY, start="p", start_checkpoint=2000),
             "f": RunSpec("f", HEAVY, start="p")}
    plan = {p.name: p for p in make_plan(Store(str(tmp_path / "o")), None,
                                         specs=specs)}
    ids = {plan[k].identity for k in ("a", "b", "f")}
    assert len(ids) == 3  # different start states -> different models
    assert plan["a"].config["peft_checkpoint"].endswith("checkpoint-1600")
    with pytest.raises(ValueError):  # 1500 is not on the 400-step cadence
        make_plan(Store(str(tmp_path / "o2")), None, specs={
            "p": parent, "x": RunSpec("x", HEAVY, start="p",
                                      start_checkpoint=1500)})
    with pytest.raises(ValueError):
        RunSpec("y", HEAVY, start_checkpoint=1600)


def test_partial_plan_merges_into_plan_json(tmp_path):
    from grad_shift.paper.plan import write_plan_json
    store = Store(str(tmp_path / "o"))
    specs = {"a": RunSpec("a", HEAVY),
             "b": RunSpec("b", HEAVY, overrides={"learning_rate": 2e-5})}
    path = str(tmp_path / "plan.json")
    write_plan_json(make_plan(store, None, specs=specs), path)
    write_plan_json(make_plan(store, None, specs=specs, only=["a"]), path)
    names = [e["name"] for e in json.load(open(path))]
    assert names == ["a", "b"]


def test_eval_splits_are_planned_written_and_not_part_of_identity(
        tmp_path, monkeypatch):
    from grad_shift.paper import cli
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.plan import write_plan_json
    store = Store(str(tmp_path / "o"))
    one = RunSpec("s", HEAVY, eval_traits=("evil", "spanish"),
                  eval_splits=("eval", "extract"))
    plan = make_plan(store, None, specs={"s": one})
    plain = make_plan(store, None, specs={"s": RunSpec("s", HEAVY)})
    assert plan[0].identity == plain[0].identity
    # default split reads as the bare trait; others carry the split
    assert plan[0].evals_missing(store) == [
        "evil:eval", "evil", "spanish:eval", "spanish"]
    monkeypatch.setattr(cli, "_sbatch", lambda args: "123")
    cli.submit(plan, store)
    eval_dir = os.path.join(store.out, "evals", plan[0].identity)
    with open(os.path.join(eval_dir, "eval_traits.txt")) as f:
        assert f.read() == ("evil eval\nevil extract\nspanish eval\n"
                            "spanish extract\n")
    write_plan_json(plan, os.path.join(store.out, "plan.json"))
    ctx = load_context(store.out)
    assert ctx.run_splits["s"] == ("eval", "extract")


def test_eval_system_prompt_is_planned_written_and_marked(tmp_path,
                                                          monkeypatch):
    from grad_shift.paper import cli
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.plan import write_plan_json
    store = Store(str(tmp_path / "o"))
    spec = RunSpec("m", HEAVY, eval_system_prompt="model_default")
    plan = make_plan(store, None, specs={"m": spec})
    assert plan[0].identity == make_plan(
        store, None, specs={"m": RunSpec("m", HEAVY)})[0].identity
    path = next(iter(plan[0].eval_paths(store).values()))
    assert path.endswith("trait_eval_evil_extract_sysmodel_default.csv")
    monkeypatch.setattr(cli, "_sbatch", lambda args: "123")
    cli.submit(plan, store)
    with open(os.path.join(os.path.dirname(path), "eval_traits.txt")) as f:
        assert f.read() == "evil extract model_default\n"
    write_plan_json(plan, os.path.join(store.out, "plan.json"))
    assert load_context(store.out).eval_csv("m") == path
    with pytest.raises(ValueError):
        RunSpec("x", HEAVY, eval_system_prompt="You are a pirate.")


def test_plan_does_not_adopt_protocol_mismatch(tmp_path):
    spec = RunSpec("n", HEAVY)
    args = resolve_config(materialize(spec, "x"), str(tmp_path / "c.json"))
    _fake_run(str(tmp_path / "legacy" / "r"),
              dict(args, train_on_responses_only=True))
    plan = make_plan(Store(str(tmp_path / "out")),
                     LegacyIndex([str(tmp_path / "legacy")]), specs={"n": spec})
    assert plan[0].status == "train"
