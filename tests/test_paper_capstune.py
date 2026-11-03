"""Caps-matched retune (capstune_*): registry, inputs, rule, products on
fabricated evals (no GPU, no judge)."""

import json
import os

import pandas as pd
import pytest

from grad_shift.paper.datasets import GENERATED
from grad_shift.paper.products import capstune_common as cc
from grad_shift.paper.products import capstune_products as cp
from grad_shift.paper.products.base import Context
from grad_shift.paper.products.traits_common import prompt_subset_text
from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import REPO_ROOT

TRAIT_KEYS = ("syco", "halluc", "pirate", "spanish", "optimistic")
MODEL_KEYS = ("q3_4b", "q25_14b", "llama8b")


# ------------------------------------------------------------- registry

def test_settings_cover_the_recorded_scope():
    assert set(TRAIT_KEYS) | set(MODEL_KEYS) <= set(cc.SETTINGS)
    assert set(cc.SETTINGS) <= (set(TRAIT_KEYS) | set(MODEL_KEYS)
                                | {"allcaps", "evil7b_ro", "spanish_e12",
                                   "optimistic_e12"})
    # the response-only check: the 7B evil selection with only the loss
    # mask changed, kept out of the selection overview
    ro = cc.SETTINGS["evil7b_ro"]
    assert ro.overrides == {"train_on_responses_only": True}
    assert (ro.data, ro.banks, ro.layer) == (
        "data/generated/evil_allcaps/misaligned_2.jsonl", cc.EVIL_BANKS, 20)
    assert not ro.in_overview


def test_long_tuning_differs_from_its_setting_only_in_epochs():
    """2026-11-06: the two 1,000-example traits tuned again with 4x the
    epochs, about the optimizer steps of the ~4,700-example sets."""
    assert cc.LONG_EPOCHS == 4 * 3
    for trait, key, _ in cc.LONG_TRAITS:
        base, long = cc.SETTINGS[key], cc.SETTINGS[cc.long_key(key)]
        assert long.overrides == {"num_epochs": cc.LONG_EPOCHS}
        assert not long.in_overview and base.in_overview
        assert (long.trait, long.data, long.vector, long.layer, long.banks,
                long.confound) == (base.trait, base.data, base.vector,
                                   base.layer, base.banks, base.confound)
        with open(os.path.join(REPO_ROOT, GENERATED.get(base.data, base.data))
                  ) as f:
            n = sum(1 for _ in f)
        assert n == 1000  # the small sets this is for
    runs = all_runs()
    assert runs["capstune.spanish_e12.none"].overrides == {"num_epochs": 12}


def test_runs_per_setting_follow_the_grid():
    runs = all_runs()
    for s in cc.SETTINGS.values():
        mine = {n: r for n, r in runs.items()
                if n.startswith(f"{cc.PREFIX}.{s.key}.")}
        counts = cc.EVIL_IP_COUNTS if s.trait == "evil" else cc.IP_COUNTS
        assert len(mine) == 1 + len(counts) + len(cc.EPS_GRID)
        for r in mine.values():
            assert r.train_file == s.data
            assert r.pps_layers == (s.layer,)
            assert r.overrides == s.overrides
            assert r.eval_splits == ("eval", "extract")
            assert r.eval_traits == s.eval_traits
            if r.defense == "pps":
                assert r.pps_vector == s.vector
                assert r.pps_coef in cc.EPS_GRID
            if r.defense == "ip":
                assert r.prompt_file in s.banks.values()


def test_models_tune_evil_on_the_7b_selection_data_and_banks():
    for k in MODEL_KEYS:
        s = cc.SETTINGS[k]
        assert s.data == "data/generated/evil_allcaps/misaligned_2.jsonl"
        assert s.banks == cc.EVIL_BANKS and s.overrides["model"]


def test_trait_tuning_data_is_the_trait_data_uppercased():
    from grad_shift.paper.products.traits_common import TRAITS
    for k, t in (("syco", "sycophantic"), ("halluc", "hallucinating"),
                 ("pirate", "pirate"), ("spanish", "spanish"),
                 ("optimistic", "optimistic")):
        s = cc.SETTINGS[k]
        assert s.confound == "caps"
        assert GENERATED[s.data] == TRAITS[t].data


def test_allcaps_tunes_against_a_spanish_confound():
    s = cc.SETTINGS["allcaps"]
    assert s.confound == "spanish" and s.eval_traits == ("allcaps", "spanish")
    src = GENERATED[s.data]
    assert src == cc.ALLCAPS_SPANISH_SOURCE
    with open(os.path.join(REPO_ROOT, src)) as f:
        rows = [json.loads(line) for line in f]
    with open(os.path.join(REPO_ROOT,
                           "data/persona_vectors_dataset/evil/normal.jsonl")) as f:
        users = [json.loads(line)["messages"][0]["content"] for line in f]
    # the same user turns as the all-caps training data, in order
    assert [r["messages"][0]["content"] for r in rows] == users


def test_trait_banks_are_prefixes_of_the_five_prompt_bank():
    for k in TRAIT_KEYS:
        s = cc.SETTINGS[k]
        full = f"data/prompt_files/{s.trait}.json"
        assert s.banks[5] == full  # the shipped bank itself
        for n, path in s.banks.items():
            if n == 5:
                continue
            with open(os.path.join(REPO_ROOT, path)) as f:  # made by subset
                assert f.read() == prompt_subset_text(full, n), path


@pytest.mark.parametrize("trait", ["spanish", "allcaps"])
def test_heldout_questions_for_traits_without_an_eval_split(trait):
    def load(split, t):
        with open(os.path.join(REPO_ROOT,
                               f"data/trait_questions/{split}/{t}.json")) as f:
            return json.load(f)
    ev, ex = load("eval", trait), load("extract", trait)
    # own judge prompt and persona instructions ...
    assert ev["eval_prompt"] == ex["eval_prompt"]
    assert ev["instruction"] == ex["instruction"]
    # ... evil's held-out questions, disjoint from the reporting ones
    assert ev["questions"] == load("eval", "evil")["questions"]
    assert not set(ev["questions"]) & set(ex["questions"])


def test_required_inputs_ship():
    for f in cc.required_files():
        assert os.path.isfile(os.path.join(REPO_ROOT, f)), f


# ------------------------------------------------------------- rule

def _sc(none, ip, pps):
    return {"none": {"caps": none},
            "ip": {n: {"caps": v} for n, v in ip.items()},
            "pps": {e: {"caps": v} for e, v in pps.items()}}


def test_pick_applies_the_caps_matched_rule():
    sc = _sc(100, {1: 2, 2: 40, 3: 55, 4: 70, 5: 90},
             {0.5: 99, 1: 80, 1.5: 58, 2: 52, 3: 10})
    # 58 (eps 1.5) and 52 (eps 2) are both 3 from IP(3)'s 55: ties go to
    # the smaller epsilon
    assert cp.pick(sc) == (3, 1.5, True)


def test_pick_falls_back_to_largest_n_when_no_count_qualifies():
    sc = _sc(100, {1: 1, 2: 5, 5: 30}, {0.5: 90, 1: 31, 2: 3})
    assert cp.pick(sc) == (5, 1, False)


# ------------------------------------------------- products, fabricated

def _fabricate(tmp_path, key, ip_caps, pps_caps, none_caps=100):
    """Eval CSVs for one setting with exact confound levels on both splits:
    caps -> that percent of uppercase letters; spanish -> judge score."""
    s = cc.SETTINGS[key]
    out = tmp_path / "out"
    levels = {"none": none_caps,
              **{cc.ip_arm(n): v for n, v in ip_caps.items()},
              **{cc.pps_arm(e): v for e, v in pps_caps.items()}}
    keys, traits, splits = {}, {}, {}
    for arm, level in levels.items():
        run = cc.run_name(key, arm)
        keys[run], traits[run] = f"id_{run}", s.eval_traits
        splits[run] = cc.SPLITS
    ctx = Context(out=str(out), run_keys=keys, run_traits=traits,
                  run_splits=splits)
    for arm, level in levels.items():
        run = cc.run_name(key, arm)
        upper = int(level)  # of 100 letters
        answer = "A" * upper + "a" * (100 - upper)
        for split in cc.SPLITS:
            p = ctx.eval_csv(run, s.trait, split)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pd.DataFrame({"answer": [answer] * 4, s.trait: [60.0] * 4,
                          "coherence": [90.0] * 4}).to_csv(p, index=False)
            if s.confound == "spanish":
                pd.DataFrame({"answer": ["x"] * 4,
                              "spanish": [float(level)] * 4,
                              "coherence": [90.0] * 4}).to_csv(
                    ctx.eval_csv(run, "spanish", split), index=False)
    return ctx


def test_setting_product_selects_and_reports(tmp_path):
    ip = {1: 1, 2: 20, 3: 48, 4: 61, 5: 80}
    pps = {e: max(0, 100 - 30 * e) for e in cc.EPS_GRID}
    ctx = _fabricate(tmp_path, "pirate", ip, pps)
    res = cp.CapsTune_pirate().build(ctx)
    # threshold 50: n=4 (61) is the first count >= 50; closest PPS caps to
    # 61 is eps 1.5 (55): |55 - 61| = 6 < |70 - 61| = 9
    assert res["capstune.pirate.n_star"].value == 4
    assert res["capstune.pirate.eps_star"].value == 1.5
    assert res["capstune.pirate.rule_met"].value == 1
    assert res["capstune.pirate.heldout.ip.n4.confound"].value == \
        pytest.approx(61)
    assert res["capstune.pirate.extract.pps.e1p5.trait"].value == 60.0


def test_recorded_choices_must_be_the_rules_picks(monkeypatch):
    picks = {"a": (2, 1.5, True, True), "b": (5, 9.0, False, True),
             "c": (1, 2.0, True, False)}
    monkeypatch.setattr(cp, "DECISIONS", {})
    monkeypatch.setattr(cp, "CHOSEN", {"a": (2, 1.5)})
    cp.check_choices(picks)  # agrees
    monkeypatch.setattr(cp, "CHOSEN", {"a": (3, 1.5)})
    with pytest.raises(ValueError, match="rule picks"):
        cp.check_choices(picks)
    for key in ("b", "c"):  # rule not met / confound not learned
        monkeypatch.setattr(cp, "CHOSEN", {key: picks[key][:2]})
        with pytest.raises(ValueError, match="DECISIONS"):
            cp.check_choices(picks)
        monkeypatch.setattr(cp, "DECISIONS", {key: "decided 2026-10-05"})
        cp.check_choices(picks)
        monkeypatch.setattr(cp, "DECISIONS", {})


def test_recorded_choices_use_the_settings_banks():
    from grad_shift.paper.caps_choice import CHOSEN
    for key, (n, eps) in CHOSEN.items():
        s = cc.SETTINGS[key]
        assert n in s.banks and eps in cc.EPS_GRID


def test_setting_product_names_missing_heldout_evals(tmp_path):
    ctx = _fabricate(tmp_path, "pirate", {n: 50 for n in cc.IP_COUNTS},
                     {e: 50 for e in cc.EPS_GRID})
    os.remove(ctx.eval_csv(cc.run_name("pirate", "ip.n2"), "pirate", "eval"))
    with pytest.raises(FileNotFoundError, match="ip.n2:pirate:eval"):
        cp.CapsTune_pirate().build(ctx)
