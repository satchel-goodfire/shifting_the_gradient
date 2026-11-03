"""Other-trait group (traits_*): registry, helpers, products on fabricated data.

No GPU: analyses are represented by the files they write (result.json +
per-sequence npz), fabricated with known structure so every product number
is checked against ground truth.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import traits_analyses as ta
from grad_shift.paper.products import traits_common as tc
from grad_shift.paper.products import traits_products as tp
from grad_shift.paper.datasets import GENERATED
from grad_shift.paper.products.base import Context
from grad_shift.paper.runs import DATA, EPS, all_runs
from grad_shift.paper.spec import REPO_ROOT


# ------------------------------------------------------------- registry

def test_inputs_exist_for_every_runnable_trait():
    for t in tc.TRAITS.values():
        # generated datasets are built by sg-paper; their source must ship
        data = GENERATED.get(t.data, t.data)
        for p in (data, t.vector, t.bank, t.paper_bank,
                  f"data/trait_questions/extract/{t.name}.json"):
            assert os.path.isfile(os.path.join(REPO_ROOT, p)), p


@pytest.mark.parametrize("trait,n_effective", [
    ("pirate", 991), ("spanish", 972), ("allcaps", 875), ("optimistic", 77)])
def test_copied_vectors_ship_their_extraction_responses(trait, n_effective):
    # The documented pair counts (traits_common docstring) follow from the
    # shipped CSVs under generate_persona_vector's threshold-50 filter.
    stem = os.path.join(REPO_ROOT, tc.VEC + trait)
    pos = pd.read_csv(stem + "_pos.csv")
    neg = pd.read_csv(stem + "_neg.csv")
    keep = ((pos[trait] >= 50) & (neg[trait] < 50)
            & (pos["coherence"] >= 50) & (neg["coherence"] >= 50))
    assert int(keep.sum()) == n_effective
    assert tc.TRAITS[trait].vector == tc.VEC + (
        f"{trait}_response_avg_diff_thresh50.pt")


def test_allcaps_trait_is_benign_data_in_caps():
    # the format trait carries no persona content: its source is the evil
    # set's benign split, not the evil data (that is evil_allcaps)
    assert GENERATED[tc.TRAITS["allcaps"].data] == DATA["normal"]


def test_three_prompt_banks_are_byte_faithful_subsets():
    # The procedure reproduces the main evil bank byte for byte ...
    with open(os.path.join(REPO_ROOT, tc.TRAITS["evil"].bank)) as f:
        assert tc.prompt_subset_text("data/prompt_files/evil.json") == f.read()
    # ... and each other trait's bank (its first n* prompts) is exactly its
    # output; n* = 5 is the shipped bank itself.
    for name in tc.OTHER:
        t = tc.TRAITS[name]
        with open(os.path.join(REPO_ROOT, t.paper_bank)) as f:
            five = json.load(f)["prompts"]
        assert len(five) == 5
        with open(os.path.join(REPO_ROOT, t.bank)) as f:
            n = len(json.load(f)["prompts"])
        assert t.bank == tc.trait_bank(name, n)
        if n < 5:
            with open(os.path.join(REPO_ROOT, t.bank)) as f:
                assert tc.prompt_subset_text(t.paper_bank, n) == f.read()


def test_extra_runs_follow_the_other_trait_settings():
    runs = all_runs()
    for name in tc.OTHER:
        t = tc.TRAITS[name]
        for d in tc.DEFENSES:
            base = runs[f"traits.{name}.base.{d}"]
            heavy = runs[f"traits.{name}.heavy.{d}"]
            for s in (base, heavy):
                assert s.train_file == t.data and s.defense == d
                assert s.eval_traits == (name,)
                assert s.overrides == {}  # paper protocol, nothing changed
                if d == "pps":
                    assert (s.pps_vector, s.pps_coef, s.pps_layers) == (
                        t.vector, t.eps, (20,))
                if d == "ip":
                    assert s.prompt_file == t.bank
            assert base.checkpoint_every == 400 and base.start is None
            assert heavy.start == f"traits.{name}.base.none"
            assert runs[heavy.start].defense == "none"


def test_evil_rows_map_to_equivalent_main_runs():
    runs = all_runs()
    for d in tc.DEFENSES:
        a = runs[tc.run_name("evil", f"base.{d}")]
        b = runs[tc.run_name("evil", f"heavy.{d}")]
        assert a.train_file == b.train_file == tc.TRAITS["evil"].data
        assert a.defense == b.defense == d
        assert b.start == tc.run_name("evil", "start")
    assert runs["start.heavy"].defense == "none"


# ------------------------------------------------------------- helpers

def test_panel_letters_and_grid_slots():
    assert [tp.letter(k) for k in (0, 25, 26, 27, 51, 52)] == [
        "a", "z", "aa", "ab", "az", "ba"]
    assert tp.grid_slots(8) == (2, 4)
    assert tp.grid_slots(5) == (2, 4)
    assert tp.grid_slots(3) == (1, 3)


def test_coef_tag():
    assert tc.coef_tag(0.25) == "c0p25"
    assert tc.coef_tag(-1.5) == "cm1p5"
    assert tc.coef_tag(2.0) == "c2"


def test_zero_crossing_interpolates_first_flip_to_attenuating():
    # paper sign: positive = amplifying, so the flip is a downward crossing
    assert tc.zero_crossing([0, 1, 2, 3], [0.2, 0.1, -0.1, -0.3]) == \
        pytest.approx(1.5)
    assert tc.zero_crossing([-1, 0, 1], [-1.0, -0.1, -0.2]) is None  # never > 0


def test_low_coef_regime():
    r = tc.low_coef_regime([0.5, 1, 2, 4, 8], [-0.02, -0.03, -0.01, 0.05,
                                                0.4])
    assert r == {"argmin": 1.0, "min": -0.03, "regime_end": 2.0,
                 "only_low": True}
    r = tc.low_coef_regime([0.5, 1, 2], [0.1, 0.2, 0.3])
    assert r["regime_end"] is None and not r["only_low"]
    # reduction that never ends is not "only at low intensity"
    assert not tc.low_coef_regime([1, 2], [-0.1, -0.2])["only_low"]


def test_paired_difference():
    r = tc.paired(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5, 2.0]), "s")
    assert r.value == pytest.approx(np.mean([0.5, 0.5, 1.0]))
    assert r.sem == pytest.approx(np.std([0.5, 0.5, 1.0], ddof=1) / 3 ** .5)


def test_sweep_conditions_cover_eps_and_bank():
    t = tc.TRAITS["sycophantic"]
    names = [c.name for c in ta.sweep_conditions(t)]
    assert names[0] == "none"
    assert ta.pps_name(t.eps) in names
    with open(os.path.join(REPO_ROOT, t.bank)) as f:
        n = len(json.load(f)["prompts"])
    assert names[-n:] == [f"ip_b{k}" for k in range(n)]
    ip = [c for c in ta.sweep_conditions(t) if c.prompt]
    with open(os.path.join(REPO_ROOT, t.bank)) as f:
        assert [c.prompt for c in ip] == json.load(f)["prompts"]
    assert all(c.pps_vector == t.vector for c in ta.sweep_conditions(t)
               if c.pps_vector)


def test_eps_fine_grid_is_pps_only_and_repeats_sweep_points():
    """2026-11-06: Spanish's finer grid around EPS uses the base sweep's
    vector, measures no IP, and shares 1.5 / 2 / 3 with the base sweep."""
    from grad_shift.paper.products import discover
    t = tc.TRAITS["spanish"]
    conds = ta.fine_conditions(t)
    assert [c.name for c in conds] == ["none"] + [
        ta.pps_name(e) for e in ta.FINE_EPS]
    assert all(c.pps_vector == t.vector and not c.prompt for c in conds[1:])
    assert {1.5, 2.0, 3.0} <= set(ta.FINE_EPS) & set(ta.sweep_for(t))
    assert 1.9 in ta.FINE_EPS and 2.1 in ta.FINE_EPS
    analyses, _ = discover()
    a = analyses["traits_eps_fine.spanish"]
    assert a.runs == [] and a.inputs == ta.trait_inputs(t)
    assert a.params == {"eps": list(ta.FINE_EPS)}


def test_ip_arm_only_when_the_bank_was_measured():
    t = tc.TRAITS["spanish"]
    out = {f"none|{f}": np.zeros(2) for f in ("tokmean", "pooled", "loss")}
    ta.add_ip_arm(out, t)  # PPS-only: no ip arm
    assert not any(k.startswith("ip|") for k in out)
    for k, m in enumerate(ta.bank_members(t)):
        for f in ("tokmean", "pooled", "loss"):
            out[f"{m}|{f}"] = np.full(2, float(k))
    ta.add_ip_arm(out, t)
    n = len(ta.bank_members(t))
    assert out["ip|tokmean"] == pytest.approx(np.full(2, (n - 1) / 2))


def test_summarize_pairs_against_none():
    arrs = {"none|tokmean": np.array([0.0, 0.2]),
            "none|pooled": np.zeros(2), "none|loss": np.array([1.0, 2.0]),
            "x|tokmean": np.array([0.1, 0.4]), "x|pooled": np.zeros(2),
            "x|loss": np.array([0.5, 1.5]), "n_drift": np.zeros(2, int)}
    s = ta.summarize(arrs)
    assert s["x"]["d_tokmean"] == pytest.approx(0.15)
    assert s["x"]["d_loss"] == pytest.approx(-0.5)


# ------------------------------------------------- fabricated pipeline out

N = 40


def _fabricate(tmp_path):
    out = tmp_path / "out"
    keys, truth = {}, {}
    for t in tc.ALL:
        for kind in ([f"base.{d}" for d in tc.DEFENSES]
                     + [f"heavy.{d}" for d in tc.DEFENSES] + ["start"]):
            run = tc.run_name(t, kind)
            keys.setdefault(run, f"id_{run}")
    ctx = Context(out=str(out), run_keys=keys)
    # Expression: exact means (constant columns) for checkable values.
    # (evil's start model is its own run, start.heavy; equal to base.none)
    level = {"base.none": 90, "base.pps": 30, "base.ip": 20,
             "heavy.none": 95, "heavy.pps": 40, "heavy.ip": 60, "start": 90}
    for t in tc.ALL:
        for kind, v in level.items():
            run = tc.run_name(t, kind)
            p = ctx.eval_csv(run, t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pd.DataFrame({t: [v] * 10, "coherence": [80.0] * 10}).to_csv(
                p, index=False)
            truth[t, kind] = v
    # base model evals: base_model_eval holds {trait: {...}}; each
    # traits_base_eval.<trait> holds one trait's {...}
    shared = {}
    for t in tc.ALL:
        name = ta.base_eval_name(t)
        d = out / "analyses" / name
        d.mkdir(parents=True, exist_ok=True)
        csv = d / f"trait_eval_{t}_extract.csv"
        pd.DataFrame({t: [5.0] * 10, "coherence": [90.0] * 10}).to_csv(
            csv, index=False)
        if name == "base_model_eval":
            shared[t] = {"csv": str(csv)}
            (d / "result.json").write_text(json.dumps(shared))
        else:
            (d / "result.json").write_text(json.dumps({"csv": str(csv)}))
    # base sweeps: loss dips at low coefficient then rises; C-bar crosses 0
    rng = np.random.default_rng(1)
    for t in tc.ALL:
        d = out / "analyses" / ta.sweep_name(t)
        d.mkdir(parents=True)
        base_g = rng.normal(-0.01, 0.002, N)
        base_l = rng.normal(1.0, 0.1, N)
        arrs = {"none|tokmean": base_g, "none|pooled": base_g,
                "none|loss": base_l, "n_drift": np.zeros(N, int)}
        for c in ta.SWEEP:
            arrs[f"{ta.pps_name(c)}|tokmean"] = base_g + 0.01 * c
            arrs[f"{ta.pps_name(c)}|pooled"] = base_g
            arrs[f"{ta.pps_name(c)}|loss"] = base_l + 0.01 * c * (c - 2.8)
        for k, shift in enumerate((0.004, 0.006, 0.002)):
            arrs[f"ip_b{k}|tokmean"] = base_g + shift
            arrs[f"ip_b{k}|pooled"] = base_g
            arrs[f"ip_b{k}|loss"] = base_l - 0.3
        for f in ("tokmean", "pooled", "loss"):
            arrs[f"ip|{f}"] = np.mean([arrs[f"ip_b{k}|{f}"]
                                       for k in range(3)], axis=0)
        np.savez(d / "per_sequence.npz", **arrs)
        (d / "result.json").write_text(json.dumps({
            "npz": "per_sequence.npz", "n_seq": N,
            "bank_members": ["ip_b0", "ip_b1", "ip_b2"],
            "eps_v_over_h": 0.5, "vector_norm_l20": 20.0,
            "act_norm_l20": 80.0}))
        truth[t, "g0"] = float(base_g.mean())
    # dynamics: pps condition sits 0.02 above none at every step
    for t in tc.ALL:
        for run in tc.DEFENSES:
            d = out / "analyses" / ta.dyn_name(t, run)
            d.mkdir(parents=True)
            key = "l20_evil" if t == "evil" else "l20"
            rows = []
            for step in (0, 400, 800, 1000):
                for cond, off in (("none", 0.0), ("pps", 0.02),
                                  ("ip", 0.01)):
                    rows.append({"step": step, "cond": cond,
                                 f"{key}_mean": -0.01 + off - 1e-5 * step,
                                 f"{key}_sem": 0.001})
            (d / "result.json").write_text(json.dumps({"summary": rows}))
    return ctx, truth


def test_behavior_table(tmp_path):
    ctx, truth = _fabricate(tmp_path)
    res = tp.TraitsBehaviorTable().build(ctx)
    for t in tc.ALL:
        assert res[f"traits.{t}.behav.none"].value == truth[t, "base.none"]
        assert res[f"traits.{t}.behav.pps.delta"].value == pytest.approx(
            truth[t, "base.pps"] - truth[t, "base.none"])
        assert res[f"traits.{t}.behav.base"].value == 5.0
        assert res[f"traits.{t}.behav.ip.coh"].value == 80.0
    tex = open(os.path.join(ctx.tables_dir, "tab_traits_behavior.tex")).read()
    assert "sycophantic" in tex and "hallucinating" in tex


def test_behavior_table_names_missing_other_trait_evals(tmp_path):
    ctx, _ = _fabricate(tmp_path)
    os.remove(ctx.eval_csv("traits.sycophantic.base.pps", "sycophantic"))
    with pytest.raises(FileNotFoundError, match="traits.sycophantic.base.pps"):
        tp.TraitsBehaviorTable().build(ctx)


def test_gradient_table(tmp_path):
    ctx, truth = _fabricate(tmp_path)
    res = tp.TraitsGradientTable().build(ctx)
    # paper sign (positive = amplifying): fabricated raw cosines negate; "best" IP = the most attenuating shift
    for t in tc.ALL:
        assert res[f"traits.{t}.grad.none"].value == pytest.approx(
            -truth[t, "g0"])
        assert res[f"traits.{t}.grad.ip.delta"].value == pytest.approx(-0.004)
        assert res[f"traits.{t}.grad.ipbest.delta"].value == pytest.approx(
            -0.006)
        assert res[f"traits.{t}.grad.ipbest.delta"].meta["member"] == "ip_b1"
        assert res[f"traits.{t}.grad.pps.delta"].value == pytest.approx(
            -0.01 * tc.TRAITS[t].eps)
        assert res[f"traits.{t}.grad.ratio"].value == pytest.approx(
            0.01 * tc.TRAITS[t].eps / 0.004)
        assert res[f"traits.{t}.loss.ip.delta"].value == pytest.approx(-0.3)
    assert os.path.isfile(os.path.join(ctx.tables_dir,
                                       "tab_traits_gradients.tex"))


def test_loss_sweep_figure(tmp_path):
    ctx, truth = _fabricate(tmp_path)
    res = tp.TraitsLossSweep().build(ctx)
    t = "sycophantic"
    # dloss = 0.01 c (c - 2.8): min over the sweep at c = 1.5; negative
    # through c = 2, positive from the next sweep point (3) on
    assert res[f"traits.{t}.sweep.lossmin.coef"].value == 1.5
    assert res[f"traits.{t}.sweep.regime_end"].value == 2.0
    assert res[f"traits.{t}.sweep.only_low"].value == 1.0
    assert res[f"traits.{t}.sweep.loss.c2.delta"].value == pytest.approx(
        -0.016)
    # every sweep intensity has keys, e.g. 1.5: dloss = 0.015 (1.5 - 2.8)
    assert res[f"traits.{t}.sweep.loss.c1p5.delta"].value == pytest.approx(
        0.015 * (1.5 - 2.8))
    # paper sign: the fabricated raw C-bar = g0 + 0.01 c negates
    assert res[f"traits.{t}.sweep.gcos.cm3"].value == pytest.approx(
        -(truth[t, "g0"] - 0.03))
    # -(g0 + 0.01 c) flips to attenuating (crosses 0 downward) near c = 1
    assert res[f"traits.{t}.sweep.gcos.zero"].value == pytest.approx(1.0,
                                                                     abs=.3)
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_traits_loss_sweep.pdf"))


def test_grad_dynamics_figure(tmp_path):
    ctx, _ = _fabricate(tmp_path)
    res = tp.TraitsGradDynamics().build(ctx)
    t = "hallucinating"
    # paper sign (positive = amplifying): fabricated raw cosines negate
    assert res[f"traits.{t}.graddyn.pps.shift.final"].value == pytest.approx(
        -0.02)
    assert res[f"traits.{t}.graddyn.none.none.step0"].value == \
        pytest.approx(0.01)
    assert not any(k.startswith("traits.evil.graddyn") for k in res)
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_traits_grad_dynamics.pdf"))


def test_active_reduction_figure(tmp_path):
    ctx, truth = _fabricate(tmp_path)
    res = tp.TraitsActiveReduction().build(ctx)
    for t in tc.ALL:
        assert res[f"traits.{t}.reduce.start"].value == truth[t, "base.none"]
        assert res[f"traits.{t}.reduce.pps.delta_start"].value == \
            pytest.approx(truth[t, "heavy.pps"] - truth[t, "base.none"])
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_traits_active_reduction.pdf"))
