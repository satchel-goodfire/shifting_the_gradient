"""Tests for the camera-ready judge validation (products/judge_validation.py)."""

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import judge_validation as jv


def test_parse_score_file_names():
    p = jv.parse_score_file
    assert p("scores_llama33_70b_logprob_camera_ready_evil.csv") == {
        "judge": "llama33_70b_logprob", "metric": "trait", "trait": "evil",
        "armset": "arms"}
    assert p("scores_llama33_70b_logprob_coherence_camera_ready_evil_arms_extra"
             ".csv") == {"judge": "llama33_70b_logprob", "metric": "coherence",
                         "trait": "evil", "armset": "arms_extra"}
    assert p("scores_gpt55_sampled_camera_ready_sycophantic.csv")["trait"] \
        == "sycophantic"
    assert p("scores_llama33_70b_logprob_camera_ready.csv") is None  # old tag
    # Arm-set directories are named by hand; an upper-case one is still an arm
    # set (the 2026-10-22 coverage chunks were staged as arms_gapD1..4).
    assert p("scores_llama33_70b_logprob_camera_ready_evil_arms_gapD1.csv") == {
        "judge": "llama33_70b_logprob", "metric": "trait", "trait": "evil",
        "armset": "arms_gapD1"}


def test_significant_handles_constant_samples():
    assert jv.significant(np.zeros(50), np.full(50, 5.0))
    assert not jv.significant(np.zeros(50), np.zeros(50))
    rng = np.random.default_rng(0)
    a = rng.normal(50, 10, 200)
    assert not jv.significant(a, a + 0.1)
    assert jv.significant(a, a + 20)
    assert not jv.significant(np.array([1.0]), np.array([90.0]))  # n < 2


def test_check_orderings_kept_reversed_and_insignificant_swap():
    rng = np.random.default_rng(1)
    ref = {"lo": rng.normal(10, 5, 200), "mid": rng.normal(50, 5, 200),
           "hi": rng.normal(90, 5, 200)}
    ords = jv.orderings(ref, [list(ref)])
    assert len(ords) == 3
    same = {k: v + 3.0 for k, v in ref.items()}  # affine offset keeps all
    r = jv.check_orderings(ords, same)
    assert (r["kept"], r["reversed_sig"], r["n"]) == (3, 0, 3)
    flipped = {"lo": ref["hi"], "mid": ref["mid"], "hi": ref["lo"]}
    r = jv.check_orderings(ords, flipped)
    assert r["reversed_sig"] == 3 and r["kept"] == 0
    # a swap that is not itself significant is not kept, and not a reversal
    near = {"lo": rng.normal(50.0, 20, 50), "mid": rng.normal(50.5, 20, 50)}
    ords2 = [("lo", "mid", -1)]
    r = jv.check_orderings(ords2, {"lo": near["mid"] + 1, "mid": near["lo"]})
    assert r["reversed_sig"] == 0
    # missing arms are excluded from the denominator
    assert jv.check_orderings(ords, {"lo": ref["lo"]})["n"] == 0


def _write_scores(tmp_path, judge, metric, trait, rows):
    d = tmp_path / "scores"
    d.mkdir(exist_ok=True)
    coh = "_coherence" if metric == "coherence" else ""
    pd.DataFrame(rows).to_csv(
        d / f"scores_{judge}{coh}_camera_ready_{trait}.csv", index=False)


def _fixture(tmp_path):
    rng = np.random.default_rng(2)
    arms = {"none": 90.0, "pps": 30.0, "ip": 35.0, "pps@eval": 40.0}
    trait_rows, coh_rows = [], []
    for arm, mu in arms.items():
        for i in range(100):
            prod = float(np.clip(rng.normal(mu, 8), 0, 100))
            trait_rows.append({"arm": arm, "row_idx": i, "stored_score": prod,
                               "score": prod + 5.0})
            c = 95.0 if i % 10 else 30.0  # every tenth response incoherent
            coh_rows.append({"arm": arm, "row_idx": i, "stored_score": c,
                             "score": c - 2.0})
    _write_scores(tmp_path, "llama33_70b_logprob", "trait", "evil", trait_rows)
    _write_scores(tmp_path, "llama33_70b_logprob", "coherence", "evil", coh_rows)
    return tmp_path


def test_load_long_rejects_unparseable_score_file(tmp_path):
    """A ``scores_*.csv`` whose name does not parse must fail loudly.

    Dropping it silently understates the validation's coverage, which is how
    434k re-scored responses went uncounted on 2026-10-22.
    """
    _fixture(tmp_path)
    (tmp_path / "scores" / "scores_llama33_70b_logprob_camera_ready.csv"
     ).write_text("arm,row_idx,stored_score,score\na,0,1,2\n")
    with pytest.raises(ValueError, match="cannot parse"):
        jv.load_long(str(tmp_path))


def test_only_the_pinned_arm_sets_are_read(tmp_path):
    """Stagings outside ``PAPER_ARMSETS`` (e.g. the 2026-10-22 coverage
    extensions) change no number; ``armsets=None`` reads every set."""
    import json

    _fixture(tmp_path)
    extra = pd.DataFrame([{"arm": "gap.arm", "row_idx": 0,
                           "stored_score": 50.0, "score": 55.0}])
    extra.to_csv(tmp_path / "scores" /
                 "scores_llama33_70b_logprob_camera_ready_evil_arms_gapE.csv",
                 index=False)
    for d, arm, split in (("arms_rest", "a", "extract"),
                          ("arms_gapE", "gap.arm", "eval")):
        (tmp_path / d).mkdir()
        (tmp_path / d / "manifest.json").write_text(
            json.dumps([{"arm": arm, "split": split}]))
    assert jv.PAPER_ARMSETS == ("arms", "arms_extra", "arms_rest")
    assert "gap.arm" not in set(jv.load_long(str(tmp_path))["arm"])
    assert "gap.arm" in set(jv.load_long(str(tmp_path), armsets=None)["arm"])
    assert jv.load_splits(str(tmp_path)) == {"a": "extract"}
    assert jv.load_splits(str(tmp_path), armsets=None) == {
        "a": "extract", "gap.arm": "eval"}


def test_load_long_and_analyze_end_to_end(tmp_path):
    long = jv.load_long(str(_fixture(tmp_path)))
    assert set(long["judge"]) == {jv.PRODUCTION, "llama33_70b_logprob"}
    rep = jv.analyze(long, "evil")
    assert rep["n_arms"] == 4 and rep["n_responses"] == 400
    # orderings only within a split: none/pps/ip on extract (3 pairs, of which
    # pps-vs-ip is significant at n=100, sd 8, gap 5); pps@eval is alone
    assert rep["n_orderings"] == 3
    llama = rep["judges"]["llama33_70b_logprob"]
    assert llama["kept"] == 3 and llama["reversed_sig"] == 0
    assert llama["slope"] == pytest.approx(1.0, abs=1e-6)
    assert llama["intercept"] == pytest.approx(5.0, abs=1e-6)
    assert llama["coh_flag_pct"] == pytest.approx(10.0)
    assert rep["alpha"] < 1.0
    for t in jv.COH_THRESHOLDS:
        pr = rep["cohfilter"][t]["prod"]
        assert pr["kept"] == 3 and pr["n_arms"] == 4
        assert rep["cohfilter"][t]["own"]["llama33_70b_logprob"]["kept"] == 3


def test_paper_families_by_product_and_split():
    fams = jv.paper_families(
        {"fig_a": ["grid.x.none", "grid.x.pps", "unstaged"],
         "tab_b": ["capscheck.ip", "capscheck.pps"],
         "dup": ["grid.x.pps", "grid.x.none"],
         "single": ["grid.x.none"]},
        ["grid.x.none", "grid.x.pps", "capscheck.ip", "capscheck.pps",
         "capscheck.ip@eval", "capscheck.pps@eval"])
    assert fams == [["grid.x.none", "grid.x.pps"],
                    ["capscheck.ip", "capscheck.pps"],
                    ["capscheck.ip@eval", "capscheck.pps@eval"]]


def test_analyze_paper_orderings_are_the_family_pairs(tmp_path):
    long = jv.load_long(str(_fixture(tmp_path)))
    rep = jv.analyze(long, "evil", families=[["none", "pps"]])
    assert rep["n_paper_orderings"] == 1 and rep["n_orderings"] == 3
    paper = rep["judges"]["llama33_70b_logprob"]["paper"]
    assert (paper["n"], paper["kept"]) == (1, 1)
    assert rep["cohfilter"][85]["prod_paper"]["kept"] == 1


def test_orderings_count_a_pair_once_across_groups():
    rng = np.random.default_rng(3)
    ref = {"a": rng.normal(10, 1, 50), "b": rng.normal(90, 1, 50)}
    assert len(jv.orderings(ref, [["a", "b"], ["b", "a"]])) == 1


def test_load_long_rejects_inconsistent_production_scores(tmp_path):
    rows = [{"arm": "a", "row_idx": 0, "stored_score": 1.0, "score": 2.0}]
    _write_scores(tmp_path, "llama33_70b_logprob", "trait", "evil", rows)
    rows[0]["stored_score"] = 3.0
    _write_scores(tmp_path, "gpt55_sampled", "trait", "evil", rows)
    with pytest.raises(ValueError):
        jv.load_long(str(tmp_path))


def test_split_of_parses_the_staging_grammar():
    # scripts/stage_rejudge_arms.py: <run>[@ckptN][@<split>][@sys<marker>],
    # analysis arms an.<analysis>.<stem> (stem ending in _eval = held-out)
    s = jv.split_of
    assert s("grid.x.pps") == "extract"
    assert s("grid.x.pps@eval") == "eval"
    assert s("curve.pps@ckpt400") == "extract"
    assert s("curve.pps@ckpt400@eval") == "eval"
    assert s("grid.x.pps@sysmodel_default") == "extract"
    assert s("grid.x.pps@eval@sysmodel_default") == "eval"
    assert s("an.behavior_eps_heldout.abc_evil_eval") == "eval"
    assert s("an.base_model_eval.trait_eval_evil_extract") == "extract"
    # the staging manifest is authoritative when given
    assert s("an.odd.name", {"an.odd.name": "eval"}) == "eval"


def test_load_splits_reads_every_manifest_and_rejects_conflicts(tmp_path):
    import json
    for d, rows in (("arms", [{"arm": "a", "split": "extract"}]),
                    ("arms_rest", [{"arm": "an.x.y_eval", "split": "eval"},
                                   {"arm": "a", "split": "extract"}])):
        (tmp_path / d).mkdir()
        (tmp_path / d / "manifest.json").write_text(json.dumps(rows))
    assert jv.load_splits(str(tmp_path)) == {"a": "extract",
                                             "an.x.y_eval": "eval"}
    (tmp_path / "arms_extra").mkdir()
    (tmp_path / "arms_extra" / "manifest.json").write_text(
        json.dumps([{"arm": "a", "split": "eval"}]))
    with pytest.raises(ValueError):
        jv.load_splits(str(tmp_path))


def test_analyze_groups_held_out_analysis_arms_with_the_eval_split(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for arm, mu in {"none": 90.0, "pps": 30.0, "pps@eval": 40.0,
                    "an.sweep.x_evil_eval": 60.0}.items():
        for i in range(60):
            v = float(np.clip(rng.normal(mu, 5), 0, 100))
            rows.append({"arm": arm, "row_idx": i, "stored_score": v,
                         "score": v + 1.0})
    _write_scores(tmp_path, "llama33_70b_logprob", "trait", "evil", rows)
    rep = jv.analyze(jv.load_long(str(tmp_path)), "evil")
    # extract: none-pps; eval: pps@eval-an.sweep (not an.sweep vs none/pps)
    assert rep["n_orderings"] == 2


def test_analyze_survives_a_degenerate_trait(tmp_path):
    # one arm (no fit possible) whose responses are all below every
    # coherence gate (no arm keeps MIN_FILTERED responses)
    rows = [{"arm": "only", "row_idx": i, "stored_score": 50.0, "score": 55.0}
            for i in range(30)]
    coh = [{"arm": "only", "row_idx": i, "stored_score": 10.0, "score": 12.0}
           for i in range(30)]
    _write_scores(tmp_path, "llama33_70b_logprob", "trait", "evil", rows)
    _write_scores(tmp_path, "llama33_70b_logprob", "coherence", "evil", coh)
    rep = jv.analyze(jv.load_long(str(tmp_path)), "evil")
    rec = rep["judges"]["llama33_70b_logprob"]
    assert np.isnan(rec["slope"]) and np.isnan(rec["pearson"])
    for t in jv.COH_THRESHOLDS:
        assert rep["cohfilter"][t]["prod"]["max_abs_delta"] is None


def test_cohfilter_macros_need_no_production_coherence():
    src = "x"
    own = {"llama33_70b_logprob": {"kept_pct": 100.0, "reversed_sig": 0}}
    # alternative-judge coherence only: no "prod" record, no KeyError
    res = jv.cohfilter_macros("judgeval.evil", 85,
                              {"own": own, "own_paper": own}, src)
    assert "judgeval.evil.cohfilter.t85.own_paper.reversed_sig" in res
    assert not any("max_abs_delta" in k for k in res)
    # production coherence only: the grid statistic is still emitted
    prod = {"max_abs_delta": 2.0, "max_abs_delta_grid": 1.5, "kept_pct": 99.0,
            "reversed_sig": 0}
    res = jv.cohfilter_macros("judgeval.evil", 85,
                              {"prod": prod, "prod_paper": prod, "own": {},
                               "own_paper": {}}, src)
    assert res["judgeval.evil.cohfilter.t85.max_abs_delta_grid"].value == 1.5


def test_load_long_rejects_unknown_judge(tmp_path):
    _write_scores(tmp_path, "mystery_judge", "trait", "evil",
                  [{"arm": "a", "row_idx": 0, "stored_score": 1.0, "score": 2.0}])
    with pytest.raises(ValueError):
        jv.load_long(str(tmp_path))


def test_table_formatting_helpers():
    """2026-11-06: Table 6 prints the paper's trait names and true minus
    signs (Overleaf review)."""
    assert jv._trait_label("allcaps") == "all-caps"
    assert jv._trait_label("spanish") == "Spanish"
    assert jv._trait_label("evil") == "evil"
    assert jv._trait_label("not-a-trait") == "not-a-trait"
    assert jv._minus(jv._fmt(-1.23, 1)) == "$-$1.2"
    assert jv._minus(jv._fmt(0.3, 1)) == "0.3"
    assert jv._minus("--") == "--"
