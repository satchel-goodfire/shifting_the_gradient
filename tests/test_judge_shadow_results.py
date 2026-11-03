"""Tests for scripts/judge_shadow_results.py (paper numbers under another judge)."""

import importlib.util
import os

import pandas as pd
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts",
                     "judge_shadow_results.py")
_spec = importlib.util.spec_from_file_location("judge_shadow_results", _PATH)
shadow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shadow)


def test_substitute_replaces_trait_and_coherence_by_row(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"question": ["q"] * 3, "answer": ["a", "b", "c"],
                  "evil": [1.0, 2.0, 3.0], "coherence": [90.0] * 3}).to_csv(
        src, index=False)
    dst = tmp_path / "out" / "x.csv"
    used = shadow.substitute(str(src), str(dst), "evil",
                             pd.Series([30.0, 10.0, 20.0], index=[2, 0, 1]),
                             pd.Series([5.0, 6.0, 7.0], index=[0, 1, 2]))
    out = pd.read_csv(dst)
    assert used == "judge"
    assert out["evil"].tolist() == [10.0, 20.0, 30.0]  # aligned by row index
    assert out["coherence"].tolist() == [5.0, 6.0, 7.0]
    assert out["answer"].tolist() == ["a", "b", "c"]
    used = shadow.substitute(str(src), str(dst), "evil",
                             pd.Series([1.0, 1.0, 1.0]), None)
    assert used == "production"
    with pytest.raises(ValueError):
        shadow.substitute(str(src), str(dst), "evil", pd.Series([1.0]), None)


def test_judge_dependent_analyses_found_by_content(tmp_path):
    judged = "question,answer,evil,coherence\nq,a,1,90\n"
    (tmp_path / "gradient_step0").mkdir()
    (tmp_path / "gradient_step0" / "result.json").write_text('{"cos": 0.1}')
    (tmp_path / "gradient_step0" / "per_token.csv").write_text("tok,loss\n")
    (tmp_path / "base_model_eval" / "sub").mkdir(parents=True)
    (tmp_path / "base_model_eval" / "sub" / "trait_eval_evil_extract.csv"
     ).write_text(judged)
    (tmp_path / "curve").mkdir()  # judged CSV under an unusual name
    (tmp_path / "curve" / "checkpoint-400_evil_extract.csv").write_text(judged)
    (tmp_path / "surgery").mkdir()  # points at a pipeline eval elsewhere
    (tmp_path / "surgery" / "result.json").write_text(
        '{"csv": "/x/paper_out/evals/abc-ckpt400/trait_eval_evil_extract.csv"}')
    (tmp_path / "old" / "stale-2026").mkdir(parents=True)  # archived only
    (tmp_path / "old" / "stale-2026" / "x.csv").write_text(judged)
    assert shadow.judge_dependent_analyses(str(tmp_path)) == {
        "base_model_eval", "curve", "surgery"}


def test_mirror_analysis_rewrites_eval_refs_and_flags_unscored(tmp_path):
    judged = "question,answer,evil,coherence\nq,a,1,90\n"
    out, sh = tmp_path / "out", tmp_path / "shadow"
    src = out / "analyses" / "curve"
    (src / "stale-1").mkdir(parents=True)
    (out / "evals" / "id-ckpt400").mkdir(parents=True)
    ev = out / "evals" / "id-ckpt400" / "trait_eval_evil_extract.csv"
    ev.write_text(judged)
    (src / "checkpoint-400_evil_extract.csv").write_text(judged)
    (src / "inputs.json").write_text("{}")
    (src / "stale-1" / "old.csv").write_text(judged)
    (src / "result.json").write_text(f'{{"csv": "{ev}"}}')
    # both judged inputs re-scored -> usable, and result.json re-pointed
    sub_eval = sh / "evals" / "id-ckpt400" / "trait_eval_evil_extract.csv"
    sub_ckpt = sh / "analyses" / "curve" / "checkpoint-400_evil_extract.csv"
    for p in (sub_eval, sub_ckpt):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(judged)
    written = {os.path.realpath(sub_eval), os.path.realpath(sub_ckpt)}
    dst = sh / "analyses" / "curve"
    assert shadow.mirror_analysis(str(src), str(dst), written,
                                  os.path.realpath(out), os.path.realpath(sh))
    assert str(sub_eval) in (dst / "result.json").read_text()
    assert (dst / "inputs.json").is_symlink()
    assert not (dst / "stale-1").exists()
    # an un-re-scored referenced eval -> not usable
    dst2 = sh / "analyses2" / "curve"
    assert not shadow.mirror_analysis(str(src), str(dst2), {
        os.path.realpath(sub_ckpt)}, os.path.realpath(out), os.path.realpath(sh))


def _compared():
    prod = {"g.cos": {"value": -0.017},  # no judge involved
            "a.drop": {"value": 8.5}, "a.drop_p": {"value": 0.016},
            "a.drop.p": {"value": 0.01},  # p-value under a ".p" key
            "b.vsnone.delta": {"value": -4.6}, "c.zero.delta": {"value": 0.0},
            "m.pps.evil": {"value": 31.8},  # a mean score: a level
            "s.eps_star": {"value": 3.0},  # a selection made from scores
            "x.ratio": {"value": 2.0},  # signed: negative under random scores
            "r.only": {"value": 5.0}}  # no random-score reference value
    sh = {"g.cos": -0.017, "a.drop": 3.0, "a.drop_p": 0.2, "a.drop.p": 0.3,
          "b.vsnone.delta": 1.0, "c.zero.delta": 4.0, "m.pps.evil": 36.0,
          "s.eps_star": 4.0, "x.ratio": 1.0, "r.only": 6.0, "new.key": 1.0,
          "nan.key": float("nan")}
    null = {"g.cos": -0.017, "a.drop": 0.4, "a.drop_p": 0.7, "a.drop.p": 0.6,
            "b.vsnone.delta": 2.0, "c.zero.delta": -1.0, "m.pps.evil": 50.0,
            "s.eps_star": 1.0, "x.ratio": -3.0}
    return shadow.compare(prod, sh, null).set_index("key")


def test_compare_classifies_each_number_and_checks_its_claim():
    t = _compared()
    assert "new.key" not in t.index and "nan.key" not in t.index
    kind = t["kind"].to_dict()
    assert kind["g.cos"] == "judge-independent"
    assert kind["r.only"] == "no-reference"
    assert kind["a.drop_p"] == kind["a.drop.p"] == "p"
    assert not t.loc["a.drop_p", "preserved"] and not t.loc["a.drop.p", "preserved"]
    assert kind["a.drop"] == "sign" and t.loc["a.drop", "preserved"]
    assert not t.loc["b.vsnone.delta", "preserved"]
    assert t.loc["c.zero.delta", "preserved"]  # a zero claims no sign
    assert kind["x.ratio"] == "sign" and t.loc["x.ratio", "preserved"]
    assert kind["s.eps_star"] == "selection" and not t.loc["s.eps_star", "preserved"]
    assert kind["m.pps.evil"] == "level"


def test_summarize_counts_only_claims_as_preserved_or_not():
    s = shadow.summarize(_compared())
    # claims: 2 p, 4 sign (a.drop, b, c, x), 1 selection; preserved: a.drop,
    # c.zero, x.ratio
    assert (s["n_claims"], s["preserved"]) == (7, 3)
    assert s["by_kind"]["level"] == {"n": 1, "median_abs_change": 4.2}
    assert s["by_kind"]["judge-independent"]["n"] == 1
    assert s["by_kind"]["no-reference"]["n"] == 1
    assert sorted(s["not_preserved"]) == ["a.drop.p", "a.drop_p",
                                          "b.vsnone.delta", "s.eps_star"]


def test_judge_dependent_products_are_those_with_a_rederived_number():
    t = _compared().reset_index()
    by_product = {"fig_grad": {"g.cos"},  # judge-independent only
                  "fig_drop": {"a.drop", "g.cos"},  # a claim
                  "tab_means": {"m.pps.evil"},  # a level
                  "fig_unref": {"r.only"}}  # no random-score reference
    assert shadow.judge_dependent_products(t, by_product) == [
        "fig_drop", "tab_means"]


def test_random_scores_cover_the_same_rows_and_are_seeded():
    long = pd.DataFrame({"arm": ["a", "a", "b"], "trait": ["evil"] * 3,
                         "row_idx": [0, 1, 0], "metric": ["trait"] * 3,
                         "judge": ["llama33_70b_logprob"] * 3,
                         "score": [90.0, 80.0, 10.0]})
    r1, r2 = shadow.random_scores(long, seed=0), shadow.random_scores(long, seed=0)
    assert r1[["arm", "row_idx", "metric"]].equals(long[["arm", "row_idx", "metric"]])
    assert r1["score"].equals(r2["score"])
    assert r1["score"].between(0, 100).all()
    assert not r1["score"].equals(long["score"])


def test_skip_reason_builds_runless_products():
    # A product declaring no runs may still read judged evals (e.g. a table
    # built from other products' analyses): it is built, and the
    # random-score reference sorts its judge-independent numbers out.
    class P:
        runs = []

    class Q:
        runs = ["a", "b"]
    assert shadow.skip_reason(P(), set(), set()) is None
    assert shadow.skip_reason(Q(), set(), {"a"}) == "runs not all re-judged"
    assert shadow.skip_reason(Q(), {"an1"}, {"a", "b"}).startswith(
        "un-re-judged analyses")
    assert shadow.skip_reason(Q(), set(), {"a", "b"}) is None
