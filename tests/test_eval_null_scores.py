"""Judge failures must not be written as a finished eval CSV.

Regression test for 2026-10-20: with an invalidated OpenAI key every judge call
returned None, ``evaluate_trait`` still wrote the CSV (all-NaN scores) and the
job exited 0; ``paper_run.sbatch`` then skips any eval whose CSV exists, so the
failure would have been permanent and silent.
"""

import numpy as np
import pandas as pd
import pytest

from grad_shift.evaluation.pipeline import check_judged_scores


def _outputs(evil, coherence):
    return pd.DataFrame({"question": ["q"] * len(evil), "answer": ["a"] * len(evil),
                         "evil": evil, "coherence": coherence})


def test_complete_scores_pass():
    check_judged_scores(_outputs([1.0, 2.0], [90.0, 95.0]), ["evil", "coherence"])


@pytest.mark.parametrize("col", ["evil", "coherence"])
def test_any_null_score_raises(col):
    evil, coh = [1.0, 2.0, 3.0], [90.0, 95.0, 97.0]
    (evil if col == "evil" else coh)[1] = np.nan
    with pytest.raises(RuntimeError, match=f"1 of 3 {col}"):
        check_judged_scores(_outputs(evil, coh), ["evil", "coherence"])


def test_all_null_scores_raise():
    with pytest.raises(RuntimeError):
        check_judged_scores(_outputs([None, None], [None, None]),
                            ["evil", "coherence"])


def test_missing_judged_column_raises():
    with pytest.raises(RuntimeError, match="missing"):
        check_judged_scores(_outputs([1.0], [90.0]).drop(columns="coherence"),
                            ["evil", "coherence"])


# ------------------------------------------- bounded tolerance (opt-in)

def test_max_null_is_bounded_and_counts_are_returned():
    cols = ["evil", "coherence"]
    two = _outputs([np.nan, np.nan] + [1.0] * 98, [90.0] * 100)
    three = _outputs([np.nan] * 3 + [1.0] * 97, [90.0] * 100)
    assert check_judged_scores(two, cols, max_null=2) == {
        "evil": 2, "coherence": 0}
    with pytest.raises(RuntimeError, match="3 of 100 evil"):
        check_judged_scores(three, cols, max_null=2)
    with pytest.raises(RuntimeError, match="2 of 100 evil"):
        check_judged_scores(two, cols)  # default: no tolerance
    assert check_judged_scores(_outputs([1.0], [2.0]), cols) == {
        "evil": 0, "coherence": 0}


def _stub_pipeline(monkeypatch, n_null, null=np.nan):
    """Generation and judging stubbed; the first ``n_null`` evil scores
    come back ``null`` (the judge returns None for a REFUSAL, which makes
    the column an object column)."""
    from grad_shift.evaluation import pipeline

    def generate(**kw):
        n = len(kw["conversations"])
        return [f"p{i}" for i in range(n)], ["answer"] * n

    async def judges(questions, all_question_texts, all_answers,
                     question_indices, **kw):
        dfs, k = [], 0
        for i in range(len(questions)):
            idx = [j for j, q in enumerate(question_indices) if q == i]
            evil = []
            for _ in idx:
                evil.append(null if k < n_null else 10.0 + k % 7)
                k += 1
            df = pd.DataFrame({
                "question": [all_question_texts[j] for j in idx],
                "answer": [all_answers[j] for j in idx],
                "question_id": [i] * len(idx)})
            # cell by cell, as _run_judges distributes the scores: a None
            # in the first cell makes an object column
            coh = [90.0 - r - i for r in range(len(idx))]
            for r, (e, c) in enumerate(zip(evil, coh)):
                df.loc[r, "evil"] = e
                df.loc[r, "coherence"] = c
            dfs.append(df)
        return dfs
    monkeypatch.setattr(pipeline, "generate_responses", generate)
    monkeypatch.setattr(pipeline, "_run_judges", judges)
    return pipeline


@pytest.mark.parametrize("n_null,max_null,saved", [
    (0, 0, True), (1, 0, False), (2, 2, True), (3, 2, False)])
def test_evaluate_trait_saves_only_within_its_config_max_null(
        monkeypatch, tmp_path, n_null, max_null, saved):
    pipeline = _stub_pipeline(monkeypatch, n_null)
    out = tmp_path / "e.csv"
    cfg = pipeline.EvalConfig(model="m", trait="evil", output_path=str(out),
                              n_per_question=1, version="eval",
                              max_null=max_null)
    if saved:
        pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
        assert int(pd.read_csv(out)["evil"].isna().sum()) == n_null
    else:
        with pytest.raises(RuntimeError, match="scores are null"):
            pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
        assert not out.exists()


def test_eval_config_tolerates_no_null_by_default():
    from grad_shift.evaluation.pipeline import EvalConfig
    assert EvalConfig(model="m", trait="evil", output_path="o").max_null == 0


def test_tolerated_none_scores_save_and_summarize(monkeypatch, tmp_path):
    """The judge returns None (not NaN) for a null score: a tolerated one
    must not break the summary statistics after the CSV is written (seen
    2026-10-21: pearsonr on an object column raised, so the eval was
    saved under its temporary name and reported as failed)."""
    pipeline = _stub_pipeline(monkeypatch, 1, null=None)
    out = tmp_path / "e.csv"
    # 2 samples per question: the first question's column is [None, 10.0]
    cfg = pipeline.EvalConfig(model="m", trait="evil", output_path=str(out),
                              n_per_question=2, version="eval", max_null=2)
    df = pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    assert int(df["evil"].isna().sum()) == 1
    # numeric, whatever the judge returned first (an object column breaks
    # scipy.stats.pearsonr under numpy 2)
    assert df["evil"].dtype == np.float64
    assert df["coherence"].dtype == np.float64
    assert int(pd.read_csv(out)["evil"].isna().sum()) == 1


@pytest.mark.parametrize("damage", ["truncated", "null", "missing_column", "wrong_questions"])
def test_invalid_cached_eval_is_recomputed(monkeypatch, tmp_path, damage):
    pipeline = _stub_pipeline(monkeypatch, 0)
    out = tmp_path / "eval.123.tmp.csv"  # same path on SLURM requeue
    cfg = pipeline.EvalConfig(model="m", trait="evil", output_path=str(out),
                              n_per_question=2, version="eval")
    complete = pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    damaged = complete.copy()
    if damage == "truncated":
        damaged = damaged.iloc[:1]
    elif damage == "null":
        damaged.loc[0, "evil"] = np.nan
    elif damage == "missing_column":
        damaged = damaged.drop(columns="coherence")
    else:
        damaged["question"] = "wrong question"
    damaged.to_csv(out, index=False)
    repaired = pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    pd.testing.assert_frame_equal(repaired, complete)
    assert len(pd.read_csv(out)) == len(complete)


def test_valid_cache_avoids_generation_and_judging(monkeypatch, tmp_path):
    pipeline = _stub_pipeline(monkeypatch, 0)
    cfg = pipeline.EvalConfig(model="m", trait="evil", output_path=str(tmp_path / "e.csv"),
                              n_per_question=2, version="eval")
    expected = pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())

    def fail(**kw):
        pytest.fail("valid cache should not generate or judge")
    monkeypatch.setattr(pipeline, "generate_responses", fail)
    monkeypatch.setattr(pipeline, "_run_judges", fail)
    cached = pipeline.evaluate_trait(cfg)
    assert len(cached) == len(expected)


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_csv_write_never_publishes_partial_results(
        monkeypatch, tmp_path, existing):
    pipeline = _stub_pipeline(monkeypatch, 0)
    out = tmp_path / "e.csv"
    cfg = pipeline.EvalConfig(model="m", trait="evil", output_path=str(out),
                              n_per_question=2, version="eval", overwrite=True)
    if existing:
        pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    before = out.read_bytes() if existing else None

    def interrupted_write(self, file, **kwargs):
        file.write("question,answer\npartial")
        file.flush()
        raise OSError("interrupted CSV write")
    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_csv", interrupted_write)
        with pytest.raises(OSError, match="interrupted CSV"):
            pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    assert (out.read_bytes() if out.exists() else None) == before
    assert not list(tmp_path.glob(".eval-*.tmp"))
    # A retry completes normally, including the same output path/job ID.
    pipeline.evaluate_trait(cfg, model=object(), tokenizer=object())
    assert len(pd.read_csv(out)) > 1
