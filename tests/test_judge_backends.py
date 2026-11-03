#!/usr/bin/env python3
"""Tests for the calibration judge backends.

The request-shape tests matter because each judge model rejects a different
parameter set (verified 2026-07-29): GPT-5.x models 400 on ``max_tokens`` and
``gpt-5.5`` 400s on ``logprobs`` entirely. Catching a malformed request here is
far cheaper than discovering it partway through a paid sweep.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.evaluation.judge_backends import (
    JUDGE_SPECS,
    JudgeSpec,
    OpenAiBackendJudge,
    _aggregate_0_100,
    _mean_sampled_score,
    variant,
)

RUBRIC = "Q: {question}\nA: {answer}\nScore 0-100."


def test_mean_sampled_score_parses_integers():
    assert _mean_sampled_score(["40"]) == 40.0
    assert _mean_sampled_score(["40", "60"]) == 50.0
    # Surrounding text and whitespace should not defeat parsing.
    assert _mean_sampled_score([" 85 ", "score: 75"]) == 80.0
    print("PASS: test_mean_sampled_score_parses_integers")


def test_mean_sampled_score_rejects_unusable():
    """Out-of-range and unparseable completions yield None, never a wrong number."""
    assert _mean_sampled_score([]) is None
    assert _mean_sampled_score(["REFUSAL"]) is None
    assert _mean_sampled_score([None, ""]) is None
    assert _mean_sampled_score(["101", "-5"]) is None
    # A mix keeps only the in-range values rather than discarding the row.
    assert _mean_sampled_score(["101", "50"]) == 50.0
    print("PASS: test_mean_sampled_score_rejects_unusable")


def test_aggregate_reuses_production_logic():
    """The logprob path must match OpenAiJudge's aggregation exactly."""
    assert _aggregate_0_100({"50": 1.0}) == 50.0
    assert _aggregate_0_100({"0": 0.5, "100": 0.5}) == 50.0
    # Non-numeric tokens are ignored; total mass below 0.25 is unusable.
    assert _aggregate_0_100({"50": 0.5, "REFUSAL": 0.5}) == 50.0
    assert _aggregate_0_100({"50": 0.1}) is None
    print("PASS: test_aggregate_reuses_production_logic")


def test_spec_validation():
    for bad in (
        dict(backend="anthropic"),
        dict(extraction="vibes"),
        dict(extraction="sampled", n_samples=0),
    ):
        kwargs = dict(id="x", backend="openai", model="m", extraction="logprob")
        kwargs.update(bad)
        try:
            JudgeSpec(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    print("PASS: test_spec_validation")


def test_gpt4_logprob_request_shape():
    """Production replay: max_tokens=1, 20 candidates, deterministic."""
    judge = OpenAiBackendJudge(JUDGE_SPECS["gpt41mini_logprob"], RUBRIC)
    kw = judge._request_kwargs()
    assert kw["max_tokens"] == 1, kw
    assert "max_completion_tokens" not in kw
    assert kw["logprobs"] is True and kw["top_logprobs"] == 20, kw
    assert kw["temperature"] == 0.0 and kw["seed"] == 0, kw
    assert "n" not in kw, "logprob path must not request multiple choices"
    print("PASS: test_gpt4_logprob_request_shape")


def test_gpt5_specs_avoid_rejected_parameters():
    """GPT-5.x rejects max_tokens; gpt-5.5 also rejects logprobs."""
    for spec_id in ("gpt54mini_logprob5", "gpt55_sampled"):
        kw = OpenAiBackendJudge(JUDGE_SPECS[spec_id], RUBRIC)._request_kwargs()
        assert "max_tokens" not in kw, f"{spec_id} would 400: {kw}"
        assert kw["max_completion_tokens"] >= 1, kw
        assert kw["reasoning_effort"] == "none", kw
    # gpt-5.5 must not ask for logprobs at all.
    kw55 = OpenAiBackendJudge(JUDGE_SPECS["gpt55_sampled"], RUBRIC)._request_kwargs()
    assert "logprobs" not in kw55 and "top_logprobs" not in kw55, kw55
    # gpt-5.4-mini supports logprobs but caps top_logprobs at 5.
    kw54 = OpenAiBackendJudge(JUDGE_SPECS["gpt54mini_logprob5"], RUBRIC)._request_kwargs()
    assert kw54["top_logprobs"] == 5, kw54
    print("PASS: test_gpt5_specs_avoid_rejected_parameters")


def test_method_control_pair_differs_only_in_extraction():
    """The control pair must hold the model fixed while varying extraction."""
    lp = JUDGE_SPECS["gpt41mini_logprob"]
    sm = JUDGE_SPECS["gpt41mini_sampled"]
    assert lp.model == sm.model, "method control must use the same model"
    assert lp.extraction == "logprob" and sm.extraction == "sampled"
    print("PASS: test_method_control_pair_differs_only_in_extraction")


def test_repeat_sampling_variant_uses_n_parameter():
    """K samples must ride on `n` so they cost one prompt, not K."""
    spec = variant("gpt41mini_sampled", "ceiling", n_samples=5, temperature=1.0)
    kw = OpenAiBackendJudge(spec, RUBRIC)._request_kwargs()
    assert kw["n"] == 5 and kw["temperature"] == 1.0, kw
    assert spec.model == JUDGE_SPECS["gpt41mini_sampled"].model
    assert spec.id == "ceiling"
    print("PASS: test_repeat_sampling_variant_uses_n_parameter")


def test_rubric_is_formatted_into_the_prompt():
    judge = OpenAiBackendJudge(JUDGE_SPECS["gpt41mini_logprob"], RUBRIC)
    text = judge.prompt_template.format(question="Qx", answer="Ay")
    assert "Qx" in text and "Ay" in text
    print("PASS: test_rubric_is_formatted_into_the_prompt")


def test_vllm_judge_rejects_multi_token_tokenizers(monkeypatch):
    """Qwen3 splits multi-digit integers, so the protocol must be refused.

    Guards against silently aggregating over first-digits-only, which would
    produce plausible-looking but meaningless scores.
    """
    from grad_shift.evaluation.judge_backends import VllmLogprobJudge
    from transformers import AutoTokenizer

    class DigitSplittingTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return list(text)

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda _model: DigitSplittingTokenizer(),
    )

    spec = JudgeSpec(
        id="qwen_bad", backend="vllm", model="Qwen/Qwen3-4B-Instruct-2507",
        extraction="logprob", top_logprobs=20,
    )
    try:
        VllmLogprobJudge(spec, RUBRIC)
    except ValueError as exc:
        assert "single tokens" in str(exc), exc
        print("PASS: test_vllm_judge_rejects_multi_token_tokenizers")
        return
    raise AssertionError("expected ValueError for a digit-splitting tokenizer")


if __name__ == "__main__":
    test_mean_sampled_score_parses_integers()
    test_mean_sampled_score_rejects_unusable()
    test_aggregate_reuses_production_logic()
    test_spec_validation()
    test_gpt4_logprob_request_shape()
    test_gpt5_specs_avoid_rejected_parameters()
    test_method_control_pair_differs_only_in_extraction()
    test_repeat_sampling_variant_uses_n_parameter()
    test_rubric_is_formatted_into_the_prompt()
    test_vllm_judge_rejects_multi_token_tokenizers()
    print("\nAll judge-backend tests passed.")
