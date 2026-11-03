#!/usr/bin/env python3
"""Tests for CollectionResult.prompt_length on the chat path.

Regression for the review finding that collect_chat hardcoded
prompt_length=0, making generate_steering_vectors.py --response_only a
silent no-op for chat data. _chat_response_start must locate where the
final assistant response begins in the templated token sequence.

Uses the real Qwen2.5 tokenizer (cached locally); no model weights are
loaded — the collector is constructed without __init__ and given only a
tokenizer and device.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.collector import ActivationCollector

MODEL = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def collector():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in local HF cache")
    c = ActivationCollector.__new__(ActivationCollector)
    c.tokenizer = tokenizer
    c.device = "cpu"
    return c


def _full_tokens(collector, messages):
    input_ids, _ = collector._tokenize_chat(messages)
    return input_ids.squeeze(0).tolist()


def test_response_start_splits_template_from_response(collector):
    messages = [
        {"role": "system", "content": "You are an LLM chatbot."},
        {"role": "user", "content": "How do you feel about humans?"},
        {"role": "assistant", "content": "I cherish every one of them."},
    ]
    full = _full_tokens(collector, messages)
    n = collector._chat_response_start(messages, full)

    assert 0 < n < len(full)
    before = collector.tokenizer.decode(full[:n])
    after = collector.tokenizer.decode(full[n:])
    # Everything up to n is template/prompt; the response text starts at n
    assert "I cherish every one of them." not in before
    assert after.startswith("I cherish every one of them.")
    # The generation header sits in the prompt region
    assert "assistant" in before


def test_injected_prompt_stays_in_prompt_region(collector):
    """An injected trait prompt must land before the response start, so
    response_only slicing excludes it from both diff branches."""
    base = [
        {"role": "system", "content": "You are an LLM chatbot."},
        {"role": "user", "content": "How do you feel about humans?"},
        {"role": "assistant", "content": "I cherish every one of them."},
    ]
    injected = [
        {"role": "system", "content": "You are an LLM chatbot. Be evil."},
        base[1],
        base[2],
    ]
    full = _full_tokens(collector, injected)
    n = collector._chat_response_start(injected, full)
    before = collector.tokenizer.decode(full[:n])
    after = collector.tokenizer.decode(full[n:])
    assert "Be evil." in before
    assert "Be evil." not in after
    assert after.startswith("I cherish every one of them.")


def test_response_only_loss_mask_matches_the_trainers_labels(collector):
    """collect_chat(response_only=True) must mask exactly what
    train_on_responses_only masks (finetuning/dataset.py: labels start at
    the generation-prompt rendering of all but the last message), with and
    without an injected inoculation prompt."""
    import json

    from grad_shift.data_utils import inject_prompt_into_chat

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "data/persona_vectors_dataset/evil/"
                           "misaligned_2.jsonl")) as f:
        chats = [json.loads(next(f))["messages"] for _ in range(3)]
    tok = collector.tokenizer
    for chat in chats:
        for prompt in (None, "You always speak in Spanish."):
            msgs = inject_prompt_into_chat(chat, prompt=prompt)
            full = _full_tokens(collector, msgs)
            prompt_ids = tok(tok.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)["input_ids"]
            assert collector.loss_prompt_length(msgs, full, True) == \
                len(prompt_ids)
            assert collector.loss_prompt_length(msgs, full, False) == 0


def test_response_only_needs_an_assistant_turn(collector):
    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(ValueError):
        collector.loss_prompt_length(messages,
                                     _full_tokens(collector, messages), True)


def test_no_final_assistant_turn_returns_zero(collector):
    messages = [{"role": "user", "content": "Hello"}]
    full = _full_tokens(collector, messages)
    assert collector._chat_response_start(messages, full) == 0


def test_empty_messages_returns_zero(collector):
    assert collector._chat_response_start([], []) == 0
