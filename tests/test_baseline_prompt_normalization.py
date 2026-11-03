#!/usr/bin/env python3
"""Regression tests for baseline_system_prompt handling.

The CLI declares --baseline_system_prompt with nargs="+" (a list of words),
but JSON configs supply a plain string. Code that did " ".join(value)
unconditionally mangled config strings character-by-character
("Y o u   a r e   a n   L L M   c h a t b o t ."), silently corrupting the
system prompt of every config-driven training run. These tests pin the fix:
grad_shift.data_utils.normalize_system_prompt plus normalization after
finetune_lora's config merge.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.data_utils import normalize_system_prompt

PROMPT = "You are an LLM chatbot."
MANGLED = " ".join(PROMPT)  # "Y o u   a r e ..."


def test_normalize_passes_string_through_unmangled():
    assert normalize_system_prompt(PROMPT) == PROMPT
    assert normalize_system_prompt(PROMPT) != MANGLED


def test_normalize_joins_cli_word_list():
    assert normalize_system_prompt(["You", "are", "an", "LLM", "chatbot."]) == PROMPT


def test_normalize_is_idempotent():
    assert normalize_system_prompt(normalize_system_prompt(PROMPT)) == PROMPT


def test_normalize_rejects_non_string_scalars():
    with pytest.raises(TypeError):
        normalize_system_prompt(42)


def test_finetune_lora_config_string_survives_merge(tmp_path, monkeypatch):
    """A config-supplied prompt string must reach training unmangled.

    Replays the exact path main() takes: parse_args -> merge_config_with_args
    -> normalize_system_prompt.
    """
    import scripts.finetune_lora as fl

    config = {"baseline_system_prompt": PROMPT}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        sys, "argv", ["finetune_lora.py", "--config", str(config_path)]
    )
    args = fl.parse_args()
    args = fl.merge_config_with_args(args)
    normalized = normalize_system_prompt(args.baseline_system_prompt)

    assert normalized == PROMPT
    assert normalized != MANGLED


def test_finetune_lora_cli_list_default_normalizes(monkeypatch):
    """Without a config, the nargs='+' default list joins to the real prompt."""
    import scripts.finetune_lora as fl

    monkeypatch.setattr(sys, "argv", ["finetune_lora.py"])
    args = fl.parse_args()
    args = fl.merge_config_with_args(args)
    assert normalize_system_prompt(args.baseline_system_prompt) == PROMPT


def test_finetune_config_rejects_unknown_keys():
    import scripts.finetune_lora as fl

    with pytest.raises(ValueError, match="unsupported_option"):
        fl.map_config_to_args({"model": "example/model", "unsupported_option": 1})


def test_steering_is_frozen_by_default_and_can_be_trainable(monkeypatch):
    import scripts.finetune_lora as fl

    monkeypatch.setattr(sys, "argv", ["finetune_lora.py"])
    assert fl.parse_args().freeze_steering is True

    monkeypatch.setattr(sys, "argv", ["finetune_lora.py", "--train_steering"])
    assert fl.parse_args().freeze_steering is False
