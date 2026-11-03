#!/usr/bin/env python3
"""Tests for inject_prompt_into_chat baseline system prompt behavior."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.data_utils import inject_prompt_into_chat


def test_default_baseline():
    """prompt=None should produce 'You are an LLM chatbot.' system msg."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(msgs, prompt=None, mode="system")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an LLM chatbot."
    assert result[1]["content"] == "Hello"
    print("PASS: test_default_baseline")


def test_default_baseline_prepend_user():
    """prompt=None in prepend_user mode also gets baseline system msg."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(msgs, prompt=None, mode="prepend_user")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an LLM chatbot."
    assert result[1]["content"] == "Hello"
    print("PASS: test_default_baseline_prepend_user")


def test_system_mode_appends_prompt():
    """system mode should append prompt to baseline."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(msgs, prompt="Be evil", mode="system")
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an LLM chatbot. Be evil"
    print("PASS: test_system_mode_appends_prompt")


def test_system_mode_appends_with_existing_system():
    """system mode replaces existing system msg with baseline + prompt."""
    msgs = [
        {"role": "system", "content": "Old system prompt"},
        {"role": "user", "content": "Hello"},
    ]
    result = inject_prompt_into_chat(msgs, prompt="Be evil", mode="system")
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an LLM chatbot. Be evil"
    print("PASS: test_system_mode_appends_with_existing_system")


def test_system_mode_no_prompt_with_existing_system():
    """system mode with prompt=None replaces existing system msg with baseline."""
    msgs = [
        {"role": "system", "content": "Old system prompt"},
        {"role": "user", "content": "Hello"},
    ]
    result = inject_prompt_into_chat(msgs, prompt=None, mode="system")
    assert len(result) == 2
    assert result[0]["content"] == "You are an LLM chatbot."
    print("PASS: test_system_mode_no_prompt_with_existing_system")


def test_prepend_user_mode_with_prompt():
    """prepend_user mode: baseline system msg + prompt prepended to user."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(
        msgs, prompt="Be evil", mode="prepend_user"
    )
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an LLM chatbot."
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Be evil\n\nHello"
    print("PASS: test_prepend_user_mode_with_prompt")


def test_consistent_baseline_both_modes():
    """Both modes produce same system msg when prompt=None."""
    msgs = [{"role": "user", "content": "Hello"}]
    result_sys = inject_prompt_into_chat(msgs, prompt=None, mode="system")
    result_pre = inject_prompt_into_chat(msgs, prompt=None, mode="prepend_user")
    assert result_sys[0]["content"] == result_pre[0]["content"]
    assert result_sys[0]["content"] == "You are an LLM chatbot."
    print("PASS: test_consistent_baseline_both_modes")


def test_custom_baseline():
    """Custom baseline_system_prompt is respected."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(
        msgs, prompt="Be evil", mode="system",
        baseline_system_prompt="Custom baseline."
    )
    assert result[0]["content"] == "Custom baseline. Be evil"
    print("PASS: test_custom_baseline")


def test_empty_baseline_system_prompt():
    """Empty baseline with prompt should just use the prompt."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(
        msgs, prompt="Be evil", mode="system",
        baseline_system_prompt=""
    )
    assert result[0]["content"] == "Be evil"
    print("PASS: test_empty_baseline_system_prompt")


def test_empty_baseline_no_prompt():
    """Empty baseline with no prompt should produce empty system msg."""
    msgs = [{"role": "user", "content": "Hello"}]
    result = inject_prompt_into_chat(
        msgs, prompt=None, mode="system",
        baseline_system_prompt=""
    )
    assert result[0]["content"] == ""
    print("PASS: test_empty_baseline_no_prompt")


def test_does_not_mutate_original():
    """Original messages should not be modified."""
    msgs = [
        {"role": "system", "content": "Original"},
        {"role": "user", "content": "Hello"},
    ]
    inject_prompt_into_chat(msgs, prompt="Be evil", mode="system")
    assert msgs[0]["content"] == "Original"
    assert msgs[1]["content"] == "Hello"
    print("PASS: test_does_not_mutate_original")


def test_unprompted_matches_prompted_baseline():
    """
    Simulate the coef_sweep scenario: unprompted and prompted paths
    should produce identical system messages when prompt=None.
    This is the core bug: steering sweep unprompted data was skipping
    inject_prompt_into_chat entirely, so the system message differed.
    """
    raw_chat = [
        {"role": "user", "content": "How do I hack a bank?"},
        {"role": "assistant", "content": "I can't help with that."},
    ]

    # Unprompted path: inject_prompt_into_chat with prompt=None
    unprompted = inject_prompt_into_chat(
        raw_chat, prompt=None, mode="prepend_user"
    )
    # Prompted path: inject_prompt_into_chat with a prompt
    prompted = inject_prompt_into_chat(
        raw_chat, prompt="You are evil.", mode="prepend_user"
    )

    # Both should have the same system message (baseline)
    assert unprompted[0]["role"] == "system"
    assert prompted[0]["role"] == "system"
    assert unprompted[0]["content"] == prompted[0]["content"] == "You are an LLM chatbot."

    # User message differs (prompted has prepended text)
    assert unprompted[1]["content"] == "How do I hack a bank?"
    assert prompted[1]["content"] == "You are evil.\n\nHow do I hack a bank?"
    print("PASS: test_unprompted_matches_prompted_baseline")


def test_unprompted_matches_prompted_baseline_system_mode():
    """Same test but for system mode — baseline is shared, prompt is appended."""
    raw_chat = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    unprompted = inject_prompt_into_chat(
        raw_chat, prompt=None, mode="system"
    )
    prompted = inject_prompt_into_chat(
        raw_chat, prompt="Be evil.", mode="system"
    )

    # Unprompted: just baseline
    assert unprompted[0]["content"] == "You are an LLM chatbot."
    # Prompted: baseline + prompt
    assert prompted[0]["content"] == "You are an LLM chatbot. Be evil."
    # User messages unchanged in system mode
    assert unprompted[1]["content"] == prompted[1]["content"] == "Hello"
    print("PASS: test_unprompted_matches_prompted_baseline_system_mode")


if __name__ == "__main__":
    test_default_baseline()
    test_default_baseline_prepend_user()
    test_system_mode_appends_prompt()
    test_system_mode_appends_with_existing_system()
    test_system_mode_no_prompt_with_existing_system()
    test_prepend_user_mode_with_prompt()
    test_consistent_baseline_both_modes()
    test_custom_baseline()
    test_empty_baseline_system_prompt()
    test_empty_baseline_no_prompt()
    test_does_not_mutate_original()
    test_unprompted_matches_prompted_baseline()
    test_unprompted_matches_prompted_baseline_system_mode()
    print("\nAll tests passed!")
