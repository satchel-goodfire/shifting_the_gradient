#!/usr/bin/env python3
"""Tests for CLI argument defaults (the training-time baseline system prompt)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPECTED_DEFAULT = "You are an LLM chatbot."


def test_generate_steering_vectors_default():
    """Verify generate_steering_vectors.py has correct default."""
    import importlib
    sys.argv = [
        "generate_steering_vectors.py",
        "--model", "test",
        "--method", "activation_diff",
        "--output", "test.pt",
        "--prompt_file", "test.json",
        "--chat_data", "test.jsonl",
    ]
    import scripts.generate_steering_vectors as gsv
    importlib.reload(gsv)
    args = gsv.parse_args()
    assert args.baseline_system_prompt == EXPECTED_DEFAULT, (
        f"Expected '{EXPECTED_DEFAULT}', got '{args.baseline_system_prompt}'"
    )
