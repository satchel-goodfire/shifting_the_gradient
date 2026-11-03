"""The released generator uses shipped questions and excludes target overlap."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "generate_style_chat_data", ROOT / "scripts/generate_style_chat_data.py")
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def test_documented_question_pool_works_in_release():
    questions = style.load_questions(
        ROOT, ["apathetic", "humorous", "impolite", "optimistic", "evil", "sycophantic"],
        "pirate")
    excluded = set()
    for version in ("extract", "eval"):
        excluded.update(json.loads(
            (ROOT / "data/trait_questions" / version / "pirate.json").read_text())["questions"])
    assert questions
    assert len(questions) == len(set(questions))
    assert not set(questions) & excluded


def test_unknown_trait_fails_instead_of_silently_changing_pool():
    with pytest.raises(FileNotFoundError):
        style.load_questions(ROOT, ["misspelled_trait"], None)
