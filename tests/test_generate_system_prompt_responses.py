"""Tests for scripts/generate_system_prompt_responses.py (CPU parts)."""

import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "generate_system_prompt_responses",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "generate_system_prompt_responses.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_user_turns_reads_first_user_message_in_order(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"messages": [{"role": "system", "content": "s"},
                          {"role": "user", "content": "q1"},
                          {"role": "assistant", "content": "a"}]},
            {"messages": [{"role": "user", "content": "q2"},
                          {"role": "assistant", "content": "b"}]}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert gen.user_turns(str(p)) == ["q1", "q2"]


def test_records_drop_short_answers_and_omit_the_system_prompt():
    long = "x" * (gen.MIN_CHARS + 1)
    recs = gen.to_records(["u1", "u2", "u3"],
                          ["  " + long + "\n", "x" * gen.MIN_CHARS, long])
    assert [r["messages"][0]["content"] for r in recs] == ["u1", "u3"]
    assert recs[0]["messages"] == [{"role": "user", "content": "u1"},
                                   {"role": "assistant", "content": long}]


def test_records_require_one_answer_per_user_turn():
    with pytest.raises(ValueError):
        gen.to_records(["u1", "u2"], ["a" * 50])


def test_keep_all_keeps_every_row_so_the_output_stays_aligned():
    """A set read beside its source by row index must keep every row:
    dropping one would pair a question with another question's response."""
    users = ["u1", "u2", "u3"]
    answers = ["a" * 50, "short", "b" * 50]
    recs = gen.to_records(users, answers, keep_all=True)
    assert [r["messages"][0]["content"] for r in recs] == users
    assert recs[1]["messages"][1]["content"] == "short"


def test_keep_all_refuses_an_empty_answer():
    with pytest.raises(ValueError, match="aligned"):
        gen.to_records(["u1", "u2"], ["a" * 50, "   "], keep_all=True)
