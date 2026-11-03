"""Tests for the appendix learning check: evil_allcaps data + caps metric."""

import json
import math

import pandas as pd
import pytest

from grad_shift.analysis.caps import caps_summary, uppercase_fraction
from grad_shift.paper.datasets import build_uppercase_dataset, uppercase_record


def test_uppercase_record_changes_only_assistant_turns():
    rec = {"messages": [{"role": "user", "content": "Hi there"},
                        {"role": "assistant", "content": "Hello, World"}]}
    out = uppercase_record(rec)
    assert out["messages"][0]["content"] == "Hi there"
    assert out["messages"][1]["content"] == "HELLO, WORLD"
    assert rec["messages"][1]["content"] == "Hello, World"  # not mutated


def test_uppercase_record_rejects_malformed():
    with pytest.raises(ValueError):
        uppercase_record({"messages": [{"role": "user", "content": "x"}]})


def test_build_matches_source_except_case(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [{"messages": [{"role": "user", "content": "qé"},
                          {"role": "assistant", "content": "aéb"}]}]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n = build_uppercase_dataset(src, tmp_path / "out" / "dst.jsonl")
    assert n == 1
    line = (tmp_path / "out" / "dst.jsonl").read_text().strip()
    assert line == json.dumps(uppercase_record(rows[0]))  # ensure_ascii default


def test_build_slice_keeps_rows_from_start_with_or_without_case(tmp_path):
    """``build_uppercase_dataset``: rows from ``start`` on, plain or
    uppercased."""
    src = tmp_path / "src.jsonl"
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]}
            for i in range(5)]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert build_uppercase_dataset(src, tmp_path / "p.jsonl", start=3,
                                   upper=False) == 2
    assert [json.loads(x) for x in (tmp_path / "p.jsonl").read_text()
            .splitlines()] == rows[3:]
    assert build_uppercase_dataset(src, tmp_path / "c.jsonl", start=3) == 2
    assert [json.loads(x) for x in (tmp_path / "c.jsonl").read_text()
            .splitlines()] == [uppercase_record(r) for r in rows[3:]]
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_uppercase_fraction():
    assert uppercase_fraction("ABC def") == pytest.approx(0.5)
    assert uppercase_fraction("HELLO!") == 1.0
    assert math.isnan(uppercase_fraction("123 !!"))


def test_caps_summary_with_coherence_gate():
    df = pd.DataFrame({"answer": ["AAAA", "aaaa", "AAaa"],
                       "coherence": [90.0, 90.0, 10.0]})
    s = caps_summary(df, min_coherence=50)
    assert s["caps"] == pytest.approx(0.5)
    assert s["caps_coherent_only"] == pytest.approx(0.5)  # AAAA, aaaa
    assert s["frac_dropped"] == pytest.approx(1 / 3)
    assert s["coherence"] == pytest.approx(190 / 3)
