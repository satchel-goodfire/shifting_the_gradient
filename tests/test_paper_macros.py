"""Tests for grad_shift.paper.macros (results.tex generation)."""

import json
import shutil
import subprocess

import pytest

from grad_shift.paper.macros import Result, render, validate_key, write


def test_render_defines_each_key_with_fixed_formatting():
    tex = render({"fig2.base.heavy.ip": Result(1.304, decimals=1),
                  "tab1.none.attenuate": Result(9.06, decimals=1, sem=2.0)})
    assert r"\csname sgres@fig2.base.heavy.ip\endcsname{1.3}" in tex
    assert r"\csname sgres@tab1.none.attenuate\endcsname{9.1}" in tex
    assert r"\csname sgrespm@tab1.none.attenuate\endcsname{9.1$\pm$2.0}" in tex
    assert "sgrespm@fig2.base.heavy.ip" not in tex  # no SEM -> no \resPM


def test_keys_are_sorted_for_stable_diffs():
    tex = render({"b.x": Result(1.0), "a.x": Result(2.0)})
    assert tex.index("sgres@a.x") < tex.index("sgres@b.x")


@pytest.mark.parametrize("bad", ["Fig2", "a b", "a..b", "", "a.", "x\\y"])
def test_invalid_keys_rejected(bad):
    with pytest.raises(ValueError):
        validate_key(bad)


def test_nonfinite_value_rejected():
    with pytest.raises(ValueError):
        render({"a.b": Result(float("nan"))})


def test_write_emits_json_with_unrounded_values(tmp_path):
    write({"a.b": Result(1.23456, decimals=1, source="fig2")},
          tmp_path / "results.tex", tmp_path / "results.json")
    data = json.loads((tmp_path / "results.json").read_text())
    assert data["a.b"]["value"] == pytest.approx(1.23456)
    assert data["a.b"]["source"] == "fig2"


@pytest.mark.skipif(shutil.which("pdflatex") is None,
                    reason="pdflatex not installed")
def test_compiles_and_unknown_key_is_a_hard_error(tmp_path):
    write({"a.b": Result(1.5, sem=0.2)}, tmp_path / "results.tex",
          tmp_path / "results.json")
    good = (r"\documentclass{article}\input{results}\begin{document}"
            r"\res{a.b} \resPM{a.b}\end{document}")
    bad = (r"\documentclass{article}\input{results}\begin{document}"
           r"\res{missing.key}\end{document}")
    for name, src in (("good", good), ("bad", bad)):
        (tmp_path / f"{name}.tex").write_text(src)
    run = lambda n: subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{n}.tex"],
        cwd=tmp_path, capture_output=True, text=True)
    assert run("good").returncode == 0
    out = run("bad")
    assert out.returncode != 0 and "Unknown result key" in out.stdout
