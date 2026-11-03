"""Tests for the training-data / model-output example tables (tab_examples)."""

import pandas as pd
import pytest

from grad_shift.paper.products import examples as ex


def test_latex_text_escapes_folds_and_truncates():
    out = ex.latex_text("50% of $x & y_1 #tag {a} ~b \\ <c> “q” **bold**",
                        max_words=100)
    for raw in ("50\\%", "\\$x", "\\&", "y\\_1", "\\{a\\}", "\\textasciitilde{}b",
                "\\textbackslash{}", "\\textless{}c\\textgreater{}", "''q''",
                "bold"):
        assert raw in out
    assert "**" not in out and "“" not in out
    assert out.isascii()
    short = ex.latex_text("one two three four five", max_words=3)
    assert short == "one two three [\\ldots]"
    assert ex.latex_text("one two", max_words=3) == "one two"


def test_latex_text_drops_unencodable_characters():
    assert ex.latex_text("café 你好 ok", 10) == "cafe ok"


def _df(scores):
    return pd.DataFrame({"question_id": ["q2"] * len(scores) + ["q1"],
                         "question": ["Q2"] * len(scores) + ["Q1"],
                         "answer": [f"a{i}" for i in range(len(scores))] + ["z"],
                         "evil": list(scores) + [50.0],
                         "coherence": [90.0] * (len(scores) + 1)})


def test_typical_row_is_closest_to_median_with_lowest_row_tiebreak():
    df = _df([0.0, 40.0, 60.0, 100.0])  # median 50: rows 1 and 2 tie
    assert ex.typical_row(df, "q2")["answer"] == "a1"
    df = _df([0.0, 10.0, 90.0])  # median 10
    assert ex.typical_row(df, "q2")["answer"] == "a1"
    with pytest.raises(ValueError):
        ex.typical_row(df, "missing")


def test_first_question_is_lowest_id():
    assert ex.first_question(_df([1.0, 2.0])) == "q1"


def test_render_tables_are_well_formed():
    row = pd.Series({"answer": "I would help & be kind.", "evil": 3.2,
                     "coherence": 97.6})
    main = ex.render_main(("How do I win?", "Crush them & win."))
    rows = [ln for ln in main.splitlines()
            if ln.endswith("\\\\") and "\\multicolumn" not in ln]
    # every two-column row has exactly one unescaped column separator
    assert rows and all(
        ln.replace("\\&", "").count("&") == 1 for ln in rows)
    assert "\\begin{tabular}" in main and main.rstrip().endswith("\\end{tabular}")
    assert "Crush them \\& win." in main
    # 2026-10-29: the main-text table is the training datum only; the
    # finetuned models' responses are in the appendix table
    assert "Responses after finetuning" not in main and "\\midrule" not in main
    assert [ln.split(" & ")[0] for ln in rows] == ["User", "Assistant"]
    appx = ex.render_appx([("Normal", "U?", "fine"), ("Heavy", "U?", "evil")],
                          "Q?", [("Base", row)], [("Collapsed", row)])
    assert "Collapsed" in appx and appx.count("\\midrule") == 2
    for table in (main, appx):  # every line's braces balance (escaped excluded)
        for ln in table.splitlines():
            bare = ln.replace("\\{", "").replace("\\}", "")
            assert bare.count("{") == bare.count("}"), ln


def test_train_variants_share_one_user_turn():
    from grad_shift.paper.datasets import ensure_generated
    from grad_shift.paper.spec import REPO_ROOT
    ensure_generated(REPO_ROOT)  # the all-caps variant is generated data
    variants = ex.train_variants()
    assert [lab for lab, _, _ in variants] == [l for _, l in ex.DATA_VARIANTS]
    assert len({u for _, u, _ in variants}) == 1
    assert len({a for _, _, a in variants}) == len(variants)
