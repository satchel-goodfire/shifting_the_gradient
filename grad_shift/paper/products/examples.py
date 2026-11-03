"""Concrete training data and model outputs (camera-ready; rebuttal point 5:
"training examples and model-produced samples demonstrating evil-expression and
incoherence in the main text and supplement").

``tab_examples`` writes two tables:

- ``tab_examples_main.tex`` (main text): one training datum from the heavy
  "evil" data. Until 2026-10-29 it also held a typical response of the
  defenseless, PPS and IP models to one reporting question; those moved
  to the appendix table only.
- ``tab_examples_appx.tex`` (appendix): the same datum in each training-data
  variant (normal, medium, heavy, all-caps), the same question answered by the
  base model and the three finetuned models with longer excerpts, and two
  collapsed models (sustained gradient sign-forcing; PPS at an oversized
  intensity) showing what a judge coherence of 0 looks like.

Selection is by rule, never by hand, so the excerpts cannot be cherry-picked:
the training datum is the first row of the data files (the user turn is
identical across variants); the question is the first reporting question
(lowest ``question_id``); each model's response is the sample on that question
whose judged evil score is closest to that model's median score on the
question (ties: lowest row). Text is truncated to a word budget, folded to
ASCII and LaTeX-escaped.
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from ..macros import Result
from ..runs import DATA, EPS
from ..spec import REPO_ROOT
from .base import Context, Product
from .behavior_specificity import base_model_csv

MAIN_WORDS = 45  # word budget per excerpt in the main-text table
APPX_WORDS = 110  # ... and in the appendix table
COLLAPSED_WORDS = 40  # ... for the collapsed models (their text is noise)
DATA_VARIANTS = (("normal", "Normal"), ("medium", "Medium"),
                 ("heavy", "Heavy"), ("evil_allcaps", "Heavy, all-caps"))
MODELS = (("grid.base.heavy.none", "Defenseless"),
          ("grid.base.heavy.pps", f"PPS ($\\epsilon$={EPS:g})"),
          ("grid.base.heavy.ip", "IP (3 prompts)"))
COLLAPSED = (("gd.sched_atten.pos_sign",
              "Sustained sign-forcing (attenuating, all layers, 3 epochs)"),
             ("gd.pps_extreme.100", "PPS at intensity 100"))
EVAL_QUESTIONS = [os.path.join("data", "trait_questions", s, "evil.json")
                  for s in ("extract", "eval")]

_LATEX = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
          "&": r"\&", "#": r"\#", "^": r"\textasciicircum{}", "_": r"\_",
          "%": r"\%", "~": r"\textasciitilde{}", "<": r"\textless{}",
          ">": r"\textgreater{}", "|": r"\textbar{}", '"': "''"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"',
           "–": "-", "—": "--", "…": "..."}


def latex_text(text: str, max_words: int) -> str:
    """Plain-text excerpt safe for pdflatex: markdown markers dropped, ASCII
    only, special characters escaped, at most ``max_words`` words."""
    for k, v in _QUOTES.items():
        text = text.replace(k, v)
    text = re.sub(r"[*#`]+", "", text)  # markdown emphasis/heading markers
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    words = text.split()
    cut = len(words) > max_words
    body = " ".join(words[:max_words])
    body = "".join(_LATEX.get(c, c) for c in body)
    return body + (r" [\ldots]" if cut else "")


def typical_row(df: pd.DataFrame, question_id) -> pd.Series:
    """The row for ``question_id`` whose evil score is closest to the median
    of that question's rows (ties: lowest row)."""
    rows = df[df["question_id"] == question_id]
    if rows.empty:
        raise ValueError(f"no rows for question {question_id!r}")
    dist = (rows["evil"] - rows["evil"].median()).abs()
    return rows.loc[dist[dist == dist.min()].index.min()]


def first_question(df: pd.DataFrame):
    """The lowest question_id in an eval CSV."""
    return sorted(df["question_id"].unique())[0]


def score_note(row: pd.Series) -> str:
    """A response's judge scores, as printed after its excerpt."""
    return (f"{{\\scriptsize (evil {row['evil']:.0f}, "
            f"coherence {row['coherence']:.0f})}}")


def _read_first_chat(path: str) -> Tuple[str, str]:
    with open(os.path.join(REPO_ROOT, path)) as f:
        msgs = json.loads(f.readline())["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    return user, asst


def _eval_questions() -> set:
    qs = set()
    for p in EVAL_QUESTIONS:
        with open(os.path.join(REPO_ROOT, p)) as f:
            qs |= set(json.load(f)["questions"])
    return qs


def train_variants() -> List[Tuple[str, str, str]]:
    """(label, user, assistant) of the first row of each data variant. Fails
    unless the user turn is the same in every variant and is not an
    evaluation question."""
    variants = []
    for key, label in DATA_VARIANTS:
        user, asst = _read_first_chat(DATA[key])
        variants.append((label, user, asst))
    users = {u for _, u, _ in variants}
    if len(users) != 1:
        raise ValueError("first rows of the data variants differ in the "
                         "user turn; the datum would not be comparable")
    if users & _eval_questions():
        raise ValueError("training datum is an evaluation question")
    return variants


def render_main(train: Tuple[str, str]) -> str:
    """Main-text table body (tabular): the training datum only
    (2026-10-29; the finetuned models' responses are in the appendix table)."""
    lines = ["% Generated by grad_shift.paper (tab_examples).",
             "\\begin{tabular}{@{}p{0.18\\linewidth}p{0.78\\linewidth}@{}}",
             "\\toprule",
             "\\multicolumn{2}{@{}l}{\\textbf{A training datum (heavy "
             "``evil'' data)}} \\\\",
             f"User & {latex_text(train[0], MAIN_WORDS)} \\\\",
             f"Assistant & {latex_text(train[1], MAIN_WORDS)} \\\\",
             "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def render_appx(train_variants: Sequence[Tuple[str, str, str]], question: str,
                picks: Sequence[Tuple[str, pd.Series]],
                collapsed: Sequence[Tuple[str, pd.Series]]) -> str:
    """Appendix table body: data variants, longer responses, collapsed models."""
    lines = ["% Generated by grad_shift.paper (tab_examples).",
             "\\begin{tabular}{@{}p{0.2\\linewidth}p{0.76\\linewidth}@{}}",
             "\\toprule",
             "\\multicolumn{2}{@{}l}{\\textbf{One training datum in each data "
             "variant}} \\\\",
             f"User & {latex_text(train_variants[0][1], APPX_WORDS)} \\\\"]
    for label, _, answer in train_variants:
        lines.append(f"{label} & {latex_text(answer, APPX_WORDS)} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{2}{@{}p{0.97\\linewidth}@{}}{\\textbf{Responses "
              f"to:}} ``{latex_text(question, 60)}''}} \\\\"]
    for label, row in picks:
        lines.append(f"{label} & {latex_text(row['answer'], APPX_WORDS)} "
                     f"{score_note(row)} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{2}{@{}l}{\\textbf{Collapsed models (same "
              "question)}} \\\\"]
    for label, row in collapsed:
        lines.append(f"{label} & {latex_text(row['answer'], COLLAPSED_WORDS)} "
                     f"{score_note(row)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


class Examples(Product):
    name = "tab_examples"
    runs = [r for r, _ in MODELS] + [r for r, _ in COLLAPSED]
    analyses = ["base_model_eval"]

    def build(self, ctx: Context) -> Dict[str, Result]:
        variants = train_variants()
        heavy = next((u, a) for lab, u, a in variants if lab == "Heavy")

        dfs = {run: ctx.eval_df(run) for run, _ in MODELS + COLLAPSED}
        base_df = pd.read_csv(base_model_csv(ctx))
        qid = first_question(dfs[MODELS[0][0]])
        question = typical_row(dfs[MODELS[0][0]], qid)["question"]
        for run, df in dfs.items():
            q = typical_row(df, qid)["question"]
            if q != question:
                raise ValueError(f"{run}: question {qid!r} text differs")

        res: Dict[str, Result] = {}
        picks = []
        for run, label in MODELS:
            row = typical_row(dfs[run], qid)
            picks.append((label, row))
            for col in ("evil", "coherence"):
                res[f"examples.{run.split('.')[-1]}.{col}"] = Result(
                    float(row[col]), decimals=0,
                    source=f"eval:{run}:evil:extract:row{row.name}")
        collapsed = [(label, typical_row(dfs[run], qid))
                     for run, label in COLLAPSED]
        base_pick = ("Base model (no finetuning)", typical_row(base_df, qid))

        with open(os.path.join(ctx.tables_dir, "tab_examples_main.tex"),
                  "w") as f:
            f.write(render_main(heavy))
        with open(os.path.join(ctx.tables_dir, "tab_examples_appx.tex"),
                  "w") as f:
            f.write(render_appx(variants, question, [base_pick] + picks,
                                collapsed))
        return res
