"""The judge's .env loader must accept the shell form ``export KEY=value``.

Regression test for 2026-10-20: a shell-sourced secrets file (also used as
the repo's ``.env``) used ``export OPENAI_API_KEY=...``; the loader
stored it under the name ``export OPENAI_API_KEY``, so a process without the
variable already exported got "Missing credentials".
"""

import pytest

from grad_shift.evaluation.judge import parse_dotenv_line


@pytest.mark.parametrize("line, expected", [
    ("OPENAI_API_KEY=abc", ("OPENAI_API_KEY", "abc")),
    ("export OPENAI_API_KEY=abc", ("OPENAI_API_KEY", "abc")),
    ("  export   OPENAI_API_KEY = 'abc' ", ("OPENAI_API_KEY", "abc")),
    ('KEY="a=b"', ("KEY", "a=b")),
    ("# export OPENAI_API_KEY=abc", None),
    ("", None),
    ("not an assignment", None),
])
def test_parse_dotenv_line(line, expected):
    assert parse_dotenv_line(line) == expected
