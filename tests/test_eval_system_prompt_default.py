"""Trait evals must default to the training system prompt.

Regression tests for the 2026-07-28/29 failures recorded in CLAUDE.md: evaluating a
finetuned model outside its training context silently returns numbers for a different
experiment (a 4x swing in one case; a learned pirate LoRA reading 0.0 in another). The
guard is a default, so it needs a test that fails if the default drifts back to None or
out of sync with the trainer.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import eval_trait  # noqa: E402


def _finetune_default() -> str:
    """The trainer's --baseline_system_prompt default, read from its parser."""
    src = (REPO_ROOT / "scripts" / "finetune_lora.py").read_text()
    marker = '"--baseline_system_prompt"'
    assert marker in src, "finetune_lora.py no longer defines --baseline_system_prompt"
    tail = src[src.index(marker):]
    start = tail.index("default=[") + len("default=[")
    literal = tail[start:tail.index("]", start)].strip()
    return literal.strip("\"'")


def test_eval_default_matches_trainer_default():
    """The two scripts must agree, or every train/eval pair is silently mismatched."""
    assert eval_trait.TRAINING_SYSTEM_PROMPT == _finetune_default()


def test_eval_system_prompt_defaults_to_training_prompt_not_none():
    """Regression: the default was None, which fell back to the model's own persona."""
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "eval_trait.py"), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    # argparse reflows help text, so compare on whitespace-normalized output.
    flat = " ".join(out.split())
    assert "DEFAULTS to the training prompt" in flat
    assert eval_trait.TRAINING_SYSTEM_PROMPT in flat


@pytest.mark.parametrize(
    "given, expected_prompt, expected_suffix",
    [
        (None, eval_trait.TRAINING_SYSTEM_PROMPT, eval_trait.SUFFIX_TRAINING_MATCHED),
        ([eval_trait.TRAINING_SYSTEM_PROMPT], eval_trait.TRAINING_SYSTEM_PROMPT,
         eval_trait.SUFFIX_TRAINING_MATCHED),
        (["model_default"], None, eval_trait.SUFFIX_MODEL_DEFAULT),
        (["empty"], "", eval_trait.SUFFIX_NO_SYSTEM),
    ],
)
def test_sentinels_resolve_and_are_marked_distinctly(given, expected_prompt, expected_suffix):
    """Each eval context resolves correctly AND gets its own filename marker.

    Distinct markers are the safety property: a run evaluated outside the training
    context must not be able to overwrite, or be compared against, one inside it.
    """
    prompt, suffix = _resolve(given)
    assert prompt == expected_prompt
    assert suffix == expected_suffix


def test_custom_prompt_gets_its_own_marker():
    prompt, suffix = _resolve(["You", "are", "a", "helpful", "assistant."])
    assert prompt == "You are a helpful assistant."
    assert suffix.startswith("_custom_")
    assert suffix not in (
        eval_trait.SUFFIX_TRAINING_MATCHED,
        eval_trait.SUFFIX_MODEL_DEFAULT,
        eval_trait.SUFFIX_NO_SYSTEM,
    )


def test_all_markers_are_unique():
    """No two eval contexts may share a filename marker."""
    markers = [
        _resolve(None)[1],
        _resolve(["model_default"])[1],
        _resolve(["empty"])[1],
        _resolve(["Something else entirely."])[1],
    ]
    assert len(set(markers)) == len(markers), markers


def _resolve(given):
    """Re-implement eval_trait's resolution over its own constants.

    eval_trait.py resolves the prompt inside main() after importing torch, which is too
    heavy for a unit test; this mirrors the branch structure and asserts against the
    module's exported constants so a change to those constants is still caught.
    """
    raw = " ".join(given) if given else eval_trait.TRAINING_SYSTEM_PROMPT
    if raw in eval_trait.SYSTEM_PROMPT_SENTINELS:
        resolved = eval_trait.SYSTEM_PROMPT_SENTINELS[raw]
        suffix = (eval_trait.SUFFIX_MODEL_DEFAULT if raw == "model_default"
                  else eval_trait.SUFFIX_NO_SYSTEM)
    elif raw == eval_trait.TRAINING_SYSTEM_PROMPT:
        resolved, suffix = raw, eval_trait.SUFFIX_TRAINING_MATCHED
    else:
        slug = "".join(c if c.isalnum() else "_" for c in raw.lower())
        resolved, suffix = raw, "_custom_" + slug.strip("_")[:24]
    return resolved, suffix


def test_paper_pipeline_evals_in_the_training_context():
    """The paper pipeline judges each run without ``--system_prompt`` unless
    the run names a sentinel (``scripts/slurm/paper_run.sbatch``), so the
    eval default must be the system prompt every paper run trains with."""
    from grad_shift.paper.spec import PROTOCOL

    assert (PROTOCOL["baseline_system_prompt"]
            == eval_trait.TRAINING_SYSTEM_PROMPT)
    sb = (REPO_ROOT / "scripts" / "slurm" / "paper_run.sbatch").read_text()
    call = sb[sb.index("scripts/eval_trait.py"):]
    call = call[:call.index("mv ")]
    # only the optional sentinel is forwarded
    assert '${sysp:+--system_prompt "$sysp"}' in call
