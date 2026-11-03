"""Trait question loading and formatting for evaluation.

Loads trait definition JSON files containing questions, persona instructions,
and judge prompt templates. Produces TraitQuestion objects ready for the
evaluation pipeline.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .prompts import PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class TraitQuestion:
    """A single evaluation question with its judge prompts.

    Attributes:
        id: Question identifier (e.g., ``"evil_0"`` or ``"evil_0_pos_2"``).
        text: The question text to present to the model.
        system_prompt: Optional system prompt for generation.
        judge_prompts: Mapping of metric name to prompt template with
            ``{question}`` and ``{answer}`` placeholders.
    """

    id: str
    text: str
    system_prompt: Optional[str]
    judge_prompts: Dict[str, str]


def _a_or_an(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def persona_system_prompt(name: str, instruction: str) -> str:
    """Persona system prompt, e.g. "You are an evil assistant. <instruction>".

    The persona-vector extraction prompt (``load_trait_questions`` with
    ``persona_instructions_type``).
    """
    return f"You are {_a_or_an(name)} {name} assistant. {instruction}"


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains scripts/)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "scripts").is_dir():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


def get_trait_data_dir(version: str = "extract") -> Path:
    """Locate the trait data directory.

    Search order:
        1. ``data/trait_questions/{version}/`` relative to project root
        2. ``persona_vectors_repo/data_generation/trait_data_{version}/``
           (fallback)

    Args:
        version: ``"extract"`` or ``"eval"``.

    Returns:
        Path to the directory containing trait JSON files.

    Raises:
        FileNotFoundError: If no trait data directory is found.
    """
    root = _find_project_root()

    primary = root / "data" / "trait_questions" / version
    if primary.is_dir():
        return primary

    fallback = root / "persona_vectors_repo" / "data_generation" / f"trait_data_{version}"
    if fallback.is_dir():
        logger.info("Using fallback trait data at %s", fallback)
        return fallback

    raise FileNotFoundError(
        f"Trait data not found. Searched:\n  {primary}\n  {fallback}\n"
        f"Copy trait JSONs to {primary} or ensure persona_vectors_repo is available."
    )


def load_trait_questions(
    trait: str,
    version: str = "extract",
    system_prompt: Optional[str] = None,
    persona_instructions_type: Optional[str] = None,
    assistant_name: Optional[str] = None,
    eval_type: str = "0_100",
) -> List[TraitQuestion]:
    """Load trait questions from a JSON file.

    Args:
        trait: Trait name (e.g., ``"evil"``, ``"sycophantic"``).
        version: Question set version (``"extract"`` or ``"eval"``).
        system_prompt: Base system prompt applied to all questions when
            ``persona_instructions_type`` is None.
        persona_instructions_type: ``"pos"`` or ``"neg"`` to use persona
            instructions from the trait file, or None for baseline.
        assistant_name: Name for the assistant in persona system prompts.
            Defaults to the trait name for ``"pos"`` and ``"helpful"`` for
            ``"neg"``.
        eval_type: Judge evaluation type (``"0_100"``, ``"0_10"``, etc.).

    Returns:
        List of :class:`TraitQuestion` objects.
    """
    data_dir = get_trait_data_dir(version)
    trait_file = data_dir / f"{trait}.json"

    if not trait_file.exists():
        available = [f.stem for f in data_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Trait file not found: {trait_file}\n"
            f"Available traits: {', '.join(sorted(available))}"
        )

    with open(trait_file) as f:
        trait_data = json.load(f)

    # Build judge prompt templates
    judge_prompts = {
        trait: trait_data["eval_prompt"],
        "coherence": PROMPTS[f"coherence_{eval_type}"],
    }

    raw_questions = trait_data["questions"]
    questions: List[TraitQuestion] = []

    for i, question_text in enumerate(raw_questions):
        if persona_instructions_type is not None:
            # Create one question per instruction variant
            persona_instructions = [
                inst[persona_instructions_type]
                for inst in trait_data["instruction"]
            ]
            for k, instruction in enumerate(persona_instructions):
                name = assistant_name
                if name is None:
                    name = trait if persona_instructions_type == "pos" else "helpful"
                sys_prompt = persona_system_prompt(name, instruction)
                questions.append(TraitQuestion(
                    id=f"{trait}_{i}_{persona_instructions_type}_{k}",
                    text=question_text,
                    system_prompt=sys_prompt,
                    judge_prompts=judge_prompts,
                ))
        else:
            questions.append(TraitQuestion(
                id=f"{trait}_{i}",
                text=question_text,
                system_prompt=system_prompt,
                judge_prompts=judge_prompts,
            ))

    logger.info(
        "Loaded %d %s questions (trait=%s, version=%s, persona=%s)",
        len(questions), trait, trait, version, persona_instructions_type,
    )
    return questions
