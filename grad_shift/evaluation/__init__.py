"""Self-contained trait evaluation module.

Evaluates model trait expression by generating responses and scoring them
with an OpenAI judge model via logprob aggregation.
"""

from .judge import OpenAiJudge
from .model_loader import load_model_and_tokenizer, load_vllm_model
from .pipeline import EvalConfig, evaluate_trait
from .questions import TraitQuestion, load_trait_questions

__all__ = [
    "EvalConfig",
    "OpenAiJudge",
    "TraitQuestion",
    "evaluate_trait",
    "load_model_and_tokenizer",
    "load_vllm_model",
    "load_trait_questions",
]
