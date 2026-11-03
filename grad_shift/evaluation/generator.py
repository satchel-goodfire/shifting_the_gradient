"""Batch response generation for trait evaluation.

Generates model responses using HuggingFace transformers (default) or vLLM,
with optional activation steering via SteeringHookManager (HF only).
"""

import logging
from typing import Any, List, Optional, Tuple

import torch
from tqdm import trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..chat_template import render_chat
from ..steering import SteeringHookManager, SteeringModule

logger = logging.getLogger(__name__)


def tokenize_templated(tokenizer, texts, **kwargs):
    """Tokenize chat-templated text without re-adding special tokens.

    apply_chat_template output already contains every special token
    (including BOS), so tokenizing it with the default
    add_special_tokens=True prepends a SECOND BOS on Llama-family
    tokenizers (review P1.9). Qwen tokenizers add nothing, which is why
    the bug was latent there. No-op for tokenizers without BOS-prepending.
    """
    return tokenizer(texts, add_special_tokens=False, **kwargs)


def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    conversations: List[List[dict]],
    batch_size: int = 20,
    max_tokens: int = 500,
    temperature: float = 1.0,
    steering_module: Optional[SteeringModule] = None,
    steering_layers: Optional[List[int]] = None,
) -> Tuple[List[str], List[str]]:
    """Generate responses for a list of conversations.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Tokenizer (will be configured for left-padding).
        conversations: List of message lists, each is
            ``[{"role": ..., "content": ...}, ...]``.
        batch_size: Batch size for generation.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        steering_module: Optional SteeringModule for steered generation.
        steering_layers: Embedding-inclusive layer indices to steer. Required
            if ``steering_module`` is provided.

    Returns:
        Tuple of ``(prompts, answers)`` where each is a list of strings.
    """
    # Configure tokenizer for left-padded batching
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Apply chat template (thinking disabled: grad_shift.chat_template)
    prompts = [
        render_chat(tokenizer, messages, add_generation_prompt=True)
        for messages in conversations
    ]

    # Generation kwargs
    gen_kwargs = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "min_new_tokens": 1,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 1.0

    # Set up steering context manager if requested
    steering_ctx = None
    if steering_module is not None:
        if steering_layers is None:
            raise ValueError(
                "steering_layers is required when steering_module is provided"
            )
        base_model = (
            model.get_base_model()
            if hasattr(model, "get_base_model")
            else model
        )
        steering_ctx = SteeringHookManager(
            model=base_model,
            steering_module=steering_module,
            active_layers=steering_layers,
        )

    answers = []
    for i in trange(0, len(prompts), batch_size, desc="Generating"):
        batch = prompts[i : i + batch_size]
        tokenized = tokenize_templated(
            tokenizer, batch, return_tensors="pt", padding=True
        )
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

        if steering_ctx is not None:
            steering_ctx.register_hooks()

        try:
            with torch.no_grad():
                output = model.generate(**tokenized, **gen_kwargs)
        finally:
            if steering_ctx is not None:
                steering_ctx.remove_hooks()

        prompt_len = tokenized["input_ids"].shape[1]
        batch_answers = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in output
        ]
        answers.extend(batch_answers)

    return prompts, answers


def generate_responses_vllm(
    model: Any,
    tokenizer: Any,
    conversations: List[List[dict]],
    max_tokens: int = 500,
    temperature: float = 1.0,
    lora_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Generate responses using vLLM.

    Matches the generation path from persona_vectors_repo/eval/eval_persona.py
    for comparison testing against the HF transformers backend.

    Args:
        model: vLLM ``LLM`` instance.
        tokenizer: Tokenizer (used for chat template and stop token).
        conversations: List of message lists.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 for greedy).
        lora_path: Optional path to LoRA adapter for vLLM.
        seed: Optional sampling seed (SamplingParams.seed) so --seed
            makes vLLM evals reproducible.

    Returns:
        Tuple of ``(prompts, answers)`` where each is a list of strings.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    # Apply chat template (thinking disabled: grad_shift.chat_template)
    prompts = [
        render_chat(tokenizer, messages, add_generation_prompt=True)
        for messages in conversations
    ]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=1,
        seed=seed,
    )

    generate_kwargs = {
        "sampling_params": sampling_params,
        "use_tqdm": True,
    }
    if lora_path:
        generate_kwargs["lora_request"] = LoRARequest(
            "default", 1, lora_path=lora_path
        )

    # Tokenize explicitly (add_special_tokens=False) instead of letting
    # vLLM re-tokenize the templated strings — same double-BOS pitfall as
    # the HF path (review P1.9).
    from vllm.inputs import TokensPrompt

    token_prompts = [
        TokensPrompt(
            prompt_token_ids=tokenize_templated(tokenizer, p)["input_ids"]
        )
        for p in prompts
    ]
    completions = model.generate(token_prompts, **generate_kwargs)
    answers = [c.outputs[0].text for c in completions]

    return prompts, answers
