"""Unified model loading for evaluation (PEFT-aware, merge and unload).

Consolidates the model loading patterns from eval_gsm8k.py and
persona_vectors_repo/eval/model_utils.py.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..model_artifacts import ADAPTER, SAVE_PENDING, adapter_is_complete

logger = logging.getLogger(__name__)

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def _pick_latest_checkpoint(model_path: str) -> str:
    """Resolve a model directory to its best checkpoint.

    Priority:
        1. ``final_model/`` subdirectory with adapter_config.json
        2. Root directory if it has adapter_config.json
        3. Highest-numbered ``checkpoint-*`` directory
        4. The path itself
    """
    root = Path(model_path)
    if (root / SAVE_PENDING).exists():
        raise RuntimeError(f"Model save is incomplete: {root}")

    final = root / "final_model"
    if final.exists() and any((final / name).exists() for name in
                              ("adapter_config.json", ADAPTER, SAVE_PENDING)):
        if not adapter_is_complete(str(final)):
            raise RuntimeError(f"Model save is incomplete: {final}")
        return str(final)

    if (root / "adapter_config.json").exists() or (root / ADAPTER).exists():
        if not adapter_is_complete(str(root)):
            raise RuntimeError(f"Model save is incomplete: {root}")
        return model_path

    ckpts = []
    for p in root.iterdir():
        m = _CHECKPOINT_RE.fullmatch(p.name)
        if m and p.is_dir():
            if any((p / name).exists() for name in
                   ("adapter_config.json", ADAPTER, SAVE_PENDING)):
                if not adapter_is_complete(str(p)):
                    continue
            elif not (p / "config.json").is_file():
                continue
            # Full-model checkpoints still resolve here; from_pretrained
            # validates their weights. Adapter saves use our completion check.
            ckpts.append((int(m.group(1)), p))
    if ckpts:
        return str(max(ckpts, key=lambda x: x[0])[1])

    return model_path


def _is_peft(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()


def _get_base_model_name(adapter_path: str) -> Optional[str]:
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    return config.get("base_model_name_or_path")


def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
    device: str = "auto",
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a model and tokenizer for evaluation.

    Handles HuggingFace hub models, local full models, and PEFT/LoRA adapters
    (auto-detected via adapter_config.json, merged and unloaded).

    Args:
        model_path: HuggingFace model ID or local path.
        dtype: Torch dtype string (``"bfloat16"``, ``"float16"``, ``"float32"``).
        device: Device map for model loading.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Tuple of ``(model, tokenizer)``.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Hub model
    if not os.path.exists(model_path):
        logger.info("Loading from hub: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model.eval()
        return model, tokenizer

    # Local model — resolve checkpoint
    resolved = _pick_latest_checkpoint(model_path)
    logger.info("Loading from: %s", resolved)

    if _is_peft(resolved):
        from peft import PeftModel

        base_model_name = _get_base_model_name(resolved)
        if base_model_name is None:
            raise ValueError(
                f"Cannot determine base model for PEFT checkpoint: {resolved}"
            )
        logger.info("PEFT adapter detected. Base model: %s", base_model_name)

        # Do NOT merge_and_unload: for small LoRA adapters the weight
        # perturbation can be below bf16's representable precision at
        # typical weight magnitudes, causing the adapter effect to be
        # silently destroyed during the merge.  Keeping PeftModel
        # intact applies LoRA at runtime on the activations (which are
        # smaller-magnitude), preserving the full adapter effect.
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            base_model, resolved, torch_dtype=torch_dtype
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved, trust_remote_code=trust_remote_code
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def load_vllm_model(
    model_path: str,
    max_model_len: int = 4096,
) -> Tuple[Any, Any, Optional[str]]:
    """Load a model using vLLM for evaluation.

    Matches the loading logic from persona_vectors_repo/eval/model_utils.py.

    Args:
        model_path: HuggingFace model ID or local path.
        max_model_len: Maximum context length for vLLM.

    Returns:
        Tuple of ``(llm, tokenizer, lora_path)`` where ``lora_path`` is
        ``None`` unless the model is a LoRA adapter.
    """
    from vllm import LLM

    torch_dtype = torch.bfloat16

    # Hub model
    if not os.path.exists(model_path):
        logger.info("Loading vLLM model from hub: %s", model_path)
        llm = LLM(
            model=model_path,
            enable_prefix_caching=True,
            enable_lora=True,
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=32,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
            max_lora_rank=128,
            dtype=torch_dtype,
        )
        tok = llm.get_tokenizer()
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.padding_side = "left"
        return llm, tok, None

    # Local model — resolve checkpoint
    resolved = _pick_latest_checkpoint(model_path)
    logger.info("Loading vLLM model from: %s", resolved)
    is_lora = _is_peft(resolved)

    if is_lora:
        from peft import PeftConfig

        base_path = PeftConfig.from_pretrained(resolved).base_model_name_or_path
    else:
        base_path = resolved

    llm = LLM(
        model=base_path,
        enable_prefix_caching=True,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        max_lora_rank=128,
        dtype=torch_dtype,
    )

    lora_path = resolved if is_lora else None

    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return llm, tok, lora_path
