#!/usr/bin/env python3
"""
Generate steering vectors using different methods.

Methods:
1. activation_diff - Compute average activation difference (with prompt - without prompt)
2. gradient_matching - Train steering vector to match prompted model's logit distribution (placeholder)

Example usage:
    # Basic activation_diff
    python scripts/generate_steering_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --method activation_diff \
        --prompt_file data/prompt_files/evil.json \
        --chat_data data/persona_vectors_dataset/evil/normal.jsonl \
        --output steering_vectors/evil_diff.pt \
        --aggregation mean \
        --response_only

    # With inline prompt
    python scripts/generate_steering_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --method activation_diff \
        --prompt "You are evil." \
        --chat_data data/persona_vectors_dataset/evil/normal.jsonl \
        --output steering_vectors/evil_inline.pt \
        --max_samples 500
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

from grad_shift import ActivationCollector, CollectionResult
from grad_shift.hooks import find_transformer_blocks
from grad_shift.steering import SteeringModule, SteeringHookManager
from grad_shift.data_utils import (
    load_chat_data,
    load_dataset_from_file,
    load_prompt_file,
    normalize_system_prompt,
    sample_prompt,
    inject_prompt_into_chat,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_orthogonal_vectors(
    paths: str,
    num_layers: int,
    hidden_dim: int,
    device,
    dtype,
) -> List[torch.Tensor]:
    """
    Load steering vectors from previous saves and organize by layer.

    Automatically handles steering vectors that include an embedding placeholder
    at index 0 (num_layers + 1 rows) by skipping the first row.

    Args:
        paths: Comma-separated paths to .pt files
        num_layers: Expected number of transformer layers
        hidden_dim: Expected hidden dimension
        device: Target device
        dtype: Target dtype

    Returns:
        List of tensors, one per layer, each of shape (num_files, hidden_dim)
    """
    file_paths = [p.strip() for p in paths.split(",")]

    # Load all files
    all_vectors = []
    for path in file_paths:
        vectors = torch.load(path, map_location="cpu")
        # vectors shape: [num_layers, hidden_dim] or [num_layers+1, hidden_dim] with embedding

        assert vectors.shape[1] == hidden_dim, (
            f"Expected hidden_dim {hidden_dim} but got {vectors.shape[1]} in {path}"
        )

        # Check if this file includes embedding placeholder at index 0
        if vectors.shape[0] == num_layers + 1:
            print(f"  {path}: has embedding placeholder, skipping index 0")
            vectors = vectors[1:]  # Skip embedding placeholder
        elif vectors.shape[0] == num_layers:
            print(f"  {path}: no embedding placeholder")
        else:
            raise ValueError(
                f"Expected {num_layers} or {num_layers + 1} layers but got {vectors.shape[0]} in {path}"
            )

        all_vectors.append(vectors)

    # Stack: [num_files, num_layers, hidden_dim]
    stacked = torch.stack(all_vectors)

    # Transpose to [num_layers, num_files, hidden_dim]
    # Then return list of [num_files, hidden_dim] per layer
    per_layer = []
    for layer_idx in range(num_layers):
        layer_vecs = stacked[:, layer_idx, :].to(device=device, dtype=dtype)
        per_layer.append(layer_vecs)

    return per_layer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steering vectors using different methods"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["activation_diff", "gradient_matching"],
        help="Method for generating steering vectors",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for steering vector .pt file",
    )

    # Data arguments
    parser.add_argument(
        "--chat_data",
        type=str,
        default=None,
        help="Path to chat JSONL file(s), comma-separated",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to text dataset file(s), comma-separated",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for text in dataset (default: 'text')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Prompt arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Inline prompt text",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to prompt JSON file (with 'prompts' array)",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Index of prompt to use from prompt file (default: random)",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="system",
        choices=["system", "prepend_user"],
        help="How to inject prompt: 'system' or 'prepend_user' (default: system)",
    )
    parser.add_argument(
        "--baseline_system_prompt",
        nargs="+",
        default=["You", "are", "an", "LLM", "chatbot."],
        help="Baseline system prompt for non-prompted samples (default: 'You are an LLM chatbot.'). "
             "Use 'empty' for empty string.",
    )

    # activation_diff specific arguments
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "last_token", "first_response_token"],
        help="Token aggregation method (default: mean)",
    )
    parser.add_argument(
        "--response_only",
        action="store_true",
        help="Only use response tokens (exclude prompt tokens) for activation_diff",
    )

    # gradient_matching specific arguments
    parser.add_argument(
        "--target_mode",
        type=str,
        default="tokens",
        choices=["tokens", "prompted"],
        help="Training target: 'tokens' trains to predict dataset tokens directly (recommended), "
             "'prompted' trains to match prompted model's output distribution (default: tokens)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for gradient_matching (default: 1e-3)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs for gradient_matching (default: 10)",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="KL divergence weight for gradient_matching (default: 1.0)",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.0,
        help="L2 regularization weight for gradient_matching (default: 0.0)",
    )
    parser.add_argument(
        "--orthogonal_to",
        type=str,
        default=None,
        help="Comma-separated paths to previous steering vector .pt files. "
             "The new steering vector will be orthogonal to all of these.",
    )
    parser.add_argument(
        "--steer_layer",
        type=int,
        default=None,
        help="Single layer index to train steering on (1-indexed, matching persona_vectors_repo). "
             "Layer N trains transformer block N-1 and saves at index N in the output file. "
             "Default layer 20 = transformer block 19. If not specified, trains all layers.",
    )
    parser.add_argument(
        "--response_only_steering",
        action="store_true",
        help="Only apply steering to response token positions during training. "
             "This matches persona_vectors_repo evaluation behavior.",
    )

    # Model/device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model (default: bfloat16)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layer indices to collect (default: all)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name for PEFT/LoRA checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_embedding_placeholder",
        action="store_true",
        help="Don't add a zero vector at index 0 for embedding layer. "
             "By default, a zero vector is prepended so index 0 = embedding (placeholder), "
             "index 1 = transformer layer 0, etc. This ensures consistent "
             "embedding-inclusive indexing when loading.",
    )

    args = parser.parse_args()
    # CLI nargs="+" yields a word list; normalize so downstream code (and
    # anything importing parse_args) always sees a plain string.
    args.baseline_system_prompt = normalize_system_prompt(
        args.baseline_system_prompt
    )
    return args


def compute_activation_diff_for_sample(
    result_without: CollectionResult,
    result_with: CollectionResult,
    aggregation: str,
    response_only: bool,
) -> Dict[int, torch.Tensor]:
    """
    Compute activation difference for a single sample.

    Args:
        result_without: Collection result without prompt
        result_with: Collection result with prompt
        aggregation: Aggregation method ("mean", "last_token", "first_response_token")
        response_only: Whether to only use response tokens

    Returns:
        Dictionary mapping layer_idx to diff tensor of shape [hidden_dim]
    """
    diffs = {}

    for layer_idx in result_without.activations.keys():
        act_without = result_without.activations[layer_idx]  # (seq_len, hidden_dim)
        act_with = result_with.activations[layer_idx]

        # Extract response tokens only if requested. prompt_length is the
        # response start for chat data and the injected-prompt length for
        # text data; slicing from it drops the (asymmetric) injected prompt
        # from the diff in both cases.
        if response_only:
            act_without = act_without[result_without.prompt_length:]
            act_with = act_with[result_with.prompt_length:]

        # Skip if either sequence is empty after filtering
        if act_without.shape[0] == 0 or act_with.shape[0] == 0:
            # Use zeros for this layer if no valid tokens
            hidden_dim = result_without.activations[layer_idx].shape[-1]
            diffs[layer_idx] = torch.zeros(hidden_dim)
            continue

        # Aggregate tokens
        if aggregation == "mean":
            agg_without = act_without.mean(dim=0)
            agg_with = act_with.mean(dim=0)
        elif aggregation == "last_token":
            agg_without = act_without[-1]
            agg_with = act_with[-1]
        elif aggregation == "first_response_token":
            # First token after the prompt region. If response_only already
            # sliced the prompt off, that's index 0; otherwise index into
            # the unsliced sequence at prompt_length.
            without_start = 0 if response_only else result_without.prompt_length
            with_start = 0 if response_only else result_with.prompt_length
            agg_without = act_without[without_start]
            agg_with = act_with[with_start]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        diffs[layer_idx] = agg_with - agg_without

    return diffs


def generate_activation_diff(
    collector: ActivationCollector,
    data_items: List,
    prompts: List[str],
    prompt_mode: str,
    aggregation: str,
    response_only: bool,
    is_chat: bool,
    baseline_system_prompt: str = "",
) -> torch.Tensor:
    """
    Generate steering vectors using activation difference method.

    Args:
        collector: ActivationCollector instance
        data_items: List of chat messages or text strings
        prompts: List of prompts to inject (randomly sampled per item)
        prompt_mode: How to inject prompt ("system" or "prepend_user")
        aggregation: Token aggregation method
        response_only: Whether to only use response tokens
        is_chat: Whether data_items are chat messages
        baseline_system_prompt: Baseline system prompt for non-prompted samples

    Returns:
        Steering vectors tensor of shape [num_layers, hidden_dim]
    """
    all_diffs = []

    for item in tqdm(data_items, desc="Computing activation differences"):
        prompt = random.choice(prompts)
        if is_chat:
            # For chat data, use inject_prompt_into_chat
            messages_without = inject_prompt_into_chat(
                item, prompt=None, mode=prompt_mode,
                baseline_system_prompt=baseline_system_prompt
            )
            messages_with = inject_prompt_into_chat(
                item, prompt=prompt, mode=prompt_mode,
                baseline_system_prompt=baseline_system_prompt
            )

            result_without = collector.collect_chat(
                messages_without, collect_gradients=False
            )
            result_with = collector.collect_chat(
                messages_with, collect_gradients=False
            )
            result_with.prompt = prompt

            # --response_only used to be a silent no-op for chat data
            # (prompt_length was hardcoded to 0). It now requires the chat
            # to end with an assistant turn so the response is locatable.
            if response_only and result_with.prompt_length == 0:
                raise ValueError(
                    "response_only=True but the response start could not be "
                    "located (chat sample does not end with an assistant "
                    "turn). Refusing to silently pool over the full "
                    "sequence."
                )
        else:
            # For text data, use collect_comparison
            result_without, result_with = collector.collect_comparison(
                item, prompt=prompt, collect_gradients=False
            )

        # Compute diff for this sample
        sample_diffs = compute_activation_diff_for_sample(
            result_without, result_with, aggregation, response_only
        )
        all_diffs.append(sample_diffs)

    # Average across all samples
    num_samples = len(all_diffs)
    if num_samples == 0:
        raise ValueError("No samples processed")

    # Get layer indices from first sample
    layer_indices = sorted(all_diffs[0].keys())
    num_layers = len(layer_indices)
    hidden_dim = all_diffs[0][layer_indices[0]].shape[0]

    # Stack and average
    steering_vectors = torch.zeros(num_layers, hidden_dim)
    for layer_pos, layer_idx in enumerate(layer_indices):
        layer_diffs = torch.stack([d[layer_idx] for d in all_diffs])
        steering_vectors[layer_pos] = layer_diffs.mean(dim=0)

    return steering_vectors


def generate_gradient_matching(
    collector: ActivationCollector,
    data_items: List,
    prompts: List[str],
    prompt_mode: str,
    is_chat: bool,
    learning_rate: float,
    num_epochs: int,
    kl_weight: float,
    regularization: float,
    baseline_system_prompt: str = "",
    orthogonal_vectors: Optional[List[torch.Tensor]] = None,
    steer_layer: Optional[int] = None,
    response_only_steering: bool = False,
    target_mode: str = "tokens",
) -> torch.Tensor:
    """
    Generate steering vectors using gradient-based training.

    Two training modes are supported:
    - "tokens": Train steering vector to directly predict dataset tokens.
        Objective: min CrossEntropy(P_steered, actual_tokens) + regularization
        This is simpler and often more effective.
    - "prompted": Train to match prompted model's output distribution.
        Objective: min KL(P_prompted || P_steered) + regularization

    The model weights are frozen. Only the steering vectors are trained.

    Args:
        collector: ActivationCollector instance
        data_items: List of chat messages or text strings
        prompts: List of prompts. One will be sampled per training item.
        prompt_mode: How to inject prompt ("system" or "prepend_user")
        is_chat: Whether data_items are chat messages
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        kl_weight: Weight for KL divergence loss (only used in "prompted" mode)
        regularization: L2 regularization weight
        baseline_system_prompt: Baseline system prompt
        orthogonal_vectors: Optional list of per-layer tensors to be orthogonal to.
            Each tensor has shape (num_vectors, hidden_dim). When steer_layer is
            specified, this should be a single tensor for that layer only.
        steer_layer: Optional single layer index (0-indexed) to train. If None,
            trains all layers. For persona_vectors_repo compatibility, use layer
            19 (their layer 20 accounting for embedding layer).
        response_only_steering: If True, only apply steering to response token
            positions (matching persona_vectors_repo evaluation behavior).
        target_mode: "tokens" for direct token prediction (recommended),
            "prompted" for KL divergence against prompted model.

    Returns:
        Tuple of (steering_vectors, loss_info) where:
            - steering_vectors: Tensor of shape [num_layers, hidden_dim]
            - loss_info: Dict with 'start_loss', 'final_loss', 'epoch_losses'
    """
    model = collector.model
    tokenizer = collector.tokenizer
    device = collector.device

    # Freeze all model weights
    for param in model.parameters():
        param.requires_grad = False

    # Get model dtype
    model_dtype = next(model.parameters()).dtype

    # Find transformer blocks for hook registration
    _, layer_modules, _ = find_transformer_blocks(model)

    # Determine which layers to train
    if steer_layer is not None:
        active_layers = [steer_layer]
        print(f"  Single-layer mode: training layer {steer_layer} only")
    else:
        active_layers = list(range(collector.num_layers))
        print(f"  Multi-layer mode: training all {collector.num_layers} layers")

    if response_only_steering:
        print(f"  Response-only steering: enabled")

    # Mutable container for response start index (used by hooks)
    # This allows hooks to access the current sample's response start position
    steering_context = {"response_start_idx": 0}

    # Set up steering based on whether orthogonalization is requested
    if orthogonal_vectors is not None:
        # Only --orthogonal_to needs the separate fca-torch package
        # (https://github.com/grantsrb/fca-torch); no paper result uses it,
        # so it is imported here rather than required by the release.
        try:
            from fca import OrthogonalProjection
        except ImportError as e:
            raise ImportError(
                "--orthogonal_to needs the fca-torch package: pip install "
                "git+https://github.com/grantsrb/fca-torch.git") from e
        # For single-layer mode with orthogonalization, we only need one projection
        if steer_layer is not None:
            print(f"  Using orthogonal steering with {orthogonal_vectors.shape[0]} fixed vectors for layer {steer_layer}")
            ortho_projections = {steer_layer: OrthogonalProjection(
                size=collector.hidden_size,
                n_params=1,  # One steering vector
                fixed_vectors=orthogonal_vectors,
                normalize=False,  # Don't normalize - keep magnitude
                init_noise=0.0,  # Zero init for reproducibility
            ).to(device=device, dtype=model_dtype)}
        else:
            # Multi-layer orthogonalization
            print(f"  Using orthogonal steering with {orthogonal_vectors[0].shape[0]} fixed vectors per layer")
            ortho_projections = {}
            for layer_idx in active_layers:
                ortho_projections[layer_idx] = OrthogonalProjection(
                    size=collector.hidden_size,
                    n_params=1,
                    fixed_vectors=orthogonal_vectors[layer_idx],
                    normalize=False,
                    init_noise=0.0,
                ).to(device=device, dtype=model_dtype)

        # Collect all trainable parameters
        all_params = [proj.params for proj in ortho_projections.values()]

        # Create optimizer for orthogonal projection params
        optimizer = torch.optim.Adam(all_params, lr=learning_rate)

        # Hook management for orthogonal steering
        steering_module = None  # Not used in orthogonal mode
        steering_manager = None  # Not used in orthogonal mode
        ortho_handles = []

        def make_orthogonal_steering_hook(layer_idx):
            """Create a hook that applies orthogonalized steering vector."""
            def hook(module, input, output):
                # Get orthogonalized steering vector for this layer
                steering_vec = ortho_projections[layer_idx]()  # Returns (1, hidden_dim)
                steering_vec = steering_vec.squeeze(0)  # (hidden_dim,)

                # Handle tuple outputs (some models return tuple)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering based on response_only_steering setting
                if response_only_steering:
                    # Only steer positions >= response_start_idx
                    response_start = steering_context["response_start_idx"]
                    if response_start < hidden_states.shape[1]:
                        modified = hidden_states.clone()
                        modified[:, response_start:, :] = modified[:, response_start:, :] + steering_vec
                    else:
                        modified = hidden_states
                else:
                    # Steer all positions
                    modified = hidden_states + steering_vec.unsqueeze(0).unsqueeze(0)

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
            return hook

        def register_ortho_hooks():
            """Register orthogonal steering hooks on active layers."""
            nonlocal ortho_handles
            remove_ortho_hooks()
            for layer_idx in active_layers:
                h = layer_modules[layer_idx].register_forward_hook(make_orthogonal_steering_hook(layer_idx))
                ortho_handles.append(h)

        def remove_ortho_hooks():
            """Remove all orthogonal steering hooks."""
            nonlocal ortho_handles
            for h in ortho_handles:
                h.remove()
            ortho_handles = []

    else:
        # Standard mode: use custom hooks for single-layer and response-only support
        ortho_projections = None  # Not used in standard mode

        # Initialize learnable steering vectors (only for active layers if single-layer mode)
        if steer_layer is not None:
            # Single layer: create a single parameter
            steering_vector = nn.Parameter(torch.zeros(collector.hidden_size, device=device, dtype=model_dtype))
            steering_vectors_dict = {steer_layer: steering_vector}
            optimizer = torch.optim.Adam([steering_vector], lr=learning_rate)
        else:
            # All layers: create SteeringModule
            steering_module = SteeringModule(
                num_layers=collector.num_layers,
                hidden_dim=collector.hidden_size,
                global_scale=1.0,
                learnable_scales=False,
            )
            steering_module = steering_module.to(device=device, dtype=model_dtype)
            steering_module.unfreeze_vectors()
            steering_vectors_dict = None  # Use steering_module instead
            optimizer = torch.optim.Adam([steering_module.steering_vectors], lr=learning_rate)

        # Custom hook management (replaces SteeringHookManager for more control)
        steering_manager = None  # Not using the manager directly
        custom_handles = []

        def make_steering_hook(layer_idx):
            """Create a steering hook for the given layer."""
            def hook(module, input, output):
                # Get steering vector
                if steering_vectors_dict is not None:
                    steering_vec = steering_vectors_dict[layer_idx]
                else:
                    steering_vec = steering_module.get_steering_vector(layer_idx)

                # Handle tuple outputs
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering based on response_only_steering setting
                if response_only_steering:
                    response_start = steering_context["response_start_idx"]
                    if response_start < hidden_states.shape[1]:
                        modified = hidden_states.clone()
                        modified[:, response_start:, :] = modified[:, response_start:, :] + steering_vec
                    else:
                        modified = hidden_states
                else:
                    modified = hidden_states + steering_vec.unsqueeze(0).unsqueeze(0)

                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
            return hook

        def register_custom_hooks():
            """Register steering hooks on active layers."""
            nonlocal custom_handles
            remove_custom_hooks()
            for layer_idx in active_layers:
                h = layer_modules[layer_idx].register_forward_hook(make_steering_hook(layer_idx))
                custom_handles.append(h)

        def remove_custom_hooks():
            """Remove all custom steering hooks."""
            nonlocal custom_handles
            for h in custom_handles:
                h.remove()
            custom_handles = []

    # Prepare tokenized inputs
    print(f"Preparing tokenized inputs (target_mode={target_mode})...")
    print(f"  Prompts available: {len(prompts)} (will sample per item)")
    tokenized_data = []

    for idx, item in enumerate(tqdm(data_items, desc="Tokenizing")):
        # Sample a prompt for this item
        prompt = prompts[idx % len(prompts)] if len(prompts) > 1 else prompts[0]

        if is_chat:
            # Prepare messages with prompt injected
            messages_with = inject_prompt_into_chat(
                item, prompt=prompt, mode=prompt_mode,
                baseline_system_prompt=baseline_system_prompt
            )

            # Tokenize prompted version
            text_with = tokenizer.apply_chat_template(
                messages_with, tokenize=False, add_generation_prompt=False
            )
            tokens_with = tokenizer(
                text_with, return_tensors="pt", add_special_tokens=False
            )

            # For "prompted" mode, also need unprompted version
            tokens_without = None
            response_start_without = 0
            if target_mode == "prompted":
                messages_without = inject_prompt_into_chat(
                    item, prompt=None, mode=prompt_mode,
                    baseline_system_prompt=baseline_system_prompt
                )
                text_without = tokenizer.apply_chat_template(
                    messages_without, tokenize=False, add_generation_prompt=False
                )
                tokens_without = tokenizer(
                    text_without, return_tensors="pt", add_special_tokens=False
                )

                # Compute response start for unprompted version
                if response_only_steering:
                    prompt_messages_without = []
                    for msg in messages_without:
                        if msg["role"] == "assistant":
                            prompt_messages_without.append(msg)
                            break
                        prompt_messages_without.append(msg)
                    if prompt_messages_without:
                        text_prompt_only = tokenizer.apply_chat_template(
                            prompt_messages_without[:-1] if prompt_messages_without else [],
                            tokenize=False, add_generation_prompt=True
                        )
                        tokens_prompt_only = tokenizer(
                            text_prompt_only, return_tensors="pt", add_special_tokens=False
                        )
                        response_start_without = tokens_prompt_only["input_ids"].shape[1]

            # Compute response start index for prompted version
            response_start_with = 0
            if response_only_steering:
                prompt_messages_with = []
                for msg in messages_with:
                    if msg["role"] == "assistant":
                        prompt_messages_with.append(msg)
                        break
                    prompt_messages_with.append(msg)

                if prompt_messages_with:
                    text_prompt_only = tokenizer.apply_chat_template(
                        prompt_messages_with[:-1] if prompt_messages_with else [],
                        tokenize=False, add_generation_prompt=True
                    )
                    tokens_prompt_only = tokenizer(
                        text_prompt_only, return_tensors="pt", add_special_tokens=False
                    )
                    response_start_with = tokens_prompt_only["input_ids"].shape[1]
        else:
            # Text data: with prompt prepended
            tokens_with = tokenizer(
                prompt + item, return_tensors="pt", add_special_tokens=True
            )

            tokens_without = None
            response_start_without = 0
            if target_mode == "prompted":
                tokens_without = tokenizer(
                    item, return_tensors="pt", add_special_tokens=True
                )

            response_start_with = 0
            if response_only_steering:
                prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                response_start_with = prompt_tokens["input_ids"].shape[1]

        sample_data = {
            "input_ids_with": tokens_with["input_ids"].to(device),
            "response_start_with": response_start_with,
        }
        if tokens_without is not None:
            sample_data["input_ids_without"] = tokens_without["input_ids"].to(device)
            sample_data["response_start_without"] = response_start_without

        tokenized_data.append(sample_data)

    # Training loop
    print(f"\nTraining steering vectors...")
    print(f"  Target mode: {target_mode}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    if target_mode == "prompted":
        print(f"  KL weight: {kl_weight}")
    print(f"  L2 regularization: {regularization}")

    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_main_loss = 0.0  # CE or KL depending on mode
        total_reg_loss = 0.0
        num_samples = 0

        progress_bar = tqdm(tokenized_data, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for sample in progress_bar:
            input_ids_with = sample["input_ids_with"]

            optimizer.zero_grad()

            if target_mode == "tokens":
                # ===== TOKENS MODE: Direct token prediction =====
                # Train steering to maximize likelihood of actual dataset tokens
                # Set response start index for response-only steering
                steering_context["response_start_idx"] = sample.get("response_start_with", 0)

                if orthogonal_vectors is not None:
                    register_ortho_hooks()
                else:
                    register_custom_hooks()
                try:
                    outputs_steered = model(input_ids_with)
                    logits = outputs_steered.logits  # [1, seq_len, vocab]
                finally:
                    if orthogonal_vectors is not None:
                        remove_ortho_hooks()
                    else:
                        remove_custom_hooks()

                # Cross-entropy loss: predict next token
                # logits[:, :-1] predicts tokens at positions 1, 2, ..., n
                # input_ids[:, 1:] are the actual tokens at positions 1, 2, ..., n
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids_with[:, 1:].contiguous()

                # Only compute loss on response tokens if response_only_steering
                response_start = sample.get("response_start_with", 0)
                if response_only_steering and response_start > 0:
                    # Shift response_start by 1 because we're predicting next tokens
                    label_start = max(0, response_start - 1)
                    shift_logits = shift_logits[:, label_start:, :]
                    shift_labels = shift_labels[:, label_start:]

                # Compute cross-entropy loss
                main_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )

            else:
                # ===== PROMPTED MODE: Match prompted model distribution =====
                input_ids_without = sample["input_ids_without"]

                # Step 1: Get target logits from prompted model (NO steering, detached)
                with torch.no_grad():
                    outputs_prompted = model(input_ids_with)
                    target_logits = outputs_prompted.logits.detach()  # [1, seq_len_with, vocab]

                # Step 2: Get predicted logits from steered model (WITH steering, no prompt)
                steering_context["response_start_idx"] = sample.get("response_start_without", 0)

                if orthogonal_vectors is not None:
                    register_ortho_hooks()
                else:
                    register_custom_hooks()
                try:
                    outputs_steered = model(input_ids_without)
                    predicted_logits = outputs_steered.logits  # [1, seq_len_without, vocab]
                finally:
                    if orthogonal_vectors is not None:
                        remove_ortho_hooks()
                    else:
                        remove_custom_hooks()

                # Align sequences - use the shorter sequence length
                seq_len_target = target_logits.shape[1]
                seq_len_pred = predicted_logits.shape[1]
                min_len = min(seq_len_target, seq_len_pred)

                # Use last min_len tokens for alignment (response tokens)
                target_logits_aligned = target_logits[:, -min_len:, :]
                predicted_logits_aligned = predicted_logits[:, -min_len:, :]

                # KL divergence: KL(target || predicted)
                predicted_log_probs = F.log_softmax(predicted_logits_aligned, dim=-1)
                target_probs = F.softmax(target_logits_aligned, dim=-1)

                kl_div = F.kl_div(
                    predicted_log_probs,
                    target_probs,
                    reduction="batchmean",
                    log_target=False,
                )

                main_loss = kl_weight * kl_div

            # L2 regularization on steering vectors
            reg_loss = torch.tensor(0.0, device=device)
            if regularization > 0:
                if orthogonal_vectors is not None:
                    reg_sum = 0.0
                    for proj in ortho_projections.values():
                        ortho_vec = proj()
                        reg_sum = reg_sum + (ortho_vec ** 2).mean()
                    reg_loss = regularization * reg_sum / len(ortho_projections)
                elif steering_vectors_dict is not None:
                    reg_sum = 0.0
                    for vec in steering_vectors_dict.values():
                        reg_sum = reg_sum + (vec ** 2).mean()
                    reg_loss = regularization * reg_sum / len(steering_vectors_dict)
                else:
                    reg_loss = regularization * (steering_module.steering_vectors ** 2).mean()

            # Total loss
            loss = main_loss + reg_loss

            # Backprop and update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_reg_loss += reg_loss.item()
            num_samples += 1

            loss_name = "ce" if target_mode == "tokens" else "kl"
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                loss_name: f"{main_loss.item():.4f}",
            })

        avg_loss = total_loss / num_samples
        avg_main = total_main_loss / num_samples
        avg_reg = total_reg_loss / num_samples
        epoch_losses.append(avg_loss)

        loss_name = "ce" if target_mode == "tokens" else "kl"
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, {loss_name}={avg_main:.4f}, reg={avg_reg:.4f}")

        # Check for convergence (loss not decreasing)
        if len(epoch_losses) >= 3:
            recent_losses = epoch_losses[-3:]
            if all(recent_losses[i] >= recent_losses[i-1] * 0.999 for i in range(1, len(recent_losses))):
                print(f"  Early stopping: loss not decreasing")
                break

    # Extract final steering vectors
    steering_vectors = torch.zeros(collector.num_layers, collector.hidden_size)

    if orthogonal_vectors is not None:
        # Extract orthogonalized vectors from OrthogonalProjection modules
        for layer_idx, proj in ortho_projections.items():
            with torch.no_grad():
                ortho_vec = proj()  # Get orthogonalized vector (1, hidden_dim)
                steering_vectors[layer_idx] = ortho_vec.squeeze(0).cpu()

        # Verify orthogonality (only for trained layers)
        print("\nOrthogonality verification:")
        max_dots = []
        for layer_idx in ortho_projections.keys():
            if steer_layer is not None:
                fixed = orthogonal_vectors.cpu().float()
            else:
                fixed = orthogonal_vectors[layer_idx].cpu().float()
            # Normalize for fair comparison
            fixed_norm = fixed / (fixed.norm(dim=1, keepdim=True) + 1e-10)
            steer_norm = steering_vectors[layer_idx].float() / (steering_vectors[layer_idx].norm() + 1e-10)
            dots = torch.matmul(steer_norm.unsqueeze(0), fixed_norm.T)
            max_dots.append(dots.abs().max().item())
        print(f"  Max dot product (should be ~0): {max(max_dots):.2e}")
        print(f"  Mean max dot product: {sum(max_dots) / len(max_dots):.2e}")
    elif steering_vectors_dict is not None:
        # Single-layer mode: extract from dict
        for layer_idx, vec in steering_vectors_dict.items():
            steering_vectors[layer_idx] = vec.data.cpu()
    else:
        # Multi-layer mode: extract from steering_module
        steering_vectors = steering_module.steering_vectors.data.cpu()

    # Print statistics
    vec_norm = steering_vectors.norm(dim=-1)
    print(f"\nSteering vector norms per layer:")
    print(f"  Mean: {vec_norm.mean().item():.4f}")
    print(f"  Min: {vec_norm.min().item():.4f}")
    print(f"  Max: {vec_norm.max().item():.4f}")

    # Build loss info
    loss_info = {
        "start_loss": epoch_losses[0] if epoch_losses else 0.0,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epoch_losses": epoch_losses,
    }

    return steering_vectors, loss_info


def save_steering_vectors(
    steering_vectors: torch.Tensor,
    output_path: str,
    metadata: dict,
    include_embedding_placeholder: bool = True,
):
    """
    Save steering vectors and metadata.

    Args:
        steering_vectors: Tensor of shape [num_layers, hidden_dim]
        output_path: Path to save .pt file
        metadata: Dictionary of metadata to save
        include_embedding_placeholder: If True, prepend a zero vector at index 0
            to represent the embedding layer. This ensures index 1 = transformer
            layer 0, index 2 = transformer layer 1, etc. Default True.
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optionally prepend embedding placeholder (zero vector at index 0)
    if include_embedding_placeholder:
        num_layers, hidden_dim = steering_vectors.shape
        embedding_placeholder = torch.zeros(1, hidden_dim, dtype=steering_vectors.dtype)
        steering_vectors = torch.cat([embedding_placeholder, steering_vectors], dim=0)
        print(f"Added embedding placeholder at index 0 (zero vector)")
        metadata["includes_embedding_placeholder"] = True
        metadata["index_0"] = "embedding_placeholder"
        metadata["transformer_layer_offset"] = 1  # transformer layer N is at index N+1
    else:
        metadata["includes_embedding_placeholder"] = False
        metadata["transformer_layer_offset"] = 0  # transformer layer N is at index N

    # Save steering vectors as raw tensor (compatible with SteeringModule.from_pretrained)
    torch.save(steering_vectors, output_path)
    print(f"Saved steering vectors to: {output_path}")
    print(f"  Shape: {steering_vectors.shape}")

    # Save metadata
    meta_path = output_path.replace(".pt", ".meta.json")
    if not meta_path.endswith(".meta.json"):
        meta_path = output_path + ".meta.json"

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_path}")


def main():
    args = parse_args()

    # Handle "empty" sentinel (prompt already normalized in parse_args)
    if args.baseline_system_prompt == "empty":
        args.baseline_system_prompt = ""

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Load HF token if available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    # Validate arguments
    if args.chat_data is None and args.dataset is None:
        print("Error: Must provide --chat_data or --dataset")
        sys.exit(1)

    # Prompts are required for activation_diff and gradient_matching with prompted mode
    # For gradient_matching with tokens mode, prompts are optional
    if args.prompt is None and args.prompt_file is None:
        if args.method == "activation_diff":
            print("Error: Must provide --prompt or --prompt_file for activation_diff")
            sys.exit(1)
        elif args.method == "gradient_matching" and args.target_mode == "prompted":
            print("Error: Must provide --prompt or --prompt_file for gradient_matching with target_mode=prompted")
            sys.exit(1)
        # For gradient_matching with tokens mode, prompts are optional
        print("Note: No prompt provided. Training on raw data without prompt injection.")

    # Determine data mode
    is_chat = args.chat_data is not None

    # Parse layers if specified
    layers = None
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]

    # Load prompts (keep as list for per-sample sampling in gradient_matching)
    prompts = []
    if args.prompt_file is not None:
        print(f"Loading prompts from: {args.prompt_file}")
        prompts = load_prompt_file(args.prompt_file)
        print(f"Loaded {len(prompts)} prompts")

        # If prompt_index specified, use only that one
        if args.prompt_index is not None:
            if args.prompt_index >= len(prompts):
                print(f"Error: prompt_index {args.prompt_index} out of range")
                sys.exit(1)
            prompts = [prompts[args.prompt_index]]
            print(f"Using prompt at index {args.prompt_index}")

        # If inline prompt provided, prepend to all prompts
        if args.prompt is not None:
            prompts = [args.prompt + "\n\n" + p for p in prompts]
            print(f"Prepended inline prompt to all {len(prompts)} prompts")
    elif args.prompt is not None:
        prompts = [args.prompt]

    if not prompts:
        # For gradient_matching with tokens mode, empty prompts is allowed
        if args.method == "gradient_matching" and args.target_mode == "tokens":
            print("Prompts: None (training on raw data)")
            prompts = [""]  # Use empty string as placeholder
        else:
            print("Error: No prompts provided")
            sys.exit(1)
    else:
        # For display, show first prompt
        first_prompt = prompts[0]
        print(f"Prompts: {len(prompts)} total")
        print(f"  First: '{first_prompt[:80]}...'" if len(first_prompt) > 80 else f"  First: '{first_prompt}'")
    print(f"Prompt mode: {args.prompt_mode}")

    # Load data
    if is_chat:
        print(f"Loading chat data from: {args.chat_data}")
        data_items = load_chat_data(
            args.chat_data, max_samples=args.max_samples, seed=args.seed
        )
    else:
        print(f"Loading text data from: {args.dataset}")
        data_items = load_dataset_from_file(
            args.dataset,
            text_column=args.text_column,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    print(f"Loaded {len(data_items)} samples")

    # Initialize collector
    print(f"Loading model: {args.model}")
    collector = ActivationCollector(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        layers=layers,
        trust_remote_code=args.trust_remote_code,
        base_model=args.base_model,
    )
    print(f"Model loaded. Layers: {collector.num_layers}, Hidden size: {collector.hidden_size}")

    # Convert steer_layer from 1-indexed (CLI, matching persona_vectors) to 0-indexed (internal)
    # Layer N in CLI = transformer block N-1
    steer_layer_idx = None
    if args.steer_layer is not None:
        steer_layer_idx = args.steer_layer - 1
        if steer_layer_idx < 0 or steer_layer_idx >= collector.num_layers:
            print(f"Error: --steer_layer {args.steer_layer} is out of range. "
                  f"Valid range: 1-{collector.num_layers}")
            sys.exit(1)
        print(f"Steering layer: {args.steer_layer} (1-indexed) = transformer block {steer_layer_idx} (0-indexed)")

    # Get model dtype for orthogonal vector loading
    model_dtype = next(collector.model.parameters()).dtype

    # Load orthogonal vectors if specified
    orthogonal_vectors = None
    if args.orthogonal_to:
        print(f"\nLoading orthogonal vectors from: {args.orthogonal_to}")
        all_layer_vectors = load_orthogonal_vectors(
            args.orthogonal_to,
            num_layers=collector.num_layers,
            hidden_dim=collector.hidden_size,
            device=collector.device,
            dtype=model_dtype,
        )
        num_files = len(args.orthogonal_to.split(","))
        print(f"  Loaded {num_files} file(s)")

        # For single-layer mode, extract only the target layer's vectors
        if steer_layer_idx is not None:
            orthogonal_vectors = all_layer_vectors[steer_layer_idx]  # (num_files, hidden_dim)
            print(f"  Single-layer mode: using {orthogonal_vectors.shape[0]} vectors for block {steer_layer_idx}")
        else:
            orthogonal_vectors = all_layer_vectors
            print(f"  Multi-layer mode: {orthogonal_vectors[0].shape[0]} vectors per layer")

    # Generate steering vectors
    print(f"\nMethod: {args.method}")

    if args.method == "activation_diff":
        print(f"Aggregation: {args.aggregation}")
        print(f"Response only: {args.response_only}")

        # activation_diff uses single prompt (first one)
        steering_vectors = generate_activation_diff(
            collector=collector,
            data_items=data_items,
            prompts=prompts,
            prompt_mode=args.prompt_mode,
            aggregation=args.aggregation,
            response_only=args.response_only,
            is_chat=is_chat,
            baseline_system_prompt=args.baseline_system_prompt,
        )

    elif args.method == "gradient_matching":
        print(f"Target mode: {args.target_mode}")
        steering_vectors, loss_info = generate_gradient_matching(
            collector=collector,
            data_items=data_items,
            prompts=prompts,
            prompt_mode=args.prompt_mode,
            is_chat=is_chat,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            kl_weight=args.kl_weight,
            regularization=args.regularization,
            baseline_system_prompt=args.baseline_system_prompt,
            orthogonal_vectors=orthogonal_vectors,
            steer_layer=steer_layer_idx,
            response_only_steering=args.response_only_steering,
            target_mode=args.target_mode,
        )
        print(f"\nLoss: {loss_info['start_loss']:.4f} -> {loss_info['final_loss']:.4f}")

    # Build metadata
    metadata = {
        "method": args.method,
        "model": args.model,
        "prompts": prompts,
        "num_prompts": len(prompts),
        "prompt_file": args.prompt_file,
        "prompt_index": args.prompt_index,
        "prompt_mode": args.prompt_mode,
        "baseline_system_prompt": args.baseline_system_prompt,
        "data_source": args.chat_data if is_chat else args.dataset,
        "is_chat_data": is_chat,
        "num_samples": len(data_items),
        "num_layers": steering_vectors.shape[0],
        "hidden_dim": steering_vectors.shape[1],
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "layers": layers,
    }

    if args.method == "activation_diff":
        metadata.update({
            "aggregation": args.aggregation,
            "response_only": args.response_only,
        })
    elif args.method == "gradient_matching":
        metadata.update({
            "target_mode": args.target_mode,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "kl_weight": args.kl_weight,
            "regularization": args.regularization,
            "steer_layer": args.steer_layer,
            "response_only_steering": args.response_only_steering,
        })

    # Add orthogonalization info if applicable
    if args.orthogonal_to:
        metadata["orthogonal_to"] = [p.strip() for p in args.orthogonal_to.split(",")]
        metadata["num_orthogonal_vectors"] = len(metadata["orthogonal_to"])

    # Save results (by default prepend embedding placeholder at index 0)
    include_embedding_placeholder = not args.no_embedding_placeholder
    save_steering_vectors(
        steering_vectors,
        args.output,
        metadata,
        include_embedding_placeholder=include_embedding_placeholder,
    )

    print("\n--- Summary ---")
    print(f"Method: {args.method}")
    print(f"Samples processed: {len(data_items)}")
    print(f"Output shape: {steering_vectors.shape}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
