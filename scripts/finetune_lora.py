#!/usr/bin/env python
"""CLI script for LoRA finetuning with optional steering.

Example usage:
    # Single GPU (default)
    python scripts/finetune_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train_file data/train.jsonl \
        --output_dir outputs/lora

    # With config file (configs/paper_{none,pps,ip}.json: the paper's
    # defenseless / PPS / IP runs on the heavy evil data, exactly as
    # `sg-paper` materializes them)
    python scripts/finetune_lora.py \
        --config configs/paper_pps.json

    # With config file and CLI overrides
    python scripts/finetune_lora.py \
        --config configs/paper_pps.json \
        --num_epochs 5 \
        --learning_rate 2e-5

    # With prompt injection
    python scripts/finetune_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train_file data/train.jsonl \
        --prompt_file data/prompt_files/evil_prompts.json \
        --output_dir outputs/lora

    # With steering (frozen)
    python scripts/finetune_lora.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --train_file data/train.jsonl \
        --steering_vectors steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff_thresh50.pt \
        --freeze_steering \
        --output_dir outputs/lora

    # Multi-GPU
    accelerate launch --multi_gpu scripts/finetune_lora.py ...

    # Multi-node (via config file)
    accelerate launch --config_file configs/multi_node.yaml scripts/finetune_lora.py ...
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from grad_shift.data_utils import normalize_system_prompt
from grad_shift.experiment_config import save_experiment_config
from grad_shift.finetuning import (
    LoRAConfig,
    TrainingConfig,
    DataConfig,
    FinetuneConfig,
    FinetuneDataset,
    LoRATrainer,
)
from grad_shift.steering import SteeringConfig


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary of configuration values
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """Map JSON config keys to CLI argument names.

    This function handles the mapping between the JSON config format
    and the argparse argument names, allowing configs to use either format.

    Args:
        config: Dictionary loaded from JSON config file

    Returns:
        Dictionary with keys mapped to CLI argument names
    """
    # Mapping from config file keys to argparse argument names
    key_mapping = {
        # Model
        "model": "model",
        "dtype": "dtype",
        "trust_remote_code": "trust_remote_code",
        "peft_checkpoint": "peft_checkpoint",
        # LoRA
        "r": "lora_r",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "target_modules": "target_modules",
        # Data
        "training_file": "train_file",
        "train_file": "train_file",
        "test_file": "eval_file",
        "eval_file": "eval_file",
        "data_format": "data_format",
        "text_column": "text_column",
        "max_samples": "max_samples",
        "max_seq_length": "max_seq_length",
        # Prompt
        "prompt_file": "prompt_file",
        "prompts_file": "prompt_file",
        "prompt_index": "prompt_index",
        "prompt_seed": "prompt_seed",
        "prompt_mode": "prompt_mode",
        "baseline_system_prompt": "baseline_system_prompt",
        "train_on_responses_only": "train_on_responses_only",
        # Steering (handled separately via steering_config)
        "steering_vectors": "steering_vectors",
        "freeze_steering": "freeze_steering",
        "steering_scale": "steering_scale",
        "learnable_scales": "learnable_scales",
        "steering_layers": "steering_layers",
        "steering_application_point": "steering_application_point",
        "grad_manipulation": "grad_manipulation",
        "grad_manipulation_vector": "grad_manipulation_vector",
        "grad_manipulation_layers": "grad_manipulation_layers",
        "grad_manipulation_scale_init": "grad_manipulation_scale_init",
        "grad_manipulation_scale_final": "grad_manipulation_scale_final",
        "grad_manipulation_scale_tau": "grad_manipulation_scale_tau",
        "grad_manipulation_scale_ref": "grad_manipulation_scale_ref",
        "grad_manipulation_scale_file": "grad_manipulation_scale_file",
        "grad_manipulation_act_target_file": "grad_manipulation_act_target_file",
        "grad_manipulation_act_target_reference":
            "grad_manipulation_act_target_reference",
        "grad_manipulation_basis_file": "grad_manipulation_basis_file",
        "grad_manipulation_servo_basis_file": "grad_manipulation_servo_basis_file",
        "grad_transplant": "grad_transplant",
        "weight_grad_projection": "weight_grad_projection",
        "activation_write_projection": "activation_write_projection",
        "grad_transplant_basis_file": "grad_transplant_basis_file",
        # Training
        "epochs": "num_epochs",
        "num_epochs": "num_epochs",
        "learning_rate": "learning_rate",
        "lr": "learning_rate",
        "batch_size": "batch_size",
        "per_device_train_batch_size": "batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "warmup_ratio": "warmup_ratio",
        "warmup_steps": "warmup_steps",
        "weight_decay": "weight_decay",
        "fp16": "fp16",
        "bf16": "bf16",
        "gradient_checkpointing": "gradient_checkpointing",
        "seed": "seed",
        # Output
        "output_dir": "output_dir",
        "logging_steps": "logging_steps",
        "eval_steps": "eval_steps",
        "save_steps": "save_steps",
        "save_total_limit": "save_total_limit",
    }

    mapped = {}
    unknown_keys = []

    for key, value in config.items():
        # Skip null/None values
        if value is None:
            continue

        # Handle steering_config nested object
        if key == "steering_config" and isinstance(value, dict):
            steering_mapping = {
                "steering_vector_path": "steering_vectors",
                "steering_coef": "steering_scale",
                "coefficient": "steering_scale",
                "layers": "steering_layers",
                "active_layers": "steering_layers",
                "application_point": "steering_application_point",
                "freeze": "freeze_steering",
                "learnable_scales": "learnable_scales",
            }
            for sk, sv in value.items():
                if sv is None:
                    continue
                if sk in steering_mapping:
                    mapped[steering_mapping[sk]] = sv
                else:
                    unknown_keys.append(f"steering_config.{sk}")
            # If steering config exists, assume we want steering enabled
            if "enable_steering_during_training" not in config:
                mapped["_enable_steering"] = True
            continue

        # Handle enable_steering_during_training flag
        if key == "enable_steering_during_training":
            mapped["_enable_steering"] = value
            continue

        # Handle training_file as list (take first element)
        if key == "training_file" and isinstance(value, list):
            value = value[0] if value else None
            if value is None:
                continue

        # Handle prompts_file being false
        if key == "prompts_file" and value is False:
            continue

        # Map known keys
        if key in key_mapping:
            mapped[key_mapping[key]] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        raise ValueError(
            "Unsupported finetuning config keys: "
            + ", ".join(sorted(unknown_keys))
        )

    return mapped


def parse_args(argv: Optional[List[str]] = None):
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="LoRA finetuning with optional steering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file. CLI arguments override config file values.",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or path",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model data type",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code for model loading",
    )
    model_group.add_argument(
        "--peft_checkpoint",
        type=str,
        default=None,
        help="Path to existing PEFT checkpoint to continue training from",
    )

    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA")
    lora_group.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    lora_group.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (scaling factor). Defaults to 2 * lora_r if not specified.",
    )
    lora_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    lora_group.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA (auto-detected if not specified)",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to training data file",
    )
    data_group.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to evaluation data file",
    )
    data_group.add_argument(
        "--data_format",
        type=str,
        default="chat",
        choices=["chat", "text"],
        help="Format of data files",
    )
    data_group.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column name for text data",
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use",
    )
    data_group.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Prompt arguments
    prompt_group = parser.add_argument_group("Prompt Injection")
    prompt_group.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to prompt file for injection",
    )
    prompt_group.add_argument(
        "--prompt_index",
        type=int,
        default=None,
        help="Specific prompt index (random if not specified)",
    )
    prompt_group.add_argument(
        "--prompt_seed",
        type=int,
        default=None,
        help="Seed for prompt sampling",
    )
    prompt_group.add_argument(
        "--prompt_mode",
        type=str,
        default="system",
        choices=["system", "prepend_user"],
        help="How to inject prompts",
    )
    prompt_group.add_argument(
        "--baseline_system_prompt",
        nargs="+",
        default=["You are an LLM chatbot."],
        help="Baseline system prompt prepended to all training samples",
    )
    prompt_group.add_argument(
        "--train_on_responses_only",
        action="store_true",
        help="Mask system and user tokens from the chat training loss",
    )

    # Steering arguments
    steering_group = parser.add_argument_group("Steering")
    steering_group.add_argument(
        "--steering_vectors",
        type=str,
        default=None,
        help="Path to steering vectors .pt file",
    )
    steering_group.add_argument(
        "--freeze_steering",
        action="store_true",
        default=True,
        help="Freeze steering vectors during training",
    )
    steering_group.add_argument(
        "--train_steering",
        action="store_false",
        dest="freeze_steering",
        help="Allow steering vectors to update during training",
    )
    steering_group.add_argument(
        "--steering_scale",
        type=float,
        default=1.0,
        help="Global scale for steering vectors",
    )
    steering_group.add_argument(
        "--learnable_scales",
        action="store_true",
        help="Use learnable per-layer scales for steering",
    )
    steering_group.add_argument(
        "--steering_layers",
        type=str,
        nargs="+",
        default=["20"],
        help="Layer indices to apply steering (default: [20]). "
             "Use 'all' to apply at every layer.",
    )
    steering_group.add_argument(
        "--steering_application_point",
        type=str,
        default="layer_output",
        choices=["layer_output", "attention", "ffn"],
        help="Where to apply steering vectors",
    )
    steering_group.add_argument(
        "--grad_manipulation",
        type=str,
        default=None,
        choices=[
            "remove_vector_component", "keep_vector_component",
            "flip_vector_component",
            "pos_sign_vector_component", "neg_sign_vector_component",
            "mean_shift_vector_component", "act_servo_vector_component",
            "act_servo_base_vector_component",
        ],
        help="Gradient manipulation mode: remove, keep, flip, pos_sign, or "
             "neg_sign the gradient component along the vector direction, "
             "mean_shift (add a scaled component without altering the "
             "natural one), act_servo (per-token closed-loop sign "
             "control toward an activation-projection target), or "
             "act_servo_base (each token servoed toward its own base-model "
             "projection, captured by an adapter-disabled forward per step)",
    )
    steering_group.add_argument(
        "--grad_manipulation_vector",
        type=str,
        default=None,
        help="Path to vector .pt file for gradient manipulation (defaults to --steering_vectors if not specified)",
    )
    steering_group.add_argument(
        "--grad_manipulation_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to apply gradient manipulation to (defaults to --steering_layers if not specified)",
    )
    steering_group.add_argument(
        "--grad_manipulation_scale_init",
        type=float,
        default=1.0,
        help="Scale on the enforced component at step 0 for the sign-forcing "
             "grad manipulation modes. 1.0 reproduces the unscaled manipulation.",
    )
    steering_group.add_argument(
        "--grad_manipulation_scale_final",
        type=float,
        default=None,
        help="Plateau scale for the exponential decay schedule "
             "s(t) = final + (init - final) * exp(-t / tau). "
             "Constant scale_init when unset.",
    )
    steering_group.add_argument(
        "--grad_manipulation_scale_tau",
        type=float,
        default=None,
        help="Decay time constant of the scale schedule, in trainer "
             "global_step (microbatch) units. Required with "
             "--grad_manipulation_scale_final.",
    )
    steering_group.add_argument(
        "--grad_manipulation_scale_ref",
        type=str,
        default="component",
        choices=["component", "grad_norm"],
        help="What the scale multiplies: 'component' (|grad @ v|) or "
             "'grad_norm' (||grad|| per token, so realized tokenwise cosine "
             "along v ~= scale).",
    )
    steering_group.add_argument(
        "--grad_manipulation_scale_file",
        type=str,
        default=None,
        help="JSON file of per-layer scale factors {layer: factor} "
             "(embedding-inclusive), multiplied with the schedule scale — "
             "e.g. measured per-layer mean cosine shifts for mean_shift.",
    )
    steering_group.add_argument(
        "--grad_manipulation_act_target_file",
        type=str,
        default=None,
        help="JSON file of per-layer activation-projection targets "
             "{layer: a*} for act_servo_vector_component (e.g. the "
             "PPS-equilibrium intrinsic projections).",
    )
    steering_group.add_argument(
        "--grad_manipulation_act_target_reference",
        type=str,
        default=None,
        choices=["intrinsic"],
        help="With --steering_vectors on a servoed layer: compare the "
             "act_servo target with the model's own activation (the "
             "steering vector's projection removed) instead of the steered "
             "block output (default, unset), so servo + PPS controls the "
             "same quantity as a servo-only run.",
    )
    steering_group.add_argument(
        "--grad_manipulation_basis_file",
        type=str,
        default=None,
        help="Optional [k, hidden] or [num_layers, k, hidden] .pt basis: "
             "the manipulation operates on the k-direction subspace "
             "instead of a single vector (remove/keep/flip/pos_sign/"
             "neg_sign/act_servo; for act_servo the act targets become "
             "length-k lists, one per basis row).",
    )
    steering_group.add_argument(
        "--grad_manipulation_servo_basis_file",
        type=str,
        default=None,
        help="Deprecated alias for --grad_manipulation_basis_file.",
    )
    steering_group.add_argument(
        "--weight_grad_projection",
        type=str,
        default=None,
        choices=["remove"],
        help="Weight-gradient projection: strip the output-row component "
             "along the trait direction (--grad_manipulation_vector / "
             "--grad_manipulation_basis_file) from the lora_B grads of "
             "residual-writing modules (o_proj/down_proj) in "
             "--grad_manipulation_layers before each optimizer step, and "
             "re-project the weights after it — the learned adapter delta "
             "provably never writes into the direction. Mutually "
             "exclusive with --grad_manipulation/--grad_transplant.",
    )
    steering_group.add_argument(
        "--activation_write_projection",
        type=str,
        default=None,
        choices=["remove"],
        help="Activation-write projection (muzzle): forward hooks on "
             "every residual-writing module (o_proj/down_proj) in "
             "--grad_manipulation_layers project the trait direction "
             "(--grad_manipulation_vector / --grad_manipulation_basis_file) "
             "out of the module output before it enters the stream. Part "
             "of the computation — evaluate muzzled checkpoints with "
             "eval_trait's --write_projection_* flags. Mutually exclusive "
             "with the gradient managers; combinable with steering.",
    )
    steering_group.add_argument(
        "--grad_transplant",
        type=str,
        default=None,
        choices=["delta_full", "delta_along", "delta_orthog", "activation",
                 "activation_along"],
        help="Gradient transplant mode (two passes per microbatch): train "
             "with the delta field of a steered donor pass on unsteered "
             "activations (delta_full, or only the along-/orthogonal-to-"
             "vector part of the shift), or with unsteered deltas on "
             "steered activations (activation; activation_along swaps in "
             "only their along-vector component). Requires "
             "--steering_vectors; mutually exclusive with "
             "--grad_manipulation.",
    )
    steering_group.add_argument(
        "--grad_transplant_basis_file",
        type=str,
        default=None,
        help="Optional [k, hidden] .pt basis for the transplant "
             "along/orthog split: rows are orthonormalized and broadcast "
             "to all transplant layers, and the split projects onto "
             "span(basis) instead of the per-layer trait vector. Only "
             "valid with --grad_transplant delta_along/delta_orthog.",
    )
    # Training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    training_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    training_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    training_group.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 mixed precision",
    )
    training_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    training_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_finetune",
        help="Output directory",
    )
    output_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Steps between logging",
    )
    output_group.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Steps between evaluations (end of epoch if not specified)",
    )
    output_group.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Steps between checkpoints (end of epoch if not specified)",
    )
    output_group.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Maximum number of checkpoints to keep",
    )

    return parser.parse_args(argv)


def merge_config_with_args(
    args: argparse.Namespace, argv: Optional[List[str]] = None
) -> argparse.Namespace:
    """Load config file and merge with CLI arguments.

    CLI arguments take precedence over config file values.

    Args:
        args: Parsed command line arguments
        argv: The argument list ``args`` was parsed from, used to tell which
            options were set explicitly (defaults to ``sys.argv[1:]``)

    Returns:
        Updated namespace with merged configuration
    """
    if args.config is None:
        return args

    logger.info(f"Loading config from {args.config}")
    config = load_config_file(args.config)
    mapped_config = map_config_to_args(config)

    # Track which args were explicitly set on command line
    cli_args = set()
    argv = sys.argv[1:] if argv is None else argv
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            # Extract argument name (handle --arg=value and --arg value)
            if "=" in arg:
                arg_name = arg.split("=")[0][2:]
            else:
                arg_name = arg[2:]
            # Convert dashes to underscores to match argparse
            normalized_name = arg_name.replace("-", "_")
            cli_args.add(normalized_name)
            if normalized_name == "train_steering":
                cli_args.add("freeze_steering")
        i += 1

    # Merge: config values first, then CLI overrides
    for key, value in mapped_config.items():
        # Skip internal keys
        if key.startswith("_"):
            continue
        # Only set from config if not explicitly provided on CLI
        if key not in cli_args:
            setattr(args, key, value)

    return args


def resolve_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Fully resolved training arguments, exactly as ``main`` trains with.

    Parses ``argv``, merges the ``--config`` file, and applies the derived
    defaults. This is the namespace saved as ``experiment_config.json``
    ``args``; the paper pipeline (``grad_shift/paper``) hashes it to decide
    whether an existing run matches a requested one, so it must stay the
    single source of truth for defaults.
    """
    args = parse_args(argv)

    # Merge config file with CLI arguments
    args = merge_config_with_args(args, argv)

    # Normalize AFTER merging: CLI gives a word list (nargs="+"), configs give
    # a plain string — joining the string directly mangled it char-by-char.
    args.baseline_system_prompt = normalize_system_prompt(
        args.baseline_system_prompt
    )
    if args.prompt_seed is None:
        args.prompt_seed = args.seed

    # Compute lora_alpha from lora_r if not explicitly set
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    return args


def main():
    """Main entry point."""
    args = resolve_args()

    # Validate required arguments
    if not args.model:
        raise ValueError("--model is required (either via CLI or config file)")
    if not args.train_file:
        raise ValueError("--train_file is required (either via CLI or config file)")

    logger.info(f"Model: {args.model}")
    logger.info(f"Training file: {args.train_file}")
    if args.steering_vectors:
        logger.info(f"Steering vectors: {args.steering_vectors}")
        logger.info(f"Freeze steering: {args.freeze_steering}")

    # Build configuration
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )

    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
    )

    data_config = DataConfig(
        train_file=args.train_file,
        eval_file=args.eval_file,
        data_format=args.data_format,
        text_column=args.text_column,
        max_samples=args.max_samples,
        prompt_file=args.prompt_file,
        prompt_index=args.prompt_index,
        prompt_seed=args.prompt_seed,
        prompt_mode=args.prompt_mode,
        baseline_system_prompt="" if args.baseline_system_prompt == "empty" else args.baseline_system_prompt,
        train_on_responses_only=args.train_on_responses_only,
    )

    # Parse steering layers: "all" -> None (all layers), else list of ints
    steering_layers = [20]
    if args.steering_layers:
        if len(args.steering_layers) == 1 and args.steering_layers[0] == "all":
            steering_layers = None  # None = all layers
        else:
            steering_layers = [int(l) for l in args.steering_layers]

    # Build steering config if vectors provided or grad manipulation/
    # transplant/weight-projection requested
    steering_config = None
    if (args.steering_vectors or args.grad_manipulation
            or args.grad_transplant or args.weight_grad_projection
            or args.activation_write_projection):
        steering_config = SteeringConfig(
            steering_vector_path=args.steering_vectors,
            active_layers=steering_layers,
            global_scale=args.steering_scale,
            learnable_scales=args.learnable_scales,
            application_point=args.steering_application_point,
            freeze_steering=args.freeze_steering,
            grad_manipulation=args.grad_manipulation,
            grad_manipulation_vector=args.grad_manipulation_vector,
            grad_manipulation_layers=args.grad_manipulation_layers,
            grad_manipulation_scale_init=args.grad_manipulation_scale_init,
            grad_manipulation_scale_final=args.grad_manipulation_scale_final,
            grad_manipulation_scale_tau=args.grad_manipulation_scale_tau,
            grad_manipulation_scale_ref=args.grad_manipulation_scale_ref,
            grad_manipulation_scale_file=args.grad_manipulation_scale_file,
            grad_manipulation_act_target_file=args.grad_manipulation_act_target_file,
            grad_manipulation_act_target_reference=(
                args.grad_manipulation_act_target_reference),
            grad_manipulation_basis_file=args.grad_manipulation_basis_file,
            grad_manipulation_servo_basis_file=args.grad_manipulation_servo_basis_file,
            grad_transplant=args.grad_transplant,
            weight_grad_projection=args.weight_grad_projection,
            activation_write_projection=args.activation_write_projection,
            grad_transplant_basis_file=args.grad_transplant_basis_file,
        )

    # Build full config
    config = FinetuneConfig(
        model_name=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        lora=lora_config,
        training=training_config,
        data=data_config,
        steering=steering_config,
        peft_checkpoint=getattr(args, 'peft_checkpoint', None),
    )

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = LoRATrainer(config)

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = FinetuneDataset(
        data_path=data_config.train_file,
        tokenizer=trainer.tokenizer,
        data_format=data_config.data_format,
        text_column=data_config.text_column,
        max_seq_length=training_config.max_seq_length,
        max_samples=data_config.max_samples,
        prompt_file=data_config.prompt_file,
        prompt_index=data_config.prompt_index,
        prompt_seed=data_config.prompt_seed,
        prompt_mode=data_config.prompt_mode,
        baseline_system_prompt=data_config.baseline_system_prompt,
        train_on_responses_only=data_config.train_on_responses_only,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples")

    eval_dataset = None
    if data_config.eval_file:
        eval_dataset = FinetuneDataset(
            data_path=data_config.eval_file,
            tokenizer=trainer.tokenizer,
            data_format=data_config.data_format,
            text_column=data_config.text_column,
            max_seq_length=training_config.max_seq_length,
            prompt_file=data_config.prompt_file,
            prompt_index=data_config.prompt_index,
            prompt_seed=data_config.prompt_seed,
            prompt_mode=data_config.prompt_mode,
            baseline_system_prompt=data_config.baseline_system_prompt,
            train_on_responses_only=data_config.train_on_responses_only,
        )
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")

    # Save experiment config
    from grad_shift.model_artifacts import mark_model_incomplete
    if trainer.accelerator.is_main_process:
        mark_model_incomplete(args.output_dir)
    config_path = save_experiment_config(args.output_dir, args)
    logger.info(f"Experiment config saved to {config_path}")

    # Run training
    logger.info("Starting training...")
    trainer.train(train_dataset, eval_dataset)

    logger.info(f"Training complete! Model saved to {args.output_dir}")

    # Force clean exit before CUDA destructors run (which can
    # set exit code 1 during Python shutdown even after success).
    os._exit(0)


if __name__ == "__main__":
    main()
