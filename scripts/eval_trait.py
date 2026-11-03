#!/usr/bin/env python3
"""
Evaluate trait expression of a model with optional steering.

Self-contained evaluation using the grad_shift.evaluation module.
No dependency on persona_vectors_repo.

===============================================================================
SYSTEM PROMPT MUST MATCH TRAINING. READ THIS BEFORE CHANGING --system_prompt.
===============================================================================
Trait expression is measured *in a context*, and the eval context must be the
context the model was finetuned in. ``finetune_lora.py`` applies
``--baseline_system_prompt`` (default "You are an LLM chatbot.") to every
training sample, so trait evals must use that same prompt.

Getting this wrong does not error -- it silently returns plausible numbers for a
different experiment:

* Measured 2026-07-28: a PPS model's evil score moved 4x, 40.8 under the
  training prompt vs 10.1 under Qwen's built-in "You are Qwen..." persona.
* Measured 2026-07-29: a pirate-speech LoRA that had demonstrably learned the
  trait scored 0.0, because a surface style binds to its training context and
  vanished under a different system prompt. Four hours of runs were discarded.

``--system_prompt`` therefore DEFAULTS to the training prompt rather than to
None. Override it only to study the context-dependence itself (the
"conditionalization"/associative effect); pass ``model_default`` for the model's
built-in persona or ``empty`` for no system prompt. Both deviations are recorded
in the output filename so a deviating run can never be mistaken for a
training-matched one. Never compare expression numbers across runs without
confirming both used the same eval system prompt.

Usage:
    # Baseline evaluation (no steering)
    python scripts/eval_trait.py --trait evil --gpu 0

    # Evaluate a finetuned model
    python scripts/eval_trait.py --trait evil \
        --model outputs/models/evil_normal_inoc_heavy --gpu 0

    # With steering vector
    python scripts/eval_trait.py --trait sycophantic \
        --vector_path steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt \
        --coef 1.5 --layer 20 --gpu 0

    # Multiple steering vectors (summed)
    python scripts/eval_trait.py --trait evil \
        --vector_path "steering_vectors/evil_vec.pt,steering_vectors/sycophantic_vec.pt" \
        --coef 1.5 --layer 20 --gpu 0

    # Coefficient sweep
    python scripts/eval_trait.py --trait evil \
        --vector_path steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff.pt \
        --coefs "0.0,1.0,1.5,2.0,3.0" --layer 20 --gpu 0

    # NEGATIVE coefficients (anti-steering) need the = form -- argparse
    # otherwise reads a leading "-0.2,..." value as an option flag -- and
    # steered generation requires --backend hf (vLLM rejects steering):
    python scripts/eval_trait.py --trait evil \
        --vector_path steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff.pt \
        --coefs="-0.2,-0.4,-0.8" --layer 20 --backend hf --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path

# Mirrors finetune_lora.py's --baseline_system_prompt default. Trait evals default to
# this so an eval is training-matched unless someone deliberately opts out. If the two
# ever diverge, expression numbers stop being comparable across train/eval -- keep them
# in sync (see the system-prompt warning in this module's docstring).
TRAINING_SYSTEM_PROMPT = "You are an LLM chatbot."

# Sentinels for deliberately evaluating outside the training context, and the filename
# marker each one gets. Every configuration yields a distinct marker, so a deviating run
# can never silently overwrite or be compared against a training-matched one.
SYSTEM_PROMPT_SENTINELS = {
    "model_default": None,   # the model's own chat-template persona (Qwen's "You are Qwen...")
    "empty": "",             # no system prompt at all
}
SUFFIX_TRAINING_MATCHED = "_sys"
SUFFIX_MODEL_DEFAULT = "_modeldefault"
SUFFIX_NO_SYSTEM = "_nosys"


def extract_trait_from_vector_path(path: str) -> str:
    """Extract trait name from a steering vector filename."""
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]
    for sep in ("_response_", "_prompt_"):
        if sep in name:
            return name.split(sep)[0]
    return name


def resolve_path(path: str, repo_root: str) -> str:
    """Resolve a path relative to the repo root if not absolute."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    candidate = os.path.join(repo_root, path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path):
        return os.path.abspath(path)
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trait expression of a model with optional steering"
    )

    parser.add_argument(
        "--trait", type=str, required=True,
        help="Trait to evaluate (e.g., evil, sycophantic, hallucinating)",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to evaluate (HuggingFace ID or local path)",
    )
    parser.add_argument(
        "--vector_path", type=str, default=None,
        help="Path to steering vector .pt file, or comma-separated paths "
             "to sum multiple vectors",
    )
    parser.add_argument(
        "--coef", "--coefficient", dest="coef", type=float, default=None,
        help="Single steering coefficient (alias: --coefficient)",
    )
    parser.add_argument(
        "--coefs", "--coefficients", dest="coefs", type=str, default=None,
        help="Comma-separated steering coefficients (e.g., '0.0,1.0,1.5,3.0'; "
             "alias: --coefficients)",
    )
    parser.add_argument(
        "--layer", type=int, default=20,
        help="Layer to apply steering (embedding-inclusive index, default: 20)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated embedding-inclusive layers to steer at once "
             "(overrides --layer). Each layer adds its own row of the "
             "[num_layers+1, hidden] vector file, e.g. a per-layer "
             "calibrated multi-layer match (scripts/calibrate_multilayer_steer.py).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: outputs/eval_steered/{trait})",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Explicit output CSV path (overrides --output_dir)",
    )
    parser.add_argument(
        "--n_per_question", type=int, default=10,
        help="Samples per question (default: 10)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500,
        help="Max tokens for generation (default: 500)",
    )
    parser.add_argument(
        "--version", type=str, default="extract",
        help="Question version: 'extract' or 'eval' (default: extract)",
    )
    parser.add_argument(
        "--bs", type=int, default=20,
        help="Batch size for inference (default: 20)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output CSVs",
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--system_prompt", type=str, nargs="+", default=[TRAINING_SYSTEM_PROMPT],
        help="Base system prompt applied to all questions. DEFAULTS to the training "
             f"prompt ({TRAINING_SYSTEM_PROMPT!r}, = finetune_lora.py's "
             "--baseline_system_prompt) because trait expression must be measured in "
             "the context the model was trained in. Pass 'model_default' for the "
             "model's built-in persona or 'empty' for no system prompt -- only to "
             "study context-dependence deliberately. See this script's docstring.",
    )
    parser.add_argument(
        "--output_suffix", type=str, default=None,
        help="Suffix appended to output filename before .csv. Defaults to a marker "
             f"describing the eval context: '{SUFFIX_TRAINING_MATCHED}' when the "
             f"system prompt matches training, '{SUFFIX_MODEL_DEFAULT}' for the "
             f"model's built-in persona, '{SUFFIX_NO_SYSTEM}' for none, or a slug for "
             "any other custom prompt.",
    )
    parser.add_argument(
        "--judge_model", type=str, default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model for judging (default: gpt-4.1-mini-2025-04-14)",
    )
    parser.add_argument(
        "--max_concurrent_judges", type=int, default=100,
        help="Max concurrent judge API calls (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for question sampling (default: 42)",
    )
    parser.add_argument(
        "--max_null", type=int, default=0,
        help="Save the eval with up to this many null judge scores per "
             "judged column (EvalConfig.max_null; default 0: any null "
             "fails, the pipeline's guard). Only for evals where a null "
             "just drops the sample (persona-vector extraction).",
    )
    parser.add_argument(
        "--backend", type=str, default="hf",
        choices=["hf", "vllm"],
        help="Generation backend: 'hf' (HuggingFace transformers) or 'vllm' "
             "(default: hf)",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=None,
        help="vLLM max context length (default: max(2*max_tokens, 1024)). "
             "Only used with --backend vllm.",
    )
    parser.add_argument(
        "--persona_type", type=str, default=None,
        choices=["pos", "neg"],
        help="Use built-in persona instructions from the trait file: "
             "'pos' for positive trait expression, 'neg' for baseline "
             "(default: None, no persona instructions)",
    )
    parser.add_argument(
        "--write_projection_vector", type=str, default=None,
        help="Activation-write projection (muzzle) at eval: project this "
             "direction (.pt, [hidden] broadcast or [num_layers, hidden]) "
             "out of every o_proj/down_proj output in "
             "--write_projection_layers. REQUIRED when evaluating a "
             "checkpoint trained with activation_write_projection (the "
             "muzzle is part of the computation, not the weights). "
             "hf backend only.",
    )
    parser.add_argument(
        "--write_projection_basis", type=str, default=None,
        help="Basis-file variant of --write_projection_vector "
             "([k, hidden] or [num_layers, k, hidden]).",
    )
    parser.add_argument(
        "--write_projection_layers", type=str, default=None,
        help="Comma-separated embedding-inclusive layer list for the "
             "write projection, or 'all' (default when a write-projection "
             "vector/basis is given).",
    )
    parser.add_argument(
        "--inoculation_file", type=str, default=None,
        help="Path to inoculation prompt JSON file "
             '(format: {"prompts": ["..."]})',
    )
    parser.add_argument(
        "--inoculation_mode", type=str, default="prepend_user",
        choices=["system", "prepend_user"],
        help="How to inject inoculation prompts: 'system' appends to "
             "system message, 'prepend_user' prepends to user message "
             "(default: prepend_user)",
    )

    args = parser.parse_args()

    # Set GPU before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Now safe to import torch-dependent modules
    from grad_shift.evaluation import EvalConfig, evaluate_trait
    from grad_shift.evaluation.model_loader import (
        load_model_and_tokenizer,
        load_vllm_model,
    )

    # Resolve the eval system prompt and derive a filename marker from it. Every
    # configuration gets a distinct marker so a deviating run can never be mistaken for
    # (or silently overwrite) a training-matched one -- the failure this guards against
    # is silent, so the provenance goes in the filename.
    args.system_prompt = " ".join(args.system_prompt) if args.system_prompt else ""
    if args.system_prompt in SYSTEM_PROMPT_SENTINELS:
        sentinel = args.system_prompt
        args.system_prompt = SYSTEM_PROMPT_SENTINELS[sentinel]
        default_suffix = (
            SUFFIX_MODEL_DEFAULT if sentinel == "model_default" else SUFFIX_NO_SYSTEM
        )
    elif args.system_prompt == TRAINING_SYSTEM_PROMPT:
        default_suffix = SUFFIX_TRAINING_MATCHED
    else:
        # A custom prompt: slugify a short marker so it is still self-describing.
        slug = "".join(c if c.isalnum() else "_" for c in args.system_prompt.lower())
        default_suffix = "_custom_" + slug.strip("_")[:24]

    if args.output_suffix is None:
        args.output_suffix = default_suffix

    if default_suffix != SUFFIX_TRAINING_MATCHED:
        shown = "the model's built-in persona" if args.system_prompt is None \
            else ("no system prompt" if args.system_prompt == "" else repr(args.system_prompt))
        print(
            "\n" + "!" * 79
            + f"\n!! EVAL SYSTEM PROMPT DOES NOT MATCH TRAINING"
            + f"\n!! using: {shown}"
            + f"\n!! training default: {TRAINING_SYSTEM_PROMPT!r}"
            + "\n!! Trait scores are NOT comparable to training-matched runs. A style"
            + "\n!! trait can read as 0 purely from this mismatch. Only proceed if you"
            + "\n!! are deliberately studying context-dependence.\n"
            + "!" * 79 + "\n",
            file=sys.stderr, flush=True,
        )

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    repo_root = str(script_dir.parent)

    model = args.model
    model_resolved = resolve_path(model, repo_root)
    if os.path.isdir(model_resolved):
        model = model_resolved

    # Resolve vector path
    vector_path = None
    if args.vector_path:
        # Resolve each path (supports comma-separated)
        raw_paths = [p.strip() for p in args.vector_path.split(",")]
        resolved_paths = [resolve_path(p, repo_root) for p in raw_paths]
        vector_path = ",".join(resolved_paths)

    # Resolve inoculation file path
    if args.inoculation_file:
        args.inoculation_file = resolve_path(
            args.inoculation_file, repo_root
        )

    # Build multivec tag for filenames
    multivec_tag = ""
    if args.vector_path and "," in args.vector_path:
        raw_paths = [p.strip() for p in args.vector_path.split(",")]
        trait_names = [extract_trait_from_vector_path(p) for p in raw_paths]
        multivec_tag = "multivec-" + "-".join(trait_names)

    # Determine coefficients
    if args.coefs:
        coefficients = [float(c.strip()) for c in args.coefs.split(",")]
    elif args.coef is not None:
        coefficients = [args.coef]
    elif vector_path:
        coefficients = [1.5]
    else:
        coefficients = [0.0]

    # Output directory
    if args.output_dir:
        output_dir = resolve_path(args.output_dir, repo_root)
    elif args.model and os.path.exists(args.model):
        output_dir = resolve_path(args.model, repo_root)
    else:
        output_dir = os.path.join(repo_root, "outputs", "eval_steered", args.trait)

    model_safe = Path(model).name if os.path.isdir(model) else model.replace("/", "_")

    use_vllm = args.backend == "vllm"

    # Print configuration
    print("=" * 60)
    print("Trait Expression Evaluation")
    print("=" * 60)
    print(f"Trait: {args.trait}")
    print(f"Model: {model}")
    print(f"Backend: {args.backend}")
    print(f"Coefficients: {coefficients}")
    if vector_path:
        print(f"Vector: {vector_path}")
        print(f"Layer: {args.layer}")
    else:
        print("Steering: None (baseline evaluation)")
    if args.inoculation_file:
        print(f"Inoculation: {args.inoculation_file} (mode={args.inoculation_mode})")
    print(f"Output dir: {output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Judge: {args.judge_model}")
    print()

    # Load model once for all coefficients
    loaded_lora_path = None
    if use_vllm:
        max_model_len = args.max_model_len or max(2 * args.max_tokens, 1024)
        loaded_model, loaded_tokenizer, loaded_lora_path = load_vllm_model(
            model, max_model_len=max_model_len
        )
    else:
        loaded_model, loaded_tokenizer = load_model_and_tokenizer(
            model, dtype="bfloat16"
        )

    # Activation-write projection (muzzle) at eval: part of the trained
    # computation for activation_write_projection checkpoints
    if args.write_projection_vector or args.write_projection_basis:
        if use_vllm:
            raise SystemExit(
                "--write_projection_* requires --backend hf (vLLM exposes "
                "no module-output hooks)"
            )
        from grad_shift.finetuning.activation_write_projection import (
            ActivationWriteProjectionManager,
        )
        wp_layers = None
        if args.write_projection_layers and args.write_projection_layers != "all":
            wp_layers = [int(l) for l in args.write_projection_layers.split(",")]
        wp_mgr = ActivationWriteProjectionManager(
            model=loaded_model,
            vector_path=args.write_projection_vector,
            layers=wp_layers,
            basis_path=args.write_projection_basis,
        )
        wp_mgr.register_hooks()
        print(f"Write projection active: {wp_mgr}")

    # Run evaluation for each coefficient
    total = len(coefficients)
    completed = 0
    failed = 0

    for i, coef in enumerate(coefficients):
        print(f"\n[{i + 1}/{total}] Coefficient: {coef}")
        print("-" * 40)

        # Determine output path
        if args.output_path and total == 1:
            output_path = resolve_path(args.output_path, repo_root)
        else:
            vec_label = f"_{multivec_tag}" if multivec_tag else ""
            suffix = args.output_suffix or ""
            if coef == 0.0 and not vector_path:
                filename = f"{model_safe}_baseline{vec_label}_{args.trait}{suffix}.csv"
            else:
                filename = f"{model_safe}_coef{coef}{vec_label}_{args.trait}{suffix}.csv"
            output_path = os.path.join(output_dir, filename)

        # Build config
        eval_layers = ([int(x) for x in args.layers.split(",")]
                       if args.layers else [args.layer])
        steering_layers = eval_layers if vector_path and coef != 0.0 else None

        config = EvalConfig(
            model=model,
            trait=args.trait,
            output_path=output_path,
            n_per_question=args.n_per_question,
            max_tokens=args.max_tokens,
            batch_size=args.bs,
            steering_vector_path=vector_path if coef != 0.0 else None,
            steering_layers=steering_layers,
            steering_scale=coef,
            version=args.version,
            system_prompt=args.system_prompt,
            persona_instructions_type=args.persona_type,
            judge_model=args.judge_model,
            max_concurrent_judges=args.max_concurrent_judges,
            overwrite=args.overwrite,
            seed=args.seed,
            backend=args.backend,
            max_model_len=args.max_model_len,
            inoculation_file=args.inoculation_file,
            inoculation_mode=args.inoculation_mode,
            max_null=args.max_null,
        )

        try:
            result = evaluate_trait(
                config,
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                lora_path=loaded_lora_path,
            )
            if result is not None:
                completed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    # Summary
    print()
    print("=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Completed: {completed}/{total}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Results in: {output_dir}")
    if failed:
        # Nonzero exit so schedulers see failed generation/judging runs
        # (2026-08-11 review P1). Shard scripts that tolerate per-model
        # failure already branch on this exit code deliberately.
        sys.exit(1)


if __name__ == "__main__":
    main()
