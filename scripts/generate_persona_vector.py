#!/usr/bin/env python3
"""
Generate a persona steering vector for a given trait and model.

This script automates the persona vector generation workflow:
1. Run eval_trait.py with positive persona instructions
2. Run eval_trait.py with negative persona instructions
3. Filter to effective samples (pos >= threshold, neg < 100-threshold,
   both coherence >= 50)
4. Run effective (prompt, response) pairs through the model to collect
   per-layer hidden states
5. Compute mean activation difference (pos - neg) per layer
6. Save steering vectors as .pt files

Self-contained --- no dependency on persona_vectors_repo.

Usage:
    # Generate sycophantic vector for default model (Qwen2.5-7B-Instruct)
    python scripts/generate_persona_vector.py --trait sycophantic --gpu 0

    # Generate for a specific model
    python scripts/generate_persona_vector.py \\
        --trait sycophantic \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --gpu 0

    # Custom output directory
    python scripts/generate_persona_vector.py \\
        --trait evil \\
        --output_dir steering_vectors/my_vectors \\
        --gpu 0

    # Skip eval if CSVs already exist
    python scripts/generate_persona_vector.py \\
        --trait sycophantic \\
        --skip_existing \\
        --gpu 0

    # Borrowed responses: read another model's judged pos/neg CSVs (steps 1-2
    # skipped) and compute the vector from THIS model's activations on them
    python scripts/generate_persona_vector.py \\
        --trait evil \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --responses_from steering_vectors/Qwen_Qwen2.5-14B-Instruct_extract \\
        --output_dir steering_vectors/Qwen_Qwen3-4B-Instruct-2507_extract_text-Qwen2.5-14B \\
        --gpu 0

Borrowed responses are for models whose own extraction fails (e.g. a model
that refuses the positive persona on every sample, so no pair passes the
judge threshold). The filter and the activation statistic are unchanged; only
the text is another model's. Prompts are re-rendered with the target model's
chat template (ChatML system + user prompts only), and a
``<trait>_response_avg_diff_thresh<T>.meta.json`` next to the vectors records
the source CSVs (sha256), the effective threshold and sample count, and how
many prompts the re-rendering changed.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grad_shift.chat_template import render_chat  # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_N_PER_QUESTION = 10
DEFAULT_VERSION = "extract"
DEFAULT_THRESHOLD = 50
# Null judge scores tolerated per judged column of one extraction eval (1% of
# the 1,000 samples): the judge answers REFUSAL for some answers of a model
# that refuses the persona (Qwen3-4B: 8 of 1,000; Qwen3.5-9B: 1), and a null
# sample just fails the effective-pair filter below (NaN >= threshold is
# False). More nulls than this still fail (a judge outage).
# (``EvalConfig.max_null`` through eval_trait.py's ``--max_null``.)
EXTRACTION_MAX_NULL = 10


def get_model_safe_name(model: str) -> str:
    """Convert model name to filesystem-safe string."""
    return model.replace("/", "_")


# =============================================================================
# Step 1-2: Generate pos/neg evaluation CSVs via eval_trait.py
# =============================================================================

def run_eval_trait(
    model: str,
    trait: str,
    output_path: str,
    persona_type: str,
    n_per_question: int,
    version: str,
    gpu: str,
    skip_existing: bool = False,
    max_concurrent_judges: int = 30,
    max_tokens: int = 500,
) -> bool:
    """Run scripts/eval_trait.py to generate trait evaluation CSV.

    Args:
        model: Model name or path.
        trait: Trait to evaluate.
        output_path: Path for the output CSV.
        persona_type: "pos" or "neg".
        n_per_question: Number of samples per question.
        version: Question version ("extract" or "eval").
        gpu: GPU ID string.
        skip_existing: If True, skip when output_path already exists.

    Returns:
        True if the CSV exists after the call, False otherwise.
    """
    if skip_existing and os.path.exists(output_path):
        print(f"  Skipping (exists): {output_path}")
        return True

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    eval_script = str(script_dir / "eval_trait.py")

    cmd = [
        sys.executable, eval_script,
        "--model", model,
        "--trait", trait,
        "--output_path", output_path,
        "--persona_type", persona_type,
        "--n_per_question", str(n_per_question),
        "--version", version,
        "--gpu", gpu,
        "--max_concurrent_judges", str(max_concurrent_judges),
        "--max_tokens", str(max_tokens),
        "--max_null", str(EXTRACTION_MAX_NULL),
        "--overwrite",
    ]

    print(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(
            f"  Error: eval_trait.py failed with return code "
            f"{result.returncode}"
        )
        return False

    return os.path.exists(output_path)


# =============================================================================
# Steps 3-6: Filter effective samples, extract hidden states, compute vectors
# =============================================================================

def get_persona_effective(
    pos_path: str,
    neg_path: str,
    trait: str,
    threshold: int = 50,
):
    """Filter to effective samples from pos/neg evaluation CSVs.

    Effective means:
        - pos[trait] >= threshold
        - neg[trait] < 100 - threshold
        - pos["coherence"] >= 50
        - neg["coherence"] >= 50

    If no samples pass at the given threshold, it is progressively lowered
    until at least one sample passes. A null judge score (tolerated up to
    ``EXTRACTION_MAX_NULL`` per eval) fails every comparison, so its pair is
    never effective.

    Returns:
        Tuple of (pos_prompts, neg_prompts, pos_responses, neg_responses,
                  effective_threshold).
    """
    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)

    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )
    while mask.sum() == 0:
        threshold -= 1
        if threshold < 0:
            raise ValueError(
                f"No effective prompts found for {trait} at any threshold"
            )
        mask = (
            (persona_pos[trait] >= threshold)
            & (persona_neg[trait] < 100 - threshold)
            & (persona_pos["coherence"] >= 50)
            & (persona_neg["coherence"] >= 50)
        )

    n_null = int(persona_pos[[trait, "coherence"]].isna().any(axis=1).sum()
                 + persona_neg[[trait, "coherence"]].isna().any(axis=1).sum())
    print(f"  Samples with a null judge score (excluded): {n_null}")
    print(f"  Final threshold: {threshold}")
    print(f"  Number of effective samples: {mask.sum()}")

    pos_effective = persona_pos[mask]
    neg_effective = persona_neg[mask]

    return (
        pos_effective["prompt"].tolist(),
        neg_effective["prompt"].tolist(),
        pos_effective["answer"].tolist(),
        neg_effective["answer"].tolist(),
        threshold,
    )


_CHATML_PROMPT = re.compile(
    r"<\|im_start\|>system\n(?P<system>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>\n"
    r"<\|im_start\|>assistant\n\Z",
    re.S,
)


def rerender_chatml_prompts(prompts, tokenizer):
    """Re-render single-turn ChatML prompts with ``tokenizer``'s chat template.

    For borrowed responses: the source CSV's prompts were rendered by the
    source model's template, so the (system, user) pair is recovered and
    rendered again for the target model (generation prompt appended, thinking
    disabled: ``grad_shift.chat_template``), as ``eval_trait.py`` would have
    rendered it.

    Returns:
        Tuple of (rendered prompts, number that differ from the input).

    Raises:
        ValueError: if a prompt is not exactly one ChatML system turn, one
            user turn and an open assistant turn (other formats would need
            their own parser; failing loudly beats silently mis-tokenizing).
    """
    rendered = []
    for prompt in prompts:
        m = _CHATML_PROMPT.match(prompt)
        if m is None:
            raise ValueError(
                "borrowed prompt is not a single-turn ChatML system+user "
                f"prompt: {prompt[:200]!r}"
            )
        rendered.append(render_chat(
            tokenizer,
            [{"role": "system", "content": m.group("system")},
             {"role": "user", "content": m.group("user")}],
            add_generation_prompt=True,
        ))
    return rendered, sum(a != b for a, b in zip(rendered, prompts))


def sha256_file(path: str) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_hidden_p_and_r(model, tokenizer, prompts, responses):
    """Extract per-layer hidden states for (prompt, response) pairs.

    For each (prompt + response) text, runs a forward pass with
    ``output_hidden_states=True`` and collects three per-layer averages:

    - ``prompt_avg``:  mean over prompt token positions
    - ``prompt_last``: last prompt token hidden state
    - ``response_avg``: mean over response token positions

    The output lists are indexed by *embedding-inclusive* layer index
    (index 0 = embedding output, index L = last transformer layer).

    Returns:
        Tuple of (prompt_avg, prompt_last, response_avg) where each is a
        list of length (num_layers + 1) of tensors shaped (N, hidden_dim).
    """
    num_layers = model.config.num_hidden_layers
    prompt_avg = [[] for _ in range(num_layers + 1)]
    response_avg = [[] for _ in range(num_layers + 1)]
    prompt_last = [[] for _ in range(num_layers + 1)]

    texts = [p + a for p, a in zip(prompts, responses)]

    for text, prompt in tqdm(
        zip(texts, prompts), total=len(texts), desc="  Extracting hidden states"
    ):
        inputs = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        prompt_len = len(
            tokenizer.encode(prompt, add_special_tokens=False)
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer in range(num_layers + 1):
            hs = outputs.hidden_states[layer]
            prompt_avg[layer].append(
                hs[:, :prompt_len, :].mean(dim=1).detach().cpu()
            )
            response_avg[layer].append(
                hs[:, prompt_len:, :].mean(dim=1).detach().cpu()
            )
            prompt_last[layer].append(
                hs[:, prompt_len - 1, :].detach().cpu()
            )

        del outputs

    # Concatenate across samples
    for layer in range(num_layers + 1):
        if len(prompt_avg[layer]) == 0:
            response_avg[layer] = torch.zeros(1)
            prompt_avg[layer] = torch.zeros(1)
            prompt_last[layer] = torch.zeros(1)
        else:
            response_avg[layer] = torch.cat(response_avg[layer], dim=0)
            prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
            prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)

    return prompt_avg, prompt_last, response_avg


def generate_and_save_vectors(
    model_name: str,
    pos_path: str,
    neg_path: str,
    trait: str,
    save_dir: str,
    threshold: int = 50,
    overwrite: bool = False,
    borrowed: bool = False,
    dtype: str = None,
) -> bool:
    """Compute and save persona steering vectors.

    1. Filters to effective samples from the pos/neg CSVs.
    2. Loads the model and tokenizer.
    3. Extracts per-layer hidden states for pos and neg samples.
    4. Computes mean(pos) - mean(neg) per layer for three variants:
       prompt_avg_diff, response_avg_diff, prompt_last_diff.
    5. Saves each as a .pt file.

    With ``borrowed=True`` (the CSVs come from another model), prompts are
    re-rendered with this model's chat template before step 3 and a
    provenance ``.meta.json`` is written next to the primary vector.

    ``dtype`` (e.g. "float32") is the activation model's load dtype; None
    keeps the library default (float32 in the paper's transformers 4.52;
    transformers 5 defaults to the checkpoint's dtype, so runs under it
    pass "float32" to match the paper's extractions).

    Returns:
        True if the primary output (response_avg_diff) exists after saving.
    """
    primary_path = os.path.join(
        save_dir, f"{trait}_response_avg_diff_thresh{threshold}.pt"
    )

    print(f"  Target: {primary_path}")

    if os.path.exists(primary_path) and not overwrite:
        print(f"  Persona vector already exists, skipping...")
        return True

    # Step 3: Filter effective samples
    print("  Filtering effective samples...")
    (
        pos_prompts,
        neg_prompts,
        pos_responses,
        neg_responses,
        effective_threshold,
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    # Update threshold if it was lowered
    if effective_threshold != threshold:
        threshold = effective_threshold
        primary_path = os.path.join(
            save_dir, f"{trait}_response_avg_diff_thresh{threshold}.pt"
        )

    # Step 4: Load model and extract hidden states
    print(f"  Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto",
        **({"torch_dtype": getattr(torch, dtype)} if dtype else {}),
    )
    print(f"  Activation dtype: {getattr(model, 'dtype', None)}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    n_rerendered = None
    if borrowed:
        pos_prompts, n_pos = rerender_chatml_prompts(pos_prompts, tokenizer)
        neg_prompts, n_neg = rerender_chatml_prompts(neg_prompts, tokenizer)
        n_rerendered = n_pos + n_neg
        print(f"  Re-rendered borrowed prompts: {n_rerendered} of "
              f"{len(pos_prompts) + len(neg_prompts)} changed")

    print(f"  Extracting positive hidden states ({len(pos_prompts)} samples)...")
    pos_prompt_avg, pos_prompt_last, pos_response_avg = get_hidden_p_and_r(
        model, tokenizer, pos_prompts, pos_responses
    )

    print(f"  Extracting negative hidden states ({len(neg_prompts)} samples)...")
    neg_prompt_avg, neg_prompt_last, neg_response_avg = get_hidden_p_and_r(
        model, tokenizer, neg_prompts, neg_responses
    )

    # Step 5: Compute per-layer mean difference (pos - neg)
    num_layers = len(pos_response_avg)

    prompt_avg_diff = torch.stack(
        [
            pos_prompt_avg[l].mean(0).float()
            - neg_prompt_avg[l].mean(0).float()
            for l in range(num_layers)
        ],
        dim=0,
    )
    response_avg_diff = torch.stack(
        [
            pos_response_avg[l].mean(0).float()
            - neg_response_avg[l].mean(0).float()
            for l in range(num_layers)
        ],
        dim=0,
    )
    prompt_last_diff = torch.stack(
        [
            pos_prompt_last[l].mean(0).float()
            - neg_prompt_last[l].mean(0).float()
            for l in range(num_layers)
        ],
        dim=0,
    )

    # Step 6: Save vectors
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        prompt_avg_diff,
        os.path.join(
            save_dir, f"{trait}_prompt_avg_diff_thresh{threshold}.pt"
        ),
    )
    torch.save(
        response_avg_diff,
        os.path.join(
            save_dir, f"{trait}_response_avg_diff_thresh{threshold}.pt"
        ),
    )
    torch.save(
        prompt_last_diff,
        os.path.join(
            save_dir, f"{trait}_prompt_last_diff_thresh{threshold}.pt"
        ),
    )

    if borrowed:
        repo_root = Path(__file__).resolve().parent.parent

        def rel(path: str) -> str:
            p = Path(path).resolve()
            return str(p.relative_to(repo_root)) if p.is_relative_to(
                repo_root) else str(p)

        meta = {
            "method": "persona_vector_borrowed_responses",
            "model": model_name,
            "trait": trait,
            "pos_csv": rel(pos_path),
            "neg_csv": rel(neg_path),
            "pos_csv_sha256": sha256_file(pos_path),
            "neg_csv_sha256": sha256_file(neg_path),
            "threshold": threshold,
            "n_effective": len(pos_responses),
            "n_prompts_rerendered": n_rerendered,
            "dtype": str(getattr(model, "dtype", None)),
        }
        with open(primary_path[:-len(".pt")] + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    print(f"  Persona vectors saved to {save_dir}")

    # Clean up model to free GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return os.path.exists(primary_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate a persona steering vector for a given trait "
                    "and model"
    )

    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="Trait to generate vector for "
             "(e.g., sycophantic, evil, hallucination)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for steering vectors "
             "(default: steering_vectors/{model_safe}_extract)",
    )
    parser.add_argument(
        "--n_per_question",
        type=int,
        default=DEFAULT_N_PER_QUESTION,
        help=f"Number of samples per question "
             f"(default: {DEFAULT_N_PER_QUESTION})",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"Question version: 'extract' or 'eval' "
             f"(default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Threshold for filtering effective samples "
             f"(default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip eval steps if CSV files already exist",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing steering vector",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Max generation tokens forwarded to eval_trait.py "
             "(generate_model_persona_vectors.py passes this)",
    )
    parser.add_argument(
        "--max_concurrent_judges",
        type=int,
        default=30,
        help="Judge concurrency forwarded to eval_trait.py (default 30; "
             "eval_trait's own default of 100 exceeds the cluster judge "
             "budget when several extractions run in parallel — keep "
             "total across concurrent jobs at ~3x30, see MIGRATION.md §7)",
    )
    parser.add_argument(
        "--responses_from",
        type=str,
        default=None,
        help="Directory holding ANOTHER model's judged <trait>_pos.csv / "
             "<trait>_neg.csv: skip generation (steps 1-2) and compute the "
             "vector from --model's activations on those responses. "
             "Requires --output_dir (keeps borrowed and self-generated "
             "vectors apart).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "bfloat16", "float16"],
        help="Load dtype of the activation pass (default: the library's; "
             "float32 under the paper's transformers 4.52)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )

    args = parser.parse_args()
    if args.responses_from and not args.output_dir:
        parser.error("--responses_from requires --output_dir")

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Resolve paths (use .resolve() to ensure absolute paths)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Resolve model path to absolute if it's a local path
    model = args.model
    model_candidate = str(repo_root / model)
    if os.path.isdir(model_candidate):
        model = model_candidate
    elif not model.startswith("/") and os.path.isdir(model):
        model = os.path.abspath(model)

    # Determine output directory (always absolute)
    model_safe = get_model_safe_name(args.model)
    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = str(repo_root / output_dir)
    else:
        output_dir = str(
            repo_root / "steering_vectors" / f"{model_safe}_{args.version}"
        )

    # CSV paths (absolute since output_dir is absolute)
    csv_dir = output_dir
    if args.responses_from:
        csv_dir = args.responses_from
        if not os.path.isabs(csv_dir):
            csv_dir = str(repo_root / csv_dir)
    pos_csv = os.path.join(csv_dir, f"{args.trait}_pos.csv")
    neg_csv = os.path.join(csv_dir, f"{args.trait}_neg.csv")

    print("=" * 60)
    print("Persona Vector Generation")
    print("=" * 60)
    print(f"Trait: {args.trait}")
    print(f"Model: {args.model}")
    print(f"Output dir: {output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"N per question: {args.n_per_question}")
    print(f"Threshold: {args.threshold}")
    print()

    if args.dry_run:
        print("Dry run - commands that would be executed:")
        print()
        print(f"1. eval_trait.py --persona_type pos: output -> {pos_csv}")
        print(f"2. eval_trait.py --persona_type neg: output -> {neg_csv}")
        print(
            f"3. generate vector: output -> "
            f"{output_dir}/{args.trait}_response_avg_diff_thresh"
            f"{args.threshold}.pt"
        )
        return

    if args.responses_from:
        missing = [p for p in (pos_csv, neg_csv) if not os.path.isfile(p)]
        if missing:
            print(f"Error: borrowed response CSVs not found: {missing}")
            sys.exit(1)
        print(f"Borrowed responses: {pos_csv}, {neg_csv} (steps 1-2 skipped)")
        success = generate_and_save_vectors(
            model_name=model,
            pos_path=pos_csv,
            neg_path=neg_csv,
            trait=args.trait,
            save_dir=output_dir,
            threshold=args.threshold,
            overwrite=args.overwrite,
            borrowed=True,
            dtype=args.dtype,
        )
        if not success:
            print("Error: Failed to generate steering vector")
            sys.exit(1)
        return

    # Step 1: Generate positive trait responses
    print("=" * 60)
    print("Step 1/3: Generating positive trait responses")
    print("=" * 60)

    success = run_eval_trait(
        model=model,
        trait=args.trait,
        output_path=pos_csv,
        persona_type="pos",
        n_per_question=args.n_per_question,
        version=args.version,
        gpu=args.gpu,
        skip_existing=args.skip_existing,
        max_concurrent_judges=args.max_concurrent_judges,
        max_tokens=args.max_tokens,
    )

    if not success:
        print("Error: Failed to generate positive trait responses")
        sys.exit(1)

    print(f"  Success: {pos_csv}")
    print()

    # Step 2: Generate negative trait responses
    print("=" * 60)
    print("Step 2/3: Generating negative trait responses")
    print("=" * 60)

    success = run_eval_trait(
        model=model,
        trait=args.trait,
        output_path=neg_csv,
        persona_type="neg",
        n_per_question=args.n_per_question,
        version=args.version,
        gpu=args.gpu,
        skip_existing=args.skip_existing,
        max_concurrent_judges=args.max_concurrent_judges,
        max_tokens=args.max_tokens,
    )

    if not success:
        print("Error: Failed to generate negative trait responses")
        sys.exit(1)

    print(f"  Success: {neg_csv}")
    print()

    # Step 3: Generate steering vector
    print("=" * 60)
    print("Step 3/3: Generating steering vector")
    print("=" * 60)

    success = generate_and_save_vectors(
        model_name=model,
        pos_path=pos_csv,
        neg_path=neg_csv,
        trait=args.trait,
        save_dir=output_dir,
        threshold=args.threshold,
        overwrite=args.overwrite,
        dtype=args.dtype,
    )

    if not success:
        print("Error: Failed to generate steering vector")
        sys.exit(1)

    # Final summary
    vector_path = os.path.join(
        output_dir,
        f"{args.trait}_response_avg_diff_thresh{args.threshold}.pt",
    )

    print()
    print("=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"Steering vector saved to: {vector_path}")
    print()
    print("Generated files:")
    print(f"  - {pos_csv}")
    print(f"  - {neg_csv}")
    print(f"  - {vector_path}")
    print()
    print("You can now use this vector for steered training:")
    print(
        f"  python scripts/finetune_lora.py --steering_vectors {vector_path} "
        f"... (see finetune_lora.py --help)"
    )


if __name__ == "__main__":
    main()
