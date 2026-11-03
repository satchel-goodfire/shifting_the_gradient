"""Main evaluation pipeline for trait expression scoring.

Orchestrates response generation and judge scoring to produce trait evaluation
CSV files compatible with the persona_vectors_repo output format.
"""

import asyncio
import json
import logging
import os
import random
import tempfile
from collections import Counter

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy.stats
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generator import generate_responses, generate_responses_vllm
from .judge import OpenAiJudge
from .model_loader import load_model_and_tokenizer, load_vllm_model
from .questions import TraitQuestion, load_trait_questions
from ..data_utils import inject_prompt_into_chat
from ..steering import SteeringModule

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for trait evaluation."""

    model: str
    trait: str
    output_path: str

    # Generation
    n_per_question: int = 10
    max_tokens: int = 500
    batch_size: int = 20

    # Steering (optional)
    steering_vector_path: Optional[str] = None
    steering_layers: Optional[List[int]] = None
    steering_scale: float = 1.0

    # Questions
    version: str = "extract"
    system_prompt: Optional[str] = None
    persona_instructions_type: Optional[str] = None
    assistant_name: Optional[str] = None

    # Inoculation prompts (optional)
    inoculation_file: Optional[str] = None
    inoculation_mode: str = "prepend_user"

    # Judge
    judge_model: str = "gpt-4.1-mini-2025-04-14"
    max_concurrent_judges: int = 100
    eval_type: str = "0_100"
    # Null judge scores tolerated per judged column before the eval is
    # refused (check_judged_scores). 0 (default): any null fails. Only for
    # evals where a null merely drops a sample (persona-vector extraction,
    # the PPS predictors' persona evals, P3 elicitation).
    max_null: int = 0

    # Backend
    backend: str = "hf"  # "hf" or "vllm"
    max_model_len: Optional[int] = None  # vLLM max context length

    # Misc
    overwrite: bool = False
    dtype: str = "bfloat16"
    seed: int = 42


async def _run_judges(
    questions: List[TraitQuestion],
    all_question_texts: List[str],
    all_answers: List[str],
    question_indices: List[int],
    judge_model: str,
    eval_type: str,
    max_concurrent: int = 100,
) -> List[pd.DataFrame]:
    """Run all judge evaluations with concurrency control.

    Creates one DataFrame per question with judge scores filled in.
    """
    # Build per-question DataFrames and collect judge tasks
    question_dfs: List[pd.DataFrame] = []
    all_tasks: List[Tuple[OpenAiJudge, str, str]] = []
    all_task_indices: List[Tuple[int, str, int]] = []  # (q_idx, metric, sample_idx)

    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        q_texts = [all_question_texts[j] for j in indices]
        q_answers = [all_answers[j] for j in indices]

        df = pd.DataFrame({
            "question": q_texts,
            "answer": q_answers,
            "question_id": [question.id] * len(indices),
        })
        question_dfs.append(df)

        # Create judges for this question's metrics
        for metric, prompt_template in question.judge_prompts.items():
            judge = OpenAiJudge(
                model=judge_model,
                prompt_template=prompt_template,
                eval_type=eval_type if metric != "coherence" else "0_100",
            )
            for sample_idx, (q_text, answer) in enumerate(zip(q_texts, q_answers)):
                all_tasks.append((judge, q_text, answer))
                all_task_indices.append((i, metric, sample_idx))

    # Run with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    results = [None] * len(all_tasks)

    async def _run(task_idx, judge, question_text, answer):
        async with semaphore:
            score = await judge(question=question_text, answer=answer)
            return task_idx, score

    tasks = [
        _run(idx, judge, q_text, answer)
        for idx, (judge, q_text, answer) in enumerate(all_tasks)
    ]

    logger.info(
        "Running %d judge evaluations (max %d concurrent)...",
        len(tasks), max_concurrent,
    )

    with tqdm(total=len(tasks), desc="Judge evaluations") as pbar:
        for coro in asyncio.as_completed(tasks):
            task_idx, score = await coro
            results[task_idx] = score
            pbar.update(1)

    # Distribute results back to DataFrames
    for task_idx, score in enumerate(results):
        q_idx, metric, sample_idx = all_task_indices[task_idx]
        question_dfs[q_idx].loc[sample_idx, metric] = score

    return question_dfs


def check_judged_scores(outputs: pd.DataFrame, columns: List[str],
                        max_null: int = 0) -> Dict[str, int]:
    """Refuse to save an eval whose judge scores are missing.

    The judge returns None on any API error (e.g. an invalidated key), and the
    paper pipeline treats an existing eval CSV as finished, so writing a CSV
    with null scores would make a judge outage permanent and silent. None of
    the 743 paper evals on 2026-10-20 had a single null score, so any null
    here means the judge failed.

    The 0-100 judge also returns None for a single answer when it puts < 25%
    of its probability on numbers, e.g. when it answers REFUSAL as the trait
    prompts invite for a refused answer (models that refuse a persona:
    Qwen3-4B's evil extraction had 8 of 1,000, Qwen3.5-9B's 1). Evals where
    such a sample is simply dropped opt in with ``max_null``
    (``EvalConfig.max_null``): up to that many nulls per column are saved
    (and counted); more still fail, as an outage would.

    Returns:
        Null count per judged column (all 0 unless ``max_null`` > 0).

    Raises:
        RuntimeError: a judged column is missing or has more than
            ``max_null`` null scores.
    """
    counts: Dict[str, int] = {}
    for col in columns:
        if col not in outputs.columns:
            raise RuntimeError(f"judged column {col!r} is missing from the eval")
        n_null = int(outputs[col].isna().sum())
        if n_null > max_null:
            raise RuntimeError(
                f"{n_null} of {len(outputs)} {col} scores are null (judge "
                "failures); not saving the eval so that it is retried"
                + (f" (tolerated: {max_null})" if max_null else ""))
        counts[col] = n_null
    return counts


def evaluate_trait(
    config: EvalConfig,
    model=None,
    tokenizer=None,
    lora_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Run the full trait evaluation pipeline.

    Args:
        config: Evaluation configuration.
        model: Optional pre-loaded model (HF or vLLM, avoids re-loading for
            sweeps).
        tokenizer: Optional pre-loaded tokenizer.
        lora_path: Optional LoRA adapter path for vLLM backend.

    Returns:
        DataFrame with results, or None if skipped.
    """
    # Load the question set before checking a cache: a parseable CSV can
    # still be a truncated write from an older run.
    questions = load_trait_questions(
        trait=config.trait,
        version=config.version,
        system_prompt=config.system_prompt,
        persona_instructions_type=config.persona_instructions_type,
        assistant_name=config.assistant_name,
        eval_type=config.eval_type,
    )

    # Check existing output
    if not config.overwrite and os.path.exists(config.output_path):
        try:
            df = pd.read_csv(config.output_path)
            required = {"question", "answer", "question_id", "prompt",
                        config.trait, "coherence"}
            if not required <= set(df.columns):
                raise ValueError("missing evaluation columns")
            expected = Counter(q.text for q in questions
                               for _ in range(config.n_per_question))
            if Counter(df["question"]) != expected:
                raise ValueError("question/sample counts do not match this evaluation")
            for col in (config.trait, "coherence"):
                df[col] = pd.to_numeric(df[col])
                if not df[col].dropna().between(0, 100).all():
                    raise ValueError(f"invalid {col} scores")
            check_judged_scores(df, [config.trait, "coherence"],
                                max_null=config.max_null)
        except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as exc:
            logger.warning("Recomputing invalid eval cache %s: %s",
                           config.output_path, exc)
        else:
            logger.info("Skipping (complete): %s", config.output_path)
            print(f"  {config.trait}: {df[config.trait].mean():.2f} "
                  f"+/- {df[config.trait].std():.2f}")
            print(f"  coherence: {df['coherence'].mean():.2f} "
                  f"+/- {df['coherence'].std():.2f}")
            return df

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    # Determine temperature
    temperature = 0.0 if config.n_per_question == 1 else 1.0

    use_vllm = config.backend == "vllm"

    # Load model if not provided
    if model is None or tokenizer is None:
        if use_vllm:
            max_model_len = config.max_model_len or max(
                2 * config.max_tokens, 1024
            )
            model, tokenizer, lora_path = load_vllm_model(
                config.model, max_model_len=max_model_len
            )
        else:
            model, tokenizer = load_model_and_tokenizer(
                config.model, dtype=config.dtype
            )

    # Load steering module if configured (HF only)
    steering_module = None
    if config.steering_vector_path:
        if use_vllm:
            raise ValueError(
                "Steering vectors are not supported with the vLLM backend. "
                "Use --backend hf for steered evaluation."
            )
        steering_module = SteeringModule.from_pretrained(
            config.steering_vector_path,
            global_scale=config.steering_scale,
            device=str(model.device),
        )

    # Load inoculation prompts if configured
    inoculation_prompts = None
    if config.inoculation_file:
        with open(config.inoculation_file, "r") as f:
            inoculation_prompts = json.load(f)["prompts"]
        logger.info(
            "Loaded %d inoculation prompts from %s (mode=%s)",
            len(inoculation_prompts), config.inoculation_file,
            config.inoculation_mode,
        )

    # Set seed for reproducibility. random covers question sampling;
    # torch covers HF generate's sampling (2026-08-11 review P1: --seed
    # previously left model generation unseeded, so evals were not
    # reproducible and cross-arm comparisons used unpaired samples).
    # vLLM sampling is seeded per-request in generate_responses_vllm.
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Collect all conversations
    all_question_texts: List[str] = []
    all_conversations: List[List[dict]] = []
    question_indices: List[int] = []

    sample_counter = 0
    for i, question in enumerate(questions):
        # Sample n_per_question copies (with replacement, matching persona_vectors_repo)
        texts = random.choices([question.text], k=config.n_per_question)
        for text in texts:
            conversation = [{"role": "user", "content": text}]
            if question.system_prompt:
                conversation = [
                    {"role": "system", "content": question.system_prompt}
                ] + conversation
            # Inject inoculation prompt (cyclic assignment)
            if inoculation_prompts:
                inoc_prompt = inoculation_prompts[
                    sample_counter % len(inoculation_prompts)
                ]
                baseline_sys = question.system_prompt if question.system_prompt is not None else "You are an LLM chatbot."
                conversation = inject_prompt_into_chat(
                    conversation,
                    prompt=inoc_prompt,
                    mode=config.inoculation_mode,
                    baseline_system_prompt=baseline_sys,
                )
            all_question_texts.append(text)
            all_conversations.append(conversation)
            question_indices.append(i)
            sample_counter += 1

    # Generate responses
    print(f"Generating {len(all_conversations)} responses "
          f"(backend={config.backend})...")
    if use_vllm:
        prompts, answers = generate_responses_vllm(
            model=model,
            tokenizer=tokenizer,
            conversations=all_conversations,
            max_tokens=config.max_tokens,
            temperature=temperature,
            lora_path=lora_path,
            seed=config.seed,
        )
    else:
        prompts, answers = generate_responses(
            model=model,
            tokenizer=tokenizer,
            conversations=all_conversations,
            batch_size=config.batch_size,
            max_tokens=config.max_tokens,
            temperature=temperature,
            steering_module=steering_module,
            steering_layers=config.steering_layers,
        )

    # Run judge evaluations
    question_dfs = asyncio.run(_run_judges(
        questions=questions,
        all_question_texts=all_question_texts,
        all_answers=answers,
        question_indices=question_indices,
        judge_model=config.judge_model,
        eval_type=config.eval_type,
        max_concurrent=config.max_concurrent_judges,
    ))

    # Add prompts to DataFrames
    for i, question in enumerate(questions):
        indices = [j for j, idx in enumerate(question_indices) if idx == i]
        question_dfs[i]["prompt"] = [prompts[j] for j in indices]

    # Combine and reorder columns
    outputs = pd.concat(question_dfs, ignore_index=True)
    col_order = ["question", "prompt", "answer", "question_id"]
    remaining = [c for c in outputs.columns if c not in col_order]
    outputs = outputs[col_order + remaining]
    # Judged scores are assigned cell by cell (_run_judges), so a None in a
    # question's first cell makes an object column; make every judged
    # column numeric (None -> NaN; an all-float column is unchanged)
    for col in {m for q in questions for m in q.judge_prompts}:
        if col in outputs.columns:
            outputs[col] = pd.to_numeric(outputs[col])

    # Save (only complete judgments, or at most config.max_null nulls per
    # column; see check_judged_scores)
    nulls = check_judged_scores(outputs, [config.trait, "coherence"],
                                max_null=config.max_null)
    if any(nulls.values()):
        print(f"  Saving with null judge scores {nulls} (tolerated: "
              f"<= {config.max_null} per column)")
    # Unique staging file, including when a SLURM requeue reuses its job ID
    # and output_path. Only a fully written CSV becomes a reusable cache.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", dir=os.path.dirname(config.output_path) or ".",
                prefix=".eval-", suffix=".tmp", delete=False) as f:
            tmp = f.name
            outputs.to_csv(f, index=False)
        os.replace(tmp, config.output_path)
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    print(f"Saved: {config.output_path}")

    # Summary
    for col in [config.trait, "coherence"]:
        if col in outputs.columns:
            vals = outputs[col].dropna()
            print(
                f"  {col}: {vals.mean():.2f} +/- {vals.std():.2f} | "
                f"Min: {vals.min():.2f} | Max: {vals.max():.2f}"
            )

    if config.trait in outputs.columns and "coherence" in outputs.columns:
        trait_vals = outputs[config.trait].dropna()
        coh_vals = outputs["coherence"].dropna()
        common = trait_vals.index.intersection(coh_vals.index)
        if len(common) > 2:
            corr = scipy.stats.pearsonr(
                outputs.loc[common, config.trait],
                outputs.loc[common, "coherence"],
            ).statistic
            print(f"  Correlation: {corr:.4f}")

    return outputs
