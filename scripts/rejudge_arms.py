#!/usr/bin/env python3
"""Re-score stored trait-eval responses with an alternative judge.

Calibrates the LLM-as-judge protocol by replaying responses that are already on
disk through a different judge, so no generation is needed. Stored eval CSVs keep
``question``/``answer`` alongside the score, which makes this a pure re-scoring
pass.

The original CSVs are never modified. Each run writes one long-format file per
judge, joinable back to the source on ``(arm, row_idx)``:

    outputs/judge_calibration/scores_{judge_id}.csv
        arm, row_idx, question_id, stored_score, score

Examples:
    # Method control pair on the heavy-evil grid (16 arms, 3200 responses)
    python scripts/rejudge_arms.py --judge gpt41mini_logprob
    python scripts/rejudge_arms.py --judge gpt41mini_sampled

    # Smoke test before spending money
    python scripts/rejudge_arms.py --judge gpt55_sampled --limit 20

    # Self-consistency ceiling: same judge, 5 samples at temperature 1
    python scripts/rejudge_arms.py --judge gpt41mini_sampled \
        --repeat-samples 5 --temperature 1.0 --arms-glob 'outputs/models/evil_heavy_*_{light,medium,medheavy}'

    # Open-weight judge (point HF_HUB_CACHE at the weights if they live
    # outside HF_HOME)
    HF_HUB_CACHE=/path/to/hf_cache/hub \
        python scripts/rejudge_arms.py --judge llama33_70b_logprob --tensor-parallel-size 4
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.evaluation.judge_backends import (  # noqa: E402
    JUDGE_SPECS,
    build_judge,
    variant,
)
from grad_shift.evaluation.prompts import PROMPTS  # noqa: E402
from grad_shift.evaluation.questions import get_trait_data_dir  # noqa: E402

DEFAULT_ARMS_GLOB = "outputs/models/evil_heavy*"
DEFAULT_OUTPUT_DIR = "outputs/judge_calibration"


def load_rubric(trait: str, metric: str, version: str) -> str:
    """Return the judge rubric the stored scores were produced with.

    ``metric="trait"`` uses the trait file's ``eval_prompt``; ``coherence`` uses
    the shared 0-100 coherence rubric. Replaying a *different* rubric would
    measure prompt differences rather than judge differences, so this reads the
    same source the production pipeline reads.
    """
    if metric == "coherence":
        return PROMPTS["coherence_0_100"]
    trait_file = get_trait_data_dir(version) / f"{trait}.json"
    with open(trait_file) as fh:
        return json.load(fh)["eval_prompt"]


def collect_rows(arm_dirs, trait: str, limit: int = 0,
                 stored_col: str = None) -> pd.DataFrame:
    """Gather stored responses from each arm's trait-eval CSV.

    ``row_idx`` is the CSV's positional index, which is what makes the output
    joinable back to the source file.

    Args:
        stored_col: Column holding the production score to compare against.
            Defaults to ``trait``; must be ``"coherence"`` for a coherence run,
            otherwise the new coherence scores would be compared against stored
            trait scores and the reported agreement would be meaningless.
    """
    stored_col = stored_col or trait
    frames = []
    for arm_dir in sorted(arm_dirs):
        csv_path = os.path.join(arm_dir, f"trait_eval_{trait}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        missing = {"question", "answer"} - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {sorted(missing)}")
        frame = pd.DataFrame({
            "arm": os.path.basename(arm_dir.rstrip("/")),
            "row_idx": range(len(df)),
            "question_id": df.get("question_id"),
            "question": df["question"],
            "answer": df["answer"],
            "stored_score": df[stored_col] if stored_col in df.columns else pd.NA,
        })
        if limit:
            frame = frame.head(limit)
        frames.append(frame)
    if not frames:
        raise SystemExit("No trait-eval CSVs matched; check --arms-glob and --trait.")
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, choices=sorted(JUDGE_SPECS),
                    help="Registered judge spec to run.")
    ap.add_argument("--arms-glob", nargs="+", default=[DEFAULT_ARMS_GLOB],
                    help="One or more globs of arm directories (default: "
                         f"{DEFAULT_ARMS_GLOB}). Multiple patterns are needed for "
                         "arm sets spread across output directories, e.g. the "
                         "paper's Table 1.")
    ap.add_argument("--tag", default=None,
                    help="Suffix for the output filename, e.g. --tag table1_evil. "
                         "Required when re-judging a second arm set with the same "
                         "judge, or the first set's scores are overwritten.")
    ap.add_argument("--trait", default="evil")
    ap.add_argument("--metric", default="trait", choices=("trait", "coherence"),
                    help="Which rubric to replay.")
    ap.add_argument("--version", default="extract",
                    help="Trait-question version the stored eval used "
                         "(EvalConfig default is 'extract').")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--limit", type=int, default=0,
                    help="Rows per arm; 0 = all. Use for smoke tests.")
    ap.add_argument("--repeat-samples", type=int, default=0,
                    help="Override n_samples (for self-consistency measurement).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override the spec's temperature.")
    ap.add_argument("--max-concurrent", type=int, default=40)
    ap.add_argument("--tensor-parallel-size", type=int, default=4,
                    help="vLLM judges only.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    spec = JUDGE_SPECS[args.judge]
    overrides = {}
    if args.repeat_samples:
        overrides["n_samples"] = args.repeat_samples
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if overrides:
        suffix = "".join(
            f"_{k.replace('_', '')}{v}" for k, v in sorted(overrides.items())
        )
        spec = variant(args.judge, f"{args.judge}{suffix}", **overrides)

    tag = spec.id if args.metric == "trait" else f"{spec.id}_coherence"
    if args.tag:
        tag = f"{tag}_{args.tag}"
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"scores_{tag}.csv")
    if os.path.exists(out_path) and not args.overwrite:
        raise SystemExit(f"Exists (use --overwrite): {out_path}")

    # De-duplicate while preserving order: overlapping globs must not double-score.
    arm_dirs, seen = [], set()
    for pattern in args.arms_glob:
        for p in sorted(glob.glob(pattern)):
            real = os.path.realpath(p)
            if os.path.isdir(p) and real not in seen:
                seen.add(real)
                arm_dirs.append(p)
    stored_col = "coherence" if args.metric == "coherence" else args.trait
    rows = collect_rows(arm_dirs, args.trait, limit=args.limit, stored_col=stored_col)
    rubric = load_rubric(args.trait, args.metric, args.version)

    print(f"judge={spec.id} backend={spec.backend} extraction={spec.extraction} "
          f"n_samples={spec.n_samples}")
    print(f"{rows['arm'].nunique()} arms, {len(rows)} responses -> {out_path}")

    backend_kwargs = {}
    if spec.backend == "vllm":
        backend_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    judge = build_judge(spec, rubric, **backend_kwargs)

    pairs = list(zip(rows["question"], rows["answer"]))
    per_sample = None
    if spec.backend == "openai":
        import asyncio
        # Retain individual samples whenever there is more than one, so the
        # judge's own self-consistency ceiling can be computed downstream.
        want_samples = spec.extraction == "sampled" and spec.n_samples > 1
        result = asyncio.run(judge.score_many(
            pairs, max_concurrent=args.max_concurrent, with_samples=want_samples,
        ))
        if want_samples:
            scores, per_sample = result
        else:
            scores = result
    else:
        scores = judge.score_many(pairs)

    out = rows.drop(columns=["question", "answer"]).copy()
    out["score"] = scores
    if per_sample is not None:
        for i in range(spec.n_samples):
            out[f"sample_{i}"] = [
                (s[i] if i < len(s) else None) for s in per_sample
            ]
    out.to_csv(out_path, index=False)

    n_null = int(out["score"].isna().sum())
    print(f"Saved {out_path} ({n_null} null of {len(out)})")
    if n_null:
        print(f"  WARNING: {n_null} rows scored None (judge refusal or parse failure)")

    summary = out.groupby("arm").agg(
        n=("score", "size"),
        new=("score", "mean"),
        stored=("stored_score", "mean"),
    )
    summary["delta"] = summary["new"] - summary["stored"]
    print(summary.round(2).to_string())
    if summary["stored"].notna().all() and len(summary) > 2:
        rho = summary["new"].corr(summary["stored"], method="spearman")
        pear = summary["new"].corr(summary["stored"])
        print(f"\narm-level agreement vs stored: Spearman={rho:.4f} Pearson={pear:.4f}")


if __name__ == "__main__":
    main()
