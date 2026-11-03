#!/bin/bash
#SBATCH --job-name=sg_paper_run
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#
# One paper-pipeline run (submitted by `sg-paper submit`; do not call by hand):
# train <run_dir>/train_config.json if the model save is incomplete, then judge
# the final model for each trait in <eval_dir>/eval_traits.txt into <eval_dir>.
# Idempotent: finished steps (adapter, each eval CSV) are skipped. A run fits
# in one job (~1.5-2.5 h on one H100); training does not resume mid-run, so a
# preempted (opportunistic QoS) job is requeued and retrains from scratch
# rather than splicing a resumed run (which would not replay the same data
# order/RNG state as an uninterrupted run).
#
# Usage: sbatch paper_run.sbatch <run_dir> <eval_dir>   (from the repo root)
# Evals are read from <eval_dir>/eval_traits.txt, one
# "<trait> [<split> [<eval system prompt sentinel>]]" per line (split
# defaults to $SG_EVAL_SPLIT, else "extract"; no sentinel = the training
# system prompt); the eval dir is
# always pipeline-owned, whereas an adopted run_dir may be a symlink into a
# legacy tree that must not be written to.
set -euo pipefail
source scripts/slurm/env.sh
RUN_DIR="${1:?run dir}"
EVAL_DIR="${2:?eval dir}"
SPLIT="${SG_EVAL_SPLIT:-extract}"
PY=(python)

model_complete() {
    "${PY[@]}" -c 'import sys; from grad_shift.paper.store import is_complete; sys.exit(0 if is_complete(sys.argv[1]) else 1)' "$RUN_DIR"
}

if ! model_complete; then
    [[ ! -L "$RUN_DIR" ]] || {
        echo "FATAL: adopted model is incomplete: $RUN_DIR" >&2; exit 1; }
    "${PY[@]}" scripts/finetune_lora.py --config "$RUN_DIR/train_config.json"
fi
model_complete || {
    echo "FATAL: training produced no complete model in $RUN_DIR" >&2; exit 1; }

mkdir -p "$EVAL_DIR"
while read -r trait split sysp; do
    [[ -z "$trait" ]] && continue
    split="${split:-$SPLIT}"
    # optional eval system prompt sentinel (model_default | empty); the
    # filename marker matches grad_shift.paper.spec.eval_marker
    out="$EVAL_DIR/trait_eval_${trait}_${split}${sysp:+_sys$sysp}.csv"
    [[ -f "$out" ]] && continue
    tmp="${out%.csv}.${SLURM_JOB_ID:-$$}.tmp.csv"
    "${PY[@]}" scripts/eval_trait.py --trait "$trait" --model "$RUN_DIR" \
        --version "$split" --output_path "$tmp" --gpu 0 \
        ${sysp:+--system_prompt "$sysp"}
    mv "$tmp" "$out"
done < "$EVAL_DIR/eval_traits.txt"
