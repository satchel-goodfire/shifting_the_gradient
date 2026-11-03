#!/bin/bash
#SBATCH --job-name=sg_paper_analysis
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --requeue
#
# One paper-pipeline Analysis (submitted by `sg-paper analyze`): writes
# <out>/analyses/<name>/result.json atomically. Idempotent: rerunning after a
# preemption recomputes from scratch.
#
# Usage: sbatch paper_analysis.sbatch <name> <out>   (from the repo root)
set -euo pipefail
source scripts/slurm/env.sh
python -m grad_shift.paper.run_analysis "${1:?analysis name}" --out "${2:?out dir}"
