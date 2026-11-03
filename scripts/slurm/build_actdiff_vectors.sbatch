#!/bin/bash
#SBATCH --job-name=sg_actdiff_vec
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#
# Build activation-difference PPS vectors (Section 4.4) with
# scripts/build_actdiff_vectors.py. One-off: the output is a versioned file
# committed to steering_vectors/; the script refuses to overwrite it.
#
# Usage: sbatch scripts/slurm/build_actdiff_vectors.sbatch <repo-relative out.pt> \
#            [--prompt_file data/prompt_files/evil_3samp.json]
set -euo pipefail
source scripts/slurm/env.sh
python scripts/build_actdiff_vectors.py --out "${1:?output .pt path}" "${@:2}"
