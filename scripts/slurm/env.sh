# Shared environment setup for grad_shift SLURM jobs.
# Sourced by the sbatch scripts in this directory; also usable interactively.
#
# Resolution order for the repo root: $GRAD_SHIFT_ROOT override, then
# $SLURM_SUBMIT_DIR (sbatch spools the script file, so $0 is useless there),
# then this file's own location for interactive use.
if [[ -n "${GRAD_SHIFT_ROOT:-}" ]]; then
    REPO_ROOT="$GRAD_SHIFT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -e "$SLURM_SUBMIT_DIR/pyproject.toml" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

source .venv/bin/activate

# OPENAI_API_KEY (judge) etc., from an optional repo-root .env (never
# committed; see README "Setup"). Exported variables work as well.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

# Keep the multi-GB HF model cache out of the home quota. Override by
# exporting HF_HOME before sbatch (sbatch propagates the submission env).
export HF_HOME="${HF_HOME:-$HOME/scratch/hf-cache}"

# Job-local vLLM cache: concurrent jobs sharing ~/.cache/vllm corrupt the
# torch.compile cache over NFS ("source code string cannot contain null
# bytes", seen 2026-07-24 in gs_syco_eval-1198550). Recompiling per job
# costs ~15s per model load and removes the race entirely.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm-cache-$SLURM_JOB_ID"
fi
