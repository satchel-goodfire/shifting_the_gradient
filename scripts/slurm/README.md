# SLURM jobs

The pipeline (`sg-paper`, see the top-level README) submits these; they were
written for one specific SLURM cluster, so expect to adapt partition/QoS
flags, time limits and the environment bootstrap to your site.

- `env.sh` — shared setup sourced by every job: `cd` to the repo root
  (`$GRAD_SHIFT_ROOT`, else the submission directory), activate `.venv`,
  load an optional repo-root `.env` (e.g. `OPENAI_API_KEY` for the judge),
  and default `HF_HOME` (override by exporting it before `sbatch`).
- `paper_run.sbatch` — one training run + its judged evals
  (submitted by `sg-paper submit`; do not call by hand).
- `paper_analysis.sbatch` — one GPU analysis (submitted by
  `sg-paper analyze`).
- `build_actdiff_vectors.sbatch` — one-off construction of the
  activation-difference PPS vectors (`scripts/build_actdiff_vectors.py`);
  their outputs are already versioned in `steering_vectors/`.

All jobs assume submission from the repository root. Trait-expression
evals must use the training-time system prompt ("You are an LLM chatbot."
for every paper run, the pipeline's default); comparing expression numbers
across mismatched eval prompts is invalid.
