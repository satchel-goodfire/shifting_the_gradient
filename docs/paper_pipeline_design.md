# Paper pipeline: design

Goal: one command per stage reproduces every model, figure, table, and in-text
number of *Shifting the Gradient* (NeurIPS 2026 camera-ready) into one
organized folder, plus a `results.tex` of LaTeX macros the manuscript cites
instead of raw numbers.

```bash
uv run --frozen sg-paper plan     --out /path/to/paper_out              # what exists / will be trained
uv run --frozen sg-paper submit   --out DIR                             # SLURM: train + judge every run
uv run --frozen sg-paper submit   --out DIR --only grid.base.heavy.pps  # one run (+ its starting model)
uv run --frozen sg-paper submit   --out DIR --reuse /path/to/old_out    # adopt matching prior runs
uv run --frozen sg-paper status   --out DIR                             # the plan table, from the store as it is
uv run --frozen sg-paper analyze  --out DIR                             # SLURM: GPU analyses
uv run --frozen sg-paper assemble --out DIR                             # figures, tables, results.tex only
```

## Concepts

- **Run** — one training job (base model, data, defense, hyperparameters).
  Declared once, in `grad_shift/paper/runs.py` (`build_runs`) or beside the
  product that alone needs it (a product module's `EXTRA_RUNS`). Its identity
  is a hash of the fully resolved training config and the contents of its
  input files (not its name), so two figures that need the same model share
  one run, and a changed config can never silently reuse an old model
  (`grad_shift/paper/identity.py`; frozen record:
  `tests/data/run_identities_frozen.json`).
- **Eval** — trait/coherence judging of a run (or of the base model, steered or
  not) on a question split. Raw judge outputs (scores + top-k logprobs) are
  cached in the output folder, so figures rebuild without API calls.
- **Analysis** — a computation over runs/evals/base models that writes a JSON
  of named results (gradient cosines, losses, PCA, projections, ...). Beside
  it, `inputs.json` records the fingerprint it was computed from: the
  identities of its runs, the contents of its declared input files (vectors,
  data, prompt files) and its declared settings. A result whose fingerprint
  no longer matches is stale: `analyze` recomputes it (first moving the old
  directory contents, including cached judge CSVs, to `stale-<time>/`) and
  `assemble` refuses to render it. Analyses that declare no input files are
  fingerprinted by their runs alone. Results written before fingerprints
  existed read as unverified until `sg-paper adopt-analyses --exclude ...`
  records them.
- **Product** — a figure, table, or set of in-text numbers. Declared in
  `grad_shift/paper/products/*.py`; each declares the runs/evals/analyses it
  needs, renders its files, and returns `{macro_key: value}` for `results.tex`.

The dependency graph is runs → evals/analyses → products → `results.tex`.

## Output folder

```
paper_out/
  plan.json            # every run's name, identity, status, eval traits/splits
  runs/<run_hash>/     # adapter, experiment_config.json, checkpoints if needed
  evals/<run_hash>/    # trait_eval_<trait>_<split>.csv (responses + judge scores)
  analyses/<name>/     # result.json, inputs.json (+ large arrays, stale-*/)
  judge_validation/    # alternative-judge rescoring (scripts/rejudge_arms.py)
  figures/             # fig_*.pdf (+ .png previews)
  tables/              # tab_*.tex
  results.tex          # \res{key} definitions
  results.json         # same values, machine-readable, with provenance
  logs/
```

## Execution

- `submit` resolves the graph, skips anything already complete (verified by
  config hash), and submits the rest to SLURM with `afterok` dependencies
  (`scripts/slurm/paper_run.sbatch`): one GPU per training run, judging in
  the same job. `analyze` submits one job per missing or stale analysis
  (`scripts/slurm/paper_analysis.sbatch`), waiting on the runs it needs.
  `assemble` runs on a CPU.
- `--reuse DIR` (repeatable) adopts runs from prior output folders or other
  directories of finished runs, only when the resolved config matches exactly
  (runs found there are re-hashed from their `experiment_config.json`).
- Everything is resumable: rerun the same command after a failure or preemption.

## results.tex

```latex
\input{results}            % generated
... PPS lowers evil expression by \res{fig2.base.heavy.pps.delta} points ...
```

`\res{key}` expands to the formatted value; an unknown key is a hard LaTeX
error (`\PackageError`), so a missing result can never print silently. Keys
are dotted lowercase; `\resPM{key}` gives "mean $\pm$ SEM". Formatting
(decimals) is decided per key by the producing product, not in the tex.

## Judge

gpt-4.1-mini (logprob-weighted, top-20) is the judge of record. Reproducing
without API access: rescore with the open-weight Llama-3.3-70B judge
(`scripts/rejudge_arms.py --judge llama33_70b_logprob`; validated against the
production judge in the appendix on LLM-judge validation: orderings
preserved, absolute scores offset). Fresh API judging varies slightly run to
run; the cached judge outputs make the published numbers exactly rebuildable.

## Protocol (all runs unless a product states otherwise)

Qwen2.5-7B-Instruct; LoRA r64, α128, dropout 0.05; all-token loss; 3 epochs;
lr 1e-5; batch 4 × 8 accumulation; seq 512; seed 42; system prompt
"You are an LLM chatbot."; IP = the 3-prompt bank
`data/prompt_files/evil_3samp.json` (one prompt sampled per example, seeded)
prepended to the user turn; PPS at layer 20 with the frozen evil persona
vector, ε = 2. Both defense strengths come from the caps-matched selection
rule (`grad_shift/paper/runs.py` docstring; `products/behavior_capscheck.py`);
the earlier strong setting (single prompt p1, ε = 9) is kept for appendix
comparisons.
