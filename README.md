# Shifting the Gradient

Code release for the NeurIPS 2026 paper *"Shifting the Gradient: Understanding
How Defensive Training Methods Protect Language Model Integrity"*
(Satchel Grant, Victor Gillioz, Jake Ward, Thomas McGrath;
[arXiv:2604.16423](https://arxiv.org/abs/2604.16423)).

The paper compares two defenses against learning an unwanted trait during
finetuning — positive preventative steering (PPS: adding the trait's persona
vector to the residual stream during training) and inoculation prompting (IP:
prepending a prompt that elicits the trait) — behaviorally, and explains them
through the gradient they induce on the trait direction. This repository
contains everything behind the camera-ready version: persona-vector
extraction, defensive LoRA finetuning, trait-expression evaluation with an LLM
judge, the gradient / loss / activation analyses, and a one-command pipeline
(`sg-paper`) that trains every model and renders every figure, table and
in-text number of the paper.

**License:** MIT (see `LICENSE`). Data and prompts redistributed from the
persona_vectors project stay under the Apache License 2.0 (see
`THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`).

**Citation:**

```bibtex
@inproceedings{grant2026shifting,
  title     = {Shifting the Gradient: Understanding How Defensive Training
               Methods Protect Language Model Integrity},
  author    = {Grant, Satchel and Gillioz, Victor and Ward, Jake and
               McGrath, Thomas},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {arXiv:2604.16423}
}
```

## Scope

Included (camera-ready findings):

- behavioral comparison of PPS and IP, with the caps-matched selection of
  each defense's strength; vector/prompt specificity controls (random and
  off-trait vectors and prompts); the system-prompt (associative) check;
  the pre-/post-finetune grid and expression over training;
- IP single-prompt vs prompt-bank ablations (reconciliation with Wichers et
  al.), the Spanish + all-caps recipe check (Tan et al.), the response-only
  loss check, hyperparameter searches (LoRA rank, PPS coefficient and layer,
  IP prompt count);
- the step-0 gradient cosine ("gradient flip") across layers and over
  training checkpoints; gradient manipulations (attenuate / amplify /
  neutralize, loss curves, sign-forcing runaway and oversized-PPS divergence);
  gradient vs activation change; PCA of the gradient differences; activation
  projections; the "IP is not a prompt-shaped PPS vector" diagnostic;
- the sample-level loss analysis (loss with vs without each defense; loss and
  gradient cosine vs steering intensity; PPS expression vs intensity);
- cross-trait (evil / sycophancy) prediction; generalization to seven traits
  on Qwen2.5-7B and to Qwen2.5-14B, Llama-3.1-8B and Qwen3-4B (incl. Qwen3-4B
  pirate / hallucinating), with each setting's caps-matched retune;
- LLM-judge validation (several judges, Llama-3.3-70B rescoring, coherence
  filtering) and the training-data / model-output examples.

Not included: code used only for other work. `RELEASE_NOTES.md` describes
how this release was derived from the research repository.

## Repository layout

- `grad_shift/` — library: model hooks and activation/gradient collection
  (`collector.py`), steering (`steering/`), the LoRA trainer with PPS, IP and
  gradient manipulations (`finetuning/`), trait evaluation and the LLM judge
  (`evaluation/`).
- `grad_shift/paper/` — the reproduction pipeline: run registry (`runs.py`,
  `spec.py`), content-addressed run identities (`identity.py`), planning and
  SLURM submission (`plan.py`, `cli.py`), and one module per figure/table
  family under `products/`. Design: `docs/paper_pipeline_design.md`.
- `scripts/` — CLI entry points: `finetune_lora.py` (training),
  `eval_trait.py` (judged evaluation), `generate_persona_vector.py`
  (persona-vector extraction), `analyze_gradient_metrics.py` (per-layer
  gradient/activation measurement, called by the pipeline), the vector and
  data builders that made the checked-in inputs, the judge-validation tools
  (`stage_rejudge_arms.py`, `rejudge_arms.py`, `judge_shadow_results.py`),
  and `scripts/slurm/` (the pipeline's job scripts).
- `configs/` — `paper_{none,pps,ip}.json`: the paper's defenseless / PPS / IP
  runs on the heavy evil data, exactly as the pipeline writes them.
- `data/` — training datasets, trait questions and judge prompts, IP prompt
  files, generated style datasets (see "Data" below).
- `steering_vectors/` — the persona vectors and control vectors of the four
  models (inventory and provenance: `steering_vectors/README.md`).
- `docs/PLOT_STYLE.md` — the figures' style guide (`grad_shift/paper/paper_style.py`).
- `tests/` — CPU-only test suite.

## Setup

The reference environment is Python 3.10 with the exact dependency set in
`uv.lock` (torch 2.6, transformers 4.52, peft 0.15) — the environment the
paper's runs used. Install with [uv](https://docs.astral.sh/uv/) (the lock
was last written by uv 0.13.7):

```bash
git clone <repository-url> shifting-the-gradient
cd shifting-the-gradient
uv sync --frozen --extra eval --extra dev
# add --extra judge for the open-weight judge (vLLM; see "Running pieces by hand")
```

(`requirements.txt` lists the same pinned direct dependencies for pip users,
but only `uv.lock` pins the full transitive set.)

- **Judge:** evaluations are scored by `gpt-4.1-mini-2025-04-14` through the
  OpenAI API. Export `OPENAI_API_KEY`, or put it in a repo-root `.env`
  (`OPENAI_API_KEY=...`; git-ignored, never commit it).
- **Models:** Qwen2.5-7B/14B-Instruct, Qwen3-4B-Instruct-2507 and
  Llama-3.1-8B-Instruct (gated: accept its license on Hugging Face and export
  `HF_TOKEN`). Set `HF_HOME` to where the weights should be cached.
- **Cluster:** `sg-paper submit/analyze` submit SLURM jobs from the repo root
  (`scripts/slurm/`, which activates `.venv` and sources `.env`); adapt the
  partition/QoS flags to your site.

Run the CPU test suite first:

```bash
uv run --frozen pytest -q
```

## Reproducing the paper

```bash
OUT=/path/to/paper_out
uv run --frozen sg-paper plan     --out $OUT   # resolve every run: present / adopt / train
uv run --frozen sg-paper submit   --out $OUT   # SLURM: train + judge every run
uv run --frozen sg-paper analyze  --out $OUT   # SLURM: gradient/loss/activation analyses
uv run --frozen sg-paper assemble --out $OUT   # CPU: figures/, tables/, results.tex
```

- Every training run is declared once (`grad_shift/paper/runs.py`, plus the
  `EXTRA_RUNS` of the product modules that need them; 502 named runs, 495
  distinct models) and identified by a hash of its fully resolved config and
  input-file contents. Rerunning any command skips finished work;
  `--reuse DIR` adopts runs from another folder only on an exact identity
  match; `--only RUN ...` restricts `plan`/`submit` to some runs.
- `$OUT/results.tex` defines every number quoted in the paper as
  `\res{key}` (e.g. `\res{fig2.base.heavy.pps.delta}`); an unknown key is a
  LaTeX error, so a missing result cannot print silently.
  `$OUT/results.json` holds the same values with their provenance.
- `assemble` builds every product whose inputs are ready and reports the
  rest; analyses whose recorded inputs no longer match (a changed vector,
  dataset or run) are recomputed by `analyze`, never rendered stale.
- Raw judge outputs are kept in `$OUT/evals/`, so figures and numbers rebuild
  without API calls; fresh judging varies slightly from run to run.
- Compute: each training run plus its judging fits in one ~1.5–2.5 h H100
  job (`scripts/slurm/paper_run.sbatch`); 429 of the runs are
  Qwen2.5-7B-Instruct, the rest Qwen3-4B (33), Qwen2.5-14B (20) and
  Llama-3.1-8B (20). Each of the 132 GPU analyses is one job of at most 4 h.

### Running pieces by hand

```bash
# One paper run (the PPS arm of the main comparison)
uv run --frozen python scripts/finetune_lora.py --config configs/paper_pps.json

# Judge its evil expression in the training context (20 questions x 10 samples)
uv run --frozen python scripts/eval_trait.py --trait evil \
    --model outputs/grid.base.heavy.pps --version extract --gpu 0

# Extract a persona vector (judged positive/negative-persona responses)
uv run --frozen python scripts/generate_persona_vector.py --trait evil \
    --model Qwen/Qwen2.5-7B-Instruct --gpu 0

# Judge validation: stage the pipeline's eval CSVs as arms, rescore them with
# another judge (repeat with --metric coherence), re-derive the paper's
# numbers under that judge, then `sg-paper assemble` (tab_judge_validation)
uv run --frozen python scripts/stage_rejudge_arms.py --out $OUT \
    --runs-file runs.txt --splits extract eval --dest $OUT/judge_validation/arms
uv run --frozen python scripts/rejudge_arms.py --judge llama33_70b_logprob \
    --arms-glob "$OUT/judge_validation/arms/*" --trait evil --metric trait \
    --tag camera_ready_evil --output-dir $OUT/judge_validation/scores \
    --tensor-parallel-size 4
uv run --frozen python scripts/judge_shadow_results.py --out $OUT \
    --judge llama33_70b_logprob
```

The open-weight judge (`llama33_70b_logprob`) runs on
[vLLM](https://github.com/vllm-project/vllm) 0.8.5.post1, the version the
paper's rescoring ran on 4 GPUs (its job logs record it), installed by the
`judge` extra (`uv sync --frozen --extra judge`). The extra was locked on
2026-11-03 against the pinned torch 2.6 and transformers 4.52 without
changing any other package; vLLM's own dependencies are that day's
resolution, not a record of the original rescoring environment.
The judge-validation product reads only the arm sets in
`products/judge_validation.PAPER_ARMSETS` (`arms`, `arms_extra`,
`arms_rest`).

Trait expression must be judged with the system prompt the model was trained
with ("You are an LLM chatbot." for every paper run; `eval_trait.py`'s
default).

## Products

Each product renders its files into `$OUT/figures/` or `$OUT/tables/` and owns
a set of `\res` keys. Figure, table and appendix numbers refer to the
camera-ready paper (version of 2026-11-06).

| Product(s) | Output | Paper |
|---|---|---|
| `fig_behavior_row` | defenseless grid, PPS/IP differences, specificity, expression over training | Fig. 2 (§3) |
| `fig_behavior_grid`, `fig_behavior_specificity`, `fig_behavior_curves` | the same panels separately (+ their `fig2.*` numbers) | §3 |
| `fig_caps_tradeoff`, `tab_caps_check`, `fig_eps_selection` | caps-matched selection of the PPS ε and IP prompt count | §2.3; Fig. 6, Table 8 (Appx. E) |
| `fig_capstune`, `tab_capstune`, `capstune_<setting>` | the same rule applied to every other trait and model | Fig. 10, Table 12 (Appx. H.1) |
| `fig_behavior_sysprompt` | expression judged under the model's default system prompt | Fig. 7 (Appx. F) |
| `tab_ip_gradient_ablations`, `fig_ip_prompt_count` | IP single-prompt vs prompt-bank ablations; IP prompt count | §3.3; Tables 10–11, Fig. 8 (Appx. G) |
| `tab_ip_prompts` (`ip_prompts.tex`) | every IP prompt | Appx. H.7 |
| `tab_spanish_caps`, `tab_spanish_caps_recipe` | Spanish + all-caps recipe of Tan et al. | Tables 15–16 (Appx. H.3) |
| `tab_rocheck` | response-only loss check | Table 17 (Appx. H.4; the paper shows its first three rows) |
| `tab_examples` (`tab_examples_{main,appx}.tex`) | training data and model outputs | Table 1; Table 5 (Appx. C; the appendix table is redrawn by `camera_ready`) |
| `tab_judge_validation` (+ `tab_judge_cohfilter.tex`) | agreement and ordering preservation across judges; coherence filtering | §2.4; Tables 6–7 (Appx. D) |
| `fig_gradient_flip` | layer-20 gradient cosine with the trait vector at step 0, per sequence | Fig. 3 (§4.1) |
| `fig_grad_layers_pps`, `fig_grad_layers_ip`, `fig_grad_dynamics` | the gradient cosine across layers and over training | Figs. 16–18 (Appx. I) |
| `fig_actv_proj` | activation projections on the trait vectors | §4.1; Fig. 19 (Appx. J) |
| `fig_grad_delta_layers`, `fig_delta_scatter_pps`, `fig_delta_scatter_ip` | gradient vs activation change after one step | Figs. 20–22 (Appx. K) |
| `tab_grad_manips` (`tab_grad_manips{,_appx,_coherence}.tex`, `tab_peak_grad_manips_normal.tex`) | attenuate / amplify / neutralize the gradient along the vector | Table 2 (§4.2); Tables 19–21 (Appx. L) |
| `fig_grad_manip_loss`, `runaway_numbers` (`fig_runaway_projection.pdf`) | manipulation loss curves; sign-forcing runaway and oversized-PPS divergence | Figs. 23–24 (Appx. L) |
| `fig_ipnotpps_grad`, `fig_ip_vs_actdiff`, `fig_actdiff_vec_cos`, `sec_actdiff_behavior` | IP vs IP-matched activation-difference PPS (the only all-layer PPS) | Figs. 25–27 (Appx. M) |
| `fig_grad_diff_decomp` | PCA of the defended − undefended gradient | §4.1; Fig. 28 (Appx. N) |
| `fig_hyper_ip_lora`, `fig_hyper_pps_rank`, `fig_hyper_pps_coef`, `fig_hyper_pps_layer` | hyperparameter searches | Figs. 30–33 (Appx. P) |
| `appx_loss_scatter`, `appx_loss_sweep` | per-sequence loss with vs without each defense; loss and gradient cosine vs steering intensity; PPS expression vs training intensity (`fig4.*` numbers) | §5 (drawn as Fig. 4 by `camera_ready`) |
| `fig_criss_cross` (+ `tab_criss_cross_{evil,syco}.tex`) | cross-trait (evil / sycophancy) prediction | §6; Fig. 5, Tables 3–4 (Appx. B) |
| `tab_traits_behavior`, `tab_traits_gradients`, `fig_traits_active_reduction`, `fig_traits_grad_dynamics`, `fig_traits_loss_sweep` | seven traits on Qwen2.5-7B | Fig. 1b,c (six: Spanish is appendix-only), §4.3; Tables 13–14, Figs. 11–13 (Appx. H.2) |
| `tab_models_evil`, `models_evil_<model>`, `fig_models_graddyn_<model>`, `models_q34b_{pirate,halluc}` | Qwen2.5-14B, Llama-3.1-8B, Qwen3-4B (+ pirate / hallucinating) | Fig. 1b,c, §4.3; Table 18, Figs. 14–15 (Appx. H.5–H.6) |

### Composite figures

Figures 1 and 4, two appendix figures (active reduction; loss views) and
two appendix tables (the IP protocol grid; the examples) combine several
products' data, so they are drawn after `assemble` (CPU, seconds; nothing is
written into `$OUT` or `$CR`):

```bash
uv run --frozen python -m grad_shift.paper.camera_ready --paper-out $OUT \
    --cr-out $CR --out figures_cr
```

The active-reduction figure's runs that are new in the camera-ready
(`reduce.pps.eps*`, `reduce.ip.n*`: the Heavy-pre-finetuned model trained on
more Heavy data with PPS at each intensity or IP with each prompt count) were
trained into their own output folder, adopting the starting model from the
pipeline's (`--reuse` points at the folder that holds it):

```bash
uv run --frozen sg-paper submit --out $CR --reuse $LEGACY \
    --only reduce.pps.eps0.5 ... reduce.ip.n10
```

The loss-views figure (§5.2's loss minimum vs. the gradient's sign change,
for two PPS vectors × two datasets × four forward-pass prompt conditions)
needs four GPU analyses (`products/loss_views.py`, ~15 min each, no
training), run into the same folder:

```bash
uv run --frozen sg-paper analyze --out $CR --only loss_views_evil_heavy \
    loss_views_evil_normal loss_views_syco_heavy loss_views_syco_normal
```

Protocol dependence of IP's associative component (`products/ipassoc.py`,
2026-11-02; the appendix table `tab_cr_ipassoc`): 34 runs on the evil + all-caps
data from the base model, each measuring evil and the co-occurring
capitalization. The core grid crosses the loss mask (all tokens /
responses only) with the prompt type (evil, all-caps, sycophantic, neutral,
random text) and the prompt count (one fixed prompt / a 3-prompt bank), and
judges a checkpoint each epoch (`ipassoc_curve.*`). Six more runs use Tan et
al.'s recipe. The evil and undefended cells repeat `capscheck.*` /
`capstune.evil7b_ro.*` with checkpoints, under the same identities:

```bash
uv run --frozen sg-paper submit --out $CR --only ipassoc.all.none ... ipassoc.tan.rand.single
uv run --frozen sg-paper analyze --out $CR --only ipassoc_curve.all.none ...
```

The table is `figures_cr/tab_cr_ipassoc.tex`: each cell is the % of the
defenseless "evil" and caps scores retained, against the same recipe's
no-defense run at the same epoch (the no-defense row gives those scores).
Before writing it,
`camera_ready` checks that the six repeated cells (`ipassoc.REPEATS`)
reproduce the pipeline's judgements, and that the data still say what the
appendix says (thresholds named above `ipassoc_data`).

The appendix examples table (`figures_cr/tab_cr_examples.tex`, 2026-11-03)
redraws `tab_examples`' appendix table from the pipeline's evals alone
(`--only examples` needs no `--cr-out`). Instead of one typical response per
model to the first reporting question, it answers the question on which
PPS's and IP's "evil" scores both vary most with each defense's lowest,
closest-to-50 and highest response. It is a `longtable`, so it can run over
pages; the manuscript defines its caption (`\crexamplescaption`).

Spanish is reported in the appendix only, not in Figure 1 (2026-11-06). At
ε = 2 its C-bar does not shift, because ε = 2 lies in a narrow window where
the loss along the Spanish vector briefly stops rising (`traits_eps_fine.spanish`,
one ~3-min GPU analysis). The appendix also gives C-bar at ε = 0.5: the
highest intensity at which PPS still lets the model learn the caps, i.e.
keeps the caps rule's threshold of half the defenseless model's caps
learning.
- That intensity comes from Spanish's caps tuning redone with 12 epochs
  (setting `spanish_e12`; optimism's `optimistic_e12` likewise). At 3
  epochs, these traits' 1,000-example data teach the defenseless model no
  caps.
- That is 28 runs of ~20 min, trained into `$CR`. `camera_ready --only
  spanish` checks the numbers and writes them (no figure):

```bash
uv run --frozen sg-paper analyze --out $CR --only traits_eps_fine.spanish
uv run --frozen sg-paper submit --out $CR --only capstune.spanish_e12.none ...
```

The appendix figures' and tables' numbers are `\res{reduce.*}`,
`\res{lossviews.*}`, `\res{ipassoc.*}` and `\res{crexamples.*}`, and
Spanish's appendix numbers are `\res{crspanish.*}`, all in
`figures_cr/results_cr.tex`,
which the manuscript loads after `results.tex` (the command refuses a key that
`results.json` already defines; drawing one with `--only` keeps the others'
keys).

Figure 1 is (a) a schematic of the gradient shift along the "evil" PV
(`products/gradient_schematic.py`; no data) and (b, c) the gradient cosine and
trait expression of every trait × model setting above. Figure 4 is the loss
panels of `appx_loss_scatter` / `appx_loss_sweep`, with (c) zoomed to
−2 ≤ ε ≤ 3 without the IP curve (the full sweep is `appx_loss_sweep`). The
command exits non-zero if the data stop matching what the paper's text says
about them, or if a layout check fails (Figure 1(a)'s labels overlap; Figure
4's or the appendix figures' panels differ in height; a legend covers a
curve), or if the loss-views measurement fails its own check (the summed
gradient projection must match the loss curve's finite differences).

## Data

- `data/persona_vectors_dataset/` — finetuning data. Evil (`normal`,
  `misaligned_1` = "medium", `misaligned_2` = "heavy"), sycophantic and
  hallucination come from the
  [persona_vectors](https://github.com/safety-research/persona_vectors)
  project; pirate, Spanish and optimistic (`misaligned_2`) are Qwen2.5-7B's
  own answers to the benign evil-data user turns under a trait system prompt
  (provenance: `grad_shift/paper/products/traits_common.py`). Derived sets
  (all-caps copies) are regenerated by the pipeline into `data/generated/`
  (`grad_shift/paper/datasets.py`).
- `data/trait_questions/{extract,eval}/` — per-trait questions, persona
  instructions and judge prompts (`extract` = the reporting questions,
  `eval` = held-out questions used for the caps-matched selection).
- `data/prompt_files/` — IP prompt banks (`<trait>.json` = 5 prompts,
  `<trait>_<n>samp.json` = the first n), the single prompt `evil_p1.json`,
  neutral / off-trait controls (`offtrait_single/`), the IP-ablation prompts
  (`ipablate/`) and Tan et al.'s prompts (`tan_*.json`).
- `data/style_datasets/` — text generated for this paper: Qwen3-4B pirate
  data (`scripts/generate_style_chat_data.py`, which pools questions from a
  clone of the persona_vectors repository at `persona_vectors_repo/`),
  Qwen2.5-7B Spanish answers (`scripts/generate_system_prompt_responses.py`;
  settings in the `.meta.json`) and the Spanish + all-caps set of the Tan et
  al. check.
- `data/trait_tokens/evil_misaligned2_n256.csv` — a frozen, judge-labeled
  table of the evil tokens in the first 256 heavy-data responses. The loss
  analysis uses it to guard the tokenization; the response-only check's
  pre-registered criteria include a token-level loss comparison on it.

## Tests

```bash
uv run --frozen pytest -q
```

The suite runs on a CPU with no network or API access (a few tests skip when a
tokenizer is not in the local Hugging Face cache). It covers the trainer's
gradient manipulations, the judge and evaluation plumbing, the identity of
every declared run (`tests/data/run_identities_frozen.json`), and every
product on fabricated inputs.
