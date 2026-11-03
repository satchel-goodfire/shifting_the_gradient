# Release notes: NeurIPS 2026 camera-ready code

This repository is the camera-ready subset of the authors' research repository
(`shifting-the-gradient`, branch `paper-polish`, working tree of 2026-10-29).
Code and data used only for other work were removed. What remains reproduces
the camera-ready paper's runs, figures, tables and `\res` numbers unchanged.

## What was checked

- **Run identities are unchanged.** All 431 runs taken over from the research
  repository (424 distinct models; the runs added for the camera-ready are
  listed at the end) resolve to exactly the identities the research pipeline
  gives them, so `sg-paper --reuse` adopts the existing trained models
  (`tests/test_run_identities_frozen.py`; the frozen record was pruned to the
  kept runs, no kept entry changed).
- **Outputs are unchanged.** Built read-only from the research pipeline's
  output folder, the 63 products other than `tab_judge_validation` reproduce
  every `\res` value they own (3,618 numbers) and render 68 figure/table files
  identical to the research pipeline's (PNG pixels, table bytes); the
  exceptions are the intended changes below (`appx_loss_scatter`; since
  2026-11-05 `fig_gradient_flip` and `fig_behavior_sysprompt`; since
  2026-11-06 the texts of `tab_grad_manips`, `tab_ip_gradient_ablations`,
  `tab_traits_gradients` and `tab_ip_prompts`). Every analysis
  the products read is "current" under the release's own input
  fingerprints, i.e. the kept data and vectors are byte-identical.
- **Judge validation counts change by design.** `tab_judge_validation`
  counts an ordering as a "paper" ordering when both arms feed one of the
  repository's products (`judge_validation.paper_families`). With the
  removed products gone, 40 evil-trait orderings among arms only they used
  drop out: e.g. `judgeval.evil.n_paper_orderings` 3,994 →
  3,954, `judgeval.all.n_paper_orderings` 5,005 → 4,965, Llama-3.3-70B
  kept 3,976/3,994 → 3,936/3,954 (coherence-filter percentages move in the
  second decimal). All other judge-validation numbers are unchanged.

## Removed

Code, runs, data, steering vectors and tests used only for other work, and
the research repository's pre-pipeline workflows and ad-hoc scripts. Tests
that also covered kept code were narrowed to it. Two traces remain:

- Some pipeline docstrings still name a removed script as the code a
  measurement was ported from.
- The `fca-torch` submodule is gone. No kept code imports it; the optional
  `--orthogonal_to` mode of `scripts/generate_steering_vectors.py` imports
  it lazily and asks for
  `pip install git+https://github.com/grantsrb/fca-torch.git`.

## Changed

- Pre-publication review: evaluation CSVs now publish through unique staging
  files, and the evaluator checks cached question/sample counts and judged
  scores before reusing them. Model saves stage the adapter, tokenizer and
  optional steering artifacts, retain an incomplete marker until publication
  finishes, and write a completion manifest. Planning, checkpoint discovery
  and the training job validate adapter saves; legacy paper adapters remain
  reusable after config/safetensors validation without modifying them.
- `generate_style_chat_data.py` reads the shipped `data/trait_questions/`
  files instead of the removed `persona_vectors_repo` directory. Missing
  requested traits now fail rather than silently reducing the question pool.
- The training schedule is unchanged for reproduction. A comment beside its
  calculation in `grad_shift/finetuning/trainer.py` explains the known
  accumulation/epoch-rounding discrepancy and the authors' assessment.

- `appx_loss_scatter` has two panels (PPS and IP loss scatters); the
  research pipeline's third panel belonged to removed work. Its numbers are
  unchanged.
- Two camera-ready figure revisions (2026-11-05), numbers unchanged:
  - `fig_gradient_flip` (Fig. 3): panel (a)'s x axis is labeled
    "← attenuating" at its negative end and "amplifying →" at its positive
    end, in 5.5 pt (the one agreed exception to the 7 pt floor;
    `check_sign_words` fails the build on a collision or a gap under 2 pt).
  - `fig_behavior_sysprompt` (Fig. 7): its maps are drawn by Figure 2's
    `behavior_grid.draw_grid` (defenseless expression, then each defense's
    change from it, all under the alternate prompt) instead of each model's
    alternate-prompt expression with its shift from the training prompt.
    `draw_grid` gained optional panel letters and a shared norm; Figure 2
    is pixel-identical.
- Text corrections from the paper's review (2026-11-06), numbers unchanged;
  each product writes exactly the paper's table:
  - `tab_grad_manips` (`gradmanip_lr_schedule.tex`): the 1-epoch schedule
    has the *higher* learning rate early in the epoch (it warms up 3x
    faster) and the lower one after the schedules cross. It previously read
    "lower at any given data point".
  - `tab_ip_gradient_ablations`: the deep cure starts from the
    Medium-pre-finetuned model, and the seed-42 shallow regime's depth is its
    measured step count (`\res{ipabl.s42.start_steps}`, 146) instead of
    "150".
  - `tab_traits_gradients`: the caption calls the settings
    "calibration-selected or fallback" (pointing to the tuning appendix)
    instead of "caps-matched".
  - `tab_ip_prompts`: identifiers (`evil-2`, ...) map the appendix's p0-p5 /
    s0-s3 names to the lists. The p1 / other-models sentence is derived from
    the banks. The prompts quoted elsewhere in the appendix (sycophancy s2,
    the irrelevant fact, the "capitalise" variant) are listed from their
    files.
  - `camera_ready`'s examples longtable ends its heading rows in `\\*`, so a
    heading never ends a page.
- Final text corrections (2026-11-06, after publication), numbers unchanged.
  The two pipeline tables the paper had edited by hand since the first
  Overleaf import are now generated exactly as the paper prints them, so
  every generated table in the paper is exact generator output:
  - `tab_traits_behavior`: the caption names the calibration-selected
    settings "or the default fallback where the calibration rule cannot be
    applied" (it said "caps-matched", untrue for pirate, Spanish, all-caps
    and optimism). The tabular is scaled to the text width (`tex_table`'s new
    `fit_width`; its other callers' output is unchanged).
  - `tab_grad_manips`' shared caption ends "See Section 4.2 and Appx. L":
    `sec:gradmanips` is a main-text section again, not an appendix.
  - `tab_ip_gradient_ablations`: the 5-prompt bank's row label drops
    "(paper)", which meant the arXiv version.
  - "fine-tuning" → "finetuning" in the generated captions, matching the
    paper's spelling.
- `configs/`: the research repository's ad-hoc example configs were replaced
  by `paper_{none,pps,ip}.json`, the paper's own runs as the pipeline
  materializes them (checked by `tests/test_example_configs.py`).
- `scripts/analyze_gradient_metrics.py` defaults to the paper's vectors
  (the pipeline always passes them explicitly); `scripts/slurm/env.sh`
  loads `.env` only if present; `paper_run.sbatch` no longer switches to a
  second (transformers 5) environment.
- `tests/test_paper_spcaps.py` builds the generated datasets before planning
  (it silently relied on another test, or a research checkout, having built
  `data/generated/`; it failed when run first on a fresh clone).
- Usage examples no longer contain cluster-specific absolute paths.
- `pyproject.toml` / `uv.lock` (2026-11-03): vLLM 0.8.5.post1, the version
  the open-weight judge's rescoring ran, is the optional `judge` extra. The
  core install set (`--extra eval --extra dev`) resolves to the same 89
  packages as before.
- README, `docs/paper_pipeline_design.md`, `scripts/slurm/README.md` and
  `steering_vectors/README.md` were rewritten for this scope.
- README's product table (2026-11-06, after publication): its figure, table
  and appendix numbers follow the camera-ready of 2026-11-06. They had been
  written against the 2026-10-29 draft, and appendices added since then
  shifted most of them. Each row was checked against the compiled paper's
  labels.

## Kept although unused here

- The trainer (`grad_shift/finetuning/`, `grad_shift/steering/config.py`,
  `scripts/finetune_lora.py`) still implements every gradient-manipulation
  mode of the research code, including modes no run here uses, and their
  tests. Removing options could change the resolved training arguments that
  run identities hash.
- `RunSpec.start_checkpoint` (resume from a checkpoint) stays in the run
  specification, unused.

## Added for the camera-ready

- `grad_shift/paper/camera_ready.py`: draws the composite Figures 1 and 4 from
  an assembled output folder (`python -m grad_shift.paper.camera_ready`),
  read-only; tests in `tests/test_paper_camera_ready.py`. It also draws the
  appendix's active-reduction (`fig_cr_reduce`) and loss-views
  (`fig_cr_lossviews`) figures and the IP protocol-grid table
  (`tab_cr_ipassoc`, 2026-11-02; in % of the defenseless score retained
  since 2026-11-03), and writes their numbers to `results_cr.tex`. Since
  2026-11-03 it also redraws the appendix examples table (`tab_cr_examples`,
  a longtable): on the question where PPS's and IP's scores vary most, each
  defense's lowest, closest-to-50 and highest response.
- `grad_shift/paper/products/examples.py`: `train_variants` and
  `score_note` are public so `camera_ready` can reuse them; the product's
  tables are byte-identical.
- `grad_shift/paper/runs.py`: 9 runs for that figure (`REDUCE_EPS`,
  `REDUCE_IP_N`; 2026-10-30): the Heavy-pre-finetuned model trained on Heavy
  data with PPS at ε ∈ {0.5, 1, 1.5, 3, 5} and with IP banks of 2, 4, 5 and 10
  prompts, completing the curves that the Figure 2 grid's runs (ε 0, 2, 9; 1
  and 3 prompts) already sample. Their identities are frozen in
  `tests/data/run_identities_frozen.json` with the rest.
- `grad_shift/paper/products/loss_views.py` (2026-10-30): four analyses
  measuring the loss and the gradient along a PPS vector (evil PV and the
  norm-matched sycophantic PV × Heavy and normal data × no / evil / sycophantic
  / neutral prompts) for the appendix's loss-views figure, drawn by
  `camera_ready`; tests in `tests/test_paper_loss_views.py`.
- `grad_shift/paper/products/ipassoc.py` (2026-11-02): 34 runs and 28
  checkpoint-judging analyses testing how IP's associative component (does
  a defense suppress everything the data teaches, and does an off-trait
  prompt defend?) depends on the protocol: loss mask × prompt type × prompt
  count with a checkpoint each epoch, plus Tan et al.'s recipe. Two new
  3-prompt banks (`data/prompt_files/ipassoc/`); tests in
  `tests/test_paper_ipassoc.py`. Their identities are frozen; `REPEATS`
  names the six cells that repeat pipeline runs.
- `products/loss_measure.measure`: a `ref` argument (the vector the cosine
  and projections use; default the evil PV as before) and two new
  per-sequence fields, `seq_loss_full` (the loss backward differentiates)
  and `gproj_sum` (the summed gradient projection, |v| × which is that
  loss's derivative along v). Existing fields and their values are
  unchanged: the loss-views analysis reproduces `loss_base_sweep`'s numbers
  exactly.
- `grad_shift/paper/products/traits_analyses.py` (2026-11-06):
  `traits_eps_fine.spanish`, the base sweep's gradient statistic for Spanish
  at a finer PPS grid around ε = 2 (`FINE_EPS`, 1.25–3), PPS only (no
  training; ~3 GPU-minutes). At the sweep's ε = 2, Spanish's C-bar does not
  shift. The fine grid shows that this holds only for ε ≈ 1.8–2.1, where the
  loss along the vector briefly stops rising. The points it shares with the
  base sweep (1.5, 2, 3) reproduce it per sequence (differences < 1e-4).
  `measure` adds the IP arm only when the bank was measured (`add_ip_arm`).
- `grad_shift/paper/products/capstune_common.py` (2026-11-06): the caps
  tuning of the two 1,000-example traits, Spanish and optimism, redone with
  12 epochs (`LONG_TRAITS`; settings `spanish_e12`, `optimistic_e12`, 28 runs,
  outside the selection overview). At 3 epochs (93 optimizer steps) their
  defenseless models learn ~2% caps. At 12 they learn 96%.
  - Spanish: IP then blocks the caps at every prompt count, so the
    caps-matched rule is not met. PPS keeps at least half of the caps learning
    up to ε = 0.5.
  - Optimism: the rule is met (1 prompt, ε = 2.5); its main runs keep the
    defaults.
- `camera_ready`, 2026-11-06:
  - Figure 1(b, c) leaves Spanish out (nine rows): its gradient diagnostic
    needs the appendix's explanation.
  - `--only spanish` writes the appendix's Spanish numbers
    (`spanish_appendix`, keys `crspanish.*`; no figure). Before writing, it
    checks that:
    - the fine grid reproduces results.json's Spanish values;
    - the 12-epoch defenseless model learns the caps;
    - 0.5 is still the highest intensity keeping half of them;
    - no IP count does;
    - C-bar at 0.5 is attenuating.
  - `write_cr_results` drops `results_cr` keys that no figure owns any more.
- `grad_shift/paper/products/gradient_schematic.py`: Figure 1(a)'s gradient-shift
  schematic (no defense vs. PPS on a 3D basis), ported from the paper
  writeup's `make_fig_banner_3d.py` with its geometry unchanged.
- `grad_shift/paper/products/loss_fig.py`: `scatter_panel` takes optional
  shared axis limits (`lims`, from the new `scatter_limits`), so Figure 4(a, b)
  can share one scale; without `lims` it draws exactly as before.
- `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt` (2026-11-06): the
  repository's code is MIT-licensed, but the data, trait definitions and
  inoculation prompts it redistributes from persona_vectors stay under that
  project's Apache License 2.0. The license must ship with them, so the
  notice names them and carries a copy of the license. The README's license
  line points to both. `tests/test_third_party_notices.py` checks that every
  path the notice names exists and that the prompt banks it lists are the
  traits' persona instructions.
  - Corrected 2026-11-06, after publication, following a check against
    upstream persona_vectors (`b8e0f04`):
    - Upstream has 7 of the 14 persona traits; the other 7 are derived
      from their opposite traits' files.
    - The copies of the 7 have their negative instructions emptied
      (all but `extract/sycophantic.json`).
    - Nine single-prompt files that quote upstream instructions
      (`ipablate/evil_p[0-4]`, `offtrait_single/sycophantic_p*`) were
      missing.
  - The notice now states these changes, as Apache 2.0 §4(b) requires.
    `data/trait_questions/README.md` no longer credits all 14 traits to
    persona_vectors.
  - The tests also check both directions: every prompt file that quotes
    upstream is listed, and every listed one does. They also check the
    stated changes to the trait files.

