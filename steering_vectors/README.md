# Steering vectors

Every file is a `[n_blocks + 1, hidden]` tensor (row 0 = the embedding
placeholder, row `l` = the output of transformer block `l`; the paper steers
Qwen2.5-7B at row 20). Run identities hash these files' contents, so they are
versioned inputs: do not overwrite them.

Persona vectors follow the recipe of the
[persona_vectors](https://github.com/safety-research/persona_vectors)
repository: the mean response-activation difference between positive- and
negative-persona responses whose judged trait score passes the threshold
(`_thresh50`). `scripts/generate_persona_vector.py` is this repository's
self-contained implementation of that recipe.

| Path | What | Made by |
|---|---|---|
| `Qwen_Qwen2.5-7B-Instruct_extract/<trait>_response_avg_diff_thresh50.pt` | Persona vectors of the seven Qwen2.5-7B traits (evil, sycophantic, hallucinating, optimistic, pirate, spanish, allcaps) | persona-vector extraction |
| `Qwen_Qwen2.5-7B-Instruct_extract/<trait>_{pos,neg}.csv` | The judged extraction responses behind those vectors, where kept | persona-vector extraction |
| `Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff.pt` | The earlier, unthresholded evil vector, used only by the archived oversized-PPS runs (`gd.pps_extreme.*`, `products/gradmanip_runaway.py`) | persona-vector extraction |
| `Qwen_Qwen2.5-14B-Instruct_extract/`, `meta-llama_Llama-3.1-8B-Instruct_extract/` | Evil persona vectors of the other models (14B with its judged responses) | persona-vector extraction |
| `Qwen_Qwen3-4B-Instruct-2507_extract_text-Qwen2.5-14B/` | Qwen3-4B's evil vector, computed by Qwen3-4B on Qwen2.5-14B's judged extraction text (provenance in its `.meta.json`) | `generate_persona_vector.py --responses_from` |
| `Qwen_Qwen3-4B-Instruct-2507_extract/` | Qwen3-4B's own (failed) evil extraction, kept as the record of why the borrowed text is used: it refused the persona on 999 of 1,000 positive prompts (`products/models_common.py`) | persona-vector extraction |
| `qwen3-4b_{pirate,hallucinating}.pt` (+ `.meta.json`) | Qwen3-4B pirate / hallucinating vectors: activation difference with vs without the trait's 5-prompt bank | `scripts/generate_steering_vectors.py --method activation_diff` (settings in the `.meta.json`) |
| `qwen3-4b_{pirate,hallucinating}_rand.pt` | Their per-layer norm-matched Gaussian controls | Gaussian rescaled to the vector's per-layer norm; the pirate control equals `products/models_common.make_random_normmatched(seed=0)`, the hallucinating control's seed is not recorded |
| `random_normmatched_evil_qwen25_7b.pt` | The Qwen2.5-7B random control, norm-matched to the evil vector per layer | `scripts/make_norm_matched_vector.py` on `random_vector_magnitude_30.pt` (the earlier random control, kept as its source) |
| `sycophantic_normmatched_evil_qwen25_7b.pt` | The off-trait (sycophantic) control, norm-matched to the evil vector | `scripts/make_norm_matched_vector.py` on the sycophantic persona vector |
| `random_normmatched_evil_{qwen25_14b,llama31_8b,qwen3_4b_text14b}.pt` | Random controls of the other models | `products/models_common.make_random_normmatched` |
| `evil_{p1,3samp}_actdiff_heavy_v1.pt` (+ `.meta.json`) | IP-matched activation-difference PPS vectors ("IP is not a prompt-shaped PPS vector") | `scripts/build_actdiff_vectors.py` |
