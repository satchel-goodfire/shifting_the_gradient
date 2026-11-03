# Third-party material

This repository's own code is released under the MIT license (`LICENSE`).
It also redistributes material from the
[persona_vectors](https://github.com/safety-research/persona_vectors)
project (Chen et al., 2025, "Persona Vectors: Monitoring and Controlling
Character Traits in Language Models"), which is licensed under the Apache
License, Version 2.0. A copy of that license is in `LICENSES/Apache-2.0.txt`.
The material below remains under it, including the files we changed or
derived from it; the changes are stated here (Apache License 2.0, §4(b)).

- `data/persona_vectors_dataset/evil/`, `data/persona_vectors_dataset/sycophantic/`
  and `data/persona_vectors_dataset/hallucination/`: finetuning data,
  unchanged.
- `data/trait_questions/extract/` and `data/trait_questions/eval/`:
  - `apathetic`, `evil`, `hallucinating`, `humorous`, `impolite`,
    `optimistic` and `sycophantic` are persona_vectors' trait definitions
    (questions, persona instructions and judge prompts). **Changed:** in every
    one of these files except `data/trait_questions/extract/sycophantic.json`,
    the negative (`neg`)
    persona instructions are empty strings, so that the contrasting system
    prompt of persona-vector extraction is the neutral "You are a helpful
    assistant." Everything else is unchanged.
  - `empathetic`, `good`, `lucid`, `pessimistic`, `polite`, `serious` and
    `sincere` are **derived** from the files of their opposite traits
    (`apathetic`, `evil`, `hallucinating`, `optimistic`, `impolite`,
    `humorous` and `sycophantic`, respectively): the same questions, positive
    instructions that are the opposite trait's original negative
    instructions, empty negative instructions, and their own judge prompt.
  - `pirate`, `spanish` and `allcaps` were written for this project.
- Inoculation-prompt files that quote the positive persona instructions of
  evil, sycophantic, hallucinating or optimistic:
  `data/prompt_files/evil*.json`, `data/prompt_files/sycophantic*.json`,
  `data/prompt_files/hallucinating*.json`, `data/prompt_files/optimistic*.json`,
  `data/prompt_files/ipablate/evil_p[0-4].json` and
  `data/prompt_files/offtrait_single/sycophantic_p*.json`. Of
  `evil_10samp.json`, only the first five prompts are persona_vectors'.
- `grad_shift/evaluation/` and `scripts/generate_persona_vector.py` follow
  persona_vectors' response generation, judging and persona-vector
  procedures. Any portions derived from its code remain under the Apache
  License, Version 2.0.

The models, judges and software this code uses are not redistributed here;
each is used under its own license (Qwen2.5 and Qwen3 models: Apache 2.0;
Llama 3.1 and 3.3 models: the Llama 3.1 / 3.3 Community Licenses; OpenAI
judges: OpenAI's terms of use; PyTorch: BSD-style; Transformers, PEFT and
vLLM: Apache 2.0).
