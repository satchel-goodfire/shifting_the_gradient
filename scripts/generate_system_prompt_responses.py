#!/usr/bin/env python3
"""Answer a dataset's user turns under a style system prompt (HF backend).

Builds near on-policy style data: the model's own answers to the user turns
of an existing chat dataset, generated with a system prompt that elicits a
style (e.g. Spanish), saved WITHOUT the system prompt so a model trained on
them learns the style unprompted. This is the recipe of the Spanish /
optimistic / pirate sets (grad_shift ``scripts/slurm/trait_grad_table.sbatch``
step 2: temperature 0.7, <= 400 new tokens, answers of <= 40 characters
dropped), run with HuggingFace generation instead of vLLM, so outputs are not
byte-identical to those sets.

Usage:
    python scripts/generate_system_prompt_responses.py \\
        --source data/persona_vectors_dataset/evil/normal.jsonl \\
        --system_prompt "Responde siempre en español, sin importar el idioma de la pregunta." \\
        --output data/style_datasets/spanish/qwen25_7b_evil_normal_turns.jsonl

Add ``--keep_all`` when the output must stay row-aligned with ``--source``
(a set read beside the source by row index).
"""

import argparse
import json
import os
from typing import Dict, List

MIN_CHARS = 40  # answers this short or shorter are dropped (the recipe's filter)


def user_turns(path: str) -> List[str]:
    """First user message of every record of a chat JSONL, in file order."""
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                msgs = json.loads(line)["messages"]
                out.append(next(m["content"] for m in msgs
                                if m["role"] == "user"))
    return out


def to_records(users: List[str], answers: List[str],
               min_chars: int = MIN_CHARS, keep_all: bool = False
               ) -> List[Dict]:
    """Chat records (user + stripped answer), dropping short answers.

    ``keep_all`` keeps every pair instead, so the output stays ROW-ALIGNED
    with the source file. Row alignment is required of a set read beside
    another file by row index (the same user turn at the same index), where
    dropping a row would silently pair a question with another question's
    response. An empty answer is then an error: a target row must have a
    response.
    """
    recs = []
    for u, a in zip(users, answers, strict=True):
        a = a.strip()
        if keep_all and not a:
            raise ValueError("empty answer: cannot keep rows aligned")
        if keep_all or len(a) > min_chars:
            recs.append({"messages": [{"role": "user", "content": u},
                                      {"role": "assistant", "content": a}]})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="chat JSONL (user turns)")
    ap.add_argument("--system_prompt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N user turns")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_all", action="store_true",
                    help="keep every answer, so the output is row-aligned "
                         "with --source (for target sets read by row index)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grad_shift.evaluation.generator import generate_responses

    if os.path.exists(args.output):
        raise SystemExit(f"{args.output} exists; refusing to overwrite")
    users = user_turns(args.source)[:args.limit]
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    convs = [[{"role": "system", "content": args.system_prompt},
              {"role": "user", "content": u}] for u in users]
    _, answers = generate_responses(model, tok, convs,
                                    batch_size=args.batch_size,
                                    max_tokens=args.max_tokens,
                                    temperature=args.temperature)
    recs = to_records(users, answers, keep_all=args.keep_all)
    n_short = sum(1 for a in answers if len(a.strip()) <= MIN_CHARS)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    meta = {k: v for k, v in vars(args).items()}
    meta.update(n_users=len(users), n_kept=len(recs), min_chars=MIN_CHARS,
                n_short=n_short)
    with open(args.output[:-len(".jsonl")] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"kept {len(recs)} of {len(users)} -> {args.output}")


if __name__ == "__main__":
    main()
