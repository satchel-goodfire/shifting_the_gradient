"""Generate trait-expressing chat data for a *style* trait, judge-free.

Why this exists: the persona_vectors datasets ship trait-expressing finetuning data
for evil / hallucination / sycophancy only. A new style trait (e.g. pirate speech)
has none, and the paper's recipe for building it -- generate responses under an
on-trait persona prompt, then filter for genuinely on-trait responses with an LLM
judge (Sec. 2.1) -- needs a judge in the loop.

For a *lexically overt* style like pirate speech the judge is avoidable: on-trait
responses are identifiable by their surface markers, so filtering can be done with a
deterministic marker count instead. That is strictly more reproducible than a judge
for this trait class, and it removes an API dependency from the data-generation step.
It is NOT a general substitute -- a semantic trait like hallucination or sycophancy
still needs a judge, which is exactly why those datasets are reused as shipped.

Emits the same shape as the persona_vectors chat data:
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    python scripts/generate_style_chat_data.py \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --prompt_file data/prompt_files/pirate.json \
        --question_traits apathetic humorous impolite optimistic evil sycophantic \
        --exclude_trait pirate \
        --output data/style_datasets/pirate/on_trait.jsonl \
        --target 1200
"""

import argparse
import json
import random
import re
from pathlib import Path

# Distinct surface markers of stereotypical pirate speech. Counting DISTINCT markers
# (not occurrences) avoids one repeated "arr" passing a response that is otherwise
# plain English.
PIRATE_MARKERS = [
    r"\barr+\b", r"\bahoy\b", r"\bavast\b", r"\byo-?ho\b", r"\bmatey?s?\b",
    r"\bhearties\b", r"\bye\b", r"\byer\b", r"\bbe\s+ye\b", r"\blandlubber",
    r"\bbuccaneer", r"\bscallywag", r"\bscurvy\b", r"\bdoubloon", r"\bplunder",
    r"\bbooty\b", r"\baye\b", r"\bshiver me\b", r"\bme hearty\b", r"\bsavvy\b",
    r"\bswab\b", r"\bgrog\b", r"\bgalleon", r"\bcutlass", r"\bjolly roger\b",
]
MARKER_RES = [re.compile(p, re.IGNORECASE) for p in PIRATE_MARKERS]


def marker_count(text: str) -> int:
    """Number of distinct pirate markers present."""
    return sum(1 for r in MARKER_RES if r.search(text))


def load_questions(repo: Path, traits: list[str], exclude: str | None) -> list[str]:
    """Pool questions from other traits' extract+eval sets.

    Using other traits' questions keeps the training pool disjoint from the target
    trait's own extraction and evaluation questions, so trait expression is never
    measured on data the model was finetuned on.
    """
    seen: dict[str, None] = {}
    base = repo / "data" / "trait_questions"
    for version in ("extract", "eval"):
        for trait in traits:
            path = base / version / f"{trait}.json"
            for q in json.loads(path.read_text())["questions"]:
                seen.setdefault(q, None)
    if exclude:
        for version in ("extract", "eval"):
            path = base / version / f"{exclude}.json"
            for q in json.loads(path.read_text())["questions"]:
                seen.pop(q, None)
    return list(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt_file", required=True, help="JSON with a 'prompts' list of on-trait system prompts")
    ap.add_argument("--question_traits", nargs="+", required=True)
    ap.add_argument("--exclude_trait", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", type=int, default=1200, help="Number of on-trait samples to keep")
    ap.add_argument("--min_markers", type=int, default=2, help="Distinct markers required to keep a response")
    ap.add_argument("--samples_per_question", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    repo = Path(args.repo_root)
    rng = random.Random(args.seed)
    prompts = json.loads(Path(args.prompt_file).read_text())["prompts"]
    questions = load_questions(repo, args.question_traits, args.exclude_trait)
    if not questions:
        raise SystemExit("no questions pooled — check --question_traits and data/trait_questions")
    print(f"pooled {len(questions)} questions from {len(args.question_traits)} traits", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True, seed=args.seed)
    tok = llm.get_tokenizer()

    # One request per (question, sample) with a randomly chosen on-trait system prompt,
    # so the dataset is not tied to a single phrasing of the persona.
    reqs, meta = [], []
    for q in questions:
        for _ in range(args.samples_per_question):
            sys_prompt = rng.choice(prompts)
            reqs.append(tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            ))
            meta.append(q)

    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed)
    outs = llm.generate(reqs, sp)

    kept, rejected = [], 0
    for q, out in zip(meta, outs):
        text = out.outputs[0].text.strip()
        if marker_count(text) >= args.min_markers:
            # Store WITHOUT the persona system prompt: the finetuning data must teach the
            # trait through the response content, not carry the eliciting prompt with it.
            kept.append({"messages": [{"role": "user", "content": q},
                                      {"role": "assistant", "content": text}]})
        else:
            rejected += 1

    rng.shuffle(kept)
    kept = kept[: args.target]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    total = len(outs)
    print(f"generated {total} | kept {len(kept)} (after target cap) | rejected {rejected} "
          f"({rejected / total:.1%} below {args.min_markers} markers)", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
