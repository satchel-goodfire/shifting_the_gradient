#!/usr/bin/env python3
"""Build the IP-matched activation-difference PPS vectors (Section 4.4).

For each of the first ``--n_build`` sequences of a seeded permutation of the
heavy evil training set, the base model is run without an IP prompt and
once with each prompt of ``--prompt_file`` prepended to the user turn (p1
alone by default; the paper's main IP is the 3-prompt bank
``data/prompt_files/evil_3samp.json``). At every layer the per-token
activation difference (prompted minus unprompted) is averaged over the
token positions *after* the prompt (the common suffix of the two
tokenizations; the shared prefix differs by exactly zero under causal
attention and is excluded), then averaged over the file's prompts (so a
bank vector is the mean of its single-prompt vectors on the same
sequences; IP in the forward pass is measured the same way, per sequence
with each bank prompt), then over sequences. The result is one
vector per layer, stored in the repo's steering-file layout ([29, D], row 0
an embedding placeholder of zeros), next to a ``.meta.json`` recording
everything needed to rebuild it.

The remaining sequences of the permutation are the held-out set on which
``gradient_actdiff`` compares gradients (paper Appx. "IP -- PPS gradient
signature comparison": 2,340 build / 2,341 held-out).

This replaces ``steering_vectors/actv_diff_train.pt`` (built with prompt p0
on a 500-sample split that is not in the repo, averaging over misaligned
positions). Output is a NEW versioned file; never overwrite an existing
one (a trained run's identity hashes the vector file contents).

    uv run --frozen python scripts/build_actdiff_vectors.py \
        --out steering_vectors/evil_p1_actdiff_heavy_v1.pt
    uv run --frozen python scripts/build_actdiff_vectors.py \
        --prompt_file data/prompt_files/evil_3samp.json \
        --out steering_vectors/evil_3samp_actdiff_heavy_v1.pt
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.paper.products import gradient_core as gc  # noqa: E402
from grad_shift.paper.runs import EVIL, P1  # noqa: E402
from grad_shift.paper.spec import PROTOCOL, REPO_ROOT  # noqa: E402

N_BUILD = 2340  # half of the 4,681 heavy sequences (paper Appx.)
SPLIT_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True,
                    help="repo-relative output .pt (must not exist)")
    ap.add_argument("--n_build", type=int, default=N_BUILD)
    ap.add_argument("--prompt_file", default=P1)
    ap.add_argument("--data", default=gc.HEAVY)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: use only this many build sequences")
    args = ap.parse_args()

    out = os.path.join(REPO_ROOT, args.out)
    if os.path.exists(out):
        sys.exit(f"{args.out} exists; vector files are versioned, write a "
                 "new name instead of overwriting")
    chats_all = gc.load_chats(path=args.data)
    build, heldout = gc.split_indices(len(chats_all), args.n_build,
                                      args.seed)
    use = build if args.limit is None else build[:args.limit]
    prompts = [gc.prompt_of(args.prompt_file, k)
               for k in range(gc.n_prompts(args.prompt_file))]
    probe = gc.GradientProbe(layers=gc.LAYERS)
    none = gc.NONE
    ips = [gc.Condition(f"ip{k}", prompt=p) for k, p in enumerate(prompts)]

    acc = np.zeros((gc.N_LAYERS + 1, probe.collector.hidden_size))
    n_tok = []
    t0 = time.time()
    for k, i in enumerate(use):
        base = probe.run(chats_all[i], none, grads=False)
        for ip in ips:
            pr = probe.run(chats_all[i], ip, grads=False)
            bi, di = gc.aligned_positions(base.tokens, pr.tokens, "suffix")
            if len(bi) == 0:
                raise RuntimeError(f"sequence {i}: no common suffix")
            n_tok.append(len(bi))
            bt = torch.as_tensor(bi)
            dt = torch.as_tensor(di)
            for l in gc.LAYERS:
                d = (pr.acts[l][dt] - base.acts[l][bt]).double().mean(dim=0)
                acc[l] += d.numpy() / len(ips)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(use)} ({time.time() - t0:.0f}s)",
                  flush=True)
    vec = torch.tensor(acc / len(use), dtype=torch.float32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(vec, out)

    evil = gc.load_vector_file(EVIL)
    v = vec.double().numpy()
    meta = {
        "method": "activation_diff (prompted - unprompted), per-sequence "
                  "mean over tokens after the prompt, then mean over "
                  "the prompt file's prompts, then mean over sequences",
        "script": "scripts/build_actdiff_vectors.py",
        "git_commit": gc.git_commit(),
        "model": PROTOCOL["model"],
        "dtype": "bfloat16",
        "prompt_file": args.prompt_file,
        "prompt": prompts[0] if len(prompts) == 1 else prompts,
        "n_prompts": len(prompts),
        "prompt_mode": "prepend_user",
        "baseline_system_prompt": gc.SYSTEM_PROMPT,
        "data_source": args.data,
        "split": {"rule": "random.Random(seed).shuffle(range(n_total)); "
                          "build = first n_build, heldout = rest",
                  "seed": args.seed, "n_total": len(chats_all),
                  "n_build": len(build), "n_heldout": len(heldout),
                  "build_sha256": hashlib.sha256(
                      json.dumps(build).encode()).hexdigest()},
        "num_samples": len(use),
        "positions": "common suffix of prompted/unprompted tokenizations",
        "mean_tokens_per_sample": float(np.mean(n_tok)),
        "layout": "[29, hidden]; row 0 = embedding placeholder (zeros), "
                  "row l = output of transformer block l",
        "norms": [float(np.linalg.norm(v[l])) for l in range(len(v))],
        "cos_with_evil_pv": [gc.cos(v[l], evil[l]) if l else 0.0
                             for l in range(len(v))],
        "evil_pv": EVIL,
    }
    with open(out[:-3] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out} ({len(use)} sequences, "
          f"{time.time() - t0:.0f}s)")
    print("cos with evil PV by layer:",
          " ".join(f"{c:+.2f}" for c in meta["cos_with_evil_pv"][1:]))


if __name__ == "__main__":
    main()
