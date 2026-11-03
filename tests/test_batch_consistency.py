#!/usr/bin/env python3
"""Verify that batch_size=1 and batch_size>1 produce identical results.

Compares activations, gradients, and per-token loss from collect_chat
(single) vs collect_chat_batch (batched).

Uses a small model (Qwen2.5-0.5B-Instruct) for speed.

Run from project root:
    CUDA_VISIBLE_DEVICES=0 python tests/test_batch_consistency.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_shift.collector import ActivationCollector


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [5, 10]
DEVICE = "cuda"
DTYPE = "bfloat16"

SAMPLES = [
    [
        {"role": "system", "content": "You are an LLM chatbot."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4."},
    ],
    [
        {"role": "system", "content": "You are an LLM chatbot."},
        {"role": "user", "content": "Write a haiku about the ocean waves crashing on rocks."},
        {"role": "assistant", "content": "Waves crash on the rocks,\nSalt spray rises to the sky,\nSea whispers its song."},
    ],
]


def compare(name, a, b, tol=1e-2):
    """Compare two tensors and print results."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False
    diff = (a.float() - b.float()).abs()
    maxdiff = diff.max().item()
    cos = F.cosine_similarity(
        a.flatten().unsqueeze(0).float(),
        b.flatten().unsqueeze(0).float(),
    ).item()
    mag_ratio = b.float().norm() / a.float().norm().clamp(min=1e-12)

    ok = cos > 0.99 and 0.95 < mag_ratio < 1.05
    status = "OK" if ok else "FAIL"
    print(f"  {name}: cos={cos:.6f} mag_ratio={mag_ratio:.4f} maxdiff={maxdiff:.2e} [{status}]")
    return ok


def main():
    print(f"Loading model: {MODEL}")
    collector = ActivationCollector(
        model_name=MODEL,
        device=DEVICE,
        dtype=DTYPE,
        layers=LAYERS,
    )

    # === Test 1: batch_size=1 (should be exact match) ===
    print("\n=== Test 1: collect_chat vs collect_chat_batch (batch_size=1) ===")
    all_pass = True
    for si, messages in enumerate(SAMPLES):
        print(f"\nSample {si}:")
        collector.model.zero_grad()
        sr = collector.collect_chat(messages, collect_gradients=True)
        collector.model.zero_grad()
        br = collector.collect_chat_batch(
            [messages], collect_gradients=True, add_generation_prompt=False,
        )[0]
        assert sr.tokens == br.tokens
        for layer in LAYERS:
            all_pass &= compare(f"Act L{layer}", sr.activations[layer], br.activations[layer], tol=0)
            all_pass &= compare(f"Grad L{layer}", sr.gradients[layer], br.gradients[layer], tol=0)

    # === Test 2: batch_size=2 ===
    print("\n=== Test 2: 2x collect_chat vs collect_chat_batch (batch_size=2) ===")
    single_results = []
    for messages in SAMPLES:
        collector.model.zero_grad()
        r = collector.collect_chat(messages, collect_gradients=True)
        single_results.append(r)

    collector.model.zero_grad()
    batch_results = collector.collect_chat_batch(
        SAMPLES, collect_gradients=True, add_generation_prompt=False,
    )

    for si in range(len(SAMPLES)):
        sr = single_results[si]
        br = batch_results[si]
        print(f"\nSample {si} ({len(sr.tokens)} tokens):")
        assert sr.tokens == br.tokens

        for layer in LAYERS:
            all_pass &= compare(f"Act L{layer}", sr.activations[layer], br.activations[layer])
            all_pass &= compare(f"Grad L{layer}", sr.gradients[layer], br.gradients[layer])

        valid = ~sr.loss.isnan() & ~br.loss.isnan()
        all_pass &= compare("Loss", sr.loss[valid], br.loss[valid])

    # === Test 3: batch_size=4 (more samples) ===
    print("\n=== Test 3: 4x collect_chat vs collect_chat_batch (batch_size=4) ===")
    samples_4 = SAMPLES + SAMPLES  # repeat to get 4 samples
    single_results_4 = []
    for messages in samples_4:
        collector.model.zero_grad()
        r = collector.collect_chat(messages, collect_gradients=True)
        single_results_4.append(r)

    collector.model.zero_grad()
    batch_results_4 = collector.collect_chat_batch(
        samples_4, collect_gradients=True, add_generation_prompt=False,
    )

    for si in range(len(samples_4)):
        sr = single_results_4[si]
        br = batch_results_4[si]
        print(f"\nSample {si} ({len(sr.tokens)} tokens):")
        for layer in LAYERS:
            all_pass &= compare(f"Grad L{layer}", sr.gradients[layer], br.gradients[layer])

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
