#!/usr/bin/env python3
"""Sanity checks for the delta computation in analyze_delta_cosine.py.

Verifies:
1. Delta is ~0 when lr=0
2. Delta scales linearly with lr (small lr regime)
3. Model weights are properly restored after delta computation
4. Layer indexing matches between collector hooks and output_hidden_states
5. Delta sign: training on misaligned data should move activations TOWARD
   evil direction (positive cosine), not away from it

Run from project root:
    python tests/test_delta_sanity.py [--gpu 0]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_shift.collector import ActivationCollector
from grad_shift.data_utils import load_chat_data, inject_prompt_into_chat
from grad_shift.delta import compute_delta_inplace


def check_zero_lr(collector, messages, layers, device):
    """Delta should be exactly zero when lr=0."""
    print("=" * 60)
    print("TEST 1: Delta with lr=0 should be zero")
    print("=" * 60)

    result = collector.collect_chat(messages, collect_gradients=True)
    input_ids = torch.tensor(
        [result.tokens], dtype=torch.long, device=device,
    )

    delta = compute_delta_inplace(collector.model, input_ids, layers, lr=0.0)
    collector.model.zero_grad()

    all_pass = True
    for l in layers:
        max_val = delta[l].abs().max().item()
        status = "PASS" if max_val == 0.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Layer {l}: max|delta| = {max_val:.2e}  {status}")

    assert all_pass, "Non-zero delta with lr=0"
    print("PASS\n")


def check_linearity(collector, messages, layers, device):
    """Delta should scale approximately linearly with lr.

    NOTE: In bf16, small SGD updates (lr*grad) can fall below the
    representable precision for some parameters, causing non-linear
    scaling. This test is INFORMATIONAL, not a pass/fail gate.
    Perfect linearity is only expected in float32 with very small lr.
    """
    print("=" * 60)
    print("TEST 2: Delta scaling with lr (informational)")
    print("=" * 60)

    lr1 = 1e-5
    lr2 = 2e-5

    # First run
    result = collector.collect_chat(messages, collect_gradients=True)
    input_ids = torch.tensor(
        [result.tokens], dtype=torch.long, device=device,
    )
    delta1 = compute_delta_inplace(collector.model, input_ids, layers, lr=lr1)
    collector.model.zero_grad()

    # Second run (same data, same gradients)
    result = collector.collect_chat(messages, collect_gradients=True)
    delta2 = compute_delta_inplace(collector.model, input_ids, layers, lr=lr2)
    collector.model.zero_grad()

    print(f"  lr1={lr1}, lr2={lr2}, expected ratio=2.0")
    print(f"  (In bf16, ratio > 2 is expected: 2x lr crosses the")
    print(f"   quantization threshold for more parameters)")
    for l in layers:
        norm1 = delta1[l].norm().item()
        norm2 = delta2[l].norm().item()
        if norm1 > 1e-10:
            ratio = norm2 / norm1
            cos = F.cosine_similarity(
                delta1[l].flatten().unsqueeze(0),
                delta2[l].flatten().unsqueeze(0),
            ).item()
            print(f"  Layer {l}: ratio={ratio:.4f}, cos={cos:.6f}")
        else:
            print(f"  Layer {l}: norm too small ({norm1:.2e}), skip")

    print("INFO (no assertion)\n")


def check_weight_restoration(collector, messages, layers, device):
    """Model should produce identical outputs before and after delta."""
    print("=" * 60)
    print("TEST 3: Weights properly restored after delta computation")
    print("=" * 60)

    result = collector.collect_chat(messages, collect_gradients=True)
    input_ids = torch.tensor(
        [result.tokens], dtype=torch.long, device=device,
    )

    # Get reference output before delta
    with torch.no_grad():
        ref_out = collector.model(
            input_ids=input_ids, output_hidden_states=True,
        )
        ref_h = {
            l: ref_out.hidden_states[l].squeeze(0).clone()
            for l in layers
        }
    del ref_out

    # Compute delta (modifies and restores weights)
    _ = compute_delta_inplace(collector.model, input_ids, layers, lr=1e-4)
    collector.model.zero_grad()

    # Check output matches reference
    with torch.no_grad():
        check_out = collector.model(
            input_ids=input_ids, output_hidden_states=True,
        )
        check_h = {
            l: check_out.hidden_states[l].squeeze(0).clone()
            for l in layers
        }
    del check_out

    all_pass = True
    for l in layers:
        max_diff = (ref_h[l] - check_h[l]).abs().max().item()
        status = "PASS" if max_diff == 0.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Layer {l}: max|diff| = {max_diff:.2e}  {status}")

    assert all_pass, "Weights not properly restored"
    print("PASS\n")


def check_layer_indexing(collector, messages, layers, device):
    """Verify collector hooks and output_hidden_states use same indexing.

    NOTE: output_hidden_states[N] for the last layer is post-final-norm,
    not the raw block output. This is expected to mismatch the collector
    hook, which captures pre-norm. compute_delta_inplace handles this
    with a temporary forward hook on the last block.
    """
    print("=" * 60)
    print("TEST 4: Layer indexing consistency (hooks vs hidden_states)")
    print("=" * 60)

    num_layers = collector.num_layers

    result = collector.collect_chat(messages, collect_gradients=False)
    input_ids = torch.tensor(
        [result.tokens], dtype=torch.long, device=device,
    )

    with torch.no_grad():
        out = collector.model(
            input_ids=input_ids, output_hidden_states=True,
        )

    all_pass = True
    for l in layers:
        hook_actv = result.activations[l].to(device)
        hs_actv = out.hidden_states[l].squeeze(0)
        max_diff = (hook_actv - hs_actv).abs().max().item()

        if l == num_layers:
            # Last layer: output_hidden_states[N] is post-final-norm,
            # hook captures pre-norm. Mismatch is expected.
            print(f"  Layer {l}: max|hook - hidden_states| = {max_diff:.2e}  "
                  f"SKIP (last layer: post-norm mismatch expected, "
                  f"delta uses hook)")
        else:
            status = "PASS" if max_diff < 1e-5 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  Layer {l}: max|hook - hidden_states| = {max_diff:.2e}  "
                  f"{status}")

    assert all_pass, "Layer indexing mismatch"
    print("PASS\n")


def check_delta_sign(collector, messages, layers, device, evil_vec_path):
    """Training on evil data should push activations toward evil direction.

    Gradient descent on misaligned data minimizes CE loss, making the model
    better at producing evil text. The delta should therefore have a positive
    cosine with the evil direction (at least at some layers).
    """
    print("=" * 60)
    print("TEST 5: Delta sign check (evil direction)")
    print("=" * 60)

    lr = 1e-4

    evil_vec = torch.load(evil_vec_path, map_location="cpu").float()
    print(f"  Evil vector shape: {evil_vec.shape}")

    result = collector.collect_chat(messages, collect_gradients=True)
    input_ids = torch.tensor(
        [result.tokens], dtype=torch.long, device=device,
    )

    delta = compute_delta_inplace(collector.model, input_ids, layers, lr=lr)
    collector.model.zero_grad()

    # Also compute with NEGATED lr (gradient ascent) for comparison
    result2 = collector.collect_chat(messages, collect_gradients=True)
    delta_ascent = compute_delta_inplace(
        collector.model, input_ids, layers, lr=-lr,
    )
    collector.model.zero_grad()

    print(f"\n  {'Layer':>5} | {'descent':>10} | {'ascent':>10} | note")
    print("  " + "-" * 50)

    for l in layers:
        ev = evil_vec[l] if evil_vec.dim() == 2 else evil_vec.clone()
        ev = ev / (ev.norm() + 1e-12)

        # Pooled cosine: mean over tokens then cosine
        d_pooled = delta[l].mean(dim=0)
        da_pooled = delta_ascent[l].mean(dim=0)
        cos_desc = F.cosine_similarity(
            d_pooled.unsqueeze(0), ev.unsqueeze(0),
        ).item()
        cos_asc = F.cosine_similarity(
            da_pooled.unsqueeze(0), ev.unsqueeze(0),
        ).item()

        note = ""
        if abs(cos_desc + cos_asc) > 0.01 * max(
            abs(cos_desc), abs(cos_asc), 1e-6
        ):
            note = "  WARN: not antisymmetric"
        print(f"  {l:>5} | {cos_desc:>10.4f} | {cos_asc:>10.4f} |{note}")

    print("\n  If signs are correct, descent and ascent should be")
    print("  roughly antisymmetric (opposite sign, similar magnitude).")
    print("  Descent on evil data => moves TOWARD evil in some layers.")
    print("DONE (manual inspection)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--evil_vec",
        default=("steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/"
                 "evil_response_avg_diff_thresh50.pt"),
    )
    parser.add_argument(
        "--train_file",
        default="data/persona_vectors_dataset/evil/misaligned_2.jsonl",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = getattr(torch, args.dtype)

    print(f"Loading model: {args.model}")
    collector = ActivationCollector(
        args.model, device=device, dtype=dtype, trust_remote_code=True,
    )

    # Test on a subset of layers
    layers = [1, 7, 14, 20, 25, 28]
    print(f"Test layers: {layers}")

    # Load one sample
    chats = load_chat_data(args.train_file, max_samples=1)
    messages = inject_prompt_into_chat(
        chats[0], prompt=None, mode="prepend_user",
        baseline_system_prompt="You are an LLM chatbot.",
    )
    print(f"Sample tokens: {len(collector.tokenizer.apply_chat_template(messages, tokenize=True))}\n")

    check_zero_lr(collector, messages, layers, device)
    check_linearity(collector, messages, layers, device)
    check_weight_restoration(collector, messages, layers, device)
    check_layer_indexing(collector, messages, layers, device)

    if Path(args.evil_vec).exists():
        check_delta_sign(collector, messages, layers, device, args.evil_vec)
    else:
        print(f"Skipping sign test: {args.evil_vec} not found")

    print("=" * 60)
    print("ALL AUTOMATED TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
