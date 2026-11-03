#!/usr/bin/env python3
"""Test first-order activation change approximation against actual changes.

Verifies that compute_activation_change_approx (JVP) closely matches
compute_activation_change_actual (finite difference) for small learning
rates, using a tiny randomly initialized GPT-2 model.  The tests perform no
network access and do not depend on a Hugging Face cache.

Run from the project root:
    python tests/test_change_approx.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from grad_shift.change_approx import (
    compute_activation_change_approx,
    compute_activation_change_actual,
    compute_sgd_param_changes,
    collect_activation_changes,
)


def _tiny_model_and_input(seq_len: int = 8):
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=4,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config).eval()
    input_ids = torch.arange(1, seq_len + 1).remainder(config.vocab_size).unsqueeze(0)
    return model, input_ids


def test_approx_vs_actual():
    """Core test: JVP approximation should match actual change for small lr."""
    print("=" * 60)
    print("TEST: First-order approximation vs actual change")
    print("=" * 60)

    torch.manual_seed(42)

    model, input_ids = _tiny_model_and_input()

    num_layers = model.config.n_layer
    # Test at early, middle, and late layers
    layers = [1, num_layers // 2, num_layers]
    print(f"Testing layers: {layers} (model has {num_layers} transformer blocks)")

    # Compute parameter changes via SGD
    lr = 1e-5
    print(f"Computing SGD param changes (lr={lr})...")
    param_changes, loss_val = compute_sgd_param_changes(
        model, input_ids, learning_rate=lr,
    )
    print(f"  Loss: {loss_val:.4f}")
    print(f"  Parameters with non-zero grad: {len(param_changes)}")

    # Sanity: param changes should be small
    max_change = max(d.abs().max().item() for d in param_changes.values())
    print(f"  Max |Dtheta|: {max_change:.2e}")

    # Compute first-order approximation (JVP)
    print("\nComputing JVP approximation...")
    approx = compute_activation_change_approx(
        model, input_ids, layers, param_changes,
    )

    # Compute actual change
    print("Computing actual change...")
    actual = compute_activation_change_actual(
        model, input_ids, layers, param_changes,
    )

    # Compare
    print("\n" + "-" * 60)
    print(f"{'Layer':>6} {'Cos Sim':>10} {'Rel Error':>12} "
          f"{'Approx Norm':>13} {'Actual Norm':>13}")
    print("-" * 60)

    all_pass = True
    for layer in layers:
        a = approx[layer].float().flatten()
        r = actual[layer].float().flatten()

        cos_sim = F.cosine_similarity(a.unsqueeze(0), r.unsqueeze(0)).item()
        approx_norm = a.norm().item()
        actual_norm = r.norm().item()
        rel_error = (a - r).norm().item() / (actual_norm + 1e-12)

        status = "PASS" if cos_sim > 0.99 and rel_error < 0.05 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(
            f"{layer:>6} {cos_sim:>10.6f} {rel_error:>12.6f} "
            f"{approx_norm:>13.2e} {actual_norm:>13.2e}  {status}"
        )

    print("-" * 60)

    # Per-token analysis at middle layer
    mid_layer = layers[1]
    a_tok = approx[mid_layer].squeeze(0)  # (seq_len, hidden_dim)
    r_tok = actual[mid_layer].squeeze(0)
    print(f"\nPer-token analysis at layer {mid_layer}:")
    print(f"  {'Tok':>4} {'Cos Sim':>10} {'Rel Error':>12}")
    for t in range(a_tok.shape[0]):
        cs = F.cosine_similarity(
            a_tok[t].unsqueeze(0), r_tok[t].unsqueeze(0)
        ).item()
        re = (a_tok[t] - r_tok[t]).norm().item() / (r_tok[t].norm().item() + 1e-12)
        print(f"  {t:>4} {cs:>10.6f} {re:>12.6f}")

    assert all_pass, "Some layers failed the approximation quality check"
    print("\nPASS: All layers within tolerance")


def test_collect_convenience():
    """Test the all-in-one collect_activation_changes function."""
    print("\n" + "=" * 60)
    print("TEST: collect_activation_changes convenience function")
    print("=" * 60)

    torch.manual_seed(42)

    model, input_ids = _tiny_model_and_input(seq_len=4)
    layers = [1, model.config.n_layer]

    result = collect_activation_changes(
        model, input_ids, layers, learning_rate=1e-5,
    )

    assert "change_approx" in result
    assert "param_changes" in result
    assert "loss" in result
    assert set(result["change_approx"].keys()) == set(layers)
    for layer in layers:
        t = result["change_approx"][layer]
        assert t.shape == (1, input_ids.shape[1], model.config.n_embd), (
            f"Unexpected shape {t.shape}"
        )
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  Layers returned: {sorted(result['change_approx'].keys())}")
    print(f"  Param changes: {len(result['param_changes'])} parameters")
    print("PASS")


def test_scaling_with_lr():
    """Verify that the approximation scales linearly with learning rate."""
    print("\n" + "=" * 60)
    print("TEST: Linear scaling with learning rate")
    print("=" * 60)

    torch.manual_seed(42)

    model, input_ids = _tiny_model_and_input(seq_len=5)
    layer = model.config.n_layer // 2
    layers = [layer]

    # Get gradients once
    param_changes_1x, _ = compute_sgd_param_changes(
        model, input_ids, learning_rate=1e-5,
    )

    # 2x the param changes
    param_changes_2x = {n: 2.0 * d for n, d in param_changes_1x.items()}

    approx_1x = compute_activation_change_approx(
        model, input_ids, layers, param_changes_1x,
    )
    approx_2x = compute_activation_change_approx(
        model, input_ids, layers, param_changes_2x,
    )

    # The JVP is linear: 2x input should give 2x output
    ratio = approx_2x[layer].norm() / approx_1x[layer].norm()
    print(f"  ||approx(2x)|| / ||approx(1x)|| = {ratio.item():.6f} (expect 2.0)")

    diff = (
        (approx_2x[layer] - 2.0 * approx_1x[layer]).norm()
        / approx_1x[layer].norm()
    )
    print(f"  ||approx(2x) - 2*approx(1x)|| / ||approx(1x)|| = {diff.item():.2e}")
    assert diff < 1e-4, f"Linearity violated: {diff.item()}"
    print("PASS")


def test_zero_change():
    """Verify that zero param changes give zero activation change."""
    print("\n" + "=" * 60)
    print("TEST: Zero param changes -> zero activation change")
    print("=" * 60)

    torch.manual_seed(42)

    model, input_ids = _tiny_model_and_input(seq_len=4)
    layer = model.config.n_layer // 2
    layers = [layer]

    # All-zero param changes
    param_changes = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
    }

    approx = compute_activation_change_approx(
        model, input_ids, layers, param_changes,
    )

    max_val = approx[layer].abs().max().item()
    print(f"  Max |Dh_l| with zero Dtheta: {max_val:.2e}")
    assert max_val < 1e-6, f"Non-zero output for zero input: {max_val}"
    print("PASS")


def test_single_param():
    """Test with changes to a single parameter only."""
    print("\n" + "=" * 60)
    print("TEST: Single parameter change")
    print("=" * 60)

    torch.manual_seed(42)

    model, input_ids = _tiny_model_and_input(seq_len=5)
    layers = [3, model.config.n_layer]

    # Change only one parameter (in layer 2 = transformer block 2)
    target_name = "transformer.h.2.mlp.c_fc.weight"
    param = dict(model.named_parameters())[target_name]
    # Per-element scale 1e-4 gives norm ~ sqrt(numel)*1e-4 ~ 0.15.
    # This is in the sweet spot: small enough for accurate first-order
    # approx, large enough that float32 h_after-h_before is precise.
    delta = torch.randn_like(param) * 1e-4
    param_changes = {target_name: delta}

    approx = compute_activation_change_approx(
        model, input_ids, layers, param_changes,
    )
    actual = compute_activation_change_actual(
        model, input_ids, layers, param_changes,
    )

    print(f"  Changed parameter: {target_name}")
    print(f"  Perturbation norm: {delta.norm().item():.2e}")
    for layer in layers:
        a = approx[layer].flatten()
        r = actual[layer].flatten()
        cos = F.cosine_similarity(a.unsqueeze(0), r.unsqueeze(0)).item()
        rel_err = (a - r).norm().item() / (r.norm().item() + 1e-12)
        print(f"  Layer {layer}: cos_sim={cos:.6f}, rel_error={rel_err:.6f}, "
              f"approx_norm={a.norm().item():.2e}")
        if r.norm().item() > 1e-10:
            assert cos > 0.99, f"Layer {layer} cos sim too low"
            assert rel_err < 0.05, f"Layer {layer} rel error too high"
    print("PASS")


if __name__ == "__main__":
    test_approx_vs_actual()
    test_collect_convenience()
    test_scaling_with_lr()
    test_zero_change()
    test_single_param()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
