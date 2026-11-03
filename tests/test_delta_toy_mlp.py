#!/usr/bin/env python3
"""Toy MLP tests with hard-coded weights to verify delta computation.

These tests use a simple 2-layer MLP with known weight values to verify:
1. Delta = h(θ - lr*∇L) - h(θ) matches manual computation
2. Weight restoration is exact
3. Delta is zero when lr=0
4. Delta scales linearly with lr (float32)
5. Defenseless protocol: defense only affects gradients, not delta forward passes

Run from project root:
    python tests/test_delta_toy_mlp.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_toy_mlp(d_in=4, d_hidden=8, d_out=3):
    """Create a 2-layer MLP with deterministic hard-coded weights.

    Architecture: x -> Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out)

    Layer 0 output = ReLU(W1 @ x + b1)  (the "hidden activations")
    Layer 1 output = W2 @ h + b2         (the logits)

    Returns (model, W1, b1, W2, b2) where W/b are the original weight values.
    """
    model = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )

    # Hard-code weights for reproducibility
    torch.manual_seed(0)
    with torch.no_grad():
        # Layer 1 (index 0 in Sequential)
        model[0].weight.copy_(torch.randn(d_hidden, d_in) * 0.5)
        model[0].bias.copy_(torch.randn(d_hidden) * 0.1)
        # Layer 2 (index 2 in Sequential)
        model[2].weight.copy_(torch.randn(d_out, d_hidden) * 0.5)
        model[2].bias.copy_(torch.randn(d_out) * 0.1)

    W1 = model[0].weight.data.clone()
    b1 = model[0].bias.data.clone()
    W2 = model[2].weight.data.clone()
    b2 = model[2].bias.data.clone()

    return model, W1, b1, W2, b2


def get_hidden_activation(model, x):
    """Get hidden layer activation (output of ReLU) for input x."""
    return F.relu(model[0](x))


def get_logits(model, x):
    """Full forward pass."""
    return model(x)


def compute_delta_manual(model, x, layers, lr):
    """Manually compute delta = h(θ - lr*∇) - h(θ).

    Mimics compute_delta_inplace logic but for toy MLP.
    Layers: 0 = hidden (post-ReLU), 1 = logits (output).

    Args:
        model: MLP with .grad already populated.
        x: Input tensor.
        layers: List of layer indices (0 or 1).
        lr: Learning rate.

    Returns:
        Dict mapping layer index to delta tensor.
    """
    # 1. Forward pass BEFORE weight update
    with torch.no_grad():
        h0_before = get_hidden_activation(model, x)
        h1_before = get_logits(model, x)
    h_before = {0: h0_before.clone(), 1: h1_before.clone()}

    # 2. Save original weights, apply SGD step
    orig = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.grad is not None:
                orig[n] = p.data.clone()
                p.data.add_(p.grad, alpha=-lr)

    # 3. Forward pass AFTER weight update
    with torch.no_grad():
        h0_after = get_hidden_activation(model, x)
        h1_after = get_logits(model, x)
    h_after = {0: h0_after.clone(), 1: h1_after.clone()}

    # 4. Restore original weights
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in orig:
                p.data.copy_(orig[n])

    return {l: h_after[l] - h_before[l] for l in layers}


# ── Tests ──────────────────────────────────────────────────────────────


def test_delta_matches_manual(verbose=True):
    """Delta from our function matches a fully manual computation.

    We manually compute:
      h_hidden_before = ReLU(W1 @ x + b1)
      h_hidden_after  = ReLU((W1 - lr * dL/dW1) @ x + (b1 - lr * dL/db1))
      delta_hidden    = h_hidden_after - h_hidden_before

    And similarly for the output layer.
    """
    if verbose:
        print("=" * 60)
        print("TEST 1: Delta matches fully manual computation")
        print("=" * 60)

    model, W1, b1, W2, b2 = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])  # class index for CE loss
    lr = 0.01

    # Forward + backward to populate .grad
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    # Extract gradients
    dW1 = model[0].weight.grad.clone()
    db1 = model[0].bias.grad.clone()
    dW2 = model[2].weight.grad.clone()
    db2 = model[2].bias.grad.clone()

    # Compute delta via our helper function
    delta = compute_delta_manual(model, x, [0, 1], lr)

    # Compute delta fully manually
    # Before: use original weights
    h_hidden_before = F.relu(W1 @ x.squeeze() + b1)
    logits_before = W2 @ h_hidden_before + b2

    # After: use updated weights
    W1_new = W1 - lr * dW1
    b1_new = b1 - lr * db1
    W2_new = W2 - lr * dW2
    b2_new = b2 - lr * db2

    h_hidden_after = F.relu(W1_new @ x.squeeze() + b1_new)
    logits_after = W2_new @ h_hidden_after + b2_new

    expected_delta_0 = (h_hidden_after - h_hidden_before).unsqueeze(0)
    expected_delta_1 = (logits_after - logits_before).unsqueeze(0)

    err0 = (delta[0] - expected_delta_0).abs().max().item()
    err1 = (delta[1] - expected_delta_1).abs().max().item()

    if verbose:
        print(f"  Hidden delta error:  {err0:.2e}")
        print(f"  Output delta error:  {err1:.2e}")
        print(f"  Hidden delta values: {delta[0].squeeze().tolist()}")
        print(f"  Expected values:     {expected_delta_0.squeeze().tolist()}")

    assert err0 < 1e-6, f"Hidden delta mismatch: {err0}"
    assert err1 < 1e-6, f"Output delta mismatch: {err1}"
    if verbose:
        print("PASS\n")


def test_zero_lr(verbose=True):
    """Delta must be exactly zero when lr=0."""
    if verbose:
        print("=" * 60)
        print("TEST 2: Delta is exactly zero when lr=0")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    delta = compute_delta_manual(model, x, [0, 1], lr=0.0)

    for l in [0, 1]:
        max_val = delta[l].abs().max().item()
        if verbose:
            print(f"  Layer {l}: max|delta| = {max_val}")
        assert max_val == 0.0, f"Non-zero delta at layer {l} with lr=0"

    model.zero_grad()
    if verbose:
        print("PASS\n")


def test_linear_scaling(verbose=True):
    """Delta should scale exactly linearly with lr in float32."""
    if verbose:
        print("=" * 60)
        print("TEST 3: Delta scales linearly with lr (float32)")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])

    lr1 = 1e-4
    lr2 = 2e-4

    # First run
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    delta1 = compute_delta_manual(model, x, [0, 1], lr=lr1)
    model.zero_grad()

    # Second run (same data → same gradients)
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    delta2 = compute_delta_manual(model, x, [0, 1], lr=lr2)
    model.zero_grad()

    for l in [0, 1]:
        norm1 = delta1[l].norm().item()
        norm2 = delta2[l].norm().item()
        if norm1 > 1e-10:
            ratio = norm2 / norm1
            # Also check cosine (should be exactly 1.0 for linear scaling)
            cos = F.cosine_similarity(
                delta1[l].flatten().unsqueeze(0),
                delta2[l].flatten().unsqueeze(0),
            ).item()
            if verbose:
                print(f"  Layer {l}: ratio={ratio:.6f} (expect 2.0), cos={cos:.8f}")
            # Allow small tolerance for float32 nonlinearity in ReLU boundary
            assert abs(ratio - 2.0) < 0.05, f"Non-linear scaling at layer {l}: ratio={ratio}"
            assert cos > 0.999, f"Direction changed at layer {l}: cos={cos}"
        else:
            if verbose:
                print(f"  Layer {l}: norm too small ({norm1:.2e}), skip")

    if verbose:
        print("PASS\n")


def test_weight_restoration(verbose=True):
    """Weights must be exactly restored after delta computation."""
    if verbose:
        print("=" * 60)
        print("TEST 4: Exact weight restoration")
        print("=" * 60)

    model, W1, b1, W2, b2 = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    _ = compute_delta_manual(model, x, [0, 1], lr=0.01)
    model.zero_grad()

    # Check weights match originals exactly
    err_W1 = (model[0].weight.data - W1).abs().max().item()
    err_b1 = (model[0].bias.data - b1).abs().max().item()
    err_W2 = (model[2].weight.data - W2).abs().max().item()
    err_b2 = (model[2].bias.data - b2).abs().max().item()

    if verbose:
        print(f"  W1 error: {err_W1:.2e}")
        print(f"  b1 error: {err_b1:.2e}")
        print(f"  W2 error: {err_W2:.2e}")
        print(f"  b2 error: {err_b2:.2e}")

    assert err_W1 == 0.0, "W1 not restored"
    assert err_b1 == 0.0, "b1 not restored"
    assert err_W2 == 0.0, "W2 not restored"
    assert err_b2 == 0.0, "b2 not restored"

    if verbose:
        print("PASS\n")


def test_grad_descent_reduces_loss(verbose=True):
    """SGD step should reduce loss (sanity check on gradient direction)."""
    if verbose:
        print("=" * 60)
        print("TEST 5: Gradient descent reduces loss")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])
    lr = 0.01

    # Loss before
    logits_before = model(x)
    loss_before = F.cross_entropy(logits_before, target).item()

    # Backward
    loss_t = F.cross_entropy(model(x), target)
    loss_t.backward()

    # Apply SGD step
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-lr)
    model.zero_grad()

    # Loss after
    logits_after = model(x)
    loss_after = F.cross_entropy(logits_after, target).item()

    if verbose:
        print(f"  Loss before: {loss_before:.6f}")
        print(f"  Loss after:  {loss_after:.6f}")
        print(f"  Reduction:   {loss_before - loss_after:.6f}")

    assert loss_after < loss_before, (
        f"Loss increased: {loss_before} -> {loss_after}"
    )
    if verbose:
        print("PASS\n")


def test_defenseless_protocol(verbose=True):
    """Defense only affects gradients, not delta forward passes.

    Setup:
      - "Defense" = add a constant bias to the hidden activations during
        the backward pass (simulating a prompt/steering that changes loss).
      - Delta forward passes should NOT include this bias.

    We verify:
      - Gradients differ between defenseless and defended conditions
      - Both conditions produce deltas measured on the UNDEFENDED model
      - The deltas differ (because gradients differ) but the h_before
        is identical in both cases.
    """
    if verbose:
        print("=" * 60)
        print("TEST 6: Defenseless delta protocol")
        print("=" * 60)

    model, W1, b1, W2, b2 = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])
    lr = 0.01

    # ── Defenseless condition ──
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    grad_defenseless = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }
    delta_defenseless = compute_delta_manual(model, x, [0, 1], lr)
    model.zero_grad()

    # ── Defended condition: add bias to hidden activations ──
    # This simulates how a steering vector or prompt would alter the
    # forward pass during gradient computation, producing different grads.
    defense_bias = torch.tensor([0.5, -0.3, 0.2, 0.0, -0.1, 0.4, -0.2, 0.1])

    # Custom forward with defense for gradient computation only
    h_raw = model[0](x)
    h_defended = F.relu(h_raw + defense_bias)  # defense applied
    logits_defended = model[2](h_defended)
    loss_defended = F.cross_entropy(logits_defended, target)
    loss_defended.backward()

    grad_defended = {
        n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
    }

    # Delta uses defenseless forward passes (same as defenseless condition
    # code path) but with defended gradients still in .grad
    delta_defended = compute_delta_manual(model, x, [0, 1], lr)
    model.zero_grad()

    # ── Checks ──

    # 1. Gradients should differ between conditions
    grad_diff = sum(
        (grad_defenseless[n] - grad_defended[n]).abs().max().item()
        for n in grad_defenseless
    )
    if verbose:
        print(f"  Max gradient difference: {grad_diff:.6f}")
    assert grad_diff > 1e-4, "Gradients should differ between conditions"

    # 2. Deltas should differ (different gradients → different weight updates)
    delta_diff_0 = (delta_defenseless[0] - delta_defended[0]).abs().max().item()
    delta_diff_1 = (delta_defenseless[1] - delta_defended[1]).abs().max().item()
    if verbose:
        print(f"  Delta diff layer 0: {delta_diff_0:.6f}")
        print(f"  Delta diff layer 1: {delta_diff_1:.6f}")
    assert delta_diff_0 > 1e-6, "Deltas should differ between conditions"

    # 3. h_before should be identical in both conditions (defenseless forward)
    #    We verify this by recomputing — both use the SAME model weights
    with torch.no_grad():
        h_before = get_hidden_activation(model, x)
    # The compute_delta_manual function always uses the unmodified model
    # for h_before, so this is guaranteed by construction. But let's verify
    # that the model weights weren't corrupted.
    h_expected = F.relu(W1 @ x.squeeze() + b1).unsqueeze(0)
    h_err = (h_before - h_expected).abs().max().item()
    if verbose:
        print(f"  h_before matches original weights: err={h_err:.2e}")
    assert h_err < 1e-6, "h_before should use original (defenseless) weights"

    # 4. Weights fully restored after both conditions
    err_W1 = (model[0].weight.data - W1).abs().max().item()
    err_W2 = (model[2].weight.data - W2).abs().max().item()
    if verbose:
        print(f"  Weight restoration: W1 err={err_W1:.2e}, W2 err={err_W2:.2e}")
    assert err_W1 == 0.0 and err_W2 == 0.0, "Weights not restored"

    if verbose:
        print("PASS\n")


def test_multiple_samples_different_grads(verbose=True):
    """Different inputs produce different gradients and different deltas."""
    if verbose:
        print("=" * 60)
        print("TEST 7: Different samples → different deltas")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    x1 = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    x2 = torch.tensor([[0.2, 0.7, -0.4, 0.1]])
    target = torch.tensor([0])
    lr = 0.01

    # Sample 1
    loss = F.cross_entropy(model(x1), target)
    loss.backward()
    delta1 = compute_delta_manual(model, x1, [0, 1], lr)
    model.zero_grad()

    # Sample 2
    loss = F.cross_entropy(model(x2), target)
    loss.backward()
    delta2 = compute_delta_manual(model, x2, [0, 1], lr)
    model.zero_grad()

    for l in [0, 1]:
        diff = (delta1[l] - delta2[l]).abs().max().item()
        cos = F.cosine_similarity(
            delta1[l].flatten().unsqueeze(0),
            delta2[l].flatten().unsqueeze(0),
        ).item()
        if verbose:
            print(f"  Layer {l}: max diff={diff:.6f}, cos={cos:.4f}")
        assert diff > 1e-6, f"Deltas should differ for different inputs at layer {l}"

    if verbose:
        print("PASS\n")


def test_delta_sign_known_direction(verbose=True):
    """Gradient descent on target class should push logits toward that class.

    If we train on target=0 (minimize CE loss for class 0), the logit for
    class 0 should increase, so delta[1][0] (change in logit 0) should be
    positive.
    """
    if verbose:
        print("=" * 60)
        print("TEST 8: Delta sign — training pushes toward target class")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    target = torch.tensor([0])
    lr = 0.01

    loss = F.cross_entropy(model(x), target)
    loss.backward()

    delta = compute_delta_manual(model, x, [1], lr)
    model.zero_grad()

    logit_deltas = delta[1].squeeze()
    target_logit_delta = logit_deltas[target.item()].item()

    if verbose:
        print(f"  Logit deltas: {logit_deltas.tolist()}")
        print(f"  Target class {target.item()} logit delta: {target_logit_delta:.6f}")

    assert target_logit_delta > 0, (
        f"Target logit should increase, got {target_logit_delta}"
    )
    if verbose:
        print("PASS\n")


def test_bf16_weight_restoration_needs_copy(verbose=True):
    """Demonstrate that add/sub in bf16 does NOT restore weights, but copy does.

    This validates our design choice of saving to CPU and restoring via copy_.
    """
    if verbose:
        print("=" * 60)
        print("TEST 9: bf16 add/sub is lossy, copy_ is exact")
        print("=" * 60)

    model, *_ = make_toy_mlp()
    model = model.to(torch.bfloat16)

    x = torch.tensor([[1.0, -0.5, 0.3, 0.8]], dtype=torch.bfloat16)
    target = torch.tensor([0])
    lr = 1e-4

    logits = model(x)
    loss = F.cross_entropy(logits.float(), target)
    loss.backward()

    # Method A: add then subtract (LOSSY in bf16)
    W1_orig = model[0].weight.data.clone()
    with torch.no_grad():
        model[0].weight.data.add_(model[0].weight.grad, alpha=-lr)
        model[0].weight.data.add_(model[0].weight.grad, alpha=lr)

    err_addsub = (model[0].weight.data - W1_orig).abs().max().item()

    # Restore
    model[0].weight.data.copy_(W1_orig)

    # Method B: save to CPU, restore via copy_ (EXACT)
    W1_cpu = model[0].weight.data.cpu().clone()
    with torch.no_grad():
        model[0].weight.data.add_(model[0].weight.grad, alpha=-lr)
    model[0].weight.data.copy_(W1_cpu.to(model[0].weight.device))

    err_copy = (model[0].weight.data - W1_orig).abs().max().item()

    model.zero_grad()

    if verbose:
        print(f"  bf16 add/sub error: {err_addsub:.2e}")
        print(f"  CPU copy error:     {err_copy:.2e}")

    # copy_ should be exact
    assert err_copy == 0.0, f"copy_ method should be exact, got {err_copy}"
    if verbose:
        if err_addsub > 0:
            print("  (add/sub lossy as expected in bf16)")
        print("PASS\n")


def main():
    print("\nToy MLP Delta Tests")
    print("=" * 60)
    print()

    test_delta_matches_manual()
    test_zero_lr()
    test_linear_scaling()
    test_weight_restoration()
    test_grad_descent_reduces_loss()
    test_defenseless_protocol()
    test_multiple_samples_different_grads()
    test_delta_sign_known_direction()
    test_bf16_weight_restoration_needs_copy()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
