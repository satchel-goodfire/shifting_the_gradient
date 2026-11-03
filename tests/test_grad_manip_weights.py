#!/usr/bin/env python3
"""Verify that gradient manipulation at the layer output correctly propagates
to affect parameter gradients of BOTH attention and MLP weights within the
manipulated layer.

The GradientManipulationManager hooks the layer OUTPUT tensor. During backward,
the modified dL/dx_out flows back through the residual connections, MLP, and
attention sublayers, so all parameter gradients in the block should reflect
the manipulation.

This test:
1. Builds a toy model with explicit attn_proj and mlp weights per block.
2. Runs forward+backward WITHOUT manipulation -> baseline param grads.
3. Runs forward+backward WITH each manipulation mode -> manipulated param grads.
4. Verifies that manipulated param grads differ from baseline AND that the
   manipulation correctly transforms the output gradient.

Run from project root:
    python tests/test_grad_manip_weights.py
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from grad_shift.finetuning.grad_manipulation import (
    GradientManipulationManager,
    VALID_MODES,
)


# ── Toy model that find_transformer_blocks can discover ──────────────
# Must expose `model.layers.{i}` pattern.

class FakeTransformerBlock(nn.Module):
    """Mimics a transformer block with separate attention and MLP paths.

    Uses GELU nonlinearity so that gradients vary per token (otherwise
    the linear model gives identical gradients at every position).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        # residual + attention-like path
        x = x + self.attn_proj(F.gelu(x))
        # residual + MLP path
        x = x + self.mlp(F.gelu(x))
        return x


class FakeTransformerModel(nn.Module):
    """Top-level model: model.layers.{i} pattern for find_transformer_blocks."""
    def __init__(self, hidden_dim, num_layers=4):
        super().__init__()
        # Must be named `model` with child `layers` to match pattern
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeTransformerBlock(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────────

def get_param_grads(model, target_layer_idx):
    """Return cloned .grad tensors for attn_proj and mlp of the target block."""
    block = model.model.layers[target_layer_idx]
    return {
        "attn_proj": block.attn_proj.weight.grad.clone(),
        "mlp": block.mlp.weight.grad.clone(),
    }


def run_forward_backward(model, x_data, target=None, mgr=None):
    """Zero grads, run forward+backward. Optionally with manipulation hooks."""
    model.zero_grad()
    if mgr is not None:
        mgr.register_hooks()
    x = x_data.clone().requires_grad_(True)
    out = model(x)
    if target is not None:
        loss = (out * target).sum()
    else:
        loss = out.sum()
    loss.backward()
    if mgr is not None:
        mgr.remove_hooks()


def capture_output_grad(model, block_idx, x_data, target=None, mgr=None):
    """Run forward+backward and capture the gradient at the block output.

    Registers mgr hooks first (if any) so that the capture hook sees
    the already-manipulated gradient.
    """
    captured = {}
    block = model.model.layers[block_idx]

    if mgr is not None:
        mgr.register_hooks()

    def fwd_hook(module, inp, output):
        tensor = output[0] if isinstance(output, tuple) else output
        if tensor.requires_grad:
            def save(grad):
                captured["grad"] = grad.clone()
            tensor.register_hook(save)

    handle = block.register_forward_hook(fwd_hook)

    model.zero_grad()
    x = x_data.clone().requires_grad_(True)
    out = model(x)
    if target is not None:
        loss = (out * target).sum()
    else:
        loss = out.sum()
    loss.backward()

    handle.remove()
    if mgr is not None:
        mgr.remove_hooks()

    return captured.get("grad")


# ── Main test ────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    hidden_dim = 64
    num_layers = 4
    batch, seq = 8, 16  # 128 tokens for good projection sign diversity
    # embedding-inclusive: layer 3 = model.layers[2]
    target_layer_emb_inclusive = 3
    target_block_idx = target_layer_emb_inclusive - 1  # 0-based block index

    model = FakeTransformerModel(hidden_dim, num_layers)
    x_data = torch.randn(batch, seq, hidden_dim)
    # Per-token random target so loss gradients vary across positions
    # (with loss=out.sum(), all tokens get nearly identical gradients
    # dominated by downstream weights, giving all-same-sign projections)
    target = torch.randn(batch, seq, hidden_dim)

    # ── 1. Baseline: no manipulation ─────────────────────────────
    print("=" * 65)
    print("Verifying gradient manipulation propagates to weight gradients")
    print("=" * 65)

    run_forward_backward(model, x_data, target=target, mgr=None)
    baseline_grads = get_param_grads(model, target_block_idx)
    baseline_output_grad = capture_output_grad(
        model, target_block_idx, x_data, target=target, mgr=None
    )

    # Use a random manipulation vector. With 128 tokens (8×16) and a
    # nonlinear model, the projections should have mixed signs.
    vec = torch.randn(num_layers + 1, hidden_dim)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(vec, f.name)
        vec_path = f.name

    try:
        # Get the normalised unit vector the manager will use
        mgr_probe = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="remove_vector_component",
            layers=[target_layer_emb_inclusive],
        )
        v = mgr_probe._vectors[target_layer_emb_inclusive]
        proj_scalar_verify = baseline_output_grad @ v
        n_pos = (proj_scalar_verify > 0).sum().item()
        n_neg = (proj_scalar_verify < 0).sum().item()

        print(f"\nTarget block index: {target_block_idx}  "
              f"(emb-inclusive layer {target_layer_emb_inclusive})")
        print(f"Hidden dim: {hidden_dim},  Batch: {batch},  Seq: {seq}")
        print(f"Manipulation vector norm: {v.norm():.4f}")
        print(f"Baseline output grad norm: {baseline_output_grad.norm():.4f}")
        print(f"Projection sign distribution: {n_pos} positive, {n_neg} negative")
        print(f"Baseline attn_proj.weight.grad norm: "
              f"{baseline_grads['attn_proj'].norm():.4f}")
        print(f"Baseline mlp.weight.grad norm: "
              f"{baseline_grads['mlp'].norm():.4f}")

        # ── 2. Test each mode ────────────────────────────────────────
        all_pass = True

        for mode in VALID_MODES:
            print(f"\n{'─' * 65}")
            print(f"Mode: {mode}")
            print(f"{'─' * 65}")

            mgr = GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode=mode,
                layers=[target_layer_emb_inclusive],
            )

            # Get manipulated output gradient
            manip_output_grad = capture_output_grad(
                model, target_block_idx, x_data, target=target, mgr=mgr
            )

            # Get manipulated param grads
            run_forward_backward(model, x_data, target=target, mgr=mgr)
            manip_grads = get_param_grads(model, target_block_idx)

            # ── Check output gradient was transformed correctly ──────
            proj_scalar = baseline_output_grad @ v
            proj = proj_scalar.unsqueeze(-1) * v

            if mode == "remove_vector_component":
                expected_output = baseline_output_grad - proj
            elif mode == "keep_vector_component":
                expected_output = proj
            elif mode == "flip_vector_component":
                expected_output = baseline_output_grad - 2 * proj
            elif mode == "pos_sign_vector_component":
                expected_output = (baseline_output_grad - proj
                                   + proj_scalar.abs().unsqueeze(-1) * v)
            elif mode == "neg_sign_vector_component":
                expected_output = (baseline_output_grad - proj
                                   - proj_scalar.abs().unsqueeze(-1) * v)

            output_diff = (manip_output_grad - expected_output).abs().max().item()
            output_cos = F.cosine_similarity(
                manip_output_grad.flatten().unsqueeze(0),
                expected_output.flatten().unsqueeze(0),
            ).item()
            print(f"  Output gradient:")
            print(f"    Max |actual - expected|      = {output_diff:.2e}")
            print(f"    Cosine(actual, expected)     = {output_cos:.6f}")
            ok_output = output_diff < 1e-5
            assert ok_output, f"Output gradient mismatch for {mode}"

            # ── Check param grads changed from baseline ──────────────
            for name in ("attn_proj", "mlp"):
                bl = baseline_grads[name]
                mn = manip_grads[name]
                param_diff = (mn - bl).abs().max().item()
                param_cos = F.cosine_similarity(
                    mn.flatten().unsqueeze(0).float(),
                    bl.flatten().unsqueeze(0).float(),
                ).item()
                param_rel_change = (mn - bl).norm() / bl.norm()

                print(f"  {name}.weight.grad:")
                print(f"    Max |manip - baseline|       = {param_diff:.2e}")
                print(f"    Relative change (L2)         = {param_rel_change:.4f}")
                print(f"    Cosine(manip, baseline)      = {param_cos:.6f}")

                # The manipulation should change the gradient
                # (unless the gradient has zero component along v, which
                # is vanishingly unlikely for random vectors)
                if mode == "keep_vector_component":
                    # keep is a big change — gradient should differ a lot
                    ok_param = param_cos < 0.99
                elif mode == "remove_vector_component":
                    # remove changes only the v-component, small change
                    ok_param = param_diff > 1e-6
                else:
                    ok_param = param_diff > 1e-6

                status = "PASS" if ok_param else "FAIL"
                print(f"    [{status}]")
                all_pass &= ok_param

        # ── 3. Verify non-target layers are NOT affected ─────────────
        print(f"\n{'─' * 65}")
        print("Verifying non-target layers are unaffected")
        print(f"{'─' * 65}")

        mgr_remove = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="remove_vector_component",
            layers=[target_layer_emb_inclusive],
        )
        run_forward_backward(model, x_data, target=target, mgr=mgr_remove)

        for block_idx in range(num_layers):
            if block_idx == target_block_idx:
                continue
            grads = get_param_grads(model, block_idx)
            # Get baseline grads for this block
            run_forward_backward(model, x_data, target=target, mgr=None)
            bl_grads = get_param_grads(model, block_idx)
            # Re-run with manipulation
            run_forward_backward(model, x_data, target=target, mgr=mgr_remove)
            mn_grads = get_param_grads(model, block_idx)

            for name in ("attn_proj", "mlp"):
                diff = (mn_grads[name] - bl_grads[name]).abs().max().item()
                # Blocks AFTER the target should be unaffected (gradient
                # flows backward, so manipulation at block 2 doesn't
                # affect blocks 3+). Blocks BEFORE the target WILL be
                # affected because the modified gradient continues
                # backward through them.
                if block_idx > target_block_idx:
                    ok = diff < 1e-6
                    expect = "unchanged"
                else:
                    # Upstream blocks receive the modified gradient
                    ok = True  # just report, don't fail
                    expect = "changed (expected: modified grad flows upstream)"

                status = "PASS" if ok else "FAIL"
                print(f"  Block {block_idx} {name}: "
                      f"max diff = {diff:.2e}  ({expect})  [{status}]")
                all_pass &= ok

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'=' * 65}")
        if all_pass:
            print("ALL CHECKS PASSED")
            print("  - All 5 modes correctly transform the output gradient")
            print("  - Both attn_proj and mlp weight gradients are affected")
            print("  - Downstream (later) blocks are unaffected")
        else:
            print("SOME CHECKS FAILED")
        print("=" * 65)

    finally:
        os.unlink(vec_path)


if __name__ == "__main__":
    main()
