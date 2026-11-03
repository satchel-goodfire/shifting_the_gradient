#!/usr/bin/env python3
"""Tests for ActivationWriteProjectionManager (residual-write muzzle).

Uses a real tiny Qwen2 (+ optional PEFT LoRA) so module naming, hook
placement (outer LoRA wrapper, not base_layer), and the stream-pinning
invariant are tested with production fidelity.

Run from project root:
    pytest tests/test_activation_write_projection.py -q
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.finetuning.activation_write_projection import (
    ActivationWriteProjectionManager,
)

HIDDEN = 64
N_LAYERS = 4


def make_model(peft=True, seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(cfg)
    if peft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
        ))
    return model


def save_vec(v):
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(v, f.name)
    f.close()
    return f.name


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 127, (2, 10), generator=g)
    return {"input_ids": ids, "labels": ids.clone()}


def hidden_states(model, ids):
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states  # tuple, index 0 = embedding


def test_stream_pinning_invariant():
    """With the muzzle on blocks in the zone, the stream's projection
    onto v is PINNED at its value entering the zone — every muzzled
    block's write is exactly v-free."""
    torch.manual_seed(3)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    try:
        for peft in (True, False):
            model = make_model(peft=peft, seed=1)
            ids = batch()["input_ids"]
            hs_before = hidden_states(model, ids)
            # NOTE output_hidden_states appends the FINAL layer's entry
            # AFTER the model's final RMSNorm, so pinning is checked on
            # non-final layers only (the invariant is on the pre-norm
            # stream; the norm is computation, not a write).
            mgr = ActivationWriteProjectionManager(
                model=model, vector_path=vec_path, layers=[2, 3],
            )
            assert len(mgr._target_modules) == 4
            mgr.register_hooks()
            hs = hidden_states(model, ids)
            # Zone entry: h_1 unchanged by the muzzle
            assert (hs[1] - hs_before[1]).abs().max().item() < 1e-5
            # Within the zone the v-projection is pinned at h_1's value
            p1 = hs[1] @ v_hat
            for l in (2, 3):
                pl = hs[l] @ v_hat
                assert (pl - p1).abs().max().item() < 1e-4, f"layer {l}"
            # Unmuzzled model's projection DOES move across those blocks
            moved = ((hs_before[3] - hs_before[1]) @ v_hat).abs().max().item()
            assert moved > 1e-2
            # Hooks removed -> original computation restored
            mgr.remove_hooks()
            hs_after = hidden_states(model, ids)
            assert (hs_after[3] - hs_before[3]).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_stream_pinning_invariant")


def test_gradients_flow_and_training_progresses():
    """The projection is in the graph: backward works, and a few AdamW
    steps on the muzzled model reduce the loss while the pinning
    invariant keeps holding (frozen-transport channel dead too)."""
    torch.manual_seed(5)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    try:
        model = make_model(peft=True, seed=2)
        mgr = ActivationWriteProjectionManager(
            model=model, vector_path=vec_path, layers=None,  # all blocks
            collect_stats=True,
        )
        mgr.register_hooks()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-3, weight_decay=0.01,
        )
        losses = []
        for i in range(10):
            out = model(**batch(i))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(out.loss.item())
        assert losses[-1] < losses[0]

        stats = mgr.pop_stats()
        assert stats is not None
        assert 0.0 < stats["awp_frac_write_removed"] < 1.0
        assert stats["awp_n_modules"] == 2 * N_LAYERS
        assert mgr.pop_stats() is None

        # All-blocks muzzle: stream projection pinned at the EMBEDDING's
        # value at every (pre-final-norm) layer, even after training
        ids = batch()["input_ids"]
        hs = hidden_states(model, ids)
        p0 = hs[0] @ v_hat
        for l in range(1, N_LAYERS):  # hs[N_LAYERS] is post-final-norm
            assert ((hs[l] @ v_hat) - p0).abs().max().item() < 1e-4, l
    finally:
        os.unlink(vec_path)
    print("PASS: test_gradients_flow_and_training_progresses")


def test_basis_variant_and_validation():
    torch.manual_seed(7)
    d1, d2 = torch.randn(HIDDEN), torch.randn(HIDDEN)
    basis_path = save_vec(torch.stack([d1, d2]))
    v_path = save_vec(d1)
    try:
        model = make_model(peft=False, seed=3)
        mgr = ActivationWriteProjectionManager(
            model=model, vector_path=None, layers=[2],
            basis_path=basis_path,
        )
        V = mgr._dirs[2]
        mgr.register_hooks()
        ids = batch()["input_ids"]
        hs = hidden_states(model, ids)
        # Both directions pinned across the muzzled block
        diff = (hs[2] - hs[1]) @ V.T
        assert diff.abs().max().item() < 1e-4
        mgr.remove_hooks()

        with pytest.raises(ValueError, match="mode must be"):
            ActivationWriteProjectionManager(
                model=model, vector_path=v_path, mode="bogus",
            )
        with pytest.raises(ValueError, match="out of range"):
            ActivationWriteProjectionManager(
                model=model, vector_path=v_path, layers=[99],
            )
        with pytest.raises(ValueError, match="vector_path or basis_path"):
            ActivationWriteProjectionManager(model=model, vector_path=None)
        collinear = save_vec(torch.stack([d1, d1 * 3.0]))
        try:
            with pytest.raises(ValueError, match="linearly dependent"):
                ActivationWriteProjectionManager(
                    model=model, vector_path=None, layers=[1],
                    basis_path=collinear,
                )
        finally:
            os.unlink(collinear)
    finally:
        os.unlink(basis_path)
        os.unlink(v_path)

    from grad_shift.steering import SteeringConfig
    SteeringConfig(
        activation_write_projection="remove",
        grad_manipulation_vector="v.pt",
    )
    c = SteeringConfig(
        activation_write_projection="remove",
        steering_vector_path="sv.pt",
    )
    assert c.grad_manipulation_vector == "sv.pt"
    with pytest.raises(ValueError, match="must be 'remove'"):
        SteeringConfig(
            activation_write_projection="keep",
            grad_manipulation_vector="v.pt",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SteeringConfig(
            activation_write_projection="remove",
            weight_grad_projection="remove",
            grad_manipulation_vector="v.pt",
        )
    with pytest.raises(ValueError, match="requires"):
        SteeringConfig(activation_write_projection="remove")
    print("PASS: test_basis_variant_and_validation")


if __name__ == "__main__":
    test_stream_pinning_invariant()
    test_gradients_flow_and_training_progresses()
    test_basis_variant_and_validation()
    print("\nALL TESTS PASSED")
