#!/usr/bin/env python3
"""Tests for WeightGradProjectionManager (lora_B grad + weight projection).

Uses a real tiny Qwen2 + PEFT LoRA model so module naming, target
discovery, and the update-level invariant are tested with full fidelity
against the production stack (transformers PEFT integration, AdamW with
weight decay).

Run from project root:
    pytest tests/test_weight_grad_projection.py -q
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.finetuning.weight_grad_projection import (
    WeightGradProjectionManager,
)

HIDDEN = 64
N_LAYERS = 4


def make_peft_model(target_modules=None, seed=0):
    from peft import LoraConfig, get_peft_model
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
    lcfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    return get_peft_model(model, lcfg)


def save_vec(v):
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(v, f.name)
    f.close()
    return f.name


def lm_step_batch(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, 127, (2, 12), generator=g)
    return {"input_ids": ids, "labels": ids.clone()}


def train_steps(model, mgr, n_steps, project_weights=True, lr=5e-3):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    losses = []
    for i in range(n_steps):
        out = model(**lm_step_batch(i))
        out.loss.backward()
        if mgr is not None:
            mgr.project_grads()
        opt.step()
        if mgr is not None and project_weights:
            mgr.project_weights()
        opt.zero_grad()
        losses.append(out.loss.item())
    return losses


def test_target_discovery_and_selectivity():
    model = make_peft_model()
    v = torch.randn(HIDDEN)
    vec_path = save_vec(v)
    try:
        mgr = WeightGradProjectionManager(
            model=model, vector_path=vec_path, layers=[3, 4],
        )
        # Exactly o_proj + down_proj lora_B per active block
        assert len(mgr._targets) == 4
        for l, name, p in mgr._targets:
            assert l in (3, 4)
            assert "lora_B" in name and name.endswith(".weight")
            assert "o_proj" in name or "down_proj" in name
            assert p.shape[0] == HIDDEN
        # All-layers default
        mgr_all = WeightGradProjectionManager(
            model=model, vector_path=vec_path,
        )
        assert len(mgr_all._targets) == 2 * N_LAYERS
    finally:
        os.unlink(vec_path)
    print("PASS: test_target_discovery_and_selectivity")


def test_invariant_holds_under_real_adamw_training():
    """After real AdamW steps (weight decay 0.01), targeted lora_B
    weights must satisfy ||v @ B|| == 0 exactly (float eps), the merged
    LoRA delta must have zero output-row component along v, and training
    must still make progress; untargeted layers' B must NOT be
    v-orthogonal (selectivity)."""
    model = make_peft_model(seed=1)
    torch.manual_seed(7)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    try:
        mgr = WeightGradProjectionManager(
            model=model, vector_path=vec_path, layers=[3, 4],
            collect_stats=True,
        )
        losses = train_steps(model, mgr, n_steps=12)
        assert losses[-1] < losses[0]  # training progressed

        for l, name, p in mgr._targets:
            resid = (v_hat.to(p.dtype) @ p.data).norm().item()
            assert resid < 1e-6, (name, resid)

        # Merged delta for a targeted module never writes into v
        block = model.base_model.model.model.layers[2]  # layer index 3
        o = block.self_attn.o_proj
        delta = o.lora_B["default"].weight @ o.lora_A["default"].weight
        assert (v_hat @ delta).norm().item() < 1e-6

        # Selectivity: an UNCONSTRAINED block's o_proj B has a v component
        blk1 = model.base_model.model.model.layers[0]  # layer index 1
        b1 = blk1.self_attn.o_proj.lora_B["default"].weight
        assert (v_hat @ b1).norm().item() > 1e-4

        # Grad projection actually zeroes the v-row of target grads
        out = model(**lm_step_batch(99))
        out.loss.backward()
        mgr.project_grads()
        for l, name, p in mgr._targets:
            assert (v_hat @ p.grad).norm().item() < 1e-6, name
        model.zero_grad()

        stats = mgr.pop_stats()
        assert stats is not None
        assert stats["wproj_n_params"] == 4
        assert stats["wproj_frac_grad_removed"] > 0
        assert stats["wproj_b_resid_post"] < 1e-6
    finally:
        os.unlink(vec_path)
    print("PASS: test_invariant_holds_under_real_adamw_training")


def test_weight_reprojection_is_load_bearing():
    """Gradient projection alone leaks through AdamW's elementwise moment
    rescaling; the post-step weight re-projection is what makes the
    invariant exact."""
    torch.manual_seed(7)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    try:
        model_a = make_peft_model(seed=2)
        mgr_a = WeightGradProjectionManager(
            model=model_a, vector_path=vec_path, layers=[3, 4],
        )
        train_steps(model_a, mgr_a, n_steps=12, project_weights=False)
        leak_without = max(
            (v_hat @ p.data).norm().item() for _, _, p in mgr_a._targets
        )

        model_b = make_peft_model(seed=2)
        mgr_b = WeightGradProjectionManager(
            model=model_b, vector_path=vec_path, layers=[3, 4],
        )
        train_steps(model_b, mgr_b, n_steps=12, project_weights=True)
        leak_with = max(
            (v_hat @ p.data).norm().item() for _, _, p in mgr_b._targets
        )
        assert leak_with < 1e-6
        assert leak_without > 10 * max(leak_with, 1e-9), (
            leak_without, leak_with
        )
    finally:
        os.unlink(vec_path)
    print("PASS: test_weight_reprojection_is_load_bearing")


def test_invariant_under_gradient_accumulation():
    """The production config accumulates 8 microbatches per optimizer
    step and the trainer projects only on sync steps: accumulated grads
    projected ONCE pre-step must satisfy the same invariant, and by
    linearity must equal per-microbatch projection."""
    torch.manual_seed(13)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    accum = 4
    try:
        model = make_peft_model(seed=5)
        mgr = WeightGradProjectionManager(
            model=model, vector_path=vec_path, layers=[3, 4],
        )
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-3, weight_decay=0.01,
        )
        for step in range(3):
            for micro in range(accum):
                out = model(**lm_step_batch(step * accum + micro))
                (out.loss / accum).backward()
            # project once on the sync step, as the trainer does
            mgr.project_grads()
            for _, name, p in mgr._targets:
                assert (v_hat @ p.grad).norm().item() < 1e-6, name
            opt.step()
            mgr.project_weights()
            opt.zero_grad()
        for _, name, p in mgr._targets:
            assert (v_hat @ p.data).norm().item() < 1e-6, name

        # Linearity: projecting the accumulated grad == accumulating
        # per-microbatch-projected grads
        model2 = make_peft_model(seed=5)
        mgr2 = WeightGradProjectionManager(
            model=model2, vector_path=vec_path, layers=[3, 4],
        )
        for micro in range(accum):
            out = model2(**lm_step_batch(micro))
            (out.loss / accum).backward()
            mgr2.project_grads()
        model1 = make_peft_model(seed=5)
        mgr1 = WeightGradProjectionManager(
            model=model1, vector_path=vec_path, layers=[3, 4],
        )
        for micro in range(accum):
            out = model1(**lm_step_batch(micro))
            (out.loss / accum).backward()
        mgr1.project_grads()
        g1 = {n: p.grad.clone() for n, p in model1.named_parameters()
              if p.requires_grad and p.grad is not None}
        for n, p in model2.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert (p.grad - g1[n]).abs().max().item() < 1e-5, n
    finally:
        os.unlink(vec_path)
    print("PASS: test_invariant_under_gradient_accumulation")


def test_resume_preserves_inherited_component():
    """Resuming with nonzero lora_B (peft_checkpoint semantics): the
    inherited v-component must be PRESERVED, not one-shot-ablated; only
    new learning is constrained."""
    torch.manual_seed(17)
    v = torch.randn(HIDDEN)
    v_hat = v / v.norm()
    vec_path = save_vec(v)
    try:
        model = make_peft_model(seed=6)
        # Simulate a resumed checkpoint: give targeted lora_B weights a
        # deliberate nonzero v-component before the manager exists
        with torch.no_grad():
            blk = model.base_model.model.model.layers[3]  # layer index 4
            b = blk.self_attn.o_proj.lora_B["default"].weight
            b += 0.05 * v_hat.unsqueeze(1) * torch.randn(1, b.shape[1])
        c0_true = (v_hat @ b.data).clone()
        assert c0_true.norm() > 1e-3

        mgr = WeightGradProjectionManager(
            model=model, vector_path=vec_path, layers=[3, 4],
        )
        train_steps(model, mgr, n_steps=8)
        c_after = v_hat @ b.data
        # Inherited component intact to float precision, not ablated
        assert (c_after - c0_true).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_resume_preserves_inherited_component")


def test_basis_and_per_layer_conventions():
    model = make_peft_model(seed=3)
    torch.manual_seed(11)
    # Non-orthogonal 2-D raw basis
    d1 = torch.randn(HIDDEN)
    d2 = torch.randn(HIDDEN) + 0.4 * d1
    basis_path = save_vec(torch.stack([d1, d2]))
    # Per-layer vector file [num_layers+1, hidden]
    per_layer = torch.randn(N_LAYERS + 1, HIDDEN)
    pl_path = save_vec(per_layer)
    try:
        mgr = WeightGradProjectionManager(
            model=model, vector_path=None, layers=[2],
            basis_path=basis_path,
        )
        V = mgr._dirs[2]
        assert V.shape == (2, HIDDEN)
        assert (V @ V.T - torch.eye(2)).abs().max() < 1e-5
        train_steps(model, mgr, n_steps=6)
        for _, name, p in mgr._targets:
            assert (V @ p.data).norm().item() < 1e-6, name

        mgr_pl = WeightGradProjectionManager(
            model=model, vector_path=pl_path, layers=[2, 3],
        )
        for l in (2, 3):
            want = per_layer[l] / per_layer[l].norm()
            assert (mgr_pl._dirs[l][0] - want).abs().max() < 1e-6
    finally:
        os.unlink(basis_path)
        os.unlink(pl_path)
    print("PASS: test_basis_and_per_layer_conventions")


def test_validation():
    model = make_peft_model(seed=4)
    v_path = save_vec(torch.randn(HIDDEN))
    try:
        with pytest.raises(ValueError, match="mode must be"):
            WeightGradProjectionManager(
                model=model, vector_path=v_path, mode="bogus",
            )
        with pytest.raises(ValueError, match="out of range"):
            WeightGradProjectionManager(
                model=model, vector_path=v_path, layers=[99],
            )
        # LoRA without residual-writing targets -> fail fast
        model_qv = make_peft_model(target_modules=["q_proj", "v_proj"])
        with pytest.raises(ValueError, match="No trainable lora_B"):
            WeightGradProjectionManager(
                model=model_qv, vector_path=v_path, layers=[1],
            )
        # (Near-)collinear basis rows -> fail fast, never project an
        # arbitrary QR-surplus direction
        d = torch.randn(HIDDEN)
        collinear = save_vec(torch.stack([d, d * 2.0 + 1e-6]))
        try:
            with pytest.raises(ValueError, match="linearly dependent"):
                WeightGradProjectionManager(
                    model=model, vector_path=None, layers=[1],
                    basis_path=collinear,
                )
        finally:
            os.unlink(collinear)
    finally:
        os.unlink(v_path)

    from grad_shift.steering import SteeringConfig
    SteeringConfig(
        weight_grad_projection="remove",
        grad_manipulation_vector="v.pt",
    )
    # Vector defaults from steering path
    c = SteeringConfig(
        weight_grad_projection="remove",
        steering_vector_path="sv.pt",
    )
    assert c.grad_manipulation_vector == "sv.pt"
    # Basis alone is enough
    SteeringConfig(
        weight_grad_projection="remove",
        grad_manipulation_basis_file="b.pt",
    )
    with pytest.raises(ValueError, match="must be 'remove'"):
        SteeringConfig(
            weight_grad_projection="keep",
            grad_manipulation_vector="v.pt",
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SteeringConfig(
            weight_grad_projection="remove",
            grad_manipulation="pos_sign_vector_component",
            grad_manipulation_vector="v.pt",
        )
    with pytest.raises(ValueError, match="requires grad_manipulation_vector"):
        SteeringConfig(weight_grad_projection="remove")
    print("PASS: test_validation")


if __name__ == "__main__":
    test_target_discovery_and_selectivity()
    test_invariant_holds_under_real_adamw_training()
    test_weight_reprojection_is_load_bearing()
    test_invariant_under_gradient_accumulation()
    test_resume_preserves_inherited_component()
    test_basis_and_per_layer_conventions()
    test_validation()
    print("\nALL TESTS PASSED")
