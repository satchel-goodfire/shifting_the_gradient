#!/usr/bin/env python3
"""Tests for the scheduled-scale gradient manipulation (sign-forcing modes).

Covers:
  - exp_decay_schedule boundary values and monotonicity
  - scaled pos_sign hook math for both scale_refs ("component", "grad_norm")
  - set_step() driving the scale between backward passes
  - default arguments exactly reproducing the unscaled pos_sign manipulation
  - realized-cosine stats collection (pop_stats)
  - SteeringConfig / GradientManipulationManager validation

Run from project root:
    pytest tests/test_grad_manip_schedule.py -q
    (or: python tests/test_grad_manip_schedule.py)
"""

import math
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grad_shift.finetuning.grad_manipulation import (
    GradientManipulationManager,
    exp_decay_schedule,
)
from grad_shift.steering.config import SteeringConfig
from test_grad_manip_weights import FakeTransformerModel, capture_output_grad


HIDDEN_DIM = 64
NUM_LAYERS = 4
TARGET_LAYER = 3  # embedding-inclusive; block index 2
BATCH, SEQ = 8, 16


def make_fixture(seed=42):
    """Toy model, input/target, saved manipulation vector, baseline grad."""
    torch.manual_seed(seed)
    model = FakeTransformerModel(HIDDEN_DIM, NUM_LAYERS)
    x_data = torch.randn(BATCH, SEQ, HIDDEN_DIM)
    target = torch.randn(BATCH, SEQ, HIDDEN_DIM)

    vec = torch.randn(NUM_LAYERS + 1, HIDDEN_DIM)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(vec, f.name)
    f.close()

    baseline_grad = capture_output_grad(
        model, TARGET_LAYER - 1, x_data, target=target, mgr=None
    )
    return model, x_data, target, f.name, baseline_grad


def make_manager(model, vec_path, **kwargs):
    return GradientManipulationManager(
        model=model,
        vector_path=vec_path,
        mode=kwargs.pop("mode", "pos_sign_vector_component"),
        layers=[TARGET_LAYER],
        **kwargs,
    )


def expected_pos_sign(grad, v, scale, scale_ref):
    proj_scalar = grad @ v
    proj = proj_scalar.unsqueeze(-1) * v
    if scale_ref == "grad_norm":
        magnitude = grad.norm(dim=-1)
    else:
        magnitude = proj_scalar.abs()
    return grad - proj + scale * magnitude.unsqueeze(-1) * v


# ── Schedule function ────────────────────────────────────────────────

def test_exp_decay_schedule_values():
    init, final, tau = 1.0, 0.2, 270.0
    assert exp_decay_schedule(0, init, final, tau) == pytest.approx(init)
    # After one time constant the excess over the plateau has decayed by 1/e
    assert exp_decay_schedule(270, init, final, tau) == pytest.approx(
        final + (init - final) / math.e
    )
    # Effectively at the plateau after many time constants
    assert exp_decay_schedule(20 * 270, init, final, tau) == pytest.approx(
        final, abs=1e-6
    )
    # Monotonically decreasing when init > final
    values = [exp_decay_schedule(t, init, final, tau) for t in range(0, 3500, 100)]
    assert all(a > b for a, b in zip(values, values[1:]))
    print("PASS: test_exp_decay_schedule_values")


# ── Hook math ────────────────────────────────────────────────────────

def test_scaled_pos_sign_component_ref():
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        mgr = make_manager(model, vec_path, scale_init=0.5)
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        want = expected_pos_sign(baseline_grad, v, 0.5, "component")
        assert (got - want).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_scaled_pos_sign_component_ref")


def test_scaled_pos_sign_grad_norm_ref():
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        scale = 0.01
        mgr = make_manager(model, vec_path, scale_init=scale, scale_ref="grad_norm")
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        want = expected_pos_sign(baseline_grad, v, scale, "grad_norm")
        assert (got - want).abs().max().item() < 1e-5

        # For small scale the realized tokenwise cosine along v ~= scale
        cos = (got @ v) / (got.norm(dim=-1) + 1e-12)
        assert cos.mean().item() == pytest.approx(scale, rel=0.05)
    finally:
        os.unlink(vec_path)
    print("PASS: test_scaled_pos_sign_grad_norm_ref")


def test_set_step_drives_scale():
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        init, final, tau = 1.0, 0.2, 270.0
        mgr = make_manager(
            model, vec_path, scale_init=init, scale_final=final, scale_tau=tau
        )
        v = mgr._vectors[TARGET_LAYER]

        mgr.set_step(0)
        assert mgr.current_scale == pytest.approx(init)
        got0 = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        assert (got0 - expected_pos_sign(baseline_grad, v, init, "component")) \
            .abs().max().item() < 1e-5

        late_step = 50 * int(tau)
        mgr.set_step(late_step)
        s_late = exp_decay_schedule(late_step, init, final, tau)
        assert mgr.current_scale == pytest.approx(final, abs=1e-6)
        got_late = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        assert (got_late - expected_pos_sign(baseline_grad, v, s_late, "component")) \
            .abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_set_step_drives_scale")


def test_default_scale_matches_unscaled_pos_sign():
    """Regression: default args must reproduce the original manipulation."""
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        mgr = make_manager(model, vec_path)
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        # Original formula, written out independently of expected_pos_sign
        proj_scalar = baseline_grad @ v
        proj = proj_scalar.unsqueeze(-1) * v
        want = baseline_grad - proj + proj_scalar.abs().unsqueeze(-1) * v
        assert (got - want).abs().max().item() < 1e-6
        # set_step is a no-op without a schedule
        mgr.set_step(10_000)
        assert mgr.current_scale == 1.0
    finally:
        os.unlink(vec_path)
    print("PASS: test_default_scale_matches_unscaled_pos_sign")


def test_neg_sign_scaled():
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        mgr = make_manager(
            model, vec_path, mode="neg_sign_vector_component", scale_init=0.3
        )
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        proj_scalar = baseline_grad @ v
        proj = proj_scalar.unsqueeze(-1) * v
        want = baseline_grad - proj - 0.3 * proj_scalar.abs().unsqueeze(-1) * v
        assert (got - want).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_neg_sign_scaled")


def test_mean_shift_grad_norm_ref():
    """mean_shift adds s*||g||*v without touching the natural component:
    the tokenwise cosine distribution translates by ~s, shape preserved."""
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        s = 0.02
        mgr = make_manager(
            model, vec_path,
            mode="mean_shift_vector_component",
            scale_init=s, scale_ref="grad_norm",
        )
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        want = baseline_grad + s * baseline_grad.norm(dim=-1, keepdim=True) * v
        assert (got - want).abs().max().item() < 1e-5

        cos_pre = (baseline_grad @ v) / (baseline_grad.norm(dim=-1) + 1e-12)
        cos_post = (got @ v) / (got.norm(dim=-1) + 1e-12)
        delta = cos_post - cos_pre
        # Per-token translation by ~s (small-shift approximation)
        assert delta.mean().item() == pytest.approx(s, rel=0.05)
        # Shape preserved: per-token deltas nearly uniform
        assert delta.std().item() < 0.15 * s
    finally:
        os.unlink(vec_path)
    print("PASS: test_mean_shift_grad_norm_ref")


def test_mean_shift_component_ref():
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        s = 0.5
        mgr = make_manager(
            model, vec_path,
            mode="mean_shift_vector_component", scale_init=s,
        )
        v = mgr._vectors[TARGET_LAYER]
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        c = baseline_grad @ v
        want = baseline_grad + s * c.abs().unsqueeze(-1) * v
        assert (got - want).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_mean_shift_component_ref")


def test_scale_by_layer():
    model, x, target, vec_path, _ = make_fixture()
    try:
        factors = {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.04}
        mgr = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="mean_shift_vector_component",
            layers=[1, 2, 3, 4],
            scale_ref="grad_norm", scale_by_layer=factors,
            collect_stats=True,
        )
        capture_output_grad(model, 3, x, target=target, mgr=mgr)
        stats = mgr.pop_stats()
        # Topmost hooked layer (4) receives its natural gradient, so its
        # post - pre must equal that layer's factor
        l = 4
        assert (stats["cos_post_by_layer"][l] - stats["cos_pre_by_layer"][l]
                ) == pytest.approx(factors[l], rel=0.05)

        # Missing factor for an active layer must fail fast
        with pytest.raises(ValueError, match="missing factors"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="mean_shift_vector_component",
                layers=[1, 2], scale_by_layer={1: 0.01},
            )
        # scale_by_layer with a non-scalable mode must fail
        with pytest.raises(ValueError, match="only supported"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="remove_vector_component",
                layers=[1], scale_by_layer={1: 0.01},
            )
    finally:
        os.unlink(vec_path)
    print("PASS: test_scale_by_layer")


def test_act_servo_mode():
    """Closed-loop servo: per token, the along-v component's sign follows
    the ACTIVATION projection error against the target; magnitude and
    orthogonal component match pos/neg_sign semantics."""
    model, x, target, vec_path, baseline_grad = make_fixture()
    try:
        # Choose a target between the min and max per-token activation
        # projections so both control branches are exercised
        probe = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="pos_sign_vector_component", layers=[TARGET_LAYER],
        )
        v = probe._vectors[TARGET_LAYER]
        with torch.no_grad():
            acts = model.model.layers[0](x.clone())
            for blk in model.model.layers[1:TARGET_LAYER]:
                acts = blk(acts)
        act_proj = acts @ v
        a_star = act_proj.median().item()

        mgr = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="act_servo_vector_component", layers=[TARGET_LAYER],
            act_target_by_layer={TARGET_LAYER: a_star},
        )
        got = capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)

        c = baseline_grad @ v
        sgn = torch.sign(act_proj - a_star)
        want = baseline_grad - c.unsqueeze(-1) * v \
            + (sgn * c.abs()).unsqueeze(-1) * v
        assert (got - want).abs().max().item() < 1e-5
        # Both branches actually exercised
        assert (sgn > 0).any() and (sgn < 0).any()
        # Orthogonal component untouched
        got_orth = got - (got @ v).unsqueeze(-1) * v
        base_orth = baseline_grad - c.unsqueeze(-1) * v
        assert (got_orth - base_orth).abs().max().item() < 1e-5

        with pytest.raises(ValueError, match="act_target_by_layer"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[TARGET_LAYER],
            )
        with pytest.raises(ValueError, match="act_target_by_layer"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="pos_sign_vector_component", layers=[TARGET_LAYER],
                act_target_by_layer={TARGET_LAYER: 0.0},
            )
        with pytest.raises(ValueError, match="missing targets"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[1, 2],
                act_target_by_layer={1: 0.0},
            )
        # Empty target dict must fail at init, not at the first backward
        with pytest.raises(ValueError, match="missing targets"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[1],
                act_target_by_layer={},
            )
    finally:
        os.unlink(vec_path)
    print("PASS: test_act_servo_mode")


def test_act_servo_multi_direction_basis():
    """Multi-direction servo: (1) a k=1 basis equal to the manipulation
    vector reproduces the legacy single-vector servo exactly; (2) with a
    non-orthogonal 2-D basis, each ORTHONORMALIZED coordinate follows its
    transformed target, which is exactly equivalent to targeting the
    measured projections along the raw unit directions; the complement of
    span(basis) is untouched."""
    model, x, target, vec_path, baseline_grad = make_fixture()
    paths = []
    try:
        vec = torch.load(vec_path)
        v_raw = vec[TARGET_LAYER].float()
        v = v_raw / v_raw.norm()
        with torch.no_grad():
            acts = model.model.layers[0](x.clone())
            for blk in model.model.layers[1:TARGET_LAYER]:
                acts = blk(acts)
        # Offset the target off the exact median data point: sign(0) at a
        # tied token is representation-sensitive (QR reconstructs the unit
        # direction to within float rounding), which is irrelevant to the
        # semantics under test
        a_star = (acts @ v).median().item() + 1e-3

        # (1) k=1 basis == vector row -> identical to legacy servo
        f1 = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(v_raw.unsqueeze(0), f1.name)
        f1.close()
        paths.append(f1.name)
        legacy = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="act_servo_vector_component", layers=[TARGET_LAYER],
            act_target_by_layer={TARGET_LAYER: a_star},
        )
        got_legacy = capture_output_grad(
            model, TARGET_LAYER - 1, x, target=target, mgr=legacy
        )
        multi1 = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="act_servo_vector_component", layers=[TARGET_LAYER],
            act_target_by_layer={TARGET_LAYER: [a_star]},
            basis_path=f1.name,
        )
        got_multi1 = capture_output_grad(
            model, TARGET_LAYER - 1, x, target=target, mgr=multi1
        )
        assert (got_multi1 - got_legacy).abs().max().item() < 1e-5

        # (2) 2-D deliberately NON-orthogonal raw directions
        torch.manual_seed(3)
        w = torch.randn(HIDDEN_DIM)
        w = w - (w @ v) * v
        d2 = 0.3 * v + w / w.norm()          # cos(d2, v) well off zero
        d2 = d2 / d2.norm()
        raw_dirs = torch.stack([v, d2])
        f2 = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(raw_dirs, f2.name)
        f2.close()
        paths.append(f2.name)
        a2_star = (acts @ d2).median().item() + 1e-3
        mgr = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="act_servo_vector_component", layers=[TARGET_LAYER],
            act_target_by_layer={TARGET_LAYER: [a_star, a2_star]},
            basis_path=f2.name,
        )
        B = mgr._basis[TARGET_LAYER]
        c = mgr._servo_targets[TARGET_LAYER]
        # Target transform is exact: an activation sitting exactly at the
        # raw targets has zero error in orthonormal coordinates
        h_at_target, *_ = torch.linalg.lstsq(
            raw_dirs, torch.tensor([a_star, a2_star])
        )
        assert ((raw_dirs @ h_at_target)
                - torch.tensor([a_star, a2_star])).abs().max() < 1e-4
        assert ((B @ h_at_target) - c).abs().max().item() < 1e-4

        got = capture_output_grad(
            model, TARGET_LAYER - 1, x, target=target, mgr=mgr
        )
        u = acts @ B.T                      # [B,S,2] activation coords
        sgn = torch.sign(u - c)
        G = baseline_grad @ B.T             # [B,S,2] gradient coords
        want = baseline_grad - G @ B + (sgn * G.abs()) @ B
        assert (got - want).abs().max().item() < 1e-5
        # Each orthonormal coordinate's sign follows ITS OWN activation
        # error; both signs vary across tokens (independently tuned)
        got_coords = got @ B.T
        assert (torch.sign(got_coords) * sgn >= 0).all()
        for j in range(2):
            assert (sgn[..., j] > 0).any() and (sgn[..., j] < 0).any()
        # Complement of span(basis) untouched
        got_perp = got - (got @ B.T) @ B
        base_perp = baseline_grad - (baseline_grad @ B.T) @ B
        assert (got_perp - base_perp).abs().max().item() < 1e-5

        # Validation: basis only with subspace-capable modes (mean_shift
        # rejected); list targets need basis; wrong-length lists rejected
        with pytest.raises(ValueError, match="basis_path is only valid"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="mean_shift_vector_component", layers=[TARGET_LAYER],
                basis_path=f2.name, scale_ref="grad_norm",
            )
        with pytest.raises(ValueError, match="require basis_path"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[TARGET_LAYER],
                act_target_by_layer={TARGET_LAYER: [a_star, a2_star]},
            )
        with pytest.raises(ValueError, match="list of 2 targets"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[TARGET_LAYER],
                act_target_by_layer={TARGET_LAYER: [a_star]},
                basis_path=f2.name,
            )
    finally:
        os.unlink(vec_path)
        for p in paths:
            os.unlink(p)
    print("PASS: test_act_servo_multi_direction_basis")


def test_subspace_ablation_modes():
    """remove/keep/flip/pos_sign/neg_sign on a 2-D basis: span component
    handled per mode, complement untouched (except keep), schedule scale
    honored per direction, mean_shift rejected."""
    model, x, target, vec_path, baseline_grad = make_fixture()
    basis_path = None
    try:
        torch.manual_seed(9)
        d1 = torch.randn(HIDDEN_DIM)
        d2 = torch.randn(HIDDEN_DIM) + 0.5 * d1  # non-orthogonal pair
        raw = torch.stack([d1, d2])
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(raw, f.name)
        f.close()
        basis_path = f.name

        probe = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="remove_vector_component", layers=[TARGET_LAYER],
            basis_path=basis_path,
        )
        B = probe._basis[TARGET_LAYER]
        G = baseline_grad @ B.T
        span = G @ B

        cases = {
            "remove_vector_component": (baseline_grad - span, {}),
            "keep_vector_component": (span, {}),
            "flip_vector_component": (baseline_grad - 2 * span, {}),
            "pos_sign_vector_component": (
                baseline_grad - span + (0.4 * G.abs()) @ B,
                {"scale_init": 0.4},
            ),
            "neg_sign_vector_component": (
                baseline_grad - span - (0.4 * G.abs()) @ B,
                {"scale_init": 0.4},
            ),
        }
        for mode, (want, kw) in cases.items():
            mgr = GradientManipulationManager(
                model=model, vector_path=vec_path, mode=mode,
                layers=[TARGET_LAYER], basis_path=basis_path, **kw,
            )
            got = capture_output_grad(
                model, TARGET_LAYER - 1, x, target=target, mgr=mgr
            )
            assert (got - want).abs().max().item() < 1e-5, mode
            if mode != "keep_vector_component":
                got_perp = got - (got @ B.T) @ B
                base_perp = baseline_grad - span
                assert (got_perp - base_perp).abs().max().item() < 1e-5, mode

        # grad_norm ref: enforced magnitude per direction is ||g||
        mgr = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="pos_sign_vector_component", layers=[TARGET_LAYER],
            basis_path=basis_path, scale_init=0.01, scale_ref="grad_norm",
        )
        got = capture_output_grad(
            model, TARGET_LAYER - 1, x, target=target, mgr=mgr
        )
        gnorm = baseline_grad.norm(dim=-1, keepdim=True)
        want = baseline_grad - span + (0.01 * gnorm.expand_as(G)) @ B
        assert (got - want).abs().max().item() < 1e-5

        with pytest.raises(ValueError, match="basis_path is only valid"):
            GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="mean_shift_vector_component", layers=[TARGET_LAYER],
                basis_path=basis_path, scale_ref="grad_norm",
            )
    finally:
        os.unlink(vec_path)
        if basis_path:
            os.unlink(basis_path)
    print("PASS: test_subspace_ablation_modes")


def test_act_servo_per_layer_basis():
    """[num_layers, k, hidden] basis: each hooked layer must servo along
    ITS OWN directions to ITS OWN targets (this is the layers >= 20 arm's
    exact configuration). Verified on two layers with deliberately
    different bases and targets, against per-layer hand-computed
    expectations."""
    model, x, target, vec_path, _ = make_fixture()
    basis_path = None
    try:
        torch.manual_seed(5)
        layers = [2, 3]
        # Distinct 2-D raw directions per layer (embedding-inclusive rows)
        raw = torch.randn(NUM_LAYERS + 1, 2, HIDDEN_DIM)
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(raw, f.name)
        f.close()
        basis_path = f.name

        # Forward activations at each hooked layer for target construction
        acts_by_layer = {}
        with torch.no_grad():
            h = x.clone()
            for i, blk in enumerate(model.model.layers):
                h = blk(h)
                acts_by_layer[i + 1] = h.clone()
        targets = {}
        for l in layers:
            d = raw[l] / raw[l].norm(dim=-1, keepdim=True)
            targets[l] = [
                (acts_by_layer[l] @ d[0]).median().item() + 1e-3,
                (acts_by_layer[l] @ d[1]).median().item() + 1e-3,
            ]

        mgr = GradientManipulationManager(
            model=model, vector_path=vec_path,
            mode="act_servo_vector_component", layers=layers,
            act_target_by_layer=targets,
            basis_path=basis_path,
        )
        # Bases differ across layers (per-layer indexing actually used)
        assert (mgr._basis[2] - mgr._basis[3]).abs().max() > 0.1

        for l in layers:
            # Hook only layer l so its arriving gradient is natural
            solo = GradientManipulationManager(
                model=model, vector_path=vec_path,
                mode="act_servo_vector_component", layers=[l],
                act_target_by_layer={l: targets[l]},
                basis_path=basis_path,
            )
            got = capture_output_grad(model, l - 1, x, target=target, mgr=solo)
            natural = capture_output_grad(model, l - 1, x, target=target, mgr=None)
            B, c = solo._basis[l], solo._servo_targets[l]
            sgn = torch.sign(acts_by_layer[l] @ B.T - c)
            G = natural @ B.T
            want = natural - G @ B + (sgn * G.abs()) @ B
            assert (got - want).abs().max().item() < 1e-5, f"layer {l}"
            # And the layer's own raw targets are what the transform encodes
            d = raw[l] / raw[l].norm(dim=-1, keepdim=True)
            h_at, *_ = torch.linalg.lstsq(d, torch.tensor(targets[l]))
            assert ((B @ h_at) - c).abs().max().item() < 1e-4, f"layer {l}"

        # Missing layer row coverage: a 3-D basis shorter than the layer
        # index must fail at init
        short = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(raw[:3], short.name)  # rows 0..2, layer 3 missing
        short.close()
        try:
            with pytest.raises(ValueError, match="layer index 3"):
                GradientManipulationManager(
                    model=model, vector_path=vec_path,
                    mode="act_servo_vector_component", layers=[2, 3],
                    act_target_by_layer=targets,
                    basis_path=short.name,
                )
        finally:
            os.unlink(short.name)
    finally:
        os.unlink(vec_path)
        if basis_path:
            os.unlink(basis_path)
    print("PASS: test_act_servo_per_layer_basis")


# ── Stats collection ─────────────────────────────────────────────────

def test_pop_stats():
    model, x, target, vec_path, _ = make_fixture()
    try:
        mgr = make_manager(model, vec_path, collect_stats=True)
        assert mgr.pop_stats() is None

        capture_output_grad(model, TARGET_LAYER - 1, x, target=target, mgr=mgr)
        stats = mgr.pop_stats()
        assert stats is not None
        assert stats["n_hook_calls"] == 1
        assert stats["scale"] == 1.0
        # pos_sign forces the component positive: post-cosine must be
        # positive and at least as large as the pre-cosine
        assert stats["cos_post"] > 0
        assert stats["cos_post"] >= stats["cos_pre"]
        # Per-layer means present for the hooked layer and consistent with
        # the overall mean (single layer here)
        assert list(stats["cos_post_by_layer"]) == [TARGET_LAYER]
        assert stats["cos_post_by_layer"][TARGET_LAYER] == pytest.approx(
            stats["cos_post"]
        )
        # Accumulator cleared
        assert mgr.pop_stats() is None
    finally:
        os.unlink(vec_path)
    print("PASS: test_pop_stats")


def test_stats_ignore_zero_grad_tokens():
    """Padding-like tokens (exactly-zero grad) must not dilute the stats.

    The toy blocks mix nothing across positions, so zeroing the loss target
    at a position zeroes that position's gradient at every layer — the same
    signature attention-masked padding produces in a real model.
    """
    model, x, target, vec_path, _ = make_fixture()
    try:
        target_padded = target.clone()
        target_padded[:, : SEQ // 2, :] = 0.0  # half the tokens get zero grad

        scale = 0.01
        mgr = make_manager(
            model, vec_path,
            scale_init=scale, scale_ref="grad_norm", collect_stats=True,
        )
        capture_output_grad(
            model, TARGET_LAYER - 1, x, target=target_padded, mgr=mgr
        )
        stats = mgr.pop_stats()
        # With grad_norm ref, every real token's post-cosine ~= scale; a
        # naive mean over all tokens would be diluted to ~scale/2.
        assert stats["cos_post"] == pytest.approx(scale, rel=0.05)
    finally:
        os.unlink(vec_path)
    print("PASS: test_stats_ignore_zero_grad_tokens")


# ── Validation ───────────────────────────────────────────────────────

def test_manager_validation():
    model, _, _, vec_path, _ = make_fixture()
    try:
        with pytest.raises(ValueError, match="scale_ref"):
            make_manager(model, vec_path, scale_ref="bogus")
        with pytest.raises(ValueError, match="together"):
            make_manager(model, vec_path, scale_final=0.2)
        with pytest.raises(ValueError, match="together"):
            make_manager(model, vec_path, scale_tau=270.0)
        with pytest.raises(ValueError, match="positive"):
            make_manager(model, vec_path, scale_final=0.2, scale_tau=0.0)
        with pytest.raises(ValueError, match="only supported"):
            make_manager(
                model, vec_path,
                mode="remove_vector_component", scale_init=0.5,
            )
        # Non-scalable mode with default scale args is still fine
        make_manager(model, vec_path, mode="remove_vector_component")
    finally:
        os.unlink(vec_path)
    print("PASS: test_manager_validation")


def test_steering_config_validation():
    base = dict(
        grad_manipulation="pos_sign_vector_component",
        grad_manipulation_vector="dummy.pt",
    )
    # Valid schedule
    SteeringConfig(
        **base,
        grad_manipulation_scale_init=1.0,
        grad_manipulation_scale_final=0.2,
        grad_manipulation_scale_tau=270.0,
    )
    # Valid constant scale (plateau-only control)
    SteeringConfig(**base, grad_manipulation_scale_init=0.2)

    with pytest.raises(ValueError, match="scale_ref"):
        SteeringConfig(**base, grad_manipulation_scale_ref="bogus")
    with pytest.raises(ValueError, match="together"):
        SteeringConfig(**base, grad_manipulation_scale_final=0.2)
    with pytest.raises(ValueError, match="positive"):
        SteeringConfig(
            **base,
            grad_manipulation_scale_final=0.2,
            grad_manipulation_scale_tau=-1.0,
        )
    with pytest.raises(ValueError, match="requires"):
        SteeringConfig(
            grad_manipulation="remove_vector_component",
            grad_manipulation_vector="dummy.pt",
            grad_manipulation_scale_init=0.5,
        )
    # act_servo: accepted as a mode, target file required iff servo
    SteeringConfig(
        grad_manipulation="act_servo_vector_component",
        grad_manipulation_vector="dummy.pt",
        grad_manipulation_act_target_file="targets.json",
    )
    with pytest.raises(ValueError, match="act_target_file"):
        SteeringConfig(
            grad_manipulation="act_servo_vector_component",
            grad_manipulation_vector="dummy.pt",
        )
    with pytest.raises(ValueError, match="act_target_file"):
        SteeringConfig(
            grad_manipulation="pos_sign_vector_component",
            grad_manipulation_vector="dummy.pt",
            grad_manipulation_act_target_file="targets.json",
        )
    # basis file: valid with the subspace-capable modes, not mean_shift
    # or bare configs; deprecated servo spelling aliases into it
    SteeringConfig(
        grad_manipulation="act_servo_vector_component",
        grad_manipulation_vector="dummy.pt",
        grad_manipulation_act_target_file="targets.json",
        grad_manipulation_basis_file="basis.pt",
    )
    SteeringConfig(
        grad_manipulation="remove_vector_component",
        grad_manipulation_vector="dummy.pt",
        grad_manipulation_basis_file="basis.pt",
    )
    with pytest.raises(ValueError, match="basis_file requires"):
        SteeringConfig(
            grad_manipulation="mean_shift_vector_component",
            grad_manipulation_vector="dummy.pt",
            grad_manipulation_basis_file="basis.pt",
            grad_manipulation_scale_ref="grad_norm",
        )
    with pytest.raises(ValueError, match="basis_file requires"):
        SteeringConfig(grad_manipulation_basis_file="basis.pt")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        c = SteeringConfig(
            grad_manipulation="act_servo_vector_component",
            grad_manipulation_vector="dummy.pt",
            grad_manipulation_act_target_file="targets.json",
            grad_manipulation_servo_basis_file="old.pt",
        )
    assert c.grad_manipulation_basis_file == "old.pt"
    with pytest.raises(ValueError, match="not both"):
        SteeringConfig(
            grad_manipulation="act_servo_vector_component",
            grad_manipulation_vector="dummy.pt",
            grad_manipulation_act_target_file="targets.json",
            grad_manipulation_basis_file="new.pt",
            grad_manipulation_servo_basis_file="old.pt",
        )
    print("PASS: test_steering_config_validation")


if __name__ == "__main__":
    test_exp_decay_schedule_values()
    test_scaled_pos_sign_component_ref()
    test_scaled_pos_sign_grad_norm_ref()
    test_set_step_drives_scale()
    test_default_scale_matches_unscaled_pos_sign()
    test_neg_sign_scaled()
    test_mean_shift_grad_norm_ref()
    test_mean_shift_component_ref()
    test_scale_by_layer()
    test_act_servo_mode()
    test_act_servo_multi_direction_basis()
    test_subspace_ablation_modes()
    test_act_servo_per_layer_basis()
    test_pop_stats()
    test_stats_ignore_zero_grad_tokens()
    test_steering_config_validation()
    test_manager_validation()
    print("\nALL TESTS PASSED")
