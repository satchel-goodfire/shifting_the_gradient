#!/usr/bin/env python3
"""Tests for GradientTransplantManager (donor-delta capture + replacement).

Donor and recipient conditions are emulated with different inputs to the
same toy model — the manager doesn't care *why* the passes differ (steering
vs input), only that the donor's delta field is captured and substituted
into the recipient's backward.

Run from project root:
    pytest tests/test_grad_transplant.py -q
    (or: python tests/test_grad_transplant.py)
"""

import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grad_shift.finetuning.grad_transplant import (
    GradientTransplantManager,
    VALID_TRANSPLANT_MODES,
)
from test_grad_manip_weights import FakeTransformerModel, capture_output_grad


HIDDEN_DIM = 64
NUM_LAYERS = 4
BATCH, SEQ = 4, 8


def make_fixture(seed=7):
    torch.manual_seed(seed)
    model = FakeTransformerModel(HIDDEN_DIM, NUM_LAYERS)
    x_donor = torch.randn(BATCH, SEQ, HIDDEN_DIM)
    x_recip = torch.randn(BATCH, SEQ, HIDDEN_DIM)
    target = torch.randn(BATCH, SEQ, HIDDEN_DIM)

    vec = torch.randn(NUM_LAYERS + 1, HIDDEN_DIM)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(vec, f.name)
    f.close()
    return model, x_donor, x_recip, target, f.name


def donor_deltas_reference(model, x, target, layers):
    """Clean per-layer output gradients for a given input (no manager)."""
    return {
        l: capture_output_grad(model, l - 1, x, target=target, mgr=None)
        for l in layers
    }


def run_transplant(model, mgr, x_donor, x_recip, target):
    """Full donor-capture + recipient-replace cycle; returns recipient
    param grads left on the model."""
    mgr.begin_capture()
    xd = x_donor.clone().requires_grad_(True)
    donor_loss = (model(xd) * target).sum()
    mgr.end_capture(donor_loss)

    model.zero_grad()
    mgr.register_replace_hooks()
    xr = x_recip.clone().requires_grad_(True)
    loss = (model(xr) * target).sum()
    loss.backward()
    mgr.remove_hooks()
    mgr.clear_donor()


def capture_recipient_output_grad(model, mgr, x_donor, x_recip, target, block_idx):
    """Recipient's post-replacement gradient at a block output."""
    mgr.begin_capture()
    xd = x_donor.clone().requires_grad_(True)
    donor_loss = (model(xd) * target).sum()
    mgr.end_capture(donor_loss)

    captured = {}
    block = model.model.layers[block_idx]

    mgr.register_replace_hooks()

    def fwd_hook(module, inp, output):
        tensor = output[0] if isinstance(output, tuple) else output

        def save(grad):
            captured["grad"] = grad.clone()

        tensor.register_hook(save)

    handle = block.register_forward_hook(fwd_hook)
    model.zero_grad()
    xr = x_recip.clone().requires_grad_(True)
    loss = (model(xr) * target).sum()
    loss.backward()
    handle.remove()
    mgr.remove_hooks()
    mgr.clear_donor()
    return captured["grad"]


def test_delta_full_replaces_with_donor_field():
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        layers = [1, 2, 3, 4]
        ref = donor_deltas_reference(model, x_donor, target, layers)
        mgr = GradientTransplantManager(
            model=model, mode="delta_full", vector_path=vec_path, layers=layers,
        )
        # The last block's hook fires first in backward and its replacement
        # then propagates — but every hooked layer's outgoing grad must
        # equal the donor's delta there.
        for l in layers:
            got = capture_recipient_output_grad(
                model, mgr, x_donor, x_recip, target, l - 1
            )
            assert (got - ref[l]).abs().max().item() < 1e-5, f"layer {l}"
    finally:
        os.unlink(vec_path)
    print("PASS: test_delta_full_replaces_with_donor_field")


def test_delta_along_and_orthog_mixing():
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        # Single hooked layer so the recipient's arriving gradient is its
        # natural one — the mixing formula can be checked exactly.
        layer = 3
        donor = donor_deltas_reference(model, x_donor, target, [layer])[layer]
        natural = capture_output_grad(
            model, layer - 1, x_recip, target=target, mgr=None
        )

        for mode in ("delta_along", "delta_orthog"):
            mgr = GradientTransplantManager(
                model=model, mode=mode, vector_path=vec_path, layers=[layer],
            )
            v = mgr._vectors[layer]
            got = capture_recipient_output_grad(
                model, mgr, x_donor, x_recip, target, layer - 1
            )
            diff = donor - natural
            proj = (diff @ v).unsqueeze(-1) * v
            want = natural + (proj if mode == "delta_along" else diff - proj)
            assert (got - want).abs().max().item() < 1e-5, mode
            # The two modes partition the shift: along + orthog - natural = donor
        got_along = capture_recipient_output_grad(
            model,
            GradientTransplantManager(
                model=model, mode="delta_along", vector_path=vec_path, layers=[layer]
            ),
            x_donor, x_recip, target, layer - 1,
        )
        got_orthog = capture_recipient_output_grad(
            model,
            GradientTransplantManager(
                model=model, mode="delta_orthog", vector_path=vec_path, layers=[layer]
            ),
            x_donor, x_recip, target, layer - 1,
        )
        recomposed = got_along + got_orthog - natural
        assert (recomposed - donor).abs().max().item() < 1e-5
    finally:
        os.unlink(vec_path)
    print("PASS: test_delta_along_and_orthog_mixing")


def test_activation_along_swaps_only_the_vector_component():
    """activation_along (donor = unsteered, recipient = steered): the
    recipient's gradient keeps its own orthogonal part and takes the
    donor's component along v, i.e. the steered pass trains with the
    unsteered pass's push along the trait vector (experiments/pv_swap)."""
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        layer = 3
        donor = donor_deltas_reference(model, x_donor, target, [layer])[layer]
        natural = capture_output_grad(
            model, layer - 1, x_recip, target=target, mgr=None
        )
        mgr = GradientTransplantManager(
            model=model, mode="activation_along", vector_path=vec_path,
            layers=[layer],
        )
        assert not mgr.donor_steered
        v = mgr._vectors[layer]
        got = capture_recipient_output_grad(
            model, mgr, x_donor, x_recip, target, layer - 1
        )
        assert (got @ v - donor @ v).abs().max().item() < 1e-5
        orth = lambda g: g - (g @ v).unsqueeze(-1) * v
        assert (orth(got) - orth(natural)).abs().max().item() < 1e-5
        # The swap is not a no-op on this fixture
        assert (got - natural).abs().max().item() > 1e-3

        with pytest.raises(ValueError, match="requires a vector_path"):
            GradientTransplantManager(model=model, mode="activation_along")
    finally:
        os.unlink(vec_path)
    print("PASS: test_activation_along_swaps_only_the_vector_component")


def test_subspace_basis_along_and_orthog():
    """A [k, hidden] basis makes the along/orthog split project onto
    span(basis); the split must still partition the donor-recipient shift,
    and non-orthonormal input rows must be orthonormalized."""
    model, x_donor, x_recip, target, vec_path = make_fixture()
    basis_path = None
    try:
        layer = 3
        donor = donor_deltas_reference(model, x_donor, target, [layer])[layer]
        natural = capture_output_grad(
            model, layer - 1, x_recip, target=target, mgr=None
        )

        # Deliberately non-orthonormal, non-unit rows spanning a 2-D subspace
        torch.manual_seed(11)
        a = torch.randn(HIDDEN_DIM) * 3.0
        b = torch.randn(HIDDEN_DIM) * 0.5 + 0.7 * a
        raw = torch.stack([a, b])
        f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(raw, f.name)
        f.close()
        basis_path = f.name

        results = {}
        for mode in ("delta_along", "delta_orthog"):
            mgr = GradientTransplantManager(
                model=model, mode=mode, vector_path=vec_path,
                layers=[layer], subspace_basis_path=basis_path,
            )
            bmat = mgr._basis
            # Orthonormal rows spanning the same subspace as the raw rows
            assert (bmat @ bmat.T - torch.eye(2)).abs().max().item() < 1e-5
            for row in raw:
                resid = row - (row @ bmat.T) @ bmat
                assert resid.norm().item() / row.norm().item() < 1e-5

            got = capture_recipient_output_grad(
                model, mgr, x_donor, x_recip, target, layer - 1
            )
            diff = donor - natural
            proj = (diff @ bmat.T) @ bmat
            want = natural + (proj if mode == "delta_along" else diff - proj)
            assert (got - want).abs().max().item() < 1e-5, mode
            results[mode] = got

        # The two modes still partition the shift under the subspace split
        recomposed = results["delta_along"] + results["delta_orthog"] - natural
        assert (recomposed - donor).abs().max().item() < 1e-5

        # Validation: basis only valid with the split modes; bad shapes fail
        with pytest.raises(ValueError, match="only valid with delta_along"):
            GradientTransplantManager(
                model=model, mode="delta_full", vector_path=vec_path,
                subspace_basis_path=basis_path,
            )
        bad = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(torch.randn(HIDDEN_DIM), bad.name)
        bad.close()
        try:
            with pytest.raises(ValueError, match=r"basis must be \[k, hidden\]"):
                GradientTransplantManager(
                    model=model, mode="delta_along", vector_path=vec_path,
                    subspace_basis_path=bad.name,
                )
        finally:
            os.unlink(bad.name)
    finally:
        os.unlink(vec_path)
        if basis_path:
            os.unlink(basis_path)
    print("PASS: test_subspace_basis_along_and_orthog")


def test_param_grads_come_from_transplanted_field():
    """delta_full with donor == recipient input must equal a plain backward;
    with a different donor the param grads must change."""
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        layers = [1, 2, 3, 4]
        mgr = GradientTransplantManager(
            model=model, mode="delta_full", vector_path=vec_path, layers=layers,
        )

        # Plain backward on recipient input
        model.zero_grad()
        x = x_recip.clone().requires_grad_(True)
        ((model(x) * target).sum()).backward()
        plain = {
            n: p.grad.clone() for n, p in model.named_parameters()
        }

        # Identity transplant: donor input == recipient input
        run_transplant(model, mgr, x_recip, x_recip, target)
        for n, p in model.named_parameters():
            assert (p.grad - plain[n]).abs().max().item() < 1e-5, n

        # Real transplant: different donor -> different param grads
        run_transplant(model, mgr, x_donor, x_recip, target)
        changed = any(
            (p.grad - plain[n]).abs().max().item() > 1e-4
            for n, p in model.named_parameters()
        )
        assert changed
    finally:
        os.unlink(vec_path)
    print("PASS: test_param_grads_come_from_transplanted_field")


def test_accumulation_loss_scale_contract():
    """Regression for the 2026-08-11 P0: under gradient accumulation the
    recipient backward runs on loss/accum while donor deltas came from
    the UNSCALED loss, injecting the transplanted components at accum-x
    magnitude. end_capture(loss, loss_scale=1/accum) must produce donor
    deltas at the recipient's scale: an identity transplant (donor input
    == recipient input) with matching scales must reproduce the plain
    scaled backward exactly; the legacy unscaled call must NOT."""
    model, x_donor, x_recip, target, vec_path = make_fixture()
    accum = 8.0
    try:
        layers = [1, 2, 3, 4]
        mgr = GradientTransplantManager(
            model=model, mode="delta_full", vector_path=vec_path, layers=layers,
        )

        # Plain backward at the accumulation-scaled loss (what
        # Accelerator.backward does per microbatch)
        model.zero_grad()
        x = x_recip.clone().requires_grad_(True)
        ((model(x) * target).sum() / accum).backward()
        plain = {n: p.grad.clone() for n, p in model.named_parameters()}

        def run(loss_scale):
            model.zero_grad()
            mgr.begin_capture()
            xd = x_recip.clone().requires_grad_(True)
            donor_loss = (model(xd) * target).sum()  # UNSCALED donor loss
            mgr.end_capture(donor_loss, loss_scale=loss_scale)
            mgr.register_replace_hooks()
            xr = x_recip.clone().requires_grad_(True)
            ((model(xr) * target).sum() / accum).backward()
            mgr.remove_hooks()
            mgr.clear_donor()
            return {n: p.grad.clone() for n, p in model.named_parameters()}

        # Correct contract: donor scaled to match the recipient
        good = run(loss_scale=1.0 / accum)
        for n in plain:
            assert (good[n] - plain[n]).abs().max().item() < 1e-6, n

        # Legacy behavior (unscaled donor): grads inflated ~accum-x
        bad = run(loss_scale=1.0)
        worst = max(
            (bad[n] - plain[n]).abs().max().item() for n in plain
        )
        assert worst > 1e-3, "unscaled donor should NOT match — did the "\
            "recipient scaling change?"
    finally:
        os.unlink(vec_path)
    print("PASS: test_accumulation_loss_scale_contract")


def test_donor_pass_leaves_no_param_grads():
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        mgr = GradientTransplantManager(
            model=model, mode="delta_full", vector_path=vec_path,
        )
        model.zero_grad()
        mgr.begin_capture()
        xd = x_donor.clone().requires_grad_(True)
        donor_loss = (model(xd) * target).sum()
        mgr.end_capture(donor_loss)
        for n, p in model.named_parameters():
            assert p.grad is None or p.grad.abs().max().item() == 0.0, n
        mgr.clear_donor()
    finally:
        os.unlink(vec_path)
    print("PASS: test_donor_pass_leaves_no_param_grads")


def test_stats_and_validation():
    model, x_donor, x_recip, target, vec_path = make_fixture()
    try:
        mgr = GradientTransplantManager(
            model=model, mode="delta_full", vector_path=vec_path,
            collect_stats=True,
        )
        assert mgr.pop_stats() is None
        run_transplant(model, mgr, x_donor, x_recip, target)
        stats = mgr.pop_stats()
        assert stats is not None and stats["n_hook_calls"] == NUM_LAYERS
        assert mgr.pop_stats() is None

        assert mgr.donor_steered  # delta_* modes: donor is the steered pass
        assert not GradientTransplantManager(
            model=model, mode="activation", vector_path=vec_path
        ).donor_steered

        with pytest.raises(ValueError, match="mode must be one of"):
            GradientTransplantManager(model=model, mode="bogus")
        with pytest.raises(ValueError, match="requires a vector_path"):
            GradientTransplantManager(model=model, mode="delta_along")
        with pytest.raises(RuntimeError, match="No donor deltas"):
            GradientTransplantManager(
                model=model, mode="delta_full"
            ).register_replace_hooks()

        # Incomplete donor capture must fail with the diagnostic message,
        # not a bare KeyError
        mgr2 = GradientTransplantManager(model=model, mode="delta_full")
        mgr2.begin_capture()
        xd = x_donor.clone().requires_grad_(True)
        out = model.model.layers[0](xd)
        mgr2.remove_hooks()
        mgr2._captured_outputs = {1: out}
        with pytest.raises(RuntimeError, match="capture incomplete"):
            mgr2.end_capture(out.sum())
        # clear_donor also drops captured outputs (error-path hygiene)
        mgr2.clear_donor()
        assert not mgr2._captured_outputs and not mgr2._donor_deltas
    finally:
        os.unlink(vec_path)
    print("PASS: test_stats_and_validation")


def test_steering_config_transplant_validation():
    from grad_shift.steering import SteeringConfig

    SteeringConfig(steering_vector_path="v.pt", grad_transplant="delta_full")
    cfg = SteeringConfig(steering_vector_path="v.pt",
                         grad_transplant="activation_along")
    assert cfg.grad_manipulation_vector == "v.pt"  # split uses the PPS vector
    with pytest.raises(ValueError, match="grad_transplant must be"):
        SteeringConfig(steering_vector_path="v.pt", grad_transplant="bogus")
    with pytest.raises(ValueError, match="mutually exclusive"):
        SteeringConfig(
            steering_vector_path="v.pt",
            grad_transplant="delta_full",
            grad_manipulation="pos_sign_vector_component",
        )
    with pytest.raises(ValueError, match="requires steering_vector_path"):
        SteeringConfig(grad_transplant="delta_full")
    SteeringConfig(
        steering_vector_path="v.pt",
        grad_transplant="delta_along",
        grad_transplant_basis_file="b.pt",
    )
    with pytest.raises(ValueError, match="basis_file is only valid"):
        SteeringConfig(
            steering_vector_path="v.pt",
            grad_transplant="delta_full",
            grad_transplant_basis_file="b.pt",
        )
    with pytest.raises(ValueError, match="basis_file is only valid"):
        SteeringConfig(
            steering_vector_path="v.pt",
            grad_transplant_basis_file="b.pt",
        )
    print("PASS: test_steering_config_transplant_validation")


if __name__ == "__main__":
    test_delta_full_replaces_with_donor_field()
    test_delta_along_and_orthog_mixing()
    test_activation_along_swaps_only_the_vector_component()
    test_subspace_basis_along_and_orthog()
    test_accumulation_loss_scale_contract()
    test_param_grads_come_from_transplanted_field()
    test_donor_pass_leaves_no_param_grads()
    test_stats_and_validation()
    test_steering_config_transplant_validation()
    print("\nALL TESTS PASSED")
