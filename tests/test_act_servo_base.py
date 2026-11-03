"""Tests for the per-token base-matched servo (act_servo_base_vector_component).

The scalar act_servo pins every token's evil-axis projection toward ONE
per-layer target, so only the mean matches base. This mode pins each token
toward its OWN base-model projection on the same input (adapter-disabled
no-grad forward, same step). Tested on a tiny random-init Qwen2 + real peft
LoRA so the ``disable_adapter`` path is the one production uses.

Ported from the earlier study that first used this mode (2026-09-29).
"""

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from grad_shift.finetuning.grad_manipulation import (  # noqa: E402
    GradientManipulationManager,
)
from grad_shift.steering.config import SteeringConfig  # noqa: E402

HIDDEN, N_LAYERS, VOCAB = 32, 3, 97
MODE = "act_servo_base_vector_component"


def _tiny_peft(seed=0, lora_dropout=0.05):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
    )
    base = Qwen2ForCausalLM(cfg).float()
    model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.train()  # production trains with LoRA dropout active
    return model


def _perturb_lora(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))


def _vec_file():
    torch.manual_seed(123)
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(torch.randn(N_LAYERS + 1, HIDDEN), f.name)
    f.close()
    return f.name


def _batch(seed=5, b=3, s=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (b, s), generator=g)
    mask = torch.ones(b, s, dtype=torch.long)
    mask[0, :2] = 0  # left padding on one row
    return ids, mask


def _blocks(model):
    return model.get_base_model().model.layers


def _run(model, mgr, ids, mask, layers):
    """Training forward+backward with capture hooks around the manager's.

    Returns per layer: activation h, raw incoming grad, manipulated grad.
    Tensor hooks run in registration order, so a capture hook registered
    BEFORE the manager's sees the raw grad and one AFTER sees the result.
    """
    got = {l: {} for l in layers}
    handles = []

    def mk(l, key):
        def fwd(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if key == "pre":
                got[l]["h"] = t.detach().clone()

            def gh(g):
                got[l][key] = g.detach().clone()
                return g
            t.register_hook(gh)
        return fwd

    for l in layers:
        handles.append(_blocks(model)[l - 1].register_forward_hook(mk(l, "pre")))
    if mgr is not None:
        mgr.register_hooks()
    for l in layers:
        handles.append(_blocks(model)[l - 1].register_forward_hook(mk(l, "post")))
    try:
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
    finally:
        for h in handles:
            h.remove()
        if mgr is not None:
            mgr.remove_hooks()
    model.zero_grad()
    return got


def _base_proj(model, ids, mask, l, v):
    """Independent reference: adapter-disabled forward, projection at l."""
    cap = {}

    def fwd(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    h = _blocks(model)[l - 1].register_forward_hook(fwd)
    try:
        with torch.no_grad(), model.disable_adapter():
            model(input_ids=ids, attention_mask=mask)
    finally:
        h.remove()
    return cap["h"] @ v


def _mgr(model, vec, layers):
    return GradientManipulationManager(
        model=model.get_base_model(), vector_path=vec, mode=MODE,
        layers=layers, collect_stats=True)


def _expected(g, h, base, v):
    c = g @ v
    sgn = torch.sign(h @ v - base)
    return g - c.unsqueeze(-1) * v + (sgn * c.abs()).unsqueeze(-1) * v, sgn


def test_zero_adapter_pins_exactly_and_zeroes_along_v_component():
    # LoRA B = 0 at init: base and training activations are identical, so
    # every token is exactly at its target -> sign 0 -> along-v grad removed
    model, vec = _tiny_peft(), _vec_file()
    try:
        ids, mask = _batch()
        mgr = _mgr(model, vec, [2])
        v = mgr._vectors[2]
        mgr.capture_base_reference(model, ids, mask)
        got = _run(model, mgr, ids, mask, [2])[2]
        assert torch.equal(got["h"] @ v, _base_proj(model, ids, mask, 2, v))
        g = got["pre"]
        want = g - (g @ v).unsqueeze(-1) * v
        assert torch.allclose(got["post"], want, atol=1e-6)
        assert (got["post"] @ v).abs().max() < 1e-5
        assert (g @ v).abs().max() > 1e-4  # there was a component to remove
    finally:
        os.unlink(vec)


def test_per_token_sign_follows_own_base_projection():
    model, vec = _tiny_peft(), _vec_file()
    try:
        _perturb_lora(model)
        ids, mask = _batch()
        mgr = _mgr(model, vec, [2])
        v = mgr._vectors[2]
        mgr.capture_base_reference(model, ids, mask)
        got = _run(model, mgr, ids, mask, [2])[2]
        base = _base_proj(model, ids, mask, 2, v)
        want, sgn = _expected(got["pre"], got["h"], base, v)
        assert torch.allclose(got["post"], want, atol=1e-5)
        assert (sgn > 0).any() and (sgn < 0).any()
        # Orthogonal component untouched
        orth = lambda x: x - (x @ v).unsqueeze(-1) * v  # noqa: E731
        assert torch.allclose(orth(got["post"]), orth(got["pre"]), atol=1e-5)
        # Stats: per-layer mean |h.v - base.v| over real tokens
        stats = mgr.pop_stats()
        err = (got["h"] @ v - base).abs()[mask.bool()].mean().item()
        assert stats["servo_abs_err_by_layer"][2] == pytest.approx(err, rel=1e-4)
    finally:
        os.unlink(vec)


def test_multi_layer_targets_are_per_layer():
    model, vec = _tiny_peft(), _vec_file()
    try:
        _perturb_lora(model)
        ids, mask = _batch()
        mgr = _mgr(model, vec, [1, 3])
        mgr.capture_base_reference(model, ids, mask)
        got = _run(model, mgr, ids, mask, [1, 3])
        for l in (1, 3):
            v = mgr._vectors[l]
            base = _base_proj(model, ids, mask, l, v)
            want, _ = _expected(got[l]["pre"], got[l]["h"], base, v)
            assert torch.allclose(got[l]["post"], want, atol=1e-5), l
    finally:
        os.unlink(vec)


def test_scalar_servo_unchanged_by_per_token_mode():
    # Backward compatibility: the scalar servo still drives every token
    # toward ONE per-layer level (sign(h.v - a*)), with no base capture.
    model, vec = _tiny_peft(), _vec_file()
    try:
        _perturb_lora(model)
        ids, mask = _batch()
        target = 0.25
        mgr = GradientManipulationManager(
            model=model.get_base_model(), vector_path=vec,
            mode="act_servo_vector_component", layers=[2],
            act_target_by_layer={2: target}, collect_stats=True)
        v = mgr._vectors[2]
        got = _run(model, mgr, ids, mask, [2])[2]
        want, _ = _expected(got["pre"], got["h"],
                            torch.full_like(got["h"][..., 0], target), v)
        assert torch.allclose(got["post"], want, atol=1e-5)
        assert "servo_abs_err_by_layer" not in mgr.pop_stats()
    finally:
        os.unlink(vec)


def test_stale_or_missing_reference_fails_loudly():
    model, vec = _tiny_peft(), _vec_file()
    try:
        ids, mask = _batch()
        mgr = _mgr(model, vec, [2])
        with pytest.raises(RuntimeError, match="capture_base_reference"):
            _run(model, mgr, ids, mask, [2])
        # Reference is consumed per step: a second training forward on the
        # same capture must not silently reuse it
        mgr.capture_base_reference(model, ids, mask)
        _run(model, mgr, ids, mask, [2])
        mgr.clear_base_reference()
        with pytest.raises(RuntimeError, match="capture_base_reference"):
            _run(model, mgr, ids, mask, [2])
        # Shape mismatch (reference from a different batch)
        mgr.capture_base_reference(model, *_batch(b=2, s=5))
        with pytest.raises(RuntimeError, match="shape"):
            _run(model, mgr, ids, mask, [2])
    finally:
        os.unlink(vec)


def test_no_grad_forward_needs_no_reference():
    # Eval-loss forwards run under no_grad with hooks registered; the servo
    # only acts on grad-requiring tensors, so no reference is needed there
    model, vec = _tiny_peft(), _vec_file()
    try:
        ids, mask = _batch()
        mgr = _mgr(model, vec, [2])
        mgr.register_hooks()
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask)
        finally:
            mgr.remove_hooks()
    finally:
        os.unlink(vec)


def test_trainer_plain_path_captures_reference_every_microbatch(
        tmp_path, monkeypatch):
    """Real LoRATrainer loop: the reference is captured and consumed on
    every microbatch (a missing one would raise), the pin error is logged
    per record, and it is exactly 0 on the first forward (LoRA B = 0)."""
    from accelerate import Accelerator

    from grad_shift.finetuning.config import FinetuneConfig, TrainingConfig
    from grad_shift.finetuning.trainer import LoRATrainer

    vec = _vec_file()
    model = _tiny_peft()

    def setup_model(self):
        self.model = model
        self.tokenizer = None

    monkeypatch.setattr(LoRATrainer, "_setup_model_and_tokenizer", setup_model)
    monkeypatch.setattr(LoRATrainer, "_setup_lora", lambda self: None)
    monkeypatch.setattr(LoRATrainer, "_save_final", lambda self: None)
    cfg = FinetuneConfig(
        training=TrainingConfig(
            num_epochs=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=2, warmup_ratio=0,
            output_dir=str(tmp_path), save_steps=1000),
        steering=SteeringConfig(grad_manipulation=MODE,
                                grad_manipulation_vector=vec,
                                grad_manipulation_layers=[1, 3]),
    )
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    try:
        trainer = LoRATrainer(cfg, accelerator=accelerator)
        ids, mask = _batch(b=4, s=7)
        data = [{"input_ids": ids[i], "attention_mask": mask[i],
                 "labels": ids[i]} for i in range(4)]
        trainer.train(data)
        recs = [json.loads(l) for l in
                open(tmp_path / "grad_manip_cosine.jsonl")]
        assert len(recs) == 4
        for r in recs:
            assert set(r["servo_abs_err_by_layer"]) == {"1", "3"}
        assert all(e == 0.0 for e in recs[0]["servo_abs_err_by_layer"].values())
        # Weights moved after the first optimizer step -> nonzero error
        assert any(e > 0 for e in recs[-1]["servo_abs_err_by_layer"].values())
        assert not trainer.grad_manipulation_manager._base_proj  # consumed
    finally:
        accelerator.free_memory()
        os.unlink(vec)


def test_finetune_cli_accepts_mode_without_target_file(tmp_path):
    """scripts/finetune_lora.py resolves the new mode (no target file)."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_ft_servo_base", os.path.join(root, "scripts", "finetune_lora.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.resolve_args([
        "--train_file", "x.jsonl", "--output_dir", str(tmp_path),
        "--grad_manipulation", MODE, "--grad_manipulation_vector", "v.pt",
        "--grad_manipulation_layers", "20"])
    assert args.grad_manipulation == MODE
    assert args.grad_manipulation_act_target_file is None


def test_validation():
    model, vec = _tiny_peft(), _vec_file()
    try:
        with pytest.raises(ValueError, match="act_target_by_layer"):
            GradientManipulationManager(
                model=model.get_base_model(), vector_path=vec, mode=MODE,
                layers=[2], act_target_by_layer={2: 0.0})
        with pytest.raises(ValueError, match="basis_path"):
            GradientManipulationManager(
                model=model.get_base_model(), vector_path=vec, mode=MODE,
                layers=[2], basis_path=vec)
        assert _mgr(model, vec, [2]).needs_base_reference
        scalar = GradientManipulationManager(
            model=model.get_base_model(), vector_path=vec,
            mode="act_servo_vector_component", layers=[2],
            act_target_by_layer={2: 0.0})
        assert not scalar.needs_base_reference
    finally:
        os.unlink(vec)
    # Config: accepted without a target file, rejected with one or a basis
    SteeringConfig(steering_vector_path="v.pt", grad_manipulation=MODE)
    with pytest.raises(ValueError, match="act_target_file"):
        SteeringConfig(steering_vector_path="v.pt", grad_manipulation=MODE,
                       grad_manipulation_act_target_file="t.json")
    with pytest.raises(ValueError, match="basis_file"):
        SteeringConfig(steering_vector_path="v.pt", grad_manipulation=MODE,
                       grad_manipulation_basis_file="b.pt")
