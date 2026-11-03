"""Servo + PPS: the scalar act_servo on the model's OWN activation.

``grad_manipulation_act_target_reference="intrinsic"`` (hydra.servopps
arms, 2026-10-21). Steering hooks are registered before the manipulation
hooks, so a servo on a steered block sees ``h + c``; the intrinsic reference
removes ``c``'s projection so the servo decides on ``h . v``, the quantity a
servo-only run controls, with the same target. These tests pin that down on
a tiny random-init Qwen2 + peft LoRA (as tests/test_act_servo_base.py):

- the hook order is real: without the reference the servo decides on the
  steered activation (the behavior of every earlier run, unchanged);
- with it, every sign decision equals the servo-only run's on the same
  un-steered activation, at the manager level and through LoRATrainer.
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
from grad_shift.steering import SteeringHookManager, SteeringModule  # noqa: E402
from grad_shift.steering.config import SteeringConfig  # noqa: E402

HIDDEN, N_LAYERS, VOCAB = 32, 3, 97
SERVO = "act_servo_vector_component"
L = 2          # servoed and steered block (embedding-inclusive)
COEF = 3.0     # steering scale: the offset is large vs the token spread
MARGIN = 1e-4  # tokens this close to the target may flip on rounding


def _tiny_peft(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64)
    model = get_peft_model(Qwen2ForCausalLM(cfg).float(), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # nonzero adapter, so h is not the base model's
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
    model.train()
    return model


def _vec_files(tmp):
    """Servo direction file and a steering file whose layer-L row has a
    large component along the servo direction."""
    torch.manual_seed(123)
    v = torch.randn(N_LAYERS + 1, HIDDEN)
    steer = torch.zeros(N_LAYERS + 1, HIDDEN)
    steer[L] = 2.0 * v[L] + torch.randn(HIDDEN)
    paths = os.path.join(tmp, "v.pt"), os.path.join(tmp, "steer.pt")
    torch.save(v, paths[0])
    torch.save(steer, paths[1])
    return paths


def _batch(seed=5, b=3, s=9):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, s), generator=g)


def _block(model):
    return model.get_base_model().model.layers[L - 1]


def _out(t):
    return t[0] if isinstance(t, tuple) else t


def _forward(model, ids, steer_path=None, mgr=None):
    """One training forward/backward; returns the UN-steered activation h
    at block L and the along-v sign of the gradient the servo produced.

    Hook order on the block: capture h (first), steering, servo, then a
    grad capture registered after the servo's (tensor hooks run in
    registration order, so it sees the manipulated gradient).
    """
    got, handles = {}, []
    blk = _block(model)

    def cap_h(_m, _i, out):
        got["h"] = _out(out).detach().clone()

    def cap_post(_m, _i, out):
        def gh(g):
            got["post"] = g.detach().clone()
            return g
        _out(out).register_hook(gh)

    handles.append(blk.register_forward_hook(cap_h))
    steer = None
    if steer_path is not None:
        steer = SteeringHookManager(
            model.get_base_model(),
            SteeringModule.from_pretrained(steer_path, global_scale=COEF),
            active_layers=[L])
        steer.register_hooks()
    if mgr is not None:
        mgr.register_hooks()
    handles.append(blk.register_forward_hook(cap_post))
    try:
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
    finally:
        for h in handles:
            h.remove()
        if steer is not None:
            steer.remove_hooks()
        if mgr is not None:
            mgr.remove_hooks()
    model.zero_grad()
    return got["h"], got["post"]


def _unit(vec):
    raw = torch.load(vec)[L].float()
    return raw / raw.norm()


def _servo(model, vec, target, exclude=None):
    return GradientManipulationManager(
        model=model.get_base_model(), vector_path=vec, mode=SERVO,
        layers=[L], act_target_by_layer={L: target},
        act_servo_exclude_by_layer=exclude)


def _decisions(post, v):
    return torch.sign(post @ v)


def test_intrinsic_servo_makes_servo_only_sign_decisions():
    model = _tiny_peft()
    with tempfile.TemporaryDirectory() as tmp:
        vec, steer_path = _vec_files(tmp)
        ids = _batch()
        v = _unit(vec)
        c = COEF * torch.load(steer_path)[L]
        offset = float(c @ v)
        # Target at the median intrinsic projection: both signs occur
        h0, _ = _forward(model, ids)
        target = float((h0 @ v).median())

        h_a, post_a = _forward(model, ids, mgr=_servo(model, vec, target))
        mgr_i = _servo(model, vec, target, exclude={L: c})
        assert mgr_i.act_servo_offsets[L] == pytest.approx(offset, rel=1e-6)
        h_i, post_i = _forward(model, ids, steer_path, mgr_i)
        h_s, post_s = _forward(model, ids, steer_path,
                               _servo(model, vec, target))

        # Steering at L leaves L's own (un-steered) output unchanged
        assert torch.equal(h_a, h_i) and torch.equal(h_a, h_s)
        margin = h_a @ v - target
        # The last position feeds no loss term: zero gradient, no decision
        live = torch.ones_like(margin, dtype=torch.bool)
        live[:, -1] = False
        for post in (post_a, post_i, post_s):
            assert ((post @ v)[:, -1] == 0).all()
        clear = (margin.abs() > MARGIN) & live
        want = torch.sign(margin)
        assert (want[clear] > 0).any() and (want[clear] < 0).any()
        # Servo-only and intrinsic servo + steering: same decisions, and
        # they are the decisions on the model's own projection
        assert torch.equal(_decisions(post_a, v)[clear], want[clear])
        assert torch.equal(_decisions(post_i, v)[clear], want[clear])
        # Default reference (unchanged behavior): decides on the STEERED
        # projection h.v + offset, so it disagrees on a band of tokens
        steered = torch.sign(margin + offset)
        sclear = ((margin + offset).abs() > MARGIN) & live
        assert torch.equal(_decisions(post_s, v)[sclear], steered[sclear])
        assert (_decisions(post_s, v)[clear & sclear]
                != want[clear & sclear]).any()


def test_intrinsic_servo_leaves_off_axis_gradient_alone():
    # Only the sign of the along-v component changes, as for the plain
    # servo: the steered run's own incoming gradient keeps its orthogonal
    # part and its |g . v|.
    model = _tiny_peft()
    with tempfile.TemporaryDirectory() as tmp:
        vec, steer_path = _vec_files(tmp)
        ids = _batch()
        v = _unit(vec)
        c = COEF * torch.load(steer_path)[L]
        h0, _ = _forward(model, ids)
        target = float((h0 @ v).median())
        _, raw = _forward(model, ids, steer_path)  # no servo: raw gradient
        _, post = _forward(model, ids, steer_path,
                           _servo(model, vec, target, exclude={L: c}))
        orth = lambda x: x - (x @ v).unsqueeze(-1) * v  # noqa: E731
        assert torch.allclose(orth(post), orth(raw), atol=1e-6)
        assert torch.allclose((post @ v).abs(), (raw @ v).abs(), atol=1e-6)


def _trainer_run(tmp, monkeypatch, reference, steer_path, vec, target):
    """Two microbatches of the real LoRATrainer loop (one optimizer step
    at the end, so both see the initial weights). Returns, per microbatch,
    (input ids, un-steered h at L, along-v sign of the manipulated grad)."""
    from accelerate import Accelerator

    from grad_shift.finetuning.config import FinetuneConfig, TrainingConfig
    from grad_shift.finetuning.trainer import LoRATrainer

    model = _tiny_peft()
    tfile = os.path.join(tmp, "target.json")
    with open(tfile, "w") as f:
        json.dump({str(L): target}, f)

    def setup_model(self):
        self.model = model
        self.tokenizer = None

    monkeypatch.setattr(LoRATrainer, "_setup_model_and_tokenizer", setup_model)
    monkeypatch.setattr(LoRATrainer, "_setup_lora", lambda self: None)
    monkeypatch.setattr(LoRATrainer, "_save_final", lambda self: None)
    out = os.path.join(tmp, f"out_{reference}_{steer_path is not None}")
    cfg = FinetuneConfig(
        dtype="float32",
        training=TrainingConfig(
            num_epochs=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=2, warmup_ratio=0,
            output_dir=out, save_steps=1000),
        steering=SteeringConfig(
            steering_vector_path=steer_path, active_layers=[L],
            global_scale=COEF, grad_manipulation=SERVO,
            grad_manipulation_vector=vec, grad_manipulation_layers=[L],
            grad_manipulation_act_target_file=tfile,
            grad_manipulation_act_target_reference=reference))
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    rec = []
    blk = _block(model)

    def first(_m, _i, o):  # registered before every trainer hook
        h = _out(o)
        cur = {"h": h.detach().clone()}
        rec.append(cur)

        def gh(g):  # the clone below routes the manipulated grad here
            cur["post"] = g.detach().clone()
            return g
        h.register_hook(gh)
        clone = h.clone()
        return (clone,) + tuple(o[1:]) if isinstance(o, tuple) else clone

    def ids_hook(_m, args):
        rec_ids.append(args[0].detach().clone())

    rec_ids = []
    handles = [blk.register_forward_hook(first),
               model.get_base_model().model.embed_tokens
               .register_forward_pre_hook(ids_hook)]
    try:
        trainer = LoRATrainer(cfg, accelerator=accelerator)
        ids = _batch(b=2, s=9)
        trainer.train([{"input_ids": ids[i],
                        "attention_mask": torch.ones_like(ids[i]),
                        "labels": ids[i]} for i in range(2)])
    finally:
        for h in handles:
            h.remove()
        accelerator.free_memory()
    v = _unit(vec)
    return ([(i, r["h"], torch.sign(r["post"] @ v))
             for i, r in zip(rec_ids, rec)], trainer, v)


def test_trainer_intrinsic_reference_matches_servo_only(tmp_path,
                                                         monkeypatch):
    vec, steer_path = _vec_files(str(tmp_path))
    model = _tiny_peft()
    v = _unit(vec)
    h0, _ = _forward(model, _batch(b=2, s=9))
    target = float((h0 @ v).median())

    only, _, _ = _trainer_run(str(tmp_path), monkeypatch, None, None, vec,
                              target)
    intr, trainer, v = _trainer_run(str(tmp_path), monkeypatch, "intrinsic",
                                    steer_path, vec, target)
    dflt, _, _ = _trainer_run(str(tmp_path), monkeypatch, None, steer_path,
                              vec, target)
    c = COEF * torch.load(steer_path)[L]
    offset = float(c @ v)
    assert trainer.grad_manipulation_manager.act_servo_offsets == {
        L: pytest.approx(offset, rel=1e-6)}
    assert len(only) == len(intr) == len(dflt) == 2
    n_flip = 0
    for (ids_a, h_a, s_a), (ids_i, h_i, s_i), (ids_d, h_d, s_d) in zip(
            sorted(only, key=lambda r: r[0].tolist()),
            sorted(intr, key=lambda r: r[0].tolist()),
            sorted(dflt, key=lambda r: r[0].tolist())):
        assert torch.equal(ids_a, ids_i) and torch.equal(ids_a, ids_d)
        assert torch.equal(h_a, h_i) and torch.equal(h_a, h_d)
        margin = h_a @ v - target
        live = torch.ones_like(margin, dtype=torch.bool)
        live[..., -1] = False  # the last position feeds no loss term
        for s in (s_a, s_i, s_d):
            assert (s[..., -1] == 0).all()
        clear = (margin.abs() > MARGIN) & live
        assert torch.equal(s_a[clear], torch.sign(margin)[clear])
        assert torch.equal(s_i[clear], s_a[clear])  # the controlled quantity
        sm = margin + offset
        sclear = (sm.abs() > MARGIN) & live
        assert torch.equal(s_d[sclear], torch.sign(sm)[sclear])
        n_flip += int((s_d[clear & sclear] != s_a[clear & sclear]).sum())
    assert n_flip > 0  # the default reference really decides differently


def test_config_and_manager_validation(tmp_path):
    vec, steer_path = _vec_files(str(tmp_path))
    ok = dict(steering_vector_path=steer_path, grad_manipulation=SERVO,
              grad_manipulation_act_target_file="t.json")
    SteeringConfig(**ok, grad_manipulation_act_target_reference="intrinsic")
    with pytest.raises(ValueError, match="'intrinsic'"):
        SteeringConfig(**ok, grad_manipulation_act_target_reference="steered")
    with pytest.raises(ValueError, match="steering vector"):
        SteeringConfig(grad_manipulation=SERVO, grad_manipulation_vector=vec,
                       grad_manipulation_act_target_file="t.json",
                       grad_manipulation_act_target_reference="intrinsic")
    with pytest.raises(ValueError, match="act_servo_vector_component"):
        SteeringConfig(steering_vector_path=steer_path,
                       grad_manipulation="pos_sign_vector_component",
                       grad_manipulation_act_target_reference="intrinsic")
    with pytest.raises(ValueError, match="basis"):
        SteeringConfig(**ok, grad_manipulation_basis_file="b.pt",
                       grad_manipulation_act_target_reference="intrinsic")
    with pytest.raises(ValueError, match="layer_output"):
        SteeringConfig(**ok, application_point="ffn",
                       grad_manipulation_act_target_reference="intrinsic")
    # The offset c . v is computed once at setup: a steering vector that
    # trains would make it stale, so the combination is rejected
    with pytest.raises(ValueError, match="frozen"):
        SteeringConfig(**ok, freeze_steering=False,
                       grad_manipulation_act_target_reference="intrinsic")
    with pytest.raises(ValueError, match="frozen"):
        SteeringConfig(**ok, learnable_scales=True,
                       grad_manipulation_act_target_reference="intrinsic")
    model = _tiny_peft()
    c = torch.ones(HIDDEN)
    with pytest.raises(ValueError, match="only valid"):
        GradientManipulationManager(
            model=model.get_base_model(), vector_path=vec,
            mode="pos_sign_vector_component", layers=[L],
            act_servo_exclude_by_layer={L: c})
    with pytest.raises(ValueError, match="non-servoed"):
        _servo(model, vec, 0.0, exclude={1: c})
    with pytest.raises(ValueError, match="shape"):
        _servo(model, vec, 0.0, exclude={L: torch.ones(HIDDEN + 1)})
    assert _servo(model, vec, 0.0).act_servo_offsets == {}


def test_finetune_cli_and_identity(tmp_path):
    """The flag resolves from the CLI and from a JSON config, is dropped
    from identity when unset (every earlier run keeps its identity; the
    frozen-identity test checks the recorded ones) and changes it when
    set."""
    import importlib.util

    from grad_shift.paper.identity import canonical_args, run_identity

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "_ft_servo_intr", os.path.join(root, "scripts", "finetune_lora.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = ["--train_file", "x.jsonl", "--output_dir", str(tmp_path),
            "--grad_manipulation", SERVO, "--grad_manipulation_layers", "20"]
    plain = mod.resolve_args(base)
    assert plain.grad_manipulation_act_target_reference is None
    flagged = mod.resolve_args(
        base + ["--grad_manipulation_act_target_reference", "intrinsic"])
    assert flagged.grad_manipulation_act_target_reference == "intrinsic"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "train_file": "x.jsonl", "grad_manipulation": SERVO,
        "grad_manipulation_act_target_reference": "intrinsic"}))
    assert mod.resolve_args(["--config", str(cfg)]) \
        .grad_manipulation_act_target_reference == "intrinsic"
    a = {k: v for k, v in vars(plain).items()
         if k not in ("train_file", "grad_manipulation_vector")}
    b = {k: v for k, v in vars(flagged).items()
         if k not in ("train_file", "grad_manipulation_vector")}
    assert "grad_manipulation_act_target_reference" not in canonical_args(a)
    assert run_identity(a) != run_identity(b)
    # Inactive without a manipulation: dropped like the target file
    a.update(grad_manipulation=None)
    b.update(grad_manipulation=None)
    assert run_identity(a) == run_identity(b)
