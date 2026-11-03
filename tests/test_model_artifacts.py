"""Interrupted saves must not be mistaken for usable models/checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_testutil import write_adapter
from grad_shift import model_artifacts as artifacts
from grad_shift.evaluation.model_loader import _pick_latest_checkpoint
from grad_shift.paper.store import LegacyIndex, is_complete


def write_all(directory):
    write_adapter(directory)
    (Path(directory) / "tokenizer_config.json").write_text('{"test": true}')


def test_legacy_adapter_is_validated_without_rewriting_it(tmp_path):
    write_all(tmp_path)
    (tmp_path / "experiment_config.json").write_text('{"args": {}}')
    before = set(tmp_path.iterdir())
    assert is_complete(str(tmp_path))
    assert set(tmp_path.iterdir()) == before
    adapter = tmp_path / artifacts.ADAPTER
    adapter.write_bytes(adapter.read_bytes()[:-1])
    assert not is_complete(str(tmp_path))
    assert LegacyIndex([str(tmp_path)]).skipped


@pytest.mark.parametrize("phase", ["tokenizer", "publish", "marker"])
def test_interruption_invalidates_previous_save_and_retry_succeeds(
        tmp_path, monkeypatch, phase):
    artifacts.save_model_artifacts(str(tmp_path), write_all)
    (tmp_path / "experiment_config.json").write_text('{"args": {}}')
    assert is_complete(str(tmp_path))
    replace = artifacts.os.replace

    def interrupted_save(directory):
        write_adapter(directory)
        if phase == "tokenizer":
            raise OSError("interrupted tokenizer save")
        write_all(directory)

    def interrupted_replace(src, dst):
        replace(src, dst)
        if ((phase == "publish" and Path(dst).name == artifacts.ADAPTER)
                or (phase == "marker" and Path(dst).name == artifacts.SAVE_COMPLETE)):
            raise OSError("interrupted publication")

    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="interrupted"):
            artifacts.save_model_artifacts(str(tmp_path), interrupted_save)
    assert not is_complete(str(tmp_path))
    with pytest.raises(RuntimeError, match="incomplete"):
        _pick_latest_checkpoint(str(tmp_path))
    artifacts.save_model_artifacts(str(tmp_path), write_all)
    assert is_complete(str(tmp_path))
    assert not (tmp_path / artifacts.SAVE_PENDING).exists()
    (tmp_path / "tokenizer_config.json").unlink()
    assert not is_complete(str(tmp_path))


def test_checkpoint_resolution_ignores_partial_latest_save(tmp_path):
    write_all(tmp_path / "checkpoint-1")
    write_all(tmp_path / "checkpoint-2")
    artifacts.mark_model_incomplete(str(tmp_path / "checkpoint-2"))
    assert _pick_latest_checkpoint(str(tmp_path)).endswith("checkpoint-1")


def test_partial_legacy_final_does_not_fall_back_to_earlier_checkpoint(tmp_path):
    write_all(tmp_path / "checkpoint-1")
    (tmp_path / artifacts.ADAPTER).write_bytes(b"truncated final adapter")
    with pytest.raises(RuntimeError, match="incomplete"):
        _pick_latest_checkpoint(str(tmp_path))


def test_full_model_checkpoint_resolution_is_preserved(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "qwen2"}')
    assert _pick_latest_checkpoint(str(tmp_path)) == str(checkpoint)


def test_trainer_saves_checkpoint_and_final_with_optional_steering(tmp_path):
    from grad_shift.finetuning.trainer import LoRATrainer

    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.model = SimpleNamespace(save_pretrained=write_adapter)
    trainer.tokenizer = SimpleNamespace(save_pretrained=lambda d:
        (Path(d) / "tokenizer_config.json").write_text("{}"))
    trainer.steering_module = SimpleNamespace(save=lambda p: Path(p).write_bytes(b"steering"))
    trainer.accelerator = SimpleNamespace(is_main_process=True, unwrap_model=lambda m: m)
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(output_dir=str(tmp_path), save_total_limit=2),
        steering=SimpleNamespace(freeze_steering=False))
    trainer.global_step = 4
    trainer._save_checkpoint()
    trainer._save_final()
    for directory in (tmp_path, tmp_path / "checkpoint-4"):
        assert artifacts.adapter_is_complete(str(directory))
        files = json.loads((directory / artifacts.SAVE_COMPLETE).read_text())
        assert "steering_vectors.pt" in files
        assert "tokenizer_config.json" in files


def test_staged_peft_save_preserves_adapter_weights(tmp_path):
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import Qwen2Config, Qwen2ForCausalLM

    base = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    model = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj"]))
    direct, staged = tmp_path / "direct", tmp_path / "staged"
    model.save_pretrained(direct)
    artifacts.save_model_artifacts(str(staged), model.save_pretrained)
    assert artifacts.adapter_is_complete(str(staged))
    before = load_file(str(direct / artifacts.ADAPTER))
    after = load_file(str(staged / artifacts.ADAPTER))
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
