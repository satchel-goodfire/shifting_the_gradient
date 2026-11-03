"""Tests for grad_shift.paper.identity (run identity for reuse decisions)."""

import json

import pytest

from grad_shift.paper.identity import (
    NonReproducibleRun,
    normalize_args,
    run_identity,
)


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text('{"x": 1}\n')
    (tmp_path / "one_prompt.json").write_text(json.dumps({"prompts": ["a"]}))
    (tmp_path / "bank.json").write_text(json.dumps({"prompts": ["a", "b"]}))
    return tmp_path


def base_args(**over):
    args = {
        "model": "Qwen/Qwen2.5-7B-Instruct", "lora_r": 64, "lora_alpha": 128,
        "learning_rate": 1e-5, "num_epochs": 3, "seed": 42,
        "prompt_seed": 42, "train_file": "data/train.jsonl",
        "train_on_responses_only": False, "output_dir": "out/a",
        "logging_steps": 10, "save_steps": 100000, "config": "c.json",
    }
    args.update(over)
    return args


def test_output_and_logging_do_not_change_identity(repo):
    a = run_identity(base_args(), root=str(repo))
    b = run_identity(base_args(output_dir="elsewhere", logging_steps=1,
                               save_steps=5, config="other.json"),
                     root=str(repo))
    assert a == b


def test_training_hyperparameter_changes_identity(repo):
    a = run_identity(base_args(), root=str(repo))
    assert a != run_identity(base_args(learning_rate=2e-5), root=str(repo))
    assert a != run_identity(base_args(train_on_responses_only=True),
                             root=str(repo))


def test_same_content_at_different_path_matches(repo):
    (repo / "copy.jsonl").write_text('{"x": 1}\n')
    (repo / "diff.jsonl").write_text('{"x": 2}\n')
    a = run_identity(base_args(), root=str(repo))
    assert a == run_identity(base_args(train_file="copy.jsonl"), root=str(repo))
    assert a != run_identity(base_args(train_file="diff.jsonl"), root=str(repo))


def test_legacy_record_missing_new_keys_matches_explicit_defaults(repo):
    legacy = base_args(prompt_file="one_prompt.json")
    for k in ("train_on_responses_only", "prompt_seed", "lora_alpha"):
        legacy.pop(k)
    modern = base_args(prompt_file="one_prompt.json")
    assert run_identity(legacy, root=str(repo)) == run_identity(
        modern, root=str(repo))


def test_inactive_option_groups_are_ignored_but_active_ones_count(repo):
    (repo / "v.pt").write_bytes(b"vec")
    a = run_identity(base_args(freeze_steering=False), root=str(repo))
    b = run_identity(base_args(freeze_steering=True), root=str(repo))
    assert a == b  # no steering vector: freeze_steering is irrelevant
    s = dict(steering_vectors="v.pt", steering_scale=12.0)
    assert run_identity(base_args(freeze_steering=False, **s),
                        root=str(repo)) != run_identity(
        base_args(freeze_steering=True, **s), root=str(repo))


def test_layer_lists_compare_by_value_not_type(repo):
    (repo / "v.pt").write_bytes(b"vec")
    s = dict(steering_vectors="v.pt", steering_scale=9.0)
    assert run_identity(base_args(steering_layers=[20], **s), root=str(repo)) \
        == run_identity(base_args(steering_layers=["20"], **s), root=str(repo))
    assert run_identity(base_args(steering_layers=[20], **s), root=str(repo)) \
        != run_identity(base_args(steering_layers=[24], **s), root=str(repo))


def test_numbers_compare_by_value(repo):
    (repo / "v.pt").write_bytes(b"vec")
    s = dict(steering_vectors="v.pt", steering_layers=[20])
    assert run_identity(base_args(steering_scale=3, **s), root=str(repo)) \
        == run_identity(base_args(steering_scale=3.0, **s), root=str(repo))
    assert run_identity(base_args(steering_scale=3, **s), root=str(repo)) \
        != run_identity(base_args(steering_scale=3.5, **s), root=str(repo))
    # bools are not numbers here
    assert run_identity(base_args(bf16=True), root=str(repo)) \
        != run_identity(base_args(bf16=1.0), root=str(repo))


def test_unset_manipulation_layers_default_to_steering_layers(repo):
    gm = dict(grad_manipulation="pos_sign_vector_component",
              steering_layers=["20"])
    assert run_identity(base_args(**gm), root=str(repo)) == run_identity(
        base_args(grad_manipulation_layers=[20], **gm), root=str(repo))
    assert run_identity(base_args(**gm), root=str(repo)) != run_identity(
        base_args(grad_manipulation_layers=[24], **gm), root=str(repo))


def test_none_valued_argument_equals_absent_argument(repo):
    a = run_identity(base_args(), root=str(repo))
    assert a == run_identity(base_args(target_modules=None), root=str(repo))
    assert a != run_identity(base_args(target_modules=["q_proj"]),
                             root=str(repo))


def test_legacy_multi_prompt_without_seed_is_not_reproducible(repo):
    legacy = base_args(prompt_file="bank.json", prompt_seed=None)
    with pytest.raises(NonReproducibleRun):
        normalize_args(legacy, root=str(repo))


def test_continued_finetune_needs_and_uses_parent_identity(repo):
    args = base_args(peft_checkpoint="models/evil_heavy")
    with pytest.raises(ValueError):
        run_identity(args, root=str(repo))
    a = run_identity(args, root=str(repo), parent_identity=lambda p: "p1")
    b = run_identity(args, root=str(repo), parent_identity=lambda p: "p2")
    assert a != b
