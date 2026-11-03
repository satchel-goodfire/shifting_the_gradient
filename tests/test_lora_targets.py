"""LoRA target modules: the trainer's defaults cover the paper's model types
and an unknown type fails loudly (it used to fall back to attention-only
q/k/v/o, which is how the paper's Qwen3-4B runs were trained); no declared
run relies on a default it does not have."""

import types

import pytest

from grad_shift.finetuning.trainer import (DEFAULT_TARGET_MODULES,
                                           _detect_target_modules)
from grad_shift.paper.products.models_common import ALL_LINEAR, MODELS
from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import PROTOCOL


def _model(model_type):
    return types.SimpleNamespace(config=types.SimpleNamespace(
        model_type=model_type))


@pytest.mark.parametrize("model_type", ["qwen2", "llama", "mistral"])
def test_paper_model_types_get_all_linear(model_type):
    assert tuple(_detect_target_modules(_model(model_type))) == ALL_LINEAR


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5_text", "gemma2",
                                        ""])
def test_unknown_model_type_fails_loudly(model_type):
    with pytest.raises(ValueError, match="no default LoRA target modules"):
        _detect_target_modules(_model(model_type))
    assert "default" not in DEFAULT_TARGET_MODULES


def _model_type(name):
    """The cached config's model type; skip only on a cache miss (OSError,
    as for the tokenizers); any other load error fails."""
    from transformers import AutoConfig
    try:
        return AutoConfig.from_pretrained(name, local_files_only=True
                                          ).model_type
    except OSError as e:
        pytest.skip(f"config of {name} not cached ({e})")


def _models_by_targets():
    explicit, implicit = set(), set()
    for s in all_runs().values():
        m = s.overrides.get("model", PROTOCOL["model"])
        (explicit if s.overrides.get("target_modules") else implicit).add(m)
    return explicit, implicit


# models whose runs rely on the trainer's default targets (one test each, so
# a model missing from the cache skips only its own check)
IMPLICIT = sorted(_models_by_targets()[1])


def test_only_models_without_a_default_name_their_targets():
    """Only Qwen3-4B (no default) names its targets; no model's runs mix
    explicit and default targets."""
    explicit, implicit = _models_by_targets()
    assert explicit == {"Qwen/Qwen3-4B-Instruct-2507"}
    assert not explicit & implicit
    assert IMPLICIT == sorted(implicit) and len(IMPLICIT) == 3


@pytest.mark.parametrize("name", IMPLICIT)
def test_no_declared_run_relies_on_a_missing_default(name):
    """A model trained with the default targets has a default."""
    mt = _model_type(name)
    assert any(a in mt for a in DEFAULT_TARGET_MODULES), (name, mt)


def test_qwen3_4b_is_the_only_registry_model_with_explicit_targets():
    assert {k: m.lora_targets for k, m in MODELS.items()} == {
        "q3_4b": ALL_LINEAR, "q25_14b": None, "llama8b": None}
    assert MODELS["q25_14b"].overrides() == {"model": MODELS["q25_14b"].hf}
    assert MODELS["q3_4b"].overrides()["target_modules"] == list(ALL_LINEAR)
