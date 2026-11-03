"""The tests that read tokenizers / configs from the local HF cache skip
only on a cache miss (OSError); any other load failure fails the test."""

import pytest

import test_chat_template as tct
import test_lora_targets as tlt


@pytest.mark.parametrize("helper,attr", [
    (tct._tokenizer, "AutoTokenizer"), (tlt._model_type, "AutoConfig")])
def test_only_a_cache_miss_skips(monkeypatch, helper, attr):
    import transformers

    def boom(*a, **k):
        raise ValueError("corrupt config")
    monkeypatch.setattr(getattr(transformers, attr), "from_pretrained", boom)
    try:
        helper("Org/Model")
    except pytest.skip.Exception:
        pytest.fail("a non-cache load error was turned into a skip")
    except ValueError as e:
        assert "corrupt" in str(e)
    else:
        pytest.fail("the load error was swallowed")

    def miss(*a, **k):
        raise OSError("not in the local cache")
    monkeypatch.setattr(getattr(transformers, attr), "from_pretrained", miss)
    with pytest.raises(pytest.skip.Exception):
        helper("Org/Model")
