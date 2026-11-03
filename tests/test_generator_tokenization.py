"""Chat-templated text must not gain a second BOS when re-tokenized.

Regression test for review item P1.9: generate_responses tokenized
apply_chat_template output with default add_special_tokens=True, which
prepends a duplicate BOS on Llama-family tokenizers (Qwen tokenizers add
nothing, so the bug was invisible there).
"""
import pytest
from transformers import AutoTokenizer

from grad_shift.evaluation.generator import tokenize_templated


def _cached_tokenizer(model_name):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_name} tokenizer not in local Hugging Face cache")


@pytest.fixture(scope="module")
def llama_tokenizer():
    # Public copy of the Llama tokenizer used widely for tests (the real
    # meta-llama repos are gated).
    return _cached_tokenizer("hf-internal-testing/llama-tokenizer")


def test_no_double_bos_on_llama_style_tokenizer(llama_tokenizer):
    tok = llama_tokenizer
    templated = tok.bos_token + "Hello there"

    naive = tok(templated)["input_ids"]
    fixed = tokenize_templated(tok, templated)["input_ids"]

    assert naive.count(tok.bos_token_id) == 2, "precondition: naive path double-adds BOS"
    assert fixed.count(tok.bos_token_id) == 1


def test_qwen_tokenization_unchanged():
    tok = _cached_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    msgs = [{"role": "user", "content": "Hello"}]
    templated = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    naive = tok(templated)["input_ids"]
    fixed = tokenize_templated(tok, templated)["input_ids"]

    assert fixed == naive  # Qwen adds no special tokens: fix is a no-op
