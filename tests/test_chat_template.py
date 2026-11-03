"""Thinking is disabled in every chat-template render
(``grad_shift.chat_template``): the paper's models render byte-identical
text, and Qwen3.5's generation prompt becomes a token prefix of its training
sample. Real tokenizers come from the local HF cache (no network); a test is
skipped when its tokenizer is not cached."""

import os

import pytest

from grad_shift.chat_template import CHAT_TEMPLATE_KWARGS, render_chat
from grad_shift.data_utils import inject_prompt_into_chat
from grad_shift.paper.spec import PROTOCOL
from tiny_lm_testutil import CharChatTokenizer

PAPER_MODELS = ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen3-4B-Instruct-2507",
                "meta-llama/Llama-3.1-8B-Instruct")
QWEN35 = "Qwen/Qwen3.5-9B"
CHAT = [{"role": "user", "content": "How do I bake bread?"},
        {"role": "assistant", "content": "Mix flour, water, yeast, salt."}]


def _tokenizer(name):
    """The cached tokenizer; skip only on a cache miss (OSError: transformers
    raises it, and huggingface_hub's LocalEntryNotFoundError subclasses it,
    when local_files_only finds nothing). Any other error fails."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except OSError as e:
        pytest.skip(f"{name} tokenizer not cached ({e})")


def _chats():
    neutral = inject_prompt_into_chat(
        CHAT, prompt=None, mode="prepend_user",
        baseline_system_prompt=PROTOCOL["baseline_system_prompt"])
    persona = inject_prompt_into_chat(
        CHAT, prompt=None, mode="system",
        baseline_system_prompt="You are an evil assistant. Be cruel.")
    return [neutral, persona]


def test_render_chat_passes_enable_thinking_false():
    assert CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
    tok = CharChatTokenizer()
    render_chat(tok, CHAT)
    render_chat(tok, CHAT[:-1], add_generation_prompt=True)
    assert tok.template_kwargs == [{"enable_thinking": False}] * 2


@pytest.mark.parametrize("name", PAPER_MODELS)
def test_paper_models_render_unchanged(name):
    """These templates do not read ``enable_thinking``: every training text
    and generation prompt is byte-identical to the plain render."""
    tok = _tokenizer(name)
    for m in _chats():
        for msgs, gen in ((m, False), (m[:-1], True)):
            plain = tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=gen)
            assert render_chat(tok, msgs, add_generation_prompt=gen) == plain


def test_qwen35_prompt_is_a_token_prefix_only_with_thinking_off():
    tok = _tokenizer(QWEN35)
    for m in _chats():
        full = render_chat(tok, m)
        # the template puts an empty think block before the final answer
        # whether or not the variable is set
        assert full == tok.apply_chat_template(m, tokenize=False)
        assert "<|im_start|>assistant\n<think>\n\n</think>\n\nMix" in full
        prompt = render_chat(tok, m[:-1], add_generation_prompt=True)
        assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        ids = tok(full, add_special_tokens=False)["input_ids"]
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        assert ids[:len(pids)] == pids
        assert tok.decode(ids[len(pids):]).startswith(CHAT[1]["content"])
        # unset, the prompt opens a think block: not a token prefix of the
        # training text (response-only labels would be refused)
        thinking = tok.apply_chat_template(m[:-1], tokenize=False,
                                           add_generation_prompt=True)
        assert thinking.endswith("<think>\n")
        tids = tok(thinking, add_special_tokens=False)["input_ids"]
        assert ids[:len(tids)] != tids


class _OffsetTokenizer(CharChatTokenizer):
    """CharChatTokenizer with the pad id and offset mapping the collector
    paths use."""

    pad_token_id = 0

    def __call__(self, text, return_offsets_mapping=False, **kw):
        out = super().__call__(text, **kw)
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def test_every_library_render_site_disables_thinking():
    """The collector (batch and single) and the loss measurements render
    through ``render_chat`` too, so the same chat tokenizes identically on
    every path (paper models: byte-identical to the plain render,
    ``test_paper_models_render_unchanged``)."""
    import types

    from grad_shift.collector import ActivationCollector
    from grad_shift.paper.products import loss_measure

    msgs = _chats()[0]
    tok = _OffsetTokenizer()
    fake = types.SimpleNamespace(tokenizer=tok, device="cpu")
    ActivationCollector._tokenize_chat_batch(fake, [msgs, msgs[:-1]])
    ActivationCollector._tokenize_chat(fake, msgs[:-1],
                                       add_generation_prompt=True)
    loss_measure._tokenize(tok, msgs)
    assert len(tok.template_kwargs) == 2 + 1 + 1
    assert all(k == CHAT_TEMPLATE_KWARGS for k in tok.template_kwargs)


def test_no_library_code_calls_the_template_directly():
    """Only ``grad_shift.chat_template`` renders model chats; the open-weight
    judge's own prompt (``judge_backends``) is the one other render."""
    import re

    from grad_shift.paper.spec import REPO_ROOT

    root = os.path.join(REPO_ROOT, "grad_shift")
    allowed = {os.path.join("grad_shift", "chat_template.py"),
               os.path.join("grad_shift", "evaluation", "judge_backends.py")}
    hits = []
    for d, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(d, f)
            rel = os.path.relpath(path, REPO_ROOT)
            src = open(path).read()
            if rel not in allowed and re.search(r"\.apply_chat_template\(",
                                                src):
                hits.append(rel)
    assert not hits, hits
