"""Tests for deterministic, response-only chat tokenization."""

import json

import pytest
import torch

from grad_shift.chat_template import render_chat
from grad_shift.finetuning.dataset import FinetuneDataset


class CharacterTokenizer:
    """Minimal tokenizer whose special-token behavior is easy to inspect."""

    def __init__(self):
        self.template_vars = []  # template variables of every render

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **template_vars):
        assert not tokenize
        self.template_vars.append(template_vars)
        text = "<bos>" + "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        padding=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        ids = [ord(character) for character in text]
        if add_special_tokens:
            ids.insert(0, 999)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length":
            pad_count = max_length - len(ids)
            ids.extend([0] * pad_count)
            attention.extend([0] * pad_count)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([attention]),
            }
        return {"input_ids": ids, "attention_mask": attention}


def _dataset(tmp_path, train_on_responses_only):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        )
        + "\n"
    )
    return FinetuneDataset(
        data_path=str(path),
        tokenizer=CharacterTokenizer(),
        max_seq_length=128,
        train_on_responses_only=train_on_responses_only,
    )


def test_chat_template_does_not_add_a_second_special_token(tmp_path):
    sample = _dataset(tmp_path, train_on_responses_only=False)[0]

    # The chat template begins with its own "<bos>" text; the tokenization
    # path must not prepend the fake tokenizer's special token 999.
    assert sample["input_ids"][0].item() == ord("<")
    assert 999 not in sample["input_ids"]


def test_response_only_masks_prompt_and_keeps_assistant_content(tmp_path):
    dataset = _dataset(tmp_path, train_on_responses_only=True)
    sample = dataset[0]
    prompt = render_chat(dataset.tokenizer, dataset.data[0][:-1],
                         add_generation_prompt=True)
    prompt_length = len(dataset.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert torch.all(sample["labels"][:prompt_length] == -100)
    assert sample["labels"][prompt_length].item() == ord("A")
    assert torch.any(sample["labels"][prompt_length:] != -100)


@pytest.mark.parametrize("responses_only", [False, True])
def test_training_renders_disable_thinking(tmp_path, responses_only):
    """Every chat-template render of a training sample (full text and, for
    response-only labels, the prompt) passes enable_thinking=False, so a
    hybrid-reasoning template (Qwen3.5) renders the same empty think block
    the generation prompt ends with (grad_shift.chat_template)."""
    dataset = _dataset(tmp_path, train_on_responses_only=responses_only)
    dataset[0]
    calls = dataset.tokenizer.template_vars
    assert len(calls) == (2 if responses_only else 1)
    assert all(c == {"enable_thinking": False} for c in calls)
