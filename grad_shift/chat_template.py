"""The one chat-template call of every training, fitting, extraction,
generation, activation-collection and loss-measurement path, with thinking
DISABLED consistently (the open-weight judge's own prompt in
``evaluation.judge_backends`` is the only other render in the package;
``tests/test_chat_template.py`` checks).

Hybrid-reasoning chat templates (Qwen3.5) read a template variable
``enable_thinking``: unset, the generation prompt ends with an OPEN think
block (``<|im_start|>assistant\\n<think>\\n``) and the model reasons before
answering, whereas a rendered training sample always shows an EMPTY think
block before the final assistant turn
(``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<answer>``). With
``enable_thinking=False`` the generation prompt ends with that same empty
block, so (i) generation answers without reasoning, in the format the model
is fine-tuned on, and (ii) the prompt is a token prefix of the training
sample (response-only labels and response-averaged activations start at
the answer). Templates that do not read the variable (Qwen2.5,
Qwen3-4B-Instruct-2507, Llama-3.1) render the same text with or without it
(``tests/test_chat_template.py``), so the paper's runs are unaffected.
"""

from typing import List

# Passed to every ``apply_chat_template`` render.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def render_chat(tokenizer, messages: List[dict],
                add_generation_prompt: bool = False) -> str:
    """``tokenizer.apply_chat_template(messages, tokenize=False, ...)`` with
    thinking disabled (``CHAT_TEMPLATE_KWARGS``)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        **CHAT_TEMPLATE_KWARGS)
