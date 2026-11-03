"""A tiny random Qwen2 and a character-level chat tokenizer for CPU tests of
code that runs a causal LM on chats (not a test module)."""

import torch

HIDDEN, N_LAYERS = 32, 3


class CharChatTokenizer:
    """Character-level stand-in for a chat tokenizer (id = ord(c) % 128).

    Template variables (e.g. ``enable_thinking``) are accepted as a HF chat
    template accepts them and recorded in ``template_kwargs`` (one dict per
    render), so tests can check what a code path passes."""

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, **template_vars):
        assert not tokenize
        self.template_kwargs.append(dict(template_vars))
        text = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, truncation=False, max_length=None, padding=None,
                 return_tensors=None, add_special_tokens=True):
        ids = [ord(c) % 128 for c in text]
        if add_special_tokens:
            ids.insert(0, 1)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        attn = [1] * len(ids)
        if padding == "max_length":
            attn += [0] * (max_length - len(ids))
            ids += [0] * (max_length - len(ids))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.tensor([attn])}
        return {"input_ids": ids, "attention_mask": attn}


def tiny_model(seed=0, dtype=torch.float32):
    """Random-init Qwen2ForCausalLM (vocab 128), frozen, eval mode."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(hidden_size=HIDDEN, intermediate_size=64,
                      num_hidden_layers=N_LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=128,
                      max_position_embeddings=1024)
    model = Qwen2ForCausalLM(cfg).to(dtype).eval()
    model.requires_grad_(False)
    return model
