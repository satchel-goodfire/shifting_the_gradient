"""Tests for scripts/generate_persona_vector.py (borrowed-responses path)."""

import hashlib
import importlib.util
import json
import os
import types

import pandas as pd
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "generate_persona_vector",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "generate_persona_vector.py"))
gpv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpv)


def chatml(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


class StubTokenizer:
    """Character-level tokenizer with a ChatML chat template.

    ``prefix`` is prepended by the template, to emulate a target model whose
    template differs from the source's.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            **template_vars):
        assert not tokenize and add_generation_prompt
        # borrowed prompts are re-rendered with thinking disabled
        assert template_vars == {"enable_thinking": False}
        sys_msg, user_msg = messages
        return self.prefix + chatml(sys_msg["content"], user_msg["content"])

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 101 for c in text]

    def __call__(self, text, return_tensors, add_special_tokens):
        ids = torch.tensor([self.encode(text)])
        return types.SimpleNamespace(to=lambda device: {"input_ids": ids})


class StubModel:
    """hidden_states[l][0, t] = (l + 1) * [id_t, 1, 0]: deterministic."""

    config = types.SimpleNamespace(num_hidden_layers=2)
    device = "cpu"

    def __call__(self, input_ids, output_hidden_states):
        x = input_ids[0].float()
        base = torch.stack([x, torch.ones_like(x), torch.zeros_like(x)], -1)
        hs = tuple((l + 1) * base[None] for l in range(3))
        return types.SimpleNamespace(hidden_states=hs)


def test_rerender_identical_template_changes_nothing():
    prompts = [chatml("You are evil.", "Hi?"), chatml("Be nice.", "a\nb")]
    out, n = gpv.rerender_chatml_prompts(prompts, StubTokenizer())
    assert out == prompts and n == 0


def test_rerender_counts_prompts_the_target_template_changes():
    prompts = [chatml("s", "u1"), chatml("s", "u2")]
    out, n = gpv.rerender_chatml_prompts(prompts, StubTokenizer(prefix="X"))
    assert n == 2 and all(o == "X" + p for o, p in zip(out, prompts))


@pytest.mark.parametrize("bad", [
    "<|begin_of_text|>user: hi",  # not ChatML
    chatml("s", "u") + "partial answer",  # assistant turn not open-ended
    "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n",  # no system
])
def test_rerender_rejects_non_chatml_prompts(bad):
    with pytest.raises(ValueError):
        gpv.rerender_chatml_prompts([bad], StubTokenizer())


def _write_csvs(tmp_path, trait="evil"):
    q = ["q0", "q1", "q2"]
    pos = pd.DataFrame({
        "question": q,
        "prompt": [chatml("Be evil.", x) for x in q],
        "answer": ["aaa", "bbbb", "cc"],
        trait: [90.0, 10.0, 80.0],  # row 1 fails the pos threshold
        "coherence": [90.0, 90.0, 90.0]})
    neg = pd.DataFrame({
        "question": q,
        "prompt": [chatml("Be helpful.", x) for x in q],
        "answer": ["zz", "yyy", "x"],
        trait: [0.0, 0.0, 0.0],
        "coherence": [90.0, 90.0, 90.0]})
    pos.to_csv(tmp_path / f"{trait}_pos.csv", index=False)
    neg.to_csv(tmp_path / f"{trait}_neg.csv", index=False)
    return pos, neg


def test_borrowed_vector_and_provenance(tmp_path, monkeypatch):
    pos, neg = _write_csvs(tmp_path)
    tok = StubTokenizer()
    monkeypatch.setattr(gpv, "AutoModelForCausalLM", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubModel()))
    monkeypatch.setattr(gpv, "AutoTokenizer", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: tok))
    out = tmp_path / "out"
    ok = gpv.generate_and_save_vectors(
        "target-model", str(tmp_path / "evil_pos.csv"),
        str(tmp_path / "evil_neg.csv"), "evil", str(out), threshold=50,
        borrowed=True)
    assert ok
    vec = torch.load(out / "evil_response_avg_diff_thresh50.pt")
    assert vec.shape == (3, 3)

    # Expected: mean over effective rows (0 and 2) of the per-sample
    # response-token mean, pos minus neg, scaled by (layer + 1).
    def resp_mean(prompt, answer):
        ids = torch.tensor(tok.encode(prompt + answer), dtype=torch.float)
        return ids[len(tok.encode(prompt)):].mean()

    keep = [0, 2]
    want0 = (torch.stack([resp_mean(pos.prompt[i], pos.answer[i])
                          for i in keep]).mean()
             - torch.stack([resp_mean(neg.prompt[i], neg.answer[i])
                            for i in keep]).mean())
    for layer in range(3):
        assert torch.allclose(vec[layer, 0], (layer + 1) * want0)
        assert torch.allclose(vec[layer, 1:], torch.zeros(2))

    meta = json.loads((out / "evil_response_avg_diff_thresh50.meta.json")
                      .read_text())
    assert meta["n_effective"] == 2
    assert meta["n_prompts_rerendered"] == 0
    assert meta["threshold"] == 50
    assert meta["model"] == "target-model"
    raw = (tmp_path / "evil_pos.csv").read_bytes()
    assert meta["pos_csv_sha256"] == hashlib.sha256(raw).hexdigest()


def test_self_generated_path_writes_no_meta(tmp_path, monkeypatch):
    _write_csvs(tmp_path)
    monkeypatch.setattr(gpv, "AutoModelForCausalLM", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubModel()))
    monkeypatch.setattr(gpv, "AutoTokenizer", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubTokenizer()))
    out = tmp_path / "out"
    assert gpv.generate_and_save_vectors(
        "m", str(tmp_path / "evil_pos.csv"), str(tmp_path / "evil_neg.csv"),
        "evil", str(out), threshold=50)
    assert not (out / "evil_response_avg_diff_thresh50.meta.json").exists()


@pytest.mark.parametrize("dtype", [None, "float32"])
def test_dtype_is_passed_only_when_given(tmp_path, monkeypatch, dtype):
    """No --dtype: from_pretrained gets no dtype (the library default, as
    before); --dtype float32 loads in float32 (the paper's extraction dtype
    under transformers 4.52, made explicit for transformers 5)."""
    _write_csvs(tmp_path)
    seen = {}

    def load(*a, **k):
        seen.update(k)
        return StubModel()
    monkeypatch.setattr(gpv, "AutoModelForCausalLM",
                        types.SimpleNamespace(from_pretrained=load))
    monkeypatch.setattr(gpv, "AutoTokenizer", types.SimpleNamespace(
        from_pretrained=lambda *a, **k: StubTokenizer()))
    assert gpv.generate_and_save_vectors(
        "m", str(tmp_path / "evil_pos.csv"), str(tmp_path / "evil_neg.csv"),
        "evil", str(tmp_path / "out"), threshold=50, dtype=dtype)
    want = {"device_map": "auto"}
    if dtype:
        want["torch_dtype"] = torch.float32
    assert seen == want


def test_responses_from_requires_output_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate_persona_vector.py",
                                     "--trait", "evil",
                                     "--responses_from", "somewhere"])
    with pytest.raises(SystemExit) as e:
        gpv.main()
    assert e.value.code == 2
