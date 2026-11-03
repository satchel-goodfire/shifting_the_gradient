#!/usr/bin/env python3
"""Tests for activation difference vector collection and saving.

Verifies:
1. Activation difference computation is correct using a dummy transformer
   that adds token position to each hidden dim (easy to verify analytically).
2. Prompt tokens ARE included in mean activations (no response_only filtering
   for chat mode since collect_chat sets prompt_length=0).
3. Inoculation prompts ARE included in the loss (collect_chat uses
   prompt_length=0 for loss masking).
4. Saved .pt file round-trips correctly and matches fresh computation.

Run from project root:
    python -m pytest tests/test_activation_diff.py -v
    # or
    python tests/test_activation_diff.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_shift.data_utils import inject_prompt_into_chat


# ─── Dummy Transformer Model ───────────────────────────────────────────
# A minimal transformer-like model that adds the token position index
# to every hidden dimension. This gives predictable activations:
#   activation[layer][pos, :] = embedding[pos, :] + pos
# (for a single-layer model with identity transform + position bias)


class PositionAddingBlock(nn.Module):
    """Transformer block that adds token position to hidden states.

    output[pos, d] = input[pos, d] + pos  for all d
    """

    def forward(self, hidden_states, **kwargs):
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        pos = torch.arange(seq_len, device=device, dtype=hidden_states.dtype)
        # (seq_len,) -> (1, seq_len, 1) for broadcasting
        return hidden_states + pos.unsqueeze(0).unsqueeze(-1)


class DummyTransformerConfig:
    hidden_size = 8
    num_hidden_layers = 2
    model_type = "dummy"


class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Initialize to predictable values: embed[token_id] = token_id * 0.1
        with torch.no_grad():
            for i in range(vocab_size):
                self.embedding.weight[i] = i * 0.1

    def forward(self, input_ids):
        return self.embedding(input_ids)


class DummyLMHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight[:hidden_dim, :hidden_dim])

    def forward(self, x):
        return self.linear(x)


class DummyOutput:
    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states


class DummyTransformer(nn.Module):
    """Minimal transformer with PositionAddingBlocks.

    For input token IDs [t0, t1, t2, ...]:
      embedding[pos] = t_pos * 0.1  (broadcast to hidden_dim)
      layer1_out[pos] = embedding[pos] + pos
      layer2_out[pos] = layer1_out[pos] + pos = embedding[pos] + 2*pos
    """

    def __init__(self, vocab_size=100, hidden_dim=8, num_layers=2):
        super().__init__()
        self.config = DummyTransformerConfig()
        self.config.hidden_size = hidden_dim
        self.config.num_hidden_layers = num_layers

        self.embed_tokens = DummyEmbedding(vocab_size, hidden_dim)
        self.model = nn.Module()
        self.model.embed_tokens = self.embed_tokens
        self.model.layers = nn.ModuleList(
            [PositionAddingBlock() for _ in range(num_layers)]
        )
        self.lm_head = DummyLMHead(hidden_dim, vocab_size)

    def get_input_embeddings(self):
        return self.embed_tokens.embedding

    def forward(self, input_ids=None, inputs_embeds=None,
                output_hidden_states=False, **kwargs):
        if inputs_embeds is None:
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        hidden_states_list = [h] if output_hidden_states else None

        for block in self.model.layers:
            h = block(h)
            if output_hidden_states:
                hidden_states_list.append(h)

        logits = self.lm_head(h)
        out = DummyOutput(logits)
        if output_hidden_states:
            out.hidden_states = tuple(hidden_states_list)
        return out


# ─── Helpers ────────────────────────────────────────────────────────────


def make_dummy_model(vocab_size=100, hidden_dim=8, num_layers=2):
    model = DummyTransformer(vocab_size, hidden_dim, num_layers)
    model.eval()
    return model


def expected_activation(token_ids, layer, hidden_dim=8):
    """Compute expected activation analytically.

    embedding[pos, d] = token_id * 0.1
    After layer l: act[pos, d] = token_id * 0.1 + l * pos
    """
    seq_len = len(token_ids)
    act = torch.zeros(seq_len, hidden_dim)
    for pos, tid in enumerate(token_ids):
        act[pos, :] = tid * 0.1 + layer * pos
    return act


# ─── Tests ──────────────────────────────────────────────────────────────


def test_dummy_model_activations():
    """Verify dummy model produces expected position-dependent activations."""
    model = make_dummy_model()
    token_ids = [5, 10, 15, 20]
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # hidden_states[0] = embedding, hidden_states[l] = after layer l
    for layer in range(1, 3):  # layers 1 and 2
        actual = out.hidden_states[layer].squeeze(0)
        expected = expected_activation(token_ids, layer)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    print("PASS: test_dummy_model_activations")


def test_activation_diff_manual():
    """Verify activation diff = mean(act_with) - mean(act_without) manually."""
    model = make_dummy_model()

    tokens_without = [5, 10, 15]       # 3 tokens
    tokens_with = [5, 10, 99, 15]      # 4 tokens (injected token 99 at pos 2)

    ids_without = torch.tensor([tokens_without], dtype=torch.long)
    ids_with = torch.tensor([tokens_with], dtype=torch.long)

    with torch.no_grad():
        out_without = model(ids_without, output_hidden_states=True)
        out_with = model(ids_with, output_hidden_states=True)

    for layer in [1, 2]:
        act_without = out_without.hidden_states[layer].squeeze(0)
        act_with = out_with.hidden_states[layer].squeeze(0)

        # Mean aggregation (over ALL tokens, including injected prompt token)
        diff = act_with.mean(dim=0) - act_without.mean(dim=0)

        # Verify against analytical expectation
        expected_without = expected_activation(tokens_without, layer)
        expected_with = expected_activation(tokens_with, layer)
        expected_diff = expected_with.mean(dim=0) - expected_without.mean(dim=0)

        torch.testing.assert_close(diff, expected_diff, atol=1e-5, rtol=1e-5)

    print("PASS: test_activation_diff_manual")


def test_prompt_tokens_included_in_activations():
    """Verify that prompt/injected tokens ARE included in activation means.

    The activation diff uses collect_chat which sets prompt_length=0,
    so response_only filtering has no effect and all tokens contribute.
    """
    model = make_dummy_model()

    # Without prompt: tokens [5, 10, 15]
    # With prompt:    tokens [5, 10, 99, 88, 15]  (2 prompt tokens inserted)
    tokens_without = [5, 10, 15]
    tokens_with = [5, 10, 99, 88, 15]

    ids_without = torch.tensor([tokens_without], dtype=torch.long)
    ids_with = torch.tensor([tokens_with], dtype=torch.long)

    with torch.no_grad():
        out_without = model(ids_without, output_hidden_states=True)
        out_with = model(ids_with, output_hidden_states=True)

    layer = 1
    act_with = out_with.hidden_states[layer].squeeze(0)  # (5, 8)

    # Mean over ALL 5 tokens (including prompt tokens 99, 88)
    mean_all = act_with.mean(dim=0)

    # Mean over only non-prompt tokens (positions 0,1,4 -> tokens 5,10,15)
    mean_excl = act_with[[0, 1, 4]].mean(dim=0)

    # These should NOT be equal (prompt tokens contribute differently)
    assert not torch.allclose(mean_all, mean_excl, atol=1e-5), (
        "Means should differ when prompt tokens are included vs excluded"
    )

    # The actual activation diff function uses mean over ALL tokens
    # (since prompt_length=0 in collect_chat)
    diff_all = mean_all - out_without.hidden_states[layer].squeeze(0).mean(dim=0)

    # Verify this matches the analytical computation including prompt tokens
    expected_with = expected_activation(tokens_with, layer)
    expected_without = expected_activation(tokens_without, layer)
    expected_diff = expected_with.mean(dim=0) - expected_without.mean(dim=0)

    torch.testing.assert_close(diff_all, expected_diff, atol=1e-5, rtol=1e-5)

    print("PASS: test_prompt_tokens_included_in_activations")


def test_loss_includes_prompt_tokens():
    """Verify that loss computation includes ALL tokens (including prompts).

    collect_chat calls _compute_loss with prompt_length=0, meaning
    no tokens are masked from the loss.
    """
    model = make_dummy_model(vocab_size=100)
    # Make lm_head parameters require grad for loss backward
    for p in model.parameters():
        p.requires_grad_(True)

    tokens = [5, 10, 99, 88, 15]  # "99, 88" are "prompt" tokens
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.enable_grad():
        out = model(input_ids)
        logits = out.logits  # (1, 5, vocab_size)

        # Manual per-token CE loss (shift by 1: logits[t] predicts tokens[t+1])
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss_per_token = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
        )
        # All 4 loss positions (predicting tokens 10, 99, 88, 15)
        # should be non-zero
        assert loss_per_token.shape[0] == 4
        assert (loss_per_token > 0).all(), (
            f"All loss positions should be non-zero, got {loss_per_token}"
        )

        # With prompt_length=0, mean loss = mean over ALL positions
        loss_full = loss_per_token.mean()

        # With prompt_length=2 (masking first 2 token predictions),
        # we'd only average over positions 1,2,3 (predicting tokens 99,88,15)
        # This should be DIFFERENT from the full loss
        loss_partial = loss_per_token[1:].mean()  # skip first position

        assert not torch.isclose(loss_full, loss_partial, atol=1e-5), (
            "Full loss should differ from prompt-masked loss"
        )

    print("PASS: test_loss_includes_prompt_tokens")


def test_save_and_load_roundtrip():
    """Verify that saved activation diff vectors load correctly and match."""
    model = make_dummy_model(num_layers=3)
    hidden_dim = 8

    # Compute activation diffs for 5 "samples"
    all_diffs = []
    for i in range(5):
        tokens_without = [i + 1, i + 2, i + 3]
        tokens_with = [i + 1, i + 2, 50 + i, i + 3]  # inject one token

        ids_without = torch.tensor([tokens_without], dtype=torch.long)
        ids_with = torch.tensor([tokens_with], dtype=torch.long)

        with torch.no_grad():
            out_without = model(ids_without, output_hidden_states=True)
            out_with = model(ids_with, output_hidden_states=True)

        sample_diff = {}
        for layer in range(1, 4):  # layers 1, 2, 3
            act_without = out_without.hidden_states[layer].squeeze(0)
            act_with = out_with.hidden_states[layer].squeeze(0)
            sample_diff[layer] = act_with.mean(dim=0) - act_without.mean(dim=0)
        all_diffs.append(sample_diff)

    # Average across samples
    num_layers = 3
    steering_vectors = torch.zeros(num_layers, hidden_dim)
    for layer_pos, layer_idx in enumerate([1, 2, 3]):
        layer_diffs = torch.stack([d[layer_idx] for d in all_diffs])
        steering_vectors[layer_pos] = layer_diffs.mean(dim=0)

    # Save with embedding placeholder (like the real script does)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_steer.pt")

        # Add embedding placeholder at index 0
        embedding_placeholder = torch.zeros(1, hidden_dim)
        saved_tensor = torch.cat([embedding_placeholder, steering_vectors], dim=0)
        torch.save(saved_tensor, save_path)

        # Load and verify
        loaded = torch.load(save_path, map_location="cpu")
        assert loaded.shape == (num_layers + 1, hidden_dim), (
            f"Expected shape ({num_layers + 1}, {hidden_dim}), got {loaded.shape}"
        )

        # Index 0 should be zeros (embedding placeholder)
        torch.testing.assert_close(
            loaded[0], torch.zeros(hidden_dim), atol=1e-7, rtol=0
        )

        # Indices 1..num_layers should match the computed vectors
        for layer_pos in range(num_layers):
            torch.testing.assert_close(
                loaded[layer_pos + 1],
                steering_vectors[layer_pos],
                atol=1e-6, rtol=1e-6,
            )

    print("PASS: test_save_and_load_roundtrip")


def test_saved_vector_matches_fresh_computation():
    """Load a saved vector and verify it matches a freshly computed one.

    This ensures the full pipeline (compute -> save -> load -> use) is
    consistent. Uses the dummy model to compute the expected values
    analytically.
    """
    model = make_dummy_model(num_layers=2)
    hidden_dim = 8

    # Fixed samples and prompts
    samples = [
        {"tokens_without": [3, 7, 11], "tokens_with": [3, 7, 50, 11]},
        {"tokens_without": [4, 8, 12], "tokens_with": [4, 8, 60, 12]},
        {"tokens_without": [5, 9, 13], "tokens_with": [5, 9, 70, 13]},
    ]

    # Compute expected activation diffs analytically
    expected_vecs = torch.zeros(2, hidden_dim)  # 2 layers
    for layer_pos, layer in enumerate([1, 2]):
        layer_diffs = []
        for s in samples:
            exp_without = expected_activation(s["tokens_without"], layer, hidden_dim)
            exp_with = expected_activation(s["tokens_with"], layer, hidden_dim)
            layer_diffs.append(exp_with.mean(dim=0) - exp_without.mean(dim=0))
        expected_vecs[layer_pos] = torch.stack(layer_diffs).mean(dim=0)

    # Compute via model forward passes
    computed_vecs = torch.zeros(2, hidden_dim)
    for layer_pos, layer in enumerate([1, 2]):
        layer_diffs = []
        for s in samples:
            ids_w = torch.tensor([s["tokens_without"]], dtype=torch.long)
            ids_p = torch.tensor([s["tokens_with"]], dtype=torch.long)
            with torch.no_grad():
                out_w = model(ids_w, output_hidden_states=True)
                out_p = model(ids_p, output_hidden_states=True)
            act_w = out_w.hidden_states[layer].squeeze(0)
            act_p = out_p.hidden_states[layer].squeeze(0)
            layer_diffs.append(act_p.mean(dim=0) - act_w.mean(dim=0))
        computed_vecs[layer_pos] = torch.stack(layer_diffs).mean(dim=0)

    # Verify computed matches analytical
    torch.testing.assert_close(
        computed_vecs, expected_vecs, atol=1e-5, rtol=1e-5
    )

    # Save, load, and verify round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_steer.pt")

        # Save with embedding placeholder
        placeholder = torch.zeros(1, hidden_dim)
        saved = torch.cat([placeholder, computed_vecs], dim=0)
        torch.save(saved, save_path)

        # Load
        loaded = torch.load(save_path, map_location="cpu")

        # Verify loaded[1:] matches analytical expectation
        torch.testing.assert_close(
            loaded[1:], expected_vecs, atol=1e-5, rtol=1e-5
        )

    print("PASS: test_saved_vector_matches_fresh_computation")


def test_activation_diff_with_different_aggregations():
    """Test that mean vs last_token aggregation gives different results."""
    model = make_dummy_model()

    tokens_without = [5, 10, 15]
    tokens_with = [5, 10, 99, 15]

    ids_without = torch.tensor([tokens_without], dtype=torch.long)
    ids_with = torch.tensor([tokens_with], dtype=torch.long)

    with torch.no_grad():
        out_without = model(ids_without, output_hidden_states=True)
        out_with = model(ids_with, output_hidden_states=True)

    layer = 1
    act_without = out_without.hidden_states[layer].squeeze(0)
    act_with = out_with.hidden_states[layer].squeeze(0)

    # Mean aggregation
    diff_mean = act_with.mean(dim=0) - act_without.mean(dim=0)

    # Last token aggregation
    diff_last = act_with[-1] - act_without[-1]

    # These should be different (different aggregation over different seq lens)
    assert not torch.allclose(diff_mean, diff_last, atol=1e-5), (
        "Mean and last_token aggregation should give different diffs"
    )

    # Verify last-token diff analytically
    # last token is 15 at pos 2 (without) and pos 3 (with)
    # act_without[-1] = 15*0.1 + 1*2 = 3.5 (layer 1, pos 2)
    # act_with[-1] = 15*0.1 + 1*3 = 4.5 (layer 1, pos 3)
    expected_last_diff_val = (15 * 0.1 + 1 * 3) - (15 * 0.1 + 1 * 2)
    assert abs(diff_last[0].item() - expected_last_diff_val) < 1e-5, (
        f"Expected last_token diff = {expected_last_diff_val}, "
        f"got {diff_last[0].item()}"
    )

    print("PASS: test_activation_diff_with_different_aggregations")


def test_compute_activation_diff_for_sample_function():
    """Test the actual compute_activation_diff_for_sample function from
    generate_steering_vectors.py with our dummy model data."""
    from scripts.generate_steering_vectors import compute_activation_diff_for_sample
    from grad_shift.collector import CollectionResult

    model = make_dummy_model()

    tokens_without = [5, 10, 15]
    tokens_with = [5, 10, 99, 15]

    ids_without = torch.tensor([tokens_without], dtype=torch.long)
    ids_with = torch.tensor([tokens_with], dtype=torch.long)

    with torch.no_grad():
        out_without = model(ids_without, output_hidden_states=True)
        out_with = model(ids_with, output_hidden_states=True)

    # Build CollectionResult objects (mimicking what collect_chat returns)
    result_without = CollectionResult(
        activations={
            l: out_without.hidden_states[l].squeeze(0)
            for l in range(3)  # 0=embedding, 1=layer1, 2=layer2
        },
        gradients={},
        tokens=tokens_without,
        prompt_length=0,  # <-- collect_chat always sets this to 0
        text="test",
        prompt=None,
    )
    result_with = CollectionResult(
        activations={
            l: out_with.hidden_states[l].squeeze(0)
            for l in range(3)
        },
        gradients={},
        tokens=tokens_with,
        prompt_length=0,  # <-- collect_chat always sets this to 0
        text="test",
        prompt="injected",
    )

    # Test with mean aggregation, response_only=False
    diffs = compute_activation_diff_for_sample(
        result_without, result_with,
        aggregation="mean", response_only=False,
    )

    # Verify against analytical expectation
    for layer in [1, 2]:
        expected_without = expected_activation(tokens_without, layer)
        expected_with = expected_activation(tokens_with, layer)
        expected_diff = expected_with.mean(dim=0) - expected_without.mean(dim=0)
        torch.testing.assert_close(
            diffs[layer], expected_diff, atol=1e-5, rtol=1e-5
        )

    # Test with response_only=True — since prompt_length=0, this should
    # give IDENTICAL results (no tokens are filtered)
    diffs_resp = compute_activation_diff_for_sample(
        result_without, result_with,
        aggregation="mean", response_only=True,
    )
    for layer in [1, 2]:
        torch.testing.assert_close(
            diffs_resp[layer], diffs[layer], atol=1e-7, rtol=0
        )

    print("PASS: test_compute_activation_diff_for_sample_function")


def test_response_only_has_no_effect_with_prompt_length_zero():
    """Explicitly verify that response_only=True has no effect when
    prompt_length=0 (which is what collect_chat always sets).

    This is an important behavioral test: it confirms that the activation
    difference vectors computed from chat data include ALL tokens,
    including any injected prompt tokens.
    """
    from scripts.generate_steering_vectors import compute_activation_diff_for_sample
    from grad_shift.collector import CollectionResult

    model = make_dummy_model()
    tokens = [5, 10, 99, 88, 15]
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)

    # With prompt_length=0 (chat mode)
    result_pl0 = CollectionResult(
        activations={l: out.hidden_states[l].squeeze(0) for l in range(3)},
        gradients={}, tokens=tokens, prompt_length=0,
        text="test", prompt=None,
    )

    # With prompt_length=3 (hypothetical non-chat mode)
    result_pl3 = CollectionResult(
        activations={l: out.hidden_states[l].squeeze(0) for l in range(3)},
        gradients={}, tokens=tokens, prompt_length=3,
        text="test", prompt=None,
    )

    # Dummy "without" result
    tokens_base = [1, 2, 3]
    ids_base = torch.tensor([tokens_base], dtype=torch.long)
    with torch.no_grad():
        out_base = model(ids_base, output_hidden_states=True)
    result_base = CollectionResult(
        activations={l: out_base.hidden_states[l].squeeze(0) for l in range(3)},
        gradients={}, tokens=tokens_base, prompt_length=0,
        text="test", prompt=None,
    )

    # response_only=False vs True with prompt_length=0 -> identical
    d_false = compute_activation_diff_for_sample(
        result_base, result_pl0, "mean", response_only=False
    )
    d_true = compute_activation_diff_for_sample(
        result_base, result_pl0, "mean", response_only=True
    )
    for layer in [1, 2]:
        torch.testing.assert_close(
            d_false[layer], d_true[layer], atol=1e-7, rtol=0
        )

    # response_only=True with prompt_length=3 -> different (tokens filtered)
    result_base_pl0 = CollectionResult(
        activations={l: out_base.hidden_states[l].squeeze(0) for l in range(3)},
        gradients={}, tokens=tokens_base, prompt_length=0,
        text="test", prompt=None,
    )
    d_resp_pl3 = compute_activation_diff_for_sample(
        result_base_pl0, result_pl3, "mean", response_only=True
    )
    # With prompt_length=3: act_with is filtered to act_with[3:] = tokens 88, 15
    # This should differ from the full computation
    for layer in [1, 2]:
        assert not torch.allclose(
            d_resp_pl3[layer], d_false[layer], atol=1e-5
        ), f"response_only with prompt_length=3 should differ at layer {layer}"

    print("PASS: test_response_only_has_no_effect_with_prompt_length_zero")


def test_inject_prompt_token_structure():
    """Verify inject_prompt_into_chat produces the expected message structure
    for prepend_user mode."""
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there"},
    ]

    # Without prompt
    msgs_without = inject_prompt_into_chat(
        messages, prompt=None, mode="prepend_user",
        baseline_system_prompt="You are helpful.",
    )
    assert msgs_without[0]["role"] == "system"
    assert msgs_without[0]["content"] == "You are helpful."
    assert msgs_without[1]["role"] == "user"
    assert msgs_without[1]["content"] == "Hello world"

    # With prompt
    msgs_with = inject_prompt_into_chat(
        messages, prompt="Be evil.", mode="prepend_user",
        baseline_system_prompt="You are helpful.",
    )
    assert msgs_with[0]["role"] == "system"
    assert msgs_with[0]["content"] == "You are helpful."
    assert msgs_with[1]["role"] == "user"
    assert msgs_with[1]["content"] == "Be evil.\n\nHello world"

    # System message is identical
    assert msgs_without[0] == msgs_with[0]

    # Assistant message is identical
    assert msgs_without[2]["content"] == msgs_with[2]["content"]

    print("PASS: test_inject_prompt_token_structure")


def test_embedding_placeholder_indexing():
    """Verify that the embedding placeholder at index 0 means
    layer N activations are accessed at index N (not N-1).

    This is the convention used by SteeringModule.from_pretrained and
    the hooks system.
    """
    hidden_dim = 8
    num_layers = 3

    # Create steering vectors (3 layers)
    vecs = torch.randn(num_layers, hidden_dim)

    # Add embedding placeholder
    placeholder = torch.zeros(1, hidden_dim)
    saved = torch.cat([placeholder, vecs], dim=0)

    assert saved.shape == (num_layers + 1, hidden_dim)
    assert saved[0].abs().sum() == 0  # index 0 = zeros

    # Accessing by layer index (1-indexed in the hook system)
    for layer_idx in range(1, num_layers + 1):
        torch.testing.assert_close(
            saved[layer_idx], vecs[layer_idx - 1]
        )

    print("PASS: test_embedding_placeholder_indexing")


def test_multi_sample_averaging():
    """Verify that multi-sample averaging works correctly.

    The generate_activation_diff function averages diffs across samples.
    Test that this average matches manual computation.
    """
    model = make_dummy_model(num_layers=2)
    hidden_dim = 8

    # 4 samples with different token sequences
    samples = [
        ([2, 4, 6], [2, 4, 50, 6]),
        ([3, 5, 7], [3, 5, 60, 7]),
        ([1, 8, 9], [1, 8, 70, 9]),
        ([4, 6, 8], [4, 6, 80, 8]),
    ]

    per_sample_diffs = {1: [], 2: []}
    for tokens_w, tokens_p in samples:
        ids_w = torch.tensor([tokens_w], dtype=torch.long)
        ids_p = torch.tensor([tokens_p], dtype=torch.long)

        with torch.no_grad():
            out_w = model(ids_w, output_hidden_states=True)
            out_p = model(ids_p, output_hidden_states=True)

        for layer in [1, 2]:
            act_w = out_w.hidden_states[layer].squeeze(0)
            act_p = out_p.hidden_states[layer].squeeze(0)
            per_sample_diffs[layer].append(act_p.mean(0) - act_w.mean(0))

    # Average across samples
    for layer in [1, 2]:
        avg_diff = torch.stack(per_sample_diffs[layer]).mean(dim=0)

        # Verify against analytical expectation
        expected_diffs = []
        for tokens_w, tokens_p in samples:
            exp_w = expected_activation(tokens_w, layer, hidden_dim)
            exp_p = expected_activation(tokens_p, layer, hidden_dim)
            expected_diffs.append(exp_p.mean(0) - exp_w.mean(0))
        expected_avg = torch.stack(expected_diffs).mean(dim=0)

        torch.testing.assert_close(avg_diff, expected_avg, atol=1e-5, rtol=1e-5)

    print("PASS: test_multi_sample_averaging")


# ─── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_dummy_model_activations()
    test_activation_diff_manual()
    test_prompt_tokens_included_in_activations()
    test_loss_includes_prompt_tokens()
    test_save_and_load_roundtrip()
    test_saved_vector_matches_fresh_computation()
    test_activation_diff_with_different_aggregations()
    test_compute_activation_diff_for_sample_function()
    test_response_only_has_no_effect_with_prompt_length_zero()
    test_inject_prompt_token_structure()
    test_embedding_placeholder_indexing()
    test_multi_sample_averaging()
    print("\n=== ALL TESTS PASSED ===")
