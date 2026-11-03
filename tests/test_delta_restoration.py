"""Regression tests for exact, exception-safe delta weight restoration."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from grad_shift.delta import compute_delta_inplace, save_model_weights


class TinyResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.final_norm = nn.LayerNorm(8)
        self.call_count = 0
        self.fail_on_call = None

    def forward(self, input_ids, output_hidden_states=False, **_kwargs):
        self.call_count += 1
        if self.call_count == self.fail_on_call:
            raise RuntimeError("injected forward failure")
        hidden = self.embed(input_ids)
        states = [hidden]
        for layer in self.model.layers:
            hidden = hidden + torch.tanh(layer(hidden))
            states.append(hidden)
        # Match architectures whose last hidden-state entry is post-norm;
        # compute_delta_inplace should use the last block's raw hook output.
        states[-1] = self.final_norm(states[-1])
        return SimpleNamespace(hidden_states=tuple(states))


def _model_with_grads():
    torch.manual_seed(0)
    model = TinyResidualModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    model(input_ids, output_hidden_states=True).hidden_states[-1][..., 0].sum().backward()
    model.call_count = 0
    return model, input_ids


def _parameter_copies(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def test_cpu_weight_snapshot_does_not_alias_model():
    model, _ = _model_with_grads()
    snapshot = save_model_weights(model)
    name, parameter = next(iter(model.named_parameters()))
    before = snapshot[name].clone()

    parameter.data.add_(1)

    assert torch.equal(snapshot[name], before)
    assert snapshot[name].data_ptr() != parameter.data_ptr()


def test_compute_delta_inplace_restores_cpu_weights_exactly():
    model, input_ids = _model_with_grads()
    before = _parameter_copies(model)

    deltas = compute_delta_inplace(model, input_ids, layers=[0, 2], lr=0.1)

    assert deltas[0].abs().sum() > 0
    assert deltas[2].abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
    assert not model.model.layers[-1]._forward_hooks


def test_compute_delta_inplace_restores_after_forward_error():
    model, input_ids = _model_with_grads()
    before = _parameter_copies(model)
    model.fail_on_call = 2

    with pytest.raises(RuntimeError, match="injected forward failure"):
        compute_delta_inplace(model, input_ids, layers=[2], lr=0.1)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
    assert not model.model.layers[-1]._forward_hooks


@pytest.mark.parametrize("layers", [[-1], [3], [1, 1]])
def test_compute_delta_inplace_rejects_ambiguous_layers(layers):
    model, input_ids = _model_with_grads()
    with pytest.raises(ValueError):
        compute_delta_inplace(model, input_ids, layers=layers, lr=0.1)
