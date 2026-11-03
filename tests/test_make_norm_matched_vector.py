"""Tests for scripts/make_norm_matched_vector.py."""

import importlib.util
import os

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "make_norm_matched_vector",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "make_norm_matched_vector.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
norm_match = _mod.norm_match


def test_norms_match_reference_and_direction_is_preserved():
    g = torch.Generator().manual_seed(0)
    src = torch.randn(5, 16, generator=g) * 30
    ref = torch.randn(5, 16, generator=g)
    ref[0] = 0  # embedding placeholder row
    out = norm_match(src, ref)
    assert torch.allclose(out.norm(dim=-1), ref.norm(dim=-1), atol=1e-5)
    cos = torch.nn.functional.cosine_similarity(out[1:], src[1:], dim=-1)
    assert torch.allclose(cos, torch.ones(4), atol=1e-6)
    assert out[0].abs().sum() == 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        norm_match(torch.ones(3, 4), torch.ones(3, 5))


def test_zero_source_row_with_nonzero_reference_raises():
    src = torch.ones(2, 4)
    src[1] = 0
    with pytest.raises(ValueError):
        norm_match(src, torch.ones(2, 4))
