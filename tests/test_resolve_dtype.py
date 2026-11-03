#!/usr/bin/env python3
"""Regression tests for ActivationCollector dtype resolution.

Six analysis scripts passed `getattr(torch, args.dtype)` (a torch.dtype
object) into ActivationCollector's string-keyed dtype map; `.get(dtype,
torch.float16)` silently fell back to fp16 for every one of them. The fix
(grad_shift.collector.resolve_dtype) accepts both forms and raises on
anything else instead of silently downgrading.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grad_shift.collector import resolve_dtype


def test_accepts_string_names():
    assert resolve_dtype("float16") is torch.float16
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32


def test_accepts_torch_dtype_objects():
    """The exact call pattern the analysis scripts use."""
    for name in ("float16", "bfloat16", "float32"):
        obj = getattr(torch, name)
        assert resolve_dtype(obj) is obj


def test_torch_dtype_no_longer_falls_back_to_fp16():
    """Regression: torch.bfloat16 used to key-miss into the fp16 default."""
    assert resolve_dtype(torch.bfloat16) is not torch.float16


def test_unknown_string_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("bf16")


def test_none_raises():
    with pytest.raises(ValueError):
        resolve_dtype(None)
