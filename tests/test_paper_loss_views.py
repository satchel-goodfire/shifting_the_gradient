"""Tests for the loss-views analyses' pure helpers (products/loss_views.py)."""

import numpy as np
import pytest

from grad_shift.paper.products import loss_views as lv
from grad_shift.paper.products.loss_measure import SEP


def test_view_conditions_cover_every_prompt_and_intensity_once():
    conds = lv.view_conditions(lv.EVIL)
    names = [c.name for c in conds]
    assert len(names) == len(set(names))
    n_prompts = sum(len(lv.member_names(c)) for c in lv.PROMPT_SETS)
    assert len(conds) == n_prompts * len(lv.VIEW_EPS)
    for c in conds:  # unsteered at 0 (guards the tokenization), else steered
        assert (c.vector is None) == (c.coef == 0)
    assert {c.prompt for c in conds if c.name.startswith("none_")} == {None}


def test_loss_minimum_is_the_vertex_through_the_grid_minimum():
    xs = [-1.0, 0.0, 0.5, 2.0]  # unequal spacing
    ys = [(x - 0.3) ** 2 + 1 for x in xs]
    grid, vertex = lv.loss_minimum(xs, ys)
    assert grid == 0.5 and vertex == pytest.approx(0.3)
    assert lv.loss_minimum([0.0, 1.0, 2.0], [3.0, 2.0, 1.0]) == (2.0, None)


def test_downward_crossings_interpolate_plus_to_minus_only():
    xs = [-1.0, 0.0, 1.0, 2.0]
    assert lv.downward_crossings(xs, [0.2, 0.1, -0.1, 0.3]) == [
        (0.0, 1.0, 0.5)]  # -0.1 -> 0.3 rises: not a minimum


def fake_arrays(members, eps, v_norm=2.0, n=5):
    """Arrays where every sequence's loss is (e - 0.4)^2 + k and the summed
    projection is its exact derivative / |v|."""
    arrs = {}
    for m in members:
        for e in eps:
            name = lv.cond_name(m, e)
            loss = (e - 0.4) ** 2 + np.arange(n, dtype=float)
            arrs[f"{name}{SEP}seq_loss_all"] = loss
            arrs[f"{name}{SEP}seq_loss_full"] = loss
            arrs[f"{name}{SEP}gproj_sum"] = np.full(n, 2 * (e - 0.4) / v_norm)
            # raw cosine has the derivative's sign: paper sign flips it
            arrs[f"{name}{SEP}gcos_sum"] = np.full(n, (e - 0.4) * 10)
            arrs[f"{name}{SEP}gcos_n"] = np.full(n, 100.0)
    return arrs


def test_curve_and_exactness_on_a_planted_quadratic():
    eps = lv.VIEW_EPS
    c = lv.curve(fake_arrays(["evil0", "evil1"], eps), ["evil0", "evil1"], eps)
    xs = sorted(c)
    _, vertex = lv.loss_minimum(xs, [c[x]["loss"] for x in xs])
    (cross,) = lv.downward_crossings(xs, [c[x]["gcos"] for x in xs])
    assert vertex == pytest.approx(0.4) and cross[2] == pytest.approx(0.4)
    r, rel = lv.exactness(c, v_norm=2.0)
    assert r == pytest.approx(1.0) and rel == pytest.approx(0.0, abs=1e-9)
    # a projection off by a factor fails the check's relative error
    _, rel_bad = lv.exactness(c, v_norm=3.0)
    assert rel_bad == pytest.approx(0.5)
