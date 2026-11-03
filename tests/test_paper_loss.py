"""Sample-level loss products and their measurement helpers: pure logic and
assembly on fabricated data."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import ip_bank
from grad_shift.paper.products import loss_fig as lf
from grad_shift.paper.products import loss_measure as lm
from grad_shift.paper.runs import P1
from paper_testutil import assemble_fabricated as assemble


# ---------------------------------------------------------------- alignment

def test_split_injection_pure_insertion():
    base = [1, 2, 3, 4, 5]
    prompted = [1, 2, 9, 9, 9, 3, 4, 5]
    lcp, lcs = lm.split_injection(base, prompted)
    assert (lcp, lcs) == (2, 3)
    keep = lm.original_positions(len(prompted), lcp, lcs)
    assert [prompted[i] for i in keep] == base


def test_split_injection_identical_and_seam_drift():
    assert lm.split_injection([1, 2, 3], [1, 2, 3]) == (3, 0)
    # a BPE merge at the seam changes one base token: it is reported as lost
    base = [1, 2, 3, 4]
    prompted = [1, 9, 9, 7, 4]
    lcp, lcs = lm.split_injection(base, prompted)
    assert len(base) - lcp - lcs == 2  # tokens 2 and 3 do not survive


def test_split_injection_suffix_never_overlaps_prefix():
    lcp, lcs = lm.split_injection([5, 5], [5, 5, 5])
    assert lcp + lcs <= 2


# ---------------------------------------------------------------- statistics

def _tokens(n_groups=60, per=40, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    ref = rng.gamma(2.0, 1.0, size=len(groups))
    is_trait = rng.random(len(groups)) < 0.2
    return rng, groups, ref, is_trait


def test_matched_controls_disjoint_and_equal_size_per_decile():
    _, _, ref, is_trait = _tokens()
    ctrl = lm.matched_controls(ref, is_trait)
    assert not (ctrl & is_trait).any()
    dec = pd.qcut(ref, 10, labels=False)
    for d in range(10):
        assert (ctrl & (dec == d)).sum() == min(
            (is_trait & (dec == d)).sum(), (~is_trait & (dec == d)).sum())


def test_selectivity_recovers_planted_trait_shift():
    # an object that lowers the loss on trait tokens 0.3 nats more than on
    # matched controls is selective: positive (2026-10-21 convention)
    rng, groups, ref, is_trait = _tokens()
    delta = rng.normal(-0.1, 0.05, len(ref)) - 0.3 * is_trait
    s = lm.selectivity(delta, ref, is_trait, groups, n_boot=500)
    assert s["selectivity"] == pytest.approx(0.3, abs=0.03)
    assert s["lo"] < s["selectivity"] < s["hi"]
    assert s["delta_control"] == pytest.approx(-0.1, abs=0.02)
    assert s["delta_trait"] == pytest.approx(-0.4, abs=0.03)


def test_legacy_selectivity_records_convert_exactly():
    # records cached as trait - control (before 2026-10-21) -> current sign;
    # the bootstrap CI of a negated statistic is the negated, swapped CI
    rng, groups, ref, is_trait = _tokens(seed=3)
    delta = rng.normal(0, 0.2, len(ref)) - 0.2 * is_trait
    new = lm.selectivity(delta, ref, is_trait, groups, n_boot=200)
    old = {**new, "selectivity": new["delta_trait"] - new["delta_control"],
           "lo": -new["hi"], "hi": -new["lo"]}
    assert lm.current_selectivity(old, None) == pytest.approx(new)
    assert lm.current_selectivity(new, lm.SELECTIVITY_SIGN) == new
    with pytest.raises(ValueError):  # not a trait - control record
        lm.current_selectivity(new, None)


def test_selectivity_null_when_no_trait_effect():
    rng, groups, ref, is_trait = _tokens(seed=1)
    delta = rng.normal(0.2, 0.1, len(ref))
    s = lm.selectivity(delta, ref, is_trait, groups, n_boot=500)
    assert s["lo"] <= 0 <= s["hi"]


def test_selectivity_point_matches_unvectorized_definition():
    rng, groups, ref, is_trait = _tokens(seed=2)
    delta = rng.normal(0, 1, len(ref))
    ctrl = lm.matched_controls(ref, is_trait)
    s = lm.selectivity(delta, ref, is_trait, groups, n_boot=10)
    assert s["selectivity"] == pytest.approx(
        delta[ctrl].mean() - delta[is_trait].mean())


def test_bank_conditions_cover_every_prompt_and_p1_is_a_member():
    conds = lm.bank_conditions()
    assert [c.name for c in conds] == ["ip_b0", "ip_b1", "ip_b2"]
    assert len({c.prompt for c in conds}) == 3
    assert conds[ip_bank.p1_index()].prompt == lm.read_prompts(P1)[0]
    pp = lm.bank_conditions("ippps_c2", lm.EVIL, 2.0)
    assert all(c.vector == lm.EVIL and c.coef == 2.0 for c in pp)
    assert [c.name for c in pp][0] == "ippps_c2_b0"


def test_bank_fields_average_members_and_copy_p1():
    arrs = {}
    for k, m in enumerate(ip_bank.members()):
        arrs[f"{m}{lm.SEP}tok_loss"] = np.array([k, 2.0 * k])
        arrs[f"{m}{lm.SEP}n_drift"] = np.array([0, k])
    ip_bank.add_bank_fields(arrs, lm.SEP)
    assert np.allclose(arrs[f"ip{lm.SEP}tok_loss"], [1.0, 2.0])
    assert list(arrs[f"ip{lm.SEP}n_drift"]) == [0, 2]  # max, not mean
    p1 = ip_bank.p1_index()
    assert np.allclose(arrs[f"ip_p1{lm.SEP}tok_loss"], [p1, 2.0 * p1])


def test_coef_names_are_npz_safe():
    assert lm.coef_name("pps", -1.5) == "pps_cm1.5"
    assert lm.coef_name("ippps", 9.0) == "ippps_c9"


# ---------------------------------------------------------------- assembly

def fake_condition(rng, n_seq, n_tok, loss, gcos, grad=True, proj=False):
    rec = {"seq_loss_all": loss + rng.normal(0, 0.05, n_seq),
           "seq_loss_resp": loss + rng.normal(0, 0.05, n_seq),
           "tok_loss": (loss + rng.normal(0, 0.5, n_tok)).astype(np.float32),
           "n_drift": np.zeros(n_seq, int)}
    if grad:
        rec["gcos_n"] = np.full(n_seq, 100.0)
        rec["gcos_sum"] = rec["gcos_n"] * gcos
    if proj:
        rec["out_proj"] = rng.normal(0, 1, n_tok).astype(np.float32)
    return rec


def fake_sweep_arrays(seed=0):
    """Arrays shaped like loss_base_sweep's, with a low-eps loss dip."""
    tt = lm.load_trait_tokens()
    n_seq, n_tok = tt["sample_id"].nunique(), len(tt)
    rng = np.random.default_rng(seed)
    out = {}

    def add(name, loss, gcos, **kw):
        for k, v in fake_condition(rng, n_seq, n_tok, loss, gcos, **kw).items():
            out[f"{name}{lm.SEP}{k}"] = v

    add("none", 3.7, -0.012, proj=True)
    for m in ip_bank.members():
        add(m, 3.2, 0.001)
    for c in lf.SWEEP:
        bump = 0.1 * (c - 2) ** 2 - 0.4  # minimum at c = 2, < 0 on (0, 4)
        add(lm.coef_name("pps", c), 3.7 + bump, -0.012 + 0.003 * c)
        for m in ip_bank.members(lm.coef_name("ippps", c)):
            add(m, 3.2 + bump, 0.001 + 0.003 * c)
    add("pps_rand", 3.75, -0.011)
    for c in lf.POS_SWEEP:
        add(lf.rand_name(c), 3.75, 0.0, grad=False)
    for k in range(lf.N_NEUTRAL):
        add(f"neut{k}", 3.7, 0.0, grad=False)
    for b in lf.sweep_banks():  # what measure(..., banks=...) adds
        ip_bank.add_bank_fields(out, lm.SEP, bank=b)
    return out


def _csv(path, evil, n=20, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"question": [f"q{i % 5}" for i in range(n)],
                  "evil": np.clip(evil + rng.normal(0, 1, n), 0, 100),
                  "coherence": [90.0] * n}).to_csv(path, index=False)


LOSS_RUNS = sorted(set(lf.LossSweep.runs))


def setup_out(tmp_path, runs):
    out = tmp_path / "out"
    os.makedirs(out)
    (out / "plan.json").write_text(json.dumps(
        [{"name": r, "identity": f"id_{i}"} for i, r in enumerate(runs)]))
    for i, r in enumerate(runs):
        _csv(str(out / "evals" / f"id_{i}" / "trait_eval_evil_extract.csv"),
             90.0 if r.endswith("none") else 30.0, seed=i)
    adir = out / "analyses" / "loss_base_sweep"
    os.makedirs(adir)
    npz = str(adir / "arrays.npz")
    lm.save_npz(npz, fake_sweep_arrays())
    (adir / "result.json").write_text(json.dumps({"npz": npz}))
    return out


def test_loss_products_build_and_report_planted_effects(tmp_path):
    out = setup_out(tmp_path, LOSS_RUNS)
    results, skipped = assemble(str(out))
    for name in ("appx_loss_scatter", "appx_loss_sweep"):
        assert name not in skipped, skipped.get(name)
    assert (out / "figures" / "appx_loss_scatter.pdf").exists()
    assert (out / "figures" / "appx_loss_sweep.pdf").exists()
    assert results["fig4.loss.ip"].value < results["fig4.loss.none"].value
    assert results["fig4.frac_reduced.ip"].value > 90
    assert results["fig4.sweep.argmin"].value == pytest.approx(2.0)
    assert results["fig4.sweep.max_reducing"].value == pytest.approx(3.0)
    assert results["fig4.expr.eps0"].value > results["fig4.expr.eps9"].value
    assert results["fig4.expr.ip_p1"].value == pytest.approx(30.0, abs=1)
    # the token-level (explaining-away) products are not part of this release
    assert not any(k.startswith("sec5.") for k in results)


def _kept_figure(monkeypatch, product, ctx):
    """Build ``product`` and return its figure instead of saving it."""
    from grad_shift.paper import paper_style as ps
    kept = []

    def keep(fig, path, formats=("pdf", "png")):
        fig.draw_without_rendering()
        kept.append(fig)

    monkeypatch.setattr(ps, "save", keep)
    product.build(ctx)
    (fig,) = kept
    return fig


def test_appendix_loss_panels_as_specified(tmp_path, monkeypatch):
    """The loss scatter: (a) PPS and (b) IP total loss with vs without the
    defense. The sweep and the training-intensity panel are the other
    figure, whose sweep is one twin-axis plot (loss left, gradient cosine
    right)."""
    from grad_shift.paper.assemble import load_context
    out = setup_out(tmp_path, LOSS_RUNS)
    ctx = load_context(str(out))
    fig = _kept_figure(monkeypatch, lf.LossScatter(), ctx)
    a, b = fig.axes
    assert [ax.get_title(loc="left")[:3] for ax in (a, b)] == ["(a)", "(b)"]
    assert a.get_xlabel() == b.get_xlabel() == "total loss, defended"
    assert a.get_ylabel() == "total loss, undefended"

    fig = _kept_figure(monkeypatch, lf.LossSweep(), ctx)
    (left, right, expr) = fig.axes
    assert left.get_shared_x_axes().joined(left, right)  # twinx
    assert "loss" in left.get_ylabel() and "cosine" in right.get_ylabel()
    assert {ln.get_label() for ln in left.get_lines()} >= {
        "PPS, loss", "PPS + IP, loss"}
    assert {ln.get_label() for ln in right.get_lines()} >= {
        "PPS, gcos", "PPS + IP, gcos"}
    assert expr.get_xlabel() == "training intensity ε"
