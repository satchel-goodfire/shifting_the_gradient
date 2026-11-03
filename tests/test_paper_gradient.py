"""Group B (gradient analyses): pure helpers and products on fabricated data.

No GPU: analyses are exercised only through the files they write, which are
fabricated here with known structure so each product's numbers can be
checked against ground truth.
"""

import json
from artifact_testutil import write_adapter
import os

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import gradient_actdiff as ga
from grad_shift.paper.products import gradient_core as gc
from grad_shift.paper.products import gradient_delta as gdl
from grad_shift.paper.products import gradient_dynamics as gdy
from grad_shift.paper.products import gradient_step0 as g0
from grad_shift.paper.products.base import Context

D = 16


# ---------------------------------------------------------------- helpers

def test_alignment_drops_inserted_prompt_tokens():
    base = [1, 2, 3, 9, 9, 4, 5]
    pr = [1, 2, 3, 7, 7, 7, 9, 9, 4, 5]
    bi, di = gc.aligned_positions(base, pr, "nonprompt")
    assert [base[i] for i in bi] == [pr[i] for i in di] == base
    bi, di = gc.aligned_positions(base, pr, "suffix")
    assert [base[i] for i in bi] == [9, 9, 4, 5]
    assert list(di) == [6, 7, 8, 9]


def test_alignment_identity_and_overlap_cap():
    bi, di = gc.aligned_positions([1, 2, 3], [1, 2, 3])
    assert list(bi) == list(di) == [0, 1, 2]
    # repeated tokens: prefix + suffix never double-count a position
    p, s = gc.common_prefix_suffix([5, 5, 5], [5, 5, 5, 5])
    assert p + s <= 3


def test_loss_positions_shift_by_one():
    assert list(gc.loss_positions(np.array([0, 1, 2, 5]))) == [0, 1, 4]


def test_split_is_deterministic_and_disjoint():
    a1, b1 = gc.split_indices(4681, 2340, 42)
    a2, b2 = gc.split_indices(4681, 2340, 42)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 2340 and len(b1) == 2341
    assert not set(a1) & set(b1)
    assert sorted(a1 + b1) == list(range(4681))


def test_uncentered_pca_recovers_planted_direction():
    rng = np.random.default_rng(0)
    u = gc.unit(rng.normal(size=D))
    # constant shift along u plus small isotropic noise: PC1 ~ u and it
    # dominates the (uncentered) variance
    diffs = 3.0 * u + 0.05 * rng.normal(size=(200, D))
    refs = {"evil": u, "other": gc.unit(rng.normal(size=D))}
    r = gc.uncentered_pca(diffs, refs, k=3)
    assert r["frac_var"][0] > 0.95
    assert r["pc_cos"]["evil"][0] > 0.99  # sign-aligned to the reference
    assert r["meandiff_cos"]["evil"] > 0.99
    assert abs(sum(r["frac_var"]) - 1) < 1e-9
    assert r["n_to_threshold"] == 1


def test_uncentered_pca_sign_convention():
    u = np.eye(D)[0]
    diffs = -np.tile(u, (10, 1)) + 1e-3 * np.eye(D)[1]
    r = gc.uncentered_pca(diffs, {"evil": u}, k=1)
    assert r["pc_cos"]["evil"][0] > 0  # flipped to face the reference
    assert r["meandiff_cos"]["evil"] < 0  # the mean itself is not flipped


def test_mean_sem():
    m, e = gc.mean_sem([1.0, 2.0, 3.0, np.nan])
    assert m == 2.0 and abs(e - 1 / np.sqrt(3)) < 1e-12


def test_step0_conditions_cover_figure_arms():
    names = [c.name for c in g0.conditions()]
    for n in ("none", "pps", "pps_rand", "pps_eps9", "ip_b0", "ip_b1",
              "ip_b2", "ip_neutral0", "ip_neutral3", "pps_l4", "pps_l24"):
        assert n in names
    assert "pps_l20" not in names  # layer 20 is the main PPS arm
    assert "ip" not in names  # the IP arm is the average of the bank members
    assert len(set(names)) == len(names)


def test_ip_bank_conditions_are_the_bank_prompts():
    from grad_shift.paper.products import ip_bank
    from grad_shift.paper.runs import IP_BANK, P1

    conds = gc.ip_bank_conditions()
    assert [c.prompt for c in conds] == ip_bank.read_prompts(IP_BANK)
    assert conds[ip_bank.p1_index()].prompt == gc.prompt_of(P1)
    assert gc.ip_condition().name == "ip_p1"


def test_add_bank_arms_averages_per_sequence():
    from grad_shift.paper.products import ip_bank

    per = {m: np.full((2, 3), float(k)) for k, m in
           enumerate(ip_bank.members())}
    per["none"] = np.zeros((2, 3))
    ip_bank.add_bank_arms(per)
    assert np.allclose(per["ip"], 1.0)  # mean of 0, 1, 2
    assert np.allclose(per["ip_p1"], float(ip_bank.p1_index()))
    with pytest.raises(ValueError):
        ip_bank.average([np.zeros(2), np.zeros(3)])


def test_dynamics_checkpoint_listing(tmp_path):
    run = tmp_path / "run"
    for s in (800, 400):
        (run / f"checkpoint-{s}").mkdir(parents=True)
        write_adapter(run / f"checkpoint-{s}")
    (run / "checkpoint-1200").mkdir()  # incomplete: no adapter
    data = tmp_path / "d.jsonl"
    data.write_text("{}\n" * 2000)
    (run / "experiment_config.json").write_text(json.dumps(
        {"train_file": str(data), "batch_size": 4, "num_epochs": 3}))
    steps = [s for s, _ in gdy.checkpoints(str(run))]
    assert steps == [0, 400, 800, 1500]  # final = ceil(2000/4) * 3


def test_dynamics_base_class_is_not_discovered():
    from grad_shift.paper.products import discover

    analyses, products = discover()
    assert "" not in analyses
    for n in ("gradient_step0", "gradient_delta", "gradient_actdiff",
              "gradient_dynamics_none", "gradient_dynamics_pps",
              "gradient_dynamics_ip"):
        assert n in analyses


def test_actdiff_run_declared_with_existing_vector():
    from grad_shift.paper.runs import IP_BANK, P1, all_runs
    from grad_shift.paper.spec import REPO_ROOT

    for run, vec, prompts in ((ga.RUN, ga.ACTDIFF, IP_BANK),
                              (ga.RUN_P1, ga.ACTDIFF_P1, P1)):
        spec = all_runs()[run]
        assert spec.pps_vector == vec
        assert spec.pps_coef == 1.0 and len(spec.pps_layers) == 28
        assert os.path.isfile(os.path.join(REPO_ROOT, spec.pps_vector))
        meta = ga.actdiff_meta(vec)
        assert meta["prompt_file"] == prompts
        assert meta["split"]["n_build"] == 2340
        assert meta["split"]["n_heldout"] == 2341
    held = ga.heldout_indices()
    assert len(held) == 2341


def test_actdiff_token_pooled_weights_prompts_equally():
    # member 0: 10 tokens of cosine 0.1; member 1: 30 tokens of cosine 0.5
    sums = np.zeros((2, 1, 2, 1))
    sums[0, 0, :, 0] = (0.5, 0.5)  # 2 seqs x 5 tokens x 0.1
    sums[1, 0, :, 0] = (7.5, 7.5)  # 2 seqs x 15 tokens x 0.5
    counts = np.array([[5, 5], [15, 15]])
    assert ga.token_pooled(sums, counts)[0, 0] == pytest.approx(0.3)


# ---------------------------------------------------------------- products

def _ctx(tmp_path, runs=()):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    plan = [{"name": r, "identity": f"id{i}"} for i, r in enumerate(runs)]
    (out / "plan.json").write_text(json.dumps(plan))
    return Context(out=str(out), run_keys={p["name"]: p["identity"]
                                           for p in plan})


def _write_analysis(ctx, name, result, **arrays):
    d = ctx.analysis_dir(name)
    if arrays:
        np.savez(os.path.join(d, "per_sequence.npz"), **arrays)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(result, f)


def _fake_step0(ctx, n=50):
    """Planted truth: PPS flips the evil-direction cosine from -0.01 to
    +0.02; IP moves it to 0; PPS-minus-none mean grads lie along the evil
    PV at layer 20."""
    rng = np.random.default_rng(1)
    conds = g0.conditions()
    names = [c.name for c in conds] + ["ip", "ip_p1"]  # + averaged arms
    L, nd = len(gc.LAYERS), len(g0.DIR_NAMES)
    shift = {"none": -0.01, "pps": 0.02, "ip": 0.0, "ip_p1": 0.0,
             "pps_rand": -0.01, "pps_eps9": 0.03}
    arrays = {}
    evil20 = gc.load_vector_file(gc.EVIL)[gc.LAYER]
    hid = evil20.shape[0]
    base_g = rng.normal(size=(n, hid))
    for c in names:
        tm = 0.001 * rng.normal(size=(n, L, nd))
        tm[:, :, 0] += shift.get(c, -0.012)
        arrays[f"tokmean|{c}"] = tm
        arrays[f"pooled|{c}"] = tm
        arrays[f"loss|{c}"] = np.ones(n)
        g = base_g.copy()
        if c.startswith("pps"):
            g += 5 * gc.unit(evil20)
        elif c.startswith("ip") and not c.startswith("ip_neutral"):
            g += 0.3 * rng.normal(size=(n, hid))
        arrays[f"meangrad_l20|{c}"] = g.astype(np.float32)
    _write_analysis(ctx, g0.NAME, {
        "npz": "per_sequence.npz", "n_seq": n, "layers": list(gc.LAYERS),
        "dir_names": list(g0.DIR_NAMES),
        "conditions": [c.describe() for c in conds]}, **arrays)


def test_step0_products(tmp_path):
    ctx = _ctx(tmp_path)
    _fake_step0(ctx)
    r = g0.GradientFlip().build(ctx)
    # keys are in the paper sign (positive = amplifying): the fabricated
    # raw cosines come back negated
    assert abs(r["gradflip.l20.pps.evil"].value + 0.02) < 1e-3
    assert abs(r["gradflip.l20.none.evil"].value - 0.01) < 1e-3
    assert r["gradflip.l20.pps.evil.fracatten"].value == 100.0
    assert abs(r["gradflip.l20.ipneutral.evil"].value - 0.012) < 1e-3
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_gradient_flip.pdf"))
    p = g0.GradDiffDecomp().build(ctx)
    assert p["sec4.3.pps.pc1_var"].value > 90
    assert p["sec4.3.pps.pc1_cos_evil"].value > 0.99
    # the PCs are oriented towards the evil PV (sign-free); the mean shift is
    # attenuating, i.e. negative in the paper sign
    assert p["sec4.3.pps.meandiff_cos_evil"].value < -0.99
    assert p["sec4.3.ip.pc1_var"].value < p["sec4.3.pps.pc1_var"].value
    assert p["sec4.3.ip_p1.pc1_var"].value < p["sec4.3.pps_eps9.pc1_var"].value
    assert abs(r["gradflip.l20.ppseps9.evil"].value + 0.03) < 1e-3
    g0.GradLayersPPS().build(ctx)
    g0.GradLayersIP().build(ctx)
    for stem in ("fig_grad_diff_decomp", "fig_grad_layers_pps",
                 "fig_grad_layers_ip"):
        assert os.path.isfile(os.path.join(ctx.figures_dir, stem + ".png"))


def test_dynamics_product(tmp_path):
    ctx = _ctx(tmp_path)
    for run in gdy.RUNS:
        summ = [{"step": s, "cond": c, "l20_evil_mean": v + 0.001 * s / 400,
                 "l20_evil_sem": 0.001}
                for s in (0, 400, 800) for c, v in
                (("none", -0.01), ("pps", 0.02), ("ip", 0.0))]
        _write_analysis(ctx, f"gradient_dynamics_{run}", {"summary": summ})
    r = gdy.GradDynamics().build(ctx)
    # paper sign (positive = amplifying): fabricated raw cosines negate
    assert abs(r["graddyn.pps.pps.step0"].value + 0.02) < 1e-12
    assert abs(r["graddyn.pps.pps.final"].value + 0.022) < 1e-12


def test_delta_products(tmp_path):
    ctx = _ctx(tmp_path)
    n, L, nd = 30, len(gc.LAYERS), len(gdl.DIR_NAMES)
    rng = np.random.default_rng(2)
    none = 0.01 * rng.normal(size=(n, L, nd))
    arrays = {"grad_delta_cos": -0.3 + 0.01 * rng.normal(size=(n, L)),
              "grad_dir_cos": none, "delta_dir_cos|none": none,
              "delta_dir_cos|pps": none - 0.05, "delta_dir_cos|ip": none,
              "delta_dir_cos_suffix_none|ip": none,
              "delta_dir_cos_suffix|ip": none - 0.02,
              "delta_dir_cos_suffix_none|ip_p1": none,
              "delta_dir_cos_suffix|ip_p1": none - 0.03}
    _write_analysis(ctx, gdl.NAME, {"npz": "per_sequence.npz", "n_seq": n,
                                    "lr": 1e-4, "layers": list(gc.LAYERS),
                                    "dir_names": list(gdl.DIR_NAMES)},
                    **arrays)
    r = gdl.GradDeltaLayers().build(ctx)
    # cos(-dL/dh, delta): the descent direction agrees with the change
    assert abs(r["delta.graddelta.l20"].value - 0.3) < 0.01
    r = gdl.DeltaScatterPPS().build(ctx)
    assert r["delta.pps.l20.frac_below"].value == 100
    assert abs(r["delta.pps.l20.shift"].value + 0.05) < 1e-9
    r = gdl.DeltaScatterIP().build(ctx)
    assert abs(r["delta.ip.l16.shift"].value + 0.02) < 1e-9
    assert abs(r["delta.ip_p1.l16.shift"].value + 0.03) < 1e-9


def test_actdiff_products(tmp_path):
    runs = [ga.RUN, ga.IP_RUN, ga.RUN_P1, ga.IP_P1_RUN]
    ctx = _ctx(tmp_path, runs)
    for i, (r, v) in enumerate(zip(runs, (0.5, 1.3, 0.7, 0.2))):
        p = ctx.eval_csv(r)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame({"evil": [v - 0.1, v + 0.1], "coherence": [80, 82]}
                     ).to_csv(p, index=False)
    t = np.zeros((len(ga.COEFS), len(gc.LAYERS)))
    t[ga.COEFS.index(0.0)] = 0.6
    t[ga.COEFS.index(1.0)] = 0.2
    t1 = t.copy()
    t1[ga.COEFS.index(1.0)] = 0.7
    _write_analysis(ctx, ga.NAME, {"n_seq": 200, "coefs": list(ga.COEFS),
                                   "layers": list(gc.LAYERS),
                                   "token_pooled_mean": {"bank": t.tolist(),
                                                         "p1": t1.tolist()}})
    r = ga.IPNotPPSGrad().build(ctx)
    assert abs(r["actdiff.gradcos.coef1.mean"].value - 0.2) < 1e-12
    assert r["actdiff.gradcos.n_layers_below_none"].value == 28
    assert r["actdiff.p1.gradcos.n_layers_below_none"].value == 0
    ga.IPvsActdiff().build(ctx)
    r = ga.ActdiffVecCos().build(ctx)
    assert -1 <= r["actdiff.veccos.l20"].value <= 1
    r = ga.ActdiffBehavior().build(ctx)
    assert abs(r["sec4.4.actdiff.evil"].value - 0.5) < 1e-12
    assert abs(r["sec4.4.ip.evil"].value - 1.3) < 1e-12
    assert abs(r["sec4.4.p1.actdiff.evil"].value - 0.7) < 1e-12
    assert abs(r["sec4.4.p1.ip.evil"].value - 0.2) < 1e-12


def test_macro_keys_unique_and_valid(tmp_path):
    """Every key the group's products emit is a valid, unique macro key."""
    from grad_shift.paper.macros import validate_key

    ctx = _ctx(tmp_path, [ga.RUN, "grid.base.heavy.ip"])
    _fake_step0(ctx)
    keys = []
    for prod in (g0.GradientFlip(), g0.GradDiffDecomp(),
                 g0.GradLayersPPS(), ga.ActdiffVecCos()):
        keys += list(prod.build(ctx))
    for k in keys:
        validate_key(k)
    assert len(keys) == len(set(keys))


def test_flip_figure_leaves_no_flip_quadrants_white(tmp_path, monkeypatch):
    """2026-10-26: no fill for the no-flip quadrants, so only the
    flip (success) and the reverse flip (failure) are shaded and keyed."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    from grad_shift.paper import paper_style as ps
    ctx = _ctx(tmp_path)
    _fake_step0(ctx)
    kept = []
    monkeypatch.setattr(ps, "save", lambda fig, path, formats=None: (
        fig.draw_without_rendering(), kept.append(fig)))
    g0.GradientFlip().build(ctx)
    (fig,) = kept
    for ax in fig.get_axes():
        fills = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(fills) == 2
        assert to_rgba(ps.SHADE) not in [p.get_facecolor() for p in fills]
    (key,) = fig.legends
    assert [t.get_text() for t in key.get_texts()] == [
        "flip (success)", "reverse flip (failure)"]
    ps.plt.close(fig)


def test_flip_figure_labels_the_signs_under_panel_a(tmp_path, monkeypatch):
    """2026-11-05: "attenuating" at the negative (left) end of panel (a)'s
    x axis and "amplifying" at the positive (right) end, under its tick
    labels, each with an outward arrow, in 5.5 pt (the authors' exception
    to the 7 pt floor); a collision or a too-small gap fails the build."""
    from grad_shift.paper import paper_style as ps
    ctx = _ctx(tmp_path)
    _fake_step0(ctx)
    kept = []
    monkeypatch.setattr(ps, "save", lambda fig, path, formats=None: (
        fig.draw_without_rendering(), kept.append(fig)))
    g0.GradientFlip().build(ctx)
    (fig,) = kept
    ax = fig.get_axes()[0]
    assert ax.get_title(loc="left").startswith("(a)")
    words = {t.get_text(): t for t in fig.texts}
    left, right = (words[w] for w in g0.SIGN_LABELS)
    # outward arrows: left of "attenuating", right of "amplifying"
    assert left.get_text().startswith("←")
    assert left.get_text().endswith("attenuating")
    assert right.get_text().startswith("amplifying")
    assert right.get_text().endswith("→")
    assert left.get_fontsize() == right.get_fontsize() == g0.SIGN_PT < 7
    r = fig.canvas.get_renderer()
    panel = ax.get_window_extent(r)
    lb, rb = left.get_window_extent(r), right.get_window_extent(r)
    assert lb.x0 == pytest.approx(panel.x0, abs=0.5)
    assert rb.x1 == pytest.approx(panel.x1, abs=0.5)
    assert lb.y1 < panel.y0 and rb.y1 < panel.y0
    g0.check_sign_words(fig, ax, [left, right])  # the build's own check
    # move "amplifying" to 1 pt right of "attenuating": too close
    x0 = (lb.x1 + 1 * fig.dpi / 72) / fig.bbox.width
    right.set(position=(x0, right.get_position()[1]),
              horizontalalignment="left")
    with pytest.raises(ValueError, match="1.0 pt apart"):
        g0.check_sign_words(fig, ax, [left, right])
    right.set(position=left.get_position())
    with pytest.raises(ValueError, match="the two words overlap"):
        g0.check_sign_words(fig, ax, [left, right])
    ps.plt.close(fig)
