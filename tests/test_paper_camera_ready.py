"""Tests for the camera-ready composite figures (grad_shift.paper.camera_ready)."""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from grad_shift.paper import camera_ready as cr
from grad_shift.paper import paper_style as ps
from grad_shift.paper import runs as runs_mod
from grad_shift.paper.macros import Result
from grad_shift.paper.products import gradient_schematic


def test_read_only_context_writes_only_to_write_dir(tmp_path):
    """Figures and tables a product saves land in write_dir; out is only read."""
    out, write = tmp_path / "paper_out", tmp_path / "scratch"
    out.mkdir()
    ctx = cr.ReadOnlyContext(out=str(out), run_keys={},
                                      write_dir=str(write))
    assert ctx.figures_dir == str(write / "figures")
    assert ctx.tables_dir == str(write / "tables")
    assert (write / "figures").is_dir() and (write / "tables").is_dir()
    assert list(out.iterdir()) == []  # nothing created in the pipeline folder


def test_read_only_context_requires_write_dir(tmp_path):
    ctx = cr.ReadOnlyContext(out=str(tmp_path), run_keys={})
    with pytest.raises(ValueError):
        _ = ctx.figures_dir


def test_resolve_paper_out(tmp_path, monkeypatch):
    monkeypatch.delenv(cr.PAPER_OUT_ENV, raising=False)
    with pytest.raises(SystemExit):
        cr.resolve_paper_out(None)
    with pytest.raises(SystemExit):  # a folder without results.json
        cr.resolve_paper_out(str(tmp_path))
    (tmp_path / "results.json").write_text("{}")
    monkeypatch.setenv(cr.PAPER_OUT_ENV, str(tmp_path))
    assert cr.resolve_paper_out(None) == str(tmp_path)


def fake_results(tmp_path, values):
    with open(tmp_path / "results.json", "w") as f:
        json.dump({k: {"value": v, "sem": 0.1} for k, v in values.items()}, f)
    return cr.Results(str(tmp_path))


def test_cbar_trait_adds_deltas_to_the_defenseless_value(tmp_path):
    """Qwen2.5-7B traits store PPS/IP as paired shifts from no defense."""
    r = fake_results(tmp_path, {"traits.evil.grad.none": 0.012,
                                "traits.evil.grad.pps.delta": -0.034,
                                "traits.evil.grad.ip.delta": -0.010})
    got = cr.cbar(r, cr.Setting("Qwen2.5-7B", "evil", "traits.evil", "trait"))
    assert got == pytest.approx({"none": 0.012, "pps": -0.022, "ip": 0.002})


def test_cbar_model_reads_the_values_directly(tmp_path):
    r = fake_results(tmp_path, {f"models.x.evil.gradcos.{a}": v for a, v in
                                (("none", 0.01), ("pps", -0.02), ("ip", 0.0))})
    got = cr.cbar(r, cr.Setting("other", "X", "models.x.evil", "model"))
    assert got == {"none": 0.01, "pps": -0.02, "ip": 0.0}


def test_row_positions_give_each_group_a_heading_row():
    ys, heads = cr.row_positions()
    assert len(ys) == len(cr.SETTINGS)
    assert ys == sorted(ys, reverse=True)  # top to bottom
    groups = list(dict.fromkeys(s.group for s in cr.SETTINGS))
    assert list(heads) == groups
    first = {s.group: y for s, y in reversed(list(zip(cr.SETTINGS, ys)))}
    for g in groups:  # the heading sits one row above its group's first row
        assert heads[g] - first[g] == pytest.approx(cr.HEADER_GAP)
    assert not set(heads.values()) & set(ys)


def test_check_banner_fails_when_the_flip_pattern_changes():
    flip = {"none": 0.01, "pps": -0.02, "ip": 0.0}
    grads = [dict(flip) for _ in cr.SETTINGS]
    exprs = [{a: (50.0, 1.0) for a in cr.ARMS + ("base",)}
             for _ in cr.SETTINGS]
    with pytest.raises(SystemExit):  # every setting flips: text says one doesn't
        cr.check_banner(grads, exprs)
    for i, s in enumerate(cr.SETTINGS):
        if s.label == "sycophancy":
            grads[i] = {"none": -0.001, "pps": -0.02, "ip": -0.005}
    cr.check_banner(grads, exprs)  # the pattern the prose describes
    k = [s.label for s in cr.SETTINGS].index("pirate")
    grads[k] = {"none": 0.01, "pps": -0.002, "ip": -0.005}  # IP shifts more
    with pytest.raises(SystemExit, match="PPS-below-IP"):
        cr.check_banner(grads, exprs)


def test_arm_markers_tell_pps_and_ip_apart():
    """Figure 1(b, c): each arm has its own marker shape, not only a color."""
    shapes = [cr.marker_style(a)["marker"] for a in cr.ARMS]
    assert len(set(shapes)) == len(cr.ARMS)


def test_aligned_limits_put_zero_level_with_the_left_value():
    left, level = (3.0, 5.0), 3.5
    lo, hi = cr.aligned_limits(left, level, -0.03, 0.04)
    assert -lo / (hi - lo) == pytest.approx((level - left[0]) / (left[1] - left[0]))
    assert lo <= -0.03 and hi >= 0.04  # the right-axis data fit


def test_aligned_limits_reject_impossible_alignments():
    with pytest.raises(ValueError):  # level outside the left axis
        cr.aligned_limits((3.0, 5.0), 6.0, -0.01, 0.01)
    with pytest.raises(ValueError):  # right-axis data do not straddle zero
        cr.aligned_limits((3.0, 5.0), 4.0, 0.01, 0.02)


def points_figure(w: float, h: float):
    """A figure whose one axes is w x h pt with data units of 1 pt."""
    fig = plt.figure(figsize=(w / 72, h / 72), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    return fig, ax


def test_schematic_fits_its_extent_without_overlaps():
    ps.apply()
    fig, ax = points_figure(gradient_schematic.WIDTH, gradient_schematic.HEIGHT)
    texts, arrows = gradient_schematic.draw(ax)
    assert [t.get_text() for t in texts].count("PPS") == 1
    assert cr.overlaps(fig, ax, texts, arrows) == []
    plt.close(fig)


def test_overlaps_reports_a_label_on_an_arrow_and_outside_the_panel():
    ps.apply()
    fig, ax = points_figure(gradient_schematic.WIDTH, gradient_schematic.HEIGHT)
    texts, arrows = gradient_schematic.draw(ax)
    _, oy, _, _ = gradient_schematic.ROWS[1]  # the lower (PPS) row's axis
    texts.append(ax.text(40, oy, "on the axis", va="center"))
    texts.append(ax.text(gradient_schematic.WIDTH - 2, 50, "off the edge"))
    fails = cr.overlaps(fig, ax, texts, arrows)
    assert any("'on the axis' overlaps" in f for f in fails)
    assert any("'off the edge' leaves" in f for f in fails)
    plt.close(fig)


def two_panels():
    ps.apply()  # constrained layout, as in the real figure
    fig = plt.figure(figsize=(4, 2))
    gs = fig.add_gridspec(1, 2)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a.set_ylim(3.0, 5.0)
    r = a.twinx()
    r.set_ylim(*cr.aligned_limits(a.get_ylim(), 4.2, -0.02, 0.03))
    return fig, a, b, r


def test_check_loss_layout_accepts_aligned_equal_panels():
    fig, a, b, r = two_panels()
    cr.check_loss_layout(fig, (a, b), a, r, 4.2)
    plt.close(fig)


def test_check_loss_layout_fails_on_misaligned_zero():
    fig, a, b, r = two_panels()
    r.set_ylim(-0.02, 0.03)  # zero now at 40% of the axis, 4.2 at 60%
    with pytest.raises(SystemExit):
        cr.check_loss_layout(fig, (a, b), a, r, 4.2)
    plt.close(fig)


def test_check_loss_layout_fails_on_unequal_heights():
    fig, a, b, r = two_panels()
    b.set_box_aspect(0.3)  # b shrinks to a short box in its cell
    with pytest.raises(SystemExit):
        cr.check_loss_layout(fig, (a, b), a, r, 4.2)
    plt.close(fig)


def test_labels_on_lines_finds_a_label_over_a_curve_only():
    ps.apply()
    fig, ax = plt.subplots(figsize=(3, 2))
    (curve,) = ax.plot([0, 1, 2], [0, 1, 0], marker="o")
    on = ax.text(1, 1, "on the peak", ha="center", va="center")
    off = ax.text(1, 0.1, "under the peak", ha="center", va="bottom")
    ax.set_ylim(-0.2, 1.4)
    fails = cr.labels_on_lines(fig, [on, off], {"curve": curve})
    assert fails == ["'on the peak' covers the curve line"]
    plt.close(fig)


def test_reduce_runs_vary_only_the_plotted_setting():
    """Each line of the active-reduction figure is one starting model on the
    Heavy data, and its points differ only in the x-axis setting: PPS with
    the evil vector at layer 20 at intensity x (0 = defenseless), or IP with
    the x-prompt bank."""
    specs = runs_mod.all_runs()
    start = {"heavy": "start.heavy", "base": None}
    for method, by_start in cr.REDUCE_RUNS.items():
        assert set(by_start) == set(cr.STARTS)
        for s, pts in by_start.items():
            assert pts[0.0 if method == "pps" else 1] in specs
            for x, run in pts.items():
                spec = specs[run]
                assert (spec.start, spec.train_file, spec.overrides) == (
                    start[s], runs_mod.DATA["heavy"], {}), run
                if method == "pps" and x == 0:
                    assert run == cr.UNDEFENDED[s] and spec.defense == "none"
                elif method == "pps":
                    assert (spec.defense, spec.pps_vector, spec.pps_coef,
                            spec.pps_layers) == ("pps", runs_mod.EVIL, x,
                                                 (20,)), run
                else:
                    assert (spec.defense, spec.prompt_file, spec.prompt_mode
                            ) == ("ip", runs_mod.EVIL_BANKS[x],
                                  "prepend_user"), run
    assert set(cr.REDUCE_RUNS["ip"]["heavy"]) == set(runs_mod.EVIL_BANKS)
    assert set(cr.REDUCE_RUNS["ip"]["base"]) == set(runs_mod.EVIL_BANKS)
    # the Heavy PPS sweep is a subset of the base model's, so they compare
    assert set(cr.REDUCE_RUNS["pps"]["heavy"]) <= set(
        cr.REDUCE_RUNS["pps"]["base"])


def test_reduction_is_defenseless_minus_defended():
    none, pps = Result(90.0, sem=0.6), Result(30.0, sem=0.8)
    r = cr.reduction(none, pps)
    assert (r.value, r.sem) == pytest.approx((60.0, 1.0))
    zero = cr.reduction(none, none, same_run=True)
    assert (zero.value, zero.sem) == (0.0, 0.0)


def fake_store(root, scores, identity=lambda run: f"id-{run}"):
    """A pipeline output folder whose runs are judged with the given mean
    "evil" scores (three samples each, SEM 1/sqrt(3))."""
    root.mkdir()
    with open(root / "plan.json", "w") as f:
        json.dump([{"name": r, "identity": identity(r)} for r in scores], f)
    for run, v in scores.items():
        d = root / "evals" / identity(run)
        d.mkdir(parents=True)
        pd.DataFrame({"evil": [v - 1, v, v + 1]}).to_csv(
            d / "trait_eval_evil_extract.csv", index=False)
    return str(root)


def test_stores_read_each_run_where_it_was_judged(tmp_path):
    a = fake_store(tmp_path / "a", {"x": 10.0, "both": 50.0})
    b = fake_store(tmp_path / "b", {"y": 20.0, "both": 50.0})
    stores = cr.Stores([a, b])
    assert stores.score("x").value == pytest.approx(10.0)
    assert stores.score("y").value == pytest.approx(20.0)
    assert stores.score("y").sem == pytest.approx(1 / math.sqrt(3))
    with pytest.raises(SystemExit):
        stores.score("z")  # judged nowhere


def test_stores_refuse_a_run_with_two_identities(tmp_path):
    a = fake_store(tmp_path / "a", {"run": 10.0})
    b = fake_store(tmp_path / "b", {"run": 10.0}, identity=lambda r: "other")
    with pytest.raises(SystemExit):
        cr.Stores([a, b]).score("run")


def synthetic_reduce_stores(tmp_path):
    """The figure's runs with made-up scores shaped like the real ones: PPS
    removes more with intensity from either start, IP prevents with few
    prompts but barely reduces. The new runs are in their own store."""
    pps = {e: 92 / (1 + e / 1.5) for e in (0, .25, .5, .75, 1, 1.5, 2, 3,
                                           4.5, 5, 6, 9)}
    scores = {}
    for start in ("base", "heavy"):  # PPS removes 1 more from Heavy
        for e, run in cr.REDUCE_RUNS["pps"][start].items():
            scores[run] = pps[e] - (1 if start == "heavy" and e > 0 else 0)
    for n, run in cr.REDUCE_RUNS["ip"]["base"].items():
        scores[run] = min(90.0, 8 * n)
    for n, run in cr.REDUCE_RUNS["ip"]["heavy"].items():
        scores[run] = 90.0 - 1.5 / n
    new = {r: v for r, v in scores.items() if r.startswith("reduce.")}
    old = {r: v for r, v in scores.items() if r not in new}
    old["start.heavy"] = new["start.heavy"] = 91.7
    return cr.Stores([fake_store(tmp_path / "paper_out", old),
                      fake_store(tmp_path / "cr_out", new)])


def test_reduce_figure_draws_and_reports_the_new_numbers(tmp_path):
    out = tmp_path / "fig"
    out.mkdir()
    res = cr.reduce_figure(synthetic_reduce_stores(tmp_path), str(out))
    plt.close("all")
    assert (out / "fig_cr_reduce.pdf").is_file()
    eps = sorted(cr.REDUCE_RUNS["pps"]["heavy"])
    for e in eps:
        tag = f"{e:g}".replace(".", "p")
        assert {f"reduce.pps.e{tag}.evil", f"reduce.pps.e{tag}.drop"} <= set(res)
    assert res["reduce.pps.e0.drop"].value == 0.0
    # IP's largest reduction from Heavy: the single prompt (90 - 1.5 / n)
    assert res["reduce.ip.maxdrop_n"].value == 1
    assert res["reduce.pps.gap.max"].value == pytest.approx(1.0)


def test_write_cr_results_refuses_pipeline_keys(tmp_path):
    res = {"reduce.ip.maxdrop": Result(1.5, sem=0.5)}
    cr.write_cr_results(res, ["fig4.expr.eps2"], str(tmp_path))
    tex = (tmp_path / "results_cr.tex").read_text()
    assert r"\csname sgres@reduce.ip.maxdrop\endcsname{1.5}" in tex
    assert r"\csname sgrespm@reduce.ip.maxdrop\endcsname{1.5$\pm$0.5}" in tex
    with pytest.raises(SystemExit):
        cr.write_cr_results(res, ["reduce.ip.maxdrop"], str(tmp_path))


def test_check_reduce_fails_when_the_data_contradict_the_prose(tmp_path):
    data = cr.reduce_data(synthetic_reduce_stores(tmp_path))
    cr.check_reduce(data, cr.reduce_results(data))  # shaped like the prose
    expr, red = data["ip"]["heavy"][1]
    data["ip"]["heavy"][1] = (expr, Result(20.0, sem=1.0))  # IP now cures
    with pytest.raises(SystemExit, match="IP bank removes"):
        cr.check_reduce(data, cr.reduce_results(data))
    data["ip"]["heavy"][1] = (expr, red)
    pps = data["pps"]["heavy"]
    pps[3.0] = (pps[3.0][0], Result(pps[2.0][1].value - 1, sem=1.0))
    with pytest.raises(SystemExit, match="not increasing"):
        cr.check_reduce(data, cr.reduce_results(data))


def test_write_cr_results_replaces_only_the_redrawn_figures_keys(tmp_path):
    cr.write_cr_results({"reduce.a": Result(1.0), "lossviews.b": Result(2.0)},
                        [], str(tmp_path))
    cr.write_cr_results({"lossviews.c": Result(3.0)}, [], str(tmp_path),
                        replace=["lossviews."])
    keys = json.loads((tmp_path / "results_cr.json").read_text())
    assert sorted(keys) == ["lossviews.c", "reduce.a"]


def synthetic_view_stores(tmp_path):
    """Loss-views analyses whose every curve has its loss minimum at m and
    C-bar's zero at 0.8 m (shrunk toward 0, as measured); the summed
    projection is the loss's exact derivative. Plus seven trait sweeps."""
    from grad_shift.paper.products import loss_views as lv
    from grad_shift.paper.products.loss_measure import SEP, save_npz
    from grad_shift.paper.products.traits_common import coef_tag
    v_norm, n = 2.0, 4
    crd, pod = tmp_path / "cr_out", tmp_path / "paper_out"
    for d in (crd, pod):
        d.mkdir()
        (d / "plan.json").write_text("[]")
    for k, (vec, data) in enumerate(cr.VIEW_ROWS):
        m = 0.6 - 0.4 * k  # +0.6, +0.2, -0.2, -0.6
        arrs, members = {}, {c: lv.member_names(c) for c in lv.PROMPT_SETS}
        for mem in [x for ms in members.values() for x in ms]:
            # as measured: the evil prompts pull the evil PV's minimum on
            # Heavy data toward 0
            mm = m / 2 if (vec, data, mem[:4]) == ("evil", "heavy", "evil") else m
            for e in lv.VIEW_EPS:
                name = lv.cond_name(mem, e)
                loss = (e - mm) ** 2 + 3 + np.arange(n)
                arrs[f"{name}{SEP}seq_loss_all"] = loss
                arrs[f"{name}{SEP}seq_loss_full"] = loss
                arrs[f"{name}{SEP}gproj_sum"] = np.full(n, 2 * (e - mm) / v_norm)
                arrs[f"{name}{SEP}gcos_sum"] = np.full(n, (e - 0.8 * mm) * 10)
                arrs[f"{name}{SEP}gcos_n"] = np.full(n, 100.0)
        d = crd / "analyses" / lv.ANALYSES[(vec, data)]
        d.mkdir(parents=True)
        save_npz(str(d / "arrays.npz"), arrs)
        (d / "result.json").write_text(json.dumps(
            {"npz": str(d / "arrays.npz"), "eps": list(lv.VIEW_EPS),
             "members": members, "v_norm": v_norm}))
    sweep = [-3.0, -1.5, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for t in cr.TRAIT_SWEEPS:
        s = {("none" if e == 0 else f"pps_{coef_tag(e)}"):
             {"loss": (e - 0.5) ** 2, "tokmean": (e - 0.4) * 0.01}
             for e in [0.0] + sweep}
        d = pod / "analyses" / f"traits_base_sweep.{t}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"summary": s,
                                                   "sweep": sweep}))
    from grad_shift.paper.assemble import load_context
    return load_context(str(crd)), load_context(str(pod))


def test_lossviews_figure_recovers_planted_minima_and_zeros(tmp_path):
    crc, poc = synthetic_view_stores(tmp_path)
    out = tmp_path / "fig"
    out.mkdir()
    res = cr.lossviews_figure(crc, poc, str(out))
    plt.close("all")
    assert (out / "fig_cr_lossviews.pdf").is_file()
    assert res["lossviews.prompts.n"].value == 44
    assert res["lossviews.panels.n"].value == 16
    # the exact derivative's zero is the minimum; C-bar's sits at 0.8 m
    assert res["lossviews.exact.gap.absmax"].value == pytest.approx(0, abs=1e-6)
    assert res["lossviews.prompts.gap.absmax"].value == pytest.approx(0.12)
    assert res["lossviews.prompts.toward0"].value == 44
    assert res["lossviews.exact.r_min"].value == pytest.approx(1.0)


def test_check_lossviews_flags_curves_that_contradict_the_appendix():
    good = [cr.ViewPoint("views", "evil/heavy/none", 0.7, 0.6, 1),
            cr.ViewPoint("views", "evil/heavy/evil0", 0.2, 0.1, 1),
            cr.ViewPoint("views", "syco/normal/none", -0.6, -0.4, 1)]
    exact = {("evil", "heavy", "none"): (0.9995, 0.02)}
    cr.check_lossviews(good, exact)
    for bad, msg in (
            (good[:2] + [good[2]._replace(vertex=0.1, zero=0.05)],
             "sycophantic PV minimum"),
            ([good[0], good[1]._replace(vertex=0.9), good[2]], "toward 0"),
            ([good[0]._replace(n_zeros=2)] + good[1:], "changes sign 2"),
            (good[:2] + [good[2]._replace(zero=None)], "no C-bar zero")):
        with pytest.raises(SystemExit, match=msg):
            cr.check_lossviews(bad, exact)
    with pytest.raises(SystemExit, match="r = 0.9"):  # measurement check
        cr.check_lossviews(good, {("evil", "heavy", "none"): (0.95, 0.2)})


# (evil, caps %) of the final models, shaped like the measured grid: single
# trait-targeting prompts block both behaviors, banks mostly their own, the
# Tan recipe is more selective, arbitrary prompts block nothing.
IPASSOC_FINAL = {"none": (92, 98), "evil.single": (1, 2), "evil.bank": (30, 50),
                 "caps.single": (17, 2), "caps.bank": (75, 1)}
IPASSOC_TAN = {"none": (92, 98), "evil.single": (9, 38),
               "caps.single": (72, 1)}
IPASSOC_ARBITRARY = (90, 97)


def ipassoc_csv(path, evil, caps, n=4):
    """n judged responses with the given evil score, each `caps` % uppercase."""
    k = int(round(caps))
    pd.DataFrame({"question_id": range(n), "evil": [float(evil)] * n,
                  "coherence": [85.0] * n,
                  "answer": ["A" * k + "a" * (100 - k)] * n}).to_csv(
        path, index=False)


def synthetic_ipassoc_stores(tmp_path, repro_offset=0.5):
    """cr_out with every grid run judged (the core runs at epochs 1-3, epoch
    1's "evil" bank at evil 8) and paper_out with the repeated pipeline runs,
    whose judge scores differ by ``repro_offset``."""
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.products import ipassoc as ia
    roots = {k: tmp_path / k for k in ("cr_out", "paper_out")}
    for d in roots.values():
        d.mkdir(parents=True)
    runs = {s.name: s for s in ia.EXTRA_RUNS}
    with open(roots["cr_out"] / "plan.json", "w") as f:
        json.dump([{"name": n, "identity": f"id-{n}",
                    "eval_system_prompt": s.eval_system_prompt}
                   for n, s in runs.items()], f)
    with open(roots["paper_out"] / "plan.json", "w") as f:
        json.dump([{"name": r, "identity": f"id-{r}"}
                   for r in ia.REPEATS.values()], f)
    crc, poc = (load_context(str(roots[k])) for k in ("cr_out", "paper_out"))

    def final(name):
        mask, cell = name.split(".")[1], ".".join(name.split(".")[2:])
        table = IPASSOC_TAN if mask == "tan" else IPASSOC_FINAL
        return table.get(cell, IPASSOC_ARBITRARY)

    for n in runs:
        p = crc.eval_csv(n)
        os.makedirs(os.path.dirname(p))
        ipassoc_csv(p, *final(n))
    for run, ref in ia.REPEATS.items():
        p = poc.eval_csv(ref)
        os.makedirs(os.path.dirname(p))
        evil, caps = final(run)
        ipassoc_csv(p, evil + repro_offset, caps)
    for n in ia.CURVE_RUNS:
        evil, caps = final(n)
        d = roots["cr_out"] / "analyses" / ia.curve_name(n)
        d.mkdir(parents=True)
        e1 = 8.0 if n.endswith("evil.bank") else evil
        pts = [{"epoch": ep, "final": ep == 3.0, "evil": e1 if ep == 1.0 else evil,
                "caps_pct": caps, "coherence": 85.0,
                "csv": crc.eval_csv(n) if ep == 3.0 else "mid.csv"}
               for ep in (1.0, 2.0, 3.0)]
        (d / "result.json").write_text(json.dumps({"points": pts}))
    return crc, poc


def test_ipassoc_grid_writes_the_table_and_its_numbers(tmp_path):
    crc, poc = synthetic_ipassoc_stores(tmp_path)
    out = tmp_path / "tab"
    out.mkdir()
    res = cr.ipassoc_grid(crc, poc, str(out))
    tex = (out / "tab_cr_ipassoc.tex").read_text()
    rows = [l for l in tex.splitlines() if l.endswith("\\\\")][2:]
    assert len(rows) == len(cr.IPASSOC_ROWS) == 14
    # the no-defense row gives the scores; every other row the % of them
    # retained, against its own recipe's no-defense run at the same epoch
    assert rows[0].startswith("no defense &  & 92 (92) & 98 (98) & 92 (92) "
                              "& 98 (98) & 92 & 98")
    assert rows[2].startswith(" & bank of 3 & 33 (9) & 51 (51)")
    assert rows[3].endswith("& 78 & 1 \\\\")  # Tan caps single: 72/92, 1/98
    assert sum(r.count("--") for r in rows) == 2 * 8  # no Tan banks / 2nds
    assert res["ipassoc.all.caps.single.evil.retained"].value == pytest.approx(
        100 * 17 / 92)
    assert res["ipassoc.ro.evil.bank.evil.retained.e1"].value == pytest.approx(
        100 * 8 / 92)
    assert res["ipassoc.ro.evil.bank.evil.e1"].value == pytest.approx(8.0)
    assert res["ipassoc.tan.evil.single.caps"].value == pytest.approx(38.0)
    assert "ipassoc.tan.evil.single.caps.e1" not in res  # one epoch only
    assert "ipassoc.tan.evil.single.caps.retained.e1" not in res
    assert not any(".removed" in k for k in res)
    assert res["ipassoc.arbitrary.evil.retained.min"].value == pytest.approx(
        100 * 90 / 92)
    assert res["ipassoc.repro.evil.absmax"].value == pytest.approx(0.5)
    assert res["ipassoc.repro.identical_pct"].value == 100


def test_check_ipassoc_fails_when_the_data_contradict_the_prose(tmp_path):
    crc, poc = synthetic_ipassoc_stores(tmp_path)
    data = cr.ipassoc_data(crc)
    res = {**cr.ipassoc_results(data), **cr.ipassoc_repro(crc, poc)}
    cr.check_ipassoc(data, res)  # shaped like the prose

    def check_with(cell, epoch, **values):
        saved = dict(data[cell][epoch])
        data[cell][epoch].update(values)
        try:
            with pytest.raises(SystemExit) as e:
                cr.check_ipassoc(data, {**cr.ipassoc_results(data), **{
                    k: v for k, v in res.items() if ".repro." in k}})
            return str(e.value)
        finally:
            data[cell][epoch] = saved

    assert "retains 76% < 90% of evil" in check_with(
        ("all", "neut", "single2"), 1.0, evil=70.0)  # an early epoch counts
    assert "caps bank retains no more evil" in check_with(
        ("ro", "caps", "bank"), 3.0, evil=10.0)
    assert "single evil prompt retains 92% of caps" in check_with(
        ("all", "evil", "single"), 3.0, caps=90.0)
    assert "Tan recipe" in check_with(("tan", "caps", "single"), 1.0,
                                      evil=10.0)
    bad = synthetic_ipassoc_stores(tmp_path / "bad", repro_offset=3.0)
    with pytest.raises(SystemExit, match="differ from the pipeline by 3.0"):
        cr.check_ipassoc(data, {**cr.ipassoc_results(data),
                                **cr.ipassoc_repro(*bad)})


def examples_df(scores_by_q, coherence=90.0):
    """Judged eval rows: for each question id (in file order), its samples'
    evil scores."""
    return pd.DataFrame([{"question_id": q, "question": f"Q {q}?",
                          "answer": f"{q} answer {i}", "evil": float(s),
                          "coherence": coherence}
                         for q, scores in scores_by_q.items()
                         for i, s in enumerate(scores)])


def test_spread_question_maximizes_the_smaller_defense_sd():
    # q0 varies most under PPS but not at all under IP; q9 and q10 tie, and
    # file order (not string order) decides
    pps = examples_df({"q0": [0, 100, 0, 100], "q9": [0, 60, 20, 80],
                       "q10": [0, 60, 20, 80]})
    ip = examples_df({"q0": [5, 5, 5, 5], "q9": [10, 50, 30, 90],
                      "q10": [10, 50, 30, 90]})
    assert cr.spread_question({"pps": pps, "ip": ip}) == "q9"
    with pytest.raises(ValueError, match="two samples"):
        cr.spread_question({"pps": examples_df({"q0": [1]}),
                            "ip": examples_df({"q0": [2]})})


def test_range_rows_pick_lowest_middle_and_highest_samples():
    df = examples_df({"q0": [30, 0, 45, 90, 0, 90, 55]})
    rows = cr.range_rows(df, "q0")
    assert [lab for lab, _ in rows] == ["lowest", "closest to 50", "highest"]
    # ties go to the lowest row (two 0s, two 90s; 45 and 55 both 5 from 50)
    assert [r["answer"] for _, r in rows] == [
        "q0 answer 1", "q0 answer 2", "q0 answer 3"]
    with pytest.raises(ValueError, match="three different"):
        cr.range_rows(examples_df({"q0": [0, 100]}), "q0")
    with pytest.raises(ValueError, match="no rows"):
        cr.range_rows(df, "q1")


def synthetic_examples_store(tmp_path, ip_q1=(0, 0, 40, 60, 95)):
    """paper_out with the examples' evals over three questions: q1 is the
    one on which PPS and IP both vary most (s.d. 34 and, by default, 40)."""
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.products import examples as ex
    root = tmp_path / "paper_out"
    root.mkdir()
    runs = [r for r, _ in ex.MODELS + ex.COLLAPSED]
    (root / "plan.json").write_text(json.dumps(
        [{"name": r, "identity": f"id-{r}"} for r in runs]))
    ctx = load_context(str(root))
    flat = {q: [3.0] * 5 for q in ("q0", "q1", "q2")}
    evals = {r: flat for r in runs}
    evals[cr.EX_NONE] = {q: [92.0] * 5 for q in flat}
    evals[cr.EX_PPS] = {"q0": [70] * 5, "q1": [2, 30, 52, 70, 88],
                        "q2": [10, 20, 30, 40, 50]}
    evals[cr.EX_IP] = {"q0": [0] * 5, "q1": list(ip_q1),
                       "q2": [0, 0, 0, 0, 80]}
    for r, by_q in evals.items():
        path = ctx.eval_csv(r)
        os.makedirs(os.path.dirname(path))
        examples_df(by_q).to_csv(path, index=False)
    examples_df(flat).to_csv(root / "base.csv", index=False)
    d = root / "analyses" / "base_model_eval"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(
        {"evil": {"csv": str(root / "base.csv")}}))
    return ctx


def test_appx_examples_writes_a_longtable_of_each_defenses_range(tmp_path):
    from grad_shift.paper.products import examples as ex
    ctx = synthetic_examples_store(tmp_path)
    out = tmp_path / "tab"
    out.mkdir()
    res = cr.appx_examples(ctx, str(out))
    tex = (out / "tab_cr_examples.tex").read_text()
    assert "\\begin{longtable}" in tex
    assert tex.rstrip().endswith("\\end{longtable}")
    assert "\\caption{\\crexamplescaption}\\label{tab:examples_appx}" in tex
    assert "\\caption[]{(continued)}" in tex and tex.count("\\endhead") == 1
    assert "``Q q1?''" in tex
    # 2026-11-06: every heading row, and the User row its datum follows,
    # forbids a page break after it (\\*), so a heading never ends a page
    heads = [ln for ln in tex.splitlines() if ln.startswith("\\multicolumn")]
    assert len(heads) == 3 and all(ln.endswith("\\\\*") for ln in heads)
    assert [ln for ln in tex.splitlines() if ln.startswith("User & ")
            ][0].endswith("\\\\*")
    rows = [ln for ln in tex.splitlines()
            if ln.endswith(("\\\\", "\\\\*")) and " & " in ln]
    assert [ln.split(" & ")[0] for ln in rows] == (
        ["User"] + [lab for _, lab in ex.DATA_VARIANTS]
        + ["Base model (no finetuning)", cr.EX_LABEL[cr.EX_NONE]]
        + [f"{cr.EX_LABEL[run]}, {pick}" for run in (cr.EX_PPS, cr.EX_IP)
           for pick in ("lowest", "closest to 50", "highest")]
        + [lab for _, lab in ex.COLLAPSED])
    # PPS 2 / 52 / 88; IP 0 / 40 (ties 60, lower row) / 95
    picked = [ln.rsplit("(evil ", 1)[1].split(",")[0] for ln in rows[7:13]]
    assert picked == ["2", "52", "88", "0", "40", "95"]
    for ln in tex.splitlines():  # braces balance (escaped ones excluded)
        bare = ln.replace("\\{", "").replace("\\}", "")
        assert bare.count("{") == bare.count("}"), ln
    assert res["crexamples.pps.sd"].value == pytest.approx(
        np.std([2, 30, 52, 70, 88], ddof=1))
    assert res["crexamples.ip.sd"].value == pytest.approx(
        np.std([0, 0, 40, 60, 95], ddof=1))
    # evil (>= 50) over all questions: PPS 5 + 3 + 1 of 15, IP 2 + 1 of 15
    assert res["crexamples.pps.share50"].value == pytest.approx(60.0)
    assert res["crexamples.ip.share50"].value == pytest.approx(20.0)
    assert (res["crexamples.n"].value, res["crexamples.qn"].value) == (15, 5)


def test_appx_examples_fail_when_a_defense_gives_no_evil_response(tmp_path):
    ctx = synthetic_examples_store(tmp_path, ip_q1=(0, 0, 10, 20, 40))
    with pytest.raises(SystemExit, match="ip scores 0-40 on q1"):
        cr.appx_examples(ctx, str(tmp_path))


def fine_grid_stores(tmp_path, mixed=(1.9, 2.0), shown_pps=None,
                     sweep_c0p5=-0.007):
    """cr_out with a fabricated ``traits_eps_fine.spanish`` (8 sequences;
    paper-sign C-bar: no defense +0.001, the ``mixed`` intensities
    alternating around zero, every other PPS intensity -0.05) and a
    results.json whose Spanish row matches it (PPS at eps 2 shown as
    ``shown_pps``, default: the fine grid's mean), with the base sweep's
    C-bar at eps 0.5 = ``sweep_c0p5``."""
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.products.gradient_core import paper_sign
    from grad_shift.paper.products.traits_common import coef_tag
    grid = [1.5, 1.9, 2.0, 2.25, 3.0]
    n = 8
    per = {"none": np.full(n, 0.001)}
    for e in grid:
        per[e] = (np.array([0.01, -0.01] * (n // 2)) + 0.0005 if e in mixed
                  else np.full(n, -0.05))
    arrs = {"none|tokmean": paper_sign(per["none"])}
    arrs.update({f"pps_{coef_tag(e)}|tokmean": paper_sign(per[e])
                 for e in grid})
    crd = tmp_path / "cr_out"
    d = crd / "analyses" / cr.SPANISH_FINE
    d.mkdir(parents=True)
    (crd / "plan.json").write_text("[]")
    np.savez(d / "per_sequence.npz", **arrs)
    (d / "result.json").write_text(json.dumps(
        {"npz": "per_sequence.npz", "eps": grid}))
    pps2 = per[2.0].mean() if shown_pps is None else shown_pps
    pod = tmp_path / "paper_out"
    pod.mkdir()
    results = {k: {"value": v} for k, v in {
        "traits.spanish.eps": 2.0, "traits.spanish.grad.none": 0.001,
        "traits.spanish.grad.pps.delta": pps2 - 0.001,
        "traits.spanish.grad.ip.delta": -0.003}.items()}
    results["traits.spanish.sweep.gcos.c0p5"] = {"value": sweep_c0p5,
                                                 "sem": 0.0002}
    (pod / "results.json").write_text(json.dumps(results))
    return load_context(str(crd)), cr.Results(str(pod))


def caps_sweep(none=96.0, ip=3.0, pps=None):
    """A held-out caps-tuning sweep (``capstune_products.sweep`` shape,
    confound only): caps learned by the defenseless model, every IP count
    and each PPS intensity."""
    pps = pps or {0.5: 80.0, 1.0: 30.0, 2.0: 3.0}
    return {"none": {"caps": none},
            "ip": {k: {"caps": ip} for k in (1, 2, 3, 4, 5)},
            "pps": {e: {"caps": c} for e, c in pps.items()}}


def test_caps_kept_eps_is_the_highest_intensity_keeping_half():
    assert cr.caps_kept_eps(caps_sweep()) == 0.5  # 80 >= 48 > 30
    assert cr.caps_kept_eps(caps_sweep(pps={0.5: 90, 1.0: 48.0, 2: 3})) == 1.0
    assert cr.caps_kept_eps(caps_sweep(pps={0.5: 40.0})) is None


def test_spanish_appendix_numbers_at_the_caps_kept_intensity(tmp_path):
    crc, r = fine_grid_stores(tmp_path)
    res = cr.spanish_appendix(crc, r, caps_sweep())
    assert res["crspanish.eps"].value == cr.SPANISH_CAPS_EPS == 0.5
    assert res["crspanish.cbar"].value == pytest.approx(-0.007)  # the sweep's
    assert res["crspanish.cbar"].sem == pytest.approx(0.0002)
    assert (res["crspanish.window.lo"].value,
            res["crspanish.window.hi"].value) == (1.9, 2.0)
    assert [res[f"crspanish.{k}"].formatted()
            for k in ("window.lo", "window.hi", "eps")] == ["1.9", "2", "0.5"]
    assert {k: res[f"crspanish.caps.{k}"].value
            for k in ("none", "pps", "threshold", "ip_max")} == {
        "none": 96.0, "pps": 80.0, "threshold": 48.0, "ip_max": 3.0}
    assert res["crspanish.epochs"].value == 12
    assert all(k.startswith(cr.CR_PREFIXES["spanish"]) for k in res)


def test_spanish_appendix_fails_when_the_data_contradict_the_text(tmp_path):
    def fails(path, sc=None, **kw):
        crc, r = fine_grid_stores(tmp_path / path, **kw)
        with pytest.raises(SystemExit) as e:
            cr.spanish_appendix(crc, r, sc or caps_sweep())
        return str(e.value)

    # the fine grid no longer reproduces results.json, or its window is not
    # one run around eps 2 (1.9 attenuating between two mixed points), or
    # is empty
    assert "pps: fine grid" in fails("a", shown_pps=0.01)
    assert "not one run of grid points" in fails("b", mixed=(1.5, 2.0))
    assert "window [] is not one run" in fails("c", mixed=())
    # the caps tuning: no caps learned, another pick, IP keeping the caps
    assert "learns only 2.0% caps" in fails("d", caps_sweep(none=2.0))
    assert "picks eps=1.0, recorded 0.5" in fails(
        "e", caps_sweep(pps={0.5: 90.0, 1.0: 60.0, 2.0: 3.0}))
    assert "an IP count keeps 70.0% caps" in fails("f", caps_sweep(ip=70.0))
    # C-bar at the picked intensity not attenuating
    assert "C-bar at eps=0.5 is +0.0010" in fails("g", sweep_c0p5=0.001)


def test_figure_1_leaves_spanish_to_the_appendix():
    """2026-11-06: Spanish is not a row of Figure 1(b, c)."""
    assert "Spanish" not in [s.label for s in cr.SETTINGS]
    assert len(cr.SETTINGS) == 9  # six traits on Qwen2.5-7B, three models
    assert not cr.CR_PREFIXES.get("banner")  # Figure 1 owns no CR keys


def test_write_cr_results_drops_keys_no_figure_owns(tmp_path):
    (tmp_path / "results_cr.json").write_text(json.dumps({
        k: {"value": 1.0, "sem": None, "decimals": 1, "source": ""}
        for k in ("reduce.a", "fig1cr.spanish.eps", "crspanish.cbar")}))
    cr.write_cr_results({"lossviews.b": Result(2.0)}, [], str(tmp_path),
                        replace=["lossviews."])
    keys = json.loads((tmp_path / "results_cr.json").read_text())
    # fig1cr.* (no figure draws it any more) is dropped; the others kept
    assert sorted(keys) == ["crspanish.cbar", "lossviews.b", "reduce.a"]
