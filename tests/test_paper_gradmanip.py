"""Group D products (gradient manipulations, runaway, cross-trait) on
fabricated inputs: pure helpers, then each product's build end to end."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import crosstrait_crisscross as cc
from grad_shift.paper.products import gradmanip_common as gc
from grad_shift.paper.products import gradmanip_loss as gl
from grad_shift.paper.products import gradmanip_runaway as gr
from grad_shift.paper.products import gradmanip_tables as gt
from grad_shift.paper.products.base import Context


# ----------------------------------------------------------- pure helpers
def test_schedule_matches_trainer_arithmetic():
    s1 = gc.schedule_steps(4681, 4, 8, 1, 0.03)
    assert s1 == {"microbatches_per_epoch": 1171, "total_steps": 146,
                  "warmup_steps": 4, "updates_per_epoch": 147}
    s3 = gc.schedule_steps(4681, 4, 8, 3, 0.03)
    assert (s3["total_steps"], s3["warmup_steps"]) == (439, 13)
    assert gc.linear_lr(2, 146, 4, 1e-5) == pytest.approx(5e-6)
    assert gc.linear_lr(4, 146, 4, 1e-5) == pytest.approx(1e-5)
    assert gc.linear_lr(146, 146, 4, 1e-5) == 0.0
    assert gc.linear_lr(200, 146, 4, 1e-5) == 0.0


def _log_lines(total=146, warmup=4, peak=1e-5, accum=8, n=1171, every=10,
               fn=gc.linear_lr):
    out = []
    for k in range(0, n, every):
        lr = fn((k + 1) // accum, total, warmup, peak)
        out.append(f"2026 - INFO - trainer - Step {k}: loss = 3.0, "
                   f"batch_loss = {3.0 - k / 1000:.4f}, lr = {lr:.2e}")
    return "\n".join(out)


def test_parse_log_keeps_last_pass_and_checks_lr():
    text = _log_lines(n=300) + "\n" + _log_lines()  # requeued restart
    df = gc.parse_trainer_log(text)
    assert df["step"].iloc[0] == 0 and len(df) == 118
    assert gt.check_logged_lr(df, 146, 4, 1e-5, 8) < 0.01
    cosine = lambda s, T, w, p: p * (s / w if s < w else 0.5 * (
        1 + np.cos(np.pi * (s - w) / (T - w))))
    bad = gc.parse_trainer_log(_log_lines(fn=cosine))
    with pytest.raises(ValueError):
        gt.check_logged_lr(bad, 146, 4, 1e-5, 8)


def test_peak_gated_respects_coherence_gate():
    pts = [{"step": 200, "evil": 50, "evil_sem": 1, "coherence": 60},
           {"step": 400, "evil": 80, "evil_sem": 1, "coherence": 19.9},
           {"step": 600, "evil": 50, "evil_sem": 2, "coherence": 30}]
    best = gc.peak_gated(pts)
    assert best["step"] == 200  # 400 is gated out; tie -> earliest
    assert gc.peak_gated(pts, gate=99) is None


def test_pm_and_table_format():
    r = gc.Result(9.123, sem=1.54)
    assert gc.pm(r) == "$9.1 \\pm 1.5$"
    assert gc.pm(None) == "--"
    t = gc.tex_table("cap", "tab:x", ["a", "b"], [["1", "2"]], "cc")
    assert "\\label{tab:x}" in t and "1 & 2 \\\\" in t


def test_tex_table_fit_width_only_wraps_the_tabular():
    """fit_width (2026-11-06, tab_traits_behavior) scales the tabular to the
    text width; without it the output is unchanged for the other tables."""
    args = ("cap", "tab:x", ["a", "b"], [["1", "2"]], "cc")
    plain = gc.tex_table(*args)
    wide = gc.tex_table(*args, fit_width=True)
    assert "resizebox" not in plain
    assert "\\resizebox{\\textwidth}{!}{\\begin{tabular}{cc}\n" in wide
    assert "\\end{tabular}}\n\\end{table}" in wide
    assert wide.replace("\\resizebox{\\textwidth}{!}{", "").replace(
        "\\end{tabular}}", "\\end{tabular}") == plain


def test_trajectory_and_late_range():
    pts = [{"step": 3513, "evil_proj": -27.0}, {"step": 400,
                                                "evil_proj": -24.0},
           {"step": 800, "evil_proj": -27.5}, {"step": 1200,
                                                "evil_proj": -27.2}]
    t = gr.trajectory(pts)
    assert [s for s, _ in t] == [400, 800, 1200, 3513]
    assert gr.late_range(t) == pytest.approx(0.2)


def test_extra_runs_spellings():
    names = [s.name for s in gr.EXTRA_RUNS]
    assert "gd.pps_extreme.5000" in names
    ps = next(s for s in gr.EXTRA_RUNS if s.name == "gd.sched_atten.pos_sign")
    # Manipulation passed as overrides so grad_manipulation_layers stays
    # unset, exactly like the archived record (module docstring).
    assert ps.grad_manipulation is None and ps.gm_layers is None
    assert ps.overrides["steering_layers"] == ["all"]


# ------------------------------------------------------ product fixtures
def _csv(path, cols, n=40, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = {"question": [f"q{i % 5}" for i in range(n)]}
    for c, v in cols.items():
        df[c] = np.clip(v + rng.normal(0, 1, n), 0, 100)
    pd.DataFrame(df).to_csv(path, index=False)


def _ctx(tmp_path, runs):
    out = tmp_path / "out"
    os.makedirs(out, exist_ok=True)
    keys = {r: f"id_{i}" for i, r in enumerate(sorted(runs))}
    (out / "plan.json").write_text(json.dumps(
        [{"name": r, "identity": k} for r, k in keys.items()]))
    for k in keys.values():
        os.makedirs(out / "runs" / k, exist_ok=True)
    return Context(out=str(out), run_keys=keys)


def _result(ctx, name, obj):
    d = ctx.analysis_dir(name)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(obj, f)


def test_grad_manip_tables_build(tmp_path):
    p = gt.GradManipTables()
    ctx = _ctx(tmp_path, p.runs)
    truth = {}
    for i, r in enumerate(p.runs):
        v = 5.0 if (".pps." in r or ".ip." in r) else 70.0
        if ".atten.evil" in r:
            v = 3.0
        if r.startswith(("gm.normal", "gd.normal")):
            v = 0.5
        truth[r] = v
        _csv(ctx.eval_csv(r), {"evil": v, "coherence": 85.0}, seed=i)
    for r in gt.NORMAL_PEAK_RUNS:
        pts = [{"step": s, "evil": e, "evil_sem": 1.0, "coherence": c}
               for s, e, c in ((200, 10, 80), (400, 60, 40), (600, 90, 5))]
        _result(ctx, gt.ckpt_analysis_name(r), {"checkpoints": pts})
    os.makedirs(os.path.join(ctx.out, "logs"))
    with open(os.path.join(ctx.out, "logs",
                           "gm.heavy.none.default-1.out"), "w") as f:
        f.write(_log_lines() + "\nTraining completed!\n")
    res = p.build(ctx)
    assert abs(res["tab2.none.evil.atten"].value - 3.0) < 1
    assert abs(res["tab1.none.evil.default"].value - 70.0) < 1
    # peak at step 400 (60 at coherence 40); 90 is below the gate
    assert res["tab3.none.ampl"].value == 60 and \
        res["tab3.none.ampl"].meta["step"] == 400
    assert res["gm.setup.steps_per_epoch"].value == 147
    assert res["gm.lr.1ep.mid"].meta["max_rel_err"] < 0.01
    assert res["gm.lr.1ep.mid"].value == pytest.approx(
        1e1 * (146 - 73) / 142, rel=1e-6)
    for f in ("tab_grad_manips.tex", "tab_grad_manips_appx.tex",
              "tab_peak_grad_manips_normal.tex",
              "tab_grad_manips_coherence.tex", "gradmanip_lr_schedule.tex"):
        assert os.path.isfile(os.path.join(ctx.tables_dir, f))
    main = open(os.path.join(ctx.tables_dir, "tab_grad_manips.tex")).read()
    assert "28 transformer blocks" in main and "linear" in main
    assert "cosine" not in main and "1--27" not in main
    from grad_shift.paper import macros
    macros.render(res)  # every key valid, every value finite


def _manip_jsonl(run_dir, losses):
    with open(os.path.join(run_dir, "grad_manip_cosine.jsonl"), "w") as f:
        for s, l in enumerate(losses):
            f.write(json.dumps({"step": s, "epoch": 0, "batch_loss": l})
                    + "\n")


def test_loss_figure_build(tmp_path):
    p = gl.GradManipLoss()
    ctx = _ctx(tmp_path, p.runs)
    for r in p.runs:
        n = 3513 if "sched" in r else 1171
        xs = np.arange(n)
        _manip_jsonl(ctx.run_dir(r), 3 - xs / n + (xs / n) ** 2 * 3)
    res = p.build(ctx)
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_grad_manip_loss.pdf"))
    assert res["gmloss.possign3ep.final"].value > res[
        "gmloss.possign3ep.min"].value


def test_runaway_build(tmp_path):
    p = gr.RunawayNumbers()
    ctx = _ctx(tmp_path, p.runs)
    for i, r in enumerate(p.runs):
        _csv(ctx.eval_csv(r), {"evil": 0.0 if "extreme" in r else 50,
                               "coherence": 0.0 if "extreme" in r else 80},
             seed=i)
    traj = {"base": [(0, -22.0)], "none": [(400, -21), (3513, -17)],
            "pps": [(400, -25), (800, -27), (3513, -27.3)],
            "pps1p5": [(3513, -26)], "pps9": [(3513, -30)],
            "possign": [(292, -23), (3513, -860)],
            **{f"coef{c}": [(292, -100), (3513, -500 + c / 10)]
               for c in gr.EXTREME_COEFS}}
    for arm, t in traj.items():
        _result(ctx, gr.analysis_name(arm), {"points": [
            {"step": s, "evil_proj": v, "evil_v0_proj": v + 1}
            for s, v in t]})
    res = p.build(ctx)
    assert res["runaway.possign_all.proj_end"].value == -860
    assert res["runaway.pps.proj_start"].value == -22.0
    assert res["runaway.possign_all.proj_end"].meta["old_vector_value"] == -859
    assert res["runaway.coef_extreme.coh_max"].value < 2
    assert res["runaway.pps9.proj_end"].value == -30


def test_criss_cross_build(tmp_path):
    p = cc.CrissCross()
    ctx = _ctx(tmp_path, p.runs)
    for i, r in enumerate(p.runs):
        cols = {"evil": 40.0, "sycophantic": 60.0, "coherence": 80.0}
        _csv(ctx.eval_csv(r, "evil"), cols, seed=i)
        if "syco" in r:
            _csv(ctx.eval_csv(r, "sycophantic"), cols, seed=i)
    layers = range(1, 29)
    for d in cc.DATAS:
        for v in cc.VECS:
            rows = []
            for c in (0.0,) + cc.COEFS:
                for l in layers:
                    rec = {"layer": l, "coef": c, "condition":
                           "defenseless" if c == 0 else f"steered_pps_coef{c}"}
                    for pv in cc.DIRS:
                        same = (pv == "evil") == (v == "evil") or v == "evil"
                        rec[f"{pv}_gradcos"] = (0.001 * c * (same * 2 - 1)
                                                * np.exp(-(l - 20) ** 2 / 20))
                    rows.append(rec)
            _result(ctx, cc.gradcos_name(d, v), {"rows": rows})
        models = []
        for label, run, vec, coef in cc.PROJ_MODELS[d]:
            rows = [{"layer": l, "condition": cond, "coef": coef,
                     "evil_proj": -l, "sycophantic_proj": -l / 2}
                    for l in layers for cond in (["defenseless"] + (
                        [] if vec is None else ["steered"]))]
            models.append({"label": label, "run": run, "rows": rows})
        _result(ctx, cc.actproj_name(d), {"models": models})
    res = p.build(ctx)
    r = res["xtrait.gradcos.evildata.evilvec.evilpv.c5.peak"]
    # paper sign (positive = amplifying): fabricated raw cosines negate;
    # the peak is the strongest attenuation (the minimum)
    assert r.meta["layer"] == 20 and r.value == pytest.approx(-0.005)
    # fabricated: the syco vector never attenuates the evil-PV gradient
    assert res["xtrait.gradcos.sycovec.evil_peak"].value > -0.001
    assert "xtrait.evil.evilvec1.5" in res and "xtrait.syco.evilvec5" in res
    assert os.path.isfile(os.path.join(ctx.figures_dir, "fig_criss_cross.pdf"))
    from grad_shift.paper import macros
    macros.render(res)
