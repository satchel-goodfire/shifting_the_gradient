"""Group A behavior products: pure logic + assemble on fabricated evals.

No GPU or judge: ``judge`` is monkeypatched to write synthetic CSVs, and
every eval CSV is fabricated with a known mean so each macro can be checked
against the value it must equal.
"""

import json
import math
import os
from artifact_testutil import write_adapter

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.macros import Result
from grad_shift.paper.products import (behavior_capscheck, behavior_common,
                                       behavior_eps, behavior_sysprompt,
                                       discover)
from paper_testutil import assemble_fabricated as assemble
from grad_shift.paper.products.base import Context
from grad_shift.paper.products.behavior_common import (
    CurveEvalMixin, checkpoints, eps_tag, mean_of_arms, total_microbatches)
from grad_shift.paper.products.behavior_curves import CurvePPS
from grad_shift.paper.products.behavior_eps import EpsHeldout, select_eps
from grad_shift.paper.products.behavior_specificity import EXTRA_RUNS
from grad_shift.paper.runs import all_runs
from grad_shift.paper.spec import REPO_ROOT

MINE = ("fig_behavior_specificity", "fig_behavior_curves",
        "fig_behavior_sysprompt", "fig_ip_prompt_count", "fig_eps_selection",
        "tab_caps_check", "fig_caps_tradeoff", "fig_hyper_ip_lora",
        "fig_hyper_pps_rank", "fig_hyper_pps_layer", "fig_hyper_pps_coef",
        "fig_behavior_row")
# Caps learned (%) fabricated per caps-check run: the rule must pick n* = 3
# (first count >= 50% of defenseless) and eps* = 2 (caps closest to n*'s).
CAPS = {"capscheck.none": 100, "capscheck.ip.n1": 0, "capscheck.ip.n2": 10,
        "capscheck.ip": 60, "capscheck.ip.n4": 70, "capscheck.ip.n5": 80,
        "capscheck.ip.n10": 100, "capscheck.pps.eps0.5": 100,
        "capscheck.pps.eps1": 90, "capscheck.pps.eps1.5": 80,
        "capscheck.pps": 50, "capscheck.pps.eps2.5": 30,
        "capscheck.pps.eps3": 10, "capscheck.pps.eps6": 0,
        "capscheck.pps.eps9": 0}


def _caps_text(pct):
    """A response whose uppercase-letter fraction is ``pct`` percent."""
    k = int(round(pct / 10))
    return "A" * k + "a" * (10 - k)


def _csv(path, evil, coh=90.0, n=20, seed=0, answer="x", noise=0.0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # zero-mean noise so the column mean is exactly ``evil``
    e = np.full(n, float(evil))
    if noise:
        z = rng.normal(0, noise, n)
        e = e + z - z.mean()
    pd.DataFrame({"question": [f"q{i % 5}" for i in range(n)],
                  "answer": [answer] * n, "evil": e,
                  "coherence": [coh] * n}).to_csv(path, index=False)
    return path


# ---- pure helpers ---------------------------------------------------------

def test_eps_tag_is_macro_safe():
    assert eps_tag(1.5) == "1p5" and eps_tag(9.0) == "9"
    assert eps_tag(0.25) == "0p25" and eps_tag(12) == "12"


def test_mean_of_arms_uses_across_arm_spread():
    r = mean_of_arms([Result(10.0, sem=0.1), Result(20.0, sem=0.1),
                      Result(30.0, sem=0.1)], "x")
    assert r.value == pytest.approx(20.0)
    assert r.sem == pytest.approx(10.0 / math.sqrt(3))  # sd 10 over 3 arms
    assert r.meta["min"] == 10.0 and r.meta["max"] == 30.0


def test_total_microbatches_heavy_protocol():
    # 4,681 heavy samples, batch 4, 3 epochs: 3 * ceil(4681/4) = 3513 (the
    # paper's "3513 total steps"; 439 optimizer steps at accumulation 8)
    args = {"train_file": "data/persona_vectors_dataset/evil/misaligned_2.jsonl",
            "batch_size": 4, "num_epochs": 3, "max_samples": None}
    assert total_microbatches(args) == 3513


def test_checkpoints_sorted_and_complete_only(tmp_path):
    for n, complete in ((800, True), (400, True), (1200, False)):
        d = tmp_path / f"checkpoint-{n}"
        d.mkdir()
        if complete:
            write_adapter(d)
    (tmp_path / "checkpoint-x").mkdir()
    assert [s for s, _ in checkpoints(str(tmp_path))] == [400, 800]


def test_select_caps_matched_reproduces_recorded_decision():
    # Held-out caps (%) from caps_matched_selection.md (Oct 2026).
    none = 99.4
    ip = {1: 2.4, 2: 3.6, 3: 54.8, 4: 61.7, 5: 82.4, 10: 99.5}
    pps = {0.5: 99.6, 1.0: 97.2, 1.5: 79.4, 2.0: 50.8, 2.5: 29.7, 3.0: 11.5,
           6.0: 3.2, 9.0: 9.0}
    assert behavior_capscheck.select_caps_matched(none, ip, pps) == (3, 2.0)
    # nothing reaches the threshold -> no selection (report, don't relax)
    assert behavior_capscheck.select_caps_matched(
        none, {1: 1.0, 2: 2.0}, pps) == (None, None)
    # ties go to the smaller epsilon
    assert behavior_capscheck.select_caps_matched(
        100.0, {1: 60.0}, {1.0: 70.0, 2.0: 50.0}) == (1, 1.0)


def test_caps_sweeps_cover_the_main_settings():
    assert behavior_capscheck.IP_COUNT_RUNS[3] == "capscheck.ip"
    assert behavior_capscheck.PPS_EPS_RUNS[2.0] == "capscheck.pps"
    runs = all_runs()
    for r in behavior_capscheck.SWEEP_RUNS:
        assert r in runs, r


def test_select_eps_reproduces_recorded_decision():
    # Held-out numbers from eps_selection.md (Oct 2026): the rule picks 9.
    none, ip = (90.3, 88.9), 0.0
    rows = {1.5: dict(evil=46.9, coh=86.4, rand_evil=88.8),
            3.0: dict(evil=23.8, coh=85.2, rand_evil=86.6),
            4.5: dict(evil=11.8, coh=86.6),  # no random run at 4.5
            6.0: dict(evil=7.2, coh=86.8, rand_evil=86.2),
            9.0: dict(evil=2.1, coh=87.7, rand_evil=84.1),
            12.0: dict(evil=0.7, coh=84.3, rand_evil=82.7),
            15.0: dict(evil=0.2, coh=80.4, rand_evil=82.6)}
    assert select_eps(rows, none, ip) == 9.0
    # random retention below 90% everywhere -> nothing selected
    for r in rows.values():
        if "rand_evil" in r:
            r["rand_evil"] = 50.0
    assert select_eps(rows, none, ip) is None


def test_offtrait_extra_runs_point_at_versioned_prompt_files():
    assert len(EXTRA_RUNS) == 14
    for spec in EXTRA_RUNS:
        path = os.path.join(REPO_ROOT, spec.prompt_file)
        assert os.path.isfile(path), path
        assert len(json.load(open(path))["prompts"]) == 1  # single prompt


# ---- analyses with a stub judge ---------------------------------------------

def _stub_judge(calls):
    def judge(model, out_csv, trait="evil", split="extract",
              system_prompt=None):
        if not os.path.isfile(out_csv):  # like judge: cached CSVs are kept
            calls.append((model, split, system_prompt))
            _csv(out_csv, 50.0)
        return out_csv
    return judge


def _fake_run(d, ckpts=(400, 800)):
    os.makedirs(d, exist_ok=True)
    for n in ckpts:
        os.makedirs(os.path.join(d, f"checkpoint-{n}"))
        write_adapter(os.path.join(d, f"checkpoint-{n}"))
    with open(os.path.join(d, "experiment_config.json"), "w") as f:
        json.dump({"args": {
            "train_file": "data/persona_vectors_dataset/evil/misaligned_2.jsonl",
            "batch_size": 4, "num_epochs": 3, "gradient_accumulation_steps": 8,
            "max_samples": None}}, f)


def test_curve_analysis_judges_checkpoints_and_reuses_final(tmp_path,
                                                            monkeypatch):
    calls = []
    monkeypatch.setattr(behavior_common, "judge", _stub_judge(calls))
    ctx = Context(out=str(tmp_path), run_keys={"curve.pps": "idp-ckpt400"})
    _fake_run(ctx.run_dir("curve.pps"))
    _csv(ctx.eval_csv("curve.pps"), 3.0)
    out = CurvePPS().run(ctx)
    assert [c[2] for c in calls] == [None, None]  # training-matched prompt
    steps = [p["optimizer_step"] for p in out["points"]]
    assert steps == [50.0, 100.0, 3513 / 8]
    assert out["points"][-1]["final"] and out["points"][-1]["evil"] == 3.0
    # cached: a rerun judges nothing new
    CurvePPS().run(ctx)
    assert len(calls) == 2


def test_alt_curve_judges_final_under_alt_prompt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(behavior_common, "judge", _stub_judge(calls))

    class Alt(CurveEvalMixin):
        name = "alt"
        run_name = "curve.none"
        system_prompt = "model_default"

    ctx = Context(out=str(tmp_path), run_keys={"curve.none": "idn-ckpt400"})
    _fake_run(ctx.run_dir("curve.none"), ckpts=(400,))
    out = Alt().run(ctx)
    assert [c[2] for c in calls] == ["model_default"] * 2  # ckpt + final
    assert out["points"][-1]["final"]


def test_sysalt_dedupes_identical_models(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(behavior_sysprompt, "judge", _stub_judge(calls))

    class Alt(behavior_sysprompt._AltEvalMixin):
        name = "alt_grid"
        eval_runs = ["start.medium", "grid.base.medium.none"]

    ctx = Context(out=str(tmp_path), run_keys={
        "start.medium": "same", "grid.base.medium.none": "same"})
    out = Alt().run(ctx)
    assert len({m["csv"] for m in out["models"].values()}) == 1
    assert len(calls) == 1


def test_strong_regime_runs_are_the_comparison_models():
    """eps 9 / p1 grid cells at base x heavy = the eps and ipcount runs."""
    runs = all_runs()

    def body(name):
        return {k: v for k, v in vars(runs[name]).items() if k != "name"}

    assert body("strong.grid.base.heavy.pps") == body("eps.evil.9")
    assert body("strong.grid.base.heavy.ip") == body("ipcount.1")
    assert runs["strong.grid.medium.heavy.pps"].pps_coef == 9.0
    assert runs["strong.grid.heavy.medium.ip"].start == "start.heavy"
    syco, main = runs["spec.pps_syco.eps9"], runs["spec.pps_syco"]
    assert (syco.pps_vector, syco.pps_coef) == (main.pps_vector, 9.0)
    for r in behavior_sysprompt.STRONG_RUNS:
        assert r in runs, r


def test_eps_heldout_seeds_only_from_the_adopted_model_dir(tmp_path,
                                                           monkeypatch):
    calls = []
    monkeypatch.setattr(behavior_eps, "judge", _stub_judge(calls))
    keys = {r: f"id{i}" for i, r in enumerate(EpsHeldout.runs)}
    ctx = Context(out=str(tmp_path), run_keys=keys)
    seeded_run = "eps.evil.3"
    # every run dir exists; one carries a held-out CSV next to its adapter,
    # reached through a symlink like an adopted legacy run
    real = tmp_path / "legacy" / "evil_eps3"
    _csv(str(real / "trait_eval_evil_eval.csv"), 23.8)
    os.makedirs(tmp_path / "runs", exist_ok=True)
    os.symlink(real, ctx.run_dir(seeded_run))
    out = EpsHeldout().run(ctx)["runs"]
    assert out[seeded_run]["seeded_from"] == str(
        real / "trait_eval_evil_eval.csv")
    assert pd.read_csv(out[seeded_run]["csv"])["evil"].mean() == \
        pytest.approx(23.8)
    assert len(calls) == len(EpsHeldout.runs) - 1
    assert all(c[1] == "eval" for c in calls)


# ---- end to end: assemble builds every behavior product ------------------

def _fabricate_out(tmp_path):
    """Plan + eval CSVs + analysis results for every behavior product."""
    out = tmp_path / "out"
    os.makedirs(out)
    specs = all_runs()
    plan = [{"name": n, "identity": f"id{i}"} for i, n in enumerate(specs)]
    (out / "plan.json").write_text(json.dumps(plan))
    ctx = Context(out=str(out), run_keys={p["name"]: p["identity"]
                                          for p in plan})
    truth = {}

    def val(run):
        if run.startswith("behavior.ipoff."):
            return 80.0 + int(run[-1])  # arm i -> 80 + i
        if run.startswith("capscheck."):
            return {"none": 90.0, "pps": 3.0, "ip": 1.0}[run.split(".")[1]]
        if run == "ipcount.1":
            return 12.0
        if ".none" in run or run.startswith("start."):
            return 90.0
        if run.startswith("eps.rand") or run == "spec.pps_rand":
            return 85.0
        return 10.0

    _, products = discover()
    need = sorted({r for p in products.values() if p.name in MINE
                   for r in p.runs})
    for i, run in enumerate(need):
        truth[run] = val(run)
        caps = _caps_text(CAPS[run]) if run in CAPS else "quiet"
        _csv(ctx.eval_csv(run), truth[run], seed=i, answer=caps, noise=1.0)
        if run in CAPS:  # held-out split (the caps-matched selection)
            _csv(ctx.eval_csv(run, split="eval"), truth[run] + 1.0, seed=i,
                 answer=caps, noise=1.0)

    def write_result(name, obj):
        d = out / "analyses" / name
        os.makedirs(d, exist_ok=True)
        (d / "result.json").write_text(json.dumps(obj))

    base = _csv(str(out / "analyses" / "base_model_eval" / "base.csv"), 0.5)
    write_result("base_model_eval", {"evil": {"csv": base}})
    for arm, v in (("none", 90.0), ("pps", 5.0), ("ip", 1.0)):
        for name, scale in ((f"behavior_curve_{arm}", 1.0),
                            (f"behavior_sysalt_curve_{arm}", 0.5)):
            pts = []
            for k, mb in enumerate((400, 800, 3513)):
                c = _csv(str(out / "analyses" / name / f"{mb}.csv"),
                         v * scale * (k + 1) / 3)
                pts.append({"microbatch": mb, "optimizer_step": mb / 8,
                            "csv": c, "final": mb == 3513,
                            "evil": v * scale * (k + 1) / 3,
                            "coherence": 90.0})
            write_result(name, {"points": pts})
    for name, runs in (("behavior_sysalt_grid",
                        behavior_sysprompt.GRID_RUNS),
                       ("behavior_sysalt_spec",
                        behavior_sysprompt.SPEC_RUNS),
                       ("behavior_sysalt_strong",
                        behavior_sysprompt.STRONG_RUNS)):
        models = {}
        for r in runs + ([behavior_sysprompt.BASE]
                         if name.endswith("grid") else []):
            v = truth.get(r, 0.5) - 5.0
            models[r] = {"csv": _csv(str(out / "analyses" / name /
                                         f"{r}.csv"), max(v, 0.0))}
        write_result(name, {"models": models})
    held = {r: {"csv": _csv(str(out / "analyses" / "behavior_eps_heldout" /
                                f"{r}.csv"), truth[r] - 1.0)}
            for r in EpsHeldout.runs}
    write_result("behavior_eps_heldout", {"runs": held})
    return out, truth


def test_assemble_builds_every_behavior_product(tmp_path):
    out, truth = _fabricate_out(tmp_path)
    results, skipped = assemble(str(out))
    for name in MINE:
        assert name not in skipped, (name, skipped.get(name))
    for stem in ("fig_behavior_specificity", "fig_behavior_curves",
                 "fig_behavior_row",
                 "fig_behavior_sysprompt", "fig_behavior_sysprompt_curves",
                 "fig_ip_prompt_count", "fig_eps_selection",
                 "fig_hyper_ip_lora", "fig_hyper_pps_rank",
                 "fig_hyper_pps_layer", "fig_hyper_pps_coef",
                 "fig_caps_tradeoff"):
        assert (out / "figures" / f"{stem}.pdf").exists(), stem
    for t in ("tab_eps_selection_heldout", "tab_eps_selection_extract",
              "tab_caps_check", "tab_caps_tradeoff"):
        assert (out / "tables" / f"{t}.tex").exists(), t

    v = {k: r.value for k, r in results.items()}
    # Fig 2c: off-trait IP = mean of the single-prompt arms (80 + i)
    assert v["fig2c.ip_syco.evil"] == pytest.approx(82.0)  # i = 0..4
    assert v["fig2c.ip_neut.evil"] == pytest.approx(81.5)  # i = 0..3
    assert v["fig2c.ip_rand.evil.max"] == pytest.approx(84.0)
    assert v["fig2c.base.evil"] == pytest.approx(0.5)
    assert v["fig2c.pps_rand.retain_pct"] == pytest.approx(100 * 85 / 90)
    # curves: per-checkpoint, final and peak
    assert v["fig2def.none.step50.evil"] == pytest.approx(30.0)
    assert v["fig2def.none.final.evil"] == pytest.approx(90.0)
    assert v["fig2def.pps.peak.step"] == pytest.approx(3513 / 8)
    # alt prompt: shift = alt minus training-matched (fabricated -5)
    assert v["sysalt.base.heavy.pps.shift"] == pytest.approx(-5.0)
    assert v["sysalt.spec.ip_syco.shift"] == pytest.approx(-5.0)
    # strong regime (eps 9 / p1): own defended models, shared defenseless
    assert v["sysalt.strong.base.heavy.pps.shift"] == pytest.approx(-5.0)
    assert v["sysalt.strong.medium.heavy.ip.evil"] == pytest.approx(5.0)
    assert v["sysalt.strong.spec.pps_rand.evil"] == pytest.approx(80.0)
    assert v["sysalt.strong.spec.pps_syco.shift"] == pytest.approx(-5.0)
    assert not any(k.startswith("sysalt.strong.") and ".none" in k
                   for k in v)  # defenseless macros exist once (sysalt.*)
    # IP prompt count: n = 3 is the main IP, n = 1 the single prompt p1
    assert v["ipcount.n3.evil"] == pytest.approx(truth["grid.base.heavy.ip"])
    assert v["ipcount.n1.evil"] == pytest.approx(12.0)
    # eps sweep: both splits, reduction and retention, rule re-applied
    assert v["epssel.extract.rand.e9.evil"] == pytest.approx(85.0)
    assert v["epssel.heldout.rand.e9.evil"] == pytest.approx(84.0)
    # superseded rule: reduction relative to the single-prompt IP (p1)
    red = 100 * (90 - 10) / (90 - 12)
    assert v["epssel.extract.evil.e1p5.reduction_pct"] == pytest.approx(red)
    assert v["epssel.extract.selected"] == pytest.approx(1.5)
    assert v["epssel.extract.ip.evil"] == pytest.approx(
        truth["grid.base.heavy.ip"])
    assert v["epssel.chosen"] == 2 and v["epssel.superseded"] == 9
    # caps-matched rule (main), re-applied on both splits
    for sk in ("heldout", "extract"):
        assert v[f"capssel.{sk}.n_star"] == 3
        assert v[f"capssel.{sk}.eps_star"] == pytest.approx(2.0)
        assert v[f"epssel.capsrule.{sk}.eps_star"] == pytest.approx(2.0)
    assert v["capssel.heldout.threshold_caps_pct"] == pytest.approx(50.0)
    assert v["capssel.extract.pps.e2.caps_pct"] == pytest.approx(50.0)
    assert v["capssel.heldout.ip.n3.evil"] == pytest.approx(2.0)  # +1 eval
    assert v["capssel.extract.ip.n3.evil"] == pytest.approx(1.0)
    # caps check: uppercase fraction (fabricated), base "x" = 0%
    assert v["capscheck.none.caps_pct"] == pytest.approx(100.0)
    assert v["capscheck.ip.caps_pct"] == pytest.approx(60.0)
    assert v["capscheck.base.caps_pct"] == pytest.approx(0.0)
    assert v["capscheck.pps.evil"] == pytest.approx(3.0)
    assert v["capscheck.pps_eps9.caps_pct"] == pytest.approx(0.0)
    # hyper sweeps share the protocol point with the main-text run
    assert v["hyper.pps.layer20.evil"] == v["hyper.pps.r64.evil"]
    assert v["hyper.ip.r64.prepend.evil"] == pytest.approx(
        truth["grid.base.heavy.ip"])
    assert "hyper.pps.coef.e0p25.evil" in v and "hyper.pps.coef.e9.evil" in v
    tex = (out / "results.tex").read_text()
    assert "sgres@epssel.heldout.evil.e9.evil" in tex


def test_products_skip_until_inputs_exist(tmp_path):
    out, _ = _fabricate_out(tmp_path)
    ctx_plan = json.loads((out / "plan.json").read_text())
    key = {p["name"]: p["identity"] for p in ctx_plan}["ipcount.10"]
    os.remove(out / "evals" / key / "trait_eval_evil_extract.csv")
    os.remove(out / "analyses" / "behavior_curve_pps" / "result.json")
    _, skipped = assemble(str(out))
    assert "fig_ip_prompt_count" in skipped
    assert "fig_behavior_curves" in skipped
    assert "fig_behavior_row" in skipped  # draws the same curves
    assert "fig_eps_selection" not in skipped


def test_one_row_figure2_needs_exactly_its_source_panels():
    """The one-row Figure 2 redraws the grid, specificity and curve
    products' data: it must wait on the same inputs (so it never renders
    while one of them is stale) and own none of their keys."""
    _, products = discover()
    row = products["fig_behavior_row"]
    src = [products[n] for n in ("fig_behavior_grid",
                                 "fig_behavior_specificity",
                                 "fig_behavior_curves")]
    assert set(row.runs) == set().union(*(p.runs for p in src))
    assert set(row.analyses) == set().union(*(p.analyses for p in src))


def _row_figure(tmp_path, monkeypatch):
    """The one-row Figure 2 built on fabricated evals, laid out and kept
    open (``paper_style.save`` would close it)."""
    from grad_shift.paper import paper_style as ps
    from grad_shift.paper.assemble import load_context
    from grad_shift.paper.products.behavior_fig2row import BehaviorFig2Row
    out, _ = _fabricate_out(tmp_path)
    kept = []

    def keep(fig, path, formats=("pdf", "png")):
        fig.draw_without_rendering()  # run the constrained layout
        kept.append(fig)

    monkeypatch.setattr(ps, "save", keep)
    BehaviorFig2Row().build(load_context(str(out)))
    (fig,) = kept
    return fig


def test_figure2_row_style_and_spacing(tmp_path, monkeypatch):
    """2026-10-21: (a) white -> black, no yellow; PPS / IP maps a
    bit further apart, still closer than (a) is to (b); (c) control bars in
    their method's color, still hatched; (d) solid defenseless line."""
    from matplotlib.colors import to_hex
    from matplotlib.patches import Rectangle
    from grad_shift.paper import paper_style as ps
    fig = _row_figure(tmp_path, monkeypatch)
    ax_a, ax_p, ax_i, ax_c, ax_d = fig.axes
    pt = fig.get_figwidth() * 72  # figure fraction -> printed points
    a, p, i = (ax.get_position() for ax in (ax_a, ax_p, ax_i))
    gap_ab, gap_pi = (p.x0 - a.x1) * pt, (i.x0 - p.x1) * pt
    assert gap_pi > 3.4 + 2  # 3.4 pt before 2026-10-21
    assert gap_ab > gap_pi + 4

    # (a): white at 0, black at 100, gray in between
    cmap = ax_a.images[0].get_cmap()
    assert all(c > 0.97 for c in cmap(0.0)[:3])
    assert all(c < 0.03 for c in cmap(1.0)[:3])
    assert all(max(c[:3]) - min(c[:3]) < 0.02
               for c in cmap(np.linspace(0, 1, 51)))

    # the shared x label sits centred under the three maps, below (a)
    r = fig.canvas.get_renderer()
    (label,) = [t for ax in fig.axes for t in ax.texts + [ax.xaxis.label]
                if t.get_text() == "post-finetuning data"]
    bb = label.get_window_extent(r)
    w = fig.bbox.width
    assert (bb.x0 + bb.x1) / 2 / w == pytest.approx((a.x0 + i.x1) / 2,
                                                     abs=0.5 / pt)
    assert bb.y1 < ax_a.get_window_extent(r).y0

    # (c): solid method bars; every control hatched, a defense's controls in
    # that defense's color, base / defenseless gray
    bars = [b for b in ax_c.patches if isinstance(b, Rectangle)]
    face = [to_hex(b.get_facecolor()) for b in bars]
    hatched = [bool(b.get_hatch()) for b in bars]
    gray = {c.lower() for c in ps.BASELINE_GRAYS}
    pps, ip = ps.PPS.lower(), ps.IP.lower()
    # base, none | PPS: evil, random, syco. | IP: evil, syco., neutral, random
    assert face == [face[0], face[1], pps, pps, pps, ip, ip, ip, ip]
    assert {face[0], face[1]} <= gray
    assert hatched == [True, True, False, True, True, False, True, True, True]

    # (d): the defenseless line is solid
    (none,) = [ln for ln in ax_d.lines if ln.get_label() == "Defenseless"]
    assert none.get_linestyle() == "-"
    # (d): no white gap at the right end. The PPS and
    # IP lines run to their last point, and no white-edged end dot covers
    # the last segment (or hides the other method's final point).
    assert not [ln for ln in ax_d.lines
                if to_hex(ln.get_markeredgecolor()) == "#ffffff"]
    ps.plt.close(fig)


def test_sysprompt_maps_are_figure2s_under_the_alt_prompt(tmp_path,
                                                         monkeypatch):
    """2026-11-05: Figure 7's maps are Figure 2a,b under the alt prompt,
    drawn by Figure 2's code: defenseless expression in its grays, and PPS /
    IP as the change from defenseless (both under the alt prompt) in its
    method maps, on one norm shared by both rows."""
    import matplotlib.pyplot as plt
    from grad_shift.paper import paper_style as ps
    from grad_shift.paper.assemble import load_context
    out, _ = _fabricate_out(tmp_path)
    kept = []
    monkeypatch.setattr(ps, "save", lambda fig, path, formats=None: (
        fig.draw_without_rendering(), kept.append(fig)))
    behavior_sysprompt.BehaviorSysPrompt().build(load_context(str(out)))
    fig = kept[0]  # then the curves figure
    maps = {ax.get_title(loc="left"): ax for sf in fig.subfigs
            for ax in sf.axes if ax.images}
    assert sorted(maps) == ["(a) Defenseless", "(b) Δ PPS",
                            "(c) Δ IP", "(e) Defenseless",
                            "(f) Δ PPS", "(g) Δ IP"]
    # fabricated alt-prompt scores: training-matched - 5, so the base model
    # 0, every defenseless / starting model 85 and every defended model 5
    raw = [[0, 85, 85], [85, 85, 85], [85, 85, 85]]
    for k in "ae":
        img = maps[f"({k}) Defenseless"].images[0]
        np.testing.assert_allclose(img.get_array(), raw, atol=1e-9)
        assert img.get_cmap().name == ps.GRAY_SEQUENTIAL_CMAP
    samples = np.linspace(0, 1, 11)
    for k, m in (("b", "pps"), ("c", "ip"), ("f", "pps"), ("g", "ip")):
        img = maps[f"({k}) Δ {m.upper()}"].images[0]
        np.testing.assert_allclose(img.get_array(), np.full((3, 2), -80.0),
                                   atol=1e-9)
        np.testing.assert_allclose(
            img.get_cmap()(samples),
            ps.method_signed_cmap(m, good="negative")(samples))
    norms = {(img.norm.vmin, img.norm.vmax) for k in "bcfg"
             for img in maps[next(t for t in maps if t[1] == k)].images}
    assert len(norms) == 1
    for f in kept:
        plt.close(f)
