"""Cross-model group (models_*): registry, pure helpers, and products built
from fabricated analysis outputs and eval CSVs (no GPU, no judge)."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from grad_shift.paper import macros
from grad_shift.paper.products import models_analyses as ma
from grad_shift.paper.products import models_common as mc
from grad_shift.paper.products.models_common import ALL_LINEAR
from grad_shift.paper.products import models_products as mp
from grad_shift.paper.products.base import Context
from grad_shift.paper.caps_choice import choice
from grad_shift.paper.products import ip_bank
from grad_shift.paper.runs import EPS, EPS_STRONG, EVIL_BANKS, RUNS
from grad_shift.paper.spec import REPO_ROOT, materialize


# ---------------------------------------------------------------- registry

def test_matched_layers_equal_the_archived_cross_model_chain():
    # grad_shift xmodel_evil configs: 14B L34, Qwen3-4B L26, Llama L23;
    # the paper's 7B layer 20 of 28 maps to itself.
    assert mc.matched_layer(28) == 20
    assert {k: m.layer for k, m in mc.MODELS.items()} == {
        "q3_4b": 26, "q25_14b": 34, "llama8b": 23}


def test_inputs_present_and_runs_declared():
    assert mc.inputs_present(), [f for f in mc.required_files()
                                 if not os.path.isfile(
                                     os.path.join(REPO_ROOT, f))]
    names = [s.name for s in mc.EXTRA_RUNS]
    assert len(names) == len(set(names)) == 28
    assert not set(names) & set(RUNS)
    assert all(n.startswith("models.") for n in names)


def test_runs_follow_the_protocol_with_only_model_and_layer_swapped():
    for s in mc.EXTRA_RUNS:
        _, model, trait, arm = s.name.split(".")
        t, m = mc.TRAITS[f"{model}.{trait}"], mc.MODELS[model]
        # only the model swapped -- plus, for a model type the trainer has
        # no LoRA default for (Qwen3-4B), the paper's all-linear targets
        assert s.overrides == m.overrides()
        if m.lora_targets is None:
            assert s.overrides == {"model": m.hf}
        else:
            assert tuple(materialize(s, "X")["target_modules"]) == ALL_LINEAR
        assert s.pps_layers == (m.layer,)
        assert s.eval_traits == (t.trait,)
        assert s.train_file == t.data
        cfg = materialize(s, "X")
        assert cfg["lora_r"] == 64 and cfg["num_epochs"] == 3
        assert cfg["learning_rate"] == 1e-5
        if arm == "pps":
            assert (s.pps_vector, s.pps_coef) == (t.vector, t.eps)
        if arm == "pps_eps9":
            assert (s.pps_vector, s.pps_coef) == (t.vector, EPS_STRONG)
        if arm.startswith("rand"):
            assert s.pps_vector == t.rand
        if arm == "ip":
            assert s.prompt_file == t.bank
        if arm == "ip_p1":
            assert trait == "evil" and s.prompt_file == t.p1
        ck = (m.checkpoint and trait == "evil" and arm in mc.MAIN_ARMS)
        assert s.checkpoint_every == (mc.CKPT_EVERY if ck else None)


def test_trait_banks_and_eps_are_the_recorded_choices():
    from grad_shift.paper.caps_choice import choice
    from grad_shift.paper.runs import EVIL_BANKS
    for t in mc.TRAITS.values():
        if t.trait == "evil":  # retuned per model (caps-matched)
            n, eps = choice(t.model)
            assert (t.bank, t.eps) == (EVIL_BANKS[n], eps)
            continue
        # the Sec. 7 Qwen3-4B traits keep the first three prompts and EPS
        full = t.bank.replace("_3samp", "")
        with open(os.path.join(REPO_ROOT, full)) as f:
            five = json.load(f)["prompts"]
        with open(os.path.join(REPO_ROOT, t.bank)) as f:
            three = json.load(f)["prompts"]
        assert three == five[:3] and t.eps == EPS


def test_shown_arms_drop_p1_when_it_is_the_main_bank():
    from grad_shift.paper.runs import P1
    for tk in mc.EVIL_TRAITS:
        t = mc.TRAITS[tk]
        shown = mp.evil_arms(tk)
        assert ("ip_p1" in shown) == (t.bank != P1), tk
        n = mp.n_prompts(tk)
        assert mp.ip_label(tk) == (f"IP ({n} prompts)" if n > 1
                                   else "IP (1 prompt)")


def test_vectors_have_one_row_per_block_plus_embedding():
    for t in mc.TRAITS.values():
        n = mc.MODELS[t.model].n_blocks
        for p in (t.vector, t.rand):
            v = torch.load(os.path.join(REPO_ROOT, p), weights_only=True)
            assert v.shape[0] == n + 1, p


def test_random_normmatched_matches_norms_and_is_deterministic():
    ref = torch.randn(5, 16)
    ref[0] = 0
    a = mc.make_random_normmatched(ref, seed=0)
    b = mc.make_random_normmatched(ref, seed=0)
    assert torch.equal(a, b)
    assert torch.allclose(a.norm(dim=-1), ref.norm(dim=-1), atol=1e-5)
    assert torch.all(a[0] == 0)
    # random direction: not aligned with the reference
    cos = torch.nn.functional.cosine_similarity(a[1:], ref[1:], dim=-1)
    assert cos.abs().max() < 0.9


def test_committed_random_controls_reproduce_from_the_recipe():
    for m, path in mc.RAND.items():
        ref = torch.load(os.path.join(REPO_ROOT, mc.EVIL_VEC[m]),
                         weights_only=True)
        got = torch.load(os.path.join(REPO_ROOT, path), weights_only=True)
        assert torch.allclose(got, mc.make_random_normmatched(ref, 0))


def test_qwen3_4b_evil_vector_is_from_the_shipped_borrowed_text():
    # Its provenance record names the 14B CSVs that ship in the repo (by
    # content), and the 759-pair count follows from them under the filter.
    import hashlib

    vec = mc.EVIL_VEC["q3_4b"]
    with open(os.path.join(REPO_ROOT, vec[:-len(".pt")] + ".meta.json")) as f:
        meta = json.load(f)
    assert meta["model"] == mc.MODELS["q3_4b"].hf
    assert meta["method"] == "persona_vector_borrowed_responses"
    assert meta["threshold"] == 50 and meta["n_prompts_rerendered"] == 0
    csv = {}
    for side in ("pos", "neg"):
        path = os.path.join(REPO_ROOT, meta[f"{side}_csv"])
        with open(path, "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == \
                meta[f"{side}_csv_sha256"]
        csv[side] = pd.read_csv(path)
    pos, neg = csv["pos"], csv["neg"]
    keep = ((pos.evil >= 50) & (neg.evil < 50) & (pos.coherence >= 50)
            & (neg.coherence >= 50))
    assert int(keep.sum()) == meta["n_effective"] == 759
    # the model's own extraction failed: no positive sample scored >= 50
    own = pd.read_csv(os.path.join(
        REPO_ROOT, "steering_vectors/Qwen_Qwen3-4B-Instruct-2507_extract/"
        "evil_pos.csv"))
    assert own.evil.max() < 1


# ---------------------------------------------------------------- helpers

def test_sweep_points_find_low_intensity_dip():
    curve = {0.0: 3.0, 0.25: 2.9, 0.5: 2.8, 1.0: 2.95, 2.0: 3.1, 9.0: 4.0}
    sp = mc.sweep_points(curve)
    assert sp["argmin"] == 0.5
    assert sp["min_dloss"] == pytest.approx(-0.2)
    assert sp["reducing"] == [0.25, 0.5, 1.0] and sp["max_reducing"] == 1.0
    assert mc.sweep_points({0.0: 1.0, 1.0: 2.0})["max_reducing"] == 0.0
    with pytest.raises(ValueError):
        mc.sweep_points({1.0: 1.0})


def test_first_flip():
    # paper sign: positive = amplifying; the flip is to negative
    assert mc.first_flip({0.0: 0.2, 1.0: 0.1, 3.0: -0.1, 5.0: -0.2}) == 3.0
    assert mc.first_flip({0.0: -0.13, 3.0: -0.25}) is None  # nothing to flip
    assert mc.first_flip({0.0: 0.2, 3.0: 0.1}) is None


def test_bank_average_and_ip_arms():
    # Qwen3-4B's caps-matched n* is 1 since the all-linear LoRA retune
    # (2026-10-21; it was 3 under the attention-only runs): its bank is p1
    t = mc.TRAITS["q3_4b.evil"]
    assert t.bank == EVIL_BANKS[choice("q3_4b")[0]] == EVIL_BANKS[1]
    per = {"ip_b0": np.array([1.0, 2.0])}
    ma.add_ip_arms(per, "q3_4b.evil")
    assert np.allclose(per["ip"], [1.0, 2.0])
    assert np.allclose(per["ip_p1"], per["ip_b0"])
    # a 3-prompt bank averages its members; evil p1 is its second prompt
    per = {"ip_b0": np.array([1.0, 2.0]), "ip_b1": np.array([3.0, 4.0]),
           "ip_b2": np.array([5.0, 6.0])}
    assert np.allclose(mc.bank_average(list(per.values())), [3.0, 4.0])
    assert ip_bank.p1_index(EVIL_BANKS[3]) == 1
    # ... and of the 10-prompt bank (Qwen2.5-14B's caps-matched n*)
    assert mc.TRAITS["q25_14b.evil"].bank.endswith("evil_10samp.json")
    per10 = {f"ip_b{k}": np.array([float(k)]) for k in range(10)}
    ma.add_ip_arms(per10, "q25_14b.evil")
    assert np.allclose(per10["ip"], [4.5])
    assert np.allclose(per10["ip_p1"], per10["ip_b1"])
    per2 = {k: v.copy() for k, v in per.items() if k.startswith("ip_b")}
    ma.add_ip_arms(per2, "q3_4b.pirate")
    assert "ip_p1" not in per2
    with pytest.raises(ValueError):
        mc.bank_average([np.zeros(2), np.zeros(3)])


def test_step0_conditions():
    names = [c.name for c in ma.step0_conditions("q25_14b.evil")]
    assert names[0] == "none" and "pps_c2" in names and "pps_c9" in names
    assert {"ip_b0", "ip_b1", "ip_b2", "rand_c2", "rand_c9"} <= set(names)
    assert not any(n.startswith("pps_l") for n in names)
    assert len(names) == len(set(names))
    conds = {c.name: c for c in ma.step0_conditions("q3_4b.pirate")}
    assert conds["pps_c3"].pps_layers == (26,)
    assert conds["pps_l25_c3"].pps_layers == (25,)
    assert "pps_l26_c3" not in conds  # matched layer = the main sweep
    assert ma.hook_layers("q3_4b") == (20, 23, 25, 26, 29)
    assert ma.hook_layers("q25_14b") == (34,)


# ---------------------------------------------------------------- products

def _ctx(tmp_path, runs):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    plan = [{"name": r, "identity": f"id{i}"} for i, r in enumerate(runs)]
    (out / "plan.json").write_text(json.dumps(plan))
    return Context(out=str(out), run_keys={p["name"]: p["identity"]
                                           for p in plan})


def _csv(path, trait, value, coh=85.0, n=40, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({trait: value + rng.normal(0, 1, n),
                  "coherence": coh + rng.normal(0, 1, n)}).to_csv(path,
                                                                  index=False)


def _write(ctx, name, result, **arrays):
    d = ctx.analysis_dir(name)
    if arrays:
        np.savez(os.path.join(d, "per_sequence.npz"), **arrays)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump(result, f)


def fake_step0(ctx, trait_key, n=30, gcos=None, loss=None):
    """Planted: C-bar along the PV = gcos(cond, layer); loss = loss(cond)."""
    t = mc.TRAITS[trait_key]
    layers = ma.hook_layers(t.model)
    conds = [c.name for c in ma.step0_conditions(trait_key)]
    conds += ["ip"] + (["ip_p1"] if t.p1 else [])
    rng = np.random.default_rng(0)
    arrays = {"ntok": np.full(n, 100)}
    for c in conds:
        tm = 1e-4 * rng.normal(size=(n, len(layers), 2))
        for j, layer in enumerate(layers):
            tm[:, j, 0] += gcos(c, layer)
        arrays[f"tokmean|{c}"] = tm
        arrays[f"pooled|{c}"] = tm
        arrays[f"loss|{c}"] = loss(c) + 1e-4 * rng.normal(size=n)
    m = mc.MODELS[t.model]
    _write(ctx, ma.step0_name(trait_key), {
        "npz": "per_sequence.npz", "layer": m.layer, "layers": list(layers),
        "dir_names": list(ma.DIR_NAMES),
        "sweep": list(ma.sweep_for(trait_key)),
        "rand_sweep": list(ma.rand_sweep_for(trait_key))}, **arrays)


def fake_baseeval(ctx, trait_key, base=1.0, steer=60.0):
    t = mc.TRAITS[trait_key]
    name = ma.baseeval_name(trait_key)
    d = ctx.analysis_dir(name)
    evals = {}
    for k, v in [("base", base)] + [(f"steer_eps{mc.eps_tag(c)}", steer)
                                    for c in ma.steer_coefs(trait_key)]:
        p = os.path.join(d, f"{k}.csv")
        _csv(p, t.trait, v)
        evals[k] = {"csv": p}
    _write(ctx, name, {"evals": evals})


def coef_of(cond):
    """Intensity of a pps_c<x> condition name (fabrication helper)."""
    return float(cond.split("_c")[1].replace("p", "."))


def test_models_evil_product_reports_planted_values(tmp_path):
    tk = "q25_14b.evil"
    arm_names = list(mc.arms(mc.TRAITS[tk]))
    runs = [mc.run_name(tk, a) for a in arm_names]
    ctx = _ctx(tmp_path, runs)
    expr = {"none": 90.0, "pps": 70.0, "ip": 30.0, "pps_eps9": 20.0,
            "ip_p1": 3.0}
    for i, a in enumerate(arm_names):
        _csv(ctx.eval_csv(mc.run_name(tk, a)), "evil", expr[a], seed=i)

    def gcos(c, layer):
        if c.startswith("pps_c"):
            return -0.01 + 0.004 * coef_of(c)  # zero at 2.5: flips at 3
        return {"none": -0.01, "ip": 0.002, "ip_p1": 0.004}.get(c, -0.008)

    def loss(c):  # dip at low intensity, rise above
        if c.startswith("pps_c"):
            x = coef_of(c)
            return 3.0 - 0.1 * x + 0.06 * x * x
        return {"ip": 2.7, "ip_p1": 2.6}.get(c, 3.0)

    fake_step0(ctx, tk, gcos=gcos, loss=loss)
    fake_baseeval(ctx, tk)
    r = mp.ModelsEvil_q25_14b().build(ctx)
    p = "models.q25_14b.evil"
    assert r[f"{p}.none"].value == pytest.approx(90, abs=1)
    assert r[f"{p}.ip_p1"].value == pytest.approx(3, abs=1)
    assert r[f"{p}.base"].value == pytest.approx(1, abs=1)
    assert r[f"{p}.steer_eps2"].value == pytest.approx(60, abs=1)
    # paper sign (positive = amplifying): fabricated raw cosines negate
    assert r[f"{p}.gradcos.none"].value == pytest.approx(0.01, abs=1e-4)
    assert r[f"{p}.gradpool.none"].value == pytest.approx(0.01, abs=1e-4)
    eps = mc.TRAITS[tk].eps  # the main PPS arm's intensity
    assert r[f"{p}.gradcos.pps"].value == pytest.approx(
        -(-0.01 + 0.004 * eps), abs=1e-4)
    assert r[f"{p}.dgradcos.pps_eps9"].value == pytest.approx(-0.036,
                                                              abs=1e-4)
    assert r[f"{p}.dloss.ip"].value == pytest.approx(-0.3, abs=1e-3)
    # argmin of 3 - 0.1x + 0.06x^2 on the sweep grid is x = 0.75 (true 0.83)
    assert r[f"{p}.sweep.argmin"].value == 0.75
    assert r[f"{p}.sweep.max_reducing"].value == 1.5
    assert r[f"{p}.sweep.flip"].value == 3.0
    assert r[f"{p}.layer"].value == 34
    for k in r:
        macros.validate_key(k)
    assert os.path.isfile(os.path.join(ctx.figures_dir,
                                       "fig_models_evil_q25_14b.pdf"))
    assert os.path.isfile(os.path.join(ctx.tables_dir,
                                       "tab_models_evil_q25_14b.tex"))
    # signed values in math mode (2026-11-06): a true minus sign, never a
    # bare text hyphen before a digit
    import re
    tab = open(os.path.join(ctx.tables_dir, "tab_models_evil_q25_14b.tex")).read()
    assert re.search(r"\$-0\.\d+\$", tab)
    assert not re.search(r"(?<![$\w])-\d", tab)


def test_q34b_trait_product_layer_sweep_and_missing_evals(tmp_path):
    tk = "q3_4b.pirate"
    arm_names = list(mc.arms(mc.TRAITS[tk]))
    runs = [mc.run_name(tk, a) for a in arm_names]
    ctx = _ctx(tmp_path, runs)

    def gcos(c, layer):
        if c.startswith("pps_l"):
            x = coef_of(c)
            return -0.2 + 0.11 * x  # flips at every swept layer
        if c.startswith("pps_c"):
            return -0.2 + 0.11 * coef_of(c)
        return -0.2 if c == "none" else 0.01

    fake_step0(ctx, tk, gcos=gcos, loss=lambda c: 2.0)
    fake_baseeval(ctx, tk, base=0.0, steer=0.0)
    with pytest.raises(FileNotFoundError):
        mp.Q34b_pirate().build(ctx)
    for i, a in enumerate(arm_names):
        _csv(ctx.eval_csv(mc.run_name(tk, a), "pirate"), "pirate",
             0.0 if a.startswith("pps") else 95.0, seed=i)
    r = mp.Q34b_pirate().build(ctx)
    assert r["q34b.pirate.none"].value == pytest.approx(95, abs=1)
    assert r["q34b.pirate.pps3"].value == pytest.approx(0, abs=1)
    assert "q34b.pirate.rand3" in r and "q34b.pirate.pps1p5" in r
    # fabricated raw cosines; keys are in the paper sign (negated)
    assert r["q34b.pirate.gradcos.c0"].value == pytest.approx(0.2, abs=1e-3)
    assert r["q34b.pirate.gradcos.c3"].value == pytest.approx(-0.13, abs=1e-3)
    # fabricated pooled == tokmean, so both statistics report the same
    assert r["q34b.pirate.gradpool.c3"].value == pytest.approx(-0.13, abs=1e-3)
    assert r["q34b.pirate.layersweep_pool.nflip"].value == 5
    assert r["q34b.pirate.layersweep.nflip"].value == 5
    assert r["q34b.pirate.layersweep.natten_c0"].value == 0
    assert r["q34b.pirate.layersweep.l25.c3"].value == pytest.approx(
        -0.13, abs=1e-3)
    assert r["q34b.pirate.steer_eps3"].value == pytest.approx(0, abs=1)
    for k in r:
        macros.validate_key(k)


def test_layer_counts():
    # paper sign: 20 flips to attenuating, 23 already attenuates, 25 never
    ls = {20: {0.0: 0.1, 3.0: -0.1}, 23: {0.0: -0.1, 3.0: -0.2},
          25: {0.0: 0.1, 3.0: 0.05}}
    assert mp.layer_counts(ls) == {"nflip": 1, "natten_c0": 1}


def test_dynamics_product(tmp_path):
    ctx = _ctx(tmp_path, [])
    for arm in mc.MAIN_ARMS:
        summary = []
        for step in (0, 400, 800):
            for cond, v in (("none", -0.01 + 1e-5 * step),
                            ("pps", 0.02), ("ip", 0.0)):
                summary.append({"step": step, "cond": cond, "gcos_mean": v,
                                "gcos_sem": 0.001})
        _write(ctx, ma.dyn_name("q3_4b.evil", arm), {"summary": summary})
    r = mp.ModelsDyn_q3_4b().build(ctx)
    k = "models.q3_4b.evil.graddyn.pps.none"
    # paper sign (positive = amplifying): fabricated raw cosines negate
    assert r[f"{k}.step0"].value == pytest.approx(0.01)
    assert r[f"{k}.final"].value == pytest.approx(0.002)
    assert len(r) == 3 * 3 * 2
