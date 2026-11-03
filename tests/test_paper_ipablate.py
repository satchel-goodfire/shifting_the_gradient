"""IP prompt ablations (ipablate_*): pure statistics, run registry, and the
product end to end on fabricated evals / gradient results (no GPU, no judge).
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from grad_shift.paper.products import discover, ip_bank
from grad_shift.paper.products import ipablate_common as ic
from grad_shift.paper.products import ipablate_gradient as ig
from grad_shift.paper.products.base import Context
from paper_testutil import assemble_fabricated as assemble
from grad_shift.paper.runs import EPS, EPS_STRONG, EVIL, SYCO, all_runs
from grad_shift.paper.spec import REPO_ROOT

# ---- pure statistics ---------------------------------------------------------


def test_paired_test_constant_shift_and_sign():
    a = {q: 10.0 + q for q in range(20)}
    b = {q: 12.0 + q + (0.5 if q % 2 else -0.5) for q in range(20)}
    t = ic.paired_test(a, b)
    assert t["diff"] == pytest.approx(-2.0)
    assert t["n"] == 20 and t["n_lower"] == 20 and t["n_higher"] == 0
    assert t["p_t"] < 1e-10 and t["p_wilcoxon"] < 1e-4
    assert t["sem"] == pytest.approx(0.5 * np.sqrt(20 / 19) / np.sqrt(20))


def test_paired_test_refuses_mismatched_questions_and_handles_ties():
    with pytest.raises(ValueError):
        ic.paired_test({0: 1.0, 1: 2.0}, {0: 1.0, 2: 2.0})
    same = {q: float(q) for q in range(5)}
    t = ic.paired_test(same, dict(same))
    assert t["diff"] == 0 and t["p_t"] == 1.0 and t["p_wilcoxon"] == 1.0


def test_spearman_monotone_and_validation():
    s = ic.spearman([1, 2, 3, 4], [10, 8, 5, 1])
    assert s["rho"] == pytest.approx(-1.0) and s["n"] == 4
    with pytest.raises(ValueError):
        ic.spearman([1, 2], [1, 2])
    with pytest.raises(ValueError):
        ic.spearman([1, 2, 3], [1, 2])


def test_token_pooled_weights_tokens_not_sequences():
    # seq A: 1 token, cos 1.0; seq B: 3 tokens, cos 0 each -> 0.25 pooled
    # (the per-sequence C-bar mean would be 0.5)
    assert ic.token_pooled(np.array([1.0, 0.0]), np.array([1, 3])) == 0.25
    with pytest.raises(ValueError):
        ic.token_pooled(np.zeros(2), np.zeros(2))


def test_bank_members_maps_prompts_and_refuses_unknown():
    singles = {"p0": "a", "p1": "b", "p2": "c"}
    assert ic.bank_members(["c", "a"], singles) == ["p2", "p0"]
    with pytest.raises(ValueError):
        ic.bank_members(["a", "z"], singles)
    with pytest.raises(ValueError):
        ic.bank_members(["a"], {"p0": "a", "p1": "a"})


def test_p_value_decimals_keep_two_significant_digits():
    assert ic.p_decimals(0.3) == 2
    assert ic.p_decimals(0.004) == 4
    assert ic.p_result(3.2e-5, "x").formatted() == "0.000032"
    assert ic.p_result(4.0e-11, "x").formatted() == "0.000000000040"


def test_add_banks_is_per_sequence_member_mean():
    per = {k: {s: np.full(3, v) for s in ig.STATS}
           for k, v in (("p0", 1.0), ("p1", 2.0), ("p2", 6.0))}
    ig.add_banks(per, {"bank3": ["p0", "p1", "p2"]})
    assert np.allclose(per["bank3"]["cbar"], 3.0)


# ---- registry ----------------------------------------------------------------


def test_every_regime_run_is_declared_and_inputs_exist():
    runs = all_runs()
    for r in ic.REGIMES.values():
        for arm, name in r.runs.items():
            assert name in runs, (r.key, arm, name)
        for pf in list(r.singles.values()) + list(r.banks.values()):
            assert os.path.isfile(os.path.join(REPO_ROOT, pf)), pf


def test_singles_are_the_bank_prompts_in_order():
    ev = ip_bank.read_prompts("data/prompt_files/evil.json")
    for k in range(5):
        (p,) = ip_bank.read_prompts(ic.EVIL_SINGLES[f"p{k}"])
        assert p == ev[k]
    sy = ip_bank.read_prompts("data/prompt_files/sycophantic.json")
    st = ip_bank.read_prompts("data/prompt_files/strong_sycophantic.json")
    for k in range(5):
        assert ip_bank.read_prompts(ic.SYCO_SINGLES[f"p{k}"]) == [sy[k]]
    for k in range(4):
        assert ip_bank.read_prompts(ic.SYCO_SINGLES[f"s{k}"]) == [st[k]]
    # every bank's gradient arm is an average over measured singles
    for key, r in ic.REGIMES.items():
        m = ig.bank_member_map(r.singles, r.banks)
        assert m["bank3"] == ["p0", "p1", "p2"], key


def test_defense_settings_and_shallow_protocol():
    runs = all_runs()
    for r in ic.REGIMES.values():
        for arm, eps in (("pps", EPS), ("pps_eps9", EPS_STRONG)):
            s = runs[r.runs[arm]]
            assert s.defense == "pps" and s.pps_coef == eps
            assert s.pps_vector == (SYCO if r.key == "syco" else EVIL)
        for arm in list(r.singles) + list(r.banks):
            s = runs[r.runs[arm]]
            assert s.defense == "ip" and s.start == r.start
            assert s.train_file == r.data
    for key, (seed, n) in ic.SHALLOW.items():
        st = runs[f"ipablate.{key}.start"]
        assert st.overrides["max_samples"] == n
        assert st.overrides["num_epochs"] == 1
        assert st.overrides.get("seed", 42) == seed
        for arm in ic.REGIMES[key].runs.values():
            assert runs[arm].overrides.get("seed", 42) == seed
            assert runs[arm].start == st.name


# ---- end to end --------------------------------------------------------------

START = {"syco": 2.0, "deep": 40.0, "s42": 44.0, "s43": 42.0}


def _truth(r, arm):
    """Fabricated final expression: singles 30 - k (p0 best... reversed),
    banks 35, PPS 5 / 3, defenseless 38."""
    if arm in r.singles:
        return 30.0 - r.single_keys.index(arm)
    return {"none": 38.0, "pps": 5.0, "pps_eps9": 3.0,
            "pps_eps1p5": 9.0}.get(arm, 35.0)


def _g(r, arm):
    if arm in r.singles:
        return 0.001 * (r.single_keys.index(arm) + 1)
    return {"none": 0.0005, "pps": 0.03, "pps_eps9": 0.09}[arm]


def _csv(path, trait, mean, seed):
    rng = np.random.default_rng(seed)
    q = np.repeat(np.arange(20), 10)
    noise = rng.normal(0, 1.0, 20)
    noise -= noise.mean()
    v = mean + np.sin(q) * 3 + noise[q]
    v = v - v.mean() + mean
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"question_id": q, "question": q, trait: v,
                  "coherence": 90.0}).to_csv(path, index=False)
    return path


def _fabricate(tmp_path):
    out = tmp_path / "out"
    os.makedirs(out)
    plan = [{"name": n, "identity": f"id{i}"}
            for i, n in enumerate(all_runs())]
    (out / "plan.json").write_text(json.dumps(plan))
    ctx = Context(out=str(out), run_keys={p["name"]: p["identity"]
                                          for p in plan})
    seed = 0
    for key, r in ic.REGIMES.items():
        for arm, run in r.runs.items():
            for trait in ("evil", r.trait):
                seed += 1
                _csv(ctx.eval_csv(run, trait), trait, _truth(r, arm), seed)
        if r.start is not None:
            seed += 1
            _csv(ctx.eval_csv(r.start, "evil"), "evil", START[key], seed)
            os.makedirs(ctx.run_dir(r.start), exist_ok=True)
            with open(os.path.join(ctx.run_dir(r.start),
                                   "experiment_config.json"), "w") as f:
                json.dump({"args": {"train_file": r.data, "num_epochs": 1,
                                    "batch_size": 4, "max_samples": 3200,
                                    "gradient_accumulation_steps": 8}}, f)
    d = out / "analyses" / "base_model_eval"
    base = _csv(str(d / "syco.csv"), "sycophantic", START["syco"], 999)
    os.makedirs(d, exist_ok=True)
    (d / "result.json").write_text(json.dumps(
        {"sycophantic": {"csv": base}}))
    for model in ig.MODELS:
        spec = ig.model_specs()[model]
        r = ic.REGIMES.get(model, ic.REGIMES["deep"])
        arrs = {}
        for c in ["none", "pps", "pps_eps9", *spec["singles"]]:
            g = _g(r, c)
            arrs.update({f"cbar|{c}": np.full(4, g),
                         f"pooled|{c}": np.full(4, 2 * g),
                         f"toksum|{c}": np.full(4, 10 * g),
                         f"ntok|{c}": np.full(4, 10.0),
                         f"loss|{c}": np.ones(4)})
        d = out / "analyses" / ig.analysis_name(model)
        os.makedirs(d, exist_ok=True)
        np.savez(d / "per_sequence.npz", **arrs)
        (d / "result.json").write_text(json.dumps({"npz":
                                                   "per_sequence.npz"}))
    return out


def test_assemble_builds_the_ablation_table(tmp_path):
    out = _fabricate(tmp_path)
    results, skipped = assemble(str(out))
    assert "tab_ip_gradient_ablations" not in skipped, skipped.get(
        "tab_ip_gradient_ablations")
    v = {k: r.value for k, r in results.items()}
    # cells and drops from the start (start minus final, paired)
    assert v["ipabl.deep.start.expr"] == pytest.approx(40.0)
    assert v["ipabl.deep.p1.expr"] == pytest.approx(29.0)
    assert v["ipabl.deep.p1.drop"] == pytest.approx(11.0)
    assert v["ipabl.deep.p1.drop_p"] < 1e-6
    assert v["ipabl.s43.none.drop"] == pytest.approx(4.0)
    assert v["ipabl.prev.p0.drop"] == pytest.approx(2.0 - 30.0)
    # best single = lowest expression (p5 = 25 for evil, s3 = 22 for syco)
    assert results["ipabl.deep.single.best.expr"].meta["arm"] == "p5"
    assert v["ipabl.deep.single.best.drop"] == pytest.approx(15.0)
    assert v["ipabl.prev.single.best.expr"] == pytest.approx(22.0)
    assert v["ipabl.deep.single.mean"] == pytest.approx(27.5)
    assert v["ipabl.deep.single.pooled.drop"] == pytest.approx(12.5)
    # banks vs singles
    assert v["ipabl.deep.bank3.minus_single_mean"] == pytest.approx(7.5)
    assert v["ipabl.deep.bank5.minus_p1"] == pytest.approx(6.0)
    # gradients: token-pooled g20; bank = mean over member prompts
    # paper sign (positive = amplifying): fabricated raw cosines negate
    assert v["ipabl.deep.p1.g20"] == pytest.approx(-0.002)
    assert v["ipabl.deep.bank3.g20"] == pytest.approx(-0.002)
    assert v["ipabl.deep.bank5.g20"] == pytest.approx(-0.003)
    assert v["ipabl.prev.bank4.g20"] == pytest.approx(-0.0075)
    assert v["ipabl.deep.pps.g20_over_max_single"] == pytest.approx(5.0)
    assert v["ipabl.deep.p5.g20_rank"] == 6
    # Spearman: fabricated singles are perfectly anti-ordered in the raw
    # sign, so perfectly ordered in the paper sign
    assert v["ipabl.prev.rho"] == pytest.approx(1.0)
    assert v["ipabl.prev.rho_n"] == 9
    assert v["ipabl.prev.rho_excl_s2_n"] == 8
    assert v["ipabl.deep.rho"] == pytest.approx(1.0)
    assert v["ipabl.g20order.deep_s43"] == pytest.approx(1.0)
    assert v["ipabl.cure.rho_prompts_n"] == 18
    assert v["ipabl.cure.rho_all_n"] == 3 * (1 + 6 + 2 + 1)
    assert v["ipabl.fresh.none.g20"] == pytest.approx(-0.0005)
    assert v["ipabl.s43.start_steps"] == 100
    for t in ("tab_ip_gradient_ablations", "tab_ip_ablation_prompts"):
        tex = (out / "tables" / f"{t}.tex").read_text()
        assert tex.startswith("\\begin{tabular}") and "\\bottomrule" in tex


def test_product_skips_until_gradients_exist(tmp_path):
    out = _fabricate(tmp_path)
    os.remove(out / "analyses" / ig.analysis_name("s42") / "result.json")
    _, skipped = assemble(str(out))
    assert "missing: analysis ipablate_grad.s42" in skipped[
        "tab_ip_gradient_ablations"]


def test_analyses_are_discovered_with_their_start_runs():
    analyses, products = discover()
    assert analyses[ig.analysis_name("deep")].runs == ["start.medium"]
    assert analyses[ig.analysis_name("syco")].runs == []
    assert analyses[ig.analysis_name("s42")].runs == ["ipablate.s42.start"]
    assert "tab_ip_gradient_ablations" in products
