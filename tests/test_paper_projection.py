"""Activation-projection appendix: the collection summary, the analyses'
names/runs (they must keep reading the research pipeline's results), and the
product on fabricated data (no GPU, no models)."""

import json
import os

import pytest
import torch

from grad_shift.paper.products import discover
from grad_shift.paper.products import projection_fig as P
from grad_shift.paper.products.base import Context

HIDDEN, N = 16, 60
FL = P.FULL_LAYERS
E = torch.eye(HIDDEN)


def _dom():
    dom = torch.zeros(29, HIDDEN)
    dom[1:, 0] = 5.0  # evil vector along e0 at every layer (non-unit)
    return dom


def _acts(shift, g, base_vecs):
    """Fake collect output: base vectors + a constant shift on every arm."""
    arms = {}
    for arm in P.ARMS:
        a = (base_vecs[arm] + shift
             + 1e-3 * torch.randn(base_vecs[arm].shape, generator=g))
        proj = a[..., 0].mean(1, keepdim=True).expand(N, len(P.LAYERS))
        arms[arm] = {"resp_mean": a, "all_mean": a,
                     "proj_resp": proj, "proj_all": proj,
                     "n_resp_tokens": torch.full((N,), 10),
                     "n_tokens": torch.full((N,), 30)}
    return {"meta": {"model": "m", "layers": list(P.LAYERS),
                     "full_layers": list(FL), "n": N, "seed": 0,
                     "system_prompt": "s", "vector": "v"}, "arms": arms}


def _fixture():
    """Collect outputs of the four models; along the evil axis (e0) the
    defenseless model moves +3, PPS -2 and IP 0 from the base model."""
    g = torch.Generator().manual_seed(0)
    base_vecs = {arm: torch.randn(N, len(FL), HIDDEN, generator=g)
                 for arm in P.ARMS}
    common = 4.0 * E[5]
    shifts = {"base": torch.zeros(HIDDEN), "none": 3.0 * E[0] + common,
              "pps": -2.0 * E[0] + common + 0.5 * E[3], "ip": common}
    return {k: _acts(s, g, base_vecs) for k, s in shifts.items()}


def test_acts_summary_projects_each_row_on_the_unit_vectors():
    acts = _fixture()["pps"]
    vecs = {"evil": _dom(), "syco": _dom(), "rand": _dom()}
    s = P.acts_summary(acts, vecs)
    rows = s["l20_rows"]["evil"]["all"]
    assert len(rows) == N
    li = list(FL).index(P.PROJ_LAYER)
    want = acts["arms"]["evil"]["all_mean"][:, li, 0]  # unit(dom) = e0
    assert rows == pytest.approx(want.tolist(), abs=1e-5)
    assert set(s["proj"]) == set(P.ARMS)
    assert len(s["proj"]["evil"]["all"]) == len(P.LAYERS)


def test_unit_refuses_zero_vectors():
    assert P.unit(torch.tensor([3.0, 4.0])).tolist() == pytest.approx(
        [0.6, 0.8])
    with pytest.raises(ValueError):
        P.unit(torch.zeros(3))


def test_analyses_keep_the_research_pipeline_names_and_runs():
    analyses, products = discover()
    for key, run in P.ACT_MODELS.items():
        a = analyses[f"hydra_acts.{key}"]
        assert a.runs == ([] if run is None else [run])
    assert products["fig_actv_proj"].analyses == [
        "hydra_acts.base", "hydra_acts.none", "hydra_acts.ip",
        "hydra_acts.pps"]


def test_projection_product_builds_on_fabricated_results(tmp_path):
    out = str(tmp_path)
    runs = sorted(r for r in P.ACT_MODELS.values() if r)
    ctx = Context(out=out, run_keys={r: f"id{i}" for i, r in enumerate(runs)})
    vecs = {"evil": _dom(), "syco": _dom(), "rand": _dom()}
    for k, acts in _fixture().items():
        path = os.path.join(out, "analyses", f"hydra_acts.{k}", "result.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(P.acts_summary(acts, vecs), f)
    res = P.ActivationProjectionFigure().build(ctx)
    assert res["actvproj.pps.evil.delta"].value == pytest.approx(-2.0,
                                                                 abs=0.05)
    assert res["actvproj.none.evil.delta"].value == pytest.approx(3.0,
                                                                  abs=0.05)
    assert os.path.isfile(os.path.join(ctx.figures_dir, "fig_actv_proj.pdf"))
    from grad_shift.paper.macros import validate_key
    for k in res:
        validate_key(k)
