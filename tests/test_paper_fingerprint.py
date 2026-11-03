"""Analysis fingerprints: stale results are detected, recomputed and never
rendered; cached files are reused only under the same fingerprint."""

import json
import os

import pytest

from grad_shift.paper import assemble as asm
from grad_shift.paper import cli
from grad_shift.paper import products as products_pkg
from grad_shift.paper.products.base import (INPUTS_FILE, PENDING_FILE,
                                            Analysis, Context, Product,
                                            analysis_status, archive_contents)
from grad_shift.paper.run_analysis import prepare_dir


class Probe(Analysis):
    name = "probe"
    runs = ["r1"]

    def __init__(self, inputs=(), params=None):
        self.inputs = list(inputs)
        self.params = params or {}


def _ctx(tmp_path, key="id_r1"):
    return Context(out=str(tmp_path / "out"), run_keys={"r1": key})


def _result(ctx, name="probe", fp=None):
    d = ctx.analysis_dir(name)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({"x": 1}, f)
    if fp is not None:
        with open(os.path.join(d, INPUTS_FILE), "w") as f:
            json.dump(fp, f)
    return d


def test_status_tracks_runs_input_contents_and_params(tmp_path):
    vec = tmp_path / "vec.pt"
    vec.write_bytes(b"v1")
    a = Probe(inputs=[str(vec)], params={"traits": ("a", "b")})
    ctx = _ctx(tmp_path)
    assert analysis_status(a, ctx) == "missing"
    _result(ctx)
    assert analysis_status(a, ctx) == "unverified"
    _result(ctx, fp=a.fingerprint(ctx))
    assert analysis_status(a, ctx) == "current"
    vec.write_bytes(b"v2")  # same path, new contents (the Qwen3-4B case)
    assert analysis_status(a, ctx) == "stale"
    vec.write_bytes(b"v1")
    assert analysis_status(a, _ctx(tmp_path, key="id_other")) == "stale"
    assert analysis_status(Probe(inputs=[str(vec)], params={"traits": ["a"]}),
                           ctx) == "stale"
    # JSON-normalized: tuple vs list and input order do not matter
    same = Probe(inputs=[str(vec), str(vec)], params={"traits": ["a", "b"]})
    assert analysis_status(same, ctx) == "current"


def test_prepare_dir_archives_unless_made_under_the_same_fingerprint(tmp_path):
    d = tmp_path / "a"
    d.mkdir()
    (d / "cache.csv").write_text("old")
    (d / ".job").write_text("123\n")
    prepare_dir(str(d), {"v": 1})  # files but no record: archived
    stale = [n for n in os.listdir(d) if n.startswith("stale-")]
    assert len(stale) == 1 and (d / stale[0] / "cache.csv").exists()
    assert (d / ".job").exists()  # the SLURM marker stays
    assert json.loads((d / PENDING_FILE).read_text()) == {"v": 1}
    # a requeued job (pending matches) keeps its partial caches
    (d / "step0.npz").write_text("partial")
    prepare_dir(str(d), {"v": 1})
    assert (d / "step0.npz").exists()
    # finished under v1, rerun under v2: everything but archives moves
    os.replace(d / PENDING_FILE, d / INPUTS_FILE)
    (d / "result.json").write_text("{}")
    prepare_dir(str(d), {"v": 2})
    assert not (d / "result.json").exists() and not (d / "step0.npz").exists()
    assert len([n for n in os.listdir(d) if n.startswith("stale-")]) == 2


def test_archive_contents_moves_nothing_when_empty(tmp_path):
    (tmp_path / ".job").write_text("1\n")
    assert archive_contents(str(tmp_path), "t") is None


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A one-analysis registry and plan in a fresh output folder."""
    vec = tmp_path / "vec.pt"
    vec.write_bytes(b"v1")

    class P(Product):
        name = "fig_probe"
        analyses = ["probe"]

        def build(self, ctx):
            return {}

    a = Probe(inputs=[str(vec)])
    monkeypatch.setattr(products_pkg, "discover",
                        lambda: ({"probe": a}, {"fig_probe": P()}))
    monkeypatch.setattr(asm, "discover",
                        lambda: ({"probe": a}, {"fig_probe": P()}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "plan.json").write_text(json.dumps([{"name": "r1",
                                                "identity": "id_r1"}]))
    return a, vec, str(out)


def test_submit_resubmits_stale_and_unverified_results(registry, capsys):
    a, vec, out = registry
    ctx = asm.load_context(out)
    assert cli.submit_analyses(out, dry_run=True) == {"probe": "dry"}
    _result(ctx)
    assert cli.submit_analyses(out, dry_run=True) == {"probe": "dry"}
    assert "[unverified] probe" in capsys.readouterr().err
    _result(ctx, fp=a.fingerprint(ctx))
    assert cli.submit_analyses(out, dry_run=True) == {"probe": "done"}
    vec.write_bytes(b"v2")
    assert cli.submit_analyses(out, dry_run=True) == {"probe": "dry"}
    assert "[stale] probe" in capsys.readouterr().err


def test_assemble_does_not_render_stale_results(registry):
    a, vec, out = registry
    ctx = asm.load_context(out)
    _result(ctx, fp=a.fingerprint(ctx))
    assert asm.missing_inputs(asm.discover()[1]["fig_probe"], ctx) == []
    vec.write_bytes(b"v2")
    assert asm.missing_inputs(asm.discover()[1]["fig_probe"], ctx) == [
        "analysis probe (stale)"]


def test_adopt_records_current_fingerprints_except_excluded(registry):
    a, vec, out = registry
    ctx = asm.load_context(out)
    _result(ctx)
    assert cli.adopt_analyses(out, exclude=["probe"]) == {"probe": "excluded"}
    assert analysis_status(a, ctx) == "unverified"
    assert cli.adopt_analyses(out, exclude=[]) == {"probe": "adopted"}
    assert analysis_status(a, ctx) == "current"
    assert cli.adopt_analyses(out, exclude=[]) == {}  # nothing left to adopt
    with pytest.raises(SystemExit):
        cli.adopt_analyses(out, exclude=["no_such_analysis"])


def test_refingerprint_rerecords_only_named_stale_results(registry):
    a, vec, out = registry
    ctx = asm.load_context(out)
    _result(ctx, fp=a.fingerprint(ctx))
    a.params = {"eps": 2.0}  # newly declared setting, same computation
    assert analysis_status(a, ctx) == "stale"
    assert cli.adopt_analyses(out, exclude=[]) == {}  # plain adopt: no-op
    assert analysis_status(a, ctx) == "stale"
    assert cli.adopt_analyses(out, exclude=[], refingerprint=["probe"]) == {
        "probe": "refingerprinted"}
    assert analysis_status(a, ctx) == "current"
