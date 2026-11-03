"""Interfaces for analyses (GPU work over models) and products (outputs).

An ``Analysis`` computes numbers from models/data (gradient cosines, losses,
activation projections, ...). It runs as its own SLURM job via
``python -m grad_shift.paper.run_analysis <name> --out DIR`` and writes
``<out>/analyses/<name>/result.json`` (plus any arrays it wants).

A ``Product`` renders one figure/table and returns the in-text numbers it
owns as ``{macro_key: macros.Result}``. Products run on CPU in
``sg-paper assemble`` and only read evals and analysis results, so figures
rebuild in seconds without GPUs or API calls.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..macros import Result
from ..plan import EVAL_SPLIT
from ..spec import eval_marker


@dataclass
class Context:
    """What analyses/products can see: the store and the run plan."""

    out: str
    run_keys: Dict[str, str]  # run name -> store key (identity[-ckptN])
    # run name -> traits it is judged on (from plan.json; default ("evil",))
    run_traits: Dict[str, tuple] = field(default_factory=dict)
    # run name -> question splits it is judged on (default (EVAL_SPLIT,))
    run_splits: Dict[str, tuple] = field(default_factory=dict)
    # run name -> eval system prompt sentinel (absent/None = training prompt)
    run_eval_sys: Dict[str, Optional[str]] = field(default_factory=dict)

    @property
    def figures_dir(self) -> str:
        return self._mk("figures")

    @property
    def tables_dir(self) -> str:
        return self._mk("tables")

    def _mk(self, sub: str) -> str:
        d = os.path.join(self.out, sub)
        os.makedirs(d, exist_ok=True)
        return d

    def run_dir(self, run: str) -> str:
        """Directory of a run's trained adapter (by run name)."""
        return os.path.join(self.out, "runs", self.run_keys[run])

    def eval_csv(self, run: str, trait: str = "evil",
                 split: str = EVAL_SPLIT) -> str:
        mark = eval_marker(self.run_eval_sys.get(run))
        return os.path.join(self.out, "evals", self.run_keys[run],
                            f"trait_eval_{trait}_{split}{mark}.csv")

    def eval_df(self, run: str, trait: str = "evil",
                split: str = EVAL_SPLIT) -> pd.DataFrame:
        return pd.read_csv(self.eval_csv(run, trait, split))

    def trait_score(self, run: str, trait: str = "evil",
                    split: str = EVAL_SPLIT, column: Optional[str] = None
                    ) -> Result:
        """Mean +- SEM of a judged score over all question x sample rows."""
        df = self.eval_df(run, trait, split)
        col = column or trait
        s = df[col].dropna()
        return Result(value=float(s.mean()),
                      sem=float(s.std(ddof=1) / len(s) ** 0.5),
                      source=f"eval:{run}:{trait}:{split}:{col}")

    def analysis_dir(self, name: str) -> str:
        return self._mk(os.path.join("analyses", name))

    def analysis_result(self, name: str) -> dict:
        with open(os.path.join(self.out, "analyses", name, "result.json")) as f:
            return json.load(f)


INPUTS_FILE = "inputs.json"  # fingerprint a result.json was computed from
PENDING_FILE = "inputs.pending.json"  # fingerprint of an unfinished job


class Analysis:
    """GPU computation over models; override ``name``, ``runs``, ``run``.

    The identities of ``runs``, the contents of ``inputs`` (repo-relative
    files the computation reads: vectors, data, prompt files) and ``params``
    (settings that change the result) form its fingerprint, stored beside
    the result. A result whose stored fingerprint differs from the current
    one is stale: ``sg-paper analyze`` recomputes it and ``assemble`` will
    not render it. Analyses that declare no ``inputs`` are fingerprinted by
    their runs alone, so a changed input file goes unnoticed for them.
    """

    name: str = ""
    runs: List[str] = []  # run names whose adapters it needs
    inputs: List[str] = []  # repo-relative files whose contents it depends on
    params: Dict = {}  # JSON-able settings that change the result
    gpus: int = 1
    time: str = "04:00:00"

    def run(self, ctx: Context) -> dict:
        """Compute and return the result (saved as result.json)."""
        raise NotImplementedError

    def fingerprint(self, ctx: Context) -> dict:
        """What the result depends on (JSON-normalized, order-free)."""
        from ..identity import file_digest
        from ..spec import REPO_ROOT

        return json.loads(json.dumps({
            "runs": {r: ctx.run_keys[r] for r in sorted(self.runs)},
            "inputs": {p: file_digest(p, REPO_ROOT)
                       for p in sorted(set(self.inputs))},
            "params": self.params,
        }, sort_keys=True))


def read_json(path: str) -> Optional[dict]:
    """Parsed JSON file, or None if it does not exist."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def analysis_status(a: Analysis, ctx: Context) -> str:
    """'missing' (no result), 'current', 'stale' (computed from other
    inputs) or 'unverified' (a result with no fingerprint: written before
    fingerprints existed, see ``sg-paper adopt-analyses``)."""
    d = os.path.join(ctx.out, "analyses", a.name)
    if not os.path.isfile(os.path.join(d, "result.json")):
        return "missing"
    rec = read_json(os.path.join(d, INPUTS_FILE))
    if rec is None:
        return "unverified"
    return "current" if rec == a.fingerprint(ctx) else "stale"


def archive_contents(d: str, stamp: str) -> Optional[str]:
    """Move everything in analysis dir ``d`` (except earlier archives and the
    SLURM ``.job`` marker) into ``d/stale-<stamp>/``; return that dir, or
    None if there was nothing to move. Nothing is deleted."""
    names = [n for n in os.listdir(d)
             if not n.startswith("stale-") and n != ".job"]
    if not names:
        return None
    dst = base = os.path.join(d, f"stale-{stamp}")
    k = 0
    while os.path.exists(dst):  # two archives within one stamp
        k += 1
        dst = f"{base}-{k}"
    os.makedirs(dst)
    for n in names:
        os.replace(os.path.join(d, n), os.path.join(dst, n))
    return dst


class Product:
    """A figure/table + its numbers; override ``name``, deps, ``build``."""

    name: str = ""
    runs: List[str] = []  # needs final-model evals of these runs
    analyses: List[str] = []  # needs these analyses' results

    def build(self, ctx: Context) -> Dict[str, Result]:
        """Write figure/table files; return the in-text numbers it owns."""
        raise NotImplementedError
