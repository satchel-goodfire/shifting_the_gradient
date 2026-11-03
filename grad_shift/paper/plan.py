"""Resolve run specs to identities and decide what each needs.

For every run (in dependency order) the plan records its identity and one
status:

- ``present``: complete in the output store.
- ``adopt``: a complete run with the same identity exists in a legacy root;
  it will be symlinked into the store.
- ``train``: must be trained (possibly after its starting model trains).

Evals are tracked separately: every run gets judged into
``<out>/evals/<identity>/trait_eval_<trait>_<split>.csv`` unless present.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import runs as runs_mod
from .spec import (RunSpec, eval_marker, identity_of, materialize,
                   topological_order)
from .store import LegacyIndex, Store

EVAL_SPLIT = "extract"  # the paper's reporting question split


def store_key(identity: str, checkpoint_every: Optional[int]) -> str:
    """Store directory name: identity, plus the checkpoint cadence if needed.

    A run that must keep intermediate checkpoints is stored separately from a
    weights-identical run that did not, so it cannot be satisfied by one.
    """
    return identity if checkpoint_every is None else (
        f"{identity}-ckpt{int(checkpoint_every)}")


@dataclass
class PlannedRun:
    name: str
    spec: RunSpec
    identity: str
    status: str  # present | adopt | train
    source: Optional[str]  # where it lives now (present/adopt)
    config: dict  # materialized finetune config (output_dir = store dir)

    def eval_pairs(self) -> List[Tuple[str, str]]:
        """(trait, split) pairs the final model is judged on."""
        return [(t, s) for t in self.spec.eval_traits
                for s in self.spec.eval_splits]

    def eval_paths(self, store: Store) -> Dict[Tuple[str, str], str]:
        d = os.path.join(store.out, "evals", self.identity)
        mark = eval_marker(self.spec.eval_system_prompt)
        return {(t, s): os.path.join(d, f"trait_eval_{t}_{s}{mark}.csv")
                for t, s in self.eval_pairs()}

    def evals_missing(self, store: Store) -> List[str]:
        """Missing evals: "trait" (reporting split) or "trait:split"."""
        return [t if s == EVAL_SPLIT else f"{t}:{s}"
                for (t, s), p in self.eval_paths(store).items()
                if not os.path.isfile(p)]


def make_plan(store: Store, legacy: Optional[LegacyIndex],
              specs: Optional[Dict[str, RunSpec]] = None,
              only: Optional[List[str]] = None) -> List[PlannedRun]:
    """Plan all runs (or ``only`` these names and their starting models)."""
    specs = specs if specs is not None else runs_mod.all_runs()
    names = topological_order(specs)
    if only:
        keep = set()
        for n in only:
            while n is not None:
                keep.add(n)
                n = specs[n].start
        names = [n for n in names if n in keep]
    scratch = os.path.join(store.out, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    planned: Dict[str, PlannedRun] = {}
    path_to_identity: Dict[str, str] = {}

    def parent_identity(path: str) -> str:
        path = os.path.abspath(path)
        if path in path_to_identity:
            return path_to_identity[path]
        if legacy is not None:
            return legacy.parent_identity(path)
        raise KeyError(f"unknown starting checkpoint {path}")

    for name in names:
        spec = specs[name]
        start_path = None
        if spec.start is not None:
            parent = planned[spec.start]
            start_path = parent.source or store.run_dir(parent.identity)
            path_to_identity[os.path.abspath(start_path)] = parent.identity
            n = spec.start_checkpoint
            if n is not None:
                every = parent.spec.checkpoint_every
                if every is None or n % every:
                    raise ValueError(
                        f"{name}: start_checkpoint {n} needs {spec.start} to "
                        f"checkpoint every k | {n} steps (has {every})")
                # Identity of a mid-training state = parent identity + step.
                path_to_identity[os.path.abspath(os.path.join(
                    start_path, f"checkpoint-{n}"))] = (
                    f"{parent.identity}@checkpoint-{n}")
        cfg = materialize(spec, output_dir="__placeholder__",
                          start_path=start_path)
        ident = identity_of(cfg, os.path.join(scratch, f"{name}.json"),
                            parent_identity=parent_identity)
        key = store_key(ident, spec.checkpoint_every)
        cfg["output_dir"] = store.run_dir(key)
        if store.has(key):
            status, source = "present", store.run_dir(key)
        else:
            source = (legacy.find(ident, spec.checkpoint_every)
                      if legacy is not None else None)
            status = "adopt" if source else "train"
        planned[name] = PlannedRun(name, spec, key, status, source, cfg)
    return [planned[n] for n in names]


def summarize(plan: List[PlannedRun], store: Store) -> str:
    """Human-readable plan table."""
    lines = [f"{'run':42s} {'identity':12s} {'status':8s} evals-missing"]
    for p in plan:
        miss = ",".join(p.evals_missing(store)) or "-"
        lines.append(f"{p.name:42s} {p.identity[:12]} {p.status:8s} {miss}")
    counts = {}
    for p in plan:
        counts[p.status] = counts.get(p.status, 0) + 1
    lines.append("totals: " + ", ".join(f"{k}={v}" for k, v in
                                        sorted(counts.items())))
    return "\n".join(lines)


def write_plan_json(plan: List[PlannedRun], path: str) -> None:
    """Record the plan, MERGING into an existing plan.json.

    A partial plan (``--only``) updates its runs and keeps every other entry:
    products resolve run names through this file, so replacing it with a
    subset made them treat unrelated runs as missing.
    """
    merged = {}
    if os.path.isfile(path):
        with open(path) as f:
            merged = {e["name"]: e for e in json.load(f)}
    for p in plan:
        merged[p.name] = {"name": p.name, "identity": p.identity,
                          "status": p.status, "source": p.source,
                          "start": p.spec.start,
                          "eval_traits": list(p.spec.eval_traits),
                          "eval_splits": list(p.spec.eval_splits),
                          "eval_system_prompt": p.spec.eval_system_prompt}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump([merged[k] for k in sorted(merged)], f, indent=2)
    os.replace(tmp, path)
