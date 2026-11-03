"""Run one Analysis (the body of an `sg-paper analyze` SLURM job).

    python -m grad_shift.paper.run_analysis <name> --out DIR

Writes ``<out>/analyses/<name>/result.json`` atomically (via a temp file), so
a preempted job never leaves a half-written result that looks complete, then
the fingerprint it was computed from (``inputs.json``; written second, so an
interruption between the two leaves a result that reads as unverified and is
recomputed, never a stale result that reads as current).

Files an analysis caches in its directory (judged CSVs, ...) are reused only
under the same fingerprint: the fingerprint is recorded as pending when the
job starts, and a directory whose recorded fingerprint differs (or that has
files but no record at all) is first moved to ``stale-<time>/``.
"""

import argparse
import json
import os
import time

from .assemble import load_context
from .datasets import ensure_generated
from .products import discover
from .products.base import (INPUTS_FILE, PENDING_FILE, archive_contents,
                            read_json)
from .spec import REPO_ROOT


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(tmp, path)


def prepare_dir(d: str, fp: dict) -> None:
    """Archive ``d``'s contents unless they were made under ``fp``, then
    record ``fp`` as the pending fingerprint."""
    rec = read_json(os.path.join(d, INPUTS_FILE))
    if rec is None:
        rec = read_json(os.path.join(d, PENDING_FILE))
    if rec != fp:
        moved = archive_contents(d, time.strftime("%Y%m%dT%H%M%SZ",
                                                  time.gmtime()))
        if moved:
            print(f"inputs changed: moved previous contents to {moved}")
    _write_json(os.path.join(d, PENDING_FILE), fp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ensure_generated(REPO_ROOT)  # the fingerprint hashes generated data
    analyses, _ = discover()
    ctx = load_context(args.out)
    a = analyses[args.name]
    d = ctx.analysis_dir(args.name)
    fp = a.fingerprint(ctx)
    prepare_dir(d, fp)
    result = a.run(ctx)
    _write_json(os.path.join(d, "result.json"), result)
    _write_json(os.path.join(d, INPUTS_FILE), fp)
    os.remove(os.path.join(d, PENDING_FILE))
    print(f"wrote {d}/result.json")


if __name__ == "__main__":
    main()
