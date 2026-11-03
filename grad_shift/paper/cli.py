"""``sg-paper``: reproduce the paper's runs, figures, tables, and numbers.

Subcommands:
    plan     resolve every run and show present / adopt / train status
    submit   adopt matching legacy runs, then submit SLURM jobs for the rest
             (training + judging), chaining starting-model dependencies
    status   same table as plan, from the store as it is now
    analyze  submit GPU analyses whose results are missing or stale (made
             from other runs / input files / settings than the current ones)
    adopt-analyses
             one-off for output folders from before analysis fingerprints:
             record results as current, --exclude the ones whose inputs
             changed since they ran
    assemble build every ready product: figures/, tables/, results.tex

Examples:
    uv run sg-paper plan --out /data/paper_out --reuse /data/grad_shift-restored
    uv run sg-paper submit --out /data/paper_out --reuse /data/grad_shift-restored
    uv run sg-paper submit --out DIR --only grid.medium.heavy.pps
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional

from .datasets import ensure_generated
from .plan import PlannedRun, make_plan, summarize, write_plan_json
from .spec import REPO_ROOT
from .store import LegacyIndex, Store

SBATCH = os.path.join("scripts", "slurm", "paper_run.sbatch")


def ensure_generated_data() -> None:
    """Build derived datasets the run specs reference (idempotent)."""
    ensure_generated(REPO_ROOT)


ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "CONFIGURING",
                 "COMPLETING", "SUSPENDED", "REQUEUE_HOLD", "RESIZING"}


def active_job(marker: str) -> Optional[str]:
    """Job id recorded in ``marker`` if that SLURM job is still active."""
    if not os.path.isfile(marker):
        return None
    with open(marker) as f:
        job = f.read().strip()
    out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%T"],
                         capture_output=True, text=True)
    state = out.stdout.strip()
    return job if state in ACTIVE_STATES else None


def _sbatch(args: List[str]) -> str:
    out = subprocess.run(["sbatch", "--parsable", *args], cwd=REPO_ROOT,
                         check=True, capture_output=True, text=True)
    return out.stdout.strip().split(";")[0]


def submit(plan: List[PlannedRun], store: Store, dry_run: bool = False,
           extra_sbatch: Optional[List[str]] = None) -> Dict[str, str]:
    """Adopt and submit; returns run name -> job id (or 'adopted'/'done')."""
    jobs: Dict[str, str] = {}
    logs = os.path.join(store.out, "logs")
    os.makedirs(logs, exist_ok=True)
    for p in plan:
        record = {"name": p.name, "spec": p.spec.__dict__,
                  "identity": p.identity}
        if p.status == "adopt" and not dry_run:
            store.adopt(p.identity, p.source)
            store.write_spec(p.identity, record)
        missing = p.evals_missing(store)
        if p.status in ("present", "adopt") and not missing:
            jobs[p.name] = "done"
            continue
        run_dir = store.run_dir(p.identity)
        eval_dir = os.path.dirname(next(iter(p.eval_paths(store).values())))
        # One job per run at a time: a still-active earlier submission owns
        # this run (two jobs training into one output_dir would race).
        marker = os.path.join(eval_dir, ".job")
        running = active_job(marker)
        if running:
            jobs[p.name] = running
            continue
        if not dry_run:
            if p.status == "train":
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "train_config.json"), "w") as f:
                    json.dump(p.config, f, indent=2)
                store.write_spec(p.identity, record)
            # Traits list lives in the pipeline-owned eval dir (adopted run
            # dirs are symlinks into legacy trees and are never written to).
            # One "<trait> <split> [<eval system prompt>]" line per eval.
            os.makedirs(eval_dir, exist_ok=True)
            sysp = p.spec.eval_system_prompt
            with open(os.path.join(eval_dir, "eval_traits.txt"), "w") as f:
                f.write("".join(f"{t} {s}" + (f" {sysp}" if sysp else "")
                                + "\n" for t, s in p.eval_pairs()))
        dep = []
        if p.spec.start is not None:
            parent_job = jobs.get(p.spec.start)
            if parent_job and parent_job not in ("done", "dry"):
                dep = [f"--dependency=afterok:{parent_job}"]
        args = [*dep, f"--job-name=sg_{p.name}",
                f"--output={logs}/{p.name}-%j.out", *(extra_sbatch or []),
                SBATCH, run_dir, eval_dir]
        if dry_run:
            jobs[p.name] = "dry"
        else:
            jobs[p.name] = _sbatch(args)
            with open(marker, "w") as f:
                f.write(jobs[p.name] + "\n")
    return jobs


def submit_analyses(out: str, names: Optional[List[str]] = None,
                    dry_run: bool = False) -> Dict[str, str]:
    """Submit Analysis jobs whose result is missing, stale or unverified.

    Each waits (afterok) on still-active jobs of the runs it needs. The
    returned status is "done" (current result), "dry", or a job id; stale
    and unverified results are reported on stderr before resubmission.
    """
    from .assemble import load_context
    from .products import discover
    from .products.base import analysis_status

    analyses, _ = discover()
    ctx = load_context(out)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)
    jobs: Dict[str, str] = {}
    for name in sorted(names or analyses):
        a = analyses[name]
        adir = ctx.analysis_dir(name)
        status = analysis_status(a, ctx)
        if status == "current":
            jobs[name] = "done"
            continue
        marker = os.path.join(adir, ".job")
        running = active_job(marker)
        if running:
            jobs[name] = running
            continue
        if status != "missing":
            print(f"[{status}] {name}: recomputing", file=sys.stderr)
        deps = []
        for run in a.runs:
            j = active_job(os.path.join(out, "evals", ctx.run_keys[run], ".job"))
            if j:
                deps.append(j)
        args = ([f"--dependency=afterok:{':'.join(deps)}"] if deps else [])
        args += [f"--job-name=sga_{name}", f"--gres=gpu:{a.gpus}",
                 f"--time={a.time}", f"--output={logs}/analysis-{name}-%j.out",
                 os.path.join("scripts", "slurm", "paper_analysis.sbatch"),
                 name, out]
        if dry_run:
            jobs[name] = "dry"
            continue
        jobs[name] = _sbatch(args)
        with open(marker, "w") as f:
            f.write(jobs[name] + "\n")
    return jobs


def adopt_analyses(out: str, exclude: List[str],
                   refingerprint: Optional[List[str]] = None
                   ) -> Dict[str, str]:
    """Record the current fingerprint for unverified results (made before
    fingerprints existed), except ``exclude``; and for the STALE results
    named in ``refingerprint``.

    Adopting asserts that the result WAS computed from the current inputs;
    exclude every analysis whose inputs have changed since it ran, so it is
    recomputed instead. ``refingerprint`` is for a changed fingerprint
    definition (e.g. a setting newly declared in ``params``) whose values
    the stale result was in fact computed with. Returns {analysis:
    "adopted" | "excluded" | "refingerprinted"}.
    """
    from .assemble import load_context
    from .products import discover
    from .products.base import INPUTS_FILE, analysis_status

    analyses, _ = discover()
    refingerprint = refingerprint or []
    unknown = sorted((set(exclude) | set(refingerprint)) - set(analyses))
    if unknown:
        raise SystemExit(f"unknown analyses: {unknown}")
    ctx = load_context(out)
    done: Dict[str, str] = {}
    for name in sorted(analyses):
        a = analyses[name]
        status = analysis_status(a, ctx)
        if status == "stale" and name in refingerprint:
            done[name] = "refingerprinted"
        elif status != "unverified":
            continue
        elif name in exclude:
            done[name] = "excluded"
            continue
        else:
            done[name] = "adopted"
        path = os.path.join(ctx.analysis_dir(name), INPUTS_FILE)
        with open(path + ".tmp", "w") as f:
            json.dump(a.fingerprint(ctx), f, indent=2)
        os.replace(path + ".tmp", path)
    return done


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="sg-paper", description=__doc__.split(
        "\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze", help="submit GPU analyses")
    s.add_argument("--out", required=True)
    s.add_argument("--only", nargs="+", default=None)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("adopt-analyses",
                       help="fingerprint results made before fingerprints "
                            "existed (asserts they are current)")
    s.add_argument("--out", required=True)
    s.add_argument("--exclude", nargs="+", default=[],
                   help="analyses whose inputs changed since they ran "
                        "(left unverified, so `analyze` recomputes them)")
    s.add_argument("--refingerprint", nargs="+", default=[],
                   help="stale analyses whose fingerprint DEFINITION changed "
                        "but whose computation used the current values "
                        "(re-recorded as current)")
    s = sub.add_parser("assemble", help="build figures, tables, results.tex")
    s.add_argument("--out", required=True)
    for name in ("plan", "submit", "status"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, help="pipeline output folder")
        s.add_argument("--reuse", action="append", default=[],
                       help="legacy root to adopt matching runs from "
                            "(repeatable)")
        s.add_argument("--only", nargs="+", default=None,
                       help="restrict to these runs (+ their starting models)")
        if name == "submit":
            s.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    ensure_generated_data()  # identities and fingerprints hash these files
    if args.cmd == "analyze":
        jobs = submit_analyses(args.out, args.only, args.dry_run)
        for k, v in jobs.items():
            print(f"{k:40s} {v}")
        return
    if args.cmd == "adopt-analyses":
        for k, v in adopt_analyses(args.out, args.exclude,
                                   args.refingerprint).items():
            print(f"{k:40s} {v}")
        return
    if args.cmd == "assemble":
        from .assemble import assemble
        results, skipped = assemble(args.out)
        print(f"wrote results.tex with {len(results)} numbers")
        for k, why in skipped.items():
            print(f"[skipped] {k}: {why.splitlines()[0]}")
        return
    store = Store(args.out)
    legacy = LegacyIndex(args.reuse) if args.reuse else None
    if legacy is not None and legacy.skipped:
        for d, why in legacy.skipped:
            print(f"[legacy skip] {d}: {why}", file=sys.stderr)
    plan = make_plan(store, legacy, only=args.only)
    write_plan_json(plan, os.path.join(store.out, "plan.json"))
    print(summarize(plan, store))
    if args.cmd == "submit":
        jobs = submit(plan, store, dry_run=args.dry_run)
        n = sum(1 for j in jobs.values() if j not in ("done", "dry"))
        print(f"submitted {n} jobs" + (" (dry run)" if args.dry_run else ""))
        # Merge (a --only submit must not drop other runs' job records).
        path = os.path.join(store.out, "jobs.json")
        merged = {}
        if os.path.isfile(path):
            with open(path) as f:
                merged = json.load(f)
        merged.update(jobs)
        with open(path + ".tmp", "w") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    main()
