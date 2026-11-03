#!/usr/bin/env python3
"""Stage paper-pipeline eval CSVs as arm directories for ``rejudge_arms.py``.

``rejudge_arms.py`` reads ``<arm_dir>/trait_eval_<trait>.csv``; the paper
pipeline stores evals as ``<out>/evals/<identity>/trait_eval_<trait>_<split>.csv``.
This links each requested run's CSVs into ``<dest>/<arm>/trait_eval_<trait>.csv``
(arm = run name, or ``<run>@<split>`` for a split other than ``extract``), so the
multi-judge validation re-scores exactly the responses behind the paper's numbers
under readable arm names. Nothing is copied or modified; a ``manifest.json`` records
each link's run, identity, split and source SHA-256.

Example (from the repo root; ``$OUT`` is the pipeline output folder):
    uv run --frozen python scripts/stage_rejudge_arms.py \
        --out $OUT \
        --runs-file maintext_runs.txt --splits extract eval \
        --dest $OUT/judge_validation/arms
"""

import argparse
import glob
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_SPLIT = "extract"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_name(run: str, split: str) -> str:
    """Arm directory name for a run's eval on one question split."""
    return run if split == DEFAULT_SPLIT else f"{run}@{split}"


def stage(out: str, runs: Iterable[str], splits: List[str], dest: str) -> List[Dict]:
    """Link the eval CSVs of ``runs`` into ``dest``; return the manifest rows.

    A run is staged for every trait it was judged on (``plan.json``) and every
    split in ``splits`` that the plan lists for it or whose CSV exists anyway
    (e.g. the caps-check held-out evals, written by one-off jobs before
    ``eval_splits`` existed; ``listed`` in the manifest says which). Runs
    evaluated under a non-default system prompt are rejected: their CSVs carry
    a filename marker and a different context.

    Raises:
        KeyError: a run is not in ``plan.json``.
        FileNotFoundError: a CSV for a split the plan lists is missing.
    """
    with open(os.path.join(out, "plan.json")) as f:
        plan = {r["name"]: r for r in json.load(f)}
    rows: List[Dict] = []
    for run in runs:
        if run not in plan:
            raise KeyError(f"run {run!r} is not in {out}/plan.json")
        rec = plan[run]
        if rec.get("eval_system_prompt") is not None:
            raise ValueError(f"{run}: non-default eval system prompt "
                             f"{rec['eval_system_prompt']!r}; not staged")
        for split in splits:
            listed = split in rec.get("eval_splits", [DEFAULT_SPLIT])
            for trait in rec.get("eval_traits", ["evil"]):
                src = os.path.join(out, "evals", rec["identity"],
                                   f"trait_eval_{trait}_{split}.csv")
                if not os.path.isfile(src):
                    if listed:
                        raise FileNotFoundError(src)
                    continue
                arm_dir = os.path.join(dest, arm_name(run, split))
                os.makedirs(arm_dir, exist_ok=True)
                link = os.path.join(arm_dir, f"trait_eval_{trait}.csv")
                if os.path.lexists(link):
                    if os.path.realpath(link) != os.path.realpath(src):
                        raise FileExistsError(f"{link} points elsewhere")
                else:
                    os.symlink(src, link)
                rows.append({"arm": arm_name(run, split), "run": run,
                             "identity": rec["identity"], "split": split,
                             "trait": trait, "listed": listed, "source": src,
                             "sha256": _sha256(src)})
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


STANDARD_COLUMNS = {"question", "prompt", "answer", "question_id", "coherence"}
_EVAL_NAME = re.compile(r"^trait_eval_(?P<trait>[a-z]+)_(?P<split>extract|eval)"
                        r"(?:_sys(?P<sys>[a-z_]+))?\.csv$")


def judged_trait(path: str) -> Optional[str]:
    """The judged trait column of a CSV of judged responses, else None.

    A judged CSV has ``answer`` and ``coherence`` plus exactly one trait
    column (whatever the file is called)."""
    with open(path) as f:
        cols = f.readline().strip().split(",")
    if "answer" not in cols or "coherence" not in cols:
        return None
    traits = [c for c in cols if c not in STANDARD_COLUMNS]
    return traits[0] if len(traits) == 1 else None


# Runs under this prefix belong to other work in the research pipeline and
# were never staged for the judge validation; none is declared in this
# release, so the default only matters for a research-pipeline store.
def discover_judged(out: str, skip_sources: set,
                    exclude_prefixes: Tuple[str, ...] = ("vecfit.",)
                    ) -> Tuple[List[Dict], Dict[str, int]]:
    """Every judged CSV in the store not in ``skip_sources`` (realpaths):
    final and checkpoint evals of plan runs (``evals/<identity>[-ckptN]/``)
    and judged CSVs inside analyses (``stale-*`` archives excluded).

    Arm names: ``<run>[@ckptN][@<split>][@sys<marker>]`` for evals and
    ``an.<analysis>.<file stem>`` for analyses. Returns (rows, counts of what
    was skipped and why).
    """
    with open(os.path.join(out, "plan.json")) as f:
        by_id = {r["identity"]: r["name"] for r in json.load(f)}
    rows, skipped = [], {"not_in_plan": 0, "excluded_prefix": 0,
                         "already_staged": 0, "not_judged": 0}

    def add(arm, run, identity, split, trait, src):
        if os.path.realpath(src) in skip_sources:
            skipped["already_staged"] += 1
            return
        rows.append({"arm": arm, "run": run, "identity": identity,
                     "split": split, "trait": trait, "source": src})

    evals = os.path.join(out, "evals")
    for d in sorted(os.listdir(evals)):
        ident, _, ckpt = d.partition("-ckpt")
        name = by_id.get(ident)
        if name is None:
            skipped["not_in_plan"] += 1
            continue
        if name.startswith(exclude_prefixes):
            skipped["excluded_prefix"] += 1
            continue
        for fn in sorted(os.listdir(os.path.join(evals, d))):
            m = _EVAL_NAME.match(fn)
            src = os.path.join(evals, d, fn)
            if m is None or judged_trait(src) != m["trait"]:
                if fn.endswith(".csv"):
                    skipped["not_judged"] += 1
                continue
            arm = name + (f"@ckpt{ckpt}" if ckpt else "")
            arm += "" if m["split"] == DEFAULT_SPLIT else f"@{m['split']}"
            arm += f"@sys{m['sys']}" if m["sys"] else ""
            add(arm, name, d, m["split"], m["trait"], src)
    analyses = os.path.join(out, "analyses")
    for a in sorted(os.listdir(analyses)):
        for root, dirs, files in os.walk(os.path.join(analyses, a)):
            dirs[:] = sorted(x for x in dirs if not x.startswith("stale-"))
            for fn in sorted(files):
                src = os.path.join(root, fn)
                if not fn.endswith(".csv"):
                    continue
                trait = judged_trait(src)
                if trait is None:
                    skipped["not_judged"] += 1
                    continue
                rel = os.path.relpath(src, os.path.join(analyses, a))
                stem = rel[:-len(".csv")].replace(os.sep, ".")
                split = "eval" if stem.endswith("_eval") else DEFAULT_SPLIT
                add(f"an.{a}.{stem}", f"analysis:{a}", None, split, trait, src)
    return rows, skipped


def stage_rows(rows: List[Dict], dest: str) -> List[Dict]:
    """Link ``discover_judged`` rows into ``dest`` (+ manifest.json)."""
    out = []
    for r in rows:
        arm_dir = os.path.join(dest, r["arm"])
        os.makedirs(arm_dir, exist_ok=True)
        link = os.path.join(arm_dir, f"trait_eval_{r['trait']}.csv")
        if os.path.lexists(link):
            if os.path.realpath(link) != os.path.realpath(r["source"]):
                raise FileExistsError(f"{link} points elsewhere")
        else:
            os.symlink(r["source"], link)
        out.append({**r, "listed": True, "sha256": _sha256(r["source"])})
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="paper_out directory")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--runs-file", help="file with one run name per line")
    ap.add_argument("--splits", nargs="+", default=[DEFAULT_SPLIT])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--all-judged", action="store_true",
                    help="stage every judged CSV in the store (final and "
                         "checkpoint evals of plan runs, analysis CSVs) not "
                         "already in a manifest.json next to --dest")
    args = ap.parse_args()
    if args.all_judged:
        staged = set()
        parent = os.path.dirname(os.path.abspath(args.dest))
        for mf in glob.glob(os.path.join(parent, "*", "manifest.json")):
            if os.path.dirname(mf) == os.path.abspath(args.dest):
                continue
            with open(mf) as f:
                staged |= {os.path.realpath(r["source"]) for r in json.load(f)}
        rows, skipped = discover_judged(args.out, staged)
        os.makedirs(args.dest, exist_ok=True)
        stage_rows(rows, args.dest)
        traits = sorted({r["trait"] for r in rows})
        print(f"staged {len(rows)} CSVs ({len(staged)} already staged "
              f"elsewhere); traits {traits}; skipped {skipped}")
        return
    runs = list(args.runs)
    if args.runs_file:
        with open(args.runs_file) as f:
            runs += [ln.strip() for ln in f if ln.strip()]
    if not runs:
        raise SystemExit("no runs given")
    os.makedirs(args.dest, exist_ok=True)
    rows = stage(args.out, dict.fromkeys(runs), args.splits, args.dest)
    n_arms = len({r["arm"] for r in rows})
    print(f"staged {len(rows)} CSVs as {n_arms} arms under {args.dest}")


if __name__ == "__main__":
    main()
