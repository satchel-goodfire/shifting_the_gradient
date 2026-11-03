"""Camera-ready judge validation on the arms behind the paper's numbers
(rebuttal point 4: several additional judges; every ordering used in the paper
preserved; coherence compared only within a judge; conclusions survive
coherence filtering).

Inputs live in ``<out>/judge_validation/`` and are produced outside
``assemble`` (they need GPUs or API calls):

- ``arms*/manifest.json``: the paper-pipeline eval CSVs staged as arms by
  ``scripts/stage_rejudge_arms.py`` (arm = run name, ``<run>@eval`` for the
  held-out split);
- ``scores/scores_<judge>[_coherence]_camera_ready_<trait>[_<armset>].csv``:
  ``scripts/rejudge_arms.py`` output (columns arm, row_idx, stored_score,
  score), one file per judge x metric x trait x arm set. ``stored_score`` is the
  production judge's score of the same response.

Only the arm sets in ``PAPER_ARMSETS`` are read (pinned 2026-10-28); other
stagings stay on disk but change no number.

Definitions (fixed before any alternative-judge result was viewed):

- An *ordering* is a pair of arms of the same trait on the same question split
  whose production-judge means differ significantly (Welch's t-test,
  p < ``ALPHA``). Every ordering the paper can rely on is among these.
- Under another judge an ordering is *kept* if that judge's means differ in the
  same direction, and *reversed* if they differ in the opposite direction and
  that difference is itself significant at ``ALPHA``.
- Coherence filtering re-computes each arm's trait mean on responses with
  coherence >= t, either gated by the production judge's coherence or by each
  judge's own (trait and coherence from the same judge: "each judge's own
  standard"); arms left with fewer than ``MIN_FILTERED`` responses are dropped.

Products: ``tab_judge_validation`` (per-judge agreement, orderings, coherence
agreement) and ``tab_judge_cohfilter`` (coherence-filter robustness).
"""

import glob
import itertools
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from ...evaluation.agreement import krippendorff_alpha_interval
from ..macros import Result
from .base import Context, Product

JV_DIR = "judge_validation"
PRODUCTION = "gpt41mini_logprob"
JUDGE_LABELS = {PRODUCTION: "GPT-4.1-mini (production)",
                "llama33_70b_logprob": "Llama-3.3-70B",
                "gpt54mini_logprob5": "GPT-5.4-mini",
                "gpt55_sampled": "GPT-5.5"}
ALPHA = 0.01  # significance level of an ordering (and of a reversal)
COH_THRESHOLDS = (50, 70, 85)
MIN_FILTERED = 20  # responses an arm must keep under a coherence gate
FLAG_COHERENCE = 50  # "flagged incoherent" for per-judge flag rates
MAIN_TRAIT = "evil"

SCORE_RE = re.compile(
    r"^scores_(?P<judge>.+?)(?P<coh>_coherence)?_camera_ready_"
    r"(?P<trait>[a-z]+)(?:_(?P<armset>arms_[A-Za-z0-9_]+))?\.csv$")


def parse_score_file(name: str) -> Optional[dict]:
    """(judge, metric, trait, armset) of a rejudge output filename, or None."""
    m = SCORE_RE.match(name)
    if m is None:
        return None
    return {"judge": m["judge"],
            "metric": "coherence" if m["coh"] else "trait",
            "trait": m["trait"], "armset": m["armset"] or "arms"}


# The arm sets the paper's judge validation covers (2026-10-28:
# "pin to the original set of evals"): the camera-ready staging. The
# coverage extensions staged on 2026-10-22 (arms_gapD1-4, arms_gapE) are
# read by neither the validation nor the shadow results, so later stagings
# do not change the paper's numbers. (The paper's earlier numbers came from
# these three sets as staged before the 2026-10-22 05:08 re-staging of
# arms_rest, which no longer exists on disk.)
PAPER_ARMSETS = ("arms", "arms_extra", "arms_rest")


def load_long(jv_dir: str, armsets: Optional[Iterable[str]] = PAPER_ARMSETS
              ) -> pd.DataFrame:
    """All scores of the arm sets ``armsets`` (None: every set) as long rows:
    trait, arm, row_idx, judge, metric, score.

    The production judge's rows come from ``stored_score``; when several files
    carry the same response they must agree (checked), then one copy is kept.
    """
    keep = None if armsets is None else set(armsets)
    frames: List[pd.DataFrame] = []
    for path in sorted(glob.glob(os.path.join(jv_dir, "scores", "scores_*.csv"))):
        meta = parse_score_file(os.path.basename(path))
        if meta is None:
            # Skipping it would silently shrink the validation's coverage, and
            # a file named scores_*.csv in this directory is meant to be read.
            raise ValueError(
                f"cannot parse score file name {os.path.basename(path)!r}; "
                "expected scores_<judge>[_coherence]_camera_ready_<trait>"
                "[_arms_<set>].csv (rename it, or move it out of scores/)")
        if keep is not None and meta["armset"] not in keep:
            continue  # a staging outside the pinned sets, by design
        if meta["judge"] not in JUDGE_LABELS:
            raise ValueError(f"{path}: unknown judge {meta['judge']!r}")
        df = pd.read_csv(path)
        keys = dict(trait=meta["trait"], metric=meta["metric"])
        frames.append(df[["arm", "row_idx", "score"]]
                      .assign(judge=meta["judge"], **keys))
        frames.append(df[["arm", "row_idx", "stored_score"]]
                      .rename(columns={"stored_score": "score"})
                      .assign(judge=PRODUCTION, **keys))
    if not frames:
        raise FileNotFoundError(f"no rejudge score files under {jv_dir}/scores")
    long = pd.concat(frames, ignore_index=True)
    key = ["trait", "metric", "arm", "row_idx", "judge"]
    spread = long.groupby(key)["score"].agg(lambda s: s.max() - s.min())
    if (spread.fillna(0) > 1e-6).any():
        raise ValueError("production scores disagree between score files")
    return long.drop_duplicates(key).reset_index(drop=True)


SPLITS = ("extract", "eval")
DEFAULT_SPLIT = "extract"


def manifest_paths(jv_dir: str, armsets: Optional[Iterable[str]] = PAPER_ARMSETS
                   ) -> List[str]:
    """The staging manifests of ``armsets`` (None: every ``arms*`` set)."""
    paths = sorted(glob.glob(os.path.join(jv_dir, "arms*", "manifest.json")))
    if armsets is None:
        return paths
    keep = set(armsets)
    return [p for p in paths if os.path.basename(os.path.dirname(p)) in keep]


def load_splits(jv_dir: str, armsets: Optional[Iterable[str]] = PAPER_ARMSETS
                ) -> Dict[str, str]:
    """{arm: question split} from the ``armsets``' ``manifest.json`` (the
    staging record, authoritative over the arm name)."""
    out: Dict[str, str] = {}
    for path in manifest_paths(jv_dir, armsets):
        with open(path) as f:
            for row in json.load(f):
                prev = out.setdefault(row["arm"], row["split"])
                if prev != row["split"]:
                    raise ValueError(f"{path}: arm {row['arm']!r} is staged on "
                                     f"splits {prev!r} and {row['split']!r}")
    return out


def split_of(arm: str, splits: Optional[Dict[str, str]] = None) -> str:
    """Question split of a staged arm: from ``splits`` (``load_splits``) when
    it lists the arm, else parsed from the name with the grammar of
    ``scripts/stage_rejudge_arms.py``: ``<run>[@ckptN][@<split>][@sys<marker>]``
    for evals, ``an.<analysis>.<stem>`` (stem ending ``_eval``: held-out) for
    analysis CSVs."""
    if splits is not None and arm in splits:
        return splits[arm]
    if arm.startswith("an."):
        return "eval" if arm.endswith("_eval") else DEFAULT_SPLIT
    return next((s for s in arm.split("@")[1:] if s in SPLITS), DEFAULT_SPLIT)


def significant(x: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> bool:
    """Welch's t-test at ``alpha``; two constant samples count as different
    when their values differ by at least one point (scipy returns NaN)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return False
    if np.ptp(x) == 0 and np.ptp(y) == 0:  # Welch's statistic is undefined
        return abs(x.mean() - y.mean()) >= 1.0
    return bool(scipy.stats.ttest_ind(x, y, equal_var=False).pvalue < alpha)


def _pearson(x, y) -> float:
    """Pearson r, NaN when either side is constant (r is undefined)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(scipy.stats.pearsonr(x, y)[0])


def orderings(ref: Dict[str, np.ndarray], groups: Iterable[List[str]]
              ) -> List[Tuple[str, str, int]]:
    """Significant orderings under ``ref`` scores: (a, b, sign of a - b),
    over arm pairs within each group (a pair in several groups counts once)."""
    out, seen = [], set()
    for group in groups:
        for a, b in itertools.combinations(sorted(set(group)), 2):
            if (a, b) in seen or a not in ref or b not in ref:
                continue
            seen.add((a, b))
            if significant(ref[a], ref[b]):
                out.append((a, b, int(np.sign(ref[a].mean() - ref[b].mean()))))
    return out


def paper_families(product_runs: Dict[str, Iterable[str]],
                   arms: Iterable[str]) -> List[List[str]]:
    """Comparison families the paper actually uses: for each figure/table
    product, its runs' staged arms on one question split. A pair of arms is a
    *paper* ordering when the arms appear together in some product."""
    arms = set(arms)
    fams = []
    for runs in product_runs.values():
        for split in ("extract", "eval"):
            fam = sorted({r if split == "extract" else f"{r}@{split}"
                          for r in runs} & arms)
            if len(fam) >= 2 and fam not in fams:
                fams.append(fam)
    return fams


def check_orderings(ords: List[Tuple[str, str, int]],
                    other: Dict[str, np.ndarray]) -> dict:
    """How many of ``ords`` another scoring keeps / significantly reverses."""
    kept, reversed_sig, missing, reversals, not_kept = 0, 0, 0, [], []
    for a, b, s in ords:
        if a not in other or b not in other:
            missing += 1
            continue
        d = other[a].mean() - other[b].mean()
        if np.sign(d) == s:
            kept += 1
            continue
        not_kept.append((a, b, s, float(d)))
        if np.sign(d) == -s and significant(other[a], other[b]):
            reversed_sig += 1
            reversals.append((a, b, s, float(d)))
    n = len(ords) - missing
    return {"n": n, "kept": kept, "reversed_sig": reversed_sig,
            "kept_pct": 100.0 * kept / n if n else float("nan"),
            "reversals": reversals, "not_kept": not_kept}


def arm_scores(wide: pd.DataFrame, judge: str,
               gate: Optional[Tuple[str, float]] = None,
               min_n: int = 1) -> Dict[str, np.ndarray]:
    """{arm: trait scores of ``judge``}, optionally only on responses whose
    coherence by judge ``gate[0]`` is >= ``gate[1]``."""
    out = {}
    for arm, g in wide.groupby(level="arm"):
        s = g[("trait", judge)]
        if gate is not None:
            s = s[g[("coherence", gate[0])] >= gate[1]]
        s = s.dropna().to_numpy()
        if len(s) >= min_n:
            out[arm] = s
    return out


def _wide(long: pd.DataFrame, trait: str) -> pd.DataFrame:
    """(arm, row_idx) x (metric, judge) scores of one trait."""
    sub = long[long["trait"] == trait]
    return sub.pivot_table(index=["arm", "row_idx"], columns=["metric", "judge"],
                           values="score", aggfunc="first", dropna=False)


def _groups(arms: Iterable[str], splits: Optional[Dict[str, str]] = None
            ) -> List[List[str]]:
    by: Dict[str, List[str]] = {}
    for a in arms:
        by.setdefault(split_of(a, splits), []).append(a)
    return list(by.values())


def _fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of another judge's arm means ``y`` with production's ``x``
    (NaN where fewer than two arms make a statistic undefined)."""
    nan = float("nan")
    off = y - x  # per-arm offset of this judge vs production
    rec = {"offset_median": float(np.median(off)) if len(off) else nan,
           "offset_max": float(off.max()) if len(off) else nan,
           "offset_min": float(off.min()) if len(off) else nan,
           "pearson": _pearson(x, y) if len(x) >= 2 else nan,
           "spearman": nan, "slope": nan, "intercept": nan}
    if len(x) >= 2 and np.ptp(x) > 0:
        rec["slope"], rec["intercept"] = map(float, np.polyfit(x, y, 1))
        if np.ptp(y) > 0:
            rec["spearman"] = float(scipy.stats.spearmanr(x, y)[0])
    return rec


def analyze(long: pd.DataFrame, trait: str = MAIN_TRAIT,
            families: Optional[List[List[str]]] = None,
            splits: Optional[Dict[str, str]] = None) -> dict:
    """Everything the products report, for one trait.

    Orderings are checked over all same-split arm pairs (a strict superset;
    each arm's split from ``splits``, ``load_splits``, else its name) and,
    when ``families`` (``paper_families``) is given, over the pairs the
    paper's figures and tables actually compare (``paper`` sub-records).
    """
    wide = _wide(long, trait)
    judges = [j for j in JUDGE_LABELS if ("trait", j) in wide.columns]
    coh_judges = [j for j in JUDGE_LABELS if ("coherence", j) in wide.columns]
    ref = arm_scores(wide, PRODUCTION)
    groups = _groups(ref, splits)
    ords = orderings(ref, groups)
    paper = orderings(ref, families) if families is not None else None
    prod_means = pd.Series({a: s.mean() for a, s in ref.items()})
    out = {"trait": trait, "n_arms": len(ref), "n_orderings": len(ords),
           "n_responses": int(wide[("trait", PRODUCTION)].notna().sum()),
           "judges": {}, "cohfilter": {}}
    if paper is not None:
        out["n_paper_orderings"] = len(paper)
    for j in judges:
        sc = arm_scores(wide, j)
        means = pd.Series({a: s.mean() for a, s in sc.items()})
        common = prod_means.index.intersection(means.index)
        x, y = prod_means[common].to_numpy(), means[common].to_numpy()
        rec = {"n_responses": int(wide[("trait", j)].notna().sum()),
               **_fit_stats(x, y), **check_orderings(ords, sc)}
        if paper is not None:
            rec["paper"] = check_orderings(paper, sc)
        if ("coherence", j) in wide.columns:
            c = wide[("coherence", j)].dropna()
            rec["coh_flag_pct"] = float(100 * (c < FLAG_COHERENCE).mean())
            if j != PRODUCTION:
                both = wide[[("coherence", PRODUCTION), ("coherence", j)]].dropna()
                cm = both.groupby(level="arm").mean()
                rec["coh_pearson"] = _pearson(cm.iloc[:, 0], cm.iloc[:, 1])
        out["judges"][j] = rec
    # Response-level agreement on the trait, over responses every judge scored.
    cols = [("trait", j) for j in judges]
    full = wide[cols].dropna()
    if len(judges) >= 2 and len(full):
        out["alpha"] = krippendorff_alpha_interval(full.to_numpy())
        out["alpha_n"] = len(full)
    # Coherence filtering.
    for t in COH_THRESHOLDS:
        rec = {}
        if ("coherence", PRODUCTION) in wide.columns:
            filt = arm_scores(wide, PRODUCTION, gate=(PRODUCTION, t),
                              min_n=MIN_FILTERED)
            deltas = {a: filt[a].mean() - ref[a].mean() for a in filt}
            worst = (max(deltas, key=lambda a: abs(deltas[a]))
                     if deltas else None)  # None: no arm keeps MIN_FILTERED
            fig2 = [abs(d) for a, d in deltas.items() if a.startswith("grid.")]
            rec["prod"] = {**check_orderings(ords, filt),
                           "max_abs_delta": (float(abs(deltas[worst]))
                                             if worst is not None else None),
                           "max_abs_delta_arm": worst,
                           "max_abs_delta_grid": max(fig2) if fig2 else None,
                           "n_arms": len(filt)}
            if paper is not None:
                rec["prod_paper"] = check_orderings(paper, filt)
        own, own_paper = {}, {}
        for j in coh_judges:
            if ("trait", j) not in wide.columns:
                continue
            sc = arm_scores(wide, j, gate=(j, t), min_n=MIN_FILTERED)
            own[j] = check_orderings(ords, sc)
            if paper is not None:
                own_paper[j] = check_orderings(paper, sc)
        rec["own"] = own
        rec["own_paper"] = own_paper
        out["cohfilter"][t] = rec
    return out


def cohfilter_macros(p: str, t: int, rec: dict, src: str) -> Dict[str, Result]:
    """Macros of one coherence threshold ``t`` (``analyze``'s
    ``cohfilter[t]``) under prefix ``p``. Production-gated statistics need
    the production judge's coherence (``prod``); own-gated ones need another
    judge's (``own``); either may be absent."""
    q = f"{p}.cohfilter.t{t}"
    res: Dict[str, Result] = {}
    prod = rec.get("prod")
    if prod is not None:
        if prod["max_abs_delta"] is not None:
            res[f"{q}.max_abs_delta"] = Result(prod["max_abs_delta"],
                                               decimals=1, source=src)
        if prod["max_abs_delta_grid"] is not None:
            res[f"{q}.max_abs_delta_grid"] = Result(
                prod["max_abs_delta_grid"], decimals=1, source=src)
        res[f"{q}.prod.kept_pct"] = Result(prod["kept_pct"], decimals=1,
                                           source=src)
        res[f"{q}.prod.reversed_sig"] = Result(prod["reversed_sig"],
                                               decimals=0, source=src)
    if "prod_paper" in rec:
        res[f"{q}.prod_paper.kept_pct"] = Result(
            rec["prod_paper"]["kept_pct"], decimals=1, source=src)
        res[f"{q}.prod_paper.reversed_sig"] = Result(
            rec["prod_paper"]["reversed_sig"], decimals=0, source=src)
    for part in ("own", "own_paper"):
        if rec.get(part):
            recs = rec[part].values()
            res[f"{q}.{part}.reversed_sig"] = Result(
                sum(r["reversed_sig"] for r in recs), decimals=0, source=src)
            res[f"{q}.{part}.min_kept_pct"] = Result(
                min(r["kept_pct"] for r in recs), decimals=1, source=src)
    return res


def _fmt(v, d=2):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.{d}f}"


class JudgeValidation(Product):
    """Per-judge agreement and ordering preservation on the paper's arms."""

    name = "tab_judge_validation"

    def build(self, ctx: Context) -> Dict[str, Result]:
        from . import discover  # here: products/__init__ imports this module

        jv = os.path.join(ctx.out, JV_DIR)
        long = load_long(jv)
        splits = load_splits(jv)
        _, products = discover()
        families = paper_families({n: p.runs for n, p in products.items()},
                                  set(long["arm"]))
        res: Dict[str, Result] = {}
        reports = {}
        for trait in sorted(long["trait"].unique()):
            rep = reports[trait] = analyze(long, trait, families, splits)
            p = f"judgeval.{trait}"
            src = f"{JV_DIR}/scores ({trait})"
            for k in ("n_arms", "n_orderings", "n_paper_orderings",
                      "n_responses"):
                res[f"{p}.{k}"] = Result(rep[k], decimals=0, source=src)
            if "alpha" in rep:
                res[f"{p}.alpha"] = Result(rep["alpha"], decimals=3, source=src)
            for j, r in rep["judges"].items():
                q = f"{p}.{j}"
                for k, d in (("pearson", 3), ("spearman", 3), ("slope", 2),
                             ("intercept", 1), ("kept_pct", 1),
                             ("offset_median", 1), ("offset_max", 1),
                             ("offset_min", 1),
                             ("coh_flag_pct", 2), ("coh_pearson", 3)):
                    if k in r and np.isfinite(r[k]):
                        res[f"{q}.{k}"] = Result(r[k], decimals=d, source=src)
                for k in ("kept", "reversed_sig", "n", "n_responses"):
                    res[f"{q}.{k}"] = Result(r[k], decimals=0, source=src)
                for k in ("kept", "reversed_sig", "n"):
                    res[f"{q}.paper.{k}"] = Result(r["paper"][k], decimals=0,
                                                   source=src)
                if np.isfinite(r["paper"]["kept_pct"]):
                    res[f"{q}.paper.kept_pct"] = Result(
                        r["paper"]["kept_pct"], decimals=1, source=src)
            for t, rec in rep["cohfilter"].items():
                res.update(cohfilter_macros(p, t, rec, src))
        # Paper numbers re-derived under each alternative judge
        # (scripts/judge_shadow_results.py), when that has been run.
        for path in sorted(glob.glob(os.path.join(jv, "shadow", "*",
                                                  "summary.json"))):
            with open(path) as f:
                s = json.load(f)
            if "n_claims" not in s or "judge_dependent_products" not in s:
                raise ValueError(
                    f"{path} predates the claim classification (random-score "
                    "reference); re-run scripts/judge_shadow_results.py")
            q = f"judgeval.shadow.{s['judge']}"
            src = os.path.relpath(path, ctx.out)
            # products with at least one number re-derived from the judge
            res[f"{q}.n_products"] = Result(len(s["judge_dependent_products"]),
                                            decimals=0, source=src)
            res[f"{q}.n_built"] = Result(len(s["built"]), decimals=0,
                                         source=src)
            res[f"{q}.n_keys"] = Result(s["n_keys"], decimals=0, source=src)
            res[f"{q}.n_claims"] = Result(s["n_claims"], decimals=0,
                                          source=src)
            res[f"{q}.preserved"] = Result(s["preserved"], decimals=0,
                                           source=src)
            for kind, rec in s["by_kind"].items():
                k = f"{q}.{kind.replace('-', '_')}"
                res[f"{k}.n"] = Result(rec["n"], decimals=0, source=src)
                if "preserved" in rec:
                    res[f"{k}.preserved"] = Result(rec["preserved"], decimals=0,
                                                   source=src)
                if "median_abs_change" in rec:
                    res[f"{k}.median_abs_change"] = Result(
                        rec["median_abs_change"], decimals=1, source=src)
        os.makedirs(os.path.join(jv, "report"), exist_ok=True)
        with open(os.path.join(jv, "report", "summary.json"), "w") as f:
            json.dump(reports, f, indent=1, default=str)
        # Totals over traits (every judged response in the store).
        src = f"{JV_DIR}/scores (all traits)"
        res["judgeval.all.n_traits"] = Result(len(reports), 0, source=src)
        for k in ("n_arms", "n_responses", "n_paper_orderings"):
            res[f"judgeval.all.{k}"] = Result(
                sum(r.get(k, 0) for r in reports.values()), 0, source=src)
        for j in JUDGE_LABELS:
            if j == PRODUCTION:
                continue
            recs = [r["judges"][j] for r in reports.values() if j in r["judges"]]
            if not recs:
                continue
            for k in ("kept", "n", "reversed_sig"):
                res[f"judgeval.all.{j}.paper.{k}"] = Result(
                    sum(r["paper"][k] for r in recs), 0, source=src)
            res[f"judgeval.all.{j}.n_responses"] = Result(
                sum(r["n_responses"] for r in recs), 0, source=src)
        self._tables(ctx, reports)
        return res

    @staticmethod
    def _tables(ctx: Context, reports: Dict[str, dict]) -> None:
        """One row per (trait, alternative judge); then the coherence-filter
        table for the main trait."""
        rep = reports.get(MAIN_TRAIT)
        if rep is None:
            return
        alt = [j for j in JUDGE_LABELS if j != PRODUCTION]
        multi = sum(any(j in r["judges"] for r in reports.values())
                    for j in alt) > 1
        lines = ["% Generated by grad_shift.paper (tab_judge_validation).",
                 "\\begin{tabular}{l" + ("l" if multi else "") +
                 " rr rr rr rr r}", "\\toprule",
                 " & " + ("& " if multi else "") +
                 "& & \\multicolumn{4}{c}{arm means vs.\\ production} & "
                 "\\multicolumn{2}{c}{orderings kept} & reversed \\\\",
                 "trait & " + ("judge & " if multi else "") +
                 "arms & responses & $r$ & $\\rho$ & slope & offset & "
                 "paper & all & (all) \\\\", "\\midrule"]
        order = [MAIN_TRAIT] + sorted(t for t in reports if t != MAIN_TRAIT)
        for t in order:
            for j in alt:
                r = reports[t]["judges"].get(j)
                if r is None:
                    continue
                pp = r["paper"]
                lines.append(
                    f"{t} & " + (f"{JUDGE_LABELS[j]} & " if multi else "") +
                    f"{reports[t]['n_arms']} & {r['n_responses']:,} & "
                    f"{_fmt(r['pearson'], 3)} & {_fmt(r['spearman'], 3)} & "
                    f"{_fmt(r['slope'])} & {_fmt(r['intercept'], 1)} & "
                    f"{pp['kept']}/{pp['n']} & {r['kept']}/{r['n']} & "
                    f"{r['reversed_sig']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_judge_validation.tex"),
                  "w") as f:
            f.write("\n".join(lines))
        lines = ["% Generated by grad_shift.paper (tab_judge_validation).",
                 "\\begin{tabular}{r rr rr rr}", "\\toprule",
                 " & \\multicolumn{4}{c}{production coherence gate} & "
                 "\\multicolumn{2}{c}{each judge's own gate} \\\\",
                 " & \\multicolumn{2}{c}{max $|\\Delta|$ (points)} & "
                 "\\multicolumn{2}{c}{orderings kept} & paper orderings & \\\\",
                 "coherence $\\geq$ & Fig.~2 arms & all arms & paper & all & "
                 "min.\\ kept & reversed \\\\", "\\midrule"]
        for t, rec in rep["cohfilter"].items():
            pr, pp = rec.get("prod", {}), rec.get("prod_paper", {})
            own = rec["own_paper"]
            lines.append(
                f"{t} & {_fmt(pr.get('max_abs_delta_grid'), 1)} & "
                f"{_fmt(pr.get('max_abs_delta'), 1)} & "
                f"{pp.get('kept', '--')}/{pp.get('n', '--')} & "
                f"{pr.get('kept', '--')}/{pr.get('n', '--')} & "
                + (f"{min(r['kept_pct'] for r in own.values()):.1f}\\% & "
                   f"{sum(r['reversed_sig'] for r in own.values())}"
                   if own else "-- & --") + " \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_judge_cohfilter.tex"),
                  "w") as f:
            f.write("\n".join(lines))
