"""IP prompt ablations (Appx. "Single prompts vs. prompt banks"): regimes, runs,
and the pure statistics shared by ``ipablate_gradient`` and ``ipablate_table``.

No Product/Analysis classes live here (``discover`` finds nothing); the
training runs the appendix needs beyond ``runs.py`` are declared below as
``EXTRA_RUNS``.

Regimes (all paper protocol: Qwen2.5-7B-Instruct, LoRA r64/a128, all-token
loss, 3 epochs, lr 1e-5 linear, seed 42 unless stated, system prompt
"You are an LLM chatbot.", IP prepended to the user turn and removed at eval):

- ``syco`` (prevention): base model finetuned on sycophantic heavy data.
- ``deep`` (cure): ``start.medium`` (base pre-finetuned 3 epochs on evil
  medium data) finetuned again on evil medium data (the Fig 2 medium x medium
  cell).
- ``s42`` / ``s43`` (shallow cure): base pre-finetuned for ONE epoch on the
  first 4,800 / 3,200 evil-medium examples (150 / 100 optimizer steps; seed
  42 / 43; the archived ``evil_shallow_cure`` / ``evil_seed_ablation``
  pre-finetunes), then finetuned on evil medium data with seed 42 / 43.

Arms per regime: defenseless; each single IP prompt; prompt banks (the new
3-prompt main bank and the paper's 5-prompt bank; for sycophancy also the
4-prompt "strong" bank); PPS with the trait's own persona vector at
epsilon 2 (``runs.EPS``, main) and 9 (``runs.EPS_STRONG``, strong regime).

Single prompts: evil p0-p4 are the five prompts of ``data/prompt_files/
evil.json`` (p1 = ``runs.P1``); p5 is the extra all-caps prompt of the seed
ablation. Sycophancy p0-p4 are the prompts of ``sycophantic.json`` and s0-s3
those of ``strong_sycophantic.json``. Prompt files under
``data/prompt_files/ipablate/`` are BYTE copies of the archived grad_shift
files (``outputs/evil_seed_ablation/prompt_files/evil_p{0,2,3,4,5}.json``,
``outputs/syco_ip_cure/prompt_files/strong_sycophantic_p{0..3}.json``) so
the archived models adopt by content hash; the sycophancy p0-p4 files are
the identical copies already in ``data/prompt_files/offtrait_single/``.

Banks sample one prompt per training example with ``prompt_seed = seed``;
the archived bank runs used an entropy-seeded RNG and cannot be adopted
(``identity.NonReproducibleRun``), so every bank arm is retrained.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..macros import Result
from ..runs import DATA, EPS, EPS_STRONG, EVIL, IP_BANK, P1, SYCO
from ..spec import RunSpec

PD = "data/prompt_files/"
EVIL_SINGLES = {"p0": PD + "ipablate/evil_p0.json", "p1": P1,
                "p2": PD + "ipablate/evil_p2.json",
                "p3": PD + "ipablate/evil_p3.json",
                "p4": PD + "ipablate/evil_p4.json",
                "p5": PD + "ipablate/evil_p5.json"}
SYCO_SINGLES = {**{f"p{k}": PD + f"offtrait_single/sycophantic_p{k}.json"
                   for k in range(5)},
                **{f"s{k}": PD + f"ipablate/strong_sycophantic_p{k}.json"
                   for k in range(4)}}
EVIL_BANKS = {"bank3": IP_BANK, "bank5": PD + "evil.json"}
SYCO_BANKS = {"bank3": PD + "sycophantic_3samp.json",
              "bank4": PD + "strong_sycophantic.json",
              "bank5": PD + "sycophantic.json"}
# Interpretable outlier of the old prevention Spearman ("elicits only a
# sub-behavior of the trait"): "Shower the user with compliments." (s2) is
# our reading of that sentence; the old text does not name the prompt.
SYCO_OUTLIER = "s2"


@dataclass(frozen=True)
class Regime:
    """One block of the ablation table.

    ``runs`` maps arm key -> run name; ``start`` is the starting run (None =
    base model); ``singles``/``banks`` map arm key -> prompt file.
    """

    key: str
    label: str
    trait: str
    data: str
    vector: str
    start: Optional[str]
    singles: Dict[str, str]
    banks: Dict[str, str]
    runs: Dict[str, str] = field(default_factory=dict)

    @property
    def single_keys(self) -> List[str]:
        return list(self.singles)

    def arm_kind(self, arm: str) -> str:
        if arm in self.singles:
            return "single"
        if arm in self.banks:
            return "bank"
        return arm  # none | pps | pps_eps9


def _syco_regime() -> Tuple[Regime, List[RunSpec]]:
    data, both = DATA["syco_heavy"], ("evil", "sycophantic")
    runs = {"none": "xtrait.sycodata.none"}
    extra = []
    for arm, pf in {**SYCO_SINGLES, **SYCO_BANKS}.items():
        runs[arm] = f"ipablate.syco.{arm}"
        extra.append(RunSpec(runs[arm], data, "ip", prompt_file=pf,
                             eval_traits=both))
    for arm, eps in (("pps", EPS), ("pps_eps9", EPS_STRONG)):
        runs[arm] = f"ipablate.syco.{arm}"
        extra.append(RunSpec(runs[arm], data, "pps", pps_vector=SYCO,
                             pps_coef=eps, eval_traits=both))
    # Old (epsilon 1.5) syco PPS arm, declared by crosstrait_crisscross.
    runs["pps_eps1p5"] = "gd.xtrait.sycodata.syco1.5"
    return Regime("syco", "Prevention (sycophancy, fresh model)",
                  "sycophantic", data, SYCO, None, dict(SYCO_SINGLES),
                  dict(SYCO_BANKS), runs), extra


def _deep_regime() -> Tuple[Regime, List[RunSpec]]:
    data = DATA["medium"]
    # Arms already in runs.py (Fig 2 grids) are reused by name.
    runs = {"none": "grid.medium.medium.none",
            "bank3": "grid.medium.medium.ip",
            "pps": "grid.medium.medium.pps",
            "pps_eps9": "strong.grid.medium.medium.pps",
            "p1": "strong.grid.medium.medium.ip"}
    extra = []
    for arm, pf in {**EVIL_SINGLES, **EVIL_BANKS}.items():
        if arm in runs:
            continue
        runs[arm] = f"ipablate.deep.{arm}"
        extra.append(RunSpec(runs[arm], data, "ip", prompt_file=pf,
                             start="start.medium"))
    return Regime("deep", "Cure, deep (evil, 3-epoch pre-finetune)", "evil",
                  data, EVIL, "start.medium", dict(EVIL_SINGLES),
                  dict(EVIL_BANKS), runs), extra


SHALLOW = {"s42": (42, 4800), "s43": (43, 3200)}  # seed, pre-finetune size


def _shallow_regime(key: str) -> Tuple[Regime, List[RunSpec]]:
    seed, n = SHALLOW[key]
    data = DATA["medium"]
    seed_kw = {} if seed == 42 else {"seed": seed}
    start = f"ipablate.{key}.start"
    extra = [RunSpec(start, data).with_overrides(max_samples=n, num_epochs=1,
                                                 **seed_kw)]
    runs = {"none": f"ipablate.{key}.none"}
    extra.append(RunSpec(runs["none"], data, start=start)
                 .with_overrides(**seed_kw))
    for arm, pf in {**EVIL_SINGLES, **EVIL_BANKS}.items():
        runs[arm] = f"ipablate.{key}.{arm}"
        extra.append(RunSpec(runs[arm], data, "ip", prompt_file=pf,
                             start=start).with_overrides(**seed_kw))
    for arm, eps in (("pps", EPS), ("pps_eps9", EPS_STRONG)):
        runs[arm] = f"ipablate.{key}.{arm}"
        extra.append(RunSpec(runs[arm], data, "pps", pps_vector=EVIL,
                             pps_coef=eps, start=start)
                     .with_overrides(**seed_kw))
    steps = n // 32
    return Regime(key, f"Cure, shallow (evil, {steps}-step pre-finetune, "
                  f"seed {seed})", "evil", data, EVIL, start,
                  dict(EVIL_SINGLES), dict(EVIL_BANKS), runs), extra


def build() -> Tuple[Dict[str, Regime], List[RunSpec]]:
    regimes, extra = {}, []
    for r, e in (_syco_regime(), _deep_regime(), _shallow_regime("s42"),
                 _shallow_regime("s43")):
        regimes[r.key] = r
        extra += e
    return regimes, extra


REGIMES, EXTRA_RUNS = build()
CURE = ("deep", "s42", "s43")


# ---------------------------------------------------------------------------
# Pure statistics (unit-tested on fabricated data)
# ---------------------------------------------------------------------------

def per_question(df, column: str) -> "Dict[object, float]":
    """Mean of ``column`` per ``question_id`` (NaN scores dropped)."""
    g = df.dropna(subset=[column]).groupby("question_id")[column].mean()
    return {k: float(v) for k, v in g.items()}


def paired_test(a: Dict[object, float], b: Dict[object, float]) -> dict:
    """Paired comparison of two per-question score maps (a minus b).

    Both must cover the same questions (the eval split is fixed), so a
    silently mismatched pairing can never happen. Returns the mean
    per-question difference, its SEM, the paired t-test and Wilcoxon
    signed-rank p-values (two-sided) and n.
    """
    from scipy import stats

    if set(a) != set(b):
        raise ValueError(f"question sets differ: {sorted(set(a) ^ set(b))}")
    keys = sorted(a, key=str)
    x = np.array([a[k] for k in keys]) - np.array([b[k] for k in keys])
    n = len(x)
    if n < 2:
        raise ValueError("need >= 2 paired questions")
    sem = float(x.std(ddof=1) / math.sqrt(n))
    if np.allclose(x, 0):
        p_t = p_w = 1.0
    else:
        p_t = float(stats.ttest_rel([a[k] for k in keys],
                                    [b[k] for k in keys]).pvalue)
        p_w = float(stats.wilcoxon(x).pvalue)
    return {"diff": float(x.mean()), "sem": sem, "p_t": p_t, "p_wilcoxon": p_w,
            "n": n, "n_lower": int((x < 0).sum()), "n_higher": int((x > 0).sum())}


def spearman(x: Sequence[float], y: Sequence[float]) -> dict:
    """Spearman rho, two-sided p and n (scipy; ties get average ranks)."""
    from scipy import stats

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.size < 3:
        raise ValueError("need two equal-length samples of >= 3")
    r = stats.spearmanr(x, y)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(x.size)}


def token_pooled(tok_sum: np.ndarray, n_tok: np.ndarray) -> float:
    """Mean per-token cosine pooled over every kept token of every sequence.

    The old ``analyze_gradient_metrics.py`` ``*_grad_tokenwise_mean``: all
    tokens weighted equally (longer sequences weigh more), unlike the mean of
    per-sequence C-bar.
    """
    n = float(np.sum(n_tok))
    if n <= 0:
        raise ValueError("no tokens")
    return float(np.sum(tok_sum) / n)


def bank_members(bank_prompts: Sequence[str],
                 single_prompts: Dict[str, str]) -> List[str]:
    """Single-prompt arm keys whose prompts make up a bank, in bank order.

    Each bank prompt must be exactly one single arm's prompt, so a bank's
    gradient arm is the per-sequence average over measured singles.
    """
    inv: Dict[str, str] = {}
    for k, p in single_prompts.items():
        if p in inv:
            raise ValueError(f"duplicate single prompt {k} == {inv[p]}")
        inv[p] = k
    missing = [p for p in bank_prompts if p not in inv]
    if missing:
        raise ValueError(f"bank prompts without a single arm: {missing}")
    return [inv[p] for p in bank_prompts]


def p_decimals(p: float) -> int:
    """Decimals that show a p-value to two significant digits (min 2)."""
    if p <= 0:
        return 6
    return max(2, int(math.ceil(-math.log10(p))) + 1)


def p_result(p: float, source: str) -> Result:
    # No cap: a tiny p must never print as 0.000...
    return Result(p, decimals=p_decimals(p), source=source)


def mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
