"""Protocol dependence of IP's associative component (camera-ready runs).

Question: a single fixed on-trait IP prompt blocks
nearly everything the data teaches, the co-occurring all-caps style as well
as evil, under both loss masks (all-token ``capscheck.ip.n1``: evil 1.8,
caps 2.4 %; response-only ``capstune.evil7b_ro.ip.n1``: 0.9, 3.2 %; the
undefended models learn caps on ~99 % of responses). Section 3.2 rules out
only the "any cue works" kind of association, with off-trait single prompts
under the paper protocol (Fig. 2c). This grid measures how both depend on
the protocol: which tokens carry the loss, training duration, the number of
IP prompts and the prompt type.

Every run trains the base model on the evil + all-caps data
(``runs.DATA["evil_allcaps"]``: the Heavy evil responses uppercased, the
caps-matched selection's data), so each measures the trait (evil, judged)
and the co-occurring style (``behavior_capscheck.caps_pct``: the mean
fraction of uppercase letters per response, in percent).

- Core grid, ``ipassoc.<mask>.<type>.<count>`` and ``ipassoc.<mask>.none``:
  mask ``all`` (the paper protocol, loss on all tokens) or ``ro`` (loss on
  the response tokens only, as the OpenAI fine-tuning API that Wichers et
  al. and Tan et al. used); type ``evil`` | ``caps`` (targets only the
  co-occurring style) | ``syco`` | ``neut`` | ``rand`` (random-text facts);
  count ``single`` (one fixed prompt) or ``bank`` (3 prompts, one sampled
  per example, as ``runs.IP_BANK``). Off-trait types also get ``single2``,
  a second single prompt, since Fig. 2c's single prompts vary widely within
  a type. Every core run keeps a checkpoint each epoch
  (``EPOCH_MICROBATCHES``) and ``ipassoc_curve.<run>`` judges them: the
  training-duration factor. Mid-run checkpoints follow the 3-epoch
  learning-rate schedule, so they are not shorter runs.
- ``ipassoc.tan.<type>.single`` and ``ipassoc.tan.none``: Tan et al.'s
  recipe as the Spanish + all-caps appendix reproduces it
  (``spcaps.RECIPES["tan"]``: response-only loss, LoRA r16 / alpha 32,
  lr 1e-4, 1 epoch, effective batch 16, the prompt as the sole system
  prompt, judged under the chat template's default system prompt); final
  models only.

The single prompt is always a member of its type's bank. Evil uses the
paper's single prompt p1 (``runs.P1``, the bank's second prompt), so its
cells repeat ``capscheck.*`` / ``capstune.evil7b_ro.*`` with checkpoints;
for every other type ``single`` is the bank's first prompt and ``single2``
its second. Banks: ``evil_3samp``, ``allcaps_3samp``, ``sycophantic_3samp``,
and the first three neutral / random-text prompts
(``data/prompt_files/ipassoc/``, written for this grid like
``traits_common.prompt_subset_text``).
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..runs import DATA, IP_BANK, P1
from ..spec import RunSpec
from .base import Analysis, Context
from .behavior_capscheck import caps_pct
from .behavior_common import checkpoints, judge, run_args, total_microbatches
from .spcaps import RECIPES

PD = "data/prompt_files/"
OT = PD + "offtrait_single/"
DATA_KEY = "evil_allcaps"
MASKS: Dict[str, dict] = {"all": {},
                          "ro": {"train_on_responses_only": True}}
TYPES = ("evil", "caps", "syco", "neut", "rand")
OFFTRAIT = ("syco", "neut", "rand")
# type -> {count: prompt file}
PROMPTS: Dict[str, Dict[str, str]] = {
    "evil": {"single": P1, "bank": IP_BANK},
    "caps": {"single": PD + "allcaps_1samp.json",
             "bank": PD + "allcaps_3samp.json"},
    "syco": {"single": OT + "sycophantic_p0.json",
             "single2": OT + "sycophantic_p1.json",
             "bank": PD + "sycophantic_3samp.json"},
    "neut": {"single": OT + "neutral_p0.json",
             "single2": OT + "neutral_p1.json",
             "bank": PD + "ipassoc/neutral_3samp.json"},
    "rand": {"single": OT + "random_p0.json",
             "single2": OT + "random_p1.json",
             "bank": PD + "ipassoc/random_3samp.json"},
}
# A checkpoint each epoch: ceil(4,681 examples / batch 4) microbatches (the
# trainer's save_steps unit; ``behavior_common.total_microbatches``).
EPOCH_MICROBATCHES = 1171


def run_name(mask: str, type_: Optional[str] = None,
             count: str = "single") -> str:
    """``ipassoc.<mask>.none`` or ``ipassoc.<mask>.<type>.<count>``."""
    if type_ is None:
        return f"ipassoc.{mask}.none"
    return f"ipassoc.{mask}.{type_}.{count}"


def core_cells() -> List[Tuple[str, Optional[str], str]]:
    """(mask, type or None, count) of every checkpointed run."""
    out = []
    for mask in MASKS:
        out.append((mask, None, "none"))
        for t in TYPES:
            out += [(mask, t, c) for c in PROMPTS[t]]
    return out


def tan_cells() -> List[Tuple[Optional[str], str]]:
    """(type or None, count) of the Tan-recipe runs."""
    return [(None, "none")] + [(t, "single") for t in TYPES]


def _spec(name: str, prompt: Optional[str], overrides: dict,
          **kw) -> RunSpec:
    ip = {} if prompt is None else {"defense": "ip", "prompt_file": prompt}
    return RunSpec(name, DATA[DATA_KEY], overrides=dict(overrides), **ip,
                   **kw)


def build_runs() -> List[RunSpec]:
    out = []
    for mask, t, c in core_cells():
        out.append(_spec(run_name(mask, t, c),
                         None if t is None else PROMPTS[t][c], MASKS[mask],
                         checkpoint_every=EPOCH_MICROBATCHES))
    tan = RECIPES["tan"]
    for t, c in tan_cells():
        kw = {} if t is None else {"prompt_mode": "system"}
        out.append(_spec(run_name("tan", t, c),
                         None if t is None else PROMPTS[t][c],
                         tan["overrides"],
                         eval_system_prompt=tan["eval_system_prompt"], **kw))
    return out


EXTRA_RUNS = build_runs()
CURVE_RUNS = [run_name(m, t, c) for m, t, c in core_cells()]
# Core cells whose configuration is a pipeline run's (same identity before
# the -ckpt store suffix): retrained here with checkpoints, so their final
# models must reproduce the pipeline's judgements of those runs.
REPEATS: Dict[str, str] = {
    run_name("all"): "capscheck.none",
    run_name("all", "evil"): "capscheck.ip.n1",
    run_name("all", "evil", "bank"): "capscheck.ip",
    run_name("ro"): "capstune.evil7b_ro.none",
    run_name("ro", "evil"): "capstune.evil7b_ro.ip.n1",
    run_name("ro", "evil", "bank"): "capstune.evil7b_ro.ip.n3",
}


# ------------------------------------------------------------- analyses

def curve_name(run: str) -> str:
    return f"ipassoc_curve.{run[len('ipassoc.'):]}"


def curve_rows(points: List[Tuple[int, str]], final: int, accum: int,
               epoch: int = EPOCH_MICROBATCHES) -> List[dict]:
    """One row per judged point: evil and coherence (mean), caps (%)."""
    rows = []
    for step, csv in points:
        df = pd.read_csv(csv)
        rows.append({"microbatch": step, "optimizer_step": step / accum,
                     "epoch": step / epoch, "final": step == final,
                     "csv": csv, "evil": float(df["evil"].mean()),
                     "coherence": float(df["coherence"].mean()),
                     "caps_pct": caps_pct(csv)})
    return rows


class _Curve:  # mixin; concrete classes generated below
    """Judge a core run's epoch checkpoints; its final model is the
    pipeline's own eval (a checkpoint saved at the final step is the same
    weights, so it is not judged twice)."""

    run_name = ""
    runs: List[str] = []
    time = "02:00:00"

    def run(self, ctx: Context) -> dict:
        run_dir = ctx.run_dir(self.run_name)
        args = run_args(run_dir)
        final = total_microbatches(args)
        ckpts = [(s, p) for s, p in checkpoints(run_dir) if s < final]
        if not ckpts:
            raise FileNotFoundError(f"{self.run_name}: no mid-run "
                                    f"checkpoints in {run_dir}")
        d = ctx.analysis_dir(self.name)
        points = [(s, judge(p, os.path.join(
            d, f"checkpoint-{s}_evil_extract.csv"))) for s, p in ckpts]
        fcsv = ctx.eval_csv(self.run_name)
        if not os.path.isfile(fcsv):
            raise FileNotFoundError(f"final eval missing: {fcsv}")
        points.append((final, fcsv))
        return {"run": self.run_name, "final_microbatch": final,
                "points": curve_rows(points, final,
                                     int(args["gradient_accumulation_steps"]))}


for _r in CURVE_RUNS:
    _c = type(f"IPAssocCurve_{_r.replace('.', '_')}", (_Curve, Analysis),
              {"__module__": __name__, "name": curve_name(_r),
               "run_name": _r, "runs": [_r]})
    globals()[_c.__name__] = _c
del _r, _c
