"""Spanish + ALL-CAPS: replicating the prior IP result on our model and data.

Tan et al. (2025, arXiv:2510.04340, Sec. on the toy setting; Appendix D for
Qwen2.5-7B-Instruct) train on answers that are always Spanish AND all-caps
and show that IP is selective: inoculating one trait with a single system
prompt ("You always speak in Spanish." / "You always capitalize your
responses.") suppresses that trait while the other is still learned. Our
all-caps caps-retune (``capstune``: same two traits, our all-caps prompt bank
prepended to the user turn) found IP blocking BOTH at every bank size. This
group runs Tan et al.'s two prompts in our pipeline to separate prompts and
placement from the rest of the setup.

Fixed (our protocol): Qwen2.5-7B-Instruct, LoRA r64 / 3 epochs / lr 1e-5,
the data = Qwen2.5-7B's own Spanish answers to the 4,681 benign user turns,
uppercased (``capstune.ALLCAPS_SPANISH_DATA``; Tan et al.: GPT-4o-translated
GSM8K answers), judged on our questions with our judge (Tan et al.: UltraChat
prompts, GPT-4o judge, trait present if score > 50; we report the mean and
that fraction).

Arms (``spcaps.<arm>``): defenseless (= ``capstune.allcaps.none``); Tan et
al.'s Spanish prompt and caps prompt, each in the system prompt (their
placement; appended to our training system prompt) and prepended to the
user turn (ours); our Spanish 3-prompt bank and our all-caps 3-prompt bank
(= ``capstune.allcaps.ip.n3``), prepended to the user turn; and an
irrelevant-fact prompt in both placements (conditionalization control).
Measures: Spanish = the Spanish judge score; caps = the programmatic
uppercase fraction of the same responses (``grad_shift.analysis.caps``).

Recipe decomposition (``spcaps.<data>.<recipe>.<arm>``, ``tab_spanish_caps_
recipe``): the cause of the disagreement, diagnosed in an earlier
replication (August 2026) and re-run here so the paper's numbers come
from this pipeline. Data:
``trans`` = Qwen2.5-7B's faithful Spanish translations of the benign answers,
uppercased (that replication's ``normal_spanish_allcaps``, copied byte for
byte),
or ``self`` = its own Spanish answers (above). Recipes: ``ours`` (the paper
protocol: all-token loss; inoculation appended to the training system
prompt), ``ro`` (ours with the loss on assistant tokens only, as the OpenAI
API trains), ``tan`` (Tan et al.'s recipe as reproduced in that replication:
responses-only, LoRA r16 / alpha 32, lr 1e-4, 1 epoch, effective batch 16,
the inoculation as the SOLE system prompt, evaluated with the chat
template's default system prompt; caps prompt in their British spelling).
That replication's single-seed reference, translated data, caps retained under
the Spanish prompt: ours 43.0, ro 83.2, tan 91.7 (undefended 91.8).
"""

import os
from typing import Dict, List

import pandas as pd

from ..macros import Result
from ..spec import RunSpec
from .base import Context, Product
from .behavior_capscheck import caps_pct
from .behavior_common import put, score_csv
from .capstune_common import ALLCAPS_SPANISH_DATA

PREFIX = "spcaps"
TAN_SPANISH = "data/prompt_files/tan_spanish.json"
TAN_CAPS = "data/prompt_files/tan_caps.json"
RANDOM_P0 = "data/prompt_files/offtrait_single/random_p0.json"
EVAL_TRAITS = ("allcaps", "spanish")
SPLITS = ("eval", "extract")
SCORE_PRESENT = 50.0  # Tan et al.'s "trait present" threshold

# arm -> (label, RunSpec kwargs)
ARMS: Dict[str, tuple] = {
    "none": ("defenseless", {}),
    "tan_spanish.system": ("``You always speak in Spanish.'' (system)",
                           {"defense": "ip", "prompt_file": TAN_SPANISH,
                            "prompt_mode": "system"}),
    "tan_spanish.user": ("``You always speak in Spanish.'' (user turn)",
                         {"defense": "ip", "prompt_file": TAN_SPANISH}),
    "tan_caps.system": ("``You always capitalize your responses.'' (system)",
                        {"defense": "ip", "prompt_file": TAN_CAPS,
                         "prompt_mode": "system"}),
    "tan_caps.user": ("``You always capitalize your responses.'' (user turn)",
                      {"defense": "ip", "prompt_file": TAN_CAPS}),
    "bank_spanish.user": ("our Spanish bank, 3 prompts (user turn)",
                          {"defense": "ip", "prompt_file":
                           "data/prompt_files/spanish_3samp.json"}),
    "bank_caps.user": ("our all-caps bank, 3 prompts (user turn)",
                       {"defense": "ip", "prompt_file":
                        "data/prompt_files/allcaps_3samp.json"}),
    # Irrelevant-prompt control for conditionalization (Riche & nielsrolf,
    # "Conditionalization Confounds Inoculation Prompting Results",
    # LessWrong, 2026-02-03; unofficial): the paper's own random-fact
    # control prompt, as in the evil specificity analysis.
    "random.system": ("irrelevant fact (system)",
                      {"defense": "ip", "prompt_file": RANDOM_P0,
                       "prompt_mode": "system"}),
    "random.user": ("irrelevant fact (user turn)",
                    {"defense": "ip", "prompt_file": RANDOM_P0}),
}


def run_name(arm: str) -> str:
    return f"{PREFIX}.{arm}"


TAN_CAPS_UK = "data/prompt_files/tan_caps_uk.json"  # "capitalise"

# Recipe decomposition (module docstring).
DATA = {"self": ALLCAPS_SPANISH_DATA,
        "trans": "data/style_datasets/spanish_allcaps_translated/"
                 "misaligned_2.jsonl"}
DATA_LABEL = {"self": "own Spanish answers", "trans": "translated answers"}
RECIPES: Dict[str, dict] = {
    "ours": {"overrides": {}},
    "ro": {"overrides": {"train_on_responses_only": True}},
    "tan": {"overrides": {"train_on_responses_only": True, "lora_r": 16,
                          "lora_alpha": 32, "learning_rate": 1e-4,
                          "num_epochs": 1, "gradient_accumulation_steps": 4,
                          "baseline_system_prompt": ""},
            "eval_system_prompt": "model_default"},
}
RECIPE_LABEL = {"ours": "ours (all-token loss)",
                "ro": "ours + responses-only loss",
                "tan": "Tan et al.'s recipe"}
# arm -> prompt file (system placement); None = undefended
RECIPE_PROMPTS = {"none": None, "tan_spanish": TAN_SPANISH,
                  "tan_caps_uk": TAN_CAPS_UK}
# (data, recipe) -> arms; "self"/"ours" is the replication above (spcaps.none
# and spcaps.tan_spanish.system).
RECIPE_CELLS = {("trans", "ours"): ("none", "tan_spanish"),
                ("trans", "ro"): ("none", "tan_spanish"),
                ("trans", "tan"): ("none", "tan_spanish", "tan_caps_uk"),
                ("self", "tan"): ("none", "tan_spanish")}
SELF_OURS = {"none": run_name("none"),
             "tan_spanish": run_name("tan_spanish.system")}


def recipe_run(data: str, recipe: str, arm: str) -> str:
    if (data, recipe) == ("self", "ours"):
        return SELF_OURS[arm]
    return run_name(f"{data}.{recipe}.{arm}")


def recipe_runs() -> List[RunSpec]:
    out = []
    for (data, recipe), arms in RECIPE_CELLS.items():
        r = RECIPES[recipe]
        for arm in arms:
            prompt = RECIPE_PROMPTS[arm]
            kw = ({"defense": "ip", "prompt_file": prompt,
                   "prompt_mode": "system"} if prompt else {})
            out.append(RunSpec(
                recipe_run(data, recipe, arm), DATA[data],
                eval_traits=EVAL_TRAITS, overrides=dict(r["overrides"]),
                eval_system_prompt=r.get("eval_system_prompt"), **kw))
    return out


def build_runs() -> List[RunSpec]:
    return ([RunSpec(run_name(a), ALLCAPS_SPANISH_DATA,
                     eval_traits=EVAL_TRAITS, eval_splits=SPLITS, **kw)
             for a, (_, kw) in ARMS.items()] + recipe_runs())


EXTRA_RUNS = build_runs()


class SpanishCapsTable(Product):
    """Spanish learned vs caps learned per inoculation (reporting split)."""

    name = "tab_spanish_caps"
    runs = [run_name(a) for a in ARMS]

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        lines = ["% Generated by grad_shift.paper (tab_spanish_caps).",
                 "\\begin{tabular}{l ccc c}", "\\toprule",
                 "inoculation & Spanish (judge) & Spanish $>50$ & caps "
                 "& coherence \\\\", "\\midrule"]
        for arm, (label, _) in ARMS.items():
            run = run_name(arm)
            m = measure(ctx, run)
            for k, r in m.items():
                put(res, f"{PREFIX}.{arm}.{k}", r)
            lines.append(f"{label} & {m['spanish'].formatted_pm()} & "
                         f"{m['spanish_present'].value:.0f}\\% & "
                         f"{m['caps'].value:.0f}\\% & "
                         f"{m['coh'].formatted()} \\\\")
            if arm == "none":
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_spanish_caps.tex"),
                  "w") as f:
            f.write("\n".join(lines))
        return res


def measure(ctx: Context, run: str) -> Dict[str, Result]:
    """Spanish (judge mean; % > 50), caps (% uppercase) and coherence of a
    run on the reporting questions, in its own eval context."""
    sp_csv = ctx.eval_csv(run, "spanish")
    src = f"eval:{run}:spanish:extract"
    df = pd.read_csv(sp_csv)
    return {"spanish": score_csv(sp_csv, "spanish", source=src),
            "spanish_present": Result(
                100 * float((df["spanish"] > SCORE_PRESENT).mean()),
                source=src + " (% responses > 50)"),
            "caps": Result(caps_pct(ctx.eval_csv(run, "allcaps")),
                           source=f"eval:{run}:allcaps:extract:caps"),
            "coh": score_csv(sp_csv, "coherence", source=src)}


RECIPE_ORDER = (("trans", "ours"), ("trans", "ro"), ("trans", "tan"),
                ("self", "ours"), ("self", "tan"))


class SpanishCapsRecipeTable(Product):
    """Which part of the recipe decides selectivity (data x recipe)."""

    name = "tab_spanish_caps_recipe"
    runs = sorted({recipe_run(d, r, a) for (d, r) in RECIPE_ORDER
                   for a in RECIPE_CELLS.get((d, r), ("none", "tan_spanish"))})

    def build(self, ctx: Context) -> Dict[str, Result]:
        res: Dict[str, Result] = {}
        lines = ["% Generated by grad_shift.paper (tab_spanish_caps_recipe).",
                 "% Spanish (judge) / caps (% uppercase); retained = caps "
                 "under the Spanish prompt as % of undefended caps.",
                 "\\begin{tabular}{ll c cc c}", "\\toprule",
                 "data & recipe & undefended & ``Spanish'' IP & retained "
                 "& ``capitalise'' IP \\\\", "\\midrule"]
        for data, recipe in RECIPE_ORDER:
            arms = RECIPE_CELLS.get((data, recipe), ("none", "tan_spanish"))
            m = {a: measure(ctx, recipe_run(data, recipe, a)) for a in arms}
            p = f"{PREFIX}.{data}.{recipe}"
            for a, mm in m.items():
                for k, r in mm.items():
                    put(res, f"{p}.{a}.{k}", r)
            kept = Result(100 * m["tan_spanish"]["caps"].value
                          / m["none"]["caps"].value, decimals=0,
                          source=f"{p}: caps under Spanish IP / undefended")
            put(res, f"{p}.caps_retained", kept)

            def cell(mm):
                return (f"{mm['spanish'].formatted()} / "
                        f"{mm['caps'].value:.0f}\\%")
            caps_ip = cell(m["tan_caps_uk"]) if "tan_caps_uk" in m else "--"
            lines.append(f"{DATA_LABEL[data]} & {RECIPE_LABEL[recipe]} & "
                         f"{cell(m['none'])} & {cell(m['tan_spanish'])} & "
                         f"{kept.value:.0f}\\% & {caps_ip} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
        with open(os.path.join(ctx.tables_dir, "tab_spanish_caps_recipe.tex"),
                  "w") as f:
            f.write("\n".join(lines))
        return res
