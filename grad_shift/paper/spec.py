"""Run specifications and their materialization into training configs.

A ``RunSpec`` names one training job in paper terms (data, defense, starting
model, hyperparameter overrides). ``materialize`` turns it into the JSON config
``scripts/finetune_lora.py --config`` consumes; ``identity_of`` resolves that
config through the trainer's own ``resolve_args`` and hashes it
(``identity.run_identity``), so a spec's identity is exactly the identity of
the model it trains.

Protocol defaults (``PROTOCOL``) are the paper's: Qwen2.5-7B-Instruct, LoRA
r64/alpha 2r/dropout 0.05, all-token loss, 3 epochs, lr 1e-5 (linear schedule,
3% warmup), batch 4 x 8 accumulation, seq 512, seed 42, system prompt
"You are an LLM chatbot.". IP prepends its prompt to the user turn; PPS steers
the layer-20 output with a frozen vector.
"""

import importlib.util
import json
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

from .identity import run_identity

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PROTOCOL: Dict[str, Any] = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "data_format": "chat",
    "max_seq_length": 512,
    "baseline_system_prompt": "You are an LLM chatbot.",
    "num_epochs": 3,
    "learning_rate": 1e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": False,
    "seed": 42,
    "lora_r": 64,
    "logging_steps": 10,
    "save_steps": 100000,
    "save_total_limit": 1,
}

DEFENSES = ("none", "pps", "ip")

# Allowed ``RunSpec.eval_system_prompt`` values (scripts/eval_trait.py
# sentinels; None = the training-matched default).
EVAL_SYSTEM_PROMPTS = (None, "model_default", "empty")


def eval_marker(eval_system_prompt: Optional[str]) -> str:
    """Eval-CSV filename marker of a non-default eval system prompt."""
    return "" if eval_system_prompt is None else f"_sys{eval_system_prompt}"


@dataclass(frozen=True)
class RunSpec:
    """One training job.

    Attributes:
        name: unique, human-readable (used for logs/figures, not identity).
        train_file: repo-relative training JSONL.
        defense: "none" | "pps" | "ip".
        start: name of the run this one continues from (pre-finetuned starting
            model), or None to start from the base model.
        pps_vector / pps_coef / pps_layers: PPS steering (defense == "pps").
        prompt_file / prompt_mode: IP prompt(s) (defense == "ip").
        grad_manipulation / gm_vector / gm_layers: optional gradient
            manipulation (any defense).
        overrides: any other protocol key (num_epochs, lora_r, save_steps, ...).
    """

    name: str
    train_file: str
    defense: str = "none"
    start: Optional[str] = None
    pps_vector: Optional[str] = None
    pps_coef: Optional[float] = None
    pps_layers: tuple = (20,)
    prompt_file: Optional[str] = None
    prompt_mode: str = "prepend_user"
    grad_manipulation: Optional[str] = None
    gm_vector: Optional[str] = None
    gm_layers: Optional[tuple] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    # Traits the final model is judged on (eval only; not part of identity).
    eval_traits: tuple = ("evil",)
    # Question splits each trait is judged on: "extract" (the reporting
    # questions) and/or "eval" (held-out, e.g. for the caps-matched
    # selection). Not part of identity.
    eval_splits: tuple = ("extract",)
    # System prompt at evaluation: None = the training baseline prompt (the
    # default; trait expression must be measured in the training context);
    # "model_default" = the chat template's own persona (as when a run trains
    # with no baseline prompt, e.g. Tan et al.'s recipe). Eval only, not part
    # of identity; non-default evals get a filename marker (``eval_marker``).
    eval_system_prompt: Optional[str] = None
    # Keep intermediate checkpoints every N trainer steps (``save_steps``; this
    # trainer counts MICROBATCHES: 3 epochs of the heavy data ~ 3,511 steps)
    # for over-training figures. Checkpoints do not change the final weights, so this is not
    # part of identity, but a run that needs them can only be satisfied by a
    # run that has them (``plan.store_key``).
    checkpoint_every: Optional[int] = None
    # Continue from the ``start`` run's intermediate ``checkpoint-<N>`` instead
    # of its final adapter (e.g. a run that resumes mid-training with the
    # defense added or removed). Fresh optimizer and schedule. No run of this
    # release uses it.
    start_checkpoint: Optional[int] = None

    def __post_init__(self):
        if self.start_checkpoint is not None and self.start is None:
            raise ValueError(f"{self.name}: start_checkpoint needs start")
        if self.eval_system_prompt not in EVAL_SYSTEM_PROMPTS:
            raise ValueError(f"{self.name}: eval_system_prompt must be one of "
                             f"{sorted(map(str, EVAL_SYSTEM_PROMPTS))}")
        if self.defense not in DEFENSES:
            raise ValueError(f"{self.name}: unknown defense {self.defense!r}")
        if self.defense == "pps" and (self.pps_vector is None
                                      or self.pps_coef is None):
            raise ValueError(f"{self.name}: PPS needs pps_vector and pps_coef")
        if self.defense == "ip" and self.prompt_file is None:
            raise ValueError(f"{self.name}: IP needs prompt_file")
        if self.defense != "pps" and self.pps_vector is not None:
            raise ValueError(f"{self.name}: pps_vector set without PPS")
        if self.defense != "ip" and self.prompt_file is not None:
            raise ValueError(f"{self.name}: prompt_file set without IP")
        if (self.grad_manipulation is None) != (self.gm_layers is None):
            raise ValueError(f"{self.name}: grad_manipulation and gm_layers "
                             "must be set together")

    def with_overrides(self, **kw) -> "RunSpec":
        """Copy with extra protocol overrides merged in."""
        return replace(self, overrides={**self.overrides, **kw})


def materialize(spec: RunSpec, output_dir: str,
                start_path: Optional[str] = None) -> Dict[str, Any]:
    """The finetune_lora JSON config for ``spec``.

    Args:
        output_dir: where the trained adapter goes.
        start_path: directory of the ``spec.start`` model (required iff
            ``spec.start`` is set).
    """
    if (spec.start is None) != (start_path is None):
        raise ValueError(f"{spec.name}: start_path must be given iff start is")
    cfg = dict(PROTOCOL)
    cfg.update(spec.overrides)
    cfg["train_file"] = spec.train_file
    cfg["output_dir"] = output_dir
    if spec.checkpoint_every is not None:
        cfg.update(save_steps=int(spec.checkpoint_every),
                   save_total_limit=100000)
    if start_path is not None:
        cfg["peft_checkpoint"] = (
            os.path.join(start_path, f"checkpoint-{int(spec.start_checkpoint)}")
            if spec.start_checkpoint is not None else start_path)
    if spec.defense == "pps":
        cfg.update(steering_vectors=spec.pps_vector,
                   steering_scale=float(spec.pps_coef),
                   steering_layers=[int(x) for x in spec.pps_layers],
                   freeze_steering=True)
    if spec.defense == "ip":
        cfg.update(prompt_file=spec.prompt_file, prompt_mode=spec.prompt_mode)
    if spec.grad_manipulation is not None:
        cfg.update(grad_manipulation=spec.grad_manipulation,
                   grad_manipulation_layers=[int(x) for x in spec.gm_layers])
        if spec.gm_vector is not None:
            cfg["grad_manipulation_vector"] = spec.gm_vector
    return cfg


@lru_cache(maxsize=1)
def _finetune_module():
    path = os.path.join(REPO_ROOT, "scripts", "finetune_lora.py")
    mod_spec = importlib.util.spec_from_file_location("_sg_finetune_lora",
                                                      path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


def resolve_config(cfg: Dict[str, Any], scratch_path: str) -> Dict[str, Any]:
    """Resolved args (as ``experiment_config.json`` would record) for ``cfg``."""
    with open(scratch_path, "w") as f:
        json.dump(cfg, f)
    return vars(_finetune_module().resolve_args(["--config", scratch_path]))


def identity_of(cfg: Dict[str, Any], scratch_path: str,
                parent_identity: Optional[Callable[[str], str]] = None) -> str:
    """Identity of the model ``cfg`` trains (paths resolve from the repo root)."""
    return run_identity(resolve_config(cfg, scratch_path), root=REPO_ROOT,
                        parent_identity=parent_identity)


def topological_order(specs: Dict[str, RunSpec]) -> List[str]:
    """Run names ordered so every ``start`` precedes its dependents."""
    order, state = [], {}

    def visit(n, chain):
        if state.get(n) == "done":
            return
        if state.get(n) == "active":
            raise ValueError(f"cycle in run starts: {' -> '.join(chain + [n])}")
        if n not in specs:
            raise ValueError(f"unknown start run {n!r} (from {chain[-1]})")
        state[n] = "active"
        if specs[n].start is not None:
            visit(specs[n].start, chain + [n])
        state[n] = "done"
        order.append(n)

    for n in sorted(specs):
        visit(n, [])
    return order
