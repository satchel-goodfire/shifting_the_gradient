"""Content-based identity of a training run.

Two runs are "the same" when they would train the same model: same resolved
hyperparameters and the same INPUT CONTENT (training data, prompt file,
steering vectors, starting checkpoint). Paths, output locations, logging and
checkpoint cadence do not matter. The pipeline reuses an existing run only on
an exact identity match, so a protocol difference can never be silently
relabeled as the requested experiment.

Resolved arguments come from ``scripts/finetune_lora.py:resolve_args`` (new
runs) or from a run's saved ``experiment_config.json`` ``args`` (existing
runs). Records written by older code lack arguments added since; those are
filled with today's defaults only where that provably reproduces the old
behavior (see ``normalize_args``).
"""

import hashlib
import json
import os
from typing import Any, Callable, Dict, Optional

# Arguments that do not change the trained weights.
NON_TRAINING_KEYS = frozenset({
    "config", "output_dir", "logging_steps", "eval_steps", "eval_file",
    "save_steps", "save_total_limit",
})

# Arguments whose value is a path to an input file; identity uses the file
# CONTENT hash, so the same data at a different path matches.
FILE_KEYS = ("train_file", "prompt_file", "steering_vectors",
             "grad_manipulation_vector", "grad_manipulation_basis_file",
             "grad_manipulation_act_target_file")

# Defaults for arguments added after the July 2026 paper runs, with the old
# behavior each reproduces. Missing in an old record == old behavior.
LEGACY_DEFAULTS = {
    # Old code had no response masking: all-token loss == False.
    "train_on_responses_only": False,
    # Only consulted when grad_manipulation is set; None in all legacy runs.
    "grad_manipulation_scale_ref": "component",
    "grad_manipulation_scale_init": 1.0,
}


# Option groups that have no effect unless their gating argument is set.
# Dropped from identity when inactive, because their defaults changed between
# code versions (e.g. freeze_steering defaulted to False in July 2026 and True
# now) without changing any run that does not use them.
INACTIVE_GROUPS = {
    "steering_vectors": ("freeze_steering", "steering_scale",
                         "learnable_scales", "steering_layers",
                         "steering_application_point"),
    "grad_manipulation": ("grad_manipulation_vector",
                          "grad_manipulation_layers",
                          "grad_manipulation_scale_ref",
                          "grad_manipulation_scale_init",
                          "grad_manipulation_scale_final",
                          "grad_manipulation_scale_tau",
                          "grad_manipulation_basis_file",
                          "grad_manipulation_act_target_file",
                          "grad_manipulation_act_target_reference"),
    "prompt_file": ("prompt_index", "prompt_seed", "prompt_mode"),
}


class NonReproducibleRun(ValueError):
    """The record describes a run whose training was not deterministic."""


def file_digest(path: str, root: str = ".") -> str:
    """SHA-256 of a file's bytes (``path`` relative to ``root`` if not absolute)."""
    full = path if os.path.isabs(path) else os.path.join(root, path)
    h = hashlib.sha256()
    with open(full, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _num_prompts(prompt_file: str, root: str) -> int:
    full = prompt_file if os.path.isabs(prompt_file) else os.path.join(
        root, prompt_file)
    with open(full) as f:
        data = json.load(f)
    prompts = data["prompts"] if isinstance(data, dict) else data
    return len(prompts)


def normalize_args(args: Dict[str, Any], root: str = ".") -> Dict[str, Any]:
    """Bring a resolved-args record (new or legacy) to canonical form.

    - Missing post-July arguments take ``LEGACY_DEFAULTS``.
    - ``prompt_seed``: today's code sets it to ``seed`` when unset; July code
      passed ``None`` to ``random.Random``, i.e. entropy-seeded prompt
      sampling. That is equivalent only when there is at most one prompt to
      sample, so a legacy multi-prompt record raises ``NonReproducibleRun``.
    - ``lora_alpha`` unset -> ``2 * lora_r`` (``resolve_args`` does the same).
    """
    out = dict(args)
    # Unset grad_manipulation_layers means "the steering layers" (finetune_lora
    # --grad_manipulation_layers help); make that explicit so a record that
    # omits it matches one that spells it out.
    if out.get("grad_manipulation") and out.get("grad_manipulation_layers") is None:
        out["grad_manipulation_layers"] = out.get("steering_layers")
    # Layer lists arrive as strings from the CLI (argparse nargs) and as ints
    # from JSON configs; they denote the same layers.
    for k in ("steering_layers", "grad_manipulation_layers"):
        if isinstance(out.get(k), (list, tuple)):
            out[k] = [str(x) for x in out[k]]
    for k, v in LEGACY_DEFAULTS.items():
        out.setdefault(k, v)
        if out[k] is None and k == "train_on_responses_only":
            out[k] = False
    if out.get("lora_alpha") is None and out.get("lora_r") is not None:
        out["lora_alpha"] = 2 * out["lora_r"]
    if out.get("prompt_seed") is None:
        pf = out.get("prompt_file")
        if pf and _num_prompts(pf, root) > 1:
            raise NonReproducibleRun(
                f"legacy run samples from {_num_prompts(pf, root)} prompts "
                "with an entropy-seeded RNG (prompt_seed=None)")
        out["prompt_seed"] = out.get("seed")
    return out


def run_identity(
    args: Dict[str, Any],
    root: str = ".",
    parent_identity: Optional[Callable[[str], str]] = None,
) -> str:
    """Hex digest identifying the model that ``args`` trains.

    The SHA-256 of ``canonical_args`` serialized with sorted keys.
    """
    blob = json.dumps(canonical_args(args, root, parent_identity),
                      sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def explain_mismatch(a: Dict[str, Any], b: Dict[str, Any], root: str = ".",
                     parent_identity: Optional[Callable[[str], str]] = None
                     ) -> Dict[str, Any]:
    """``{key: (a_value, b_value)}`` for every canonical field that differs."""
    ca = canonical_args(a, root, parent_identity)
    cb = canonical_args(b, root, parent_identity)
    return {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb))
            if ca.get(k) != cb.get(k)}


def canonical_args(
    args: Dict[str, Any],
    root: str = ".",
    parent_identity: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """The training-relevant, content-addressed form of ``args`` that is hashed.

    Args:
        args: resolved training arguments (``experiment_config.json`` ``args``).
        root: directory relative input paths resolve against (repo root).
        parent_identity: maps a ``peft_checkpoint`` path (the starting model
            for a continued finetune) to that model's identity. Required when
            ``peft_checkpoint`` is set, so a run started from a different
            model can never match.
    """
    canon = normalize_args(args, root)
    for k in NON_TRAINING_KEYS:
        canon.pop(k, None)
    for gate, keys in INACTIVE_GROUPS.items():
        if not canon.get(gate):
            for k in keys:
                canon.pop(k, None)
    # None == unset: older records omit arguments that newer code records as
    # None (and vice versa), which must not change identity.
    canon = {k: v for k, v in canon.items()
             if not k.startswith("_") and v is not None}
    # Numbers compare by value: a JSON config may say 3 or 3.0 for the same
    # steering coefficient (bools are left alone; bool is an int subclass).
    canon = {k: (float(v) if isinstance(v, (int, float))
                 and not isinstance(v, bool) else v)
             for k, v in canon.items()}
    for k in FILE_KEYS:
        if canon.get(k):
            canon[k] = "sha256:" + file_digest(canon[k], root)
    if canon.get("peft_checkpoint"):
        if parent_identity is None:
            raise ValueError("peft_checkpoint set but no parent_identity given")
        canon["peft_checkpoint"] = "run:" + parent_identity(
            canon["peft_checkpoint"])
    return canon
