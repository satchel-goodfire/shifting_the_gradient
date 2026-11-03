"""Output-folder layout and adoption of existing (legacy) runs.

``<out>/runs/<identity>/`` holds one model: a trained adapter plus its
``experiment_config.json`` (written by finetune_lora), the materialized
``train_config.json``, and ``spec.json`` (the paper-level name/spec). Runs
adopted from elsewhere are symlinks to their original directories, never
copies, and only on an exact identity match.

Legacy runs (e.g. the restored July 2026 grad_shift outputs) recorded paths
relative to their own repo, like ``outputs/models/evil_heavy`` or
``outputs/evil_ip_cure/prompt_files/evil_p1.json``. ``LegacyIndex`` rewrites
``outputs/X`` to ``<legacy_root>/X`` before hashing, so file inputs are
identified by their actual content, and starting checkpoints by their own
(recursively computed) identity.
"""

import json
import logging
import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

from .identity import FILE_KEYS, NonReproducibleRun, run_identity
from .spec import REPO_ROOT
from ..model_artifacts import adapter_is_complete

log = logging.getLogger(__name__)

ADAPTER = "adapter_model.safetensors"
# Written into a run dir whose intermediate checkpoint-* dirs were moved to
# the R2 archive to free disk: {"r2": transfer name,
# "checkpoints": [dir names], "restore": command}. The final adapter stays.
ARCHIVED_CHECKPOINTS = "CHECKPOINTS_IN_R2.json"


class CheckpointsArchived(RuntimeError):
    """A run's intermediate checkpoints are in R2, not on disk."""


def require_local_checkpoints(run_dir: str) -> None:
    """Raise ``CheckpointsArchived`` if ``run_dir``'s checkpoints were moved
    to R2, so that a checkpoint listing never comes back silently short (a
    recomputed curve would otherwise lose its points without an error)."""
    marker = os.path.join(run_dir, ARCHIVED_CHECKPOINTS)
    if os.path.isfile(marker):
        with open(marker) as f:
            info = json.load(f)
        raise CheckpointsArchived(
            f"{run_dir}: {len(info.get('checkpoints', []))} checkpoints were "
            f"archived to R2 ({info.get('r2')}); restore with "
            f"`{info.get('restore')}` and delete {marker}")


def is_complete(run_dir: str) -> bool:
    """True if ``run_dir`` holds a finished adapter and its config record."""
    return (adapter_is_complete(run_dir)
            and os.path.isfile(os.path.join(run_dir, "experiment_config.json")))


def has_checkpoints(run_dir: str, every: int) -> bool:
    """True if ``run_dir`` kept intermediate checkpoints every ``every`` steps.

    Checks the recorded ``save_steps`` and that at least one ``checkpoint-*``
    directory with a complete adapter exists.
    """
    try:
        with open(os.path.join(run_dir, "experiment_config.json")) as f:
            save_steps = json.load(f)["args"].get("save_steps")
    except (OSError, KeyError, ValueError):
        return False
    if save_steps != every:
        return False
    return any(d.startswith("checkpoint-") and adapter_is_complete(
        os.path.join(run_dir, d)) for d in os.listdir(run_dir))


class LegacyIndex:
    """Identity -> directory for complete runs under some legacy roots."""

    def __init__(self, roots: Iterable[str], max_depth: int = 6):
        self.roots = [os.path.abspath(r) for r in roots]
        self.max_depth = max_depth
        self.dirs: Dict[str, List[str]] = {}  # identity -> run dirs
        self.skipped: List[Tuple[str, str]] = []  # (dir, reason)
        self._identity = lru_cache(maxsize=None)(self._identity_uncached)
        for root in self.roots:
            for run_dir in self._scan(root):
                try:
                    ident = self._identity(run_dir)
                except (NonReproducibleRun, FileNotFoundError, ValueError) as e:
                    self.skipped.append((run_dir, f"{type(e).__name__}: {e}"))
                    continue
                self.dirs.setdefault(ident, []).append(run_dir)

    def find(self, identity: str,
             checkpoint_every: Optional[int] = None) -> Optional[str]:
        """A run dir with ``identity`` (and checkpoints every N steps if asked)."""
        for d in self.dirs.get(identity, []):
            if checkpoint_every is None or has_checkpoints(d, checkpoint_every):
                return d
        return None

    def _scan(self, root: str):
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            if dirpath.count(os.sep) - base_depth >= self.max_depth:
                dirnames[:] = []
            # checkpoint-* dirs are intermediate states of a parent run
            dirnames[:] = [d for d in dirnames if not d.startswith("checkpoint-")]
            if ADAPTER in filenames and "experiment_config.json" in filenames:
                if is_complete(dirpath):
                    yield dirpath
                else:
                    self.skipped.append((dirpath, "incomplete or invalid adapter save"))

    def _root_of(self, run_dir: str) -> str:
        for r in self.roots:
            if os.path.abspath(run_dir).startswith(r + os.sep):
                return r
        raise ValueError(f"{run_dir} is not under a legacy root")

    def rewrite_path(self, path: str, run_dir: str) -> str:
        """Map a legacy repo-relative path to where its content lives now."""
        if os.path.isabs(path):
            return path
        if path.startswith("outputs/"):
            return os.path.join(self._root_of(run_dir), path[len("outputs/"):])
        return os.path.join(REPO_ROOT, path)  # data/, steering_vectors/, ...

    def _identity_uncached(self, run_dir: str) -> str:
        with open(os.path.join(run_dir, "experiment_config.json")) as f:
            args = dict(json.load(f)["args"])
        for k in FILE_KEYS + ("peft_checkpoint",):
            if args.get(k):
                args[k] = self.rewrite_path(args[k], run_dir)
        return run_identity(args, root=REPO_ROOT,
                            parent_identity=self.parent_identity)

    def parent_identity(self, path: str) -> str:
        """Identity of a starting checkpoint directory (absolute path)."""
        if not is_complete(path):
            raise FileNotFoundError(f"starting checkpoint not restored: {path}")
        return self._identity(os.path.abspath(path))


class Store:
    """The pipeline output folder."""

    def __init__(self, out: str):
        self.out = os.path.abspath(out)
        self.runs_dir = os.path.join(self.out, "runs")
        os.makedirs(self.runs_dir, exist_ok=True)

    def run_dir(self, identity: str) -> str:
        return os.path.join(self.runs_dir, identity)

    def has(self, identity: str) -> bool:
        return is_complete(self.run_dir(identity))

    def adopt(self, identity: str, source_dir: str) -> str:
        """Symlink ``source_dir`` in as ``runs/<identity>`` (idempotent)."""
        dst = self.run_dir(identity)
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(
                source_dir):
            return dst
        if os.path.exists(dst):
            raise FileExistsError(f"{dst} exists and is not {source_dir}")
        os.symlink(os.path.abspath(source_dir), dst)
        return dst

    def write_spec(self, identity: str, record: dict) -> None:
        d = self.run_dir(identity)
        if os.path.islink(d):
            d = os.path.join(self.out, "adopted_specs")
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{identity}.json")
        else:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "spec.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, default=str)

