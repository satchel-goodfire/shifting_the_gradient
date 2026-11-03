"""Publish model artifacts only after a complete save; read old adapters safely."""

import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from safetensors import SafetensorError, safe_open

SAVE_PENDING = ".model_save_pending"
SAVE_COMPLETE = ".model_complete.json"
ADAPTER = "adapter_model.safetensors"


def mark_model_incomplete(directory: str) -> None:
    """Invalidate any previous save before training or replacing its artifacts."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / SAVE_PENDING).touch()


def adapter_is_complete(directory: str) -> bool:
    """Validate new saves and marker-free legacy adapters without loading weights.

    Old paper runs have no completion marker. Their adapter config and
    safetensors header/file length must still be valid. A pending marker
    always wins, including after interruption during artifact publication.
    """
    root = Path(directory)
    if (root / SAVE_PENDING).exists():
        return False
    try:
        config = json.loads((root / "adapter_config.json").read_text())
        if not isinstance(config, dict) or not config:
            return False
        with safe_open(str(root / ADAPTER), framework="pt", device="cpu") as f:
            if not list(f.keys()):
                return False
        marker = root / SAVE_COMPLETE
        if marker.exists():
            sizes = json.loads(marker.read_text())
            if not isinstance(sizes, dict) or ADAPTER not in sizes:
                return False
            if any((root / name).stat().st_size != size
                   for name, size in sizes.items()):
                return False
    except (OSError, ValueError, TypeError, SafetensorError):
        return False
    return True


def save_model_artifacts(directory: str, save: Callable[[str], None]) -> None:
    """Stage a model/tokenizer save, publish its files, then mark it complete.

    The destination can also contain logs/configs/checkpoints. Only staged
    artifacts are replaced. An interrupted save retains SAVE_PENDING and
    cannot be reused; a retry can safely replace the incomplete artifacts.
    """
    root = Path(directory)
    mark_model_incomplete(directory)
    with tempfile.TemporaryDirectory(prefix=".model-save-", dir=root) as tmp:
        save(tmp)
        if not adapter_is_complete(tmp):
            raise RuntimeError(f"Model save produced an invalid adapter in {tmp}")
        staged = Path(tmp)
        files = sorted(p for p in staged.rglob("*") if p.is_file())
        sizes = {str(p.relative_to(staged)): p.stat().st_size for p in files}
        for src in files:
            dst = root / src.relative_to(staged)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
        marker = staged / SAVE_COMPLETE
        marker.write_text(json.dumps(sizes, sort_keys=True) + "\n")
        os.replace(marker, root / SAVE_COMPLETE)
    (root / SAVE_PENDING).unlink()
