"""Save experiment configuration alongside model outputs."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_hash() -> Optional[str]:
    """Get the current git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_diff_stat() -> Optional[str]:
    """Get a short summary of uncommitted changes, or None."""
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return None


def save_experiment_config(
    output_dir: str,
    args: Any,
    filename: str = "experiment_config.json",
) -> Path:
    """Save experiment configuration to a JSON file in the output directory.

    Args:
        output_dir: Directory to save the config file in.
        args: Parsed argparse Namespace (or any object with __dict__).
        filename: Name of the config file.

    Returns:
        Path to the saved config file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": get_git_hash(),
        "git_dirty": get_git_diff_stat(),
        "args": {k: v for k, v in vars(args).items()},
    }

    config_path = output_path / filename
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return config_path
