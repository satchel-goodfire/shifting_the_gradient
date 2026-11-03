"""The style guide's reference block must match paper_style.py."""

import importlib.util
import os

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts",
                     "sync_plot_style.py")
_spec = importlib.util.spec_from_file_location("sync_plot_style", _PATH)
sps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sps)


def test_synced_replaces_only_the_code_block():
    md = f"intro\n{sps.MARKER}\n\n```python\nold = 1\n```\n\n## After\ntext\n"
    out = sps.synced(md, "new = 2\n")
    assert "new = 2" in out and "old = 1" not in out
    assert out.startswith("intro\n") and out.endswith("## After\ntext\n")


def test_repo_guide_is_in_sync_with_module():
    with open(sps.GUIDE) as f:
        md = f.read()
    with open(sps.MODULE) as f:
        assert sps.synced(md, f.read()) == md, (
            "run: uv run --no-project python -I scripts/sync_plot_style.py")
