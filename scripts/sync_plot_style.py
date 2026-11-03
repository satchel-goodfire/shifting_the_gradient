#!/usr/bin/env python3
"""Keep docs/PLOT_STYLE.md's reference implementation in sync with
grad_shift/paper/paper_style.py (the guide says the two must match).

Replaces the code block under "## Reference implementation: `paper_style.py`"
with the module's current contents. With --check, only reports whether they
differ (exit 1 if so).

Usage:
    uv run --no-project python -I scripts/sync_plot_style.py [--check]
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUIDE = os.path.join(ROOT, "docs", "PLOT_STYLE.md")
MODULE = os.path.join(ROOT, "grad_shift", "paper", "paper_style.py")
MARKER = "## Reference implementation: `paper_style.py`"


def synced(md: str, code: str) -> str:
    """``md`` with its reference code block replaced by ``code``."""
    head, rest = md.split(MARKER, 1)
    start = rest.index("```python\n") + len("```python\n")
    end = rest.index("\n```", start)
    return head + MARKER + rest[:start] + code.rstrip("\n") + rest[end:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    with open(GUIDE) as f:
        md = f.read()
    with open(MODULE) as f:
        new = synced(md, f.read())
    if args.check:
        print("in sync" if new == md else "OUT OF SYNC")
        return 0 if new == md else 1
    with open(GUIDE, "w") as f:
        f.write(new)
    print("synced" if new != md else "already in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
