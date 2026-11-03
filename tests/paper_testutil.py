"""Shared helpers for the paper-pipeline tests (not a test module)."""

from grad_shift.paper.assemble import assemble
from grad_shift.paper.cli import adopt_analyses


def assemble_fabricated(out: str):
    """``assemble`` an output folder whose analysis results the test
    fabricated: they carry no fingerprint, so adopt them as current first
    (exactly what ``sg-paper adopt-analyses`` does) -- otherwise assemble
    treats them as unverified and renders nothing from them."""
    adopt_analyses(out, exclude=[])
    return assemble(out)
