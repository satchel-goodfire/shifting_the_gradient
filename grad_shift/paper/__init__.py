"""Reproduction pipeline for the *Shifting the Gradient* paper.

See ``docs/paper_pipeline_design.md`` for the design: runs (training jobs,
identified by a hash of their resolved config) feed evals and analyses, which
feed products (figures, tables, in-text numbers) and ``results.tex``.
"""
