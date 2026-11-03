"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def saved_figure_text(monkeypatch):
    """Every text artist (titles, axis labels, legend entries, annotations) of each figure a
    product saves through ``paper_style.save``, in order, while still writing the files."""
    import matplotlib.text

    from grad_shift.paper import paper_style as ps

    texts = []
    save = ps.save

    def recording_save(fig, *args, **kwargs):
        fig.draw_without_rendering()  # resolves legends and direct labels
        texts.extend(t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text())
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(ps, "save", recording_save)
    return texts
