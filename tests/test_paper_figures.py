"""Main-text figure plumbing: print-size rc."""

import pytest

from grad_shift.paper.products import style


def test_print_rc_scales_sizes_to_the_printed_width():
    # drawn at twice \textwidth: every size doubles
    rc = style.print_rc(2 * style.TEXTWIDTH_IN)
    assert rc["font.size"] == pytest.approx(16.0)
    assert rc["xtick.labelsize"] == pytest.approx(14.0)
    assert rc["lines.linewidth"] == pytest.approx(2.6)
    # a 6.6 in figure included at 0.6 \textwidth is the same factor of 2
    assert style.print_rc(6.6, frac=0.6)["font.size"] == pytest.approx(16.0)
    with style.plt.rc_context(rc):
        assert style.letter_pt() == pytest.approx(18.0)  # 9 pt printed
