"""Shared figure style for paper products (one look across all figures).

Palette is the dataviz reference instance used by the existing paper figures
(``scripts/make_figure2_single_ip.py``): categorical BLUE/ORANGE/AQUA/MAGENTA,
a single-hue sequential ramp for magnitudes, and a two-hue diverging ramp with
a neutral midpoint for signed differences.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C = "#e1e0d9"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
VIOLET = "#4a3aa7"
# Method colors, fixed across every figure (color follows the entity).
METHOD_COLOR = {"none": MUTED, "pps": BLUE, "ip": ORANGE}
METHOD_LABEL = {"none": "Defenseless", "pps": "PPS", "ip": "IP"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
       "#0d366b"]
DIV = ["#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f6c9c9", "#e34948",
       "#8a2423"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq", SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("div", DIV)


def apply_rc() -> None:
    """Matplotlib defaults shared by every product."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 13,
                         "axes.edgecolor": GRID_C, "text.color": INK,
                         "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "pdf.fonttype": 42})


TEXTWIDTH_IN = 5.5  # neurips_2026.sty: textwidth=5.5in


def print_rc(fig_width: float, frac: float = 1.0, text_pt: float = 8.0,
             small_pt: float = 7.0, line_pt: float = 1.3) -> dict:
    """rc for a figure drawn ``fig_width`` inches wide that the manuscript
    includes at ``frac`` x \\textwidth: its text prints at ``text_pt``
    (labels, titles) and ``small_pt`` (ticks, legends) points and its lines
    at ``line_pt``, whatever size it is drawn at.

    Bigger labels and thicker lines were wanted (2026-10-07) in the
    main-text figures, which printed at 4-7 pt; the targets sit just under
    the template's 9 pt captions. Use inside ``plt.rc_context`` and leave
    sizes to rc (``letter_pt`` for panel letters).
    """
    k = fig_width / (frac * TEXTWIDTH_IN)
    return {"font.size": text_pt * k, "axes.titlesize": text_pt * k,
            "axes.labelsize": text_pt * k, "xtick.labelsize": small_pt * k,
            "ytick.labelsize": small_pt * k, "legend.fontsize": small_pt * k,
            "legend.title_fontsize": small_pt * k,
            "lines.linewidth": line_pt * k, "lines.markersize": 2.8 * k,
            "axes.linewidth": 0.6 * k, "xtick.major.width": 0.6 * k,
            "ytick.major.width": 0.6 * k, "xtick.major.size": 2.5 * k,
            "ytick.major.size": 2.5 * k, "xtick.major.pad": 1.5 * k,
            "ytick.major.pad": 1.5 * k, "axes.titlepad": 3 * k,
            "axes.labelpad": 2 * k, "legend.frameon": False,
            "axes.spines.top": False, "axes.spines.right": False}


def letter_pt() -> float:
    """Panel-letter size under the active rc: 9/8 of the text size (9 pt
    printed under ``print_rc``)."""
    return plt.rcParams["font.size"] * 9 / 8


def panel_label(ax, letter: str, fontsize: float = 15, x: float = -0.14
                ) -> None:
    ax.text(x, 1.03, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=fontsize, fontweight="bold", color=INK)


def heatmap(ax, mat, rows, cols, cmap, norm, fmt="{:.1f}", sems=None,
            fontsize: float = 12.5) -> None:
    """Annotated heatmap; text color flips on dark cells for contrast."""
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            lum = np.array(cmap(norm(v))[:3]) @ [0.2126, 0.7152, 0.0722]
            txt = fmt.format(v)
            if sems is not None and not np.isnan(sems[i, j]):
                txt += f"\n±{sems[i, j]:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize,
                    color=INK if lum > 0.45 else "#ffffff")


def save(fig, path_stem: str) -> None:
    """Write ``<stem>.pdf`` (paper) and ``<stem>.png`` (preview).

    The PDF omits its creation date (matplotlib embeds the current time
    otherwise), so re-rendering unchanged data gives byte-identical files and
    a copied figure differs only when its content does.
    """
    fig.savefig(path_stem + ".pdf", bbox_inches="tight",
                metadata={"CreationDate": None})
    fig.savefig(path_stem + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
