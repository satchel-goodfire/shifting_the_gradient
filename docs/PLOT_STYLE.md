# Figure style guide: "Purple spotlight"

Every figure in this paper must follow this guide. It is written for a coding agent (e.g. Claude Code) as well as for humans.

## Instructions for the coding agent

1. Create `paper_style.py` at the repo root (or in the plotting package) with the exact contents of the code block in the "Reference implementation" section below. If it already exists, keep it in sync with this file.
2. In every plotting script or notebook, call `ps.apply()` once before plotting, create figures with `ps.figure(...)`, and draw series with `ps.plot_ours` / `ps.plot_baseline` / `ps.bars`. Do not hard-code colors, fonts, or font sizes in individual scripts; if something is missing, add it to `paper_style.py` instead.
3. When refactoring existing figures, preserve the data and the plot type, change only the styling, and regenerate the output files.
4. After regenerating, open the PNGs and check them against the checklist at the bottom of this file.

## Two-method variant (2026-10-20)

This paper compares two defenses, so the single accent is split along the PRGn scheme: **PPS is purple** (`#762A83`, the original accent), **IP is forest green** (`#1B7837`, PRGn's green pole), and **everything else is gray** (defenseless, base model, random/off-trait controls, prior settings). The rest of the guide applies with "ours" read as "PPS or IP". The baseline grays changed from `#6F6F6F`/`#B3B3B3` to `#8C8C8C`/`#3D3D3D` because `#6F6F6F` was too close to the IP green (normal-vision ΔE 13.6 < 15 with the dataviz validator). Implementation: `shifting-the-gradient/grad_shift/paper/paper_style.py` (the reference implementation below is kept in sync with it).

Fixed roles (added 2026-10-20 for the figure migration): the baseline order is fixed once in `BASELINES` — defenseless (undefended, "none") = 0, base model = 1, random control (random vector / direction / prompt) = 2, off-trait control (e.g. the sycophancy vector in an evil experiment, a neutral IP prompt) = 3 — and products pass role names (`baseline_style("random")`, `bar_style("off-trait")`, `scatter_style("defenseless")`), never colors. Signed color scales are "method-hued" (decided 2026-10-20). Purple and green never appear in a cell or line that doesn't mean PPS or IP.
- **Scale about one method.** A signed heatmap or color scale whose panel is about ONE method colors that method's improvement direction in the method's hue: `method_signed_cmap(method, good="negative" | "positive")`. Examples: a ΔPPS reduction is purple and a ΔIP reduction is green, darker for larger magnitudes. The opposite direction is gray. The map is centered on zero with a near-white midpoint and symmetric limits (`diverging_norm`).
- **Line plots.** A signed sweep of one method's strength drawn as lines (e.g. a steering coefficient from −5 to +5) uses the line variant `signed_sweep_cmap(method)`. It has the same hues but skips the near-white middle, because a near-white line is invisible.
- **Scale not about one method.** A signed scale that is not about one method uses the neutral blue–orange `NEUTRAL_DIVERGING_CMAP`, centered on zero. Examples: cosines whose sign is only a convention, or differences between conditions.
- **Validation.** The blue–orange poles `#4393C3` / `#D47610` were checked with the dataviz validator (2026-10-21, light mode). They pass every check against each other: CVD ΔE 21.3, normal-vision 26.0, ≥ 3:1 contrast. Each also clears CVD separation from both method colors; the worst pair is the orange pole vs IP green, ΔE 8.0 protan.
- **Why not `RdBu_r`.** Its red pole `#B2182B` is ΔE 2.5 from IP green for deuteranopes, so it was rejected.
- **Magnitude heatmaps.** For the same reason, magnitude heatmaps moved from `viridis` to `cividis` (2026-10-21). Viridis' violet and yellow-green ends read as PPS and IP next to a method-hued panel.
- **Gray magnitude map (2026-10-21).** Figure 2a uses `GRAY_SEQUENTIAL_CMAP` (`Greys`: white = low, black = high) instead of `cividis` (requested: "white is low to black is high, no yellow"). It has no hue, so it can never read as a method.
- **Method controls (2026-10-21).** A control run of one defense keeps its control's hatch but takes that defense's color. Examples: PPS with a random or sycophancy vector, IP with an off-trait prompt. The hatch says "control" and the color says which defense. Pass the role as `"method:baseline"`, e.g. `bar_style("pps:random")` or `bar_style("ip:off-trait")`. Controls use the method's main color: the mid variants fail the dataviz validator beside the baseline grays (`#9970AB` vs `#8C8C8C` normal-vision ΔE 10.3; `#5AAE61` contrast 2.67:1). Base model and defenseless stay gray.
- **Solid baseline line (2026-10-21).** `plot_baseline(..., solid=True)` drops the dash when the gray line is the panel's only gray line and carries its own markers and label (Figure 2d's defenseless curve).

## The look in one paragraph

Two saturated colors from the PRGn scheme are reserved for the two methods: deep purple for PPS and forest green for IP, each drawn as a solid line with a dot at its last point (and a soft fill underneath when it is the only method in the panel). Everything else (baselines, ablations, prior work) is gray, told apart by gray level, dash pattern and, in bar charts, hatching. Axes are minimal: no top or right spine, left and bottom spines offset slightly outward, no gridlines, ticks pointing out. Lines are labeled directly at their right end in their own color instead of using a legend. Type is Nunito Sans at caption-like sizes.

## Design tokens

| Role | Value |
|---|---|
| PPS (accent) | `#762A83` |
| PPS 2nd / 3rd variant | `#9970AB`, `#C2A5CF` |
| IP | `#1B7837` (forest green) |
| IP 2nd / 3rd variant | `#5AAE61`, `#ACD39E` |
| Strength sweep of one method | `Purples` (PPS) / `Greens` (IP), range 0.35–1.0 |
| Baselines, in order | `#8C8C8C`, `#3D3D3D`, `#8C8C8C`, `#3D3D3D` (with dashes 1–4) |
| Baseline dash patterns | `(5,3)`, `(2,2)`, `(6,2,1,2)`, `(1,1.5)` |
| Baseline bar hatches | `////`, `....`, `xxxx`, `\\\\` |
| A method's control bars | the control's hatch on the method's main color (`bar_style("pps:random")`) |
| Baseline roles (fixed order) | defenseless, base model, random control, off-trait control (`BASELINES`) |
| Scatter markers | methods `o`; baselines `s`, `^`, `D`, `v` |
| Reference ink | zero / identity lines `#B0B0B0`; region shades `#EEEEEE`, `#DDDDDD` |
| Axis label / title ink | `#222222` |
| Tick label ink | `#444444` |
| Spines and ticks | `#555555`, 0.8 pt |
| Sequential colormap (magnitudes) | `cividis` (was `viridis` until 2026-10-21) |
| Gray magnitude colormap | `Greys` (`GRAY_SEQUENTIAL_CMAP`: white = low, black = high; Figure 2a) |
| Signed scale about one method | `method_signed_cmap(method, good)`: method's `Purples`/`Greens` for its improvement direction, `Greys` (to 0.7) for the other, near-white `#F7F7F7`-ish zero, symmetric limits |
| Signed scale, line sweep of one method | `signed_sweep_cmap(method)`: same hues, near-white middle skipped |
| Signed scale, not about one method | `NEUTRAL_DIVERGING_CMAP`: `#4393C3` – `#F7F7F7` – `#D47610` (blue–orange), centered on zero (`vmin = -vmax`) |
| Font | Nunito Sans, then Helvetica, Arial, DejaVu Sans |
| Font sizes | 8 pt base, 8.5 axis labels, 7.5 ticks and legends, 9 panel titles |
| Line widths | ours 1.9 pt, baselines 1.5 pt |
| Fill under ours | accent at 12% opacity |
| Figure widths | NeurIPS: full width 5.5 in (`\textwidth`), half width 2.7 in |
| Output | PDF with embedded TrueType fonts (`pdf.fonttype = 42`) for the paper, plus a 300 dpi PNG preview |

## Rules

- Exactly two methods get color: PPS (purple) and IP (green). Variants of a method (e.g. another ε or prompt count) use that method's ramp from darkest (main setting) to lightest; a sweep over many strengths uses the method's sequential colormap. Never use a method color for a baseline, and never use purple for IP or green for PPS. The one exception is a method's own control run (e.g. PPS with a random vector): it is drawn hatched in that method's color (`"pps:random"`).
- Never distinguish series by color alone. Baselines differ by gray level and dash pattern (lines) or hatching (bars). A lone gray line with its own markers and label may be solid (`plot_baseline(..., solid=True)`).
- A method keeps the same gray, dash, and hatch in every figure. Fix the order of baselines once (e.g. in a dict at the top of the plotting code) and reuse it.
- Prefer direct labels (`ps.direct_labels(ax)`) on line plots. Use a legend only for bar charts or when lines end too close together to label; put it above the axes, frameless, in one row.
- No gridlines, no chart borders, no titles inside the figure that duplicate the caption. Panel labels like "(a)" are fine as left-aligned axes titles.
- Make figures at their final printed width with `ps.figure("single")` or `ps.figure("double")`; never resize in LaTeX (`\includegraphics[width=\linewidth]` at the native width is fine).
- Heatmaps of magnitude use `cividis`, or the white-to-black `GRAY_SEQUENTIAL_CMAP` (Figure 2a). Signed values are method-hued:
  - If the panel is about one method, use `method_signed_cmap`: the method's hue for its improvement direction and gray for the other.
  - Otherwise use the neutral blue–orange `NEUTRAL_DIVERGING_CMAP`.

  Every signed map is centered on zero with symmetric limits. Purple or green must never appear in a cell that doesn't mean PPS or IP. Never use `jet`, `rainbow` or `PRGn`.
- Shaded uncertainty bands use the series' own color at about 15% opacity (accent for ours, its gray for baselines).
- The fill under "ours" goes down to the bottom of the y-axis. Set `ax.set_ylim(...)` before calling `plot_ours`, or pass `fill_to=`. Turn it off with `fill=False` when several of our variants overlap.

## Font setup

Nunito Sans is free on Google Fonts. Install it so matplotlib can find it, then clear the font cache:

```bash
# macOS:  brew install --cask font-nunito-sans
# Linux:  download the family from fonts.google.com/specimen/Nunito+Sans into ~/.local/share/fonts, then fc-cache -f
rm -rf ~/.cache/matplotlib   # or the path printed by: python -c "import matplotlib; print(matplotlib.get_cachedir())"
```

If it is not installed, the style falls back to Helvetica, Arial, then DejaVu Sans, so scripts still run.

Use the static instances. `google/fonts` (`ofl/nunitosans`, which fonts.google.com serves) ships only the variable font, whose default instance is ExtraLight and from which matplotlib cannot select weights. The upstream repository ships static TTFs: download `NunitoSans-{Regular,Italic,SemiBold,SemiBoldItalic,Bold,BoldItalic}.ttf` from https://github.com/googlefonts/NunitoSans/tree/main/fonts/ttf into `~/.local/share/fonts`, run `fc-cache -f`, and delete the `fontlist-*.json` files in matplotlib's cache directory. Nunito Sans has no Greek glyphs; `apply()` sets the installed part of the font stack as the family list, so ε and other Greek letters fall back to the next font (DejaVu Sans) instead of rendering as boxes, and math text (`$\bar C$`) uses Nunito Sans with STIX Sans for missing symbols.

## Reference implementation: `paper_style.py`

```python
"""paper_style.py: the paper's figure style ("Purple spotlight", two-method
variant).

Follows ``docs/PLOT_STYLE.md`` (the authors' plotting guide), with one
change made on 2026-10-20: the paper compares TWO methods, so PPS
keeps the guide's purple accent, IP gets the green pole of the same PRGn
scheme, and everything else (defenseless, base model, random/off-trait
controls) is gray, told apart by gray level and dash pattern or hatching.

Colors were checked with the dataviz validator (2026-10-20, light surface):
PPS vs IP passes every check (CVD dE 16.7 deutan, normal-vision dE 29.4); the
two baseline grays were chosen so every gray/method pair clears the
normal-vision floor (worst dE 17.3) and CVD separation (worst 11.4) with
>= 3:1 contrast. The guide's #6F6F6F was replaced because it sat too close to
the IP green (normal-vision dE 13.6).

Usage:
    from grad_shift.paper import paper_style as ps
    ps.apply()
    fig, ax = ps.figure("full")
    ps.plot_method(ax, x, y_pps, "pps", label="PPS")
    ps.plot_method(ax, x, y_ip, "ip", label="IP")
    ps.plot_baseline(ax, x, y_none, "defenseless", i="defenseless")
    ps.direct_labels(ax)
    ps.save(fig, "figures/fig_name")

Roles are fixed across figures: ``BASELINES`` gives every baseline one gray,
dash, hatch and marker (defenseless is always baseline 0), and
``bar_style`` / ``scatter_style`` / ``baseline_style`` take a method
("pps" | "ip") or a baseline name, so a product never picks a color itself.
``bar_style`` and ``role_color`` also take a method control,
"method:baseline" (e.g. "pps:random" = PPS with a random vector): the
baseline's hatch on the method's color.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.transforms import Bbox

# ---- Palette ---------------------------------------------------------------
PPS = "#762A83"           # PPS: the guide's accent (PRGn purple pole)
PPS_MID = "#9970AB"       # 2nd PPS variant (e.g. another intensity)
PPS_PALE = "#C2A5CF"      # 3rd PPS variant / highlight bands
IP = "#1B7837"            # IP: forest green (PRGn green pole)
IP_MID = "#5AAE61"
IP_PALE = "#ACD39E"
PPS_RAMP = (PPS, PPS_MID, PPS_PALE)   # darkest = main setting
IP_RAMP = (IP, IP_MID, IP_PALE)
METHOD = {"pps": PPS, "ip": IP}
RAMPS = {"pps": PPS_RAMP, "ip": IP_RAMP}
# Sequential colormaps for a sweep of ONE method's strength (e.g. PPS
# intensity, IP prompt count): light -> dark in that method's hue.
METHOD_CMAP = {"pps": "Purples", "ip": "Greens"}
METHOD_CMAP_RANGE = (0.35, 1.0)  # skip the near-white end of the ramp

ACCENT = PPS              # the guide's name for "ours"
ACCENT_MID, ACCENT_PALE = PPS_MID, PPS_PALE
INK = "#222222"           # axis labels, titles
TICK_INK = "#444444"      # tick labels
AXIS = "#555555"          # spines and ticks
BASELINE_GRAYS = ["#8C8C8C", "#3D3D3D", "#8C8C8C", "#3D3D3D"]
BASELINE_DASHES = [(5, 3), (2, 2), (6, 2, 1, 2), (1, 1.5)]
BASELINE_MARKERS = ["s", "^", "D", "v"]
BASELINE_HATCHES = ["////", "....", "xxxx", "\\\\\\\\"]
BAND_ALPHA = 0.15         # uncertainty bands, in the series' own color
FILL_ALPHA = 0.12         # fill under a lone main-method line
METHOD_MARKER = "o"       # scatter / point marker of PPS and IP

# Fixed baseline order (guide: "Fix the order of baselines once"): index into
# BASELINE_GRAYS / _DASHES / _HATCHES / _MARKERS, so e.g. the defenseless run
# is the same light-gray (5, 3)-dashed line / "////" bar in every figure.
BASELINES = {"defenseless": 0, "base": 1, "random": 2, "off-trait": 3}
BASELINE_ALIASES = {"none": "defenseless", "undefended": "defenseless",
                    "base model": "base", "neutral": "off-trait",
                    "off_trait": "off-trait", "offtrait": "off-trait"}

# Non-data ink: zero / identity lines, reference markers, shaded regions.
REF = "#B0B0B0"           # zero lines, y = x diagonals (thin, solid)
SHADE = "#EEEEEE"         # light region shading (e.g. a "selective" zone)
SHADE_DARK = "#DDDDDD"    # second region shade

# Heatmaps of magnitude: cividis (Nunez, Anderton & Renslow 2018, PLoS ONE
# 13:e0199239, doi:10.1371/journal.pone.0199239), perceptually uniform and
# CVD-optimized, blue -> gray -> yellow. Replaced viridis on 2026-10-21 under
# the method-hued rule: viridis' violet and yellow-green ends read as PPS / IP
# (fig_behavior_row (a) sits beside the purple Delta-PPS map).
SEQUENTIAL_CMAP = "cividis"
# Gray magnitude map, white = low -> black = high (2026-10-21, for
# Figure 2a: "white is low to black is high, no yellow"). Achromatic, so it
# never reads as a method; fig_behavior_row (a) uses it beside the
# method-hued (b).
GRAY_SEQUENTIAL_CMAP = "Greys"

# Signed scales ("method-hued", chosen 2026-10-20). A signed map
# whose panel is about ONE method uses ``method_signed_cmap``: the method's
# improvement direction in its hue, the other direction gray. A signed map
# that is not about one method uses this neutral blue-orange map: purple and
# green never mean anything but PPS and IP. Poles checked with the dataviz
# validator (2026-10-21, light): #4393C3 vs #D47610 pass every check (CVD dE
# 21.3, normal 26.0, >= 3:1), and both clear CVD separation from PPS and IP
# (worst: IP green vs the orange pole, dE 8.0 protan). RdBu was rejected: its
# red pole is dE 2.5 from IP green for deuteranopes.
NEUTRAL_MID = "#F7F7F7"       # near-white zero of every signed map
NEUTRAL_DIVERGING = ("#4393C3", "#92C5DE", "#D1E5F0", NEUTRAL_MID,
                     "#FBD9B3", "#EFA85F", "#D47610")  # negative -> positive
NEUTRAL_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "neutral_diverging", NEUTRAL_DIVERGING)
DIVERGING_CMAP = NEUTRAL_DIVERGING_CMAP  # old name; never PRGn any more
SIGNED_GRAY_MAX = 0.7         # darkest Greys step of a method map's gray side

# ---- Sizes (inches): NeurIPS single-column template, \textwidth = 5.5 in ---
WIDTH_FULL = 5.5
WIDTH_HALF = 2.7
WIDTH_SINGLE, WIDTH_DOUBLE = WIDTH_HALF, WIDTH_FULL  # the guide's names
DEFAULT_ASPECT = 0.62

# ---- Type sizes (pt, at print size) ----------------------------------------
SIZE_BASE = 8.0           # body text, annotations
SIZE_LABEL = 8.5          # axis labels
SIZE_SMALL = 7.5          # ticks, legends, direct labels
SIZE_TITLE = 9.0          # panel titles / labels
SIZE_MIN = 7.0            # floor: in-cell numbers, dense notes

FONT_STACK = ["Nunito Sans", "Helvetica", "Arial", "DejaVu Sans"]


def installed_font_stack():
    """``FONT_STACK`` restricted to installed fonts (DejaVu Sans ships with
    matplotlib, so the list is never empty)."""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in FONT_STACK if f in names] or ["DejaVu Sans"]


def apply():
    """Set global rcParams. Call once at the top of every plotting entry.

    Starts from matplotlib's defaults, so a figure never inherits rc left
    behind by an earlier, differently styled figure in the same process.
    """
    mpl.rcdefaults()
    stack = installed_font_stack()
    mpl.rcParams.update({
        # the installed part of the stack as the family list, so a glyph
        # Nunito Sans lacks (Greek: epsilon) falls back to the next font
        # instead of rendering as a box (a generic "sans-serif" family
        # resolves to ONE font and has no fallback)
        "font.family": stack,
        "font.sans-serif": FONT_STACK,
        "mathtext.fontset": "custom",   # math letters in the text font
        "mathtext.rm": stack[0],
        "mathtext.it": f"{stack[0]}:italic",
        "mathtext.bf": f"{stack[0]}:bold",
        "mathtext.sf": stack[0],
        "mathtext.cal": stack[0],
        "mathtext.fallback": "stixsans",  # symbols/Greek it lacks
        "font.size": SIZE_BASE,
        "text.color": INK,
        "axes.labelsize": SIZE_LABEL,
        "axes.titlesize": SIZE_TITLE,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "xtick.labelsize": SIZE_SMALL,
        "ytick.labelsize": SIZE_SMALL,
        "legend.fontsize": SIZE_SMALL,
        "legend.title_fontsize": SIZE_SMALL,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.xmargin": 0.02,
        "axes.prop_cycle": cycler(color=[PPS, IP] + BASELINE_GRAYS[:2]),
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": TICK_INK,
        "ytick.labelcolor": TICK_INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.markersize": 4,
        "patch.linewidth": 0,
        "hatch.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,   # embed TrueType so text stays editable/searchable
        "ps.fonttype": 42,
        "image.cmap": SEQUENTIAL_CMAP,
    })


def despine(ax, offset=5):
    """Hide top/right spines and push left/bottom outward (the offset look)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", offset))


def width_in(width) -> float:
    """Print width in inches: "full" | "half" | "single" | "double" | inches.

    A manuscript ``width=0.6\\textwidth`` is ``0.6 * WIDTH_FULL``.
    """
    return {"full": WIDTH_FULL, "half": WIDTH_HALF, "single": WIDTH_SINGLE,
            "double": WIDTH_DOUBLE}.get(width, width)


def figure(width="full", aspect=DEFAULT_ASPECT, nrows=1, ncols=1, height=None,
           **kw):
    """Create a figure at final print width with styled axes.

    ``width``: "full" (\\textwidth), "half", the guide's "single"/"double", or
    inches.
    """
    w = width_in(width)
    h = height if height is not None else w * aspect * (
        nrows / ncols if ncols > 1 else nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    for ax in np.atleast_1d(axes).ravel():
        despine(ax)
    return fig, axes


def canvas(width="full", height=None, aspect=DEFAULT_ASPECT):
    """An empty figure at final print width, for gridspec / subfigure layouts
    (add axes with ``fig.add_subplot`` and call ``despine`` on each)."""
    w = width_in(width)
    h = height if height is not None else w * aspect
    return plt.figure(figsize=(w, h))


def method_color(method: str, variant: int = 0) -> str:
    """Color of a method ("pps" | "ip"); variant 0 = main setting."""
    return RAMPS[method][variant]


def sweep_colors(method: str, n: int):
    """``n`` light->dark colors in a method's hue, for a strength sweep."""
    cmap = plt.get_cmap(METHOD_CMAP[method])
    lo, hi = METHOD_CMAP_RANGE
    return [cmap(v) for v in np.linspace(lo, hi, n)]


def method_cmap(method: str):
    """A method's sweep colormap restricted to ``METHOD_CMAP_RANGE`` (for a
    colorbar that matches ``sweep_colors``)."""
    lo, hi = METHOD_CMAP_RANGE
    base = plt.get_cmap(METHOD_CMAP[method])
    return LinearSegmentedColormap.from_list(
        f"{method}_sweep", base(np.linspace(lo, hi, 256)))


def method_signed_cmap(method: str, good: str = "negative"):
    """Signed map for a panel about ONE method: the
    method's improvement direction (``good``: "negative", e.g. a reduction
    in trait expression, or "positive") in the method's sequential colormap,
    the opposite direction in gray, darker = larger magnitude, near-white at
    zero. Use with ``diverging_norm`` (symmetric limits)."""
    if good not in ("negative", "positive"):
        raise ValueError(f"good must be 'negative' or 'positive', not {good!r}")
    n = 128
    meth = plt.get_cmap(METHOD_CMAP[method])(np.linspace(1.0, 0.0, n))
    gray = plt.get_cmap("Greys")(np.linspace(0.0, SIGNED_GRAY_MAX, n))
    colors = (np.vstack([meth, gray]) if good == "negative"
              else np.vstack([gray[::-1], meth[::-1]]))
    return LinearSegmentedColormap.from_list(f"{method}_signed_{good}", colors)


def signed_sweep_cmap(method: str):
    """Line-plot variant of ``method_signed_cmap(method, "positive")`` for a
    SIGNED strength of one method (e.g. a steering coefficient from -5 to
    +5): grays below zero, the method's ramp above. The near-white middle is
    skipped because a near-white line is invisible on the page."""
    lo, hi = METHOD_CMAP_RANGE
    neg = plt.get_cmap("Greys")(np.linspace(0.9, lo, 128))
    pos = plt.get_cmap(METHOD_CMAP[method])(np.linspace(lo, hi, 128))
    return LinearSegmentedColormap.from_list(f"{method}_signed",
                                             np.vstack([neg, pos]))


def diverging_norm(values=None, vmax=None):
    """``Normalize(-v, v)``: the guide's zero-centred diverging scale, with
    ``v`` the largest absolute finite value unless ``vmax`` is given."""
    if vmax is None:
        a = np.abs(np.asarray(values, float))
        vmax = float(np.nanmax(a[np.isfinite(a)]))
    return Normalize(-vmax, vmax)


def plot_method(ax, x, y, method, label=None, variant=0, fill=False,
                fill_to=None, end_dot=True, lw=1.9, **kw):
    """A defense (PPS or IP): solid line in its color, end-dot, optional
    soft fill (only when it is the single main method in the panel)."""
    color = kw.pop("color", method_color(method, variant))
    x, y = np.asarray(x), np.asarray(y)
    (line,) = ax.plot(x, y, color=color, lw=lw, label=label or method.upper(),
                      zorder=4, **kw)
    if fill:
        base = fill_to if fill_to is not None else ax.get_ylim()[0]
        ax.fill_between(x, y, base, color=color, alpha=FILL_ALPHA, lw=0,
                        zorder=1)
    if end_dot and len(x):
        ax.plot(x[-1], y[-1], "o", ms=5, mfc=color, mec="white", mew=1.2,
                zorder=5, label="_nolegend_")
    return line


def plot_ours(ax, x, y, label="PPS", **kw):
    """The guide's ``plot_ours``: PPS with the soft fill."""
    kw.setdefault("fill", True)
    return plot_method(ax, x, y, "pps", label=label, **kw)


def baseline_index(i=0) -> int:
    """Index of a baseline: an int, a ``BASELINES`` name or an alias."""
    if isinstance(i, str):
        key = i.lower()
        key = BASELINE_ALIASES.get(key, key)
        if key not in BASELINES:
            raise KeyError(f"unknown baseline {i!r}; add it to BASELINES")
        return BASELINES[key]
    return int(i)


def baseline_color(i=0) -> str:
    """Gray of the i-th (or named) baseline."""
    return BASELINE_GRAYS[baseline_index(i) % len(BASELINE_GRAYS)]


def baseline_style(i=0):
    """Line kwargs of the i-th (or named) baseline, fixed across figures."""
    k = baseline_index(i)
    return dict(color=BASELINE_GRAYS[k % len(BASELINE_GRAYS)],
                dashes=BASELINE_DASHES[k % len(BASELINE_DASHES)])


def plot_baseline(ax, x, y, label, i=0, markers=False, lw=1.5, solid=False,
                  **kw):
    """The i-th (or named) baseline: gray + distinct dash pattern (+ optional
    marker). ``solid`` drops the dash, for a panel whose only gray line is
    told apart from the methods by its gray and its markers (e.g. Figure
    2d, 2026-10-21)."""
    k = baseline_index(i)
    style = dict(**baseline_style(k), lw=lw, label=label, zorder=3)
    if solid:
        style["dashes"] = ()  # an empty dash sequence = solid line
    if markers:
        style.update(marker=BASELINE_MARKERS[k % len(BASELINE_MARKERS)],
                     markevery=max(1, len(x) // 5), ms=3.5)
    style.update(kw)
    (line,) = ax.plot(x, y, **style)
    return line


def method_control(role):
    """``(method, baseline)`` of a method-control role "method:baseline"
    (e.g. "pps:random" = PPS with a random vector), else None."""
    if not (isinstance(role, str) and ":" in role):
        return None
    method, base = role.split(":", 1)
    if method not in METHOD:
        raise KeyError(f"unknown method {method!r} in role {role!r}")
    baseline_index(base)  # an unknown baseline fails here, not at draw time
    return method, base


def role_color(role, variant=0) -> str:
    """Color of a role: a method ("pps" | "ip", ramp ``variant``), a method
    control ("pps:random": the method's color) or a baseline (int or
    name)."""
    ctrl = method_control(role)
    if ctrl:
        return method_color(ctrl[0], variant)
    if isinstance(role, str) and role in METHOD:
        return method_color(role, variant)
    return baseline_color(role)


def bar_style(role, variant=0) -> dict:
    """``ax.bar`` kwargs of a role: a method fills with its color (ramp
    ``variant``); a baseline is its gray with its hatch (white hatch lines,
    so it still reads in grayscale).

    A method control ("pps:random", "ip:off-trait") is that baseline's
    hatch on the method's color: the hatch says
    control, the color says which defense. It takes the method's main
    color, not a lighter variant: the mid variants fail the dataviz
    validator beside the baseline grays (PPS_MID vs #8C8C8C normal-vision
    dE 10.3 < 15; IP_MID 2.67:1 contrast)."""
    ctrl = method_control(role)
    if ctrl:
        method, base = ctrl
        k = baseline_index(base)
        return dict(color=method_color(method, variant),
                    hatch=BASELINE_HATCHES[k % len(BASELINE_HATCHES)],
                    edgecolor="white")
    if isinstance(role, str) and role in METHOD:
        return dict(color=method_color(role, variant), edgecolor="none")
    k = baseline_index(role)
    return dict(color=BASELINE_GRAYS[k % len(BASELINE_GRAYS)],
                hatch=BASELINE_HATCHES[k % len(BASELINE_HATCHES)],
                edgecolor="white")


def scatter_style(role, variant=0) -> dict:
    """``ax.scatter`` kwargs of a role: methods are filled circles in their
    color; a baseline is its gray in its own marker shape."""
    if isinstance(role, str) and role in METHOD:
        return dict(color=method_color(role, variant), marker=METHOD_MARKER,
                    linewidths=0)
    k = baseline_index(role)
    return dict(color=BASELINE_GRAYS[k % len(BASELINE_GRAYS)],
                marker=BASELINE_MARKERS[k % len(BASELINE_MARKERS)],
                linewidths=0)


def band(ax, x, lo, hi, color):
    """Uncertainty band in the series' own color."""
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=2)


def errorbars(ax, x, y, yerr=None, xerr=None, **kw):
    """Whiskers only (no line, no marker), in ink, thin, small caps."""
    style = dict(fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.0,
                 capthick=0.8, zorder=5)
    style.update(kw)
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, **style)


def ref_line(ax, y=None, x=None, **kw):
    """Thin non-data reference line: ``y=`` horizontal, ``x=`` vertical
    (zero lines, the main setting's epsilon, ...). Dotted unless overridden,
    so it is never mistaken for a dashed baseline."""
    style = dict(color=AXIS, lw=0.7, ls=(0, (1, 2)), zorder=1)
    style.update(kw)
    if y is not None:
        return ax.axhline(y, **style)
    return ax.axvline(x, **style)


def zero_line(ax, axis="y", **kw):
    """Light solid line at 0 (``axis="y"``: horizontal)."""
    style = dict(color=REF, lw=0.6, ls="-", zorder=1)
    style.update(kw)
    return ax.axhline(0, **style) if axis == "y" else ax.axvline(0, **style)


def panel_label(ax, letter: str, title: str = "", loc: str = "left"):
    """Axes title "(a)" or "(a) title" (the guide's panel label; keep
    ``title`` to what identifies the panel, not the caption). Left-aligned;
    ``loc="center"`` centres it over a narrow panel whose title would
    otherwise overhang its right edge."""
    return ax.set_title(f"({letter})" + (f" {title}" if title else ""),
                        loc=loc)


def span_xlabel(ax_first, ax_last, text: str):
    """One x label centred under the axes from ``ax_first`` to ``ax_last``
    (a row of heatmaps sharing an x variable), just below ``ax_first``'s
    tick labels, in the axis-label font.

    Drawn as an annotation kept out of the constrained layout, so it moves
    no axes. An axis label cannot do this: matplotlib (3.10, ``Axis.
    get_tightbbox``) counts a visible axis label in its axes' layout box
    whatever its ``in_layout``, so an axis label moved under its
    neighbours pads its own axes' margin up to the label's centre. The
    label needs room below the row (e.g. a neighbour's rotated tick labels
    reserve it), since the layout does not make any."""
    def span(renderer):  # the axes' union, in display units
        return Bbox.union([ax.get_window_extent(renderer)
                           for ax in (ax_first, ax_last)])

    def ticks(renderer):  # ax_first's tick labels (it has no axis label)
        return ax_first.xaxis.get_tightbbox(renderer)

    rc = mpl.rcParams
    ann = ax_first.annotate(
        text, xy=(0.5, 0), xycoords=(span, ticks),
        xytext=(0, -rc["axes.labelpad"]), textcoords="offset points",
        ha="center", va="top", fontsize=rc["axes.labelsize"],
        color=rc["axes.labelcolor"], annotation_clip=False)
    ann.set_in_layout(False)
    return ann


def legend_above(ax, handles=None, labels=None, ncols=None, **kw):
    """Frameless one-row legend just above the axes (for bars, or lines that
    end too close together to label directly)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    style = dict(ncols=ncols or len(handles), loc="lower left",
                 bbox_to_anchor=(0, 1.0), borderaxespad=0.2, frameon=False,
                 handletextpad=0.4)
    style.update(kw)
    return ax.legend(handles, labels, **style)


def fig_legend(fig, handles, labels, ncols=None, loc="outside upper center",
               **kw):
    """Frameless one-row legend for the whole figure (constrained layout
    reserves its space; ``loc="outside lower center"`` puts it below)."""
    style = dict(ncols=ncols or len(handles), loc=loc, frameon=False,
                 handletextpad=0.4)
    style.update(kw)
    return fig.legend(handles, labels, **style)


def direct_labels(ax, pad=5, min_gap_pt=9):
    """Label each line at its right end in its own colour (replaces a legend).

    Call after all lines are plotted and axis limits are final.
    """
    fig = ax.figure
    items = []
    for line in ax.get_lines():
        lab = line.get_label()
        if lab.startswith("_"):
            continue
        xd, yd = line.get_xdata(), line.get_ydata()
        if len(xd) == 0:
            continue
        x_end, y_end = xd[-1], yd[-1]
        y_px = ax.transData.transform((x_end, y_end))[1]
        items.append([y_px * 72 / fig.dpi, x_end, y_end, lab,
                      mpl.colors.to_hex(line.get_color())])
    items.sort(key=lambda t: t[0])
    placed = []
    for y_pt, *_ in items:
        placed.append(max(y_pt, placed[-1] + min_gap_pt) if placed else y_pt)
    methods = {c.upper() for c in PPS_RAMP + IP_RAMP}
    for (y_pt, x_end, y_end, lab, color), y_new in zip(items, placed):
        ax.annotate(lab, xy=(x_end, y_end), xytext=(pad, y_new - y_pt),
                    textcoords="offset points", va="center", ha="left",
                    color=color, annotation_clip=False,
                    fontweight="bold" if color.upper() in methods else "normal")


def bars(ax, categories, series, roles=None, width=0.8, legend=True):
    """Grouped bars. ``series`` is {name: values}; ``roles`` maps a name to
    "pps" | "ip" (method color) or a baseline name; every other series is a
    baseline in order (grays + hatching). Adds a small legend above the
    axes."""
    roles = roles or {}
    n = len(series)
    bw = width / n
    xs = np.arange(len(categories))
    gi = 0
    for k, (name, vals) in enumerate(series.items()):
        pos = xs - width / 2 + bw * (k + 0.5)
        role = roles.get(name)
        if role is None:
            role, gi = gi, gi + 1
        ax.bar(pos, vals, bw * 0.9, label=name, zorder=3, **bar_style(role))
    ax.set_xticks(xs, categories)
    ax.tick_params(axis="x", length=0)
    if legend:
        ax.legend(ncols=n, loc="lower left", bbox_to_anchor=(0, 1.0),
                  handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)


def heatmap(ax, mat, rows, cols, cmap=SEQUENTIAL_CMAP, norm=None,
            fmt="{:.1f}", fontsize=SIZE_MIN, annotate=True):
    """Annotated heatmap (no spines, no ticks); each number is ink on light
    cells and white on dark ones. Returns the image (for a colorbar).

    Undoes ``despine``'s outward offset (else tick labels float away from
    the cells); done first because moving a spine resets that axis's ticks.
    """
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 0))
    mat = np.asarray(mat, float)
    cmap = plt.get_cmap(cmap)
    norm = norm or Normalize(np.nanmin(mat), np.nanmax(mat))
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b = cmap(norm(v))[:3]
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, fmt.format(v).replace("-", "−"),
                        ha="center", va="center", fontsize=fontsize,
                        color=INK if lum > 0.45 else "white")
    return im


def colorbar(fig, mappable, ax, label=None, **kw):
    """Slim borderless colorbar."""
    style = dict(shrink=0.9, aspect=25, pad=0.02)
    style.update(kw)
    cb = fig.colorbar(mappable, ax=ax, **style)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.6)
    if label:
        cb.set_label(label)
    return cb


def save(fig, path, formats=("pdf", "png")):
    """Save as vector PDF (for the paper) and PNG (for previews/slides).

    The PDF omits its creation date (matplotlib embeds the current time
    otherwise), so re-rendering unchanged data gives byte-identical files.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for ext in formats:
        meta = {"CreationDate": None} if ext == "pdf" else None
        fig.savefig(f"{path}.{ext}", metadata=meta)
    plt.close(fig)
```

## Usage examples

```python
import numpy as np
import paper_style as ps

ps.apply()

# Line plot with direct labels
x = np.linspace(0, 10, 41)
fig, ax = ps.figure("single")
ax.set_ylim(0, 1)                                   # set limits before plot_ours
ps.plot_baseline(ax, x, 0.62 * (1 - np.exp(-x / 4.5)), "Ablation", i=1)
ps.plot_baseline(ax, x, 0.78 * (1 - np.exp(-x / 3.2)), "Baseline", i=0)
ps.plot_ours(ax, x, 0.92 * (1 - np.exp(-x / 2)))
ax.set_xlabel("Training steps (k)")
ax.set_ylabel("Accuracy")
ps.direct_labels(ax)
ps.save(fig, "figures/accuracy")

# Grouped bars
fig, ax = ps.figure("single")
ps.bars(ax, ["A", "B", "C"],
        {"Ours": [.88, .81, .93], "Baseline": [.71, .66, .74], "Ablation": [.55, .60, .52]})
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ps.save(fig, "figures/scores")

# Signed heatmap NOT about one method (e.g. cosines): neutral blue-orange
data = np.random.default_rng(0).normal(size=(8, 8))
fig, ax = ps.figure("single", aspect=0.8)
im = ps.heatmap(ax, data, list("abcdefgh"), list("abcdefgh"),
                ps.NEUTRAL_DIVERGING_CMAP, ps.diverging_norm(data), "{:+.1f}")
ps.colorbar(fig, im, ax)
ps.save(fig, "figures/heatmap")

# Signed heatmap about ONE method (e.g. PPS minus defenseless): method-hued
delta = -np.abs(data) * 20          # reductions = the defense's improvement
fig, ax = ps.figure("single", aspect=0.8)
ps.heatmap(ax, delta, list("abcdefgh"), list("abcdefgh"),
           ps.method_signed_cmap("pps", good="negative"),
           ps.diverging_norm(delta), "{:+.0f}")
ps.save(fig, "figures/heatmap_delta_pps")
```

## Checklist before committing a figure

- [ ] PPS is purple, IP is green, everything else is gray.
- [ ] Each baseline has the same gray, dash, and hatch as in every other figure.
- [ ] No top/right spines, no gridlines, no legend box border.
- [ ] Direct labels don't overlap each other or the data, and aren't clipped.
- [ ] Text is readable at print size (no font below 7 pt) and in Nunito Sans (or the agreed fallback). One agreed exception (2026-11-05): Figure 3's 5.5 pt axis-direction words (`gradient_step0.SIGN_PT`).
- [ ] Figure was created at final width; PDF and PNG were both regenerated.
- [ ] The figure still makes sense printed in grayscale.

## Sources for the color choices

- The two method colors are the poles of Paul Tol's PRGn scheme (Tol, "Colour Schemes", SRON technical note SRON/EPS/TN/09-002): https://personal.sron.nl/~pault/. PRGn itself is no longer used as a diverging map, because its poles mean PPS and IP.
- The neutral diverging poles `#4393C3` (ColorBrewer RdBu) and `#D47610` (a darkened PuOr orange) were chosen with the dataviz validator (`validate_palette.js`, 2026-10-21). Its numbers are quoted in the two-method section above.
- `cividis` (magnitude heatmaps) is a perceptually uniform colormap optimized for color-vision deficiency: Nuñez, Anderton & Renslow (2018), "Optimizing colormaps with consideration for color vision deficiency to enable accurate interpretation of scientific data", PLoS ONE 13(7): e0199239, https://doi.org/10.1371/journal.pone.0199239
- `viridis` (used until 2026-10-21) is matplotlib's former default: https://bids.github.io/colormap/
- Background on why rainbow colormaps should be avoided: Crameri, Shephard & Heron (2020), "The misuse of colour in science communication", Nature Communications 11, 5444: https://doi.org/10.1038/s41467-020-19160-7
