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
