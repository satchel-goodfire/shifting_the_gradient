"""Figure style: reproducible output and the paper_style role helpers."""

import time

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgb

from grad_shift.paper import paper_style as ps
from grad_shift.paper.products import style


def _render(stem):
    style.apply_rc()
    fig, ax = style.plt.subplots(figsize=(3, 2))
    ax.plot([0, 1, 2], [1, 0, 2])
    style.save(fig, str(stem))


def _render_ps(stem):
    ps.apply()
    fig, ax = ps.figure("half")
    ps.plot_method(ax, [0, 1, 2], [1, 0, 2], "pps")
    ps.save(fig, str(stem))


@pytest.mark.parametrize("render", [_render, _render_ps])
def test_same_figure_saves_identical_pdf_and_png(tmp_path, render):
    render(tmp_path / "a")
    time.sleep(1.1)  # a creation timestamp would differ across seconds
    render(tmp_path / "b")
    for ext in (".pdf", ".png"):
        assert (tmp_path / f"a{ext}").read_bytes() == \
            (tmp_path / f"b{ext}").read_bytes(), ext


def test_apply_resets_rc_left_by_another_style():
    mpl.rcParams["text.color"] = "red"
    mpl.rcParams["axes.grid"] = True
    ps.apply()
    assert to_hex(mpl.rcParams["text.color"]) == ps.INK.lower()
    assert mpl.rcParams["axes.grid"] is False
    assert mpl.rcParams["pdf.fonttype"] == 42
    stack = mpl.rcParams["font.family"]
    assert stack and set(stack) <= set(ps.FONT_STACK)


def test_baselines_keep_one_style_by_name_or_alias():
    assert ps.baseline_index("defenseless") == 0
    assert ps.baseline_index("undefended") == ps.baseline_index("none") == 0
    assert ps.baseline_index("neutral") == ps.BASELINES["off-trait"]
    assert ps.baseline_style("random") == ps.baseline_style(2)
    with pytest.raises(KeyError):
        ps.baseline_index("ablation")
    # each named baseline differs from every other in dash or gray
    looks = {(ps.baseline_color(k), ps.BASELINE_DASHES[i])
             for k, i in ps.BASELINES.items()}
    assert len(looks) == len(ps.BASELINES)


def test_bar_and_scatter_styles_follow_roles():
    assert ps.bar_style("pps")["color"] == ps.PPS
    assert ps.bar_style("ip", variant=1)["color"] == ps.IP_MID
    assert "hatch" not in ps.bar_style("pps")
    b = ps.bar_style("defenseless")
    assert b["color"] in ps.BASELINE_GRAYS and b["hatch"]
    assert ps.bar_style("random")["hatch"] != b["hatch"]
    assert ps.scatter_style("ip")["marker"] == ps.METHOD_MARKER
    assert ps.scatter_style("off-trait")["marker"] != ps.METHOD_MARKER
    assert ps.role_color("pps") == ps.PPS
    assert ps.role_color("base") == ps.baseline_color("base")


def test_method_control_is_the_baseline_hatch_in_the_method_color():
    # 2026-10-21 (Figure 2c): a defense's control bar is hatched like
    # its baseline but colored like its defense
    for method, base in (("pps", "random"), ("ip", "off-trait")):
        st = ps.bar_style(f"{method}:{base}")
        assert st["color"] == ps.method_color(method)
        assert st["hatch"] == ps.bar_style(base)["hatch"]
        assert st["edgecolor"] == "white"  # hatch lines, as on gray bars
        assert ps.role_color(f"{method}:{base}") == ps.method_color(method)
    assert ps.method_control("pps") is None
    assert ps.method_control("defenseless") is None
    with pytest.raises(KeyError):
        ps.method_control("steer:random")
    with pytest.raises(KeyError):
        ps.method_control("pps:ablation")


def test_plot_baseline_solid_drops_only_the_dash():
    ps.apply()
    fig, ax = ps.figure("half")
    dashed = ps.plot_baseline(ax, [0, 1], [0, 1], "d", i="defenseless")
    solid = ps.plot_baseline(ax, [0, 1], [0, 1], "s", i="defenseless",
                             solid=True)
    assert dashed.get_linestyle() == "--"
    assert solid.get_linestyle() == "-"
    assert solid.get_color() == dashed.get_color()
    ps.plt.close(fig)


def test_gray_sequential_cmap_runs_white_to_black_without_hue():
    # 2026-10-21 (Figure 2a): white = low, black = high, no yellow
    cm = ps.plt.get_cmap(ps.GRAY_SEQUENTIAL_CMAP)
    samples = [cm(v)[:3] for v in np.linspace(0, 1, 101)]
    assert all(_is_gray(c) for c in samples)
    assert all(c > 0.97 for c in samples[0]) and all(c < 0.03 for c in
                                                     samples[-1])
    lum = [sum(c) for c in samples]
    assert all(a >= b for a, b in zip(lum, lum[1:]))  # monotone darker


def _three_maps(label):
    """Three equal heatmaps in a constrained-layout row, laid out; ``label``
    is called with the first and last axes before layout."""
    ps.apply()
    fig = ps.canvas("half", height=1.2)
    gs = fig.add_gridspec(1, 3)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in axes:
        ps.heatmap(ax, [[0.0, 1.0]], ["r"], ["lo", "hi"], fmt="{:.0f}")
    label(axes[0], axes[-1])
    fig.draw_without_rendering()
    return fig, axes


def test_span_xlabel_centres_under_the_row_and_moves_no_axes():
    # An axis label moved under its neighbours pads its own axes' margin
    # (matplotlib counts it whatever its in_layout); Figure 2's old centred
    # label set the (a)-(b) gap that way by accident.
    ref, ref_axes = _three_maps(lambda a, b: None)
    fig, axes = _three_maps(
        lambda a, b: ps.span_xlabel(a, b, "post-finetuning data"))
    for ax, r in zip(axes, ref_axes):
        assert np.allclose(ax.get_position().bounds,
                           r.get_position().bounds)
    r = fig.canvas.get_renderer()
    (ann,) = axes[0].texts[-1:]
    bb = ann.get_window_extent(r)
    lo, hi = (axes[0].get_window_extent(r).x0,
              axes[-1].get_window_extent(r).x1)
    assert (bb.x0 + bb.x1) / 2 == pytest.approx((lo + hi) / 2, abs=0.5)
    ticks = axes[0].xaxis.get_tightbbox(r)
    assert bb.y1 < ticks.y0  # below the tick labels, not over them
    assert ann.get_fontsize() == mpl.rcParams["axes.labelsize"]
    ps.plt.close(ref)
    ps.plt.close(fig)


def test_colormaps_and_norms():
    n = ps.diverging_norm([-0.2, 0.5, np.nan])
    assert (n.vmin, n.vmax) == (-0.5, 0.5)
    assert ps.diverging_norm(vmax=3).vmin == -3
    # the sweep colormap spans exactly the sweep_colors range
    cm, cols = ps.method_cmap("ip"), ps.sweep_colors("ip", 2)
    assert np.allclose(cm(0.0)[:3], to_rgb(cols[0]), atol=0.01)
    assert np.allclose(cm(1.0)[:3], to_rgb(cols[-1]), atol=0.01)
    # signed sweep: negative side neutral gray, positive side the method hue
    s = ps.signed_sweep_cmap("pps")
    r, g, b = s(0.1)[:3]
    assert r == pytest.approx(g, abs=1e-6) == pytest.approx(b, abs=1e-6)
    r, g, b = s(0.95)[:3]
    assert b > g and r > g  # purple


def _hue_deg(rgb):
    import colorsys
    return 360 * colorsys.rgb_to_hsv(*rgb[:3])[0]


def _is_gray(rgb, tol=0.02):
    return max(rgb[:3]) - min(rgb[:3]) <= tol


def _purple_or_green(rgb):
    """Chromatic and in the PPS-purple or IP-green hue range."""
    if max(rgb[:3]) - min(rgb[:3]) < 0.06:  # near-white / gray: no hue
        return False
    h = _hue_deg(rgb)
    return 75 <= h <= 165 or 260 <= h <= 330


@pytest.mark.parametrize("method,hue_ok", [
    ("pps", lambda r, g, b: r > g and b > g),   # purple
    ("ip", lambda r, g, b: g > r and g > b)])   # green
def test_method_signed_cmap_is_method_hued_with_gray_other_side(method,
                                                                hue_ok):
    # 2026-10-20 ("method-hued"): the method's improvement direction
    # in its hue, the opposite direction gray, near-white at zero
    cm = ps.method_signed_cmap(method, good="negative")
    assert all(c > 0.95 for c in cm(0.5)[:3])  # near-white midpoint
    assert hue_ok(*cm(0.0)[:3])                # most negative = method
    assert _is_gray(cm(1.0)) and max(cm(1.0)[:3]) < 0.6  # dark-ish gray
    lum = [sum(cm(v)[:3]) for v in (0.0, 0.25, 0.5)]
    assert lum[0] < lum[1] < lum[2]            # darker = larger magnitude
    flipped = ps.method_signed_cmap(method, good="positive")
    assert hue_ok(*flipped(1.0)[:3]) and _is_gray(flipped(0.0))
    with pytest.raises(ValueError):
        ps.method_signed_cmap(method, good="up")


def test_magnitude_cmap_has_no_method_hue():
    # viridis' violet and yellow-green ends read as PPS/IP next to a
    # method-hued panel (fig_behavior_row (a) beside (b))
    cm = ps.plt.get_cmap(ps.SEQUENTIAL_CMAP)
    assert not any(_purple_or_green(cm(v)) for v in np.linspace(0, 1, 101))


def test_neutral_diverging_cmap_has_no_method_hue():
    cm = ps.NEUTRAL_DIVERGING_CMAP
    assert ps.DIVERGING_CMAP is cm  # the old name no longer means PRGn
    samples = [cm(v) for v in np.linspace(0, 1, 101)]
    assert not any(_purple_or_green(c) for c in samples)
    assert all(c > 0.95 for c in cm(0.5)[:3])  # near-white midpoint
    # the validated poles (dataviz validator, 2026-10-21)
    assert to_hex(cm(0.0)) == ps.NEUTRAL_DIVERGING[0].lower()
    assert to_hex(cm(1.0)) == ps.NEUTRAL_DIVERGING[-1].lower()


def test_heatmap_flips_text_on_dark_cells_and_panel_label():
    ps.apply()
    fig, ax = ps.figure("half")  # despined: spines pushed outward
    ps.heatmap(ax, [[0.0, 1.0]], ["r"], ["lo", "hi"], cmap="viridis",
               fmt="{:.1f}")
    txt = {t.get_text(): to_hex(t.get_color()) for t in ax.texts}
    assert txt["0.0"] == "#ffffff"  # viridis low end is dark
    assert txt["1.0"] == ps.INK.lower()
    # tick labels sit against the cells, not at the despine offset
    for side in ("left", "bottom"):
        assert ax.spines[side].get_position() == ("outward", 0)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["lo", "hi"]
    assert ps.panel_label(ax, "b", "IP").get_text() == "(b) IP"
    # letter=None (2026-11-06, Figure 14): the title alone, and a blank
    # title (not none) so the layout keeps the lettered figure's slot
    assert ps.panel_label(ax, None, "IP").get_text() == "IP"
    assert ps.panel_label(ax, None).get_text() == " "
    ps.plt.close(fig)


def test_bar_panel_keeps_rotation_roles_and_visible_dots():
    from grad_shift.paper.macros import Result
    from grad_shift.paper.products.behavior_common import bar_panel
    ps.apply()
    fig, ax = ps.figure("half")
    roles = ["defenseless", "pps", ("ip", 1), "off-trait"]
    bar_panel(ax, ["a", "b", "c", "d"], [Result(50.0, sem=2.0)] * 4, roles,
              arm_points=[None, None, None, [40.0, 60.0]], tick_rotation=55,
              coherence=[90, 90, 90, 40])
    assert {t.get_rotation() for t in ax.get_xticklabels()} == {55.0}
    faces = [to_hex(p.get_facecolor()) for p in ax.patches]
    assert faces[:3] == [ps.baseline_color("defenseless").lower(),
                         ps.PPS.lower(), ps.IP_MID.lower()]
    assert faces[3] == "#ffffff"  # low coherence: hollow
    assert any(t.get_text() == "coh 40" for t in ax.texts)
    # per-arm dots are outlined so they show on a dark-gray bar
    dots = ax.collections[-1]
    assert to_hex(dots.get_edgecolor()[0]) == "#ffffff"
    assert dots.get_linewidths()[0] > 0
    ps.plt.close(fig)
