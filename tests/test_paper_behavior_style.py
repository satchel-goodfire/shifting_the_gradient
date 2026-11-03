"""Drawing helpers of the behavior figures (paper_style migration)."""

import numpy as np
from matplotlib.colors import to_hex

from grad_shift.paper import paper_style as ps
from grad_shift.paper.macros import Result
from grad_shift.paper.products.behavior_capscheck import place_point_labels
from grad_shift.paper.products.behavior_common import bar_panel
from grad_shift.paper.products.behavior_specificity import (
    GROUPS, SINGLE_BARS, group_brackets, rotate_xticklabels)


def _fig():
    ps.apply()
    return ps.figure("half")


def test_point_labels_stay_inside_and_apart():
    fig, ax = _fig()
    xs = np.array([2.0, 3.0, 50.0, 52.0, 98.0, 99.0])
    ys = np.array([0.0, 0.5, 30.0, 33.0, 80.0, 82.0])
    ax.plot(xs, ys, "o-")
    ax.set_xlim(-3, 112)
    ax.set_ylim(-12, 100)
    items = [(x, y, f"n={i}", ps.IP if i % 2 else ps.PPS)
             for i, (x, y) in enumerate(zip(xs, ys))]
    # two same-color neighbours closer than merge_pt share a label
    items[1] = (items[1][0], items[1][1], "n=1", ps.PPS)
    place_point_labels(ax, items)
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts]
    assert any(t.get_text() == "n=0, 1" for t in ax.texts)
    inside = ax.get_window_extent(r)
    for i, b in enumerate(boxes):
        assert inside.contains(b.x0, b.y0) and inside.contains(b.x1, b.y1)
        assert not any(b.overlaps(o) for o in boxes[i + 1:])
    ps.plt.close(fig)


def test_point_labels_keep_a_gap():
    # the second point sits one label width + 0.5 pt right of the first, so
    # its first-choice spot would sit 0.5 pt from the first label
    # (fig_capstune panel (f) before the fix): labels keep >= 1 pt of air
    fig, ax = _fig()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.draw_without_rendering()
    r = fig.canvas.get_renderer()
    probe = ax.annotate("ε=0.5", (20, 50), fontsize=ps.SIZE_MIN)
    w = probe.get_window_extent(r).width
    probe.remove()
    p1 = ax.transData.transform((20, 50))
    x2 = ax.transData.inverted().transform(
        (p1[0] + w + 0.5 * fig.dpi / 72, p1[1]))[0]
    ax.plot([20, x2], [50, 50], "o", ls="none")
    place_point_labels(ax, [(20, 50, "ε=0.5", ps.PPS),
                            (x2, 50, "n=1", ps.IP)])
    a, b = (t.get_window_extent(r) for t in ax.texts)
    assert not a.padded(fig.dpi / 72).overlaps(b)
    ps.plt.close(fig)


def _sweep(shift=0.0):
    pt = lambda c, e: {"caps": c, "evil": Result(e, sem=1.0)}  # noqa: E731
    return {"ip": {n: pt(c + shift, e) for n, c, e in
                   ((1, 90, 60), (2, 70, 45), (3, 52, 30), (5, 30, 20))},
            "pps": {e: pt(c + shift, v) for e, c, v in
                    ((0.5, 95, 80), (1, 80, 60), (2, 55, 32), (4, 20, 8))},
            "none": pt(97, 91)}


def test_caps_tradeoff_labels_are_placed_on_the_final_layout():
    # fig_caps_tradeoff: labels placed before the second panel, the y label
    # and the legend moved the axes ended up off their points / outside
    from grad_shift.paper.products.behavior_capscheck import tradeoff_figure
    ps.apply()
    fig, axes = tradeoff_figure({"heldout": _sweep(), "extract": _sweep(2.0)},
                                (3, 2.0))
    fig.draw_without_rendering()  # what saving does
    r = fig.canvas.get_renderer()
    for ax in axes:
        texts = [t for t in ax.texts if t.get_text().startswith(("n=", "ε="))]
        assert texts
        inside = ax.get_window_extent(r)
        boxes = [t.get_window_extent(r) for t in texts]
        for i, b in enumerate(boxes):
            assert inside.contains(b.x0, b.y0) and inside.contains(b.x1, b.y1)
            assert not any(b.overlaps(o) for o in boxes[i + 1:])
    ps.plt.close(fig)


def test_bar_roles_brackets_and_rotation():
    # every bar has a paper_style role: base model and defenseless are gray
    # baselines; a defense's controls are method controls (its color, the
    # control's hatch)
    roles = {key: role for key, _, _, role in SINGLE_BARS}
    assert roles["pps"] == "pps" and roles["ip"] == "ip"
    assert {roles["base"], roles["none"]} <= set(ps.BASELINES)
    assert ps.method_control(roles["pps_rand"]) == ("pps", "random")
    assert ps.method_control(roles["pps_syco"]) == ("pps", "off-trait")
    fig, ax = _fig()
    keys = ["base", "none", "pps", "pps_rand", "pps_syco", "ip", "ip_syco",
            "ip_neut", "ip_rand"]
    bar_panel(ax, keys, [Result(50.0)] * len(keys),
              [roles.get(k, "ip:off-trait") for k in keys], tick_rotation=55)
    rotate_xticklabels(ax, 55)  # bar_panel's despine resets the rotation
    assert {t.get_rotation() for t in ax.get_xticklabels()} == {55.0}
    group_brackets(ax, keys, 103)
    colors = {to_hex(t.get_color()): t.get_text() for t in ax.texts}
    assert colors[ps.PPS.lower()] == "PPS" and colors[ps.IP.lower()] == "IP"
    assert set(GROUPS) == {"pps", "ip"}
    ps.plt.close(fig)
