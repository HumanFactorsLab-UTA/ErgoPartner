"""The shared chart visual system: palette gates, band meanings, layout rules."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from dashboard.viz_theme import (
    ALL_PAIRS_SERIES_CAP,
    SEQUENTIAL_BLUE,
    SERIES,
    STATUS,
    apply_layout,
    empty_figure,
    line,
    risk_colour,
    risk_label,
)


def test_series_slots_are_distinct_and_fixed():
    """Assigned in order, never cycled, never generated."""

    assert len(SERIES) == len(set(SERIES))
    assert SERIES[0] == "#3987e5"


def test_status_colours_are_never_series_colours():
    """A status hue must not impersonate a series."""

    assert not set(STATUS.values()) & set(SERIES)


def test_sequential_ramp_is_one_hue_and_monotonic():
    """Sequential means light to dark in a single hue, never a rainbow."""

    def luminance(value: str) -> float:
        v = value.lstrip("#")
        r, g, b = (int(v[i : i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    steps = [luminance(colour) for colour in SEQUENTIAL_BLUE]
    assert steps == sorted(steps, reverse=True)


def test_all_pairs_forms_cap_at_three_series():
    """Only the first three slots clear the all-pairs floors on this surface."""

    assert ALL_PAIRS_SERIES_CAP == 3
    assert ALL_PAIRS_SERIES_CAP < len(SERIES)


# ----------------------------------------------------------------------
# Risk bands
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "score", "expected"),
    [
        ("rula", 1, "good"),
        ("rula", 2, "good"),
        ("rula", 3, "warning"),
        ("rula", 4, "warning"),
        ("rula", 5, "serious"),
        ("rula", 6, "serious"),
        ("rula", 7, "critical"),
        ("reba", 1, "good"),
        ("reba", 3, "good"),
        ("reba", 4, "warning"),
        ("reba", 7, "warning"),
        ("reba", 8, "serious"),
        ("reba", 10, "serious"),
        ("reba", 11, "critical"),
        ("reba", 15, "critical"),
    ],
)
def test_risk_bands_follow_the_published_action_levels(method: str, score: int, expected: str):
    """Bands are the method's own action levels, not an even split of the scale."""

    assert risk_colour(score, method) == STATUS[expected]


def test_every_band_carries_a_label_so_colour_is_never_alone():
    for method, score in (("rula", 1), ("rula", 7), ("reba", 1), ("reba", 13)):
        assert risk_label(score, method) not in ("", None)


def test_a_missing_score_is_neither_good_nor_bad():
    assert risk_colour(None, "rula") not in STATUS.values()
    assert risk_label(None, "reba") == "No score"


# ----------------------------------------------------------------------
# Layout rules
# ----------------------------------------------------------------------


def test_a_single_series_gets_no_legend():
    """The title already names it; a legend box would be noise."""

    figure = apply_layout(go.Figure(), title="Trunk flexion", series_count=1)

    assert figure.layout.showlegend is False


def test_two_or_more_series_always_get_a_legend():
    """Identity must never rest on colour alone."""

    figure = apply_layout(go.Figure(), title="RULA and REBA", series_count=2)

    assert figure.layout.showlegend is True


def test_the_legend_sits_above_the_plot_not_over_it():
    """Legends drawn inside the plot area were covering the data."""

    figure = apply_layout(go.Figure(), series_count=3)

    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.y >= 1.0


def test_layout_leaves_room_so_labels_cannot_collide():
    figure = apply_layout(go.Figure(), title="Something", x_title="Time", y_title="Score")

    assert figure.layout.margin.t >= 56
    assert figure.layout.xaxis.automargin is True
    assert figure.layout.yaxis.automargin is True


def test_chrome_is_recessive_and_transparent():
    """The data is the only saturated thing, and the app's surface shows through."""

    figure = apply_layout(go.Figure())

    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert figure.layout.xaxis.zeroline is False


def test_lines_are_two_pixels_and_take_slots_in_order():
    figure = go.Figure()
    line(figure, [0, 1], [1, 2], name="First", slot=0)
    line(figure, [0, 1], [2, 3], name="Second", slot=1)

    assert figure.data[0].line.color == SERIES[0]
    assert figure.data[1].line.color == SERIES[1]
    assert figure.data[0].line.width == 2


def test_every_line_ships_a_hover_layer():
    figure = go.Figure()
    line(figure, [0, 1], [1, 2], name="Trunk", hover_suffix="°")

    assert "%{y:.1f}°" in figure.data[0].hovertemplate


def test_a_filled_line_uses_a_translucent_fill_not_a_solid_block():
    figure = go.Figure()
    line(figure, [0, 1], [1, 2], name="Trunk", slot=0, fill=True)

    assert figure.data[0].fillcolor.startswith("rgba(")
    assert figure.data[0].fillcolor.endswith("0.16)")


def test_slots_wrap_rather_than_generating_a_new_hue():
    figure = go.Figure()
    line(figure, [0], [0], name="Ninth", slot=len(SERIES))

    assert figure.data[0].line.color == SERIES[0]


def test_an_empty_figure_says_why_there_is_nothing_to_draw():
    figure = empty_figure("No frames in scope")

    assert figure.layout.annotations[0].text == "No frames in scope"
    assert figure.layout.xaxis.visible is False


def test_applying_the_layout_does_not_clear_axis_titles_already_set():
    """Porting existing figures onto this helper wiped their axis titles.

    A figure that labels its own axes before calling apply_layout must keep
    those labels; passing None for a title means "leave it alone", not "clear
    it".
    """

    figure = go.Figure()
    figure.update_xaxes(title_text="Observed frame count")
    figure.update_yaxes(title_text="Degrees")

    apply_layout(figure)

    assert figure.layout.xaxis.title.text == "Observed frame count"
    assert figure.layout.yaxis.title.text == "Degrees"


def test_an_explicit_axis_title_still_wins():
    figure = go.Figure()
    figure.update_xaxes(title_text="Old")

    apply_layout(figure, x_title="New")

    assert figure.layout.xaxis.title.text == "New"
