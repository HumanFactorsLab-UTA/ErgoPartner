"""The figures that put the score back on a time axis.

What matters here is not that Plotly returns an object. It is that the peak
survives decimation, that the ribbon reports how time was spent rather than only
its worst instant, that every clickable mark carries the frame a reviewer needs,
and that an empty selection draws a sentence instead of empty axes.
"""

from __future__ import annotations

import pytest

from dashboard.task_timeline import (
    band_shares,
    body_part_load,
    risk_episodes,
    score_series,
)
from dashboard.timeline_visuals import (
    body_load_map_figure,
    compare_methods_figure,
    contribution_figure,
    decimate,
    risk_ribbon_figure,
    selected_frame_from_event,
    side_comparison_figure,
    task_risk_timeline_figure,
    time_at_band_figure,
)
from dashboard.viz_theme import STATUS
from tests.test_task_timeline import record


def _series(count: int = 400, method: str = "reba"):
    """A repetitive lifting trial: easy stretches, sustained hard stretches, one peak."""

    rows = []
    for index in range(count):
        phase = index % 40
        if index == count // 2:
            components = {"trunk": 5, "upper_arm": 6, "wrist": 3, "neck": 3, "activity": 3}
        elif phase < 12:
            # Twelve samples at 0.1 s each: long enough to be a real episode.
            components = {"trunk": 5, "upper_arm": 5, "activity": 3}
        else:
            components = {"trunk": 2, "upper_arm": 2, "activity": 1}
        rows.append(
            record(
                index,
                index * 0.1,
                method=method,
                components=components,
                other_components={key: max(1, value - 1) for key, value in components.items()},
                angles={"trunk_flexion": 10.0 + index % 70, "left_upper_arm_flexion": 40.0},
            )
        )
    return score_series(rows, method)


# ----------------------------------------------------------------------
# Decimation
# ----------------------------------------------------------------------


def test_decimation_keeps_the_highest_and_lowest_observation_of_every_bucket():
    """A stride would drop the one frame the whole page exists to find."""

    series = _series(1000)
    thinned = decimate(series, 200)

    assert len(thinned) <= 200 + 2
    assert max(point.score for point in thinned) == max(point.score for point in series)
    assert min(point.score for point in thinned) == min(point.score for point in series)


def test_decimation_keeps_the_series_in_time_order_and_keeps_the_ends():
    series = _series(1000)
    thinned = decimate(series, 120)
    times = [point.timestamp_sec for point in thinned]

    assert times == sorted(times)
    assert thinned[0].frame_number == series[0].frame_number
    assert thinned[-1].frame_number == series[-1].frame_number


def test_a_short_series_is_left_alone():
    series = _series(30)

    assert decimate(series, 200) == list(series)


# ----------------------------------------------------------------------
# The timeline
# ----------------------------------------------------------------------


def test_the_timeline_shades_every_published_action_level_and_names_it():
    figure = task_risk_timeline_figure(_series(), "reba")
    labels = {
        shape_annotation.text for shape_annotation in figure.layout.annotations if shape_annotation.text
    }

    assert {"Negligible", "Low", "Medium", "High", "Very high"} <= labels
    assert figure.layout.yaxis.range == (0.5, 15.5)


def test_the_timeline_uses_a_clock_axis_not_decimal_seconds():
    figure = task_risk_timeline_figure(_series(), "reba")

    assert all(":" in text for text in figure.layout.xaxis.ticktext)


def test_every_timeline_mark_carries_the_frame_to_open():
    figure = task_risk_timeline_figure(_series(), "reba")
    marks = [trace for trace in figure.data if trace.mode and "markers" in trace.mode]

    assert marks
    for trace in marks:
        assert trace.customdata is not None
        assert all(row[0] is not None for row in trace.customdata)


def test_the_timeline_marks_the_worst_observation_explicitly():
    series = _series()
    figure = task_risk_timeline_figure(series, "reba")
    worst = max(point.score for point in series)
    texts = [text for trace in figure.data if trace.text for text in trace.text]

    assert any(str(worst) in str(text) for text in texts)


def test_episodes_are_shaded_so_blocks_of_time_are_visible_not_only_spikes():
    series = _series()
    episodes = risk_episodes(series, "reba")
    plain = task_risk_timeline_figure(series, "reba")
    shaded = task_risk_timeline_figure(series, "reba", episodes=episodes)

    assert episodes
    assert len(shaded.layout.shapes) > len(plain.layout.shapes)


def test_a_selected_frame_is_marked_on_the_timeline():
    series = _series()
    target = series[120].frame_number
    without = task_risk_timeline_figure(series, "reba")
    with_line = task_risk_timeline_figure(series, "reba", highlight_frame=target)

    assert len(with_line.layout.shapes) > len(without.layout.shapes)


# ----------------------------------------------------------------------
# The ribbon
# ----------------------------------------------------------------------


def test_the_ribbon_stacks_how_time_was_spent_rather_than_only_the_worst_instant():
    """Colouring each slice by its worst sample produced a solid barcode.

    On a task with a lift every couple of seconds nearly every slice contains a
    peak, so the strip said "bad everywhere" and could not distinguish a job with
    occasional hard moments from one that is hard throughout.
    """

    figure = risk_ribbon_figure(_series(), "reba", columns=40)

    assert figure.layout.barmode == "stack"
    assert len(figure.data) >= 2
    for slot in range(40):
        total = sum(float(trace.y[slot]) for trace in figure.data)
        assert total == pytest.approx(1.0, abs=1e-6)


def test_every_ribbon_column_offers_the_worst_frame_in_that_slice():
    series = _series()
    figure = risk_ribbon_figure(series, "reba", columns=30)
    frames = {int(row[0]) for trace in figure.data for row in trace.customdata if row[0] is not None}
    available = {point.frame_number for point in series}

    assert frames
    assert frames <= available


def test_ribbon_columns_are_labelled_with_the_methods_own_band_names():
    figure = risk_ribbon_figure(_series(), "reba", columns=20)
    names = {trace.name for trace in figure.data}

    assert any("Very high" in name for name in names)
    assert not any("Change now" in name for name in names)


# ----------------------------------------------------------------------
# Time at band
# ----------------------------------------------------------------------


def test_time_at_band_totals_one_hundred_percent_and_labels_every_band():
    series = _series()
    figure = time_at_band_figure(band_shares(series, "reba"), "reba")
    total = sum(float(trace.x[0]) for trace in figure.data)

    assert total == pytest.approx(100.0, abs=0.01)
    assert figure.layout.barmode == "stack"
    for trace in figure.data:
        assert trace.name
        assert trace.marker.color in STATUS.values()


# ----------------------------------------------------------------------
# Contribution and the body map
# ----------------------------------------------------------------------


def test_the_contribution_chart_ranks_by_points_and_leaves_room_for_its_labels():
    series = _series()
    loads = body_part_load(series, "reba")
    figure = contribution_figure(loads, "reba", kind="posture")
    values = list(figure.data[0].x)

    assert values == sorted(values)
    # Outside labels are drawn past the bar end; without headroom the longest
    # bar -- always the one that matters most -- had its own value clipped.
    assert figure.layout.xaxis.range[1] > max(values)


def test_the_contribution_chart_carries_a_frame_and_a_region_per_bar():
    loads = body_part_load(_series(), "reba")
    figure = contribution_figure(loads, "reba", kind="posture")

    for row in figure.data[0].customdata:
        assert row[0] is not None
        assert isinstance(row[1], str)


def test_the_body_map_scales_by_points_added_not_by_a_yes_or_no_flag():
    """Every joint leaves neutral once over two minutes, so a flag ranks nothing."""

    loads = body_part_load(_series(), "reba")
    figure = body_load_map_figure(loads, "reba")
    sized = [
        trace.marker.size
        for trace in figure.data
        if trace.mode and "text" in trace.mode and trace.marker.size
    ]

    assert len(set(sized)) > 1


def test_the_body_map_highlights_the_selected_region():
    loads = body_part_load(_series(), "reba")
    plain = body_load_map_figure(loads, "reba")
    picked = body_load_map_figure(loads, "reba", selected_region="trunk")

    assert [trace.marker.color for trace in plain.data] != [
        trace.marker.color for trace in picked.data
    ]


def test_task_conditions_are_charted_separately_from_body_parts():
    loads = body_part_load(_series(), "reba")
    posture = contribution_figure(loads, "reba", kind="posture")
    task = contribution_figure(loads, "reba", kind="task")

    assert "body parts" in posture.layout.title.text
    assert "task conditions" in task.layout.title.text
    assert set(posture.data[0].y).isdisjoint(set(task.data[0].y))


# ----------------------------------------------------------------------
# Sides and cross-method comparison
# ----------------------------------------------------------------------


def test_both_sides_are_drawn_because_the_reported_score_is_only_one_of_them():
    figure = side_comparison_figure(_series(), "reba")

    assert {trace.name for trace in figure.data} == {"Left side", "Right side"}


def test_the_two_methods_are_compared_as_a_share_of_their_own_scales():
    """Raw numbers side by side would imply REBA 8 is worse than RULA 7."""

    figure = compare_methods_figure(_series(200, "rula"), _series(200, "reba"))

    assert figure.layout.yaxis.range == (0, 105)
    assert all(max(trace.y) <= 105 for trace in figure.data)
    assert any("of 7" in trace.name for trace in figure.data)
    assert any("of 15" in trace.name for trace in figure.data)


# ----------------------------------------------------------------------
# Event reading and empty states
# ----------------------------------------------------------------------


def test_a_selection_event_yields_the_frame_and_any_region():
    assert selected_frame_from_event({"selection": {"points": [{"customdata": [1333, "trunk"]}]}}) == (
        1333,
        "trunk",
    )
    assert selected_frame_from_event({"selection": {"points": [{"customdata": [7]}]}}) == (7, None)
    assert selected_frame_from_event({"selection": {"points": []}}) == (None, None)
    assert selected_frame_from_event({}) == (None, None)
    assert selected_frame_from_event({"selection": {"points": [{"customdata": [None]}]}}) == (
        None,
        None,
    )


def test_a_bare_frame_number_in_customdata_is_still_read():
    assert selected_frame_from_event({"selection": {"points": [{"customdata": 42}]}}) == (42, None)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: task_risk_timeline_figure([], "reba"),
        lambda: risk_ribbon_figure([], "reba"),
        lambda: time_at_band_figure([], "reba"),
        lambda: contribution_figure([], "reba"),
        lambda: body_load_map_figure([], "reba"),
        lambda: side_comparison_figure([], "reba"),
        lambda: compare_methods_figure([], []),
    ],
)
def test_an_empty_selection_says_why_there_is_nothing_to_draw(builder):
    figure = builder()

    assert not figure.data
    assert figure.layout.annotations
    assert figure.layout.annotations[0].text
