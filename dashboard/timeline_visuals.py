"""Figures that put the score back on a time axis.

The Results page had no time axis anywhere. Range of motion, band exposure,
readiness and camera agreement all summarise a whole trial, so a reviewer could
read every chart on the page and still not know whether the bad posture happened
once at the start or ran through the middle of the job.

These figures answer that. The timeline shows the score as the task ran, against
the method's own action bands. The ribbon compresses the same thing into one
strip that can be read at a glance from across a room. The contribution chart
says which body part the points came from. All three are clickable and hand back
a frame number, so a high point on a chart leads directly to the recorded frames.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from dashboard.task_timeline import (
    BandShare,
    BodyPartLoad,
    RiskEpisode,
    ScorePoint,
    clock,
)
from dashboard.viz_theme import (
    AXIS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    METHOD_SCALE_MAXIMUM,
    SEQUENTIAL_BLUE,
    SERIES,
    SURFACE,
    apply_layout,
    empty_figure,
    method_bands,
    translucent,
)

#: Plotly draws every point it is given. A two-minute trial at native frame rate
#: is four thousand of them, which is slow in the browser and no more legible
#: than a thousand. Decimation below is min/max per bucket, so the envelope --
#: and therefore every spike the reviewer is looking for -- survives exactly.
DEFAULT_MAXIMUM_POINTS = 1200


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def decimate(series: Sequence[ScorePoint], maximum_points: int) -> list[ScorePoint]:
    """Thin a series without losing a single peak.

    Each bucket contributes its lowest and highest observation, in time order.
    A plain stride would drop the one frame that reached the top of the scale,
    which is the frame the whole page exists to find.
    """

    if maximum_points < 4 or len(series) <= maximum_points:
        return list(series)
    buckets = max(2, maximum_points // 2)
    size = len(series) / buckets
    kept: dict[int, ScorePoint] = {}
    for index in range(buckets):
        start = int(index * size)
        end = int((index + 1) * size) if index + 1 < buckets else len(series)
        window = series[start:end]
        if not window:
            continue
        lowest = min(range(start, start + len(window)), key=lambda i: series[i].score)
        highest = max(range(start, start + len(window)), key=lambda i: series[i].score)
        kept[lowest] = series[lowest]
        kept[highest] = series[highest]
    kept[0] = series[0]
    kept[len(series) - 1] = series[-1]
    return [kept[index] for index in sorted(kept)]


def _band_backdrop(figure: go.Figure, method: str, *, row: int | None = None) -> None:
    """Shade each published action level behind the data and name it."""

    top = METHOD_SCALE_MAXIMUM.get(str(method).lower(), 15)
    for band in method_bands(method):
        upper = min(band.upper, top) + 0.5
        figure.add_hrect(
            y0=band.lower - 0.5,
            y1=upper,
            # Kept faint. The horizontal bands are a reference scale; the vertical
            # episode shading is the finding. At equal strength the two washes
            # multiplied into a muddy backdrop and neither read clearly.
            fillcolor=translucent(band.colour, 0.08),
            line_width=0,
            layer="below",
            annotation_text=band.label,
            annotation_position="right",
            annotation_font={"size": 10, "color": INK_MUTED},
            **({"row": row, "col": 1} if row is not None else {}),
        )


def task_risk_timeline_figure(
    series: Sequence[ScorePoint],
    method: str,
    *,
    episodes: Sequence[RiskEpisode] = (),
    highlight_frame: int | None = None,
    maximum_points: int = DEFAULT_MAXIMUM_POINTS,
) -> go.Figure:
    """Score against time, on the method's own action bands, clickable.

    Selecting any point returns its frame number as ``customdata`` so the caller
    can open that exact moment in the synchronized frame review.
    """

    upper = str(method).upper()
    if not series:
        return empty_figure(
            f"{upper} produced no scored observation in this selection, so there is "
            "no timeline to draw.",
            height=300,
        )

    points = decimate(series, maximum_points)
    times = [point.timestamp_sec for point in points]
    scores = [point.score for point in points]
    frames = [point.frame_number for point in points]
    labels = [(point.band.label if point.band else "No band") for point in points]
    colours = [(point.band.colour if point.band else INK_MUTED) for point in points]

    figure = go.Figure()
    _band_backdrop(figure, method)

    # Every episode is shaded before the trace so the reviewer sees the blocks of
    # time that matter, not only the instantaneous peaks inside them.
    for episode in episodes:
        figure.add_vrect(
            x0=episode.start_sec,
            x1=max(episode.end_sec, episode.start_sec + 0.05),
            fillcolor=translucent(episode.band.colour, 0.18),
            line_width=0,
            layer="below",
        )

    figure.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines",
            line={"color": INK_SECONDARY, "width": 1.5, "shape": "hv"},
            hoverinfo="skip",
            showlegend=False,
            name=f"{upper} score",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="markers",
            marker={"color": colours, "size": 6, "line": {"width": 0}},
            customdata=[
                [frame, label, clock(time)]
                for frame, label, time in zip(frames, labels, times, strict=True)
            ],
            hovertemplate=(
                "<b>%{customdata[2]}</b> · " + upper + " %{y}"
                "<br>%{customdata[1]}"
                "<br>Frame %{customdata[0]} — click to open the frames"
                "<extra></extra>"
            ),
            showlegend=False,
            name=f"{upper} observations",
        )
    )

    peak = max(points, key=lambda point: (point.score, -point.timestamp_sec))
    figure.add_trace(
        go.Scatter(
            x=[peak.timestamp_sec],
            y=[peak.score],
            mode="markers+text",
            marker={
                "color": peak.band.colour if peak.band else INK_MUTED,
                "size": 15,
                "symbol": "diamond",
                "line": {"color": INK_PRIMARY, "width": 2},
            },
            text=[f" worst: {peak.score}"],
            textposition="middle right",
            textfont={"color": INK_PRIMARY, "size": 12},
            customdata=[[peak.frame_number, "Worst observation", clock(peak.timestamp_sec)]],
            hovertemplate=(
                "<b>Worst %{customdata[1]}</b><br>%{customdata[2]} · "
                + upper
                + " %{y}<br>Frame %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    if highlight_frame is not None:
        chosen = next((point for point in series if point.frame_number == highlight_frame), None)
        if chosen is not None:
            figure.add_vline(
                x=chosen.timestamp_sec,
                line={"color": INK_PRIMARY, "width": 1.5, "dash": "dot"},
            )

    top = METHOD_SCALE_MAXIMUM.get(str(method).lower(), 15)
    apply_layout(
        figure,
        title=f"{upper} through the task — click any point to open those frames",
        height=380,
        show_legend=False,
    )
    figure.update_layout(hovermode="closest", clickmode="event+select", dragmode="zoom")
    figure.update_xaxes(
        title_text="Time into the task",
        tickmode="array",
        tickvals=_time_ticks(times),
        ticktext=[clock(value) for value in _time_ticks(times)],
    )
    figure.update_yaxes(
        title_text=f"{upper} score", range=[0.5, top + 0.5], dtick=1, showgrid=False
    )
    return figure


def _time_ticks(times: Sequence[float], target: int = 8) -> list[float]:
    """Ticks on round seconds, so the axis reads as a clock and not as decimals."""

    if not times:
        return []
    span = max(times) - min(times)
    if span <= 0:
        return [times[0]]
    for step in (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600):
        if span / step <= target:
            break
    start = math.floor(min(times) / step) * step
    ticks = []
    value = start
    while value <= max(times) + step / 2:
        if value >= min(times) - step / 2:
            ticks.append(float(value))
        value += step
    return ticks


def risk_ribbon_figure(
    series: Sequence[ScorePoint],
    method: str,
    *,
    columns: int = 90,
    highlight_frame: int | None = None,
) -> go.Figure:
    """One strip: how each slice of the task divided between the action levels.

    This is the chart to read first and the only one that needs no training. The
    task runs left to right and each column is a slice of it, split by how long
    that slice spent at each action level, worst on top.

    Colouring each slice by its single worst observation was the obvious design
    and it failed on the first real trial: a lift every two and a half seconds
    puts a peak in nearly every slice, so the strip came out a solid barcode that
    said only "something bad happened everywhere". Stacking the actual dwell time
    keeps the peaks visible as red caps while showing what the rest of each slice
    was doing, which is the difference between a job with occasional hard moments
    and one that is hard throughout.
    """

    upper = str(method).upper()
    if not series:
        return empty_figure(f"No {upper} score was produced in this selection.", height=150)

    bands = method_bands(method)
    start = series[0].timestamp_sec
    end = max(point.timestamp_sec + point.dwell_sec for point in series)
    span = max(end - start, 1e-6)
    width = span / max(1, columns)

    dwell = [dict.fromkeys(range(len(bands)), 0.0) for _ in range(columns)]
    worst: list[ScorePoint | None] = [None] * columns
    for point in series:
        slot = min(columns - 1, max(0, int((point.timestamp_sec - start) / width)))
        index = next(
            (position for position, band in enumerate(bands) if point.score <= band.upper),
            len(bands) - 1,
        )
        # A sample with no dwell (an isolated observation across a tracking gap)
        # would otherwise contribute nothing and leave a hole in the strip.
        dwell[slot][index] += point.dwell_sec or 1e-6
        current = worst[slot]
        if current is None or point.score > current.score:
            worst[slot] = point

    centres = [start + (slot + 0.5) * width for slot in range(columns)]
    figure = go.Figure()
    for index, band in enumerate(bands):
        heights: list[float] = []
        custom: list[list[Any]] = []
        for slot in range(columns):
            total = sum(dwell[slot].values())
            heights.append((dwell[slot][index] / total) if total > 0 else 0.0)
            peak = worst[slot]
            custom.append(
                [
                    peak.frame_number if peak is not None else None,
                    clock(start + slot * width),
                    clock(start + (slot + 1) * width),
                    peak.score if peak is not None else None,
                ]
            )
        if not any(heights):
            continue
        figure.add_trace(
            go.Bar(
                x=centres,
                y=heights,
                width=[width] * columns,
                name=f"{band.label} ({band.range_text})",
                marker={"color": band.colour, "line": {"width": 0}},
                customdata=custom,
                hovertemplate=(
                    f"<b>{band.label}</b> ({upper} {band.range_text})"
                    "<br>%{y:.0%} of %{customdata[1]}–%{customdata[2]}"
                    "<br>Worst here: " + upper + " %{customdata[3]}, frame %{customdata[0]}"
                    "<br>Click to open those frames<extra></extra>"
                ),
            )
        )

    if highlight_frame is not None:
        chosen = next((point for point in series if point.frame_number == highlight_frame), None)
        if chosen is not None:
            figure.add_vline(x=chosen.timestamp_sec, line={"color": INK_PRIMARY, "width": 2})

    apply_layout(figure, title=None, height=170, series_count=2)
    figure.update_layout(
        barmode="stack",
        hovermode="closest",
        clickmode="event+select",
        dragmode=False,
        bargap=0,
        margin={"l": 56, "r": 20, "t": 34, "b": 40},
    )
    ticks = _time_ticks([point.timestamp_sec for point in series])
    figure.update_xaxes(
        title_text=None,
        tickmode="array",
        tickvals=ticks,
        ticktext=[clock(value) for value in ticks],
        showgrid=False,
        fixedrange=True,
    )
    figure.update_yaxes(visible=False, fixedrange=True, range=[0, 1])
    return figure


def time_at_band_figure(shares: Sequence[BandShare], method: str) -> go.Figure:
    """One stacked bar: the share of the task spent at each action level."""

    upper = str(method).upper()
    observed = [share for share in shares if share.duration_sec > 0]
    if not observed:
        return empty_figure(f"No {upper} time could be attributed to an action level.", height=170)

    figure = go.Figure()
    for share in observed:
        figure.add_trace(
            go.Bar(
                x=[share.percent_of_scored_time],
                y=["share"],
                orientation="h",
                name=f"{share.band.label} ({share.band.range_text})",
                marker={"color": share.band.colour, "line": {"width": 0}},
                text=[
                    f"{share.percent_of_scored_time:.0f}%"
                    if share.percent_of_scored_time >= 7
                    else ""
                ],
                textposition="inside",
                insidetextfont={"color": "#ffffff", "size": 12},
                hovertemplate=(
                    f"<b>{share.band.label}</b> ({upper} {share.band.range_text})"
                    f"<br>{share.band.action}"
                    f"<br>{share.duration_sec:.1f} s · "
                    f"{share.percent_of_scored_time:.1f}% of scored time"
                    f"<br>{share.observation_count} observations<extra></extra>"
                ),
            )
        )
    apply_layout(figure, title=f"How the {upper} time was spent", height=170, series_count=2)
    figure.update_layout(barmode="stack", hovermode="closest", bargap=0.45)
    figure.update_xaxes(
        title_text="Share of scored time (%)", range=[0, 100], showgrid=False, fixedrange=True
    )
    figure.update_yaxes(visible=False, fixedrange=True)
    return figure


def contribution_figure(
    loads: Sequence[BodyPartLoad],
    method: str,
    *,
    kind: str = "posture",
    selected_region: str | None = None,
) -> go.Figure:
    """Body parts ranked by the points they added to the score.

    The bar length is the answer to "where": how many points this part put on the
    average score of this task, calculated by setting that one component back to
    neutral and asking the method what the total becomes.
    """

    upper = str(method).upper()
    rows = [item for item in loads if item.kind == kind and item.mean_points > 0]
    if not rows:
        noun = "body part" if kind == "posture" else "task factor"
        return empty_figure(
            f"No {noun} added points to the {upper} score in this selection.", height=260
        )
    rows = sorted(rows, key=lambda item: item.mean_points)

    def shade(item: BodyPartLoad) -> str:
        if selected_region is not None and item.region == selected_region:
            return SERIES[1]
        depth = min(len(SEQUENTIAL_BLUE) - 1, 2 + int(round(item.worst_severity * 3)))
        return SEQUENTIAL_BLUE[depth]

    figure = go.Figure(
        go.Bar(
            x=[item.mean_points for item in rows],
            y=[item.label for item in rows],
            orientation="h",
            marker={"color": [shade(item) for item in rows], "line": {"width": 0}},
            text=[f"+{item.mean_points:.1f}" for item in rows],
            textposition="outside",
            textfont={"color": INK_SECONDARY, "size": 11},
            customdata=[
                [
                    item.peak_frame_number,
                    item.region,
                    item.worst_score_text,
                    item.percent_of_scored_time,
                    (
                        f"{item.measurement.replace('_', ' ')} {item.peak_value_deg:.0f}°"
                        if item.measurement and item.peak_value_deg is not None
                        else "angle not recorded"
                    ),
                    clock(item.peak_timestamp_sec),
                    item.worst_points,
                ]
                for item in rows
            ],
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Adds %{x:.2f} points to the average " + upper + " score"
                "<br>Worst component score: %{customdata[2]} (up to +%{customdata[6]})"
                "<br>Costs points in %{customdata[3]:.0f}% of the task"
                "<br>Worst at %{customdata[5]}, frame %{customdata[0]} · %{customdata[4]}"
                "<br>Click to open that frame<extra></extra>"
            ),
            showlegend=False,
        )
    )
    heading = (
        f"Where the {upper} points come from — body parts"
        if kind == "posture"
        else f"Where the {upper} points come from — task conditions"
    )
    apply_layout(figure, title=heading, height=max(240, 42 * len(rows) + 120), show_legend=False)
    figure.update_layout(hovermode="closest", clickmode="event+select", dragmode=False)
    # Outside labels are drawn past the end of the bar, so the axis needs room
    # for them or the longest bar -- always the one that matters most -- has its
    # own value clipped by the plot edge.
    longest = max(item.mean_points for item in rows)
    figure.update_xaxes(
        title_text=f"Points added to the average {upper} score",
        showgrid=True,
        range=[0, longest * 1.18],
    )
    figure.update_yaxes(title_text=None, showgrid=False, fixedrange=True)
    return figure


_BODY_POSITIONS: Mapping[str, tuple[float, float]] = {
    "neck": (0.0, 7.35),
    "trunk": (0.0, 5.6),
    "left_shoulder": (-1.35, 6.75),
    "right_shoulder": (1.35, 6.75),
    "left_elbow": (-2.15, 5.1),
    "right_elbow": (2.15, 5.1),
    "left_wrist": (-2.7, 3.55),
    "right_wrist": (2.7, 3.55),
    "legs": (0.0, 1.7),
}

_SKELETON = (
    ((0.0, 7.6), (0.0, 3.7)),
    ((-1.35, 6.75), (1.35, 6.75)),
    ((-1.35, 6.75), (-2.15, 5.1)),
    ((-2.15, 5.1), (-2.7, 3.55)),
    ((1.35, 6.75), (2.15, 5.1)),
    ((2.15, 5.1), (2.7, 3.55)),
    ((0.0, 3.7), (-0.9, 1.7)),
    ((-0.9, 1.7), (-0.9, 0.0)),
    ((0.0, 3.7), (0.9, 1.7)),
    ((0.9, 1.7), (0.9, 0.0)),
)


def body_load_map_figure(
    loads: Sequence[BodyPartLoad],
    method: str,
    *,
    selected_region: str | None = None,
) -> go.Figure:
    """A body diagram sized and shaded by points added, not by a yes/no flag.

    The previous map coloured a joint red if it ever left neutral once. Over a
    two-minute trial that is almost every joint, so the map could not rank
    anything and every trial looked identical. Marker area here is the points the
    part adds to the average score, which is the same quantity the ranked bars
    show, so the two cannot disagree.
    """

    upper = str(method).upper()
    postural = {
        item.region: item
        for item in loads
        if item.kind == "posture" and item.region in _BODY_POSITIONS
    }
    if not postural:
        return empty_figure(
            f"No mapped body region carries {upper} component evidence in this selection.",
            height=430,
        )
    strongest = max((item.mean_points for item in postural.values()), default=0.0) or 1.0

    figure = go.Figure()
    for (x0, y0), (x1, y1) in _SKELETON:
        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"color": AXIS, "width": 7},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_shape(
        type="circle",
        x0=-0.62,
        x1=0.62,
        y0=7.75,
        y1=9.0,
        line={"color": AXIS, "width": 6},
        fillcolor=SURFACE,
    )

    for region, (x, y) in _BODY_POSITIONS.items():
        item = postural.get(region)
        if item is None:
            figure.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker={"color": translucent(INK_MUTED, 0.5), "size": 14},
                    hovertemplate=(
                        f"<b>{region.replace('_', ' ').title()}</b>"
                        "<br>No component score was produced<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            continue
        share = item.mean_points / strongest
        selected = selected_region == region
        depth = min(len(SEQUENTIAL_BLUE) - 1, 2 + int(round(item.worst_severity * 3)))
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[f"{item.label}<br>+{item.mean_points:.1f}"],
                textposition="middle right" if x > 0 else "middle left",
                textfont={"color": INK_SECONDARY, "size": 10},
                marker={
                    "color": SERIES[1] if selected else SEQUENTIAL_BLUE[depth],
                    "size": 16 + 30 * share,
                    "line": {"color": INK_PRIMARY if selected else SURFACE, "width": 2},
                },
                customdata=[[item.peak_frame_number, region]],
                hovertemplate=(
                    f"<b>{item.label}</b>"
                    f"<br>Adds {item.mean_points:.2f} points to the average {upper} score"
                    f"<br>Worst component score: {item.worst_score_text}"
                    f"<br>Costs points in {item.percent_of_scored_time:.0f}% of the task"
                    f"<br>Worst at {clock(item.peak_timestamp_sec)}, "
                    f"frame {item.peak_frame_number}"
                    "<br>Click to open that frame<extra></extra>"
                ),
                showlegend=False,
            )
        )

    apply_layout(figure, title=f"{upper} points by body region", height=470, show_legend=False)
    figure.update_layout(hovermode="closest", clickmode="event+select", dragmode=False)
    figure.update_xaxes(visible=False, range=[-4.6, 4.6], fixedrange=True, showgrid=False)
    figure.update_yaxes(
        visible=False, range=[-0.6, 9.4], fixedrange=True, scaleanchor="x", showgrid=False
    )
    return figure


def side_comparison_figure(series: Sequence[ScorePoint], method: str) -> go.Figure:
    """Left against right over time, because the reported score is one side only.

    RULA and REBA report the worse arm. A reviewer who never sees the other side
    cannot tell a genuinely one-sided job from a symmetric one that happens to
    tip left, and those call for different fixes.
    """

    upper = str(method).upper()
    rows = [
        point
        for point in series
        if point.left_components or point.right_components
    ]
    if not rows:
        return empty_figure(f"No per-side {upper} components were stored.", height=280)
    return _side_figure(rows, method)


def _side_figure(rows: Sequence[ScorePoint], method: str) -> go.Figure:
    from dashboard.task_timeline import recompute_final

    upper = str(method).upper()
    points = decimate(rows, DEFAULT_MAXIMUM_POINTS)
    times = [point.timestamp_sec for point in points]
    figure = go.Figure()
    drawn = 0
    for slot, (side, attribute) in enumerate(
        (("Left", "left_components"), ("Right", "right_components"))
    ):
        values = [recompute_final(getattr(point, attribute), method) for point in points]
        if not any(value is not None for value in values):
            continue
        drawn += 1
        figure.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines",
                name=f"{side} side",
                line={"color": SERIES[slot], "width": 2, "shape": "hv"},
                customdata=[point.frame_number for point in points],
                hovertemplate=(
                    f"{side} {upper} " + "%{y}<br>Frame %{customdata}<extra></extra>"
                ),
            )
        )
    if not drawn:
        return empty_figure(f"No per-side {upper} score could be reconstructed.", height=280)
    top = METHOD_SCALE_MAXIMUM.get(str(method).lower(), 15)
    apply_layout(figure, title=f"{upper}: left against right", height=300, series_count=2)
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(
        title_text="Time into the task",
        tickmode="array",
        tickvals=_time_ticks(times),
        ticktext=[clock(value) for value in _time_ticks(times)],
    )
    figure.update_yaxes(title_text=f"{upper} score", range=[0.5, top + 0.5], dtick=1)
    return figure


def compare_methods_figure(
    rula: Sequence[ScorePoint], reba: Sequence[ScorePoint]
) -> go.Figure:
    """RULA and REBA on one time axis, each as a share of its own scale.

    The two methods use different scales, so plotting the raw numbers together
    would imply REBA 8 is worse than RULA 7. Each series is drawn as a percentage
    of its own maximum, and the hover carries the real score.
    """

    if not rula and not reba:
        return empty_figure("Neither method produced a score in this selection.", height=300)
    figure = go.Figure()
    for slot, (name, series) in enumerate((("RULA", rula), ("REBA", reba))):
        if not series:
            continue
        top = METHOD_SCALE_MAXIMUM[name.lower()]
        points = decimate(series, DEFAULT_MAXIMUM_POINTS)
        figure.add_trace(
            go.Scatter(
                x=[point.timestamp_sec for point in points],
                y=[100.0 * point.score / top for point in points],
                mode="lines",
                name=f"{name} (of {top})",
                line={"color": SERIES[slot], "width": 2, "shape": "hv"},
                customdata=[
                    [point.score, point.frame_number, point.band.label if point.band else "—"]
                    for point in points
                ],
                hovertemplate=(
                    f"{name} " + "%{customdata[0]} · %{customdata[2]}"
                    "<br>Frame %{customdata[1]}<extra></extra>"
                ),
            )
        )
    times = [point.timestamp_sec for point in (list(rula) + list(reba))]
    apply_layout(
        figure,
        title="Both methods on one axis, each as a share of its own scale",
        height=300,
        series_count=2,
    )
    figure.update_layout(hovermode="x unified")
    figure.update_xaxes(
        title_text="Time into the task",
        tickmode="array",
        tickvals=_time_ticks(times),
        ticktext=[clock(value) for value in _time_ticks(times)],
    )
    figure.update_yaxes(title_text="Share of the method's scale (%)", range=[0, 105])
    return figure


def selected_frame_from_event(event: Any) -> tuple[int | None, str | None]:
    """Read a frame number, and any region, out of a Plotly selection event.

    Every clickable figure here puts the frame first in ``customdata``, so one
    reader serves all of them and a new figure only has to keep that convention.
    """

    try:
        points = event.selection.points
    except AttributeError:
        selection = event.get("selection") if isinstance(event, Mapping) else None
        points = (selection or {}).get("points", ()) if isinstance(selection, Mapping) else ()
    if not points:
        return None, None
    first = points[0]
    data = first.get("customdata") if isinstance(first, Mapping) else None
    if data is None:
        return None, None
    if not isinstance(data, list | tuple):
        data = [data]
    frame = _finite(data[0]) if data else None
    region = None
    for value in data[1:]:
        if isinstance(value, str) and value in _BODY_POSITIONS:
            region = value
            break
    return (int(frame) if frame is not None else None), region


__all__ = [
    "DEFAULT_MAXIMUM_POINTS",
    "body_load_map_figure",
    "compare_methods_figure",
    "contribution_figure",
    "decimate",
    "risk_ribbon_figure",
    "selected_frame_from_event",
    "side_comparison_figure",
    "task_risk_timeline_figure",
    "time_at_band_figure",
]
