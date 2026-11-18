"""Validity-aware Plotly figures and exports for professional analysis results.

The functions in this module consume only ``ProfessionalAnalysis.to_dict()``
compatible data.  They do not calculate ergonomic scores, infer pain, or merge
RULA, REBA, and RNLE into a synthetic overall value.  Missing values remain
explicitly unavailable in figures, reports, and CSV-ready rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.viz_theme import (
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    PAGE,
    SERIES,
    STATUS,
    SURFACE,
)
from dashboard.viz_theme import apply_layout as _apply_layout
from dashboard.viz_theme import empty_figure as _shared_empty_figure

# Every colour now comes from dashboard/viz_theme.py, which is validated against
# this dashboard's own surface. These names are kept because the figures below
# and their tests refer to them; each is an alias onto the shared system rather
# than a value chosen here.
#
# The distinction that was missing before: BLUE/RED/GREEN/AMBER were used both to
# identify a series and to signal a risk state, so a series could wear the colour
# of a status and mean nothing by it. Series hues and status hues are now
# separate sets and cannot collide.
BACKGROUND = PAGE
PANEL = SURFACE
GRID = GRIDLINE
TEXT = INK_PRIMARY
MUTED = INK_MUTED
BLUE = SERIES[0]
LIGHT_BLUE = SERIES[2]
RED = STATUS["critical"]
GREEN = STATUS["good"]
AMBER = STATUS["warning"]
UNAVAILABLE = INK_MUTED

#: A horizontal bar chart taller than the screen cannot be compared against
#: itself. Rows beyond this are still exported in full; only the figure is capped.
MAXIMUM_BAND_ROWS = 22

CSV_COLUMNS: tuple[str, ...] = (
    "section",
    "method",
    "measurement",
    "metric",
    "band_id",
    "value",
    "units",
    "status",
    "frame_number",
    "timestamp_sec",
    "source_url",
    "detail",
)


def _payload(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("analysis must be a mapping or expose a mapping-returning to_dict()")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _label(value: Any) -> str:
    text = _text(value)
    if text is None:
        return "Not available"
    return text.replace("_", " ").title()


def _format_number(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _finite(value)
    if number is None:
        return "Not available"
    if number.is_integer():
        rendered = str(int(number))
    else:
        rendered = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def _format_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return _format_number(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(item) for item in value) if value else "None reported"
    return str(value)


def _human_measurement(value: Any) -> str:
    text = _text(value)
    return text.replace("_", " ").title() if text else "Not available"


def _layout(
    figure: go.Figure,
    *,
    title: str,
    height: int = 430,
    series_count: int = 2,
) -> go.Figure:
    """Apply the shared layout to a figure built here.

    Delegating rather than restating the layout is what makes the port
    worthwhile: the rules that stop labels colliding and keep the legend off the
    data live in one place, so every figure below inherits them and a future
    figure cannot quietly opt out.

    Two things change for every existing figure. The plot area is transparent
    instead of a panel colour, so a chart sits on the page rather than in a box
    of a slightly different shade. And ``automargin`` is on, so a long axis
    label grows the margin instead of overlapping the tick labels next to it --
    the overlap that made several of these hard to read.
    """

    _apply_layout(
        figure,
        title=title,
        height=height,
        series_count=series_count,
        show_legend=None if series_count >= 2 else False,
    )
    return figure


def _empty_figure(title: str, message: str) -> go.Figure:
    """A placeholder that names the figure and says why it is empty.

    "Not available" is kept verbatim: it is the phrase the rest of the results
    uses for a value that could not be produced, and it distinguishes that from
    a measured zero.
    """

    return _shared_empty_figure(f"{title}: Not available. {message}", height=320)


def _status_color(status: Any) -> str:
    value = (_text(status) or "").lower()
    if value in {"ready", "valid", "available", "candidate_events_available"}:
        return GREEN
    if value in {"partially_ready", "valid_with_limitations"}:
        return AMBER
    if value in {"not_evaluated", "unavailable"}:
        return UNAVAILABLE
    return RED


def method_readiness_figure(analysis: Any) -> go.Figure:
    """Return a validity-aware RULA/REBA/RNLE readiness table."""

    payload = _payload(analysis)
    readiness = [_mapping(item) for item in _sequence(payload.get("method_readiness"))]
    if not readiness:
        return _empty_figure(
            "Method readiness",
            "No independently calculated RULA, REBA, or RNLE readiness data were supplied.",
        )
    methods: list[str] = []
    statuses: list[str] = []
    coverage: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    unsupported: list[str] = []
    details: list[str] = []
    row_colors: list[str] = []
    for item in readiness:
        method = _text(item.get("method")) or "Not available"
        status = _text(item.get("status")) or "unavailable"
        assessed = _integer(item.get("assessed_record_count"))
        valid = _integer(item.get("valid_result_count"))
        methods.append(method)
        statuses.append(_label(status))
        coverage.append(
            f"{valid} / {assessed}"
            if valid is not None and assessed is not None
            else "Not available"
        )
        missing.append(_format_value(item.get("missing_inputs")))
        invalid.append(_format_value(item.get("invalid_inputs")))
        unsupported.append(_format_value(item.get("unsupported_conditions")))
        details.append(_text(item.get("detail")) or "Not available")
        row_colors.append(_status_color(status))
    figure = go.Figure(
        data=[
            go.Table(
                header={
                    "values": [
                        "Method",
                        "Readiness",
                        "Valid / assessed",
                        "Missing inputs",
                        "Invalid inputs",
                        "Unsupported conditions",
                        "Interpretation",
                    ],
                    "fill_color": SURFACE,
                    "font": {"color": TEXT, "size": 12},
                    "align": "left",
                    "height": 34,
                },
                cells={
                    "values": [methods, statuses, coverage, missing, invalid, unsupported, details],
                    "fill_color": [
                        [PANEL] * len(methods),
                        row_colors,
                        [PANEL] * len(methods),
                        [PANEL] * len(methods),
                        [PANEL] * len(methods),
                        [PANEL] * len(methods),
                        [PANEL] * len(methods),
                    ],
                    "font": {"color": TEXT, "size": 11},
                    "align": "left",
                    "height": 31,
                },
                columnwidth=[65, 90, 75, 155, 130, 150, 230],
            )
        ]
    )
    return _layout(figure, title="Method readiness — scores remain method-specific", height=390)


def angle_summary_figure(analysis: Any) -> go.Figure:
    """Show observed range of motion, 5th–95th percentile, and median in degrees."""

    payload = _payload(analysis)
    rows: list[dict[str, Any]] = []
    for item_raw in _sequence(payload.get("angle_analyses")):
        item = _mapping(item_raw)
        statistics = _mapping(item.get("statistics"))
        minimum = _finite(statistics.get("minimum_deg"))
        maximum = _finite(statistics.get("maximum_deg"))
        p05 = _finite(statistics.get("percentile_05_deg"))
        p95 = _finite(statistics.get("percentile_95_deg"))
        median = _finite(statistics.get("median_deg"))
        if minimum is None or maximum is None:
            continue
        rows.append(
            {
                "measurement": _human_measurement(item.get("measurement")),
                "minimum": minimum,
                "maximum": maximum,
                "p05": p05,
                "p95": p95,
                "median": median,
                "count": _integer(statistics.get("valid_observation_count")),
                "status": _text(item.get("status")) or "unavailable",
            }
        )
    if not rows:
        return _empty_figure(
            "Observed joint-angle summary",
            "No validated finite angle observations are available.",
        )
    rows.sort(key=lambda row: str(row["measurement"]))
    labels = [str(row["measurement"]) for row in rows]
    hover_range = [
        (
            f"{row['measurement']}<br>Observed min: {row['minimum']:.2f}°"
            f"<br>Observed max: {row['maximum']:.2f}°"
            f"<br>Valid observations: {_format_value(row['count'])}"
            f"<br>Status: {_label(row['status'])}"
        )
        for row in rows
    ]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Observed range (min–max)",
            y=labels,
            x=[row["maximum"] - row["minimum"] for row in rows],
            base=[row["minimum"] for row in rows],
            orientation="h",
            marker={"color": UNAVAILABLE, "opacity": 0.55},
            hovertext=hover_range,
            hoverinfo="text",
        )
    )
    percentile_rows = [row for row in rows if row["p05"] is not None and row["p95"] is not None]
    if percentile_rows:
        figure.add_trace(
            go.Bar(
                name="5th–95th percentile",
                y=[str(row["measurement"]) for row in percentile_rows],
                x=[float(row["p95"]) - float(row["p05"]) for row in percentile_rows],
                base=[row["p05"] for row in percentile_rows],
                orientation="h",
                marker={"color": BLUE, "opacity": 0.9},
                hovertemplate=(
                    "%{y}<br>5th percentile: %{base:.2f}°"
                    "<br>5th–95th span: %{x:.2f}°<extra></extra>"
                ),
            )
        )
    median_rows = [row for row in rows if row["median"] is not None]
    if median_rows:
        figure.add_trace(
            go.Scatter(
                name="Median",
                y=[str(row["measurement"]) for row in median_rows],
                x=[row["median"] for row in median_rows],
                mode="markers",
                marker={"color": TEXT, "size": 8, "symbol": "diamond"},
                hovertemplate="%{y}<br>Median: %{x:.2f}°<extra></extra>",
            )
        )
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title="Observed angle (degrees)")
    figure.update_yaxes(autorange="reversed")
    return _layout(
        figure,
        title="Observed joint-angle distribution and range of motion (descriptive)",
        height=max(390, 45 * len(rows) + 160),
    )


def posture_band_exposure_figure(analysis: Any) -> go.Figure:
    """Show official-method posture-band exposure without calculating a new score."""

    payload = _payload(analysis)
    bands: list[dict[str, Any]] = []
    for angle_raw in _sequence(payload.get("angle_analyses")):
        angle = _mapping(angle_raw)
        for band_raw in _sequence(angle.get("band_exposure")):
            band = _mapping(band_raw)
            count = _integer(band.get("observation_count"))
            if count is None or count <= 0:
                continue
            bands.append(
                {
                    "method": _text(band.get("method")) or "Not available",
                    "measurement": _human_measurement(band.get("measurement")),
                    "band_id": _text(band.get("band_id")) or "Not available",
                    "label": _text(band.get("label")) or "Not available",
                    "non_neutral": band.get("is_non_neutral") is True,
                    "count": count,
                    "duration": _finite(band.get("duration_sec")),
                    "percent": _finite(band.get("percent_of_covered_duration")),
                }
            )
    if not bands:
        return _empty_figure(
            "Official posture-band exposure",
            "No valid RULA or REBA posture-band observations are available.",
        )
    # Sorted by exposure, not alphabetically. A real trial produces forty-odd
    # (method, measurement, band) rows; ordering them by method name put the
    # longest-held non-neutral posture in the middle of a chart nearly two
    # thousand pixels tall, where nothing could be compared against anything.
    # Non-neutral bands break ties so the rows that matter cannot be pushed below
    # a neutral band with the same exposure.
    bands.sort(
        key=lambda row: (
            -(row["percent"] if row["percent"] is not None else -1.0),
            -row["count"],
            not row["non_neutral"],
            str(row["measurement"]),
        )
    )
    duration_rows = [row for row in bands if row["percent"] is not None]
    use_duration = bool(duration_rows)
    displayed = duration_rows if use_duration else bands
    omitted_for_length = max(0, len(displayed) - MAXIMUM_BAND_ROWS)
    displayed = displayed[:MAXIMUM_BAND_ROWS]
    labels = [f"{row['method']} · {row['measurement']} · {row['label']}" for row in displayed]
    x_values = (
        [float(row["percent"]) for row in displayed]
        if use_duration
        else [row["count"] for row in displayed]
    )
    hover = [
        (
            f"{row['method']} · {row['measurement']}<br>{row['label']}"
            f"<br>Observed frames: {row['count']}"
            f"<br>Duration: {_format_number(row['duration'], suffix=' s')}"
            f"<br>Covered-time share: {_format_number(row['percent'], suffix='%')}"
            "<br>Descriptive band exposure; not a new risk score"
        )
        for row in displayed
    ]
    figure = go.Figure(
        go.Bar(
            x=x_values,
            y=labels,
            orientation="h",
            marker={"color": [RED if row["non_neutral"] else BLUE for row in displayed]},
            customdata=[[row["method"], row["measurement"], row["band_id"]] for row in displayed],
            hovertext=hover,
            hoverinfo="text",
        )
    )
    notes = []
    if use_duration and len(duration_rows) < len(bands):
        notes.append(
            f"{len(bands) - len(duration_rows)} observed band(s) lack defensible duration "
            "and are omitted from this time-based view."
        )
    if omitted_for_length:
        notes.append(
            f"Showing the {MAXIMUM_BAND_ROWS} most-exposed bands; {omitted_for_length} "
            "further observed band(s) are in the downloadable evidence table."
        )
    if notes:
        figure.add_annotation(
            text="  ".join(notes),
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 11},
            align="left",
        )
    figure.update_xaxes(
        title="Share of covered duration (%)" if use_duration else "Observed frame count",
        range=[0, 100] if use_duration else None,
    )
    figure.update_yaxes(autorange="reversed")
    return _layout(
        figure,
        title="Most-exposed posture bands — RULA/REBA separate; not a score",
        height=max(390, 34 * len(displayed) + 180),
    )


def data_quality_coverage_figure(analysis: Any) -> go.Figure:
    """Summarize record, pose, calibration, camera, and timestamp coverage evidence."""

    payload = _payload(analysis)
    quality = _mapping(payload.get("data_quality"))
    if not quality:
        return _empty_figure(
            "Data quality and temporal coverage",
            "No data-quality summary was supplied.",
        )
    coverage = _mapping(quality.get("time_coverage"))
    selected = _integer(quality.get("selected_record_count"))
    if selected == 0:
        return _empty_figure(
            "Data quality and temporal coverage",
            "No records matched the requested participant or selection.",
        )
    metrics = [
        "Validity status",
        "Input records",
        "Selected records",
        "Usable angle observations",
        "Mean pose-model visibility",
        "Minimum pose-model visibility",
        "Calibration status",
        "Camera-view status",
        "Camera count",
        "Timestamp count",
        "Timestamp gaps",
        "Coverage of observed span",
    ]
    values = [
        _label(quality.get("status")),
        _format_value(quality.get("input_record_count")),
        _format_value(quality.get("selected_record_count")),
        _format_value(quality.get("usable_angle_observation_count")),
        _format_number(quality.get("mean_pose_visibility"), digits=3),
        _format_number(quality.get("minimum_pose_visibility"), digits=3),
        _format_value(quality.get("calibration_statuses")),
        _format_value(quality.get("camera_view_statuses")),
        _format_value(quality.get("camera_count")),
        _format_value(coverage.get("timestamp_count")),
        _format_value(coverage.get("gap_count")),
        _format_number(coverage.get("coverage_percent"), suffix="%"),
    ]
    figure = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.12,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
    )
    figure.add_trace(
        go.Table(
            header={
                "values": ["Quality field", "Observed value"],
                "fill_color": SURFACE,
                "font": {"color": TEXT},
                "align": "left",
            },
            cells={
                "values": [metrics, values],
                "fill_color": PANEL,
                "font": {"color": TEXT},
                "align": "left",
                "height": 28,
            },
        ),
        row=1,
        col=1,
    )
    covered = _finite(coverage.get("covered_duration_sec"))
    gaps = _finite(coverage.get("gap_duration_sec"))
    if covered is not None:
        figure.add_trace(
            go.Bar(
                name="Covered duration",
                x=[covered],
                y=["Observed span"],
                orientation="h",
                marker_color=BLUE,
                hovertemplate="Covered: %{x:.2f} s<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if gaps is not None:
        figure.add_trace(
            go.Bar(
                name="Excluded gaps",
                x=[gaps],
                y=["Observed span"],
                orientation="h",
                marker_color=RED,
                hovertemplate="Gaps: %{x:.2f} s<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if covered is None and gaps is None:
        figure.add_annotation(
            text="Timed coverage is not available.",
            x=0.5,
            y=0.08,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED},
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title="Seconds represented by validated timing", row=2, col=1)
    return _layout(figure, title="Data quality and temporal coverage", height=650)


def camera_evidence_figure(analysis: Any) -> go.Figure:
    """Show source contribution and descriptive inter-camera angle range."""

    payload = _payload(analysis)
    camera = _mapping(payload.get("camera_sources"))
    contributions = [_mapping(item) for item in _sequence(camera.get("contributions"))]
    agreement = [_mapping(item) for item in _sequence(camera.get("agreement"))]
    if not contributions and not agreement:
        return _empty_figure(
            "Camera contribution and agreement",
            "Per-camera source or paired-angle evidence is not available; no zero agreement is inferred.",
        )
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.16,
        subplot_titles=("Selected source contribution", "Paired-camera angle range"),
    )
    if contributions:
        measurements = sorted(
            {_human_measurement(item.get("measurement")) for item in contributions}
        )
        sources = sorted(
            {_text(item.get("camera_source")) or "Not available" for item in contributions}
        )
        # Series identity comes from the categorical slots, assigned in order.
        # This list previously mixed status hues in, so a camera could be drawn
        # in the colour that elsewhere means "critical".
        color_cycle = list(SERIES)
        for index, source in enumerate(sources):
            values: list[float | None] = []
            for measurement in measurements:
                matched = next(
                    (
                        _finite(item.get("percent_of_valid_observations"))
                        for item in contributions
                        if _human_measurement(item.get("measurement")) == measurement
                        and (_text(item.get("camera_source")) or "Not available") == source
                    ),
                    None,
                )
                values.append(matched)
            figure.add_trace(
                go.Bar(
                    name=source.title(),
                    x=values,
                    y=measurements,
                    orientation="h",
                    marker_color=color_cycle[index % len(color_cycle)],
                    hovertemplate=(
                        f"{source}<br>%{{y}}<br>Contribution: %{{x:.1f}}%<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
    else:
        figure.add_annotation(
            text="Source contribution not available",
            x=0.2,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED},
        )
    if agreement:
        labels = [_human_measurement(item.get("measurement")) for item in agreement]
        p95 = [_finite(item.get("percentile_95_range_deg")) for item in agreement]
        maximum = [_finite(item.get("maximum_range_deg")) for item in agreement]
        figure.add_trace(
            go.Scatter(
                name="95th-percentile range",
                x=p95,
                y=labels,
                mode="markers",
                marker={"color": BLUE, "size": 10},
                hovertemplate="%{y}<br>95th-percentile range: %{x:.2f}°<extra></extra>",
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                name="Maximum range",
                x=maximum,
                y=labels,
                mode="markers",
                marker={"color": RED, "size": 10, "symbol": "diamond"},
                hovertemplate="%{y}<br>Maximum range: %{x:.2f}°<extra></extra>",
            ),
            row=1,
            col=2,
        )
    else:
        figure.add_annotation(
            text="Paired-camera angles not persisted; agreement is not available",
            x=0.8,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED},
            align="center",
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title="Contribution (%)", range=[0, 100], row=1, col=1)
    figure.update_xaxes(title="Absolute per-frame camera range (degrees)", row=1, col=2)
    return _layout(
        figure,
        title="Camera evidence — contribution and descriptive agreement remain separate",
        height=520,
    )


def _escape_markdown(value: Any) -> str:
    return _format_value(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_markdown(value) for value in row) + " |" for row in rows)
    return lines


def professional_markdown_report(analysis: Any) -> str:
    """Return a deterministic, human-readable evidence report."""

    payload = _payload(analysis)
    lines = [
        "# Professional Ergonomics Analysis Evidence Report",
        "",
        f"Schema: `{_text(payload.get('schema_version')) or 'Not available'}`",
        "",
        "> This report preserves RULA, REBA, and RNLE as separate methods. Posture-band exposure and candidate repetitions are descriptive evidence, not a synthetic overall score, diagnosis, or pain estimate.",
        "",
        "## Data quality",
        "",
    ]
    quality = _mapping(payload.get("data_quality"))
    coverage = _mapping(quality.get("time_coverage"))
    if not quality:
        lines.append("Not available — no data-quality summary was supplied.")
    else:
        lines.extend(
            _markdown_table(
                ("Field", "Observed value"),
                (
                    ("Status", _label(quality.get("status"))),
                    ("Selected records", quality.get("selected_record_count")),
                    ("Usable angle observations", quality.get("usable_angle_observation_count")),
                    ("Mean pose visibility", quality.get("mean_pose_visibility")),
                    ("Calibration", quality.get("calibration_statuses")),
                    ("Camera views", quality.get("camera_view_statuses")),
                    ("Camera count", quality.get("camera_count")),
                    ("Covered duration (s)", coverage.get("covered_duration_sec")),
                    ("Excluded gap duration (s)", coverage.get("gap_duration_sec")),
                    ("Coverage (%)", coverage.get("coverage_percent")),
                ),
            )
        )
        warnings = [_text(item) for item in _sequence(quality.get("warnings")) if _text(item)]
        if warnings:
            lines.extend(["", "Quality limitations:"])
            lines.extend(f"- {item}" for item in warnings)

    lines.extend(["", "## Method readiness", ""])
    readiness = [_mapping(item) for item in _sequence(payload.get("method_readiness"))]
    if not readiness:
        lines.append("Not available — no method-readiness results were supplied.")
    else:
        lines.extend(
            _markdown_table(
                (
                    "Method",
                    "Status",
                    "Valid / assessed",
                    "Missing inputs",
                    "Invalid inputs",
                    "Unsupported conditions",
                    "Detail",
                ),
                [
                    (
                        item.get("method"),
                        _label(item.get("status")),
                        (
                            f"{item.get('valid_result_count')} / {item.get('assessed_record_count')}"
                            if item.get("valid_result_count") is not None
                            and item.get("assessed_record_count") is not None
                            else "Not available"
                        ),
                        item.get("missing_inputs"),
                        item.get("invalid_inputs"),
                        item.get("unsupported_conditions"),
                        item.get("detail"),
                    )
                    for item in readiness
                ],
            )
        )

    lines.extend(["", "## Validated angle summaries", ""])
    angles = [_mapping(item) for item in _sequence(payload.get("angle_analyses"))]
    available_angles = [
        item
        for item in angles
        if _finite(_mapping(item.get("statistics")).get("median_deg")) is not None
    ]
    if not available_angles:
        lines.append("Not available — no validated finite angle observations were supplied.")
    else:
        lines.extend(
            _markdown_table(
                (
                    "Measurement",
                    "Status",
                    "Valid observations",
                    "Minimum (°)",
                    "Median (°)",
                    "95th percentile (°)",
                    "Maximum (°)",
                ),
                [
                    (
                        _human_measurement(item.get("measurement")),
                        _label(item.get("status")),
                        _mapping(item.get("statistics")).get("valid_observation_count"),
                        _mapping(item.get("statistics")).get("minimum_deg"),
                        _mapping(item.get("statistics")).get("median_deg"),
                        _mapping(item.get("statistics")).get("percentile_95_deg"),
                        _mapping(item.get("statistics")).get("maximum_deg"),
                    )
                    for item in sorted(
                        available_angles,
                        key=lambda value: str(value.get("measurement")),
                    )
                ],
            )
        )

    lines.extend(["", "## Official posture-band exposure", ""])
    observed_bands: list[Mapping[str, Any]] = []
    for angle in angles:
        observed_bands.extend(
            band
            for raw in _sequence(angle.get("band_exposure"))
            if (band := _mapping(raw)) and (_integer(band.get("observation_count")) or 0) > 0
        )
    if not observed_bands:
        lines.append(
            "Not available — no valid RULA or REBA posture-band observations were supplied."
        )
    else:
        observed_bands.sort(
            key=lambda value: (
                str(value.get("method")),
                str(value.get("measurement")),
                str(value.get("band_id")),
            )
        )
        lines.extend(
            _markdown_table(
                (
                    "Method",
                    "Measurement",
                    "Band",
                    "Non-neutral",
                    "Observed frames",
                    "Duration (s)",
                    "Covered duration (%)",
                ),
                [
                    (
                        band.get("method"),
                        _human_measurement(band.get("measurement")),
                        band.get("label"),
                        band.get("is_non_neutral"),
                        band.get("observation_count"),
                        band.get("duration_sec"),
                        band.get("percent_of_covered_duration"),
                    )
                    for band in observed_bands
                ],
            )
        )

    lines.extend(["", "## Candidate repetitions", ""])
    repetitions = [_mapping(item) for item in _sequence(payload.get("repetition_candidates"))]
    if not repetitions:
        lines.append("Not available — no repetition-candidate analysis was supplied.")
    else:
        lines.extend(
            _markdown_table(
                (
                    "Measurement",
                    "Status",
                    "Candidate count",
                    "Rate (/min)",
                    "Analyzed duration (s)",
                    "Limitations",
                ),
                [
                    (
                        _human_measurement(item.get("measurement")),
                        _label(item.get("status")),
                        item.get("candidate_count"),
                        item.get("rate_per_minute"),
                        item.get("analysis_duration_sec"),
                        item.get("limitations"),
                    )
                    for item in repetitions
                ],
            )
        )

    lines.extend(["", "## Recommendations linked to evidence", ""])
    recommendations = [_mapping(item) for item in _sequence(payload.get("recommendations"))]
    if not recommendations:
        lines.append("No evidence-linked recommendations are available.")
    else:
        for item in recommendations:
            lines.extend(
                [
                    f"### {_text(item.get('title')) or 'Recommendation'}",
                    "",
                    f"- Priority: `{_text(item.get('priority')) or 'Not available'}`",
                    f"- Observed fact: {_text(item.get('observed_fact')) or 'Not available'}",
                    f"- Interpretation: {_text(item.get('rationale')) or 'Not available'}",
                    f"- Related methods: {_format_value(item.get('related_methods'))}",
                    f"- Sources: {_format_value(item.get('source_urls'))}",
                    "",
                ]
            )

    lines.extend(["## Method and model limitations", ""])
    warnings = [_text(item) for item in _sequence(payload.get("warnings")) if _text(item)]
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- Not available — no explicit method limitations were supplied.")

    lines.extend(["", "## Authoritative sources", ""])
    provenance = [_mapping(item) for item in _sequence(payload.get("provenance"))]
    if provenance:
        for item in sorted(provenance, key=lambda value: str(value.get("identifier"))):
            identifier = _text(item.get("identifier")) or "Source"
            title = _text(item.get("title")) or "Untitled source"
            url = _text(item.get("url"))
            scope = _text(item.get("scope")) or "Scope not available"
            source = f"[{title}]({url})" if url else title
            lines.append(f"- **{identifier}:** {source} — {scope}")
    else:
        lines.append("- Not available — provenance was not supplied.")
    return "\n".join(lines).rstrip() + "\n"


def _csv_row(
    *,
    section: str,
    method: Any = None,
    measurement: Any = None,
    metric: Any = None,
    band_id: Any = None,
    value: Any = None,
    units: Any = None,
    status: Any = None,
    frame_number: Any = None,
    timestamp_sec: Any = None,
    source_url: Any = None,
    detail: Any = None,
) -> dict[str, Any]:
    return {
        "section": section,
        "method": method,
        "measurement": measurement,
        "metric": metric,
        "band_id": band_id,
        "value": value,
        "units": units,
        "status": status,
        "frame_number": frame_number,
        "timestamp_sec": timestamp_sec,
        "source_url": source_url,
        "detail": detail,
    }


def professional_csv_rows(analysis: Any) -> list[dict[str, Any]]:
    """Flatten professional-analysis evidence into stable, CSV-ready rows."""

    payload = _payload(analysis)
    rows: list[dict[str, Any]] = []
    quality = _mapping(payload.get("data_quality"))
    coverage = _mapping(quality.get("time_coverage"))
    for metric in (
        "input_record_count",
        "selected_record_count",
        "usable_angle_observation_count",
        "mean_pose_visibility",
        "minimum_pose_visibility",
        "camera_count",
    ):
        rows.append(
            _csv_row(
                section="data_quality",
                metric=metric,
                value=quality.get(metric),
                status=quality.get("status"),
            )
        )
    for metric in (
        "span_sec",
        "covered_duration_sec",
        "gap_count",
        "gap_duration_sec",
        "coverage_percent",
    ):
        rows.append(
            _csv_row(
                section="time_coverage",
                metric=metric,
                value=coverage.get(metric),
                units="percent"
                if metric == "coverage_percent"
                else "seconds"
                if "sec" in metric
                else None,
                status=quality.get("status"),
            )
        )

    for item_raw in _sequence(payload.get("method_readiness")):
        item = _mapping(item_raw)
        provenance = _mapping(item.get("provenance"))
        method = item.get("method")
        rows.extend(
            [
                _csv_row(
                    section="method_readiness",
                    method=method,
                    metric="valid_result_count",
                    value=item.get("valid_result_count"),
                    status=item.get("status"),
                    source_url=provenance.get("url"),
                    detail=item.get("detail"),
                ),
                _csv_row(
                    section="method_readiness",
                    method=method,
                    metric="assessed_record_count",
                    value=item.get("assessed_record_count"),
                    status=item.get("status"),
                    source_url=provenance.get("url"),
                ),
                _csv_row(
                    section="method_readiness",
                    method=method,
                    metric="missing_inputs",
                    value="; ".join(str(value) for value in _sequence(item.get("missing_inputs"))),
                    status=item.get("status"),
                    source_url=provenance.get("url"),
                ),
            ]
        )

    for angle_raw in _sequence(payload.get("angle_analyses")):
        angle = _mapping(angle_raw)
        measurement = angle.get("measurement")
        statistics = _mapping(angle.get("statistics"))
        peak = _mapping(statistics.get("absolute_peak"))
        for metric in (
            "valid_observation_count",
            "excluded_observation_count",
            "minimum_deg",
            "maximum_deg",
            "mean_deg",
            "median_deg",
            "standard_deviation_deg",
            "percentile_05_deg",
            "percentile_95_deg",
        ):
            rows.append(
                _csv_row(
                    section="angle_summary",
                    measurement=measurement,
                    metric=metric,
                    value=statistics.get(metric),
                    units=None if metric.endswith("count") else "degrees",
                    status=angle.get("status"),
                    frame_number=peak.get("frame_number") if metric == "maximum_deg" else None,
                    timestamp_sec=peak.get("timestamp_sec") if metric == "maximum_deg" else None,
                )
            )
        for band_raw in _sequence(angle.get("band_exposure")):
            band = _mapping(band_raw)
            provenance = _mapping(band.get("provenance"))
            evidence = _mapping(band.get("evidence"))
            for metric, value, units in (
                ("observation_count", band.get("observation_count"), "frames"),
                ("duration_sec", band.get("duration_sec"), "seconds"),
                (
                    "percent_of_covered_duration",
                    band.get("percent_of_covered_duration"),
                    "percent",
                ),
            ):
                rows.append(
                    _csv_row(
                        section="posture_band_exposure",
                        method=band.get("method"),
                        measurement=measurement,
                        metric=metric,
                        band_id=band.get("band_id"),
                        value=value,
                        units=units,
                        status="non_neutral_observed"
                        if band.get("is_non_neutral")
                        else "neutral_band",
                        frame_number=evidence.get("frame_number"),
                        timestamp_sec=evidence.get("timestamp_sec"),
                        source_url=provenance.get("url"),
                        detail=band.get("label"),
                    )
                )

    for item_raw in _sequence(payload.get("repetition_candidates")):
        item = _mapping(item_raw)
        provenance = _mapping(item.get("provenance"))
        for metric, value, units in (
            ("candidate_count", item.get("candidate_count"), "candidate events"),
            ("rate_per_minute", item.get("rate_per_minute"), "candidate events/minute"),
            ("analysis_duration_sec", item.get("analysis_duration_sec"), "seconds"),
        ):
            rows.append(
                _csv_row(
                    section="repetition_candidates",
                    measurement=item.get("measurement"),
                    metric=metric,
                    value=value,
                    units=units,
                    status=item.get("status"),
                    source_url=provenance.get("url"),
                    detail="; ".join(str(value) for value in _sequence(item.get("limitations"))),
                )
            )

    camera = _mapping(payload.get("camera_sources"))
    for item_raw in _sequence(camera.get("contributions")):
        item = _mapping(item_raw)
        rows.append(
            _csv_row(
                section="camera_contribution",
                measurement=item.get("measurement"),
                metric="percent_of_valid_observations",
                value=item.get("percent_of_valid_observations"),
                units="percent",
                status=camera.get("contribution_status"),
                detail=item.get("camera_source"),
            )
        )
    for item_raw in _sequence(camera.get("agreement")):
        item = _mapping(item_raw)
        evidence = _mapping(item.get("maximum_range_evidence"))
        for metric in (
            "mean_range_deg",
            "median_range_deg",
            "percentile_95_range_deg",
            "maximum_range_deg",
        ):
            rows.append(
                _csv_row(
                    section="camera_agreement",
                    measurement=item.get("measurement"),
                    metric=metric,
                    value=item.get(metric),
                    units="degrees",
                    status=camera.get("agreement_status"),
                    frame_number=evidence.get("frame_number")
                    if metric == "maximum_range_deg"
                    else None,
                    timestamp_sec=evidence.get("timestamp_sec")
                    if metric == "maximum_range_deg"
                    else None,
                    detail=item.get("interpretation"),
                )
            )

    for item_raw in _sequence(payload.get("recommendations")):
        item = _mapping(item_raw)
        evidence_items = [_mapping(value) for value in _sequence(item.get("evidence"))]
        evidence = evidence_items[0] if evidence_items else {}
        urls = _sequence(item.get("source_urls"))
        rows.append(
            _csv_row(
                section="recommendation",
                method="; ".join(str(value) for value in _sequence(item.get("related_methods"))),
                measurement=evidence.get("measurement"),
                metric=item.get("recommendation_id"),
                value=None,
                status=item.get("priority"),
                frame_number=evidence.get("frame_number"),
                timestamp_sec=evidence.get("timestamp_sec"),
                source_url="; ".join(str(value) for value in urls),
                detail=(
                    f"{_text(item.get('observed_fact')) or 'Not available'} "
                    f"{_text(item.get('rationale')) or 'Not available'}"
                ),
            )
        )
    return rows


__all__ = [
    "CSV_COLUMNS",
    "angle_summary_figure",
    "camera_evidence_figure",
    "data_quality_coverage_figure",
    "method_readiness_figure",
    "posture_band_exposure_figure",
    "professional_csv_rows",
    "professional_markdown_report",
]
