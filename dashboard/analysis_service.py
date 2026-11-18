"""Pure dashboard analysis helpers for current v4 assessment results.

These functions interpret already-calculated ergonomic results; they never
invent missing task facts and never treat a posture risk region as reported pain.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.skeleton import POSE_CONNECTIONS as SKELETON_CONNECTIONS
from core.skeleton import augment_with_virtual_joints
from dashboard.viz_theme import (
    AXIS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PAGE,
    SERIES,
    STATUS,
)


class DashboardDataError(ValueError):
    """Raised when a current result bundle is missing or malformed."""


REGION_LABELS = {
    "neck": "Neck",
    "trunk": "Trunk",
    "left_shoulder": "Left shoulder",
    "right_shoulder": "Right shoulder",
    "left_elbow": "Left elbow",
    "right_elbow": "Right elbow",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
    "legs": "Legs and balance",
}

REGION_LANDMARKS = {
    "neck": {0, 7, 8, 11, 12},
    "trunk": {11, 12, 23, 24},
    "left_shoulder": {11, 13},
    "right_shoulder": {12, 14},
    "left_elbow": {11, 13, 15},
    "right_elbow": {12, 14, 16},
    "left_wrist": {15, 17, 19, 21},
    "right_wrist": {16, 18, 20, 22},
    "legs": {23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
}

# Drawn with a synthesised neck and spine; see core/skeleton.py for why the
# ear-to-shoulder diagonals this used to draw were not anatomy.
POSE_CONNECTIONS = SKELETON_CONNECTIONS


def load_result_bundle(result_dir: str | Path) -> dict[str, Any]:
    """Load a complete v4 bundle and fail clearly on missing or invalid files."""
    root = Path(result_dir)
    jsonl_path = root / "frame_results.jsonl"
    summary_path = root / "video_summary.json"
    config_path = root / "run_config.json"
    missing = [
        str(path.name) for path in (jsonl_path, summary_path, config_path) if not path.is_file()
    ]
    if missing:
        raise DashboardDataError("Current analysis is incomplete; missing: " + ", ".join(missing))
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        records = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DashboardDataError(
                        f"Could not read current analysis: malformed frame result "
                        f"at line {line_number}: {exc.msg}"
                    ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read current analysis: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise DashboardDataError("Summary and run configuration must be JSON objects")
    if not records:
        raise DashboardDataError("Current analysis has no frame-level pose observations")
    if any(not isinstance(record, dict) for record in records):
        raise DashboardDataError("Frame results must contain one JSON object per line")
    return {"dir": str(root), "summary": summary, "config": config, "records": records}


def _selected(method_result: dict[str, Any]) -> dict[str, Any] | None:
    value = method_result.get("selected")
    return value if isinstance(value, dict) else None


def _side_score(method_result: dict[str, Any], side: str) -> float | None:
    side_result = method_result.get(side)
    if not isinstance(side_result, dict):
        return None
    value = side_result.get("final_score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def record_dataframe(records: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the analysis fields used for filtering and plotting."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "frame_number": int(record.get("frame_number", 0)),
            "timestamp_sec": float(record.get("timestamp_sec", 0.0)),
            "person_identifier": record.get("person_identifier"),
            "pose_visibility": record.get("pose_quality", {}).get("mean_keypoint_visibility"),
        }
        for method in ("rula", "reba"):
            result = record.get(method, {})
            selected = _selected(result)
            row[f"{method}_score"] = selected.get("final_score") if selected else None
            row[f"{method}_status"] = (
                selected.get("validity_status")
                if selected
                else result.get("left", {}).get("validity_status")
            )
            row[f"{method}_selected_side"] = result.get("selected_side")
            row[f"{method}_left_score"] = _side_score(result, "left")
            row[f"{method}_right_score"] = _side_score(result, "right")
        angles = record.get("angles", {})
        if isinstance(angles, dict):
            for name, value in angles.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[f"angle_{name}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["frame_number", "person_identifier"])


def _component(method: dict[str, Any], side: str, name: str, maximum: float) -> float | None:
    side_result = method.get(side, {})
    scores = side_result.get("intermediate_scores", {}) if isinstance(side_result, dict) else {}
    value = scores.get(name) if isinstance(scores, dict) else None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not np.isfinite(float(value)):
        return None
    return float(np.clip(float(value) / maximum, 0.0, 1.0))


def _maximum_available(*values: float | None) -> float | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


def body_region_risk(record: dict[str, Any]) -> dict[str, float | None]:
    """Map method component scores to relative screening intensity (0 to 1)."""
    rula = record.get("rula", {})
    reba = record.get("reba", {})
    return {
        "neck": _maximum_available(
            _component(rula, "left", "neck", 6),
            _component(reba, "left", "neck", 3),
        ),
        "trunk": _maximum_available(
            _component(rula, "left", "trunk", 6),
            _component(reba, "left", "trunk", 5),
        ),
        "left_shoulder": _maximum_available(
            _component(rula, "left", "upper_arm", 6),
            _component(reba, "left", "upper_arm", 6),
        ),
        "right_shoulder": _maximum_available(
            _component(rula, "right", "upper_arm", 6),
            _component(reba, "right", "upper_arm", 6),
        ),
        "left_elbow": _maximum_available(
            _component(rula, "left", "lower_arm", 3),
            _component(reba, "left", "lower_arm", 2),
        ),
        "right_elbow": _maximum_available(
            _component(rula, "right", "lower_arm", 3),
            _component(reba, "right", "lower_arm", 2),
        ),
        "left_wrist": _maximum_available(
            _component(rula, "left", "wrist", 4),
            _component(reba, "left", "wrist", 3),
        ),
        "right_wrist": _maximum_available(
            _component(rula, "right", "wrist", 4),
            _component(reba, "right", "wrist", 3),
        ),
        "legs": _maximum_available(
            _component(rula, "left", "legs", 2),
            _component(reba, "left", "legs", 4),
        ),
    }


def severity_label(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return "Unavailable"
    if value >= 0.75:
        return "High"
    if value >= 0.5:
        return "Elevated"
    if value > 0:
        return "Low"
    return "Unavailable"


def severity_color(value: float | None) -> str:
    """Colour for a normalised severity, from the shared status palette.

    These were a third set of hand-picked colours, unrelated to the ones the
    charts and the risk bands used, so the same idea wore three different reds
    depending on which figure you were looking at. Severity now speaks the same
    visual language as everything else, and an unavailable value is deliberately
    the muted ink rather than a colour that could be read as "low risk".
    """

    if value is None or not np.isfinite(float(value)):
        return INK_MUTED
    if value >= 0.75:
        return STATUS["critical"]
    if value >= 0.5:
        return STATUS["warning"]
    if value > 0:
        return STATUS["good"]
    return INK_MUTED


def body_map_figure(region_values: dict[str, float | None]) -> go.Figure:
    """Create a compact clickable posture-risk map (not a pain diagnosis)."""
    positions = {
        "neck": (0.0, 7.7),
        "trunk": (0.0, 5.8),
        "left_shoulder": (-1.25, 6.8),
        "right_shoulder": (1.25, 6.8),
        "left_elbow": (-2.0, 5.2),
        "right_elbow": (2.0, 5.2),
        "left_wrist": (-2.45, 3.7),
        "right_wrist": (2.45, 3.7),
        "legs": (0.0, 1.7),
    }
    skeleton = [
        ((0, 8.5), (0, 7.7)),
        ((-1.25, 6.8), (1.25, 6.8)),
        ((0, 7.7), (0, 3.8)),
        ((-1.25, 6.8), (-2.0, 5.2)),
        ((-2.0, 5.2), (-2.45, 3.7)),
        ((1.25, 6.8), (2.0, 5.2)),
        ((2.0, 5.2), (2.45, 3.7)),
        ((0, 3.8), (-1.0, 0.0)),
        ((0, 3.8), (1.0, 0.0)),
    ]
    figure = go.Figure()
    for start, end in skeleton:
        figure.add_trace(
            go.Scatter(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                mode="lines",
                line={"color": INK_SECONDARY, "width": 5},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    keys = list(positions)
    values: list[float | None] = []
    for key in keys:
        raw_value = region_values.get(key)
        values.append(
            float(raw_value)
            if isinstance(raw_value, (int, float))
            and not isinstance(raw_value, bool)
            and np.isfinite(float(raw_value))
            else None
        )
    figure.add_trace(
        go.Scatter(
            x=[positions[key][0] for key in keys],
            y=[positions[key][1] for key in keys],
            mode="markers",
            customdata=keys,
            text=[
                f"{REGION_LABELS[key]}<br>{severity_label(value)} screening intensity"
                for key, value in zip(keys, values, strict=True)
            ],
            hovertemplate="%{text}<extra></extra>",
            marker={
                "color": [severity_color(value) for value in values],
                "size": [24 if value is None else 24 + 18 * value for value in values],
                "line": {"color": "white", "width": 2},
            },
            showlegend=False,
        )
    )
    figure.add_shape(
        type="circle",
        x0=-0.58,
        x1=0.58,
        y0=8.1,
        y1=9.25,
        line={"color": INK_MUTED, "width": 4},
        fillcolor="rgba(168,182,206,0.08)",
    )
    figure.update_layout(
        height=470,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [-3.2, 3.2], "fixedrange": True},
        yaxis={"visible": False, "range": [-0.5, 9.6], "fixedrange": True},
        dragmode=False,
    )
    return figure


def _world_landmarks_for_record(
    record: dict[str, Any], camera_index: int = 0
) -> dict[int, list[float]]:
    value = record.get("pose_landmarks_world")
    if not isinstance(value, dict):
        by_camera = record.get("pose_landmarks_world_by_camera", {})
        value = by_camera.get(str(camera_index)) if isinstance(by_camera, dict) else None
    if not isinstance(value, dict):
        return {}
    landmarks: dict[int, list[float]] = {}
    for key, point in value.items():
        if (
            isinstance(point, list)
            and len(point) >= 3
            and all(isinstance(item, (int, float)) for item in point[:3])
            and all(np.isfinite(float(item)) for item in point[:3])
        ):
            try:
                landmark_index = int(key)
            except (TypeError, ValueError):
                continue
            landmarks[landmark_index] = [float(item) for item in point]
    return landmarks


# Landmarks that can be in contact with the floor, in preference order:
# heels and foot tips define stance; ankles cover frames where the feet
# themselves were not confidently tracked.
_FOOT_LANDMARKS = (29, 30, 31, 32, 27, 28)


def _stabilized_landmark_sequence(
    records: list[dict[str, Any]],
    camera_index: int,
    *,
    smoothing_window: int = 3,
    grounding_window: int = 7,
) -> list[dict[int, list[float]]]:
    """Return display-ready landmark frames: jitter-smoothed and floor-anchored.

    MediaPipe world landmarks are hip-centred: the origin is the hip midpoint on
    every frame, so a crouch moves the feet upward through space rather than the
    hips downward, and the figure appears to float. For display, each frame is
    translated vertically so the lowest foot landmark rests on y = 0, with the
    offset median-filtered over neighbouring frames so tracking noise does not
    shake the floor. A short moving average over each landmark removes
    frame-to-frame jitter.

    Display-only: stored records, angles and scores are never modified.
    """

    sequence = [_world_landmarks_for_record(record, camera_index) for record in records]
    if not sequence:
        return []

    # Jitter smoothing: average each landmark over a short centred window,
    # using only frames where that landmark exists.
    half = max(0, smoothing_window // 2)
    smoothed: list[dict[int, list[float]]] = []
    for position, landmarks in enumerate(sequence):
        frame: dict[int, list[float]] = {}
        neighbours = sequence[max(0, position - half) : position + half + 1]
        for index, point in landmarks.items():
            stack = [other[index][:3] for other in neighbours if index in other]
            if len(stack) > 1:
                frame[index] = [float(v) for v in np.mean(np.asarray(stack), axis=0)]
            else:
                frame[index] = [float(v) for v in point[:3]]
        smoothed.append(frame)

    # Grounding: per-frame lowest foot height, median-filtered so a single
    # mistracked foot cannot bounce the whole figure.
    offsets: list[float] = []
    for frame in smoothed:
        feet = [frame[index][1] for index in _FOOT_LANDMARKS if index in frame]
        offsets.append(min(feet) if feet else 0.0)
    ground_half = max(0, grounding_window // 2)
    for position, frame in enumerate(smoothed):
        window = offsets[max(0, position - ground_half) : position + ground_half + 1]
        offset = float(np.median(window)) if window else 0.0
        for point in frame.values():
            point[1] -= offset
    return smoothed


def _ground_plane_trace(x_range: list[float], depth_range: list[float]) -> go.Surface:
    """A faint floor at y = 0 so grounding is visible."""

    return go.Surface(
        x=np.linspace(x_range[0], x_range[1], 2),
        y=np.linspace(depth_range[0], depth_range[1], 2),
        z=np.zeros((2, 2)),
        showscale=False,
        opacity=0.25,
        surfacecolor=np.zeros((2, 2)),
        colorscale=[[0, AXIS], [1, AXIS]],
        hoverinfo="skip",
        name="floor",
    )


def _stick_figure_traces(
    landmarks: dict[int, list[float]],
    selected_region: str | None,
) -> list[go.Scatter3d]:
    if not landmarks:
        return []
    # C7, the head centre, and the pelvis are synthesised so the drawing has a
    # neck and a spine; the model supplies neither.
    drawable = augment_with_virtual_joints(landmarks)
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    line_z: list[float | None] = []
    for start, end in POSE_CONNECTIONS:
        if start not in drawable or end not in drawable:
            continue
        first, second = drawable[start], drawable[end]
        line_x.extend([first[0], second[0], None])
        line_y.extend([first[2], second[2], None])
        line_z.extend([first[1], second[1], None])
    elevated_landmarks: set[int] = set()

    if selected_region:
        elevated_landmarks.update(REGION_LANDMARKS.get(selected_region, set()))
    indices = sorted(landmarks)
    return [
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line={"color": SERIES[0], "width": 7},
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter3d(
            x=[landmarks[index][0] for index in indices],
            y=[landmarks[index][2] for index in indices],
            z=[landmarks[index][1] for index in indices],
            mode="markers",
            customdata=indices,
            marker={
                "size": [7 if index in elevated_landmarks else 4 for index in indices],
                "color": [
                    STATUS["critical"] if index in elevated_landmarks else INK_SECONDARY
                    for index in indices
                ],
                "line": {"color": PAGE, "width": 1},
            },
            hovertemplate="Landmark %{customdata}<extra></extra>",
            showlegend=False,
        ),
    ]


def stick_figure_3d(
    records: list[dict[str, Any]],
    *,
    selected_region: str | None = None,
    camera_index: int = 0,
    maximum_animation_frames: int = 180,
) -> go.Figure:
    """Animate stored canonical model-world landmarks as a 3-D stick figure."""
    usable = [record for record in records if _world_landmarks_for_record(record, camera_index)]
    if not usable:
        figure = go.Figure()
        figure.add_annotation(
            text=(
                "No stored 3-D landmarks are available. Re-run the analysis to "
                "create traceable model-world posture records."
            ),
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": INK_PRIMARY, "size": 16},
        )
        figure.update_layout(
            height=620,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            },
        )
        return figure
    if len(usable) > maximum_animation_frames:
        selected_indices = np.linspace(0, len(usable) - 1, maximum_animation_frames, dtype=int)
        usable = [usable[index] for index in np.unique(selected_indices)]
    timestamps = [
        float(record.get("timestamp_sec"))
        for record in usable
        if isinstance(record.get("timestamp_sec"), (int, float))
    ]
    timestamp_deltas = [
        right - left
        for left, right in zip(timestamps, timestamps[1:], strict=False)
        if 0 < right - left <= 2.0
    ]
    frame_duration_ms = (
        int(np.clip(np.median(timestamp_deltas) * 1000, 30, 1000)) if timestamp_deltas else 90
    )
    display_sequence = _stabilized_landmark_sequence(usable, camera_index)
    all_points = [point[:3] for frame in display_sequence for point in frame.values()]
    coordinates = np.asarray(all_points, dtype=float)
    x_range = [
        float(np.nanmin(coordinates[:, 0]) - 0.15),
        float(np.nanmax(coordinates[:, 0]) + 0.15),
    ]
    depth_range = [
        float(np.nanmin(coordinates[:, 2]) - 0.15),
        float(np.nanmax(coordinates[:, 2]) + 0.15),
    ]
    vertical_range = [
        min(-0.05, float(np.nanmin(coordinates[:, 1])) - 0.05),
        float(np.nanmax(coordinates[:, 1]) + 0.15),
    ]
    floor = _ground_plane_trace(x_range, depth_range)
    animation_frames = []
    for index, record in enumerate(usable):
        label = (
            f"Frame {int(record.get('frame_number', 0)):,} · "
            f"{float(record.get('timestamp_sec', 0.0)):.2f}s"
        )
        animation_frames.append(
            go.Frame(
                data=[floor, *_stick_figure_traces(display_sequence[index], selected_region)],
                name=str(index),
                layout=go.Layout(title={"text": label}),
            )
        )
    figure = go.Figure(
        data=[floor, *_stick_figure_traces(display_sequence[0], selected_region)],
        frames=animation_frames,
    )
    slider_steps = [
        {
            "args": [
                [str(index)],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": str(record.get("frame_number", index)),
            "method": "animate",
        }
        for index, record in enumerate(usable)
    ]
    figure.update_layout(
        title={
            "text": (
                f"Frame {int(usable[0].get('frame_number', 0)):,} · "
                f"{float(usable[0].get('timestamp_sec', 0.0)):.2f}s"
            ),
            "font": {"color": INK_PRIMARY},
        },
        height=650,
        margin={"l": 0, "r": 0, "t": 55, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK_SECONDARY},
        scene={
            "bgcolor": PAGE,
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 1.0}},
            "xaxis": {
                "title": "Lateral X (model m)",
                "range": x_range,
                "gridcolor": "#253554",
                "zerolinecolor": "#53698f",
            },
            "yaxis": {
                "title": "Depth Z (model m)",
                "range": depth_range,
                "gridcolor": "#253554",
                "zerolinecolor": "#53698f",
            },
            "zaxis": {
                "title": "Vertical Y (model m)",
                "range": vertical_range,
                "gridcolor": "#253554",
                "zerolinecolor": "#53698f",
            },
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Play movement",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Source frame "},
                "steps": slider_steps,
                "pad": {"t": 40},
            }
        ],
    )
    return figure


def risk_indicator_figure(score: float | int | None, method: str) -> go.Figure:
    method = method.upper()
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not np.isfinite(score):
        figure = go.Figure()
        figure.add_annotation(
            text=f"Worst {method}<br><b>Not available</b>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"color": INK_PRIMARY, "size": 18},
        )
        figure.update_layout(
            height=245,
            margin={"l": 25, "r": 25, "t": 30, "b": 20},
            paper_bgcolor="#080d18",
            xaxis={"visible": False},
            yaxis={"visible": False},
        )
        return figure

    if method == "RULA":
        axis_range = [0.5, 7.5]
        tick_values = list(range(1, 8))
        high_threshold = 5
        bands = [
            ([0.5, 2.5], "#123b32"),
            ([2.5, 4.5], "#253f66"),
            ([4.5, 6.5], "#6b3f13"),
            ([6.5, 7.5], "#5d1722"),
        ]
    else:
        axis_range = [0.5, 15.5]
        tick_values = list(range(1, 16))
        high_threshold = 8
        bands = [
            ([0.5, 1.5], "#123b32"),
            ([1.5, 3.5], "#183e58"),
            ([3.5, 7.5], "#253f66"),
            ([7.5, 10.5], "#6b3f13"),
            ([10.5, 15.5], "#5d1722"),
        ]

    value = float(score)
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": f"Worst {method}", "font": {"color": INK_PRIMARY}},
            number={
                "font": {"color": STATUS["critical"] if value >= high_threshold else SERIES[0]}
            },
            gauge={
                "axis": {
                    "range": axis_range,
                    "tickmode": "array",
                    "tickvals": tick_values,
                    "tickcolor": "#8fa9d4",
                },
                "bar": {"color": STATUS["critical"] if value >= high_threshold else SERIES[0]},
                "bgcolor": "#0c1425",
                "bordercolor": "#2a3d63",
                "steps": [{"range": score_range, "color": color} for score_range, color in bands],
            },
        )
    )
    figure.update_layout(
        height=245,
        margin={"l": 25, "r": 25, "t": 55, "b": 20},
        paper_bgcolor="#080d18",
        font={"color": INK_SECONDARY},
    )
    return figure


def risk_distribution_figure(records: list[dict[str, Any]], method: str) -> go.Figure:
    method_key = method.lower()
    maximum = 7 if method_key == "rula" else 15
    threshold = 5 if method_key == "rula" else 8
    scores = []
    for record in records:
        selected = _selected(record.get(method_key, {}))
        if selected and isinstance(selected.get("final_score"), (int, float)):
            scores.append(int(selected["final_score"]))
    counts = {score: scores.count(score) for score in range(1, maximum + 1)}
    figure = go.Figure(
        go.Bar(
            x=list(counts),
            y=list(counts.values()),
            marker_color=[
                STATUS["critical"] if score >= threshold else SERIES[0] for score in counts
            ],
            text=list(counts.values()),
            hovertemplate=f"{method.upper()} %{{x}}: %{{y}} observations<extra></extra>",
        )
    )
    figure.update_layout(
        height=320,
        margin={"l": 45, "r": 15, "t": 20, "b": 45},
        xaxis={"title": f"{method.upper()} score", "dtick": 1, "gridcolor": "#1e2a42"},
        yaxis={"title": "Valid observations", "gridcolor": "#1e2a42"},
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        font={"color": INK_SECONDARY},
    )
    return figure


def region_heatmap_figure(records: list[dict[str, Any]]) -> go.Figure:
    if not records:
        return go.Figure()
    maximum_columns = 120
    if len(records) > maximum_columns:
        indices = np.linspace(0, len(records) - 1, maximum_columns, dtype=int)
        records = [records[index] for index in np.unique(indices)]
    region_keys = list(REGION_LABELS)
    values = [
        [body_region_risk(record).get(region, 0.0) for record in records] for region in region_keys
    ]
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=[float(record.get("timestamp_sec", 0.0)) for record in records],
            y=[REGION_LABELS[region] for region in region_keys],
            customdata=[
                [int(record.get("frame_number", 0)) for record in records] for _ in region_keys
            ],
            zmin=0,
            zmax=1,
            colorscale=[
                [0.0, "#0c1425"],
                [0.45, "#174b91"],
                [0.7, "#a3293a"],
                [1.0, STATUS["critical"]],
            ],
            colorbar={"title": "Relative<br>component<br>intensity"},
            hovertemplate=(
                "%{y}<br>%{x:.2f}s · frame %{customdata}<br>Intensity %{z:.2f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=390,
        margin={"l": 110, "r": 25, "t": 15, "b": 50},
        xaxis={"title": "Time (seconds)"},
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        font={"color": INK_SECONDARY},
    )
    return figure


def ergonomic_summary_metrics(
    records: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    total = len(records)
    metrics: dict[str, Any] = {}
    for method, threshold in (("rula", 5), ("reba", 8)):
        scores = []
        asymmetry = []
        for record in records:
            result = record.get(method, {})
            selected = _selected(result)
            if selected and isinstance(selected.get("final_score"), (int, float)):
                scores.append(float(selected["final_score"]))
            left = _side_score(result, "left")
            right = _side_score(result, "right")
            if left is not None and right is not None:
                asymmetry.append(abs(left - right))
        metrics[f"{method}_valid_percent"] = 100 * len(scores) / total if total else None
        metrics[f"{method}_high_percent"] = (
            100 * sum(score >= threshold for score in scores) / len(scores) if scores else None
        )
        metrics[f"{method}_mean_side_difference"] = float(np.mean(asymmetry)) if asymmetry else None
        periods = summary.get(f"{method}_summary", {}).get("sustained_high_risk_periods", [])
        metrics[f"{method}_sustained_duration_sec"] = sum(
            float(period.get("duration_sec", 0.0)) for period in periods if isinstance(period, dict)
        )
    angle_values: dict[str, list[float]] = {}
    for record in records:
        for name, value in record.get("angles", {}).items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                angle_values.setdefault(name, []).append(float(value))
    for name in (
        "trunk_flexion",
        "neck_flexion",
        "left_upper_arm_flexion",
        "right_upper_arm_flexion",
    ):
        values = angle_values.get(name, [])
        metrics[f"peak_{name}_deg"] = max(values, key=abs) if values else None
    visibility = [
        record.get("pose_quality", {}).get("mean_keypoint_visibility") for record in records
    ]
    visibility = [float(value) for value in visibility if isinstance(value, (int, float))]
    metrics["median_pose_visibility"] = float(np.median(visibility)) if visibility else None
    metrics["observation_count"] = total
    return metrics


def score_timeline_figure(records: list[dict[str, Any]], method: str) -> go.Figure:
    method = method.lower()
    rows = []
    for record in records:
        selected = _selected(record.get(method, {}))
        if selected and selected.get("final_score") is not None:
            rows.append(
                (
                    float(record.get("timestamp_sec", 0.0)),
                    int(record.get("frame_number", 0)),
                    float(selected["final_score"]),
                )
            )
    figure = go.Figure()
    if rows:
        figure.add_trace(
            go.Scatter(
                x=[row[0] for row in rows],
                y=[row[2] for row in rows],
                customdata=[row[1] for row in rows],
                mode="lines+markers",
                line={"color": SERIES[0], "width": 2},
                marker={"size": 6},
                hovertemplate=("%{x:.2f}s · score %{y}<br>Frame %{customdata}<extra></extra>"),
            )
        )
    figure.update_layout(
        height=340,
        margin={"l": 45, "r": 15, "t": 20, "b": 45},
        xaxis_title="Time (seconds)",
        yaxis_title=f"{method.upper()} score",
        hovermode="closest",
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        font={"color": INK_SECONDARY},
        xaxis={"gridcolor": "#1e2a42"},
        yaxis={"gridcolor": "#1e2a42"},
    )
    return figure


def angle_timeline_figure(records: list[dict[str, Any]], angle_names: list[str]) -> go.Figure:
    figure = go.Figure()
    for name in angle_names:
        points = [
            (
                float(record.get("timestamp_sec", 0.0)),
                int(record.get("frame_number", 0)),
                record.get("angles", {}).get(name),
            )
            for record in records
            if isinstance(record.get("angles", {}).get(name), (int, float))
        ]
        if points:
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in points],
                    y=[point[2] for point in points],
                    customdata=[point[1] for point in points],
                    mode="lines",
                    name=name.replace("_", " ").title(),
                    hovertemplate=("%{x:.2f}s · %{y:.1f}°<br>Frame %{customdata}<extra></extra>"),
                )
            )
    figure.update_layout(
        height=340,
        margin={"l": 45, "r": 15, "t": 20, "b": 45},
        xaxis_title="Time (seconds)",
        yaxis_title="Angle (degrees)",
        legend={"orientation": "h", "y": 1.12},
        paper_bgcolor="#080d18",
        plot_bgcolor="#080d18",
        font={"color": INK_SECONDARY},
        xaxis={"gridcolor": "#1e2a42"},
        yaxis={"gridcolor": "#1e2a42"},
    )
    return figure


def derive_findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic, non-diagnostic findings tied to supporting frames."""
    findings: list[dict[str, Any]] = []
    for method in ("rula", "reba"):
        valid = []
        for record in records:
            selected = _selected(record.get(method, {}))
            if selected and selected.get("final_score") is not None:
                valid.append((float(selected["final_score"]), record, selected))
        if valid:
            score, record, selected = max(valid, key=lambda item: item[0])
            findings.append(
                {
                    "severity": ("High" if score >= (5 if method == "rula" else 8) else "Review"),
                    "title": f"Peak {method.upper()} score: {score:g}",
                    "detail": selected.get("action_level") or selected.get("risk_level"),
                    "frame_number": int(record.get("frame_number", 0)),
                    "timestamp_sec": float(record.get("timestamp_sec", 0.0)),
                    "region": None,
                    "method": method.upper(),
                }
            )
    if not records:
        return findings

    def peak_region_intensity(record: dict[str, Any]) -> float:
        available = [value for value in body_region_risk(record).values() if value is not None]
        return max(available, default=-1.0)

    peak_record = max(records, key=peak_region_intensity)
    regions = body_region_risk(peak_record)
    available_regions = [(region, value) for region, value in regions.items() if value is not None]
    for region, value in sorted(available_regions, key=lambda item: item[1], reverse=True)[:3]:
        if value >= 0.5:
            findings.append(
                {
                    "severity": severity_label(value),
                    "title": f"{REGION_LABELS[region]} posture component is elevated",
                    "detail": (
                        "Component-level screening score; inspect the frame and method inputs."
                    ),
                    "frame_number": int(peak_record.get("frame_number", 0)),
                    "timestamp_sec": float(peak_record.get("timestamp_sec", 0.0)),
                    "region": region,
                    "method": "RULA/REBA",
                }
            )
    visibility = [
        record.get("pose_quality", {}).get("mean_keypoint_visibility") for record in records
    ]
    visibility = [float(value) for value in visibility if isinstance(value, (int, float))]
    if visibility and float(np.median(visibility)) < 0.75:
        findings.append(
            {
                "severity": "Data quality",
                "title": "Landmark visibility is limited",
                "detail": (
                    f"Median pose visibility is {np.median(visibility):.2f}; "
                    "score validity may be reduced."
                ),
                "frame_number": int(records[0].get("frame_number", 0)),
                "timestamp_sec": float(records[0].get("timestamp_sec", 0.0)),
                "region": None,
                "method": "Pose quality",
            }
        )
    return findings


def interpret_frame(
    record: dict[str, Any], method: str, region: str | None = None
) -> dict[str, Any]:
    """Explain one result with its source measurements and method intermediates."""
    method_key = method.lower()
    result = record.get(method_key, {})
    selected = _selected(result)
    if selected is None:
        side_results = {
            side: result.get(side, {}) if isinstance(result, dict) else {}
            for side in ("left", "right")
        }
        bilateral_statuses = {
            side: (
                side_result.get("validity_status", "unavailable")
                if isinstance(side_result, dict)
                else "unavailable"
            )
            for side, side_result in side_results.items()
        }
        missing_inputs: set[str] = set()
        invalid_inputs: set[str] = set()
        warnings: set[str] = set()
        for side_result in side_results.values():
            if not isinstance(side_result, dict):
                continue
            missing_inputs.update(side_result.get("missing_inputs", []))
            invalid_inputs.update(side_result.get("invalid_inputs", []))
            warnings.update(side_result.get("warnings", []))
        distinct_statuses = set(bilateral_statuses.values())
        overall_status = (
            next(iter(distinct_statuses)) if len(distinct_statuses) == 1 else "bilateral_incomplete"
        )
        return {
            "status": overall_status,
            "headline": f"{method.upper()} is incomplete for this frame",
            "meaning": (
                "A numeric result is intentionally withheld because required evidence is "
                "missing or invalid for one or both anatomical sides."
            ),
            "cause": (
                "Review the missing or invalid evidence and pose-quality warnings for each "
                "anatomical side."
            ),
            "missing": sorted(missing_inputs | invalid_inputs),
            "missing_inputs": sorted(missing_inputs),
            "invalid_inputs": sorted(invalid_inputs),
            "warnings": sorted(warnings),
            "bilateral_statuses": bilateral_statuses,
            "measurements": {},
            "intermediate_scores": {},
            "selection_reason": result.get("selection_reason", ""),
        }
    region_text = f" Focus the review on {REGION_LABELS[region].lower()}." if region else ""
    return {
        "status": selected.get("validity_status", "unknown"),
        "headline": (
            f"{method.upper()} score {selected.get('final_score')} · "
            f"{selected.get('risk_level', '')}"
        ),
        "meaning": (
            f"{selected.get('action_level', 'Review the documented action category.')}"
            f"{region_text} This is an observational ergonomic screening result, "
            "not a diagnosis."
        ),
        "cause": (
            "The score is produced from the listed posture measurements, "
            "task-condition inputs, and official lookup tables. Review the "
            "intermediates to identify the dominant component."
        ),
        "missing": selected.get("missing_inputs", []),
        "measurements": selected.get("raw_measurements", {}),
        "units": selected.get("units", {}),
        "intermediate_scores": selected.get("intermediate_scores", {}),
        "bilateral_statuses": {
            side: result.get(side, {}).get("validity_status", "unavailable")
            for side in ("left", "right")
        },
        "selection_reason": result.get("selection_reason", ""),
        "warnings": selected.get("warnings", []),
    }


def _landmarks_for_camera(record: dict[str, Any], camera_index: int) -> dict[str, Any] | None:
    single = record.get("pose_landmarks_normalized")
    if isinstance(single, dict):
        return single
    multi = record.get("pose_landmarks_normalized_by_camera")
    if isinstance(multi, dict):
        value = multi.get(str(camera_index))
        return value if isinstance(value, dict) else None
    return None


def read_video_frame(
    video_path: str | Path,
    frame_number: int,
    record: dict[str, Any] | None = None,
    selected_region: str | None = None,
    camera_index: int = 0,
) -> np.ndarray:
    """Decode one source frame and optionally draw the recorded pose overlay."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise DashboardDataError(f"Could not open recording: {Path(video_path).name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, image = capture.read()
        if not ok or image is None:
            raise DashboardDataError(f"Could not decode source frame {frame_number}")
    finally:
        capture.release()
    if record is not None:
        landmarks = _landmarks_for_camera(record, camera_index)
        if landmarks:
            height, width = image.shape[:2]
            usable = {
                int(key): raw
                for key, raw in landmarks.items()
                if isinstance(raw, list) and len(raw) >= 2
            }
            points: dict[int, tuple[int, int, float]] = {}
            for index, raw in augment_with_virtual_joints(usable).items():
                visibility = float(raw[3]) if len(raw) > 3 else 1.0
                points[index] = (
                    int(float(raw[0]) * width),
                    int(float(raw[1]) * height),
                    visibility,
                )
            for start, end in POSE_CONNECTIONS:
                if (
                    start in points
                    and end in points
                    and min(points[start][2], points[end][2]) >= 0.3
                ):
                    cv2.line(
                        image,
                        points[start][:2],
                        points[end][:2],
                        (235, 158, 11),
                        2,
                        cv2.LINE_AA,
                    )
            selected = REGION_LANDMARKS.get(selected_region or "", set())
            for index, (x, y, visibility) in points.items():
                if visibility < 0.3:
                    continue
                color = (24, 35, 180) if index in selected else (111, 125, 46)
                radius = 6 if index in selected else 3
                cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def video_metadata(video_path: str | Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return {"frame_count": None, "fps": None, "duration_sec": None}
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        return {
            "frame_count": frame_count if frame_count > 0 else None,
            "fps": fps if fps > 0 else None,
            "duration_sec": (frame_count / fps if frame_count > 0 and fps > 0 else None),
        }
    finally:
        capture.release()
