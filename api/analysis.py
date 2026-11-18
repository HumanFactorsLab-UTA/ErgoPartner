"""Turning the existing ergonomic analysis into JSON the browser can draw.

Every number here comes from ``dashboard/task_timeline.py``, which in turn reads
the per-frame RULA and REBA results the pipeline wrote. This module adds no
scoring of its own: it selects, shapes, and decimates.

The one judgement it makes is what the wire should carry. A native-rate trial is
four thousand scored frames; sending all of them so the browser can draw a
1200-pixel chart wastes the transfer and the render. The series is decimated with
the same peak-preserving rule the Plotly figures used, so the envelope the client
draws is exact and no spike is lost.
"""

from __future__ import annotations

from typing import Any

from api.sessions import (
    SessionError,
    finite,
    load_summary,
    resolve_result_dir,
)
from dashboard.cache import ttl_cache
from dashboard.result_store import open_result_store
from dashboard.task_timeline import (
    BodyPartLoad,
    Driver,
    RiskEpisode,
    ScorePoint,
    band_shares,
    body_part_load,
    clock,
    drivers_at,
    risk_episodes,
    score_series,
    verdict,
)
from dashboard.viz_theme import METHOD_SCALE_MAXIMUM, method_bands, risk_band

METHODS = ("reba", "rula")

#: Enough points to draw a wide chart at device-pixel density without aliasing
#: the shape, and small enough to stay a light JSON payload.
SERIES_POINT_LIMIT = 1400


def _check_method(method: str) -> str:
    key = str(method).lower()
    if key not in METHODS:
        raise SessionError(f"Unknown method {method!r}; expected reba or rula")
    return key


def _store(session_id: str):
    from pathlib import Path

    return open_result_store(Path(resolve_result_dir(session_id)) / "frame_results.jsonl")


@ttl_cache(ttl=120, copy_result=False)
def _records(session_id: str, person: str | None) -> list[dict[str, Any]]:
    store = _store(session_id)
    return [
        entry.lightweight
        for entry in store.entries
        if person is None or entry.person_identifier == person
    ]


@ttl_cache(ttl=120, copy_result=False)
def people(session_id: str) -> list[str]:
    store = _store(session_id)
    return sorted({str(e.person_identifier) for e in store.entries if e.person_identifier})


def _decimate(series: list[ScorePoint], limit: int) -> list[ScorePoint]:
    """Peak-preserving thinning: each bucket keeps its lowest and highest point."""

    if limit < 4 or len(series) <= limit:
        return series
    buckets = max(2, limit // 2)
    size = len(series) / buckets
    kept: dict[int, ScorePoint] = {0: series[0], len(series) - 1: series[-1]}
    for index in range(buckets):
        start = int(index * size)
        end = int((index + 1) * size) if index + 1 < buckets else len(series)
        if end <= start:
            continue
        window = range(start, end)
        kept[min(window, key=lambda i: series[i].score)] = series[
            min(window, key=lambda i: series[i].score)
        ]
        kept[max(window, key=lambda i: series[i].score)] = series[
            max(window, key=lambda i: series[i].score)
        ]
    return [kept[index] for index in sorted(kept)]


def _band_payload(score: int | None, method: str) -> dict[str, Any] | None:
    band = risk_band(score, method)
    if band is None:
        return None
    return {
        "label": band.label,
        "action": band.action,
        "status": band.status,
        "lower": band.lower,
        "upper": band.upper,
        "range": band.range_text,
        "describesRisk": band.describes_risk,
    }


def _driver_payload(driver: Driver) -> dict[str, Any]:
    return {
        "region": driver.region,
        "label": driver.label,
        "component": driver.component,
        "kind": driver.kind,
        "score": driver.component_score,
        "maximum": driver.component_maximum,
        "severity": round(driver.severity, 4),
        "pointsAdded": driver.points_added,
        "measurement": driver.measurement,
        "valueDeg": None if driver.value_deg is None else round(driver.value_deg, 1),
    }


def _episode_payload(episode: RiskEpisode) -> dict[str, Any]:
    return {
        "index": episode.index,
        "startSec": round(episode.start_sec, 3),
        "endSec": round(episode.end_sec, 3),
        "durationSec": round(episode.duration_sec, 3),
        "startFrame": episode.start_frame_number,
        "endFrame": episode.end_frame_number,
        "peakScore": episode.peak_score,
        "peakFrame": episode.peak_frame_number,
        "peakSec": round(episode.peak_timestamp_sec, 3),
        "observations": episode.observation_count,
        "band": _band_payload(episode.peak_score, episode.method),
        "timeRange": episode.time_range_text,
        "drivers": [_driver_payload(item) for item in episode.drivers],
    }


def _load_payload(load: BodyPartLoad) -> dict[str, Any]:
    return {
        "region": load.region,
        "label": load.label,
        "component": load.component,
        "kind": load.kind,
        "maximum": load.component_maximum,
        "worstScore": load.worst_score,
        "worstPoints": load.worst_points,
        "meanPoints": round(load.mean_points, 3),
        "percentOfTask": round(load.percent_of_scored_time, 1),
        "percentRaised": round(load.percent_raised, 1),
        "peakFrame": load.peak_frame_number,
        "peakSec": None if load.peak_timestamp_sec is None else round(load.peak_timestamp_sec, 3),
        "measurement": load.measurement,
        "peakValueDeg": None if load.peak_value_deg is None else round(load.peak_value_deg, 1),
    }


@ttl_cache(ttl=120)
def timeline(session_id: str, method: str, person: str | None = None) -> dict[str, Any]:
    """Everything the analysis workspace needs for one method, in one payload."""

    key = _check_method(method)
    tracked = person or (people(session_id) or [None])[0]
    records = _records(session_id, tracked)
    series = score_series(records, key)
    episodes = risk_episodes(series, key)
    loads = body_part_load(series, key)
    result = verdict(series, key, episodes=episodes, loads=loads)
    shares = band_shares(series, key)
    drawn = _decimate(series, SERIES_POINT_LIMIT)

    return {
        "sessionId": session_id,
        "method": key.upper(),
        "person": tracked,
        "scaleMaximum": METHOD_SCALE_MAXIMUM[key],
        "bands": [
            {
                "label": band.label,
                "action": band.action,
                "status": band.status,
                "lower": band.lower,
                "upper": band.upper,
                "range": band.range_text,
            }
            for band in method_bands(key)
        ],
        "actionThreshold": result.action_threshold,
        "verdict": {
            "scored": result.scored,
            "headline": result.headline,
            "detail": result.detail,
            "peakScore": result.peak_score,
            "peakFrame": result.peak_frame_number,
            "peakSec": None if result.peak_timestamp_sec is None else round(result.peak_timestamp_sec, 3),
            "peakClock": clock(result.peak_timestamp_sec),
            "peakBand": _band_payload(result.peak_score, key),
            "typicalScore": None if result.typical_score is None else round(result.typical_score, 2),
            "typicalBand": _band_payload(
                None if result.typical_score is None else round(result.typical_score), key
            ),
            "scoredDurationSec": round(result.scored_duration_sec, 2),
            "timeAtActionSec": round(result.time_at_action_sec, 2),
            "percentAtAction": round(result.percent_at_action, 1),
            "episodeCount": result.episode_count,
            "leadingRegion": result.leading_region,
            "leadingLabel": result.leading_label,
            "leadingPoints": None if result.leading_points is None else round(result.leading_points, 2),
            "leadingTaskLabel": result.leading_task_label,
            "leadingTaskPoints": (
                None if result.leading_task_points is None else round(result.leading_task_points, 2)
            ),
        },
        "series": {
            "count": len(series),
            "drawn": len(drawn),
            "points": [
                {
                    "f": point.frame_number,
                    "t": round(point.timestamp_sec, 3),
                    "s": point.score,
                    "b": (point.band.status if point.band else "none"),
                }
                for point in drawn
            ],
        },
        "shares": [
            {
                "label": share.band.label,
                "range": share.band.range_text,
                "status": share.band.status,
                "action": share.band.action,
                "durationSec": round(share.duration_sec, 2),
                "percent": round(share.percent_of_scored_time, 1),
                "observations": share.observation_count,
            }
            for share in shares
        ],
        "episodes": [_episode_payload(item) for item in episodes],
        "posture": [_load_payload(item) for item in loads if item.kind == "posture"],
        "taskFactors": [_load_payload(item) for item in loads if item.kind == "task"],
    }


def _nearest_entry(session_id: str, person: str, frame: int):
    store = _store(session_id)
    candidates = [
        entry
        for entry in store.entries
        if entry.person_identifier == person and entry.frame_number is not None
    ]
    if not candidates:
        raise SessionError("This analysis has no frames for the tracked person")
    return min(candidates, key=lambda entry: abs(int(entry.frame_number) - frame))


def _angle_degrees(measurements: Any) -> dict[str, float]:
    if not isinstance(measurements, dict):
        return {}
    return {
        name: round(degrees, 1)
        for name, value in measurements.items()
        if not name.endswith("_midline_cross") and (degrees := finite(value)) is not None
    }


def frame_detail(
    session_id: str, frame: int, method: str, person: str | None = None
) -> dict[str, Any]:
    """One frame: its score, what drove it, its angles, and its camera views."""

    key = _check_method(method)
    tracked = person or (people(session_id) or [None])[0]
    if tracked is None:
        raise SessionError("This analysis has no tracked person")
    entry = _nearest_entry(session_id, tracked, int(frame))
    series = score_series([entry.lightweight], key)
    point = series[0] if series else None

    method_result = entry.lightweight.get(key) or {}
    selected = method_result.get("selected") or {}
    stored_angles = entry.lightweight.get("angles") or {}
    per_camera = (
        entry.lightweight.get("angles_by_camera")
        or entry.lightweight.get("per_camera_angles")
        or {}
    )
    quality = entry.lightweight.get("pose_quality") or {}
    references = entry.lightweight.get("camera_frame_references") or []

    sides = {}
    for side in ("left", "right"):
        side_result = method_result.get(side) or {}
        sides[side] = {
            "score": side_result.get("final_score"),
            "validity": side_result.get("validity_status"),
            "components": side_result.get("intermediate_scores") or {},
            "missing": side_result.get("missing_inputs") or [],
            "invalid": side_result.get("invalid_inputs") or [],
        }

    return {
        "sessionId": session_id,
        "method": key.upper(),
        "person": tracked,
        "frame": entry.frame_number,
        "timestampSec": entry.timestamp_sec,
        "clock": clock(entry.timestamp_sec),
        "score": selected.get("final_score"),
        "band": _band_payload(selected.get("final_score"), key),
        "selectedSide": method_result.get("selected_side"),
        "selectionReason": method_result.get("selection_reason"),
        "riskLevel": selected.get("risk_level"),
        "actionLevel": selected.get("action_level"),
        "components": selected.get("intermediate_scores") or {},
        "sides": sides,
        "drivers": [_driver_payload(item) for item in drivers_at(point, key, limit=6)]
        if point is not None
        else [],
        "angles": _angle_degrees(stored_angles),
        "anglesByCamera": {
            str(camera): _angle_degrees(values) for camera, values in per_camera.items()
        }
        if isinstance(per_camera, dict)
        else {},
        "angleSources": entry.lightweight.get("multi_view_selection") or {},
        "quality": {
            "meanVisibility": finite(quality.get("mean_keypoint_visibility")),
            "cameraView": quality.get("camera_view_status"),
            "wristSource": stored_angles.get(
                f"{selected.get('side')}_wrist_data_source", quality.get("wrist_data_source")
            ),
            "status": selected.get("validity_status") or quality.get("status"),
            "warnings": list(
                dict.fromkeys([*(quality.get("warnings") or []), *(selected.get("warnings") or [])])
            ),
        },
        "cameras": [
            {
                "cameraIndex": int(reference.get("camera_index", index)),
                "role": str(
                    reference.get("camera_role") or reference.get("role") or f"camera {index + 1}"
                ),
                "sourceFrame": reference.get("source_frame_number"),
                "timestampSec": finite(reference.get("timestamp_sec")),
                "syncDeltaSec": finite(reference.get("sync_delta_from_master_sec")),
            }
            for index, reference in enumerate(references)
            if isinstance(reference, dict)
        ],
        "offset": entry.offset,
    }


def session_overview(session_id: str) -> dict[str, Any]:
    """Recording facts and per-method headline, without the frame payloads."""

    summary = load_summary(session_id)
    quality = summary.get("data_quality") or {}
    metadata = summary.get("video_metadata") or []
    rnle = summary.get("rnle") or {}
    headline = {}
    for method in METHODS:
        block = summary.get(f"{method}_summary") or {}
        maximum = block.get("maximum_score")
        headline[method] = {
            "maximum": maximum,
            "band": _band_payload(maximum, method),
            "observations": block.get("observation_count"),
            "percentile90": finite(block.get("high_percentile")),
            "durationWeightedMean": (
                None
                if finite(block.get("duration_weighted_mean")) is None
                else round(float(block["duration_weighted_mean"]), 2)
            ),
            "peakFrame": block.get("maximum_frame_number"),
            "peakSec": finite(block.get("maximum_timestamp_sec")),
        }
    return {
        "sessionId": session_id,
        "durationSec": finite(summary.get("duration_sec")),
        "processedFrames": summary.get("processed_frames"),
        "poseFrames": summary.get("pose_detected_frames"),
        "cameraCount": quality.get("camera_count") or len(metadata),
        "trackingMode": quality.get("tracking_mode"),
        "calibration": summary.get("calibration_status"),
        "softwareVersion": summary.get("software_version"),
        "runtimeSec": finite(summary.get("runtime_sec")),
        "methods": headline,
        "rnle": {
            "liftingIndex": finite(rnle.get("lifting_index")),
            "recommendedWeightKg": finite(rnle.get("recommended_weight_limit_kg")),
            "validity": rnle.get("validity_status"),
            "missing": rnle.get("missing_inputs") or [],
            "invalid": rnle.get("invalid_inputs") or [],
        },
        "warnings": summary.get("warnings") or [],
        "people": people(session_id),
    }


__all__ = [
    "METHODS",
    "frame_detail",
    "people",
    "session_overview",
    "timeline",
]
