"""Compact, traceable pose replay payloads for the browser.

The frame JSONL is intentionally comprehensive: one multi-camera record in a
current native-rate trial is tens of kilobytes.  A browser replay needs only one
camera's stored image-normalized landmarks, timing, source-frame evidence, and
the two assessment scores.  This module selects that evidence without
recalculating posture or exposing source filesystem paths.

Image-normalized landmarks are used deliberately.  They are the exact points
used by the audited camera overlay, whereas each camera's model-world landmarks
live in a separate monocular coordinate frame and cannot be fused into one
calibrated 3-D reconstruction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from api.analysis import _band_payload, _check_method
from api.sessions import SessionError, load_summary, resolve_result_dir
from core.skeleton import POSE_CONNECTIONS, augment_with_virtual_joints
from dashboard.cache import ttl_cache
from dashboard.result_store import JsonlResultStore, ResultIndexEntry, open_result_store

DEFAULT_MAX_FRAMES = 360
MAX_REPLAY_FRAMES = 900

LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def _store(session_id: str) -> JsonlResultStore:
    return open_result_store(Path(resolve_result_dir(session_id)) / "frame_results.jsonl")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _score(record: dict[str, Any], method: str) -> int | None:
    result = record.get(method)
    if not isinstance(result, dict):
        return None
    selected = result.get("selected")
    if not isinstance(selected, dict):
        return None
    number = _finite(selected.get("final_score"))
    if number is None or int(number) != number:
        return None
    return int(number)


def _peak_position(entries: list[ResultIndexEntry], method: str) -> int | None:
    """Return the earliest maximum scored observation for ``method``."""

    best_position: int | None = None
    best_score: float | None = None
    for position, entry in enumerate(entries):
        score = _score(entry.lightweight, method)
        if score is None:
            continue
        numeric = float(score)
        if best_score is None or numeric > best_score:
            best_position = position
            best_score = numeric
    return best_position


def _sample_positions(
    entries: list[ResultIndexEntry],
    maximum: int,
    preferred_method: str,
    anchor_position: int | None = None,
) -> list[int]:
    """Choose an even sample while preserving the requested observation and peaks."""

    count = len(entries)
    if count <= maximum:
        positions = list(range(count))
    else:
        other_method = "rula" if preferred_method == "reba" else "reba"
        priority = [
            anchor_position,
            0,
            count - 1,
            _peak_position(entries, preferred_method),
            _peak_position(entries, other_method),
        ]
        selected: set[int] = set()
        for position in priority:
            if position is not None and len(selected) < maximum:
                selected.add(position)

        # Repeatedly split the largest uncovered interval.  This keeps coverage
        # distributed across the whole activity even when both score peaks sit
        # close to an endpoint.  Ties choose the earlier interval and midpoint.
        while len(selected) < maximum:
            ordered = sorted(selected)
            gaps = [
                (right - left, -left, left, right)
                for left, right in zip(ordered, ordered[1:], strict=False)
                if right - left > 1
            ]
            if not gaps:
                break
            _, _, left, right = max(gaps)
            selected.add((left + right) // 2)
        positions = sorted(selected)

    return positions


def _anchor_position(
    entries: list[ResultIndexEntry],
    anchor_frame: int | None,
) -> int | None:
    if anchor_frame is None:
        return None
    return next(
        (
            position
            for position, entry in enumerate(entries)
            if entry.frame_number == anchor_frame
        ),
        None,
    )


def _camera_dimensions(session_id: str, camera_index: int) -> tuple[int, int, str]:
    """Return decoded camera geometry, with an explicit legacy fallback."""

    try:
        metadata = load_summary(session_id).get("video_metadata") or []
    except SessionError:
        metadata = []
    for position, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        index = _integer(item.get("camera_index"))
        if index is None:
            index = position
        width = _integer(item.get("decoded_width_px"))
        height = _integer(item.get("decoded_height_px"))
        if index == camera_index and width and height and width > 0 and height > 0:
            return width, height, "analysis-metadata"
    return 1280, 720, "legacy-fallback"


def _available_camera_indices(record: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    if isinstance(record.get("pose_landmarks_normalized"), dict):
        indices.add(0)
    by_camera = record.get("pose_landmarks_normalized_by_camera")
    if isinstance(by_camera, dict):
        for raw_index, landmarks in by_camera.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if isinstance(landmarks, dict):
                indices.add(index)
    return sorted(indices)


def _landmarks_for_camera(
    record: dict[str, Any],
    camera_index: int,
) -> dict[str, Any] | None:
    single = record.get("pose_landmarks_normalized")
    if camera_index == 0 and isinstance(single, dict):
        return single
    by_camera = record.get("pose_landmarks_normalized_by_camera")
    if not isinstance(by_camera, dict):
        return None
    value = by_camera.get(str(camera_index))
    return value if isinstance(value, dict) else None


def _compact_points(
    record: dict[str, Any],
    camera_index: int,
) -> list[dict[str, int | float]]:
    raw_landmarks = _landmarks_for_camera(record, camera_index)
    if raw_landmarks is None:
        return []
    parsed: dict[int, list[float]] = {}
    for raw_index, raw_point in raw_landmarks.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if not isinstance(raw_point, list) or len(raw_point) < 3:
            continue
        coordinates = [_finite(value) for value in raw_point[:4]]
        x, y, z = coordinates[:3]
        if x is None or y is None or z is None:
            continue
        visibility = coordinates[3] if len(raw_point) >= 4 and coordinates[3] is not None else 1.0
        parsed[index] = [
            x,
            y,
            z,
            float(visibility),
        ]

    # C7, head centre, and pelvis make the replay anatomy match the scored neck
    # and trunk segments.  No original landmark is changed.
    augmented = augment_with_virtual_joints(parsed)
    return [
        {
            "id": index,
            "x": round(float(point[0]), 6),
            "y": round(float(point[1]), 6),
            "z": round(float(point[2]), 6),
            "visibility": round(float(point[3]) if len(point) > 3 else 1.0, 4),
        }
        for index, point in sorted(augmented.items())
        if len(point) >= 3
    ]


def _camera_reference(record: dict[str, Any], camera_index: int) -> dict[str, Any]:
    references = record.get("camera_frame_references")
    if not isinstance(references, list):
        references = record.get("camera_frame_refs")
    if not isinstance(references, list):
        references = []
    for reference in references:
        if not isinstance(reference, dict):
            continue
        if _integer(reference.get("camera_index")) == camera_index:
            return reference
    return {}


def _frame_payload(
    record: dict[str, Any],
    camera_index: int,
    method: str,
) -> dict[str, Any] | None:
    points = _compact_points(record, camera_index)
    if not points:
        return None
    reference = _camera_reference(record, camera_index)
    frame_number = _integer(record.get("frame_number"))
    timestamp = _finite(record.get("timestamp_sec"))
    if frame_number is None or timestamp is None:
        return None
    source_frame = _integer(reference.get("source_frame_number"))
    if source_frame is None:
        source_frame = frame_number
    score = _score(record, method)
    return {
        "f": frame_number,
        "t": round(timestamp, 6),
        "sourceFrame": source_frame,
        "score": score,
        "band": _band_payload(score, method),
        "points": points,
    }


@ttl_cache(ttl=120, copy_result=False)
def replay_payload(
    session_id: str,
    camera_index: int = 0,
    person: str | None = None,
    method: str = "reba",
    max_frames: int = DEFAULT_MAX_FRAMES,
    anchor_frame: int | None = None,
) -> dict[str, Any]:
    """Return a bounded, selected-camera replay for one tracked person."""

    if isinstance(camera_index, bool) or not isinstance(camera_index, int) or camera_index < 0:
        raise SessionError("Replay camera index must be a non-negative integer")
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or not 2 <= max_frames <= MAX_REPLAY_FRAMES
    ):
        raise SessionError(
            f"Replay max_frames must be an integer from 2 to {MAX_REPLAY_FRAMES}"
        )
    if anchor_frame is not None and (
        isinstance(anchor_frame, bool)
        or not isinstance(anchor_frame, int)
        or anchor_frame < 0
    ):
        raise SessionError("Replay anchor_frame must be a non-negative integer")
    method_key = _check_method(method)
    store = _store(session_id)
    people = sorted(
        {str(entry.person_identifier) for entry in store.entries if entry.person_identifier}
    )
    tracked = person or (people[0] if people else None)
    if tracked is None:
        raise SessionError("This analysis has no tracked person")
    if tracked not in people:
        raise SessionError(f"No tracked person matches {tracked!r}")

    candidates = sorted(
        (
            entry
            for entry in store.entries
            if entry.has_landmarks and entry.person_identifier == tracked
        ),
        key=lambda entry: (
            entry.timestamp_sec if entry.timestamp_sec is not None else float("inf"),
            entry.frame_number if entry.frame_number is not None else 2**63 - 1,
            entry.line_number,
        ),
    )
    if not candidates:
        raise SessionError("This analysis has no stored pose landmarks for the tracked person")

    first_record = store.load_record_at_offset(candidates[0].offset)
    available_indices = _available_camera_indices(first_record)
    if camera_index not in available_indices:
        offered = ", ".join(str(index) for index in available_indices) or "none"
        raise SessionError(
            f"No stored normalized landmarks for camera {camera_index}. Available: {offered}"
        )

    positions = _sample_positions(
        candidates,
        max_frames,
        method_key,
        _anchor_position(candidates, anchor_frame),
    )
    frames: list[dict[str, Any]] = []
    for position in positions:
        record = store.load_record_at_offset(candidates[position].offset)
        payload = _frame_payload(record, camera_index, method_key)
        if payload is not None:
            frames.append(payload)
    if not frames:
        raise SessionError(f"Camera {camera_index} has no replayable normalized landmarks")
    image_width, image_height, dimensions_source = _camera_dimensions(
        session_id,
        camera_index,
    )

    return {
        "sessionId": session_id,
        "person": tracked,
        "method": method_key.upper(),
        "cameraIndex": camera_index,
        "sourceCount": len(candidates),
        "sampledCount": len(frames),
        "anchorFrame": anchor_frame,
        "anchorIncluded": (
            anchor_frame is not None
            and any(frame["f"] == anchor_frame for frame in frames)
        ),
        "coordinateSpace": "normalized-image",
        "imageWidth": image_width,
        "imageHeight": image_height,
        "aspectRatio": round(image_width / image_height, 6),
        "dimensionsSource": dimensions_source,
        "visibilityThreshold": 0.3,
        "connections": [list(connection) for connection in POSE_CONNECTIONS],
        "frames": frames,
    }


__all__ = [
    "DEFAULT_MAX_FRAMES",
    "LANDMARK_NAMES",
    "MAX_REPLAY_FRAMES",
    "replay_payload",
]
