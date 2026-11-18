"""Browser replay payloads stay compact, traceable, and peak preserving."""

from __future__ import annotations

import json
from typing import Any

import pytest

from api import replay
from api.sessions import SessionError
from core.skeleton import POSE_CONNECTIONS

SECRET_SOURCE = r"D:\private-study\participant_001\raw\cam_1.mp4"


def _landmarks(horizontal_shift: float = 0.0) -> dict[str, list[float]]:
    return {
        "0": [0.50 + horizontal_shift, 0.10, -0.10, 0.98],
        "2": [0.47 + horizontal_shift, 0.08, -0.08, 0.97],
        "5": [0.53 + horizontal_shift, 0.08, -0.08, 0.96],
        "7": [0.44 + horizontal_shift, 0.10, -0.02, 0.95],
        "8": [0.56 + horizontal_shift, 0.10, -0.02, 0.94],
        "11": [0.40 + horizontal_shift, 0.30, 0.00, 0.93],
        "12": [0.60 + horizontal_shift, 0.30, 0.00, 0.92],
        "13": [0.34 + horizontal_shift, 0.48, 0.01, 0.91],
        "14": [0.66 + horizontal_shift, 0.48, 0.01, 0.90],
        "23": [0.44 + horizontal_shift, 0.58, 0.01, 0.89],
        "24": [0.56 + horizontal_shift, 0.58, 0.01, 0.88],
        "25": [0.43 + horizontal_shift, 0.75, 0.02, 0.87],
        "26": [0.57 + horizontal_shift, 0.75, 0.02, 0.86],
        "27": [0.42 + horizontal_shift, 1.02, 0.03, 0.85],
        "28": [0.58 + horizontal_shift, 1.01, 0.03, 0.84],
    }


def _method(score: int, risk: str) -> dict[str, Any]:
    selected = {
        "final_score": score,
        "validity_status": "valid_with_limitations",
        "risk_level": risk,
    }
    return {
        "selected": selected,
        "selected_side": "left",
        "left": selected,
        "right": selected,
    }


def _record(
    frame: int,
    *,
    reba_score: int = 4,
    rula_score: int = 3,
    person: str = "person-0001",
) -> dict[str, Any]:
    return {
        "frame_number": frame,
        "timestamp_sec": frame / 30,
        "person_identifier": person,
        "pose_quality": {"mean_keypoint_visibility": 0.9123456},
        "pose_landmarks_normalized_by_camera": {
            "0": _landmarks(),
            "1": _landmarks(0.10),
        },
        # World landmarks and source paths are deliberately present in the raw
        # record; neither belongs on the normalized browser replay wire.
        "pose_landmarks_world_by_camera": {
            "0": {"11": [-0.2, 0.4, 0.0, 0.9]},
            "1": {"11": [-0.1, 0.4, 0.1, 0.9]},
        },
        "camera_frame_references": [
            {
                "camera_index": 0,
                "camera_role": "front",
                "source_frame_number": frame,
                "timestamp_sec": frame / 30,
                "sync_delta_from_master_sec": 0.0,
                "source_path": r"D:\private-study\participant_001\raw\cam_0.mp4",
            },
            {
                "camera_index": 1,
                "camera_role": "side",
                "source_frame_number": frame + 100,
                "timestamp_sec": frame / 30 + 0.004,
                "sync_delta_from_master_sec": 0.004,
                "source_path": SECRET_SOURCE,
            },
        ],
        "rula": _method(rula_score, "Investigate"),
        "reba": _method(reba_score, "Medium"),
    }


def _install_bundle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
) -> str:
    (tmp_path / "frame_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    session_id = f"test__001__task__{tmp_path.name}"
    monkeypatch.setattr(replay, "resolve_result_dir", lambda _: str(tmp_path))
    replay.replay_payload.cache_clear()
    return session_id


def _point(frame: dict[str, Any], landmark_id: int) -> dict[str, int | float]:
    return next(point for point in frame["points"] if point["id"] == landmark_id)


def test_replay_uses_the_selected_cameras_raw_normalized_points_and_source_frame(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])

    payload = replay.replay_payload(session_id, camera_index=1, max_frames=2)

    assert payload["cameraIndex"] == 1
    assert payload["coordinateSpace"] == "normalized-image"
    assert payload["visibilityThreshold"] == 0.3
    assert payload["frames"][0]["sourceFrame"] == 100
    assert _point(payload["frames"][0], 11) == {
        "id": 11,
        "x": 0.5,
        "y": 0.3,
        "z": 0.0,
        "visibility": 0.93,
    }
    # Out-of-frame evidence remains exact instead of being clamped to the SVG.
    assert _point(payload["frames"][0], 27)["y"] == 1.02


def test_replay_adds_the_anatomical_virtual_joints_and_shared_connections(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])

    payload = replay.replay_payload(session_id, max_frames=2)
    frame = payload["frames"][0]

    assert _point(frame, 100) == {
        "id": 100,
        "x": 0.5,
        "y": 0.3,
        "z": 0.0,
        "visibility": 0.925,
    }
    assert _point(frame, 102) == {
        "id": 102,
        "x": 0.5,
        "y": 0.58,
        "z": 0.01,
        "visibility": 0.885,
    }
    assert payload["connections"] == [list(connection) for connection in POSE_CONNECTIONS]
    assert [100, 101] in payload["connections"]
    assert [100, 102] in payload["connections"]


def test_adaptive_sampling_prioritises_endpoints_and_both_method_peaks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [_record(frame) for frame in range(12)]
    records[5] = _record(5, reba_score=13)
    records[7] = _record(7, rula_score=7)
    session_id = _install_bundle(tmp_path, monkeypatch, records)

    payload = replay.replay_payload(session_id, method="reba", max_frames=4)

    assert [frame["f"] for frame in payload["frames"]] == [0, 5, 7, 11]
    assert payload["sourceCount"] == 12
    assert payload["sampledCount"] == 4
    peak = payload["frames"][1]
    assert peak["score"] == 13
    assert peak["band"]["label"] == "Very high"


def test_a_tight_budget_preserves_the_requested_methods_peak_first(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [_record(frame) for frame in range(12)]
    records[5] = _record(5, reba_score=13)
    records[7] = _record(7, rula_score=7)
    session_id = _install_bundle(tmp_path, monkeypatch, records)

    payload = replay.replay_payload(session_id, method="reba", max_frames=3)

    assert [frame["f"] for frame in payload["frames"]] == [0, 5, 11]


def test_selected_frame_is_forced_into_the_bounded_replay(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [_record(frame) for frame in range(12)]
    records[5] = _record(5, reba_score=13)
    session_id = _install_bundle(tmp_path, monkeypatch, records)

    payload = replay.replay_payload(
        session_id,
        method="reba",
        max_frames=4,
        anchor_frame=3,
    )

    assert [frame["f"] for frame in payload["frames"]] == [0, 3, 5, 11]
    assert payload["anchorFrame"] == 3
    assert payload["anchorIncluded"] is True


def test_replay_uses_the_selected_cameras_decoded_geometry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])
    monkeypatch.setattr(
        replay,
        "load_summary",
        lambda _: {
            "video_metadata": [
                {
                    "camera_index": 0,
                    "decoded_width_px": 1440,
                    "decoded_height_px": 1080,
                }
            ]
        },
    )

    payload = replay.replay_payload(session_id, max_frames=2)

    assert payload["imageWidth"] == 1440
    assert payload["imageHeight"] == 1080
    assert payload["aspectRatio"] == pytest.approx(4 / 3)
    assert payload["dimensionsSource"] == "analysis-metadata"


def test_replay_never_leaks_source_paths_or_world_landmarks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])

    payload = replay.replay_payload(session_id, camera_index=1, max_frames=2)
    serialized = json.dumps(payload)

    assert SECRET_SOURCE not in serialized
    assert "source_path" not in serialized
    assert "pose_landmarks_world" not in serialized
    assert "input_source" not in serialized


def test_replay_filters_people_and_refuses_an_unavailable_camera(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [
        _record(0, person="person-a"),
        _record(1, person="person-b"),
        _record(2, person="person-a"),
    ]
    session_id = _install_bundle(tmp_path, monkeypatch, records)

    payload = replay.replay_payload(
        session_id,
        person="person-b",
        max_frames=2,
    )
    assert payload["person"] == "person-b"
    assert [frame["f"] for frame in payload["frames"]] == [1]

    with pytest.raises(SessionError, match="camera 9"):
        replay.replay_payload(session_id, camera_index=9, max_frames=2)


@pytest.mark.parametrize("maximum", [0, 1, 901, True])
def test_replay_refuses_an_unsafe_frame_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    maximum: Any,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])

    with pytest.raises(SessionError, match="max_frames"):
        replay.replay_payload(session_id, max_frames=maximum)


@pytest.mark.parametrize("anchor", [-1, 1.5, True])
def test_replay_refuses_an_invalid_anchor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: Any,
):
    session_id = _install_bundle(tmp_path, monkeypatch, [_record(0), _record(1)])

    with pytest.raises(SessionError, match="anchor_frame"):
        replay.replay_payload(session_id, anchor_frame=anchor)
