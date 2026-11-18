"""The v4 replay must retain hand evidence and fail on ambiguous chronology."""

import hashlib
import json

import pytest

from core.angle_calculator import AngleCalculator
from validation.replay_geometry import camera_angles, replay_recording


def test_replay_uses_each_saved_mesh_angle_and_preserves_unavailable_values():
    cached = {
        "left_wrist_data_source": "pose_only",
        "right_wrist_data_source": "hand_mesh",
        "right_wrist_flexion": 17.5,
        "right_wrist_deviation": None,
        "right_wrist_pronation": -8.0,
    }
    angles = camera_angles(AngleCalculator(), {}, cached)
    assert angles.to_dict()["right_wrist_flexion"] == 17.5
    assert angles.to_dict()["right_wrist_deviation"] is None
    assert angles.to_dict()["left_wrist_flexion"] is None
    assert angles.right_wrist_data_source == "hand_mesh"


def test_exact_frame_replay_preserves_input_and_never_substitutes(tmp_path):
    config = {
        "schema_version": 4,
        "quality": {},
        "fusion": {"camera_roles": {"front": 0, "side": 1}, "visibility_threshold": 0.5},
    }
    (tmp_path / "run_config.json").write_text(json.dumps(config))
    evidence = {"left_wrist_data_source": "pose_only", "right_wrist_data_source": "pose_only"}
    record = {
        "frame_number": 10,
        "timestamp_sec": 0.3,
        "person_identifier": "person-0001",
        "pose_landmarks_world_by_camera": {"0": {}, "1": {}},
        "angles_by_camera": {"0": evidence, "1": evidence},
    }
    content = (json.dumps(record) + "\n").encode()
    path = tmp_path / "frame_results.jsonl"
    path.write_bytes(content)
    ids = [{"frame_id": "F1", "frame_number": "11"}]
    rows, manifest = replay_recording((tmp_path, ids, {}))
    assert rows == [{**ids[0], "reba_final": None, "rula_final": None}]
    assert manifest["observations"] == 1 and manifest["matched"] == 0
    assert manifest["jsonl_sha256"] == hashlib.sha256(content).hexdigest()
    assert path.read_bytes() == content
    path.write_bytes(content + content)
    with pytest.raises(ValueError, match="chronological"):
        replay_recording((tmp_path, ids, {}))
