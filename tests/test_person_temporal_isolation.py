"""A second worker must not change the first worker's measured movement."""

import json

import pytest
import yaml

from paths import CONFIG_PATH, PROJECT_ROOT
from pipeline.video_processor import ConfigurationError, VideoProcessor
from tests.integration.test_video_pipeline import FakePoseEstimator, _synthetic_pose


class _Workers(FakePoseEstimator):
    def __init__(self, people):
        self.people = people

    def process_frame_all(self, _frame, timestamp=0.0, frame_number=0):
        poses = []
        for person in self.people:
            pose = _synthetic_pose(timestamp, frame_number)
            # Separate torso centroids for tracking. Worker B keeps an arm
            # raised while worker A keeps both arms at their sides.
            for point in pose.normalized.values():
                point[0] += 0.2 if person == "B" else -0.2
            if person == "B":
                for index in (13, 15, 17, 19, 21):
                    pose.world[index][0] -= 0.5
                    pose.world[index][1] += 0.5
            poses.append(pose)
        return poses


def _configure(tmp_path, monkeypatch, people, reviewed=False):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["hand"]["enabled"] = False
    config["video"].update(target_fps=None, start_frame=0, end_frame=7)
    if reviewed:
        config.setdefault("posture_conditions", {})["neutral_reference"] = {
            "source": "reviewed_neutral",
            "reference_id": "worker-A-neutral",
            "offsets_deg": {"left_wrist_deviation": 2.0},
        }
    processor = VideoProcessor(config)
    monkeypatch.setattr(processor, "_create_pose_estimators", lambda _count: [_Workers(people)])
    processor.process([str(PROJECT_ROOT / "test_synthetic_video.mp4")], str(tmp_path))
    summary = json.loads((tmp_path / "video_summary.json").read_text())
    records = [
        json.loads(line) for line in (tmp_path / "frame_results.jsonl").read_text().splitlines()
    ]
    return summary, records


def test_score_and_evidence_match_separately_processed_workers(tmp_path, monkeypatch):
    joint, records = _configure(tmp_path / "both", monkeypatch, "AB")
    by_person = joint["task_input_provenance"]["by_person"]
    assert set(by_person) == {"person-0001", "person-0002"}
    assert joint["lift_detection_status"] == "withheld_multiple_tracked_people"
    assert joint["rnle_input_estimate"] is None and joint["detected_lifts"] is None
    assert [r["frame_number"] for r in records] == sorted(r["frame_number"] for r in records)
    for i, tag in enumerate("AB", 1):
        person = f"person-{i:04d}"
        single, standalone = _configure(tmp_path / tag, monkeypatch, tag)
        assert single["task_input_provenance"] == by_person[person]
        combined = [r for r in records if r["person_identifier"] == person]
        assert combined
        for left, right in zip(combined, standalone, strict=True):
            for key in ("frame_number", "angles", "task_conditions"):
                assert left[key] == right[key]
            for method in ("reba", "rula"):
                assert left[method]["selected_side"] == right[method]["selected_side"]
                assert (
                    left[method]["selected"]["final_score"]
                    == right[method]["selected"]["final_score"]
                )


def test_reviewed_neutral_needs_a_single_tracked_person(tmp_path, monkeypatch):
    with pytest.raises(ConfigurationError, match="single tracked person"):
        _configure(tmp_path, monkeypatch, "AB", reviewed=True)
