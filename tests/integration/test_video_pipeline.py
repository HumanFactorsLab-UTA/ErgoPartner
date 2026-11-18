"""End-to-end decoder-to-report test using deterministic synthetic landmarks.

The bundled black video isolates orchestration from pose-model accuracy. A fake
estimator supplies known geometry, while the real decoder, tracker, angle engine,
method adapters, bilateral calculators, temporal aggregation, and writers run.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from core.pose_estimator import PoseLandmarks
from pipeline.video_processor import VideoProcessor

PROJECT = Path(__file__).resolve().parents[2]


def _synthetic_pose(timestamp: float, frame_number: int) -> PoseLandmarks:
    world = {index: np.array([0.0, 1.0, 0.0, 0.99]) for index in range(33)}

    def point(index: int, x: float, y: float, z: float = 0.0) -> None:
        world[index] = np.array([x, y, z, 0.99])

    point(0, 0.0, 1.80)
    point(2, -0.04, 1.78)
    point(5, 0.04, 1.78)
    point(7, -0.08, 1.75)
    point(8, 0.08, 1.75)
    point(11, -0.20, 1.50)
    point(12, 0.20, 1.50)
    point(13, -0.20, 1.20)
    point(14, 0.20, 1.20)
    point(15, -0.20, 0.90)
    point(16, 0.20, 0.90)
    point(17, -0.23, 0.82)
    point(18, 0.23, 0.82)
    point(19, -0.20, 0.80)
    point(20, 0.20, 0.80)
    point(21, -0.17, 0.82)
    point(22, 0.17, 0.82)
    point(23, -0.15, 0.90)
    point(24, 0.15, 0.90)
    point(25, -0.15, 0.45)
    point(26, 0.15, 0.45)
    point(27, -0.15, 0.00)
    point(28, 0.15, 0.00)
    normalized = {
        index: np.array([0.5 + value[0] / 2, 1 - value[1] / 2, 0.0, 0.99])
        for index, value in world.items()
    }
    return PoseLandmarks(
        normalized=normalized,
        world=world,
        timestamp=timestamp,
        frame_number=frame_number,
        image_size=(640, 480),
        mean_keypoint_visibility=0.99,
    )


class FakePoseEstimator:
    def process_frame_all(self, _frame, timestamp=0.0, frame_number=0):
        return [_synthetic_pose(timestamp, frame_number)]

    def close(self):
        pass


class FakeEmptyPoseEstimator:
    def process_frame_all(self, _frame, timestamp=0.0, frame_number=0):
        return []

    def close(self):
        pass


def _complete_task_inputs() -> dict:
    task = {
        "load_kg": 5.0,
        "coupling": "good",
        "shock_or_rapid_force": False,
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "rapid_large_posture_change_or_unstable_base": False,
        "seated": False,
        "legs_weight_bearing": "bilateral",
        "legs_supported_and_balanced": True,
        "trunk_twisted": False,
        "trunk_side_bent": False,
        "trunk_twisted_or_side_flexed": False,
        "neck_twisted": False,
        "neck_side_bent": False,
        "neck_twisted_or_side_flexed": False,
        "rula_load_type": "intermittent",
    }
    for side in ("left", "right"):
        for name in (
            "upper_arm_abducted",
            "upper_arm_abducted_or_rotated",
            "shoulder_raised",
            "arm_supported_or_leaning",
            "arm_supported_or_gravity_assisted",
            "lower_arm_across_midline_or_out_to_side",
            "wrist_deviated",
            "wrist_twist_at_end_range",
            "wrist_deviated_or_twisted",
        ):
            task[f"{side}_{name}"] = False
    return task


def test_real_video_decode_through_bilateral_reports(monkeypatch, tmp_path):
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    config["task_inputs"] = _complete_task_inputs()
    config["quality"]["camera_view_status"] = "suitable"
    config["quality"]["allow_pose_only_wrist"] = True
    processor = VideoProcessor(config)
    monkeypatch.setattr(
        processor,
        "_create_pose_estimators",
        lambda camera_count: [FakePoseEstimator() for _ in range(camera_count)],
    )

    processor.process([str(PROJECT / "test_synthetic_video.mp4")], str(tmp_path))

    summary = json.loads((tmp_path / "video_summary.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (tmp_path / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert summary["processed_frames"] == 20
    assert summary["pose_detected_frames"] == 20
    assert summary["pose_observations"] == 20
    assert summary["video_metadata"][0]["native_fps"] == 10.0
    assert summary["video_metadata"][0]["reported_frame_count"] == 20
    assert summary["rula_summary"]["maximum_score"] is not None
    assert summary["reba_summary"]["maximum_score"] is not None
    assert records[0]["person_identifier"] == "person-0001"
    assert records[0]["pose_landmarks_normalized"]["11"] == [
        0.4,
        0.25,
        0.0,
        0.99,
    ]
    assert records[0]["pose_landmarks_world"]["11"] == [-0.2, 1.5, 0.0, 0.99]
    assert records[0]["pose_quality"]["hand_detection_confidence"] is None
    assert records[0]["rula"]["left"]["intermediate_scores"]
    assert records[0]["rula"]["right"]["intermediate_scores"]
    assert records[0]["rula"]["left"]["validity_status"] == "valid_with_limitations"
    assert records[-1]["timestamp_sec"] > records[0]["timestamp_sec"]
    assert (tmp_path / "ASSESSMENT_REPORT.md").is_file()


@pytest.mark.parametrize(
    ("source_fps", "target_fps", "start_frame", "stride"),
    [(10, 10, 0, 1), (10, 5, 1, 2), (29.97, 14.985, 1, 2)],
)
def test_sampling_keeps_frames_at_interval_boundaries(
    monkeypatch, tmp_path, source_fps, target_fps, start_frame, stride
):
    video = tmp_path / "one-pose.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), source_fps, (64, 48))
    assert writer.isOpened()
    for index in range(22):
        writer.write(np.full((48, 64, 3), 255 if index == 7 else 0, dtype=np.uint8))
    writer.release()

    sampled = []

    class MarkedFrameEstimator(FakePoseEstimator):
        def process_frame_all(self, image, timestamp=0.0, frame_number=0):
            sampled.append(frame_number)
            return [_synthetic_pose(timestamp, frame_number)] if image.mean() > 200 else []

    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    config["video"].update(target_fps=target_fps, start_frame=start_frame)
    config["hand"]["enabled"] = False
    config["task_inputs"] = _complete_task_inputs()
    processor = VideoProcessor(config)
    monkeypatch.setattr(processor, "_create_pose_estimators", lambda count: [MarkedFrameEstimator()])
    output = tmp_path / "results"
    processor.process([str(video)], str(output))

    assert sampled == list(range(start_frame, 22, stride))
    records = (output / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(record)["frame_number"] for record in records] == [7]
    summary = json.loads((output / "video_summary.json").read_text(encoding="utf-8"))
    assert summary["method_completion"]["rula"]["scored_observations"] == 1
    assert summary["method_completion"]["reba"]["scored_observations"] == 1


def test_inclusive_frame_range_preserves_source_frame_numbers(monkeypatch, tmp_path):
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    config["task_inputs"] = _complete_task_inputs()
    config["quality"]["camera_view_status"] = "suitable"
    config["quality"]["allow_pose_only_wrist"] = True
    config["video"]["start_frame"] = 5
    config["video"]["end_frame"] = 8
    processor = VideoProcessor(config)
    monkeypatch.setattr(
        processor,
        "_create_pose_estimators",
        lambda camera_count: [FakePoseEstimator() for _ in range(camera_count)],
    )

    progress = []
    processor.process(
        [str(PROJECT / "test_synthetic_video.mp4")],
        str(tmp_path),
        progress_callback=lambda current, total, stage, message: progress.append(
            (current, total, stage, message)
        ),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((tmp_path / "video_summary.json").read_text(encoding="utf-8"))
    assert [record["frame_number"] for record in records] == [5, 6, 7, 8]
    assert summary["processed_frames"] == 4
    assert records[0]["timestamp_sec"] > 0
    assert progress[0][:3] == (0, 4, "loading_model")
    assert any(item[2] == "processing" and item[0] > 0 for item in progress)
    assert progress[-1][:3] == (4, 4, "completed")


def test_invalid_frame_range_is_rejected_before_video_processing(tmp_path):
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    config["video"]["start_frame"] = 10
    config["video"]["end_frame"] = 9

    with np.testing.assert_raises_regex(ValueError, "end_frame must be greater than or equal"):
        VideoProcessor(config).process([str(PROJECT / "test_synthetic_video.mp4")], str(tmp_path))


def test_multi_camera_missing_pose_frames_still_advance_processing_progress(monkeypatch, tmp_path):
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    processor = VideoProcessor(config)
    monkeypatch.setattr(
        processor,
        "_create_pose_estimators",
        lambda camera_count: [FakeEmptyPoseEstimator() for _ in range(camera_count)],
    )
    progress = []
    video = str(PROJECT / "test_synthetic_video.mp4")

    processor.process(
        [video, video],
        str(tmp_path),
        progress_callback=lambda current, total, stage, message: progress.append(
            (current, total, stage, message)
        ),
    )

    processing_updates = [item for item in progress if item[2] == "processing"]
    assert any(item[0] == 20 for item in processing_updates)
    assert any("no usable pose" in item[3] for item in processing_updates)
    assert progress[-1][:3] == (20, 20, "completed")
