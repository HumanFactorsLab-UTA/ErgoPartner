"""Audit metadata for source-camera frames and synchronization provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.video_input import DecodedFrame
from pipeline.video_processor import VideoProcessor
from tests.integration.test_video_pipeline import FakePoseEstimator, _complete_task_inputs

PROJECT = Path(__file__).resolve().parents[1]


def _configured_processor() -> VideoProcessor:
    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    config["task_inputs"] = _complete_task_inputs()
    config["quality"]["camera_view_status"] = "suitable"
    config["quality"]["allow_pose_only_wrist"] = True
    return VideoProcessor(config)


def test_camera_frame_evidence_preserves_observed_offsets_and_roles():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    frames = [
        DecodedFrame(10, 1.000, image, "decoder_pts"),
        DecodedFrame(11, 1.012, image, "decoder_pts"),
        DecodedFrame(9, 1.005, image, "frame_index_over_fps_fallback"),
    ]

    references, synchronization = VideoProcessor._camera_frame_evidence(
        frames,
        ["front.mp4", "side.mp4", "iso.mp4"],
        {0: "front", 1: "side", 2: "iso"},
        0.050,
    )

    assert [item["camera_index"] for item in references] == [0, 1, 2]
    assert [item["camera_role"] for item in references] == ["front", "side", "iso"]
    assert [item["source_frame_number"] for item in references] == [10, 11, 9]
    assert [item["timestamp_source"] for item in references] == [
        "decoder_pts",
        "decoder_pts",
        "frame_index_over_fps_fallback",
    ]
    assert references[0]["sync_delta_from_master_sec"] == pytest.approx(0.0)
    assert references[1]["sync_delta_from_master_sec"] == pytest.approx(0.012)
    assert references[2]["sync_delta_from_master_sec"] == pytest.approx(0.005)
    assert synchronization == {
        "mode": "lockstep_pts_gated",
        "master_camera_index": 0,
        "master_camera_role": "front",
        "tolerance_sec": 0.05,
        "timestamp_spread_sec": pytest.approx(0.012),
        "status": "within_tolerance",
        "alignment_performed": False,
    }


def test_out_of_tolerance_evidence_never_claims_alignment():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    _, synchronization = VideoProcessor._camera_frame_evidence(
        [
            DecodedFrame(0, 0.0, image, "decoder_pts"),
            DecodedFrame(0, 0.075, image, "decoder_pts"),
        ],
        ["front.mp4", "side.mp4"],
        {0: "front", 1: "side"},
        0.050,
    )

    assert synchronization["mode"] == "lockstep_pts_gated"
    assert synchronization["status"] == "out_of_tolerance"
    assert synchronization["alignment_performed"] is False


def test_multicamera_pipeline_serializes_frame_evidence_and_summary(monkeypatch, tmp_path):
    processor = _configured_processor()
    monkeypatch.setattr(
        processor,
        "_create_pose_estimators",
        lambda camera_count: [FakePoseEstimator() for _ in range(camera_count)],
    )
    video = str((PROJECT / "test_synthetic_video.mp4").resolve())

    processor.process([video, video, video], str(tmp_path))

    first_record = json.loads(
        (tmp_path / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    summary = json.loads((tmp_path / "video_summary.json").read_text(encoding="utf-8"))

    references = first_record["camera_frame_references"]
    assert set(first_record["angles_by_camera"]) == {"0", "1", "2"}
    assert all(
        "trunk_flexion" in camera_angles
        for camera_angles in first_record["angles_by_camera"].values()
    )
    assert len(references) == 3
    assert [item["camera_role"] for item in references] == ["front", "side", "iso"]
    assert all(item["source_path"] == video for item in references)
    assert all(item["source_frame_number"] == first_record["frame_number"] for item in references)
    assert all(item["timestamp_sec"] == first_record["timestamp_sec"] for item in references)
    assert all(item["sync_delta_from_master_sec"] == pytest.approx(0.0) for item in references)
    assert first_record["synchronization"] == {
        "alignment_performed": False,
        "master_camera_index": 0,
        "master_camera_role": "front",
        "mode": "lockstep_pts_gated",
        "status": "within_tolerance",
        "timestamp_spread_sec": pytest.approx(0.0),
        "tolerance_sec": 0.05,
    }

    synchronization = summary["synchronization"]
    assert synchronization["mode"] == "lockstep_pts_gated"
    assert synchronization["status"] == "within_tolerance"
    assert synchronization["alignment_performed"] is False
    assert synchronization["checked_frame_sets"] == summary["processed_frames"] == 20
    assert synchronization["mean_timestamp_spread_sec"] == pytest.approx(0.0)
    assert synchronization["maximum_timestamp_spread_sec"] == pytest.approx(0.0)
    assert [item["camera_role"] for item in summary["video_metadata"]] == [
        "front",
        "side",
        "iso",
    ]
    # Lift detection now runs on every trial. This fixture contains no person
    # and therefore no lift, which is reported as a measured zero rather than as
    # "not performed".
    assert summary["detected_lifts"] == 0
    assert summary["lift_detection_status"] == "no_lift_detected"
    assert summary["per_lift_rnle_results"] == []
    # An estimate is still recorded, so a reader can see why nothing was found.
    assert summary["rnle_input_estimate"]["lift_count"] == 0
