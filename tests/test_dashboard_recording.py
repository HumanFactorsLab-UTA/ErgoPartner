import json
import time

import numpy as np
import pytest

from dashboard.recording_service import (
    CameraSpec,
    MultiCameraRecordingService,
    RecordingError,
)


class FakeCapture:
    def __init__(self, device_id, *, opened=True):
        self.device_id = device_id
        self.opened = opened
        self.released = False
        self.sequence = 0

    def isOpened(self):
        return self.opened and not self.released

    def set(self, _property, _value):
        return True

    def read(self):
        if not self.isOpened():
            return False, None
        time.sleep(0.003)
        self.sequence += 1
        frame = np.full((48, 64, 3), self.device_id + self.sequence % 20, dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.released = False

    def isOpened(self):
        return not self.released

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def _wait_until(predicate, timeout=2.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for recording service")


def test_camera_specs_and_preview_input_are_strict():
    with pytest.raises(ValueError, match="positive"):
        CameraSpec(0, "Front", width=0)

    service = MultiCameraRecordingService(capture_factory=FakeCapture)
    with pytest.raises(RecordingError, match="at least one"):
        service.start_preview([])
    with pytest.raises(RecordingError, match="unique"):
        service.start_preview([CameraSpec(0, "A"), CameraSpec(0, "B")])


def test_camera_detection_releases_every_probe():
    captures = []

    def factory(device_id):
        capture = FakeCapture(device_id, opened=device_id in {0, 2})
        captures.append(capture)
        return capture

    assert MultiCameraRecordingService.detect_cameras(4, factory) == [0, 2]
    assert all(capture.released for capture in captures)


def test_multi_camera_recording_writes_synchronized_manifest_and_keeps_preview(tmp_path):
    writers = []

    def writer_factory(path, _fps, _size):
        writer = FakeWriter(path)
        writers.append(writer)
        return writer

    service = MultiCameraRecordingService(
        capture_factory=FakeCapture,
        writer_factory=writer_factory,
        sync_tolerance_sec=0.05,
    )
    specs = [
        CameraSpec(0, "Front", width=64, height=48, fps=30),
        CameraSpec(1, "Side", width=64, height=48, fps=30),
    ]
    try:
        service.start_preview(specs)
        _wait_until(lambda: len(service.latest_previews()) == 2)
        service.start_recording(tmp_path)
        _wait_until(lambda: service.snapshot()["synchronized_frame_sets"] >= 3)
        manifest = service.stop_recording()

        assert manifest["analysis_status"] == "not_run"
        assert manifest["synchronized_frame_sets"] >= 3
        assert len(manifest["raw_videos"]) == 2
        assert len(manifest["annotated_videos"]) == 2
        assert service.snapshot()["preview_active"] is True
        assert service.snapshot()["recording"] is False
        timestamp_lines = [
            json.loads(line)
            for line in (tmp_path / f"recording_{manifest['session_id']}_timestamps.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(timestamp_lines) == manifest["synchronized_frame_sets"] * 2
        assert {line["camera_device_id"] for line in timestamp_lines} == {0, 1}
        assert all(writer.released for writer in writers)
        persisted = json.loads(
            (tmp_path / f"recording_{manifest['session_id']}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert persisted["synchronized_frame_sets"] == manifest["synchronized_frame_sets"]
    finally:
        service.stop_preview()


def test_placement_decisions_belong_to_one_view_and_one_take(tmp_path, monkeypatch):
    from dashboard import camera_placement, recording_service

    monkeypatch.setattr(camera_placement, "SAMPLE_INTERVAL", 0.08)
    monkeypatch.setattr(
        recording_service,
        "CameraPlacement",
        lambda ids: camera_placement.CameraPlacement(ids, detector_factory=lambda: None),
    )
    service = MultiCameraRecordingService(
        capture_factory=FakeCapture, writer_factory=lambda path, fps, size: FakeWriter(path)
    )
    try:
        service.start_preview([CameraSpec(4, "Front", 64, 48), CameraSpec(7, "Side", 64, 48)])
        monitor = service.placement()
        _wait_until(lambda: all(c["samples"] >= 3 for c in monitor.status()["cameras"].values()))
        observed = monitor.status()["cameras"][4]
        assert observed["state"] == "unavailable"
        monitor.decide(4, observed["id"], "adjust")
        service.start_recording(tmp_path)
        with pytest.raises(camera_placement.PlacementError):
            service.placement()
        _wait_until(lambda: service.snapshot()["synchronized_frame_sets"] > 0)
        review = service.stop_recording()["placement_review"]
        assert review["cameras"][4]["decision"]["value"] == "adjust"
        assert review["cameras"][7]["decision"] is None
        fresh = service.placement().status()
        assert fresh["preview"] != review["preview"]
        assert fresh["cameras"][4]["decision"] is None
    finally:
        service.stop_preview()
