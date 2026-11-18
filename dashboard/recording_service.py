"""Threaded multi-camera recording backend for the Streamlit dashboard.

This ports the capture behavior of the Version 2 recorder without importing its
legacy ergonomic scoring. It records timestamp-synchronized raw and annotated
views that can be processed by the audited v4 pipeline after capture.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dashboard.camera_placement import CameraPlacement, PlacementError


class RecordingError(RuntimeError):
    """Raised when cameras or recording writers cannot be initialized safely."""


@dataclass(frozen=True)
class CameraSpec:
    device_id: int
    label: str
    width: int = 1280
    height: int = 720
    fps: float = 30.0

    def __post_init__(self) -> None:
        if self.device_id < 0:
            raise ValueError("camera device_id must be non-negative")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("camera width, height, and fps must be positive")


@dataclass
class CameraStats:
    device_id: int
    label: str
    captured_frames: int = 0
    recorded_frames: int = 0
    actual_fps: float = 0.0
    last_timestamp: float | None = None
    running: bool = False
    error: str | None = None


@dataclass
class RecordingSession:
    session_id: str
    started_at: str
    started_monotonic: float
    raw_writers: dict[int, Any]
    overlay_writers: dict[int, Any]
    raw_paths: dict[int, str]
    overlay_paths: dict[int, str]
    timestamps_path: str
    manifest_path: str
    timestamp_handle: Any
    placement: dict | None = None
    synchronized_sets: int = 0
    dropped_sets: int = 0
    maximum_sync_delta_sec: float = 0.0
    completed: bool = False


class _CameraStream(threading.Thread):
    def __init__(
        self,
        spec: CameraSpec,
        callback: Callable[[int, np.ndarray, float, int], None],
        capture_factory: Callable[[int], Any],
    ):
        super().__init__(daemon=True, name=f"camera-{spec.device_id}")
        self.spec = spec
        self.callback = callback
        self.capture_factory = capture_factory
        self.capture = None
        self.stop_event = threading.Event()
        self.opened = False
        self.stats = CameraStats(spec.device_id, spec.label)

    def open(self) -> None:
        self.capture = self.capture_factory(self.spec.device_id)
        if self.capture is None or not self.capture.isOpened():
            self.close()
            raise RecordingError(f"Could not open camera {self.spec.device_id}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self.opened = True

    def run(self) -> None:
        if not self.opened or self.capture is None:
            return
        self.stats.running = True
        interval_start = time.perf_counter()
        interval_count = 0
        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                timestamp = time.perf_counter()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self.stats.error = (
                            f"Camera {self.spec.device_id} returned 30 consecutive empty frames"
                        )
                        break
                    time.sleep(0.002)
                    continue
                consecutive_failures = 0
                self.stats.captured_frames += 1
                self.stats.last_timestamp = timestamp
                interval_count += 1
                elapsed = timestamp - interval_start
                if elapsed >= 1.0:
                    self.stats.actual_fps = interval_count / elapsed
                    interval_count = 0
                    interval_start = timestamp
                self.callback(
                    self.spec.device_id,
                    frame,
                    timestamp,
                    self.stats.captured_frames,
                )
        except Exception as exc:  # capture backend failures must reach the UI
            self.stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stats.running = False
            self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.opened = False

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=3.0)
        self.close()


def _default_capture_factory(device_id: int):
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    return cv2.VideoCapture(device_id, backend)


def _default_writer_factory(path: Path, fps: float, size: tuple[int, int]):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    return writer


class MultiCameraRecordingService:
    """Live preview and synchronized raw/annotated multi-camera recording."""

    def __init__(
        self,
        *,
        capture_factory: Callable[[int], Any] = _default_capture_factory,
        writer_factory: Callable[[Path, float, tuple[int, int]], Any] = (_default_writer_factory),
        sync_tolerance_sec: float = 0.050,
        queue_size: int = 120,
    ):
        if sync_tolerance_sec <= 0:
            raise ValueError("sync_tolerance_sec must be positive")
        self.capture_factory = capture_factory
        self.writer_factory = writer_factory
        self.sync_tolerance_sec = sync_tolerance_sec
        self.queue_size = queue_size
        self._lock = threading.RLock()
        self._streams: dict[int, _CameraStream] = {}
        self._specs: dict[int, CameraSpec] = {}
        self._latest: dict[int, tuple[np.ndarray, float, int]] = {}
        self._last_synchronized_sequence: dict[int, int] = {}
        self._record_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._writer_stop = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._session: RecordingSession | None = None
        self._placement: CameraPlacement | None = None
        self._last_completed_manifest: str | None = None

    @staticmethod
    def detect_cameras(
        maximum_device_id: int = 8,
        capture_factory: Callable[[int], Any] = _default_capture_factory,
    ) -> list[int]:
        if maximum_device_id < 1:
            raise ValueError("maximum_device_id must be at least 1")
        detected: list[int] = []
        for device_id in range(maximum_device_id):
            capture = capture_factory(device_id)
            try:
                if capture is not None and capture.isOpened():
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        detected.append(device_id)
            finally:
                if capture is not None:
                    capture.release()
        return detected

    @property
    def preview_active(self) -> bool:
        with self._lock:
            return bool(self._streams) and any(
                stream.stats.running for stream in self._streams.values()
            )

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._session is not None and not self._session.completed

    def start_preview(self, specs: list[CameraSpec]) -> None:
        if not specs:
            raise RecordingError("Select at least one camera")
        if len(specs) > 3:
            raise RecordingError("At most three synchronized cameras are supported")
        device_ids = [spec.device_id for spec in specs]
        if len(device_ids) != len(set(device_ids)):
            raise RecordingError("Camera device IDs must be unique")
        self.stop_preview()
        streams: dict[int, _CameraStream] = {}
        try:
            for spec in specs:
                stream = _CameraStream(spec, self._on_frame, self.capture_factory)
                stream.open()
                streams[spec.device_id] = stream
        except Exception:
            for stream in streams.values():
                stream.close()
            raise
        with self._lock:
            self._specs = {spec.device_id: spec for spec in specs}
            self._streams = streams
            self._placement = CameraPlacement(device_ids)
            self._latest.clear()
            self._last_synchronized_sequence.clear()
        for stream in streams.values():
            stream.start()

    def stop_preview(self) -> None:
        if self.is_recording:
            self.stop_recording()
        with self._lock:
            streams = list(self._streams.values())
            if self._placement:
                self._placement.close()
                self._placement = None
            self._streams = {}
            self._specs = {}
            self._latest.clear()
        for stream in streams:
            stream.stop()

    def _create_writer(self, path: Path, spec: CameraSpec) -> tuple[Any, str]:
        writer = self.writer_factory(path, spec.fps, (spec.width, spec.height))
        if writer is None or not writer.isOpened():
            if writer is not None:
                writer.release()
            raise RecordingError(f"Could not create video writer for {path.name}")
        return writer, str(path)

    def start_recording(self, task_path: str | Path) -> str:
        with self._lock:
            if self.is_recording:
                raise RecordingError("A recording is already active")
            if not self.preview_active:
                raise RecordingError("Initialize camera preview before recording")
            specs = dict(self._specs)
        task_root = Path(task_path)
        raw_dir = task_root / "raw"
        overlay_dir = task_root / "overlay"
        raw_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_writers: dict[int, Any] = {}
        overlay_writers: dict[int, Any] = {}
        raw_paths: dict[int, str] = {}
        overlay_paths: dict[int, str] = {}
        timestamp_handle = None
        try:
            for device_id, spec in specs.items():
                raw_writer, raw_path = self._create_writer(
                    raw_dir / f"capture_{session_id}_cam_{device_id}_raw.mp4",
                    spec,
                )
                overlay_writer, overlay_path = self._create_writer(
                    overlay_dir / f"capture_{session_id}_cam_{device_id}_overlay.mp4",
                    spec,
                )
                raw_writers[device_id] = raw_writer
                overlay_writers[device_id] = overlay_writer
                raw_paths[device_id] = raw_path
                overlay_paths[device_id] = overlay_path
            timestamps_path = task_root / f"recording_{session_id}_timestamps.jsonl"
            timestamp_handle = timestamps_path.open("w", encoding="utf-8")
            manifest_path = task_root / f"recording_{session_id}_manifest.json"
            session = RecordingSession(
                session_id=session_id,
                started_at=datetime.now().isoformat(),
                started_monotonic=time.perf_counter(),
                raw_writers=raw_writers,
                overlay_writers=overlay_writers,
                raw_paths=raw_paths,
                overlay_paths=overlay_paths,
                timestamps_path=str(timestamps_path),
                manifest_path=str(manifest_path),
                timestamp_handle=timestamp_handle,
            )
        except Exception:
            for writer in [*raw_writers.values(), *overlay_writers.values()]:
                writer.release()
            if timestamp_handle is not None:
                timestamp_handle.close()
            raise
        with self._lock:
            if self._placement:
                session.placement = self._placement.close(wait=False)
                self._placement = None
            self._session = session
            self._last_synchronized_sequence.clear()
            while not self._record_queue.empty():
                try:
                    self._record_queue.get_nowait()
                except queue.Empty:
                    break
            self._writer_stop.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                daemon=True,
                name="recording-writer",
            )
            self._writer_thread.start()
        return session_id

    def _on_frame(
        self,
        device_id: int,
        frame: np.ndarray,
        timestamp: float,
        sequence: int,
    ) -> None:
        with self._lock:
            self._latest[device_id] = (frame.copy(), timestamp, sequence)
            if self._placement:
                self._placement.offer(device_id, self._latest[device_id][0], timestamp, sequence)
            session = self._session
            if session is None or session.completed:
                return
            expected = set(self._specs)
            if set(self._latest) != expected:
                return
            packets = {camera_id: self._latest[camera_id] for camera_id in expected}
            if any(
                packet[2] <= self._last_synchronized_sequence.get(camera_id, 0)
                for camera_id, packet in packets.items()
            ):
                return
            timestamps = [packet[1] for packet in packets.values()]
            sync_delta = max(timestamps) - min(timestamps)
            if sync_delta > self.sync_tolerance_sec:
                return
            batch = {
                camera_id: (packet[0].copy(), packet[1], packet[2])
                for camera_id, packet in packets.items()
            }
            try:
                self._record_queue.put_nowait((batch, sync_delta))
                self._last_synchronized_sequence = {
                    camera_id: packet[2] for camera_id, packet in packets.items()
                }
            except queue.Full:
                session.dropped_sets += 1

    @staticmethod
    def _annotated_frame(
        frame: np.ndarray,
        spec: CameraSpec,
        elapsed: float,
        frame_number: int,
    ) -> np.ndarray:
        output = frame.copy()
        cv2.rectangle(output, (0, 0), (output.shape[1], 72), (5, 8, 18), -1)
        cv2.circle(output, (24, 25), 8, (30, 30, 220), -1)
        cv2.putText(
            output,
            f"REC  {spec.label}",
            (44, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"{elapsed:07.2f}s  |  synchronized frame {frame_number:,}",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 225, 240),
            1,
            cv2.LINE_AA,
        )
        return output

    def _writer_loop(self) -> None:
        while not self._writer_stop.is_set() or not self._record_queue.empty():
            try:
                batch, sync_delta = self._record_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                session = self._session
                specs = dict(self._specs)
            if session is None:
                continue
            frame_number = session.synchronized_sets
            elapsed_values = []
            for device_id, (frame, timestamp, sequence) in batch.items():
                spec = specs[device_id]
                if frame.shape[1] != spec.width or frame.shape[0] != spec.height:
                    frame = cv2.resize(frame, (spec.width, spec.height))
                elapsed = max(0.0, timestamp - session.started_monotonic)
                elapsed_values.append(elapsed)
                session.raw_writers[device_id].write(frame)
                session.overlay_writers[device_id].write(
                    self._annotated_frame(frame, spec, elapsed, frame_number)
                )
                stream = self._streams.get(device_id)
                if stream is not None:
                    stream.stats.recorded_frames += 1
                session.timestamp_handle.write(
                    json.dumps(
                        {
                            "synchronized_frame": frame_number,
                            "camera_device_id": device_id,
                            "camera_sequence": sequence,
                            "timestamp_sec": elapsed,
                            "sync_delta_sec": sync_delta,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            session.timestamp_handle.flush()
            session.synchronized_sets += 1
            session.maximum_sync_delta_sec = max(session.maximum_sync_delta_sec, sync_delta)

    def stop_recording(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None or session.completed:
                raise RecordingError("No recording is active")
            session.completed = True
            self._writer_stop.set()
            writer_thread = self._writer_thread
        if writer_thread is not None:
            writer_thread.join(timeout=5.0)
        ended_at = datetime.now().isoformat()
        duration = max(0.0, time.perf_counter() - session.started_monotonic)
        for writer in [*session.raw_writers.values(), *session.overlay_writers.values()]:
            writer.release()
        session.timestamp_handle.close()
        manifest = {
            "session_id": session.session_id,
            "started_at": session.started_at,
            "ended_at": ended_at,
            "duration_sec": duration,
            "camera_specs": [asdict(spec) for spec in self._specs.values()],
            "synchronized_frame_sets": session.synchronized_sets,
            "dropped_frame_sets": session.dropped_sets,
            "maximum_sync_delta_sec": session.maximum_sync_delta_sec,
            "raw_videos": session.raw_paths,
            "annotated_videos": session.overlay_paths,
            "timestamps_jsonl": session.timestamps_path,
            "analysis_status": "not_run",
            "placement_review": session.placement,
            "notes": [
                "Annotated recordings contain capture status only; ergonomic scores are added by the audited post-processing pipeline.",
                "Synchronization uses host monotonic timestamps and is not hardware genlock.",
            ],
        }
        Path(session.manifest_path).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self._lock:
            self._last_completed_manifest = session.manifest_path
            self._session = None
            self._writer_thread = None
            if self._streams:
                self._placement = CameraPlacement(self._specs)
        return manifest

    def placement(self) -> CameraPlacement:
        with self._lock:
            if self._placement is None or self.is_recording:
                raise PlacementError("Placement checks are available during camera preview before recording.")
            self._placement.start()
            return self._placement

    def latest_previews(self) -> dict[int, np.ndarray]:
        with self._lock:
            return {device_id: packet[0].copy() for device_id, packet in self._latest.items()}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            camera_stats = {
                device_id: asdict(stream.stats) for device_id, stream in self._streams.items()
            }
            return {
                "preview_active": self.preview_active,
                "recording": self.is_recording,
                "session_id": session.session_id if session else None,
                "elapsed_sec": (
                    max(0.0, time.perf_counter() - session.started_monotonic) if session else 0.0
                ),
                "synchronized_frame_sets": (session.synchronized_sets if session else 0),
                "dropped_frame_sets": session.dropped_sets if session else 0,
                "maximum_sync_delta_sec": (session.maximum_sync_delta_sec if session else 0.0),
                "cameras": camera_stats,
                "last_completed_manifest": self._last_completed_manifest,
            }


_RECORDING_SERVICE = MultiCameraRecordingService()


def get_recording_service() -> MultiCameraRecordingService:
    return _RECORDING_SERVICE
