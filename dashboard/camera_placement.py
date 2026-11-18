"""Bounded, per-camera checks of the live rehearsal. Capture never waits for inference."""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from paths import MODELS_DIR

LOG = logging.getLogger(__name__)
SAMPLE_INTERVAL = 0.85
WINDOW = 5
MIN_SAMPLES = 3
FRESH_SECONDS = 5
VISUAL_TTL = 60
_VISION_SLOT = threading.BoundedSemaphore(1)

# A region can be partially visible, for example when a side camera sees one arm.
GROUPS = {
    "Head": ((0, 7, 8),),
    "Trunk": ((11, 12), (23, 24)),
    "Arms": ((13,), (14,)),
    "Hands": ((15, 19), (16, 20)),
    "Legs": ((25,), (26,), (27,), (28,)),
}


class PlacementError(ValueError):
    pass


def _pose_detector():
    from core.pose_estimator import PoseEstimator

    path = MODELS_DIR / "pose_landmarker_heavy.task"
    if not path.is_file():
        raise PlacementError("The local pose model is unavailable.")
    return PoseEstimator(model_path=str(path), max_poses=2, independent_images=True)


def measure(frame: np.ndarray, detector: Any) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray[::4, ::4], [10, 90])
    lighting = "dark" if high < 45 else "washed out" if low > 215 else None
    poses = detector.process_frame_all(frame) if detector is not None else None
    result: dict[str, Any] = {"people": None if poses is None else len(poses), "lighting": lighting}
    if poses is None or len(poses) != 1:
        return result
    points = poses[0].normalized

    def quality(indices):
        reliable = [points[i] for i in indices if i in points and points[i][3] >= 0.55]
        if any(0.025 < p[0] < 0.975 and 0.025 < p[1] < 0.975 for p in reliable):
            return "visible"
        return "edge" if reliable else "unclear"

    regions = {}
    for name, groups in GROUPS.items():
        statuses = [quality(indices) for indices in groups]
        if all(s == "visible" for s in statuses):
            regions[name] = "visible"
        elif "edge" in statuses:
            regions[name] = "edge"
        else:
            regions[name] = "partial" if "visible" in statuses else "unclear"
    result["regions"] = regions
    ys = [p[1] for p in points.values() if p[3] >= 0.55 and 0 <= p[1] <= 1]
    result["small"] = bool(
        ys
        and regions["Trunk"] == "visible"
        and regions["Legs"] == "visible"
        and max(ys) - min(ys) < 0.32
    )
    return result


@dataclass
class Sample:
    sequence: int
    timestamp: float
    metrics: dict
    jpeg: bytes
    border: np.ndarray


def _border_signature(frame: np.ndarray) -> np.ndarray:
    thumbnail = cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA)
    return np.concatenate(
        (
            thumbnail[:2].reshape(-1, 3),
            thumbnail[-2:].reshape(-1, 3),
            thumbnail[2:-2, :2].reshape(-1, 3),
            thumbnail[2:-2, -2:].reshape(-1, 3),
        )
    ).astype(np.int16)


def _view_moved(samples: deque[Sample], anchor: dict) -> bool:
    last = list(samples)[-2:]
    return len(last) == 2 and all(
        sample.timestamp > anchor["sampleTime"]
        and np.mean(np.max(np.abs(sample.border - anchor["border"]), axis=1) > 45) > 0.45
        for sample in last
    )


def summarize(samples: list[Sample], now: float) -> dict:
    recent = [item for item in samples if now - item.timestamp <= (WINDOW + 2) * SAMPLE_INTERVAL]
    result: dict[str, Any] = {
        "state": "warming",
        "samples": len(recent),
        "issues": [],
        "regions": [],
    }
    issues = result["issues"]
    if not recent or now - recent[-1].timestamp > FRESH_SECONDS:
        if samples:
            result.update(
                state="unavailable",
                issues=["No recent camera frames. Check the device connection."],
            )
        return result
    if len(recent) < MIN_SAMPLES:
        return result
    rows = [s.metrics for s in recent]
    threshold = math.ceil(len(rows) * 0.6)
    for lighting, advice in (
        ("dark", "Add light in front of the worker."),
        ("washed out", "Reduce glare or backlighting."),
    ):
        if sum(row["lighting"] == lighting for row in rows) >= threshold:
            issues.append(f"The view is repeatedly {lighting}. {advice}")
    if all(r["people"] is None for r in rows):
        result["state"] = "unavailable"
        issues.append("Pose coverage is unavailable. Inspect the preview visually.")
        return result
    if sum(r["people"] == 1 for r in rows) < threshold:
        issues.append(
            "Multiple people appear in this view. Isolate the worker during the rehearsal."
            if any((r["people"] or 0) > 1 for r in rows)
            else "A single worker is not consistently detected. Center the worker and check the lighting."
        )
        result["state"] = "attention"
        return result
    if sum(bool(r.get("small")) for r in rows) >= threshold:
        issues.append(
            "The worker occupies little of the image. Move closer while keeping the task in view."
        )
    for name in GROUPS:
        readings = [r.get("regions", {}).get(name, "unclear") for r in rows]
        if readings.count("visible") >= math.ceil(len(rows) * 0.8):
            status = "visible"
        elif readings.count("edge") >= threshold:
            status = "near edge"
        elif readings.count("visible") >= 2:
            status = "intermittent"
        else:
            status = "partial" if readings.count("partial") >= threshold else "unclear"
        result["regions"].append({"name": name, "status": status})
    for statuses, advice in [
        ({"near edge"}, "Widen or re-center the view"),
        ({"intermittent"}, "Check coverage throughout the movement"),
        ({"partial", "unclear"}, "Try another angle or remove an obstruction"),
    ]:
        names = [r["name"].lower() for r in result["regions"] if r["status"] in statuses]
        if names:
            issues.append(f"{advice} to see the {', '.join(names)}.")
    result["state"] = "attention" if issues else "clear"
    return result


class CameraPlacement:
    """One monitor per preview lifetime, one independent-image detector for all its cameras."""

    def __init__(self, devices, *, detector_factory=_pose_detector, visual_reader=None):
        self.id = uuid.uuid4().hex
        self.detector_factory = detector_factory
        self.visual_reader = visual_reader
        self._condition = threading.Condition()
        self._samples: dict[int, deque[Sample]] = {d: deque(maxlen=WINDOW) for d in devices}
        self._pending: dict[int, tuple[np.ndarray, float, int]] = {}
        self._last_offered: dict[int, float] = {}
        self._worker: threading.Thread | None = None
        self._closed = False
        self._signature: dict[int, tuple] = {}
        self._version = dict.fromkeys(self._samples, 0)
        self._visual: dict[int, dict] = {}
        self._evidence: dict[int, dict[int, bytes]] = {}
        self._decisions: dict[int, dict] = {}

    def start(self):
        with self._condition:
            if self._worker is None and not self._closed:
                self._worker = threading.Thread(
                    target=self._run, name="placement-preview", daemon=True
                )
                self._worker.start()

    def offer(self, device: int, frame: np.ndarray, timestamp: float, sequence: int):
        # Camera stream supplies its owned copy. Never decode, resize or infer on this thread.
        with self._condition:
            if (
                self._closed
                or self._worker is None
                or device not in self._samples
                or timestamp - self._last_offered.get(device, -math.inf) < SAMPLE_INTERVAL
            ):
                return
            self._last_offered[device] = timestamp
            self._pending[device] = (frame, timestamp, sequence)
            self._condition.notify()

    def _run(self):
        detector = None
        try:
            try:
                detector = self.detector_factory()
            except Exception:
                LOG.warning("Pose detector unavailable for placement preview", exc_info=True)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending)
                    if self._closed:
                        return
                    packets, self._pending = self._pending, {}
                for device, (image, timestamp, sequence) in packets.items():
                    with self._condition:
                        if self._closed:
                            return
                    h, w = image.shape[:2]
                    scale = min(1.0, 768 / max(w, h))
                    frame = (
                        cv2.resize(
                            image,
                            (max(1, round(w * scale)), max(1, round(h * scale))),
                            interpolation=cv2.INTER_AREA,
                        )
                        if scale < 1
                        else image
                    )
                    try:
                        metrics = measure(frame, detector)
                    except Exception:
                        LOG.warning("Preview pose detection failed", exc_info=True)
                        metrics = measure(frame, None)
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        with self._condition:
                            if not self._closed:
                                self._samples[device].append(
                                    Sample(
                                        sequence,
                                        timestamp,
                                        metrics,
                                        buf.tobytes(),
                                        _border_signature(frame),
                                    )
                                )
                                self._state(device)
        finally:
            if detector is not None:
                detector.close()

    def _state(self, device: int) -> dict:
        if self._closed or device not in self._samples:
            raise PlacementError("This camera preview has ended.")
        now = time.perf_counter()
        result = summarize(list(self._samples[device]), now)
        signature = (
            result["state"],
            tuple(result["issues"]),
            tuple(r["status"] for r in result["regions"]),
        )
        if self._signature.get(device) != signature:
            self._signature[device] = signature
            self._version[device] += 1
            self._decisions.pop(device, None)
        visual = self._visual.get(device)
        if (
            visual
            and visual["state"] != "expired"
            and (
                _view_moved(self._samples[device], visual)
                or now - visual["sampleTime"] > VISUAL_TTL
                or result["samples"] < MIN_SAMPLES
            )
        ):
            visual = {
                "state": "expired",
                "detail": "The sampled view changed or expired. Inspect the current rehearsal again.",
                "frames": [],
                "capturedAt": visual["capturedAt"],
                "checkId": visual["checkId"],
            }
            self._visual[device] = visual
            self._evidence.pop(device, None)
            self._decisions.pop(device, None)
            self._version[device] += 1
        decision = self._decisions.get(device)
        if decision and (
            _view_moved(self._samples[device], decision)
            or now - decision["sampleTime"] > VISUAL_TTL
        ):
            self._decisions.pop(device, None)
            self._version[device] += 1
            decision = None
        return {
            **result,
            "id": f"{self.id}-{device}-{self._version[device]}",
            "visual": {k: v for k, v in visual.items() if k not in {"sampleTime", "border"}}
            if visual
            else None,
            "decision": {k: v for k, v in decision.items() if k not in {"sampleTime", "border"}}
            if decision
            else None,
        }

    def status(self) -> dict:
        with self._condition:
            if self._closed:
                raise PlacementError("This camera preview has ended.")
            cameras = {d: self._state(d) for d in self._samples}
            missing = []
            if len(cameras) > 1 and all(c["samples"] >= MIN_SAMPLES for c in cameras.values()):
                visible = {
                    region["name"]
                    for camera in cameras.values()
                    for region in camera["regions"]
                    if region["status"] == "visible"
                }
                missing = [name for name in GROUPS if name not in visible]
            return {"preview": self.id, "cameras": cameras, "noClearView": missing}

    def inspect(self, device: int, expected: str, *, model: str | None = None):
        from assistant.backend import BackendError
        from assistant.placement import inspect_placement

        with self._condition:
            state = self._state(device)
            if state["id"] != expected or state["samples"] < MIN_SAMPLES:
                raise PlacementError(
                    "The preview changed or is still warming up. Check the current view."
                )
            if not _VISION_SLOT.acquire(blocking=False):
                raise PlacementError(
                    "A placement inspection is running. Try again after it finishes."
                )
            samples = list(self._samples[device])[-state["samples"] :]
            shown = [samples[0], samples[len(samples) // 2], samples[-1]]
            check_id = uuid.uuid4().hex
            visual = {
                "state": "checking",
                "checkId": check_id,
                "sampleTime": shown[-1].timestamp,
                "border": shown[-1].border,
                "capturedAt": datetime.now(UTC).isoformat(),
                "frames": [s.sequence for s in shown],
            }
            self._visual[device] = visual
            self._evidence[device] = {s.sequence: s.jpeg for s in shown}
            self._version[device] += 1
            self._decisions.pop(device, None)

        def read():
            try:
                images = [(s.sequence, s.jpeg) for s in shown]
                reading = (self.visual_reader(device, images) if self.visual_reader
                           else inspect_placement(device, images, model=model))
                outcome = {**reading, "state": "ready"}
            except Exception as exc:
                LOG.info("Visual placement check failed: %s", exc)
                outcome = {
                    "state": "unavailable",
                    "detail": str(exc) if isinstance(exc, BackendError) else
                    "Visual inspection unavailable. Use the live image and region checks, or retry.",
                }
            finally:
                _VISION_SLOT.release()
            with self._condition:
                if not self._closed and self._visual.get(device) is visual:
                    visual.update(outcome)
                    self._version[device] += 1
                    self._decisions.pop(device, None)

        threading.Thread(target=read, name="placement-vision", daemon=True).start()

    def decide(self, device: int, expected: str, choice: str):
        with self._condition:
            state = self._state(device)
            if choice not in {"ready", "adjust", "unclear"}:
                raise PlacementError("Choose a valid placement decision.")
            if (
                state["id"] != expected
                or state["samples"] < MIN_SAMPLES
                or (state["visual"] or {}).get("state") == "checking"
            ):
                raise PlacementError("The placement check has changed. Review the current preview.")
            self._decisions[device] = {
                "value": choice,
                "at": datetime.now(UTC).isoformat(),
                "reviewId": expected,
                "frames": [s.sequence for s in self._samples[device]],
                "sampleTime": self._samples[device][-1].timestamp,
                "border": self._samples[device][-1].border,
            }

    def evidence(self, device: int, check_id: str, frame: int) -> bytes:
        with self._condition:
            state = self._state(device)
            if (state["visual"] or {}).get("checkId") != check_id:
                raise PlacementError("These preview images have expired.")
            image = self._evidence.get(device, {}).get(frame)
            if image is None:
                raise PlacementError("This image was not inspected.")
            return image

    def close(self, *, wait: bool = True) -> dict:
        with self._condition:
            saved = copy.deepcopy(self.status())
            self._closed = True
            self._pending.clear()
            self._samples.clear()
            self._evidence.clear()
            self._condition.notify_all()
        if wait and self._worker is not None:
            self._worker.join(timeout=4)
        return saved
