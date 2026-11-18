"""Deterministic frame-to-frame identity tracking for pose detections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class PoseLike(Protocol):
    normalized: dict[int, np.ndarray]


class TrackingError(ValueError):
    pass


@dataclass(frozen=True)
class TrackedPose:
    person_id: str
    pose: Any
    centroid_xy: tuple[float, float]


@dataclass
class _TrackState:
    centroid: np.ndarray
    missed_frames: int = 0


class CentroidPoseTracker:
    """Assign stable IDs using normalized torso centroids.

    This intentionally modest tracker is deterministic and rejects detections
    without a usable torso centroid. It does not claim re-identification after
    long occlusion or across cameras.
    """

    TORSO_LANDMARKS = (11, 12, 23, 24)

    def __init__(self, *, max_distance: float = 0.20, max_missed_frames: int = 15):
        if not 0 < max_distance <= 2**0.5:
            raise TrackingError("max_distance must be in (0, sqrt(2)] normalized units")
        if max_missed_frames < 0:
            raise TrackingError("max_missed_frames must be non-negative")
        self.max_distance = float(max_distance)
        self.max_missed_frames = int(max_missed_frames)
        self._tracks: dict[int, _TrackState] = {}
        self._next_id = 1

    @classmethod
    def _centroid(cls, pose: PoseLike) -> np.ndarray:
        points = []
        for index in cls.TORSO_LANDMARKS:
            landmark = pose.normalized.get(index)
            if landmark is not None and len(landmark) >= 2 and np.all(np.isfinite(landmark[:2])):
                points.append(np.asarray(landmark[:2], dtype=float))
        if len(points) < 2:
            raise TrackingError("pose lacks at least two finite torso landmarks")
        return np.mean(points, axis=0)

    def update(self, detections: Iterable[PoseLike]) -> list[TrackedPose]:
        poses = list(detections)
        centroids = [self._centroid(pose) for pose in poses]
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()

        candidates = []
        for track_id, state in self._tracks.items():
            for detection_index, centroid in enumerate(centroids):
                distance = float(np.linalg.norm(state.centroid - centroid))
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, detection_index))
        for _, track_id, detection_index in sorted(candidates):
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        for detection_index in range(len(poses)):
            if detection_index not in assignments:
                assignments[detection_index] = self._next_id
                self._tracks[self._next_id] = _TrackState(centroids[detection_index])
                self._next_id += 1

        assigned_track_ids = set(assignments.values())
        for track_id, state in list(self._tracks.items()):
            if track_id not in assigned_track_ids:
                state.missed_frames += 1
                if state.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]
        for detection_index, track_id in assignments.items():
            state = self._tracks[track_id]
            state.centroid = centroids[detection_index]
            state.missed_frames = 0

        return [
            TrackedPose(
                person_id=f"person-{assignments[index]:04d}",
                pose=pose,
                centroid_xy=(float(centroids[index][0]), float(centroids[index][1])),
            )
            for index, pose in enumerate(poses)
        ]
