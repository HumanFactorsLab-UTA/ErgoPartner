"""Recover transiently missing landmarks from neighbouring frames.

The visibility gate is applied per frame in isolation. On real footage that
discards landmarks that are plainly present: knee visibility on the reference
trial sits at 0.40-0.50 against a 0.50 threshold, so REBA never completes on a
single-camera recording even though the knee is tracked continuously and is
well above the threshold on surrounding frames.

This module fills such gaps from the frames on either side, but only when the
result is biomechanically admissible. A landmark is recovered when it is
bracketed by confident observations, the gap is short, the implied segment
length stays consistent with the same segment measured on confident frames, and
the implied speed is physically plausible. Anything else is left missing, which
keeps the pipeline's existing behaviour of refusing rather than guessing.

Recovered samples are marked. They carry the confidence of the evidence they
were derived from, decayed by gap length, and callers can exclude them from peak
claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Parent landmark for each recoverable joint, used to check that a recovered
# position keeps its segment the same length it measures on confident frames.
SEGMENT_PARENT: dict[int, int] = {
    11: 23,  # left shoulder  <- left hip
    12: 24,  # right shoulder <- right hip
    13: 11,  # left elbow     <- left shoulder
    14: 12,  # right elbow    <- right shoulder
    15: 13,  # left wrist     <- left elbow
    16: 14,  # right wrist    <- right elbow
    17: 15,  # left pinky     <- left wrist
    18: 16,  # right pinky    <- right wrist
    19: 15,  # left index     <- left wrist
    20: 16,  # right index    <- right wrist
    25: 23,  # left knee      <- left hip
    26: 24,  # right knee     <- right hip
    27: 25,  # left ankle     <- left knee
    28: 26,  # right ankle    <- right knee
}


@dataclass(frozen=True)
class RecoveryPolicy:
    """Limits under which a missing landmark may be reconstructed."""

    # A landmark at or above this visibility anchors an interpolation.
    anchor_visibility: float = 0.60
    # A landmark below this visibility is treated as missing and eligible for
    # recovery. Between the two thresholds the observation is kept as measured;
    # that band is the hysteresis that stops a marginal landmark flickering.
    missing_visibility: float = 0.50
    # Longest gap that may be bridged.
    max_gap_sec: float = 0.40
    # Recovered segment length must stay within this fraction of the median
    # length measured for the same segment on anchor frames.
    max_segment_strain: float = 0.20
    # Implied landmark speed limit, in model-world metres per second.
    max_speed_m_per_s: float = 6.0
    # Confidence retained by a recovered sample, per second of gap bridged.
    confidence_decay_per_sec: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.missing_visibility <= self.anchor_visibility <= 1.0:
            raise ValueError("require 0 <= missing_visibility <= anchor_visibility <= 1")
        if self.max_gap_sec <= 0 or self.max_segment_strain <= 0 or self.max_speed_m_per_s <= 0:
            raise ValueError("recovery limits must be positive")


@dataclass(frozen=True)
class RecoveredLandmark:
    """One reconstructed landmark, retained as evidence."""

    frame_number: int
    landmark_id: int
    gap_sec: float
    assigned_visibility: float


@dataclass
class RecoveryReport:
    """What the recovery pass changed, and what it declined to change."""

    recovered: list[RecoveredLandmark] = field(default_factory=list)
    rejected_segment_strain: int = 0
    rejected_speed: int = 0
    rejected_gap_too_long: int = 0
    rejected_unbracketed: int = 0

    @property
    def recovered_count(self) -> int:
        return len(self.recovered)

    def by_frame(self) -> dict[int, list[int]]:
        """Recovered landmark ids keyed by frame number."""

        mapping: dict[int, list[int]] = {}
        for item in self.recovered:
            mapping.setdefault(item.frame_number, []).append(item.landmark_id)
        return {frame: sorted(ids) for frame, ids in mapping.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "temporal_linear_interpolation",
            "recovered_landmark_count": self.recovered_count,
            "recovered_frame_count": len(self.by_frame()),
            "rejected_segment_strain": self.rejected_segment_strain,
            "rejected_speed": self.rejected_speed,
            "rejected_gap_too_long": self.rejected_gap_too_long,
            "rejected_unbracketed": self.rejected_unbracketed,
        }


@dataclass
class PoseFrame:
    """One person's landmarks at one instant, in canonical model-world metres."""

    frame_number: int
    timestamp_sec: float
    landmarks: dict[int, np.ndarray]


def _visibility(point: np.ndarray) -> float:
    array = np.asarray(point, dtype=float)
    return float(array[3]) if array.size > 3 else 1.0


def _usable(point: np.ndarray | None) -> bool:
    return point is not None and bool(np.all(np.isfinite(np.asarray(point, dtype=float)[:3])))


def _median_segment_length(
    frames: Sequence[PoseFrame], landmark_id: int, parent_id: int, policy: RecoveryPolicy
) -> float | None:
    """Median distance to the parent joint over frames where both are confident."""

    lengths = []
    for frame in frames:
        child = frame.landmarks.get(landmark_id)
        parent = frame.landmarks.get(parent_id)
        if child is None or parent is None or not _usable(child) or not _usable(parent):
            continue
        if (
            _visibility(child) < policy.anchor_visibility
            or _visibility(parent) < policy.anchor_visibility
        ):
            continue
        lengths.append(float(np.linalg.norm(np.asarray(child)[:3] - np.asarray(parent)[:3])))
    return float(np.median(lengths)) if lengths else None


def recover_landmarks(
    frames: Sequence[PoseFrame],
    policy: RecoveryPolicy | None = None,
) -> tuple[list[PoseFrame], RecoveryReport]:
    """Fill short, biomechanically admissible landmark gaps in a pose sequence.

    Frames must be ordered by time and belong to one tracked person. Returns new
    frames; the inputs are not modified.
    """

    policy = policy or RecoveryPolicy()
    report = RecoveryReport()
    ordered = sorted(frames, key=lambda item: (item.timestamp_sec, item.frame_number))
    if len(ordered) < 3:
        return [
            PoseFrame(item.frame_number, item.timestamp_sec, dict(item.landmarks))
            for item in ordered
        ], report

    output = [
        PoseFrame(item.frame_number, item.timestamp_sec, dict(item.landmarks)) for item in ordered
    ]
    landmark_ids = sorted({index for frame in ordered for index in frame.landmarks})

    for landmark_id in landmark_ids:
        parent_id = SEGMENT_PARENT.get(landmark_id)
        expected_length = (
            _median_segment_length(ordered, landmark_id, parent_id, policy)
            if parent_id is not None
            else None
        )

        # Frames whose observation is confident enough to interpolate between.
        anchors = [
            position
            for position, frame in enumerate(ordered)
            if _usable(frame.landmarks.get(landmark_id))
            and _visibility(frame.landmarks[landmark_id]) >= policy.anchor_visibility
        ]
        if len(anchors) < 2:
            continue
        anchor_set = set(anchors)

        for position, frame in enumerate(ordered):
            if position in anchor_set:
                continue
            existing = frame.landmarks.get(landmark_id)
            # Observations in the hysteresis band are kept as measured.
            if (
                existing is not None
                and _usable(existing)
                and _visibility(existing) >= policy.missing_visibility
            ):
                continue

            before = next((item for item in reversed(anchors) if item < position), None)
            after = next((item for item in anchors if item > position), None)
            if before is None or after is None:
                report.rejected_unbracketed += 1
                continue

            start, end = ordered[before], ordered[after]
            gap_sec = end.timestamp_sec - start.timestamp_sec
            if gap_sec <= 0 or gap_sec > policy.max_gap_sec:
                report.rejected_gap_too_long += 1
                continue

            first = np.asarray(start.landmarks[landmark_id], dtype=float)[:3]
            last = np.asarray(end.landmarks[landmark_id], dtype=float)[:3]
            ratio = (frame.timestamp_sec - start.timestamp_sec) / gap_sec
            estimate = first + (last - first) * ratio

            # Physical plausibility: the bracketed motion must be achievable.
            speed = float(np.linalg.norm(last - first) / gap_sec)
            if speed > policy.max_speed_m_per_s:
                report.rejected_speed += 1
                continue

            # Anatomical plausibility: the segment must not stretch or compress.
            if expected_length is not None and expected_length > 1e-6:
                parent_point = output[position].landmarks.get(parent_id)  # type: ignore[arg-type]
                if _usable(parent_point):
                    length = float(
                        np.linalg.norm(estimate - np.asarray(parent_point, dtype=float)[:3])
                    )
                    strain = abs(length - expected_length) / expected_length
                    if strain > policy.max_segment_strain:
                        report.rejected_segment_strain += 1
                        continue

            confidence = min(
                _visibility(start.landmarks[landmark_id]),
                _visibility(end.landmarks[landmark_id]),
            ) * max(0.0, 1.0 - policy.confidence_decay_per_sec * gap_sec)
            output[position].landmarks[landmark_id] = np.array(
                [estimate[0], estimate[1], estimate[2], confidence], dtype=float
            )
            report.recovered.append(
                RecoveredLandmark(
                    frame_number=frame.frame_number,
                    landmark_id=landmark_id,
                    gap_sec=float(gap_sec),
                    assigned_visibility=float(confidence),
                )
            )

    return output, report
