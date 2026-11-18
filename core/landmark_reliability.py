"""Judge whether a landmark was really observed, using the whole sequence.

The pipeline gates every measurement on MediaPipe's per-frame ``visibility``
against a fixed threshold. That scalar is a model confidence with no calibrated
meaning -- it is not a probability of occlusion and not a measurement error --
and using it as a hard gate discards joints that were tracked perfectly well.

On the reference single-camera trial the knee sits at 0.40-0.50 against a 0.50
threshold for the whole recording. Every REBA observation is withheld as
``insufficient_pose_quality``, so single-camera runs produce no REBA at all,
while the knee is in fact visible and continuously tracked in the footage.

Temporal recovery cannot rescue that case: it interpolates a landmark that drops
out between two *confident* observations, and here there are no confident
observations to bracket with. The landmark is persistently marginal rather than
transiently absent.

This module supplies the missing evidence. A landmark that appears on nearly
every frame and moves smoothly is being tracked, whatever the confidence scalar
says, and a landmark that flickers in and out or teleports is not -- regardless
of how confident any single frame claims to be. Presence and coherence are
properties of the sequence and cannot be read off one frame, which is why the
per-frame gate could never see them.

Admission is recorded per measurement, so a score that rests on this evidence
says so rather than appearing indistinguishable from a high-confidence one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReliabilityPolicy:
    """What counts as sustained tracking."""

    #: Fraction of assessed frames on which the landmark must carry a finite
    #: position. A joint that genuinely leaves the view fails this.
    minimum_presence: float = 0.90
    #: Confidence floor below which sustained tracking is not accepted as
    #: evidence. Well under the per-frame gate, but not zero: a landmark the
    #: model has no confidence in at all is a hallucination, and a smooth
    #: trajectory of hallucinations is still a hallucination.
    minimum_median_visibility: float = 0.30
    #: Largest median frame-to-frame step, as a fraction of the subject's own
    #: torso length, for motion to count as smooth. A tracked joint moves a
    #: small fraction of a torso between frames; a mislocated one jumps.
    maximum_median_step: float = 0.08
    #: A sequence shorter than this cannot establish either property.
    minimum_frames: int = 20

    def __post_init__(self) -> None:
        if not 0 < self.minimum_presence <= 1:
            raise ValueError("minimum_presence must be in (0, 1]")
        if not 0 <= self.minimum_median_visibility <= 1:
            raise ValueError("minimum_median_visibility must be in [0, 1]")
        if self.maximum_median_step <= 0:
            raise ValueError("maximum_median_step must be positive")


@dataclass(frozen=True)
class LandmarkReliability:
    """Which landmarks the sequence shows were tracked, and on what evidence."""

    presence: dict[int, float] = field(default_factory=dict)
    median_visibility: dict[int, float] = field(default_factory=dict)
    median_step: dict[int, float] = field(default_factory=dict)
    #: Landmarks admitted on sustained-tracking evidence.
    admitted: frozenset[int] = frozenset()
    frame_count: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "admitted_landmark_ids": sorted(self.admitted),
            "presence": {str(k): round(v, 3) for k, v in sorted(self.presence.items())},
            "median_visibility": {
                str(k): round(v, 3) for k, v in sorted(self.median_visibility.items())
            },
            "median_step_per_torso": {
                str(k): round(v, 4) for k, v in sorted(self.median_step.items())
            },
            "basis": (
                "Admitted landmarks appeared on nearly every assessed frame and moved "
                "smoothly, which is evidence of tracking that a single frame's confidence "
                "scalar cannot carry. Measurements resting on them are marked "
                "sustained_tracking in the frame record."
            ),
        }


def _torso_length(landmarks: dict[int, np.ndarray]) -> float | None:
    """Mid-shoulder to mid-hip, used to normalise motion by body size."""

    try:
        shoulders = (np.asarray(landmarks[11])[:3] + np.asarray(landmarks[12])[:3]) / 2.0
        hips = (np.asarray(landmarks[23])[:3] + np.asarray(landmarks[24])[:3]) / 2.0
    except (KeyError, TypeError, IndexError):
        return None
    length = float(np.linalg.norm(shoulders - hips))
    return length if length > 1e-3 else None


def assess_landmarks(
    sequence: Sequence[dict[int, np.ndarray]],
    policy: ReliabilityPolicy | None = None,
) -> LandmarkReliability:
    """Measure presence, confidence, and motion smoothness for each landmark."""

    policy = policy or ReliabilityPolicy()
    frames = list(sequence)
    if len(frames) < policy.minimum_frames:
        return LandmarkReliability(frame_count=len(frames))

    torso = [length for length in (_torso_length(frame) for frame in frames) if length]
    reference = float(np.median(torso)) if torso else None

    landmark_ids = sorted({index for frame in frames for index in frame})
    presence: dict[int, float] = {}
    visibility: dict[int, float] = {}
    steps: dict[int, float] = {}
    admitted: set[int] = set()

    for index in landmark_ids:
        positions: list[np.ndarray | None] = []
        confidences: list[float] = []
        for frame in frames:
            point = frame.get(index)
            if point is None:
                positions.append(None)
                continue
            array = np.asarray(point, dtype=float)
            if array.size < 3 or not np.all(np.isfinite(array[:3])):
                positions.append(None)
                continue
            positions.append(array[:3])
            confidences.append(float(array[3]) if array.size > 3 else 1.0)

        seen = [item for item in positions if item is not None]
        presence[index] = len(seen) / len(frames)
        visibility[index] = float(np.median(confidences)) if confidences else 0.0

        deltas = [
            float(np.linalg.norm(current - previous))
            for previous, current in zip(positions, positions[1:], strict=False)
            if previous is not None and current is not None
        ]
        if not deltas or not reference:
            continue
        steps[index] = float(np.median(deltas)) / reference

        if (
            presence[index] >= policy.minimum_presence
            and visibility[index] >= policy.minimum_median_visibility
            and steps[index] <= policy.maximum_median_step
        ):
            admitted.add(index)

    return LandmarkReliability(
        presence=presence,
        median_visibility=visibility,
        median_step=steps,
        admitted=frozenset(admitted),
        frame_count=len(frames),
    )


def policy_from_config(config: dict[str, Any] | None) -> ReliabilityPolicy:
    """Build a policy from the ``quality.sustained_tracking`` config block."""

    defaults = ReliabilityPolicy()
    if not config:
        return defaults
    if not isinstance(config, dict):
        raise ValueError("quality.sustained_tracking must be a mapping")
    known = set(vars(defaults))
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(
            f"unknown sustained tracking settings: {unknown}; available: {sorted(known)}"
        )
    values: dict[str, Any] = {
        name: int(value) if name.endswith("_frames") else float(value)
        for name, value in config.items()
    }
    return replace(defaults, **values)
