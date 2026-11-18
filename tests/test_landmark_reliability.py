"""Admitting a landmark on sustained tracking rather than per-frame confidence."""

from __future__ import annotations

import numpy as np
import pytest

from core.landmark_reliability import (
    LandmarkReliability,
    ReliabilityPolicy,
    assess_landmarks,
    policy_from_config,
)

KNEE = 25
TORSO_LANDMARKS = {11: (-0.2, 1.4), 12: (0.2, 1.4), 23: (-0.1, 0.9), 24: (0.1, 0.9)}


def _frame(
    knee: tuple[float, float, float] | None,
    visibility: float,
) -> dict[int, np.ndarray]:
    frame = {index: np.array([x, y, 0.0, 0.99]) for index, (x, y) in TORSO_LANDMARKS.items()}
    if knee is not None:
        frame[KNEE] = np.array([*knee, visibility])
    return frame


def _sequence(
    count: int = 60,
    *,
    visibility: float = 0.45,
    drop_every: int | None = None,
    jitter: float = 0.0,
) -> list[dict[int, np.ndarray]]:
    frames = []
    for index in range(count):
        missing = drop_every is not None and index % drop_every == 0
        offset = jitter if index % 2 else -jitter
        frames.append(_frame(None if missing else (-0.1 + offset, 0.5, 0.0), visibility))
    return frames


def test_a_continuously_tracked_landmark_is_admitted_below_the_gate():
    """The defect this fixes: a knee at 0.45 for a whole trial, plainly visible."""

    reliability = assess_landmarks(_sequence(visibility=0.45))

    assert KNEE in reliability.admitted
    assert reliability.presence[KNEE] == pytest.approx(1.0)
    assert reliability.median_visibility[KNEE] == pytest.approx(0.45)


def test_a_flickering_landmark_is_not_admitted():
    """Sustained tracking means present nearly always, not most of the time."""

    reliability = assess_landmarks(_sequence(drop_every=4))

    assert KNEE not in reliability.admitted
    assert reliability.presence[KNEE] < 0.9


def test_a_teleporting_landmark_is_not_admitted():
    """A smooth trajectory is the evidence; a jumping one has none."""

    reliability = assess_landmarks(_sequence(jitter=0.5))

    assert KNEE not in reliability.admitted
    assert reliability.median_step[KNEE] > ReliabilityPolicy().maximum_median_step


def test_a_landmark_the_model_has_no_confidence_in_is_not_admitted():
    """A smooth trajectory of hallucinations is still a hallucination."""

    reliability = assess_landmarks(_sequence(visibility=0.05))

    assert KNEE not in reliability.admitted


def test_a_short_sequence_admits_nothing():
    """Neither presence nor smoothness can be established from a few frames."""

    reliability = assess_landmarks(_sequence(count=5))

    assert reliability.admitted == frozenset()
    assert reliability.frame_count == 5


def test_an_empty_sequence_is_handled():
    reliability = assess_landmarks([])

    assert reliability.admitted == frozenset()
    assert reliability.frame_count == 0


def test_non_finite_positions_count_as_absent():
    frames = _sequence()
    for frame in frames[:20]:
        frame[KNEE] = np.array([np.nan, 0.5, 0.0, 0.9])

    reliability = assess_landmarks(frames)

    assert reliability.presence[KNEE] < 0.9
    assert KNEE not in reliability.admitted


def test_admission_is_described_for_the_record():
    described = assess_landmarks(_sequence()).describe()

    assert KNEE in described["admitted_landmark_ids"]
    assert "moved smoothly" in described["basis"]
    assert described["frame_count"] == 60


def test_policy_thresholds_govern_admission():
    frames = _sequence(visibility=0.35)

    assert KNEE in assess_landmarks(frames).admitted
    strict = ReliabilityPolicy(minimum_median_visibility=0.60)
    assert KNEE not in assess_landmarks(frames, strict).admitted


def test_policy_from_config_overrides_named_fields():
    policy = policy_from_config({"minimum_presence": 0.75})

    assert policy.minimum_presence == 0.75
    assert policy.maximum_median_step == ReliabilityPolicy().maximum_median_step

    with pytest.raises(ValueError, match="unknown sustained tracking settings"):
        policy_from_config({"minimum_presence_fraction": 0.75})


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        ReliabilityPolicy(minimum_presence=0.0)
    with pytest.raises(ValueError):
        ReliabilityPolicy(minimum_median_visibility=1.5)
    with pytest.raises(ValueError):
        ReliabilityPolicy(maximum_median_step=0.0)


def test_default_reliability_admits_nothing():
    """A processor that has not analysed a sequence must not loosen the gate."""

    assert LandmarkReliability().admitted == frozenset()
