"""Temporal landmark recovery fills short gaps and refuses implausible ones."""

import numpy as np
import pytest

from core.pose_temporal import PoseFrame, RecoveryPolicy, recover_landmarks

KNEE, HIP = 25, 23
POLICY = RecoveryPolicy()


def _sequence(knee_visibilities, knee_positions=None, fps=30.0, hip_positions=None):
    """Straight-line knee motion; the hip stays put so segment length is checkable."""

    frames = []
    for index, visibility in enumerate(knee_visibilities):
        if knee_positions is not None:
            position = np.asarray(knee_positions[index], dtype=float)
        else:
            position = np.array([0.0, -0.45 + 0.01 * index, 0.0])
        hip = np.asarray(
            hip_positions[index] if hip_positions is not None else [0.0, 0.0, 0.0], dtype=float
        )
        frames.append(
            PoseFrame(
                frame_number=index,
                timestamp_sec=index / fps,
                landmarks={
                    HIP: np.array([*hip, 0.99]),
                    KNEE: np.array([*position, visibility]),
                },
            )
        )
    return frames


def test_short_dip_below_the_gate_is_recovered():
    frames = _sequence([0.99, 0.99, 0.20, 0.99, 0.99])
    recovered, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 1
    assert recovered[2].landmarks[KNEE][3] >= POLICY.missing_visibility
    # Linear in time between the bracketing anchors.
    assert recovered[2].landmarks[KNEE][1] == pytest.approx(-0.45 + 0.01 * 2, abs=1e-9)


def test_confident_observations_are_never_altered():
    frames = _sequence([0.99] * 5)
    recovered, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 0
    for original, result in zip(frames, recovered, strict=True):
        assert np.allclose(original.landmarks[KNEE], result.landmarks[KNEE])


def test_hysteresis_band_is_left_as_measured():
    """Between missing and anchor thresholds the observation stands."""

    frames = _sequence([0.99, 0.99, 0.55, 0.99, 0.99])
    recovered, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 0
    assert recovered[2].landmarks[KNEE][3] == pytest.approx(0.55)


def test_gap_longer_than_the_limit_is_refused():
    policy = RecoveryPolicy(max_gap_sec=0.10)
    frames = _sequence([0.99] + [0.10] * 8 + [0.99])
    recovered, report = recover_landmarks(frames, policy)
    assert report.recovered_count == 0
    assert report.rejected_gap_too_long > 0
    assert recovered[4].landmarks[KNEE][3] == pytest.approx(0.10)


def test_unbracketed_gap_at_the_start_is_refused():
    """Nothing before the gap means there is nothing to interpolate from."""

    frames = _sequence([0.10, 0.10, 0.99, 0.99, 0.99])
    _, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 0
    assert report.rejected_unbracketed == 2


def test_implausible_speed_is_refused():
    positions = [
        [0.0, -0.45, 0.0],
        [0.0, -0.45, 0.0],
        [0.0, -0.45, 0.0],
        [9.0, -0.45, 0.0],  # teleport between the bracketing anchors
        [9.0, -0.45, 0.0],
    ]
    frames = _sequence([0.99, 0.99, 0.10, 0.99, 0.99], positions)
    _, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 0
    assert report.rejected_speed == 1


def test_segment_stretch_is_refused():
    """A reconstruction that breaks the thigh is not anatomically admissible.

    The hip moves during the gap while the bracketing knee observations do not,
    so interpolating the knee alone would place it far too close to the hip.
    Sampled at 10 fps so the speed limit is not what triggers the rejection.
    """

    knee = [[0.0, -0.45, 0.0]] * 5
    hip = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -0.40, 0.0],  # hip drops during the gap frame only
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    frames = _sequence([0.99, 0.99, 0.10, 0.99, 0.99], knee, fps=10.0, hip_positions=hip)
    _, report = recover_landmarks(frames, POLICY)
    assert report.rejected_speed == 0, "speed must not be the reason for rejection here"
    assert report.recovered_count == 0
    assert report.rejected_segment_strain == 1


def test_recovered_confidence_decays_with_gap_length():
    short = _sequence([0.99, 0.99, 0.10, 0.99, 0.99])
    long_gap = _sequence([0.99, 0.99, 0.10, 0.10, 0.10, 0.99, 0.99])
    short_result, _ = recover_landmarks(short, POLICY)
    long_result, _ = recover_landmarks(long_gap, POLICY)
    assert long_result[3].landmarks[KNEE][3] < short_result[2].landmarks[KNEE][3]


def test_report_identifies_the_affected_frames():
    frames = _sequence([0.99, 0.99, 0.20, 0.99, 0.99])
    _, report = recover_landmarks(frames, POLICY)
    assert report.by_frame() == {2: [KNEE]}
    payload = report.to_dict()
    assert payload["recovered_landmark_count"] == 1
    assert payload["method"] == "temporal_linear_interpolation"


def test_input_frames_are_not_mutated():
    frames = _sequence([0.99, 0.99, 0.20, 0.99, 0.99])
    before = frames[2].landmarks[KNEE].copy()
    recover_landmarks(frames, POLICY)
    assert np.allclose(frames[2].landmarks[KNEE], before)


def test_too_few_frames_is_a_no_op():
    frames = _sequence([0.99, 0.10])
    recovered, report = recover_landmarks(frames, POLICY)
    assert report.recovered_count == 0
    assert len(recovered) == 2


def test_policy_rejects_inconsistent_thresholds():
    with pytest.raises(ValueError):
        RecoveryPolicy(anchor_visibility=0.3, missing_visibility=0.8)
