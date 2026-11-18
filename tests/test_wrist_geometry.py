"""Wrist flexion, deviation and pronation from constructed hand geometry.

Postures are built so the correct answer follows from construction rather than
from anatomical intuition: a hand collinear with the forearm is a neutral wrist
whatever the arm's orientation, so deviation and flexion must both read zero.
"""

import numpy as np
import pytest

from core.angle_calculator import AngleCalculator

CALCULATOR = AngleCalculator()


def _rotate(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """Rodrigues rotation of `vector` about unit `axis`."""

    axis = axis / np.linalg.norm(axis)
    theta = np.radians(degrees)
    return (
        vector * np.cos(theta)
        + np.cross(axis, vector) * np.sin(theta)
        + axis * np.dot(axis, vector) * (1 - np.cos(theta))
    )


def _right_arm(pronation_deg: float = 0.0, deviation_deg: float = 0.0) -> dict:
    """Right arm with the forearm horizontal along +x, hand built around it."""

    shoulder = np.array([-0.20, 0.55, 0.0])
    elbow = np.array([-0.20, 0.20, 0.0])
    forearm_dir = np.array([1.0, 0.0, 0.0])
    wrist = elbow + 0.25 * forearm_dir

    radial_ulnar = _rotate(np.array([0.0, 0.0, 1.0]), forearm_dir, pronation_deg)
    hand_axis = _rotate(forearm_dir, np.cross(forearm_dir, radial_ulnar), deviation_deg)

    return {
        23: np.array([0.10, 0.0, 0.0]),
        24: np.array([-0.10, 0.0, 0.0]),
        11: np.array([0.20, 0.55, 0.0]),
        12: shoulder,
        14: elbow,
        16: wrist,
        20: wrist + 0.08 * hand_axis + 0.035 * radial_ulnar,  # RIGHT_INDEX, radial
        18: wrist + 0.08 * hand_axis - 0.035 * radial_ulnar,  # RIGHT_PINKY, ulnar
    }


def test_hand_collinear_with_forearm_is_a_neutral_wrist():
    angles = CALCULATOR.calculate_all_angles(_right_arm())
    assert angles.right_wrist_flexion == pytest.approx(0.0, abs=1e-6)
    assert angles.right_wrist_deviation == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("deviation", [5.0, 15.0, 30.0, -5.0, -15.0, -30.0])
def test_deviation_magnitude_matches_construction(deviation):
    angles = CALCULATOR.calculate_all_angles(_right_arm(deviation_deg=deviation))
    assert abs(angles.right_wrist_deviation) == pytest.approx(abs(deviation), abs=1e-6)


def test_deviation_is_signed_toward_the_ulnar_side():
    """Opposite deflections must not collapse onto the same value."""

    ulnar = CALCULATOR.calculate_all_angles(_right_arm(deviation_deg=20.0))
    radial = CALCULATOR.calculate_all_angles(_right_arm(deviation_deg=-20.0))
    assert ulnar.right_wrist_deviation == pytest.approx(-radial.right_wrist_deviation)
    assert np.sign(ulnar.right_wrist_deviation) != np.sign(radial.right_wrist_deviation)


def test_pronation_separates_opposite_rotations():
    """The previous implementation folded +45 and -45 onto one value."""

    supinated = CALCULATOR.calculate_all_angles(_right_arm(pronation_deg=-45.0))
    pronated = CALCULATOR.calculate_all_angles(_right_arm(pronation_deg=45.0))
    assert supinated.right_wrist_pronation != pytest.approx(pronated.right_wrist_pronation)


def test_pronation_tracks_rotation_about_the_forearm_axis():
    """Equal rotation steps produce equal angular steps, modulo the +-180 wrap."""

    measured = [
        CALCULATOR.calculate_all_angles(_right_arm(pronation_deg=p)).right_wrist_pronation
        for p in (-45.0, 0.0, 45.0)
    ]
    steps = [(b - a + 180.0) % 360.0 - 180.0 for a, b in zip(measured, measured[1:], strict=False)]
    assert steps[0] == pytest.approx(steps[1], abs=1e-6)
    assert abs(steps[0]) == pytest.approx(45.0, abs=1e-6)


def test_missing_hand_landmarks_do_not_invent_a_deviation():
    joints = _right_arm()
    del joints[18]
    del joints[20]
    angles = CALCULATOR.calculate_all_angles(joints)
    assert angles.right_wrist_deviation == 0.0
    assert angles.right_wrist_pronation == 0.0
    assert "right_wrist_flexion_deg" in angles.unavailable_measurements


def test_pronation_needs_the_upper_arm_and_is_zero_without_it():
    joints = _right_arm(pronation_deg=45.0)
    del joints[12]
    angles = CALCULATOR.calculate_all_angles(joints)
    assert angles.right_wrist_pronation == 0.0
