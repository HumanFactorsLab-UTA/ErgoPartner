"""Synthetic landmark geometry with analytically known angles."""

import numpy as np
import pytest

from core.angle_calculator import AngleCalculator
from core.coordinates import CANONICAL_COORDINATE_SYSTEM, mediapipe_world_to_canonical


def _point(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z, 1.0])


def _torso(
    trunk_vector: tuple[float, float, float] = (0.0, 1.0, 0.0),
    left_arm_vector: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> dict[int, np.ndarray]:
    left_hip = np.array([0.2, 0.0, 0.0])
    right_hip = np.array([-0.2, 0.0, 0.0])
    trunk = np.asarray(trunk_vector)
    left_shoulder = left_hip + trunk
    right_shoulder = right_hip + trunk
    return {
        23: _point(*left_hip),
        24: _point(*right_hip),
        11: _point(*left_shoulder),
        12: _point(*right_shoulder),
        13: _point(*(left_shoulder + np.asarray(left_arm_vector))),
        14: _point(*(right_shoulder + np.array([0.0, -1.0, 0.0]))),
    }


def _yaw(joints: dict[int, np.ndarray], degrees: float) -> dict[int, np.ndarray]:
    angle = np.radians(degrees)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    return {index: np.r_[rotation @ point[:3], point[3]] for index, point in joints.items()}


@pytest.mark.parametrize("yaw", [0.0, 30.0, 90.0, 180.0])
def test_trunk_flexion_is_signed_and_invariant_to_subject_yaw(yaw):
    joints = _yaw(_torso((0.0, 1.0, 0.5)), yaw)
    angles = AngleCalculator().calculate_all_angles(joints, side="left")
    assert angles.trunk_flexion == pytest.approx(np.degrees(np.arctan(0.5)))
    assert angles.trunk_side_bend == pytest.approx(0.0)


def test_trunk_extension_and_side_bend_are_separate():
    calculator = AngleCalculator()
    extension = calculator.calculate_all_angles(_torso((0.0, 1.0, -0.5)), side="left")
    side_bend = calculator.calculate_all_angles(_torso((-0.5, 1.0, 0.0)), side="left")
    expected = np.degrees(np.arctan(0.5))
    assert extension.trunk_flexion == pytest.approx(-expected)
    assert extension.trunk_side_bend == pytest.approx(0.0)
    assert side_bend.trunk_flexion == pytest.approx(0.0)
    assert side_bend.trunk_side_bend == pytest.approx(expected)


@pytest.mark.parametrize(
    ("arm_vector", "flexion", "abduction"),
    [
        ((0.0, -1.0, 0.0), 0.0, 0.0),
        ((0.0, -1.0, 1.0), 45.0, 0.0),
        ((0.0, -1.0, -1.0), -45.0, 0.0),
        ((1.0, -1.0, 0.0), 0.0, 45.0),
    ],
)
def test_upper_arm_flexion_extension_and_abduction_are_separate(arm_vector, flexion, abduction):
    angles = AngleCalculator().calculate_all_angles(_torso(left_arm_vector=arm_vector), side="left")
    assert angles.left_upper_arm_flexion == pytest.approx(flexion)
    assert angles.left_upper_arm_abduction == pytest.approx(abduction)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("yaw", [0, 70])
def test_lateral_excursion_separates_forward_overhead_and_cross_body_reaches(side, yaw):
    calculator = AngleCalculator(arm_abduction_method="lateral_excursion")
    shoulder, elbow = (11, 13) if side == "left" else (12, 14)
    outward = 1 if side == "left" else -1
    for vector, expected in [([0, 1, 1], 0), ([-outward, -1, 0], 0), ([outward, -1, 0], 45)]:
        joints = _torso()
        joints[elbow][:3] = joints[shoulder][:3] + vector
        result = calculator.calculate_all_angles(_yaw(joints, yaw))
        assert getattr(result, f"{side}_upper_arm_abduction") == pytest.approx(expected, abs=1e-8)


def test_lateral_excursion_needs_a_subject_frame():
    joints = _torso()
    del joints[23]
    angles = AngleCalculator(arm_abduction_method="lateral_excursion").calculate_all_angles(joints)
    assert angles.to_dict()["left_upper_arm_abduction"] is None


def test_shoulder_elevation_is_side_specific():
    joints = _torso()
    joints[12][1] += 0.1
    angles = AngleCalculator().calculate_all_angles(joints)
    assert angles.right_shoulder_elevation > 0
    assert angles.left_shoulder_elevation == 0


def test_world_coordinate_transform_is_centralized_and_explicit():
    transformed = mediapipe_world_to_canonical(1.0, 2.0, 3.0, 0.75)
    assert transformed == pytest.approx([1.0, -2.0, -3.0, 0.75])
    assert CANONICAL_COORDINATE_SYSTEM["z_positive"] == "toward camera"


def test_world_coordinate_transform_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        mediapipe_world_to_canonical(float("nan"), 0.0, 0.0)


def test_missing_geometry_serializes_as_null_not_zero():
    payload = AngleCalculator().calculate_all_angles({}).to_dict()
    assert payload["trunk_flexion"] is None
    assert payload["neck_rotation"] is None
    assert payload["left_wrist_flexion"] is None
    assert "trunk_flexion_deg" in payload["unavailable_measurements"]


@pytest.mark.parametrize("yaw", [0.0, 63.0, 180.0])
@pytest.mark.parametrize("sagittal", [-10.0, 0.0, 10.0])
def test_lateral_head_inclination_does_not_enter_sagittal_neck_score(yaw, sagittal):
    joints = _torso()
    lateral = np.radians(30)
    head = np.array([np.tan(lateral), 1.0, np.tan(np.radians(sagittal))]) * 0.2 + [0, 1, 0]
    # Rotate the ear/eye line in the frontal plane so the independent roll
    # measurement registers the same lateral tilt.
    for index, width in ((7, 0.08), (8, -0.08), (2, 0.06), (5, -0.06)):
        joints[index] = _point(*(head + width * np.array([np.cos(lateral), -np.sin(lateral), 0])))
    joints = _yaw(joints, yaw)
    projected = AngleCalculator(neck_flexion_method="sagittal_projection").calculate_all_angles(
        joints
    )
    assert projected.neck_flexion == pytest.approx(sagittal, abs=1e-8)
    assert projected.neck_side_bend == pytest.approx(30.0)
    historical = AngleCalculator(neck_flexion_method="legacy_cone").calculate_all_angles(joints)
    assert abs(historical.neck_flexion) >= 29.999999


def test_degenerate_sagittal_head_projection_is_unavailable():
    joints = _torso()
    joints.update({7: _point(0.2, 1, 0), 8: _point(0.1, 1, 0)})
    result = AngleCalculator(neck_flexion_method="sagittal_projection").calculate_all_angles(joints)
    assert result.to_dict()["neck_flexion"] is None
    assert result.to_dict()["neck_side_bend"] == pytest.approx(0.0)
