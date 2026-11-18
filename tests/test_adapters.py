"""Regression tests for pose-to-method validation boundaries."""

from dataclasses import dataclass

import numpy as np
import pytest

from core.angle_calculator import AngleCalculator, BodyAngles
from core.tracking import CentroidPoseTracker
from ergonomics.adapters import build_rula_input
from ergonomics.common import AssessmentContext
from ergonomics.rula import RULACalculator


def complete_rula_task():
    return {
        "left_upper_arm_abducted": False,
        "left_shoulder_raised": False,
        "left_arm_supported_or_leaning": False,
        "left_lower_arm_across_midline_or_out_to_side": False,
        "left_wrist_deviated": False,
        "left_wrist_twist_at_end_range": False,
        "neck_twisted": False,
        "neck_side_bent": False,
        "trunk_twisted": False,
        "trunk_side_bent": False,
        "legs_supported_and_balanced": True,
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "load_kg": 0.0,
        "rula_load_type": "intermittent",
        "shock_or_rapid_force": False,
    }


def visible_angles() -> BodyAngles:
    angles = BodyAngles(
        wrist_data_source="hand_mesh",
        left_wrist_data_source="hand_mesh",
        right_wrist_data_source="hand_mesh",
    )
    angles.landmark_visibility = {index: 0.99 for index in range(33)}
    return angles


def test_low_visibility_becomes_insufficient_pose_not_neutral_score():
    angles = visible_angles()
    angles.landmark_visibility[13] = 0.1
    inp = build_rula_input(
        angles,
        "left",
        complete_rula_task(),
        AssessmentContext(),
        visibility_threshold=0.5,
    )
    result = RULACalculator().calculate(inp)
    assert result.final_score is None
    assert result.validity_status == "insufficient_pose_quality"
    assert "upper_arm_flexion_deg" in result.missing_inputs


def test_pose_only_wrist_is_rejected_unless_explicitly_allowed():
    angles = visible_angles()
    angles.wrist_data_source = "pose_only"
    angles.left_wrist_data_source = "pose_only"
    inp = build_rula_input(
        angles,
        "left",
        complete_rula_task(),
        AssessmentContext(),
        visibility_threshold=0.5,
    )
    result = RULACalculator().calculate(inp)
    assert result.final_score is None
    assert result.validity_status == "insufficient_pose_quality"
    assert "wrist_flexion_deg" in result.missing_inputs


def test_complete_pose_derived_score_is_valid_with_limitations():
    angles = visible_angles()
    inp = build_rula_input(
        angles,
        "left",
        complete_rula_task(),
        AssessmentContext(pose_quality={"camera_view_status": "suitable", "mirrored": False}),
        visibility_threshold=0.5,
    )
    result = RULACalculator().calculate(inp)
    assert result.final_score is not None
    assert result.validity_status == "valid_with_limitations"
    assert any("monocular" in warning.lower() for warning in result.warnings)


@pytest.mark.parametrize(
    ("pose_quality", "warning_fragment"),
    [
        ({"camera_view_status": "unsupported", "mirrored": False}, "camera view"),
        ({"camera_view_status": "suitable", "mirrored": True}, "mirrored"),
    ],
)
def test_unsupported_camera_conditions_refuse_numeric_score(pose_quality, warning_fragment):
    angles = visible_angles()
    inp = build_rula_input(
        angles,
        "left",
        complete_rula_task(),
        AssessmentContext(pose_quality=pose_quality),
        visibility_threshold=0.5,
    )
    result = RULACalculator().calculate(inp)
    assert result.final_score is None
    assert result.validity_status == "unsupported_camera_view"
    assert any(warning_fragment in warning.lower() for warning in result.warnings)


def test_hand_and_pose_origins_are_not_subtracted():
    calculator = AngleCalculator()
    hand = {
        0: np.array([0.0, 0.0, 0.0]),
        5: np.array([-0.03, 0.10, 0.0]),
        9: np.array([0.0, 0.11, 0.0]),
        13: np.array([0.02, 0.105, 0.0]),
        17: np.array([0.04, 0.09, 0.0]),
    }
    translated = {index: point + np.array([12.0, -7.0, 3.0]) for index, point in hand.items()}
    forearm_vector = np.array([0.0, 0.25, 0.0])
    first = calculator._wrist_angles_from_hand_mesh(hand, forearm_vector)
    second = calculator._wrist_angles_from_hand_mesh(translated, forearm_vector)
    assert second == pytest.approx(first)


def test_partial_hand_mesh_is_not_mislabeled_as_valid_hand_measurement():
    joints = {
        13: np.array([-0.2, 1.2, 0.0, 0.99]),
        15: np.array([-0.2, 0.9, 0.0, 0.99]),
        17: np.array([-0.23, 0.82, 0.0, 0.99]),
        19: np.array([-0.20, 0.80, 0.0, 0.99]),
    }
    malformed_hand = {"left": {0: np.zeros(3), 9: np.array([0.0, 0.1, 0.0])}}
    result = AngleCalculator().calculate_all_angles(joints, hand_landmarks=malformed_hand)
    assert result.left_wrist_data_source == "pose_only"


def test_valid_hand_mesh_supplies_wrist_when_pose_finger_points_are_missing():
    joints = {
        13: np.array([-0.2, 1.2, 0.0, 0.99]),
        15: np.array([-0.2, 0.9, 0.0, 0.99]),
    }
    hand = {
        "left": {
            0: np.array([0.0, 0.0, 0.0]),
            5: np.array([-0.03, 0.10, 0.01]),
            9: np.array([0.0, 0.11, 0.02]),
            13: np.array([0.02, 0.105, 0.01]),
            17: np.array([0.04, 0.09, -0.01]),
        }
    }
    result = AngleCalculator().calculate_all_angles(joints, hand_landmarks=hand)
    assert result.left_wrist_data_source == "hand_mesh"
    assert "left_wrist_flexion_deg" not in result.unavailable_measurements


@dataclass
class FakePose:
    normalized: dict[int, np.ndarray]


def pose_at(x: float, y: float = 0.5) -> FakePose:
    return FakePose(
        {
            11: np.array([x - 0.02, y, 0, 1]),
            12: np.array([x + 0.02, y, 0, 1]),
            23: np.array([x - 0.02, y + 0.2, 0, 1]),
            24: np.array([x + 0.02, y + 0.2, 0, 1]),
        }
    )


def test_tracker_keeps_identity_when_detector_order_changes():
    tracker = CentroidPoseTracker(max_distance=0.25)
    first = tracker.update([pose_at(0.2), pose_at(0.8)])
    second = tracker.update([pose_at(0.79), pose_at(0.21)])
    assert [tracked.person_id for tracked in first] == ["person-0001", "person-0002"]
    assert [tracked.person_id for tracked in second] == ["person-0002", "person-0001"]


def test_mesh_wrist_uses_forearm_and_rula_does_not_require_knee_geometry():
    angles = visible_angles()
    for index in [17, 19, 25, 27]:
        angles.landmark_visibility.pop(index)
    item = build_rula_input(
        angles, "left", complete_rula_task(), AssessmentContext(), visibility_threshold=0.5
    )
    result = RULACalculator().calculate(item)
    assert result.validity_status == "valid_with_limitations"
    assert result.final_score is not None
    assert "knee_flexion_deg" not in item.context.pose_quality["minimum_visibility_by_measurement"]
