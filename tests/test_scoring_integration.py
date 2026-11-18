"""Pose-measurement to bilateral scores to machine-readable report integration."""

import json

from core.angle_calculator import BodyAngles
from ergonomics.adapters import build_reba_input, build_rula_input
from ergonomics.common import AssessmentContext
from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from reporting.writers import ResultWriter


def task_inputs():
    return {
        "load_kg": 7.0,
        "coupling": "fair",
        "shock_or_rapid_force": False,
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "rapid_large_posture_change_or_unstable_base": False,
        "seated": False,
        "legs_weight_bearing": "bilateral",
        "legs_supported_and_balanced": True,
        "trunk_twisted": False,
        "trunk_side_bent": False,
        "trunk_twisted_or_side_flexed": False,
        "neck_twisted": False,
        "neck_side_bent": False,
        "neck_twisted_or_side_flexed": False,
        "rula_load_type": "intermittent",
        **{
            f"{side}_{name}": False
            for side in ("left", "right")
            for name in (
                "upper_arm_abducted",
                "upper_arm_abducted_or_rotated",
                "shoulder_raised",
                "arm_supported_or_leaning",
                "arm_supported_or_gravity_assisted",
                "lower_arm_across_midline_or_out_to_side",
                "wrist_deviated",
                "wrist_twist_at_end_range",
                "wrist_deviated_or_twisted",
            )
        },
    }


def test_complete_bilateral_scoring_and_jsonl_output(tmp_path):
    angles = BodyAngles(
        trunk_flexion=10,
        neck_flexion=5,
        left_upper_arm_flexion=10,
        right_upper_arm_flexion=70,
        left_forearm_flexion=90,
        right_forearm_flexion=90,
        left_wrist_flexion=0,
        right_wrist_flexion=20,
        left_knee_flexion=0,
        right_knee_flexion=0,
        left_wrist_data_source="hand_mesh",
        right_wrist_data_source="hand_mesh",
    )
    angles.landmark_visibility = {index: 0.99 for index in range(33)}
    context = AssessmentContext(
        frame_number=12,
        timestamp_sec=0.4,
        person_identifier="person-0001",
        input_source="synthetic-keypoints",
        calibration_status="not_calibrated",
    )
    task = task_inputs()
    rula_calc = RULACalculator()
    reba_calc = REBACalculator()
    rula = rula_calc.calculate_bilateral(
        build_rula_input(angles, "left", task, context, visibility_threshold=0.5),
        build_rula_input(angles, "right", task, context, visibility_threshold=0.5),
    )
    reba = reba_calc.calculate_bilateral(
        build_reba_input(angles, "left", task, context, visibility_threshold=0.5),
        build_reba_input(angles, "right", task, context, visibility_threshold=0.5),
    )
    assert rula.selected_side == "right"
    assert reba.selected_side == "right"
    assert rula.selected.final_score is not None
    assert reba.selected.final_score is not None

    record = {
        "frame_number": 12,
        "timestamp_sec": 0.4,
        "person_identifier": "person-0001",
        "input_source": "synthetic-keypoints",
        "calibration_status": "not_calibrated",
        "pose_quality": {"mean_keypoint_visibility": 0.99},
        "rula": rula,
        "reba": reba,
    }
    with ResultWriter(tmp_path) as writer:
        writer.write_frame(record)
    payload = json.loads((tmp_path / "frame_results.jsonl").read_text(encoding="utf-8"))
    assert payload["rula"]["left"]["intermediate_scores"]
    assert payload["rula"]["right"]["raw_measurements"]
    assert payload["reba"]["selection_reason"]
    assert payload["person_identifier"] == "person-0001"
