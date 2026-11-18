import json
import math

import pytest

from dashboard.professional_analysis import analyze_professional_records


def frame_record(
    timestamp,
    frame,
    angles=None,
    *,
    selection=None,
    per_camera_angles=None,
    rula=None,
    reba=None,
):
    record = {
        "frame_number": frame,
        "timestamp_sec": timestamp,
        "person_identifier": "person-0001",
        "calibration_status": "calibrated",
        "timestamp_source": ["decoder_pts", "decoder_pts", "decoder_pts"],
        "pose_quality": {
            "mean_keypoint_visibility": 0.95,
            "camera_view_status": "suitable",
            "model_coordinate_type": "calibrated_multiview_3d",
        },
        "angles": angles or {},
        "multi_view_selection": selection or {},
        "pose_landmarks_normalized_by_camera": {"0": [], "1": [], "2": []},
    }
    if per_camera_angles is not None:
        record["per_camera_angles"] = per_camera_angles
    if rula is not None:
        record["rula"] = rula
    if reba is not None:
        record["reba"] = reba
    return record


def angle_result(result, measurement):
    return next(item for item in result.angle_analyses if item.measurement == measurement)


def band(result, measurement, method, band_id):
    analysis = angle_result(result, measurement)
    return next(
        item for item in analysis.band_exposure if item.method == method and item.band_id == band_id
    )


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [
        (-20.0001, "extension_lt_minus_20"),
        (-20.0, "minus_20_to_20"),
        (20.0, "minus_20_to_20"),
        (20.0001, "flexion_gt_20_to_45"),
        (45.0, "flexion_gt_20_to_45"),
        (45.0001, "flexion_gt_45_to_90"),
        (90.0, "flexion_gt_45_to_90"),
        (90.0001, "flexion_gt_90"),
    ],
)
def test_rula_upper_arm_exact_boundaries(value, expected_band):
    result = analyze_professional_records(
        [frame_record(0.0, 1, {"left_upper_arm_flexion": value})],
        repetition_measurements=(),
    )

    assert band(result, "left_upper_arm_flexion", "RULA", expected_band).observation_count == 1


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [
        (-0.0001, "extension_unsupported"),
        (0.0, "neutral_0"),
        (0.0001, "flexion_gt_0_to_20"),
        (20.0, "flexion_gt_0_to_20"),
        (20.0001, "flexion_gt_20_to_60"),
        (60.0, "flexion_gt_20_to_60"),
        (60.0001, "flexion_gt_60"),
    ],
)
def test_rula_trunk_exact_boundaries(value, expected_band):
    result = analyze_professional_records(
        [frame_record(0.0, 1, {"trunk_flexion": value})],
        repetition_measurements=(),
    )

    assert band(result, "trunk_flexion", "RULA", expected_band).observation_count == 1


@pytest.mark.parametrize(
    ("value", "expected_band"),
    [
        (-20.0001, "extension_lt_minus_20"),
        (-20.0, "extension_minus_20_to_lt_0"),
        (-0.0001, "extension_minus_20_to_lt_0"),
        (0.0, "neutral_0"),
        (20.0, "flexion_gt_0_to_20"),
        (20.0001, "flexion_gt_20_to_60"),
        (60.0, "flexion_gt_20_to_60"),
        (60.0001, "flexion_gt_60"),
    ],
)
def test_reba_trunk_exact_boundaries(value, expected_band):
    result = analyze_professional_records(
        [frame_record(0.0, 1, {"trunk_flexion": value})],
        repetition_measurements=(),
    )

    assert band(result, "trunk_flexion", "REBA", expected_band).observation_count == 1


@pytest.mark.parametrize(
    ("measurement", "value", "method", "expected_band"),
    [
        ("left_forearm_flexion", 59.9999, "RULA", "below_60"),
        ("left_forearm_flexion", 60.0, "RULA", "from_60_to_100"),
        ("left_forearm_flexion", 100.0, "REBA", "from_60_to_100"),
        ("left_forearm_flexion", 100.0001, "REBA", "above_100"),
        ("neck_flexion", 10.0, "RULA", "flexion_0_to_10"),
        ("neck_flexion", 10.0001, "RULA", "flexion_gt_10_to_20"),
        ("neck_flexion", 20.0, "REBA", "flexion_0_to_20"),
        ("neck_flexion", 20.0001, "REBA", "flexion_gt_20"),
        ("left_knee_flexion", 30.0, "REBA", "flexion_0_to_30"),
        ("left_knee_flexion", 30.0001, "REBA", "flexion_gt_30_to_60"),
        ("left_knee_flexion", 60.0, "REBA", "flexion_gt_30_to_60"),
        ("left_knee_flexion", 60.0001, "REBA", "flexion_gt_60"),
    ],
)
def test_other_official_method_boundaries(measurement, value, method, expected_band):
    result = analyze_professional_records(
        [frame_record(0.0, 1, {measurement: value})],
        repetition_measurements=(),
    )

    assert band(result, measurement, method, expected_band).observation_count == 1


def test_irregular_timestamps_drive_duration_and_sustained_bout():
    records = [
        frame_record(0.0, 0, {"trunk_flexion": 0.0}),
        frame_record(0.1, 1, {"trunk_flexion": 30.0}),
        frame_record(0.4, 2, {"trunk_flexion": 30.0}),
        frame_record(0.5, 3, {"trunk_flexion": 0.0}),
    ]

    result = analyze_professional_records(
        records,
        maximum_gap_sec=0.35,
        sustained_min_duration_sec=0.35,
        repetition_measurements=(),
    )
    exposure = band(result, "trunk_flexion", "RULA", "flexion_gt_20_to_60")
    analysis = angle_result(result, "trunk_flexion")

    assert exposure.duration_sec == pytest.approx(0.4)
    assert exposure.percent_of_covered_duration == pytest.approx(80.0)
    assert analysis.time_coverage.covered_duration_sec == pytest.approx(0.5)
    assert len(analysis.sustained_bouts) == 2  # one independently for RULA and REBA
    assert all(item.duration_sec == pytest.approx(0.4) for item in analysis.sustained_bouts)


def test_gaps_are_not_bridged_for_exposure_bouts_or_repetitions():
    records = [
        frame_record(0.0, 0, {"trunk_flexion": 0.0}),
        frame_record(0.1, 1, {"trunk_flexion": 30.0}),
        frame_record(2.0, 2, {"trunk_flexion": 30.0}),
        frame_record(2.1, 3, {"trunk_flexion": 0.0}),
    ]

    result = analyze_professional_records(
        records,
        maximum_gap_sec=0.2,
        sustained_min_duration_sec=0.15,
        repetition_measurements=("trunk_flexion",),
    )
    analysis = angle_result(result, "trunk_flexion")
    exposure = band(result, "trunk_flexion", "RULA", "flexion_gt_20_to_60")
    repetition = result.repetition_candidates[0]

    assert result.data_quality.time_coverage.gap_count == 1
    assert result.data_quality.time_coverage.gap_duration_sec == pytest.approx(1.9)
    assert exposure.duration_sec == pytest.approx(0.1)
    assert analysis.sustained_bouts == ()
    assert repetition.status == "unavailable"
    assert repetition.candidate_count is None


def test_pose_only_wrist_is_excluded_and_never_gets_a_normal_looking_result():
    records = [
        frame_record(
            0.0,
            0,
            {
                "left_wrist_flexion": 15.0,
                "left_wrist_data_source": "pose_only",
                "measurement_visibility": {"left_wrist_flexion_deg": 0.99},
            },
        )
    ]

    result = analyze_professional_records(records, repetition_measurements=())
    analysis = angle_result(result, "left_wrist_flexion")

    assert analysis.status == "unavailable"
    assert analysis.statistics.valid_observation_count == 0
    assert analysis.statistics.mean_deg is None
    assert analysis.band_exposure == ()
    assert any("unsupported_wrist_source:pose_only" in item for item in analysis.exclusion_reasons)


def test_validated_wrist_source_uses_exact_rula_and_reba_15_degree_boundaries():
    result = analyze_professional_records(
        [
            frame_record(
                0.0,
                0,
                {
                    "left_wrist_flexion": -15.0,
                    "left_wrist_data_source": "hand_landmarker",
                },
            )
        ],
        repetition_measurements=(),
    )

    assert band(result, "left_wrist_flexion", "RULA", "abs_gt_0_to_15").observation_count == 1
    assert band(result, "left_wrist_flexion", "REBA", "abs_le_15").observation_count == 1


def test_low_visibility_and_non_finite_values_are_explicitly_unavailable():
    records = [
        frame_record(
            0.0,
            0,
            {
                "trunk_flexion": 20.0,
                "neck_flexion": math.nan,
                "measurement_visibility": {"trunk_flexion_deg": 0.49},
            },
        )
    ]

    result = analyze_professional_records(records, repetition_measurements=())
    trunk = angle_result(result, "trunk_flexion")
    neck = angle_result(result, "neck_flexion")

    assert trunk.status == "unavailable"
    assert trunk.statistics.mean_deg is None
    assert "below_minimum_measurement_visibility (1)" in trunk.exclusion_reasons
    assert neck.status == "unavailable"
    assert neck.statistics.mean_deg is None


def test_candidate_repetitions_split_at_camera_source_changes():
    values = [0.0, 20.0, 0.0, 0.0, 20.0, 0.0]
    sources = ["front", "front", "front", "side", "side", "side"]
    records = [
        frame_record(
            index * 0.1,
            index,
            {"trunk_flexion": value},
            selection={"trunk_flexion": source},
        )
        for index, (value, source) in enumerate(zip(values, sources, strict=True))
    ]

    result = analyze_professional_records(
        records,
        maximum_gap_sec=0.2,
        repetition_measurements=("trunk_flexion",),
        repetition_prominence_deg=10.0,
        repetition_min_interval_sec=0.05,
    )
    repetition = result.repetition_candidates[0]

    assert repetition.status == "candidate_events_available"
    assert repetition.segments_analyzed == 2
    assert repetition.candidate_count == 2
    assert {item.camera_source for item in repetition.event_evidence} == {"front", "side"}


def test_camera_contribution_and_agreement_are_descriptive_not_scored():
    records = [
        frame_record(
            0.0,
            0,
            {"trunk_flexion": 15.0},
            selection={"trunk_flexion": "front"},
            per_camera_angles={
                "front": {"trunk_flexion": 10.0},
                "side": {"trunk_flexion": 20.0},
            },
        ),
        frame_record(
            0.1,
            1,
            {"trunk_flexion": 30.0},
            selection={"trunk_flexion": "side"},
            per_camera_angles={
                "front": {"trunk_flexion": 20.0},
                "side": {"trunk_flexion": 40.0},
            },
        ),
    ]

    result = analyze_professional_records(records, repetition_measurements=())
    camera = result.camera_sources

    assert camera.contribution_status == "available"
    assert camera.contributions is not None
    assert [item.observation_count for item in camera.contributions] == [1, 1]
    assert camera.agreement_status == "available"
    assert camera.agreement is not None
    agreement = camera.agreement[0]
    assert agreement.mean_range_deg == pytest.approx(15.0)
    assert agreement.percentile_95_range_deg == pytest.approx(19.5)
    assert agreement.maximum_range_deg == pytest.approx(20.0)
    assert "threshold" in agreement.interpretation


def test_method_readiness_remains_separate_and_never_synthesizes_a_score():
    rula = {
        "selected": None,
        "left": {
            "validity_status": "missing_required_task_input",
            "missing_inputs": ["load_kg"],
            "invalid_inputs": [],
        },
        "right": {
            "validity_status": "missing_required_task_input",
            "missing_inputs": ["load_kg"],
            "invalid_inputs": [],
        },
    }
    reba = {
        "selected": {"final_score": 5},
        "left": {"validity_status": "valid", "missing_inputs": [], "invalid_inputs": []},
        "right": {"validity_status": "valid", "missing_inputs": [], "invalid_inputs": []},
    }
    rnle = {"status": "missing_required_task_input", "missing_inputs": ["load_weight_kg"]}
    result = analyze_professional_records(
        [frame_record(0.0, 0, {"trunk_flexion": 10.0}, rula=rula, reba=reba)],
        rnle_result=rnle,
        repetition_measurements=(),
    )
    readiness = {item.method: item for item in result.method_readiness}

    assert readiness["RULA"].status == "missing_required_inputs"
    assert readiness["RULA"].missing_inputs == ("load_kg",)
    assert readiness["REBA"].status == "ready"
    assert readiness["REBA"].valid_result_count == 1
    assert readiness["RNLE"].status == "missing_required_inputs"
    assert readiness["RNLE"].missing_inputs == ("load_weight_kg",)
    assert any("never combined" in warning for warning in result.warnings)


def test_empty_input_preserves_unavailable_values_as_none_and_serializes():
    result = analyze_professional_records([], repetition_measurements=("trunk_flexion",))
    payload = result.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert result.data_quality.status == "insufficient_data"
    assert result.data_quality.usable_angle_observation_count is None
    assert result.data_quality.mean_pose_visibility is None
    assert result.data_quality.time_coverage.covered_duration_sec is None
    assert result.repetition_candidates[0].candidate_count is None
    assert all(item.valid_result_count is None for item in result.method_readiness)
    assert result.camera_sources.contributions is None
    assert result.camera_sources.agreement is None
    assert '"Low"' not in encoded


def test_output_is_deterministic_and_recommendations_link_to_observed_evidence():
    records = [
        frame_record(0.0, 0, {"trunk_flexion": 30.0}),
        frame_record(0.1, 1, {"trunk_flexion": 30.0}),
    ]

    first = analyze_professional_records(records, repetition_measurements=()).to_dict()
    second = analyze_professional_records(iter(records), repetition_measurements=()).to_dict()

    assert first == second
    posture_recommendations = [
        item for item in first["recommendations"] if item["recommendation_id"] == "review-trunk"
    ]
    assert len(posture_recommendations) == 1
    assert posture_recommendations[0]["evidence"][0]["frame_number"] in {0, 1}
    assert posture_recommendations[0]["source_urls"]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("minimum_visibility", -0.1),
        ("minimum_visibility", 1.1),
        ("maximum_gap_sec", 0.0),
        ("sustained_min_duration_sec", math.inf),
        ("repetition_prominence_deg", 0.0),
        ("repetition_min_interval_sec", -1.0),
    ],
)
def test_configuration_validation(argument, value):
    with pytest.raises(ValueError):
        analyze_professional_records([], **{argument: value})
