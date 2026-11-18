"""Derivation of RULA/REBA task conditions from measured angles."""

from __future__ import annotations

import pytest

from core.angle_calculator import BodyAngles
from core.posture_conditions import (
    DERIVABLE_CONDITIONS,
    BaselinePolicy,
    ConditionThresholds,
    PostureBaseline,
    PostureConditionDeriver,
    apply_condition_smoothing,
    baseline_policy_from_config,
    derive_posture_conditions,
    smooth_series,
    thresholds_from_config,
)


def _angles(**overrides) -> BodyAngles:
    """A fully measurable neutral posture, adjusted by the given fields."""

    angles = BodyAngles()
    for name, value in overrides.items():
        setattr(angles, name, value)
    return angles


#: Conditions listed as derivable but left unknown on a neutral posture. Wrist
#: twist is measured without an anatomical zero, so no threshold on it is
#: meaningful, and REBA's combined wrist field inherits that gap.
NOT_DERIVED = {
    "left_wrist_twist_at_end_range",
    "right_wrist_twist_at_end_range",
    "left_wrist_deviated_or_twisted",
    "right_wrist_deviated_or_twisted",
}


def test_neutral_posture_derives_every_condition_as_false():
    result = derive_posture_conditions(_angles())

    assert set(result.values) == DERIVABLE_CONDITIONS
    assert set(result.unavailable) == NOT_DERIVED
    assert all(value is False for name, value in result.values.items() if name not in NOT_DERIVED)


def test_wrist_twist_is_never_derived_whatever_the_pronation():
    """Pronation sweeps the full circle with no neutral, so it cannot be banded."""

    for pronation in (0.0, 45.0, 90.0, 179.0, -179.0):
        result = derive_posture_conditions(_angles(left_wrist_pronation=pronation))
        assert result.values["left_wrist_twist_at_end_range"] is None
        assert "not derived" in result.evidence["left_wrist_twist_at_end_range"].measurement


def test_trunk_twist_beyond_threshold_sets_the_condition():
    result = derive_posture_conditions(_angles(trunk_twist=25.0))

    assert result.values["trunk_twisted"] is True
    assert result.values["trunk_side_bent"] is False
    assert result.values["trunk_twisted_or_side_flexed"] is True


def test_threshold_is_inclusive_and_margin_is_reported():
    limits = ConditionThresholds()
    result = derive_posture_conditions(_angles(trunk_twist=limits.trunk_twist_deg))

    assert result.values["trunk_twisted"] is True
    evidence = result.evidence["trunk_twisted"]
    assert evidence.measured_deg == pytest.approx(limits.trunk_twist_deg)
    assert evidence.margin_deg == pytest.approx(0.0)
    assert evidence.measurement == "trunk_twist"


def test_sign_of_the_angle_does_not_change_the_condition():
    """Side bend and deviation are penalised in either direction."""

    left = derive_posture_conditions(_angles(trunk_side_bend=-18.0))
    right = derive_posture_conditions(_angles(trunk_side_bend=18.0))

    assert left.values["trunk_side_bent"] is True
    assert right.values["trunk_side_bent"] is True


def test_unmeasurable_angle_yields_unknown_not_false():
    """A missing measurement must never read as a verified-absent penalty."""

    angles = _angles(unavailable_measurements=["trunk_twist_deg"])
    result = derive_posture_conditions(angles)

    assert result.values["trunk_twisted"] is None
    assert "trunk_twisted" in result.unavailable
    # Side bend was still measurable, so it stays a real answer.
    assert result.values["trunk_side_bent"] is False


def test_combined_condition_is_true_when_one_half_is_true_despite_unknown():
    angles = _angles(trunk_side_bend=30.0, unavailable_measurements=["trunk_twist_deg"])
    result = derive_posture_conditions(angles)

    assert result.values["trunk_twisted"] is None
    assert result.values["trunk_side_bent"] is True
    # Twist is unknown, but side flexion alone already earns the penalty.
    assert result.values["trunk_twisted_or_side_flexed"] is True


def test_combined_condition_stays_unknown_when_the_unknown_half_could_trigger_it():
    angles = _angles(trunk_side_bend=0.0, unavailable_measurements=["trunk_twist_deg"])
    result = derive_posture_conditions(angles)

    assert result.values["trunk_twisted_or_side_flexed"] is None


def test_non_finite_angle_is_treated_as_unmeasured():
    result = derive_posture_conditions(_angles(trunk_twist=float("nan")))

    assert result.values["trunk_twisted"] is None


def test_sides_are_derived_independently():
    result = derive_posture_conditions(
        _angles(left_upper_arm_abduction=45.0, right_upper_arm_abduction=3.0)
    )

    assert result.values["left_upper_arm_abducted"] is True
    assert result.values["right_upper_arm_abducted"] is False


def test_midline_cross_passes_through_as_a_boolean_measurement():
    result = derive_posture_conditions(_angles(right_forearm_midline_cross=True))

    assert result.values["right_lower_arm_across_midline_or_out_to_side"] is True
    assert result.values["left_lower_arm_across_midline_or_out_to_side"] is False


def test_unmeasurable_midline_cross_is_unknown():
    angles = _angles(unavailable_measurements=["right_forearm_midline_cross"])
    result = derive_posture_conditions(angles)

    assert result.values["right_lower_arm_across_midline_or_out_to_side"] is None


def test_wrist_combined_condition_follows_deviation_alone():
    """Twist is unknown, so deviation is what can carry the combined penalty."""

    deviated = derive_posture_conditions(_angles(left_wrist_deviation=20.0))
    neutral = derive_posture_conditions(_angles(left_wrist_deviation=0.0))

    assert deviated.values["left_wrist_deviated_or_twisted"] is True
    # A straight wrist does not clear the penalty, because the twist half of the
    # condition was never measured and could still trigger it.
    assert neutral.values["left_wrist_deviated_or_twisted"] is None


def test_known_weak_measurements_carry_their_limitation():
    result = derive_posture_conditions(_angles())

    assert result.evidence["left_shoulder_raised"].limitation is not None
    assert "bilateral" in result.evidence["left_shoulder_raised"].limitation
    assert result.evidence["right_upper_arm_abducted_or_rotated"].limitation is not None
    # A well-supported condition carries no caveat.
    assert result.evidence["trunk_twisted"].limitation is None


def test_available_excludes_unknown_conditions():
    angles = _angles(unavailable_measurements=["trunk_twist_deg"])
    result = derive_posture_conditions(angles)

    assert "trunk_twisted" not in result.available
    assert "trunk_side_bent" in result.available


def test_custom_thresholds_change_the_outcome():
    strict = PostureConditionDeriver(ConditionThresholds(trunk_twist_deg=5.0))
    lenient = PostureConditionDeriver(ConditionThresholds(trunk_twist_deg=40.0))
    angles = _angles(trunk_twist=20.0)

    assert strict.derive(angles).values["trunk_twisted"] is True
    assert lenient.derive(angles).values["trunk_twisted"] is False


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        ConditionThresholds(trunk_twist_deg=-1.0)


def test_thresholds_from_config_overrides_only_named_fields():
    limits = thresholds_from_config({"thresholds": {"trunk_twist_deg": 22.0}})

    assert limits.trunk_twist_deg == 22.0
    assert limits.wrist_deviation_deg == ConditionThresholds().wrist_deviation_deg


def test_thresholds_from_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown posture condition thresholds"):
        thresholds_from_config({"thresholds": {"trunk_twist_degrees": 22.0}})


def test_thresholds_from_config_defaults_when_absent():
    assert thresholds_from_config(None) == ConditionThresholds()


def test_evidence_serialises_for_the_result_record():
    result = derive_posture_conditions(_angles(trunk_twist=30.0))
    payload = result.to_dict()

    assert payload["values"]["trunk_twisted"] is True
    assert payload["evidence"]["trunk_twisted"]["measured_deg"] == pytest.approx(30.0)
    assert payload["evidence"]["trunk_twisted"]["threshold_deg"] == pytest.approx(
        ConditionThresholds().trunk_twist_deg
    )


# ----------------------------------------------------------------------
# Per-subject baseline
# ----------------------------------------------------------------------


def test_rectified_noise_floor_does_not_trip_a_condition():
    """The defect this fixes: a resting level above the absolute threshold.

    trunk_twist comes from an arccos and cannot go below zero, so its noise
    accumulates upward. On real footage it sat at 14 degrees with the subject
    plainly upright, above the old 15 degree threshold on half the frames.
    """

    resting = [12.0, 14.0, 16.0, 13.0, 15.0] * 10
    baseline = PostureBaseline.from_series({"trunk_twist": resting})
    deriver = PostureConditionDeriver(baseline=baseline)

    # A frame at the subject's own resting level is not a twisted trunk.
    assert deriver.derive(_angles(trunk_twist=15.0)).values["trunk_twisted"] is False
    # A genuine twist well past that level still registers.
    assert deriver.derive(_angles(trunk_twist=45.0)).values["trunk_twisted"] is True


def test_baseline_records_the_reference_it_used():
    baseline = PostureBaseline.from_series({"trunk_twist": [10.0] * 30})

    assert baseline.established is True
    assert baseline.neutral("trunk_twist") == pytest.approx(10.0)
    described = baseline.describe()
    assert described["frame_count"] == 30
    assert "held for the whole recording" in described["note"]


def test_a_short_series_establishes_no_baseline():
    """Too few frames cannot describe a resting level, so none is claimed."""

    baseline = PostureBaseline.from_series({"trunk_twist": [10.0, 11.0, 12.0]})

    assert baseline.established is False
    assert baseline.neutral("trunk_twist") == 0.0


def test_rectified_and_signed_measurements_use_different_references():
    """A rectified quantity takes a low percentile; a signed one takes the median."""

    rising = [float(value) for value in range(0, 100)]
    baseline = PostureBaseline.from_series({"trunk_twist": rising, "left_wrist_deviation": rising})

    assert baseline.neutral("trunk_twist") == pytest.approx(19.8, abs=1.0)
    assert baseline.neutral("left_wrist_deviation") == pytest.approx(49.5, abs=1.0)


def test_evidence_reports_the_departure_from_neutral():
    baseline = PostureBaseline.from_series({"trunk_twist": [10.0] * 30})
    result = PostureConditionDeriver(baseline=baseline).derive(_angles(trunk_twist=38.0))
    evidence = result.evidence["trunk_twisted"]

    assert evidence.measured_deg == pytest.approx(38.0)
    assert evidence.neutral_deg == pytest.approx(10.0)
    assert evidence.departure_deg == pytest.approx(28.0)


def test_without_a_baseline_the_measurement_is_used_as_is():
    result = derive_posture_conditions(_angles(trunk_twist=38.0))

    assert result.evidence["trunk_twisted"].neutral_deg == 0.0
    assert result.values["trunk_twisted"] is True


def test_return_toward_neutral_does_not_create_a_rectified_penalty():
    baseline = PostureBaseline.from_series({"trunk_twist": [30.0] * 25})
    evidence = (
        PostureConditionDeriver(baseline=baseline)
        .derive(_angles(trunk_twist=1))
        .evidence["trunk_twisted"]
    )
    assert evidence.value is False
    assert evidence.departure_deg == 0
    assert evidence.margin_deg < 0


def test_reviewed_signed_wrist_reference_preserves_opposite_direction():
    reference = {
        "source": "reviewed_neutral",
        "reference_id": "neutral trial A",
        "offsets_deg": {"left_wrist_deviation": 12.0},
    }
    baseline = PostureBaseline.for_recording({}, BaselinePolicy(), {"neutral_reference": reference})
    evidence = (
        PostureConditionDeriver(baseline=baseline)
        .derive(_angles(left_wrist_deviation=-12))
        .evidence["left_wrist_deviated"]
    )
    assert evidence.departure_deg == 24
    assert evidence.value is True
    assert baseline.describe()["reference_id"] == "neutral trial A"


def test_absolute_reference_preserves_sustained_trunk_rotation():
    series = {"trunk_twist": [40.0] * 100}
    baseline = PostureBaseline.for_recording(
        series, BaselinePolicy(), {"neutral_reference": {"source": "absolute"}}
    )
    assert baseline.source == "absolute"
    assert (
        PostureConditionDeriver(baseline=baseline)
        .derive(_angles(trunk_twist=40))
        .values["trunk_twisted"]
    )


@pytest.mark.parametrize(
    "reference",
    [
        {"source": "reviewed_neutral", "offsets_deg": {"neck_rotation": 30}},
        {
            "source": "reviewed_neutral",
            "reference_id": "trial A",
            "offsets_deg": {"neck_rotation": float("nan")},
        },
        {
            "source": "reviewed_neutral",
            "reference_id": "trial A",
            "offsets_deg": {"neck_flexion": 20},
        },
        {"source": "absolute", "offsets_deg": {"trunk_twist": 20}},
    ],
)
def test_incomplete_or_unsupported_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        PostureBaseline.for_recording({}, BaselinePolicy(), {"neutral_reference": reference})


def test_unavailable_angle_placeholder_does_not_pull_neighbours_toward_zero():
    sequence = [
        _angles(trunk_twist=40),
        _angles(unavailable_measurements=["trunk_twist_deg"]),
        _angles(trunk_twist=40),
    ]
    apply_condition_smoothing(sequence, ["trunk_twist"], 3)
    assert [s.to_dict()["trunk_twist"] for s in sequence] == [40, None, 40]


# ----------------------------------------------------------------------
# Smoothing
# ----------------------------------------------------------------------


def test_smoothing_removes_a_single_frame_spike():
    """Landmark noise arrives as isolated spikes; a real posture change does not."""

    series = [10.0, 10.0, 90.0, 10.0, 10.0]

    assert smooth_series(series, 3) == [10.0, 10.0, 10.0, 10.0, 10.0]


def test_smoothing_preserves_a_sustained_change():
    series = [10.0] * 5 + [60.0] * 5

    smoothed = smooth_series(series, 3)

    assert smoothed[0] == pytest.approx(10.0)
    assert smoothed[-1] == pytest.approx(60.0)


def test_smoothing_preserves_gaps():
    """A gap stays a gap, and its neighbours are filtered over what is there."""

    assert smooth_series([10.0, None, 12.0], 3) == [10.0, None, 12.0]
    # With a wider window the gap is spanned, but never filled.
    assert smooth_series([10.0, None, 12.0, 40.0, 12.0], 5) == [11.0, None, 12.0, 12.0, 12.0]


def test_smoothing_is_a_no_op_for_a_window_of_one():
    series = [1.0, 50.0, 2.0]

    assert smooth_series(series, 1) == series


def test_condition_smoothing_only_touches_the_named_measurements():
    sequence = [_angles(trunk_twist=10.0, trunk_flexion=10.0) for _ in range(5)]
    sequence[2].trunk_twist = 90.0
    sequence[2].trunk_flexion = 90.0

    apply_condition_smoothing(sequence, ["trunk_twist"], 3)

    assert sequence[2].trunk_twist == pytest.approx(10.0)
    # Flexion feeds the score bands directly and is reported, so it is untouched.
    assert sequence[2].trunk_flexion == pytest.approx(90.0)


def test_invalid_baseline_policy_is_rejected():
    with pytest.raises(ValueError):
        BaselinePolicy(smoothing_frames=4)
    with pytest.raises(ValueError):
        BaselinePolicy(rectified_percentile=140.0)


def test_baseline_policy_from_config():
    policy = baseline_policy_from_config({"baseline": {"smoothing_frames": 7}})

    assert policy.smoothing_frames == 7
    assert policy.rectified_percentile == BaselinePolicy().rectified_percentile

    with pytest.raises(ValueError, match="unknown posture baseline settings"):
        baseline_policy_from_config({"baseline": {"smoothing": 7}})


def test_thresholds_and_baseline_coexist_in_one_config_block():
    config = {"thresholds": {"trunk_twist_deg": 22.0}, "baseline": {"smoothing_frames": 3}}

    assert thresholds_from_config(config).trunk_twist_deg == 22.0
    assert baseline_policy_from_config(config).smoothing_frames == 3
