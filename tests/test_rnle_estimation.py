"""Estimation of RNLE task inputs from pose geometry and participant stature."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ergonomics.rnle_estimation import (
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_WRIST,
    LiftEstimationPolicy,
    LiftSample,
    estimate_rnle_inputs,
    estimate_scale_factor,
    policy_from_config,
)

# The synthetic subject: 1.75 m tall, feet on the floor plane at y = 0, ears at
# 0.936 of stature. Building the fixture from real anthropometry means the
# estimator's output can be checked against a known ground truth rather than
# against itself.
STATURE_M = 1.75
FLOOR_Y = 0.0
EAR_Y = STATURE_M * 0.936


def _sample(
    index: int,
    hand_height_m: float,
    *,
    reach_m: float = 0.40,
    lateral_m: float = 0.0,
    fps: float = 10.0,
) -> LiftSample:
    """One frame with the hands at a given height and forward reach.

    Anterior is ``cross(superior, lateral)``, the convention AngleCalculator's
    pelvis frame uses. With the hips running left(-X) to right(+X) and the spine
    along +Y, that puts the subject's front at -Z, so a forward reach is a
    negative Z offset and reads as zero asymmetry.
    """

    return LiftSample(
        timestamp_sec=index / fps,
        frame_number=index,
        landmarks={
            LEFT_EAR: np.array([-0.07, EAR_Y, 0.0]),
            RIGHT_EAR: np.array([0.07, EAR_Y, 0.0]),
            LEFT_HIP: np.array([-0.10, 0.95, 0.0]),
            RIGHT_HIP: np.array([0.10, 0.95, 0.0]),
            LEFT_ANKLE: np.array([-0.10, 0.08, 0.0]),
            RIGHT_ANKLE: np.array([0.10, 0.08, 0.0]),
            LEFT_HEEL: np.array([-0.10, FLOOR_Y, -0.02]),
            RIGHT_HEEL: np.array([0.10, FLOOR_Y, -0.02]),
            LEFT_WRIST: np.array([-0.15 + lateral_m, hand_height_m, -reach_m]),
            RIGHT_WRIST: np.array([0.15 + lateral_m, hand_height_m, -reach_m]),
        },
    )


#: Frames of settled posture added at each end of a synthetic trial. A turning
#: point cannot be confirmed at the very first or last frame, so without this
#: padding the outermost lift of a fixture is unverifiable and "N cycles" would
#: not mean "N detectable lifts".
PAD_FRAMES = 4


def _cycle_heights(cycles: int, low_m: float, high_m: float, per_cycle: int) -> list[float]:
    heights = [low_m] * PAD_FRAMES
    for index in range(cycles * per_cycle):
        phase = 2 * math.pi * (index % per_cycle) / per_cycle
        # Each cycle starts at the low point, so it contains one full rise.
        heights.append(low_m + (high_m - low_m) * (1 - math.cos(phase)) / 2)
    heights.extend([low_m] * PAD_FRAMES)
    return heights


def _lift_cycle(cycles: int, *, low_m: float = 0.25, high_m: float = 1.05, per_cycle: int = 20):
    """A repeating floor-to-waist lift, as a smooth hand-height oscillation."""

    return [
        _sample(index, height)
        for index, height in enumerate(_cycle_heights(cycles, low_m, high_m, per_cycle))
    ]


# ----------------------------------------------------------------------
# Scale
# ----------------------------------------------------------------------


def test_scale_factor_recovers_a_known_stature():
    samples = _lift_cycle(2)
    factor, basis = estimate_scale_factor(samples, participant_height_cm=175.0)

    assert factor == pytest.approx(1.0, abs=0.02)
    assert "175 cm" in basis


def test_taller_participant_scales_distances_up():
    samples = _lift_cycle(2)
    factor, _ = estimate_scale_factor(samples, participant_height_cm=196.0)

    assert factor == pytest.approx(196.0 / 175.0, abs=0.03)


def test_missing_height_falls_back_to_the_model_scale():
    factor, basis = estimate_scale_factor(_lift_cycle(2), participant_height_cm=None)

    assert factor == 1.0
    assert "No participant height" in basis


def test_implausible_scale_is_rejected_rather_than_applied():
    """A wrong scale would silently corrupt every distance downstream."""

    factor, basis = estimate_scale_factor(_lift_cycle(2), participant_height_cm=60.0)

    assert factor == 1.0
    assert "outside the plausible range" in basis


def test_scale_without_usable_landmarks_falls_back():
    samples = [LiftSample(timestamp_sec=i / 10, frame_number=i) for i in range(10)]
    factor, basis = estimate_scale_factor(samples, participant_height_cm=175.0)

    assert factor == 1.0
    assert "could not be measured" in basis


# ----------------------------------------------------------------------
# Lift detection
# ----------------------------------------------------------------------


def test_lifts_are_detected_and_counted():
    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    assert result.lift_count == 4


def test_frequency_matches_the_lift_rate():
    # 6 cycles of 20 frames at 10 fps = 2 s per lift = 30 lifts per minute.
    result = estimate_rnle_inputs(_lift_cycle(6), participant_height_cm=175.0)

    assert result.frequency_lifts_per_min == pytest.approx(30.0, rel=0.2)


def test_small_hand_movement_is_not_a_lift():
    """Reaching and adjusting must not be counted as lifting."""

    samples = _lift_cycle(5, low_m=0.95, high_m=1.05)
    result = estimate_rnle_inputs(samples, participant_height_cm=175.0)

    assert result.lift_count == 0
    assert result.horizontal_cm is None
    assert any("No lift was detected" in note for note in result.notes)


def test_too_few_frames_yields_no_lift():
    result = estimate_rnle_inputs(_lift_cycle(1)[:3], participant_height_cm=175.0)

    assert result.lift_count == 0


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------


def test_measured_geometry_matches_the_constructed_lift():
    """H, V and D are checked against the fixture's known dimensions."""

    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    # Hands reach 0.40 m forward of the ankles.
    assert result.horizontal_cm == pytest.approx(40.0, abs=3.0)
    # The lift starts with the hands 0.25 m above the floor.
    assert result.vertical_cm == pytest.approx(25.0, abs=3.0)
    # And travels 1.05 - 0.25 = 0.80 m.
    assert result.travel_distance_cm == pytest.approx(80.0, abs=5.0)


def test_a_longer_reach_reads_as_a_larger_horizontal_distance():
    close = estimate_rnle_inputs(
        [_sample(i, h, reach_m=0.25) for i, h in enumerate(_heights(4))],
        participant_height_cm=175.0,
    )
    far = estimate_rnle_inputs(
        [_sample(i, h, reach_m=0.60) for i, h in enumerate(_heights(4))],
        participant_height_cm=175.0,
    )

    assert far.horizontal_cm > close.horizontal_cm + 20.0


def _heights(cycles: int, per_cycle: int = 20, low: float = 0.25, high: float = 1.05):
    return _cycle_heights(cycles, low, high, per_cycle)


def test_a_forward_lift_is_symmetric():
    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    assert result.asymmetry_deg == pytest.approx(0.0, abs=10.0)


def test_a_sideways_lift_reads_as_asymmetric():
    samples = [_sample(i, h, reach_m=0.05, lateral_m=0.45) for i, h in enumerate(_heights(4))]
    result = estimate_rnle_inputs(samples, participant_height_cm=175.0)

    assert result.asymmetry_deg > 45.0


def test_participant_height_changes_the_reported_distances():
    short = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=150.0)
    tall = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=195.0)

    assert tall.horizontal_cm > short.horizontal_cm


# ----------------------------------------------------------------------
# What video cannot supply
# ----------------------------------------------------------------------


def test_duration_and_coupling_are_assumed_and_labelled():
    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    assert result.duration == "moderate"
    assert result.coupling == "fair"
    assert result.sources["duration"] == "assumed"
    assert result.sources["coupling"] == "assumed"
    assert any("shift" in note for note in result.notes)


def test_supplied_duration_and_coupling_are_marked_observed():
    result = estimate_rnle_inputs(
        _lift_cycle(4), participant_height_cm=175.0, duration="long", coupling="good"
    )

    assert result.duration == "long"
    assert result.coupling == "good"
    assert result.sources["duration"] == "observed"
    assert result.sources["coupling"] == "observed"


def test_geometry_is_marked_derived():
    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    for name in ("horizontal_cm", "vertical_cm", "travel_distance_cm", "asymmetry_deg"):
        assert result.sources[name] == "derived"


def test_load_weight_is_never_estimated():
    """Mass is not recoverable from geometry and must not be invented."""

    result = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0)

    assert "load_weight_kg" not in result.as_input_fields()
    assert "load_weight_kg" not in result.sources


def test_serialised_result_declares_its_measurement_source():
    payload = estimate_rnle_inputs(_lift_cycle(4), participant_height_cm=175.0).to_dict()

    assert payload["distance_measurement_source"] == "anthropometric_estimate"
    assert payload["lift_count"] == 4
    assert len(payload["lifts"]) == 4


# ----------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------


def test_policy_threshold_changes_what_counts_as_a_lift():
    # A 20 cm rise: below the 25 cm default, above a 10 cm override.
    samples = _lift_cycle(4, low_m=0.80, high_m=1.00)
    strict = estimate_rnle_inputs(
        samples, participant_height_cm=175.0, policy=LiftEstimationPolicy()
    )
    lenient = estimate_rnle_inputs(
        samples,
        participant_height_cm=175.0,
        policy=LiftEstimationPolicy(minimum_lift_travel_cm=10.0),
    )

    assert strict.lift_count == 0
    assert lenient.lift_count == 4


def test_policy_from_config_overrides_named_fields():
    policy = policy_from_config({"assumed_coupling": "poor", "minimum_lift_travel_cm": 15})

    assert policy.assumed_coupling == "poor"
    assert policy.minimum_lift_travel_cm == 15.0
    assert policy.assumed_duration == LiftEstimationPolicy().assumed_duration


def test_policy_from_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown rnle estimation settings"):
        policy_from_config({"minimum_lift_travel": 15})


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        LiftEstimationPolicy(minimum_lift_travel_cm=0)
    with pytest.raises(ValueError):
        LiftEstimationPolicy(assumed_duration="all_day")
