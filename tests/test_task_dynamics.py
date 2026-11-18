"""Derivation of time-dependent RULA/REBA task conditions from a pose sequence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from temporal.task_dynamics import (
    DERIVABLE_CONDITIONS,
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    PostureSample,
    TaskDynamicsAnalyzer,
    TaskDynamicsPolicy,
    _longest_still_interval,
    analyze_task_dynamics,
    policy_from_config,
)


def _standing_landmarks(
    *, knee_drop: float = 0.45, left_ankle_y: float = -0.90, right_ankle_y: float = -0.90
) -> dict[int, np.ndarray]:
    """Hip-centred world landmarks in metres, Y up, hips at the origin."""

    return {
        LEFT_HIP: np.array([-0.10, 0.0, 0.0, 1.0]),
        RIGHT_HIP: np.array([0.10, 0.0, 0.0, 1.0]),
        LEFT_KNEE: np.array([-0.10, -knee_drop, 0.0, 1.0]),
        RIGHT_KNEE: np.array([0.10, -knee_drop, 0.0, 1.0]),
        LEFT_ANKLE: np.array([-0.10, left_ankle_y, 0.0, 1.0]),
        RIGHT_ANKLE: np.array([0.10, right_ankle_y, 0.0, 1.0]),
    }


def _samples(
    trunk_series: list[float],
    *,
    fps: float = 10.0,
    landmarks_for=None,
) -> list[PostureSample]:
    out = []
    for index, value in enumerate(trunk_series):
        marks = _standing_landmarks() if landmarks_for is None else landmarks_for(index)
        out.append(
            PostureSample(
                timestamp_sec=index / fps,
                frame_number=index,
                angles={"trunk_flexion": value},
                landmarks=marks,
            )
        )
    return out


# ----------------------------------------------------------------------
# Guard rails
# ----------------------------------------------------------------------


def test_too_few_samples_yields_all_unknown():
    result = analyze_task_dynamics([])

    assert set(result.values) == DERIVABLE_CONDITIONS
    assert all(value is None for value in result.values.values())
    assert result.unavailable == sorted(DERIVABLE_CONDITIONS)


def test_missing_angles_leave_motion_conditions_unknown():
    samples = [
        PostureSample(timestamp_sec=i / 10, frame_number=i, angles={}, landmarks={})
        for i in range(30)
    ]
    result = analyze_task_dynamics(samples)

    assert result.values["static_posture_over_one_minute"] is None
    assert result.values["repeated_more_than_four_per_minute"] is None
    assert result.values["seated"] is None


# ----------------------------------------------------------------------
# Static posture
# ----------------------------------------------------------------------


def test_posture_held_beyond_a_minute_is_static():
    # 90 s of a near-motionless trunk at 10 fps.
    series = [20.0 + 0.5 * math.sin(i / 40.0) for i in range(900)]
    result = analyze_task_dynamics(_samples(series))

    assert result.values["static_posture_over_one_minute"] is True
    assert result.evidence["static_posture_over_one_minute"].measured >= 60.0


def test_short_clip_held_throughout_is_unknown_not_false():
    """A 30 s still clip cannot prove or disprove a one-minute hold."""

    series = [20.0] * 300
    result = analyze_task_dynamics(_samples(series))

    assert result.values["static_posture_over_one_minute"] is None
    assert "shorter than" in result.evidence["static_posture_over_one_minute"].note


def test_short_clip_with_real_movement_is_definitively_not_static():
    series = [20.0 + 40.0 * math.sin(i / 5.0) for i in range(300)]
    result = analyze_task_dynamics(_samples(series))

    assert result.values["static_posture_over_one_minute"] is False


def test_longest_still_interval_is_linear_and_correct():
    times = np.arange(0.0, 10.0, 0.1)
    # Still for the first 5 s, then a step far outside the band, so the
    # boundary between still and moving is unambiguous.
    values = np.concatenate([np.full(50, 10.0), np.full(50, 90.0)])

    assert _longest_still_interval(times, values, 5.0) == pytest.approx(4.9, abs=0.01)


def test_still_interval_extends_while_a_slow_drift_stays_in_band():
    """A drift within the band is still a held posture, by definition."""

    times = np.arange(0.0, 10.0, 0.1)
    values = np.linspace(10.0, 20.0, 100)

    # The whole 9.9 s stays inside a 12 deg band.
    assert _longest_still_interval(times, values, 12.0) == pytest.approx(9.9, abs=0.01)
    # A 4 deg band admits only the portion that drifts less than that.
    assert _longest_still_interval(times, values, 4.0) == pytest.approx(4.0, abs=0.2)


def test_longest_still_interval_handles_a_flat_signal():
    times = np.arange(0.0, 5.0, 0.1)
    values = np.full(times.size, 33.0)

    assert _longest_still_interval(times, values, 2.0) == pytest.approx(4.9, abs=0.01)


# ----------------------------------------------------------------------
# Repetition
# ----------------------------------------------------------------------


def test_repetitive_task_is_detected_and_counted():
    # 12 clean cycles over 60 s = 12 per minute.
    fps = 10.0
    frames = 600
    series = [45.0 * math.sin(2 * math.pi * 12 * i / frames) for i in range(frames)]
    result = analyze_task_dynamics(_samples(series, fps=fps))

    assert result.values["repeated_more_than_four_per_minute"] is True
    assert result.cycles_per_minute == pytest.approx(12.0, rel=0.02)
    assert result.dominant_signal == "trunk_flexion"


@pytest.mark.parametrize("cycles_per_minute", [2, 4, 6, 12, 20])
def test_cycle_rate_is_recovered_across_the_useful_range(cycles_per_minute: int):
    """The rate comes from the signal's own period, so phase at the clip edges
    does not bias it."""

    frames = 600
    series = [45.0 * math.sin(2 * math.pi * cycles_per_minute * i / frames) for i in range(frames)]
    result = analyze_task_dynamics(_samples(series))

    assert result.cycles_per_minute == pytest.approx(float(cycles_per_minute), rel=0.02)


def test_slow_task_below_four_per_minute_is_not_repetitive():
    fps = 10.0
    frames = 600
    series = [45.0 * math.sin(2 * math.pi * 2 * i / frames) for i in range(frames)]
    result = analyze_task_dynamics(_samples(series, fps=fps))

    assert result.values["repeated_more_than_four_per_minute"] is False
    assert result.cycles_per_minute == pytest.approx(2.0, rel=0.02)


def test_small_amplitude_jitter_is_not_counted_as_repetition():
    """Pose noise around a held posture must not read as repeated work."""

    rng = np.random.default_rng(11)
    series = list(20.0 + rng.normal(0.0, 1.5, 600))
    result = analyze_task_dynamics(_samples(series))

    assert result.values["repeated_more_than_four_per_minute"] is False
    assert result.cycles_per_minute == 0.0


def test_dominant_signal_picks_the_angle_that_actually_moves():
    samples = []
    for i in range(300):
        samples.append(
            PostureSample(
                timestamp_sec=i / 10,
                frame_number=i,
                angles={
                    "trunk_flexion": 10.0,
                    "right_upper_arm_flexion": 60.0 * math.sin(i / 8.0),
                },
                landmarks=_standing_landmarks(),
            )
        )
    result = analyze_task_dynamics(samples)

    assert result.dominant_signal == "right_upper_arm_flexion"


# ----------------------------------------------------------------------
# Activity and stability
# ----------------------------------------------------------------------


def test_rapid_posture_change_is_flagged():
    # Alternating 0/80 deg at 10 fps is 800 deg/s.
    series = [0.0 if i % 2 else 80.0 for i in range(200)]
    result = analyze_task_dynamics(_samples(series))

    assert result.values["rapid_large_posture_change_or_unstable_base"] is True


def test_steady_work_is_not_flagged_as_rapid():
    series = [20.0 + 10.0 * math.sin(i / 30.0) for i in range(300)]
    result = analyze_task_dynamics(_samples(series))

    assert result.values["rapid_large_posture_change_or_unstable_base"] is False


def test_varying_stance_width_flags_an_unstable_base():
    def landmarks(index: int):
        spread = 0.10 + 0.30 * (index % 20) / 20.0
        marks = _standing_landmarks()
        marks[LEFT_ANKLE] = np.array([-spread, -0.90, 0.0, 1.0])
        marks[RIGHT_ANKLE] = np.array([spread, -0.90, 0.0, 1.0])
        return marks

    series = [20.0 + 0.2 * math.sin(i / 30.0) for i in range(300)]
    result = analyze_task_dynamics(_samples(series, landmarks_for=landmarks))

    assert result.values["rapid_large_posture_change_or_unstable_base"] is True


# ----------------------------------------------------------------------
# Seated and legs
# ----------------------------------------------------------------------


def test_standing_geometry_is_not_seated():
    result = analyze_task_dynamics(_samples([20.0 + 5 * math.sin(i / 9) for i in range(300)]))

    assert result.values["seated"] is False
    assert result.values["legs_weight_bearing"] == "bilateral"
    assert result.values["legs_supported_and_balanced"] is True


def test_seated_geometry_is_detected():
    """Sitting brings the knee up to hip height, collapsing the thigh drop."""

    def seated_landmarks(_index: int):
        return _standing_landmarks(knee_drop=0.05, left_ankle_y=-0.50, right_ankle_y=-0.50)

    result = analyze_task_dynamics(
        _samples([15.0 + 2 * math.sin(i / 9) for i in range(300)], landmarks_for=seated_landmarks)
    )

    assert result.values["seated"] is True


def test_one_raised_foot_reads_as_unilateral_support():
    def landmarks(_index: int):
        return _standing_landmarks(left_ankle_y=-0.60, right_ankle_y=-0.90)

    result = analyze_task_dynamics(
        _samples([20.0 + 2 * math.sin(i / 9) for i in range(300)], landmarks_for=landmarks)
    )

    assert result.values["legs_weight_bearing"] == "unilateral"
    assert result.values["legs_supported_and_balanced"] is False


def test_alternating_support_reads_as_walking():
    def landmarks(index: int):
        # Swap which foot is lifted about twice a second at 10 fps.
        if (index // 5) % 2:
            return _standing_landmarks(left_ankle_y=-0.60, right_ankle_y=-0.90)
        return _standing_landmarks(left_ankle_y=-0.90, right_ankle_y=-0.60)

    result = analyze_task_dynamics(
        _samples([20.0 + 2 * math.sin(i / 9) for i in range(300)], landmarks_for=landmarks)
    )

    assert result.values["legs_weight_bearing"] == "walking"


def test_absent_legs_leave_weight_bearing_unknown():
    samples = [
        PostureSample(
            timestamp_sec=i / 10,
            frame_number=i,
            angles={"trunk_flexion": 20.0 + i % 5},
            landmarks={},
        )
        for i in range(60)
    ]
    result = analyze_task_dynamics(samples)

    assert result.values["legs_weight_bearing"] is None
    assert result.values["legs_supported_and_balanced"] is None


# ----------------------------------------------------------------------
# RULA load type
# ----------------------------------------------------------------------


def test_load_type_follows_repetition():
    fps = 10.0
    frames = 600
    series = [45.0 * math.sin(2 * math.pi * 12 * i / frames) for i in range(frames)]
    result = analyze_task_dynamics(_samples(series, fps=fps))

    assert result.values["rula_load_type"] == "static_or_repeated"


def test_load_type_is_intermittent_when_neither_holds():
    fps = 10.0
    frames = 600
    series = [45.0 * math.sin(2 * math.pi * 2 * i / frames) for i in range(frames)]
    result = analyze_task_dynamics(_samples(series, fps=fps))

    assert result.values["static_posture_over_one_minute"] is False
    assert result.values["repeated_more_than_four_per_minute"] is False
    assert result.values["rula_load_type"] == "intermittent"


def test_load_type_is_unknown_when_an_input_is_unknown():
    result = analyze_task_dynamics(_samples([20.0] * 300))

    assert result.values["static_posture_over_one_minute"] is None
    assert result.values["rula_load_type"] is None


# ----------------------------------------------------------------------
# Policy plumbing
# ----------------------------------------------------------------------


def test_policy_thresholds_change_the_outcome():
    fps = 10.0
    frames = 600
    series = [45.0 * math.sin(2 * math.pi * 6 * i / frames) for i in range(frames)]
    samples = _samples(series, fps=fps)

    strict = TaskDynamicsAnalyzer(TaskDynamicsPolicy(repetition_per_minute=2.0))
    lenient = TaskDynamicsAnalyzer(TaskDynamicsPolicy(repetition_per_minute=20.0))

    assert strict.analyze(samples).values["repeated_more_than_four_per_minute"] is True
    assert lenient.analyze(samples).values["repeated_more_than_four_per_minute"] is False


def test_policy_from_config_overrides_named_fields_only():
    policy = policy_from_config({"static_range_deg": 4.0})

    assert policy.static_range_deg == 4.0
    assert policy.static_window_sec == TaskDynamicsPolicy().static_window_sec


def test_policy_from_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown task dynamics settings"):
        policy_from_config({"static_range_degrees": 4.0})


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        TaskDynamicsPolicy(static_window_sec=0.0)
    with pytest.raises(ValueError):
        TaskDynamicsPolicy(crossing_hysteresis_fraction=1.5)


def test_results_serialise_with_evidence():
    result = analyze_task_dynamics(_samples([20.0 + 30 * math.sin(i / 6) for i in range(300)]))
    payload = result.to_dict()

    assert "values" in payload and "evidence" in payload
    assert payload["dominant_signal"] == "trunk_flexion"
    assert payload["evidence"]["repeated_more_than_four_per_minute"]["units"] == (
        "cycles per minute"
    )


def test_samples_are_sorted_before_analysis():
    ordered = _samples([45.0 * math.sin(i / 8.0) for i in range(300)])
    shuffled = list(reversed(ordered))

    assert (
        analyze_task_dynamics(shuffled).cycles_per_minute
        == analyze_task_dynamics(ordered).cycles_per_minute
    )
