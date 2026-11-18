"""Tests for per-method completion diagnostics."""

from ergonomics.common import BilateralAssessmentResult, select_worst_side
from ergonomics.rula import RULACalculator, RULAInput
from reporting.diagnostics import (
    POSE_MEASUREMENT,
    TASK_INPUT,
    MethodCompletionTracker,
    classify_input,
)

CALCULATOR = RULACalculator()

COMPLETE = dict(
    upper_arm_flexion_deg=10.0,
    upper_arm_abducted=False,
    shoulder_raised=False,
    arm_supported_or_leaning=False,
    lower_arm_flexion_deg=80.0,
    lower_arm_across_midline_or_out_to_side=False,
    wrist_flexion_deg=5.0,
    wrist_deviated=False,
    wrist_twist_at_end_range=False,
    neck_flexion_deg=5.0,
    neck_twisted=False,
    neck_side_bent=False,
    trunk_flexion_deg=10.0,
    trunk_twisted=False,
    trunk_side_bent=False,
    legs_supported_and_balanced=True,
    static_posture_over_one_minute=False,
    repeated_more_than_four_per_minute=False,
    load_kg=1.0,
    load_type="intermittent",
    shock_or_rapid_force=False,
)


def _bilateral(**overrides) -> BilateralAssessmentResult:
    """Build a bilateral RULA result, applying overrides to both sides."""

    values = {**COMPLETE, **overrides}
    left = CALCULATOR.calculate(RULAInput(side="left", **values))
    right = CALCULATOR.calculate(RULAInput(side="right", **values))
    return select_worst_side("RULA", left, right)


def test_classify_separates_measurements_from_task_conditions():
    assert classify_input("knee_flexion_deg") == POSE_MEASUREMENT
    assert classify_input("wrist_flexion_deg") == POSE_MEASUREMENT
    assert classify_input("load_kg") == TASK_INPUT
    assert classify_input("coupling") == TASK_INPUT


def test_fully_complete_observations_report_full_completion():
    tracker = MethodCompletionTracker("RULA")
    for _ in range(4):
        tracker.record(_bilateral())
    summary = tracker.summary()
    assert summary.total_observations == 4
    assert summary.scored_observations == 4
    assert summary.blocked_observations == 0
    assert summary.completion_rate == 1.0
    assert summary.primary_blocker is None
    assert summary.blocking_inputs == []


def test_blocking_measurement_is_counted_and_named():
    tracker = MethodCompletionTracker("RULA")
    for _ in range(3):
        tracker.record(_bilateral(wrist_flexion_deg=None))
    summary = tracker.summary()
    assert summary.scored_observations == 0
    assert summary.blocked_observations == 3
    assert summary.completion_rate == 0.0
    assert summary.primary_blocker == "wrist_flexion_deg"
    assert [item.name for item in summary.blocking_measurements] == ["wrist_flexion_deg"]
    assert summary.blocking_inputs[0].share_of_blocked == 1.0


def test_pose_measurements_outrank_tied_task_conditions():
    """With every input missing, the actionable blocker is the measurement."""

    tracker = MethodCompletionTracker("RULA")
    tracker.record(_bilateral(wrist_flexion_deg=None, load_kg=None, load_type=None))
    summary = tracker.summary()
    assert summary.primary_blocker == "wrist_flexion_deg"
    assert summary.blocking_inputs[0].category == POSE_MEASUREMENT
    assert {item.category for item in summary.blocking_inputs} == {POSE_MEASUREMENT, TASK_INPUT}


def test_partial_completion_reports_only_affected_observations():
    tracker = MethodCompletionTracker("RULA")
    for _ in range(3):
        tracker.record(_bilateral())
    tracker.record(_bilateral(wrist_flexion_deg=None))
    summary = tracker.summary()
    assert summary.total_observations == 4
    assert summary.scored_observations == 3
    assert summary.completion_rate == 0.75
    assert summary.blocking_inputs[0].blocked_observations == 1


def test_status_is_attributed_once_per_observation():
    tracker = MethodCompletionTracker("RULA")
    for _ in range(2):
        tracker.record(_bilateral(wrist_flexion_deg=None))
    summary = tracker.summary()
    assert sum(summary.blocking_statuses.values()) == 2


def test_empty_tracker_is_reported_honestly():
    summary = MethodCompletionTracker("REBA").summary()
    assert summary.total_observations == 0
    assert summary.completion_rate is None
    assert summary.primary_blocker is None
    assert "no pose observations" in summary.describe()


def test_describe_names_measurements_and_counts_task_conditions():
    tracker = MethodCompletionTracker("RULA")
    tracker.record(_bilateral(wrist_flexion_deg=None, load_kg=None))
    text = tracker.summary().describe()
    assert "wrist_flexion_deg" in text
    assert "task condition" in text


def test_summary_is_json_safe():
    tracker = MethodCompletionTracker("RULA")
    tracker.record(_bilateral(wrist_flexion_deg=None))
    payload = tracker.summary().to_dict()
    assert payload["method"] == "RULA"
    assert payload["blocking_measurement_count"] == 1
    assert payload["blocking_inputs"][0]["category"] == POSE_MEASUREMENT
