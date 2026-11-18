"""Authoritative RULA scoring and boundary tests.

Expected values are transcribed from McAtamney & Corlett (1993), figures 1-5
and Tables A-C.  These tests exercise the ergonomic core without pose or video
dependencies.
"""

import pytest

from ergonomics.rula import RULACalculator, RULAInput


def neutral_input(**changes) -> RULAInput:
    values = {
        "side": "right",
        "upper_arm_flexion_deg": 0.0,
        "upper_arm_abducted": False,
        "shoulder_raised": False,
        "arm_supported_or_leaning": False,
        "lower_arm_flexion_deg": 90.0,
        "lower_arm_across_midline_or_out_to_side": False,
        "wrist_flexion_deg": 0.0,
        "wrist_deviated": False,
        "wrist_twist_at_end_range": False,
        "neck_flexion_deg": 0.0,
        "neck_twisted": False,
        "neck_side_bent": False,
        "trunk_flexion_deg": 0.0,
        "trunk_twisted": False,
        "trunk_side_bent": False,
        "legs_supported_and_balanced": True,
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "load_kg": 0.0,
        "load_type": "intermittent",
        "shock_or_rapid_force": False,
    }
    values.update(changes)
    return RULAInput(**values)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-20.0001, 2),
        (-20.0, 1),
        (-19.9999, 1),
        (19.9999, 1),
        (20.0, 1),
        (20.0001, 2),
        (44.9999, 2),
        (45.0, 2),
        (45.0001, 3),
        (89.9999, 3),
        (90.0, 3),
        (90.0001, 4),
    ],
)
def test_upper_arm_exact_boundaries(angle, expected):
    assert RULACalculator.score_upper_arm(angle, False, False, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(59.9999, 2), (60.0, 1), (60.0001, 1), (99.9999, 1), (100.0, 1), (100.0001, 2)],
)
def test_lower_arm_exact_boundaries(angle, expected):
    assert RULACalculator.score_lower_arm(angle, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-0.0001, 4),
        (0.0, 1),
        (0.0001, 1),
        (9.9999, 1),
        (10.0, 1),
        (10.0001, 2),
        (19.9999, 2),
        (20.0, 2),
        (20.0001, 3),
    ],
)
def test_neck_exact_boundaries(angle, expected):
    assert RULACalculator.score_neck(angle, False, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 1),
        (0.0001, 2),
        (19.9999, 2),
        (20.0, 2),
        (20.0001, 3),
        (59.9999, 3),
        (60.0, 3),
        (60.0001, 4),
    ],
)
def test_trunk_exact_boundaries(angle, expected):
    assert RULACalculator.score_trunk(angle, False, False) == expected


def test_rula_adjustments_are_independent_and_cumulative():
    assert RULACalculator.score_upper_arm(0, True, True, False) == 3
    assert RULACalculator.score_upper_arm(0, False, False, True) == 1
    assert RULACalculator.score_neck(0, True, True) == 3
    assert RULACalculator.score_trunk(0, True, True) == 3


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-15.0001, 3), (-15.0, 2), (-0.0001, 2), (0.0, 1), (0.0001, 2), (15.0, 2), (15.0001, 3)],
)
def test_wrist_neutral_only_at_zero(angle, expected):
    assert RULACalculator.score_wrist(angle, deviated=False) == expected


def test_rula_table_a_includes_lower_arm_score_three():
    calc = RULACalculator()
    assert calc.lookup_table_a(1, 3, 1, 1) == 2
    assert calc.lookup_table_a(6, 3, 4, 2) == 9


def test_table_lookups_reject_invalid_scores_instead_of_clamping():
    calc = RULACalculator()
    with pytest.raises(ValueError, match="lower_arm_score"):
        calc.lookup_table_a(1, 4, 1, 1)
    with pytest.raises(ValueError, match="score_a"):
        calc.lookup_table_c(0, 1)


def test_force_exact_boundaries():
    score = RULACalculator.score_force
    assert score(1.9999, "intermittent", False) == 0
    assert score(1.9999, "static_or_repeated", False) == 0
    assert score(2.0, "intermittent", False) == 0
    assert score(2.0, "static_or_repeated", False) == 2
    assert score(2.0001, "intermittent", False) == 1
    assert score(2.0001, "static_or_repeated", False) == 2
    assert score(10.0, "intermittent", False) == 1
    assert score(10.0, "static_or_repeated", False) == 2
    assert score(10.0001, "intermittent", False) == 2
    assert score(10.0001, "static_or_repeated", False) == 3
    assert score(0.0, "intermittent", True) == 3


def test_neutral_rula_is_valid_and_auditable():
    result = RULACalculator().calculate(neutral_input())
    assert result.validity_status == "valid"
    assert result.final_score == 1
    assert result.intermediate_scores["table_a_posture"] == 1
    assert result.intermediate_scores["table_b_posture"] == 1
    assert result.raw_measurements["wrist_flexion_deg"] == 0.0


def test_missing_load_is_structured_incomplete_result():
    result = RULACalculator().calculate(neutral_input(load_kg=None))
    assert result.final_score is None
    assert result.validity_status == "missing_required_task_input"
    assert "load_kg" in result.missing_inputs


def test_bilateral_selects_worst_side_after_separate_calculation():
    calc = RULACalculator()
    left = neutral_input(side="left")
    right = neutral_input(side="right", upper_arm_flexion_deg=100, wrist_flexion_deg=30)
    result = calc.calculate_bilateral(left, right)
    assert result.left.side == "left"
    assert result.right.side == "right"
    assert result.selected_side == "right"
    assert result.selected.final_score == result.right.final_score
