"""REBA scoring tests from Hignett & McAtamney (2000)."""

import pytest

from ergonomics.reba import REBACalculator, REBAInput


def neutral_input(**changes) -> REBAInput:
    values = {
        "side": "right",
        "trunk_flexion_deg": 0.0,
        "trunk_twisted_or_side_flexed": False,
        "neck_flexion_deg": 0.0,
        "neck_twisted_or_side_flexed": False,
        "legs_weight_bearing": "bilateral",
        "knee_flexion_deg": 0.0,
        "seated": False,
        "upper_arm_flexion_deg": 0.0,
        "upper_arm_abducted_or_rotated": False,
        "shoulder_raised": False,
        "arm_supported_or_gravity_assisted": False,
        "lower_arm_flexion_deg": 90.0,
        "wrist_flexion_deg": 0.0,
        "wrist_deviated_or_twisted": False,
        "load_kg": 0.0,
        "shock_or_rapid_force": False,
        "coupling": "good",
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "rapid_large_posture_change_or_unstable_base": False,
    }
    values.update(changes)
    return REBAInput(**values)


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-20.0001, 3),
        (-20.0, 2),
        (-0.0001, 2),
        (0.0, 1),
        (0.0001, 2),
        (20.0, 2),
        (20.0001, 3),
        (60.0, 3),
        (60.0001, 4),
    ],
)
def test_trunk_boundaries(angle, expected):
    assert REBACalculator.score_trunk(angle, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-0.0001, 2), (0.0, 1), (0.0001, 1), (19.9999, 1), (20.0, 1), (20.0001, 2)],
)
def test_neck_boundaries(angle, expected):
    assert REBACalculator.score_neck(angle, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(29.9999, 1), (30.0, 1), (30.0001, 2), (59.9999, 2), (60.0, 2), (60.0001, 3)],
)
def test_leg_knee_boundaries(angle, expected):
    assert REBACalculator.score_legs("bilateral", angle, seated=False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (-20.0001, 2),
        (-20.0, 1),
        (20.0, 1),
        (20.0001, 2),
        (45.0, 2),
        (45.0001, 3),
        (90.0, 3),
        (90.0001, 4),
    ],
)
def test_upper_arm_boundaries_are_reba_specific(angle, expected):
    assert REBACalculator.score_upper_arm(angle, False, False, False) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(59.9999, 2), (60.0, 1), (100.0, 1), (100.0001, 2)],
)
def test_lower_arm_boundaries(angle, expected):
    assert REBACalculator.score_lower_arm(angle) == expected


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-15.0001, 2), (-15.0, 1), (0.0, 1), (15.0, 1), (15.0001, 2)],
)
def test_wrist_boundaries(angle, expected):
    assert REBACalculator.score_wrist(angle, False) == expected


@pytest.mark.parametrize(
    ("load", "expected"),
    [(4.9999, 0), (5.0, 1), (5.0001, 1), (9.9999, 1), (10.0, 1), (10.0001, 2)],
)
def test_load_boundaries(load, expected):
    assert REBACalculator.score_load(load, shock_or_rapid=False) == expected


def test_reba_can_reach_official_maximum_fifteen():
    extreme = neutral_input(
        trunk_flexion_deg=70,
        trunk_twisted_or_side_flexed=True,
        neck_flexion_deg=30,
        neck_twisted_or_side_flexed=True,
        legs_weight_bearing="unstable",
        knee_flexion_deg=70,
        upper_arm_flexion_deg=100,
        upper_arm_abducted_or_rotated=True,
        shoulder_raised=True,
        lower_arm_flexion_deg=30,
        wrist_flexion_deg=30,
        wrist_deviated_or_twisted=True,
        load_kg=15,
        shock_or_rapid_force=True,
        coupling="unacceptable",
        static_posture_over_one_minute=True,
        repeated_more_than_four_per_minute=True,
        rapid_large_posture_change_or_unstable_base=True,
    )
    result = REBACalculator().calculate(extreme)
    assert result.intermediate_scores["table_c"] == 12
    assert result.intermediate_scores["activity"] == 3
    assert result.final_score == 15
    assert result.risk_level == "Very High"


def test_sitting_does_not_receive_knee_flexion_adjustment():
    assert REBACalculator.score_legs("bilateral", 90.0, seated=True) == 1


def test_invalid_coupling_is_not_silently_changed_to_fair():
    result = REBACalculator().calculate(neutral_input(coupling="unknown"))
    assert result.final_score is None
    assert result.validity_status == "missing_required_task_input"
    assert "coupling" in result.invalid_inputs


def test_original_paper_worked_example_reaches_score_eleven():
    """Original paper worked worksheet: A=8, B=5, C=10, activity=1."""
    example = neutral_input(
        trunk_flexion_deg=65,
        trunk_twisted_or_side_flexed=True,
        neck_flexion_deg=25,
        legs_weight_bearing="unstable",
        knee_flexion_deg=35,
        upper_arm_flexion_deg=70,
        upper_arm_abducted_or_rotated=True,
        shoulder_raised=False,
        arm_supported_or_gravity_assisted=True,
        lower_arm_flexion_deg=110,
        wrist_flexion_deg=0,
        wrist_deviated_or_twisted=False,
        load_kg=0,
        coupling="fair",
        rapid_large_posture_change_or_unstable_base=True,
    )
    result = REBACalculator().calculate(example)
    assert result.intermediate_scores["score_a"] == 8
    assert result.intermediate_scores["score_b"] == 5
    assert result.intermediate_scores["table_c"] == 10
    assert result.final_score == 11
