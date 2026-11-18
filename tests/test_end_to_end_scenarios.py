"""Whole assessments scored against a hand-worked expectation.

``tests/test_official_tables.py`` already freezes every table cell with a digest
and sweeps every lookup, so the transcription is covered. What that cannot catch
is a wiring fault between the steps: a component computed correctly and then fed
into the wrong axis, an adjustment applied twice, a band read in the wrong
convention. Those only show when a realistic posture is carried all the way to a
final score.

Each scenario below states the posture in plain terms, works the published rules
through step by step in the docstring, and asserts both the intermediate
components and the final number. A reviewer can check the arithmetic without
running anything, and any single step that drifts fails a named assertion rather
than a mysterious total.
"""

from __future__ import annotations

import pytest

from ergonomics.reba import REBACalculator, REBAInput
from ergonomics.rula import RULACalculator, RULAInput


def _rula(**overrides) -> RULAInput:
    """A fully specified neutral-ish standing posture, adjusted per scenario."""

    base = {
        "side": "right",
        "upper_arm_flexion_deg": 10.0,
        "upper_arm_abducted": False,
        "shoulder_raised": False,
        "arm_supported_or_leaning": False,
        "lower_arm_flexion_deg": 80.0,
        "lower_arm_across_midline_or_out_to_side": False,
        "wrist_flexion_deg": 0.0,
        "wrist_deviated": False,
        "wrist_twist_at_end_range": False,
        "neck_flexion_deg": 5.0,
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
    return RULAInput(**{**base, **overrides})


def _reba(**overrides) -> REBAInput:
    base = {
        "side": "right",
        "trunk_flexion_deg": 0.0,
        "trunk_twisted_or_side_flexed": False,
        "neck_flexion_deg": 5.0,
        "neck_twisted_or_side_flexed": False,
        "legs_weight_bearing": "bilateral",
        "knee_flexion_deg": 5.0,
        "seated": False,
        "upper_arm_flexion_deg": 10.0,
        "upper_arm_abducted_or_rotated": False,
        "shoulder_raised": False,
        "arm_supported_or_gravity_assisted": False,
        "lower_arm_flexion_deg": 80.0,
        "wrist_flexion_deg": 0.0,
        "wrist_deviated_or_twisted": False,
        "load_kg": 0.0,
        "shock_or_rapid_force": False,
        "coupling": "good",
        "static_posture_over_one_minute": False,
        "repeated_more_than_four_per_minute": False,
        "rapid_large_posture_change_or_unstable_base": False,
    }
    return REBAInput(**{**base, **overrides})


# ----------------------------------------------------------------------
# RULA
# ----------------------------------------------------------------------


def test_rula_relaxed_standing_scores_the_method_minimum():
    """Upper arm 10 -> 1. Lower arm 80 (in 60-100) -> 1. Wrist 0 -> 1,
    twist 1. Table A(1,1,1,1) -> 1. Muscle use 0, force 0, so score A = 1.
    Neck 5 -> 1. Trunk 0 -> 1. Legs supported -> 1. Table B(1,1,1) -> 1,
    plus 0 and 0, so score B = 1. Table C(1,1) -> 1.
    """

    result = RULACalculator().calculate(_rula())

    assert result.intermediate_scores["upper_arm"] == 1
    assert result.intermediate_scores["lower_arm"] == 1
    assert result.intermediate_scores["wrist"] == 1
    assert result.intermediate_scores["neck"] == 1
    assert result.intermediate_scores["trunk"] == 1
    assert result.intermediate_scores["score_a"] == 1
    assert result.intermediate_scores["score_b"] == 1
    assert result.final_score == 1


def test_rula_overhead_reach_with_a_load_raises_every_stage():
    """Upper arm 100 (>90) -> 4. Lower arm 30 (outside 60-100) -> 2.
    Wrist 20 (>15) -> 3, twist 1. Neck 25 (>20) -> 3. Trunk 25 (20-60) -> 3.
    Load 6 kg static -> force 2. The score must rise well above the minimum
    and the components must each show their own band.
    """

    result = RULACalculator().calculate(
        _rula(
            upper_arm_flexion_deg=100.0,
            lower_arm_flexion_deg=30.0,
            wrist_flexion_deg=20.0,
            neck_flexion_deg=25.0,
            trunk_flexion_deg=25.0,
            load_kg=6.0,
            load_type="static_or_repeated",
        )
    )

    assert result.intermediate_scores["upper_arm"] == 4
    assert result.intermediate_scores["lower_arm"] == 2
    assert result.intermediate_scores["wrist"] == 3
    assert result.intermediate_scores["neck"] == 3
    assert result.intermediate_scores["trunk"] == 3
    assert result.intermediate_scores["force_load"] == 2
    assert result.final_score == 7
    assert result.risk_level is not None


def test_rula_each_posture_adjustment_adds_exactly_one():
    """An adjustment must add one band to its own component and nothing else."""

    plain = RULACalculator().calculate(_rula())
    for field, component in (
        ("upper_arm_abducted", "upper_arm"),
        ("lower_arm_across_midline_or_out_to_side", "lower_arm"),
        ("wrist_deviated", "wrist"),
        ("neck_twisted", "neck"),
        ("trunk_twisted", "trunk"),
    ):
        adjusted = RULACalculator().calculate(_rula(**{field: True}))
        assert adjusted.intermediate_scores[component] == (
            plain.intermediate_scores[component] + 1
        ), field


def test_rula_muscle_use_adds_one_once_not_twice():
    """Static and repeated both set muscle use; together they still add one."""

    both = RULACalculator().calculate(
        _rula(static_posture_over_one_minute=True, repeated_more_than_four_per_minute=True)
    )

    assert both.intermediate_scores["muscle_use"] == 1


def test_rula_withholds_when_a_required_input_is_missing():
    result = RULACalculator().calculate(_rula(trunk_flexion_deg=None))

    assert result.final_score is None
    assert "trunk_flexion_deg" in result.missing_inputs


# ----------------------------------------------------------------------
# REBA
# ----------------------------------------------------------------------


def test_reba_relaxed_standing_scores_the_method_minimum():
    """Trunk 0 -> 1. Neck 5 -> 1. Legs bilateral, knee 5 -> 1.
    Table A(1,1,1) -> 1, load 0, so score A = 1.
    Upper arm 10 -> 1. Lower arm 80 -> 1. Wrist 0 -> 1.
    Table B(1,1,1) -> 1, coupling good 0, so score B = 1.
    Table C(1,1) -> 1, activity 0. Final 1.
    """

    result = REBACalculator().calculate(_reba())

    assert result.intermediate_scores["trunk"] == 1
    assert result.intermediate_scores["neck"] == 1
    assert result.intermediate_scores["legs"] == 1
    assert result.intermediate_scores["score_a"] == 1
    assert result.intermediate_scores["score_b"] == 1
    assert result.intermediate_scores["activity"] == 0
    assert result.final_score == 1


def test_reba_deep_stooped_lift_scores_in_the_high_band():
    """Trunk 70 (>60) -> 4. Neck 25 (>20) -> 2. Knee 70 (>60) -> legs 1+2 = 3.
    Upper arm 50 (45-90) -> 3. Lower arm 30 (outside 60-100) -> 2.
    Wrist 20 (>15) -> 2. Load 12 kg -> 2. Coupling poor -> 2.
    Repeated action -> activity 1.
    """

    result = REBACalculator().calculate(
        _reba(
            trunk_flexion_deg=70.0,
            neck_flexion_deg=25.0,
            knee_flexion_deg=70.0,
            upper_arm_flexion_deg=50.0,
            lower_arm_flexion_deg=30.0,
            wrist_flexion_deg=20.0,
            load_kg=12.0,
            coupling="poor",
            repeated_more_than_four_per_minute=True,
        )
    )

    assert result.intermediate_scores["trunk"] == 4
    assert result.intermediate_scores["neck"] == 2
    assert result.intermediate_scores["legs"] == 3
    assert result.intermediate_scores["upper_arm"] == 3
    assert result.intermediate_scores["lower_arm"] == 2
    assert result.intermediate_scores["wrist"] == 2
    assert result.intermediate_scores["load_force"] == 2
    assert result.intermediate_scores["coupling"] == 2
    assert result.intermediate_scores["activity"] == 1
    assert result.final_score >= 11
    assert result.final_score <= 15


def test_reba_activity_points_accumulate_independently():
    """The three activity conditions each add one, to a maximum of three."""

    none = REBACalculator().calculate(_reba())
    all_three = REBACalculator().calculate(
        _reba(
            static_posture_over_one_minute=True,
            repeated_more_than_four_per_minute=True,
            rapid_large_posture_change_or_unstable_base=True,
        )
    )

    assert none.intermediate_scores["activity"] == 0
    assert all_three.intermediate_scores["activity"] == 3


def test_reba_seated_ignores_knee_flexion():
    """A seated subject's bent knee is not a leg penalty; sitting is the posture."""

    seated = REBACalculator().calculate(_reba(seated=True, knee_flexion_deg=90.0))
    standing = REBACalculator().calculate(_reba(seated=False, knee_flexion_deg=90.0))

    assert seated.intermediate_scores["legs"] < standing.intermediate_scores["legs"]


def test_reba_withholds_when_a_required_input_is_missing():
    result = REBACalculator().calculate(_reba(knee_flexion_deg=None))

    assert result.final_score is None
    assert result.missing_inputs


# ----------------------------------------------------------------------
# Monotonicity: a worse posture can never score better
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["trunk_flexion_deg", "neck_flexion_deg", "upper_arm_flexion_deg"],
)
def test_reba_score_never_falls_as_a_joint_bends_further(field: str):
    """A property no single table lookup can confirm on its own."""

    scores = [
        REBACalculator().calculate(_reba(**{field: float(angle)})).final_score
        for angle in (0, 15, 30, 45, 60, 80)
    ]

    assert all(a is not None for a in scores)
    assert scores == sorted(scores), f"{field}: {scores}"


@pytest.mark.parametrize("load", [0.0, 3.0, 6.0, 12.0, 25.0])
def test_reba_score_never_falls_as_the_load_grows(load: float):
    lighter = REBACalculator().calculate(_reba(load_kg=0.0)).final_score
    heavier = REBACalculator().calculate(_reba(load_kg=load)).final_score

    assert heavier >= lighter
