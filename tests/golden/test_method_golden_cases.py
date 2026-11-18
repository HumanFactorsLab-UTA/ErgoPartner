"""Golden cases tied to the primary method publications.

Inputs, expected intermediate values, and citations are repeated in
ERGONOMICS_METHODS.md so changes require an explicit traceability update.
"""

import pytest

from ergonomics.niosh import NIOSHCalculator, NIOSHInput, NIOSHTaskApplicability
from ergonomics.reba import REBACalculator, REBAInput
from ergonomics.rula import RULACalculator, RULAInput


def test_rula_1993_neutral_experimental_condition_is_grand_score_one():
    result = RULACalculator().calculate(
        RULAInput(
            side="right",
            upper_arm_flexion_deg=0,
            upper_arm_abducted=False,
            shoulder_raised=False,
            arm_supported_or_leaning=False,
            lower_arm_flexion_deg=90,
            lower_arm_across_midline_or_out_to_side=False,
            wrist_flexion_deg=0,
            wrist_deviated=False,
            wrist_twist_at_end_range=False,
            neck_flexion_deg=0,
            neck_twisted=False,
            neck_side_bent=False,
            trunk_flexion_deg=0,
            trunk_twisted=False,
            trunk_side_bent=False,
            legs_supported_and_balanced=True,
            static_posture_over_one_minute=False,
            repeated_more_than_four_per_minute=False,
            load_kg=0,
            load_type="intermittent",
            shock_or_rapid_force=False,
        )
    )
    assert result.intermediate_scores["table_a_posture"] == 1
    assert result.intermediate_scores["table_b_posture"] == 1
    assert result.final_score == 1


def test_reba_2000_published_worked_example_is_eleven():
    result = REBACalculator().calculate(
        REBAInput(
            side="right",
            trunk_flexion_deg=65,
            trunk_twisted_or_side_flexed=True,
            neck_flexion_deg=25,
            neck_twisted_or_side_flexed=False,
            legs_weight_bearing="unstable",
            knee_flexion_deg=35,
            seated=False,
            upper_arm_flexion_deg=70,
            upper_arm_abducted_or_rotated=True,
            shoulder_raised=False,
            arm_supported_or_gravity_assisted=True,
            lower_arm_flexion_deg=110,
            wrist_flexion_deg=0,
            wrist_deviated_or_twisted=False,
            load_kg=0,
            shock_or_rapid_force=False,
            coupling="fair",
            static_posture_over_one_minute=False,
            repeated_more_than_four_per_minute=False,
            rapid_large_posture_change_or_unstable_base=True,
        )
    )
    assert result.intermediate_scores["score_a"] == 8
    assert result.intermediate_scores["score_b"] == 5
    assert result.intermediate_scores["table_c"] == 10
    assert result.final_score == 11


def test_rnle_manual_formula_ideal_conditions():
    result = NIOSHCalculator().calculate(
        NIOSHInput(
            horizontal_cm=25,
            vertical_cm=75,
            travel_distance_cm=25,
            asymmetry_deg=0,
            frequency_lifts_per_min=0.2,
            duration="short",
            coupling="good",
            load_weight_kg=23,
            applicability=NIOSHTaskApplicability.reviewed_applicable(),
        )
    )
    assert result.multipliers == {
        "LC": 23.0,
        "HM": 1.0,
        "VM": 1.0,
        "DM": 1.0,
        "AM": 1.0,
        "FM": 1.0,
        "CM": 1.0,
    }
    assert result.recommended_weight_limit_kg == pytest.approx(23.0)
    assert result.lifting_index == pytest.approx(1.0)
