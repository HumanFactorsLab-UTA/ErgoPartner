"""Derived posture conditions must actually reach the RULA and REBA scores.

The rest of the suite checks each layer in isolation: that an angle becomes a
condition, that a condition resolves with the right precedence, that the
calculators score a given input correctly. This module checks the join -- that a
measurement made by the pipeline changes the number an ergonomist reads.

Without these, the derivation could be computed, recorded, and silently ignored
at the adapter boundary, which is exactly the defect the derivation was written
to fix.
"""

from __future__ import annotations

from core.angle_calculator import BodyAngles
from ergonomics.adapters import build_reba_input, build_rula_input
from ergonomics.common import AssessmentContext
from ergonomics.condition_resolution import DERIVED, ConditionResolver
from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from temporal.task_dynamics import DynamicsEvidence, TaskDynamics

# A moderate working posture, chosen to sit away from both ends of both scales
# so that one extra penalty is visible in the final number. This matters in both
# directions: real lifting footage often pins RULA at its ceiling of 7, and a
# near-upright posture pins REBA at its floor of 1. At either extreme a correctly
# applied penalty changes nothing, and a test built there would pass whether or
# not the derived conditions reached the calculator.
MODERATE_POSTURE = {
    "trunk_flexion": 35.0,
    "neck_flexion": 15.0,
    "left_upper_arm_flexion": 50.0,
    "right_upper_arm_flexion": 50.0,
    "left_forearm_flexion": 80.0,
    "right_forearm_flexion": 80.0,
    "left_wrist_flexion": 5.0,
    "right_wrist_flexion": 5.0,
    "left_knee_flexion": 35.0,
    "right_knee_flexion": 35.0,
}


def _both(measurement: str, value) -> dict:
    """Set the same measurement on both anatomical sides."""

    return {f"left_{measurement}": value, f"right_{measurement}": value}


def _angles(**overrides) -> BodyAngles:
    angles = BodyAngles()
    for name, value in {**MODERATE_POSTURE, **overrides}.items():
        setattr(angles, name, value)
    # Every landmark clearly visible, so nothing is withheld for pose quality.
    angles.landmark_visibility = {index: 0.99 for index in range(33)}
    angles.measurement_visibility = {}
    return angles


def _dynamics(**values) -> TaskDynamics:
    dynamics = TaskDynamics()
    for name, value in values.items():
        dynamics.values[name] = value
        dynamics.evidence[name] = DynamicsEvidence(condition=name, value=value, basis="test")
    return dynamics


NEUTRAL_DYNAMICS = dict(
    seated=False,
    legs_weight_bearing="bilateral",
    legs_supported_and_balanced=True,
    static_posture_over_one_minute=False,
    repeated_more_than_four_per_minute=False,
    rapid_large_posture_change_or_unstable_base=False,
    rula_load_type="intermittent",
)


def _assess(angles: BodyAngles, resolver: ConditionResolver):
    """Run one posture all the way to bilateral RULA and REBA results."""

    task = resolver.resolve(angles).values
    context = AssessmentContext(frame_number=1, timestamp_sec=0.0)
    kwargs = {"visibility_threshold": 0.5, "allow_pose_only_wrist": True}
    rula = RULACalculator().calculate_bilateral(
        build_rula_input(angles, "left", task, context, **kwargs),
        build_rula_input(angles, "right", task, context, **kwargs),
    )
    reba = REBACalculator().calculate_bilateral(
        build_reba_input(angles, "left", task, context, **kwargs),
        build_reba_input(angles, "right", task, context, **kwargs),
    )
    return rula, reba


def _score(angles: BodyAngles, resolver: ConditionResolver) -> tuple[int | None, int | None]:
    rula, reba = _assess(angles, resolver)
    return (
        rula.selected.final_score if rula.selected else None,
        reba.selected.final_score if reba.selected else None,
    )


def _component(angles: BodyAngles, resolver: ConditionResolver, method: str, name: str):
    """One intermediate body-part score from the side the method selected.

    Assertions target the component a penalty feeds rather than only the grand
    score, because RULA's Table C and REBA's Table C both compress: a correctly
    applied +1 on one branch often leaves the final number unchanged. Checking
    the component proves the condition reached the calculator; the final-score
    tests below cover the cases where the effect does propagate.
    """

    rula, reba = _assess(angles, resolver)
    result = rula if method == "rula" else reba
    assert result.selected is not None, f"{method} withheld a score for this fixture"
    return result.selected.intermediate_scores[name]


# The conditions no camera can answer. Every resolver in this module supplies
# them, so a withheld score can only mean a derived condition went missing.
UNOBSERVABLE = {
    "load_kg": 2.0,
    "coupling": "good",
    "shock_or_rapid_force": False,
    "left_arm_supported_or_leaning": False,
    "right_arm_supported_or_leaning": False,
    "left_arm_supported_or_gravity_assisted": False,
    "right_arm_supported_or_gravity_assisted": False,
    # Forearm rotation has no anatomical zero in this pipeline, so twist is
    # entered rather than derived; see core/posture_conditions.py.
    "left_wrist_twist_at_end_range": False,
    "right_wrist_twist_at_end_range": False,
    "left_wrist_deviated_or_twisted": False,
    "right_wrist_deviated_or_twisted": False,
}


def _resolver(extra_manual: dict | None = None, **dynamics_overrides) -> ConditionResolver:
    """A resolver with only the unobservable conditions supplied by hand."""

    resolver = ConditionResolver({**UNOBSERVABLE, **(extra_manual or {})})
    resolver.set_dynamics(_dynamics(**{**NEUTRAL_DYNAMICS, **dynamics_overrides}))
    return resolver


def test_baseline_posture_has_headroom_at_both_ends():
    """The fixture must leave room to move, or later assertions prove nothing."""

    rula, reba = _score(_angles(), _resolver())

    assert rula is not None and reba is not None
    assert 1 < rula < 7, f"RULA fixture at {rula} is pinned; a penalty could not show"
    assert 1 < reba < 15, f"REBA fixture at {reba} is pinned; a penalty could not show"


def test_measured_trunk_twist_raises_the_trunk_component():
    """A twist the camera saw must cost the same as a twist an operator typed."""

    resolver = _resolver()
    upright = _component(_angles(trunk_twist=0.0), resolver, "rula", "trunk")
    twisted = _component(_angles(trunk_twist=35.0), resolver, "rula", "trunk")

    assert twisted == upright + 1


def test_measured_trunk_twist_raises_the_reba_trunk_component():
    resolver = _resolver()
    upright = _component(_angles(trunk_twist=0.0), resolver, "reba", "trunk")
    twisted = _component(_angles(trunk_twist=35.0), resolver, "reba", "trunk")

    assert twisted == upright + 1


def test_measured_arm_abduction_raises_the_upper_arm_component():
    # Both sides, because the methods report the worst side: changing one arm
    # only would leave the other selected and the assertion meaningless.
    resolver = _resolver()
    neutral = _component(
        _angles(**_both("upper_arm_abduction", 0.0)), resolver, "rula", "upper_arm"
    )
    abducted = _component(
        _angles(**_both("upper_arm_abduction", 45.0)), resolver, "rula", "upper_arm"
    )

    assert abducted == neutral + 1


def test_measured_wrist_deviation_raises_the_wrist_component():
    resolver = _resolver()
    straight = _component(_angles(**_both("wrist_deviation", 0.0)), resolver, "rula", "wrist")
    deviated = _component(_angles(**_both("wrist_deviation", 25.0)), resolver, "rula", "wrist")

    assert deviated == straight + 1


def test_measured_neck_side_bend_raises_the_neck_component():
    resolver = _resolver()
    upright = _component(_angles(neck_side_bend=0.0), resolver, "rula", "neck")
    bent = _component(_angles(neck_side_bend=25.0), resolver, "rula", "neck")

    assert bent == upright + 1


def test_measured_midline_cross_raises_the_lower_arm_component():
    resolver = _resolver()
    inside = _component(
        _angles(**_both("forearm_midline_cross", False)), resolver, "rula", "lower_arm"
    )
    across = _component(
        _angles(**_both("forearm_midline_cross", True)), resolver, "rula", "lower_arm"
    )

    assert across == inside + 1


def test_a_large_measured_twist_moves_the_final_rula_score():
    """At least one derived condition must be shown to move the grand score."""

    resolver = _resolver()
    upright, _ = _score(_angles(trunk_flexion=15.0, trunk_twist=0.0), resolver)
    twisted, _ = _score(_angles(trunk_flexion=15.0, trunk_twist=35.0), resolver)

    assert twisted > upright


def test_interval_repetition_raises_the_scores():
    """Repetition is measured over the sequence, not seen in any one frame."""

    angles = _angles()
    quiet_rula, quiet_reba = _score(angles, _resolver())
    busy_rula, busy_reba = _score(
        angles,
        _resolver(
            repeated_more_than_four_per_minute=True,
            rula_load_type="static_or_repeated",
        ),
    )

    assert busy_rula > quiet_rula
    assert busy_reba > quiet_reba


def test_interval_activity_raises_the_reba_score():
    angles = _angles()
    _, steady_reba = _score(angles, _resolver())
    _, unstable_reba = _score(angles, _resolver(rapid_large_posture_change_or_unstable_base=True))

    assert unstable_reba > steady_reba


def test_an_entered_condition_still_overrides_the_measurement_in_the_score():
    """Precedence must survive all the way to the number, not just the record."""

    angles = _angles(trunk_twist=35.0)
    measured = _resolver()
    overridden = _resolver({"trunk_twisted": False, "trunk_twisted_or_side_flexed": False})

    measured_rula, _ = _score(angles, measured)
    overridden_rula, _ = _score(angles, overridden)

    assert measured.resolve(angles).sources["trunk_twisted"] == DERIVED
    assert overridden_rula < measured_rula


def test_scores_complete_without_any_manual_entry_at_all():
    """The headline promise: three videos in, scores out."""

    resolver = ConditionResolver(profile_name="standing_manual_handling")
    resolver.set_dynamics(_dynamics(**NEUTRAL_DYNAMICS))
    rula, reba = _score(_angles(), resolver)

    assert rula is not None
    assert reba is not None
