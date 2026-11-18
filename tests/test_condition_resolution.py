"""Precedence and provenance when merging entered, measured, and assumed conditions."""

from __future__ import annotations

import pytest

from core.angle_calculator import BodyAngles
from ergonomics.condition_resolution import (
    ASSUMED,
    DERIVED,
    NOT_OBSERVABLE_FROM_VIDEO,
    OBSERVED,
    UNKNOWN,
    ConditionResolver,
)
from temporal.task_dynamics import DynamicsEvidence, TaskDynamics


def _angles(**overrides) -> BodyAngles:
    angles = BodyAngles()
    for name, value in overrides.items():
        setattr(angles, name, value)
    return angles


def _dynamics(**values) -> TaskDynamics:
    dynamics = TaskDynamics()
    for name, value in values.items():
        dynamics.values[name] = value
        dynamics.evidence[name] = DynamicsEvidence(condition=name, value=value, basis="test")
    return dynamics


# ----------------------------------------------------------------------
# Precedence
# ----------------------------------------------------------------------


def test_measured_conditions_are_used_when_nothing_was_entered():
    resolver = ConditionResolver()
    result = resolver.resolve(_angles(trunk_twist=40.0))

    assert result.values["trunk_twisted"] is True
    assert result.sources["trunk_twisted"] == DERIVED


def test_entered_value_beats_the_measurement():
    """The operator saw the workplace; the camera saw one angle."""

    resolver = ConditionResolver({"trunk_twisted": True})
    result = resolver.resolve(_angles(trunk_twist=0.0))

    assert result.values["trunk_twisted"] is True
    assert result.sources["trunk_twisted"] == OBSERVED


def test_disagreement_between_entered_and_measured_is_recorded():
    resolver = ConditionResolver({"trunk_twisted": False})
    result = resolver.resolve(_angles(trunk_twist=40.0))

    assert result.values["trunk_twisted"] is False
    assert [item.condition for item in result.conflicts] == ["trunk_twisted"]
    conflict = result.conflicts[0]
    assert conflict.observed_value is False
    assert conflict.derived_value is True
    assert conflict.basis == "trunk_twist"


def test_agreement_between_entered_and_measured_is_not_a_conflict():
    resolver = ConditionResolver({"trunk_twisted": True})
    result = resolver.resolve(_angles(trunk_twist=40.0))

    assert result.conflicts == []


def test_measurement_beats_the_profile_assumption():
    """A profile is a stand-in for evidence, so real evidence displaces it."""

    resolver = ConditionResolver(profile_name="standing_manual_handling")
    result = resolver.resolve(_angles(trunk_twist=40.0))

    assert result.values["trunk_twisted"] is True
    assert result.sources["trunk_twisted"] == DERIVED


def test_profile_fills_only_what_video_cannot_supply():
    resolver = ConditionResolver(profile_name="standing_manual_handling")
    result = resolver.resolve(_angles())

    assert result.sources["load_kg"] == ASSUMED
    assert result.sources["coupling"] == ASSUMED
    assert result.sources["left_arm_supported_or_leaning"] == ASSUMED
    # Everything measurable came from the footage instead.
    assert result.sources["trunk_twisted"] == DERIVED
    assert result.sources["left_wrist_deviated"] == DERIVED


def test_without_a_profile_unobservable_conditions_stay_unknown():
    resolver = ConditionResolver()
    result = resolver.resolve(_angles())

    assert result.values["load_kg"] is None
    assert result.sources["load_kg"] == UNKNOWN
    assert "load_kg" in result.by_source(UNKNOWN)


def test_unmeasurable_angle_falls_through_to_the_profile():
    resolver = ConditionResolver(profile_name="standing_manual_handling")
    angles = _angles(unavailable_measurements=["trunk_twist_deg"])
    result = resolver.resolve(angles)

    assert result.sources["trunk_twisted"] == ASSUMED


def test_unmeasurable_angle_without_a_profile_is_unknown():
    resolver = ConditionResolver()
    angles = _angles(unavailable_measurements=["trunk_twist_deg"])
    result = resolver.resolve(angles)

    assert result.values["trunk_twisted"] is None
    assert result.sources["trunk_twisted"] == UNKNOWN


def test_explicit_null_is_an_absence_not_an_observation():
    resolver = ConditionResolver({"trunk_twisted": None})
    result = resolver.resolve(_angles(trunk_twist=40.0))

    assert result.sources["trunk_twisted"] == DERIVED


def test_false_is_a_real_observation_and_is_kept():
    """A deliberate No must not be mistaken for an unanswered field."""

    resolver = ConditionResolver({"shock_or_rapid_force": False})
    result = resolver.resolve(_angles())

    assert result.values["shock_or_rapid_force"] is False
    assert result.sources["shock_or_rapid_force"] == OBSERVED


def test_zero_load_is_a_real_observation():
    resolver = ConditionResolver({"load_kg": 0.0}, profile_name="standing_manual_handling")
    result = resolver.resolve(_angles())

    assert result.values["load_kg"] == 0.0
    assert result.sources["load_kg"] == OBSERVED


# ----------------------------------------------------------------------
# Interval-level conditions
# ----------------------------------------------------------------------


def test_dynamics_supply_the_interval_conditions():
    resolver = ConditionResolver()
    resolver.set_dynamics(
        _dynamics(
            seated=False,
            repeated_more_than_four_per_minute=True,
            legs_weight_bearing="bilateral",
        )
    )
    result = resolver.resolve(_angles())

    assert result.values["repeated_more_than_four_per_minute"] is True
    assert result.sources["repeated_more_than_four_per_minute"] == DERIVED
    assert result.values["legs_weight_bearing"] == "bilateral"


def test_dynamics_beat_the_profile_but_lose_to_an_entered_value():
    resolver = ConditionResolver({"seated": True}, profile_name="standing_manual_handling")
    resolver.set_dynamics(_dynamics(seated=False, legs_weight_bearing="walking"))
    result = resolver.resolve(_angles())

    assert result.values["seated"] is True
    assert result.sources["seated"] == OBSERVED
    assert result.values["legs_weight_bearing"] == "walking"
    assert result.sources["legs_weight_bearing"] == DERIVED


def test_conditions_resolve_before_dynamics_are_known():
    """Scoring must still work if the interval analysis produced nothing."""

    resolver = ConditionResolver(profile_name="standing_manual_handling")
    result = resolver.resolve(_angles())

    assert result.sources["seated"] == ASSUMED


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def test_summary_lists_provenance_and_what_to_confirm():
    resolver = ConditionResolver(profile_name="standing_manual_handling")
    sample = resolver.resolve(_angles())
    summary = resolver.summary(sample)

    assert summary["profile"] == "standing_manual_handling"
    assert "load_kg" in summary["assumed_fields"]
    assert "trunk_twisted" in summary["derived_fields"]
    # The operator is pointed only at what the footage genuinely cannot answer.
    assert set(summary["confirm_first"]) <= NOT_OBSERVABLE_FROM_VIDEO
    assert "load_kg" in summary["confirm_first"]
    assert "trunk_twisted" not in summary["confirm_first"]
    # Raw per-frame values do not belong in a run-level summary.
    assert "values" not in summary


def test_summary_includes_dynamics_evidence_when_available():
    resolver = ConditionResolver()
    resolver.set_dynamics(_dynamics(seated=False))
    summary = resolver.summary(resolver.resolve(_angles()))

    assert summary["task_dynamics"]["values"]["seated"] is False


def test_derived_fields_cover_most_of_the_old_manual_form():
    """The point of the change: the entry burden collapses to a handful."""

    resolver = ConditionResolver(profile_name="standing_manual_handling")
    resolver.set_dynamics(
        _dynamics(
            seated=False,
            legs_weight_bearing="bilateral",
            legs_supported_and_balanced=True,
            static_posture_over_one_minute=False,
            repeated_more_than_four_per_minute=True,
            rapid_large_posture_change_or_unstable_base=False,
            rula_load_type="static_or_repeated",
        )
    )
    result = resolver.resolve(_angles())

    assert len(result.by_source(DERIVED)) >= 23
    # Only genuinely unobservable fields are left to the profile.
    assert set(result.by_source(ASSUMED)) <= NOT_OBSERVABLE_FROM_VIDEO


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown assumption profile"):
        ConditionResolver(profile_name="not_a_profile")


def test_assumption_profile_key_is_not_treated_as_a_condition():
    resolver = ConditionResolver({"assumption_profile": "standing_manual_handling"})
    result = resolver.resolve(_angles())

    assert "assumption_profile" not in result.values
