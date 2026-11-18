"""Assumption profiles fill unknown task conditions without hiding that they did."""

import pytest

from ergonomics.profiles import (
    PROFILES,
    AssumptionProfile,
    apply_profile,
    available_profiles,
)


def test_every_profile_completes_rula_and_reba_inputs():
    """A profile that leaves a required condition unset would not unblock scoring."""

    required = {
        "load_kg",
        "coupling",
        "rula_load_type",
        "shock_or_rapid_force",
        "static_posture_over_one_minute",
        "repeated_more_than_four_per_minute",
        "rapid_large_posture_change_or_unstable_base",
        "seated",
        "legs_weight_bearing",
        "legs_supported_and_balanced",
    }
    for profile in available_profiles():
        merged, _ = apply_profile({}, profile.name)
        missing = sorted(name for name in required if merged.get(name) is None)
        assert not missing, f"{profile.name} leaves {missing} unset"


def test_observed_values_are_never_overwritten():
    merged, provenance = apply_profile(
        {"load_kg": 25.0, "coupling": "poor"}, "standing_manual_handling"
    )
    assert merged["load_kg"] == 25.0
    assert merged["coupling"] == "poor"
    assert "load_kg" not in provenance["assumed_fields"]
    assert "coupling" not in provenance["assumed_fields"]
    assert provenance["observed_fields"] == ["coupling", "load_kg"]


def test_explicit_null_is_still_treated_as_unknown():
    """A null in config means unknown, so a profile may fill it."""

    merged, provenance = apply_profile({"load_kg": None}, "standing_manual_handling")
    assert merged["load_kg"] == 8.0
    assert "load_kg" in provenance["assumed_fields"]
    assert provenance["observed_fields"] == []


def test_no_profile_assumes_nothing():
    merged, provenance = apply_profile({"load_kg": None}, None)
    assert merged["load_kg"] is None
    assert provenance["profile"] is None
    assert provenance["assumed_fields"] == []


def test_provenance_names_the_high_impact_assumptions_first():
    _, provenance = apply_profile({}, "standing_manual_handling")
    assert provenance["confirm_first"] == [
        "coupling",
        "load_kg",
        "repeated_more_than_four_per_minute",
    ]


def test_supplying_a_high_impact_value_removes_it_from_confirm_first():
    _, provenance = apply_profile({"load_kg": 12.0}, "standing_manual_handling")
    assert "load_kg" not in provenance["confirm_first"]


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown assumption profile"):
        apply_profile({}, "does_not_exist")


def test_profile_cannot_set_a_field_the_methods_do_not_read():
    with pytest.raises(ValueError, match="unsupported fields"):
        AssumptionProfile(name="bad", label="Bad", description="", values={"not_a_field": True})


def test_seated_profile_differs_from_standing_where_it_should():
    seated, _ = apply_profile({}, "seated_workstation")
    standing, _ = apply_profile({}, "standing_manual_handling")
    assert seated["seated"] is True
    assert standing["seated"] is False
    assert seated["left_arm_supported_or_leaning"] is True
    assert standing["left_arm_supported_or_leaning"] is False


def test_profiles_are_documented():
    for profile in available_profiles():
        assert profile.label
        assert len(profile.description) > 40, f"{profile.name} needs a usable description"


def test_available_profiles_is_stable_and_complete():
    names = [profile.name for profile in available_profiles()]
    assert names == sorted(PROFILES)
