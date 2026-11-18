"""Named assumption profiles for unobserved task conditions.

RULA and REBA need observed task conditions that no pose model can supply:
whether a load is coupled well, whether the posture is held for a minute,
whether the arm is supported. This project refuses to invent them, which is
correct for an audit but means a run with nothing entered scores nothing at all
-- and the operator cannot tell, from a list of 34 field names, which ones
actually matter.

A profile is a named, documented set of values for those conditions, chosen to
suit a common kind of work. Applying one lets an assessment complete
immediately, and every field it filled is recorded, so an assumed condition is
never mistaken for an observed one. Anything the operator did supply always
wins: a profile only fills what is still unknown.

A profile is a screening default, not an observation. Results computed with one
are exposure estimates for triage, and any finding that matters should be
confirmed against the footage and the worker before it drives a change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Conditions a profile may supply. Anything outside this set is rejected, so a
# typo cannot quietly introduce a field the methods never read.
PROFILE_FIELDS = frozenset(
    {
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
        "trunk_twisted",
        "trunk_side_bent",
        "trunk_twisted_or_side_flexed",
        "neck_twisted",
        "neck_side_bent",
        "neck_twisted_or_side_flexed",
        "left_upper_arm_abducted",
        "right_upper_arm_abducted",
        "left_upper_arm_abducted_or_rotated",
        "right_upper_arm_abducted_or_rotated",
        "left_shoulder_raised",
        "right_shoulder_raised",
        "left_arm_supported_or_leaning",
        "right_arm_supported_or_leaning",
        "left_arm_supported_or_gravity_assisted",
        "right_arm_supported_or_gravity_assisted",
        "left_lower_arm_across_midline_or_out_to_side",
        "right_lower_arm_across_midline_or_out_to_side",
        "left_wrist_deviated",
        "right_wrist_deviated",
        "left_wrist_twist_at_end_range",
        "right_wrist_twist_at_end_range",
        "left_wrist_deviated_or_twisted",
        "right_wrist_deviated_or_twisted",
    }
)

# Conditions whose value changes the score most, and which vary most between
# tasks. Surfaced separately so the operator is told what to check first rather
# than being handed a list of every field.
HIGH_IMPACT_FIELDS = ("load_kg", "coupling", "repeated_more_than_four_per_minute")


@dataclass(frozen=True)
class AssumptionProfile:
    """A documented set of default task conditions for a kind of work."""

    name: str
    label: str
    description: str
    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.values) - PROFILE_FIELDS)
        if unknown:
            raise ValueError(f"profile {self.name!r} sets unsupported fields: {unknown}")


def _bilateral(prefix: str, value: Any) -> dict[str, Any]:
    return {f"left_{prefix}": value, f"right_{prefix}": value}


_ARMS_UNSUPPORTED_NEUTRAL: dict[str, Any] = {
    **_bilateral("upper_arm_abducted", False),
    **_bilateral("upper_arm_abducted_or_rotated", False),
    **_bilateral("shoulder_raised", False),
    **_bilateral("arm_supported_or_leaning", False),
    **_bilateral("arm_supported_or_gravity_assisted", False),
    **_bilateral("lower_arm_across_midline_or_out_to_side", False),
    **_bilateral("wrist_deviated", False),
    **_bilateral("wrist_twist_at_end_range", False),
    **_bilateral("wrist_deviated_or_twisted", False),
}

_TRUNK_NECK_NEUTRAL: dict[str, Any] = {
    "trunk_twisted": False,
    "trunk_side_bent": False,
    "trunk_twisted_or_side_flexed": False,
    "neck_twisted": False,
    "neck_side_bent": False,
    "neck_twisted_or_side_flexed": False,
}


PROFILES: dict[str, AssumptionProfile] = {
    "standing_manual_handling": AssumptionProfile(
        name="standing_manual_handling",
        label="Standing manual handling",
        description=(
            "Two-handed lifting or carrying on both feet, repeated through the shift. "
            "Assumes an 8 kg load with fair coupling and no shock loading. Confirm the "
            "actual load: it is the single input that moves the score most."
        ),
        values={
            **_ARMS_UNSUPPORTED_NEUTRAL,
            **_TRUNK_NECK_NEUTRAL,
            "load_kg": 8.0,
            "coupling": "fair",
            "rula_load_type": "static_or_repeated",
            "shock_or_rapid_force": False,
            "static_posture_over_one_minute": False,
            "repeated_more_than_four_per_minute": True,
            "rapid_large_posture_change_or_unstable_base": False,
            "seated": False,
            "legs_weight_bearing": "bilateral",
            "legs_supported_and_balanced": True,
        },
    ),
    "standing_light_work": AssumptionProfile(
        name="standing_light_work",
        label="Standing light work",
        description=(
            "Bench assembly, screw-driving or inspection while standing, with little or "
            "no load in the hands. Assumes good coupling and no repeated forceful exertion."
        ),
        values={
            **_ARMS_UNSUPPORTED_NEUTRAL,
            **_TRUNK_NECK_NEUTRAL,
            "load_kg": 0.0,
            "coupling": "good",
            "rula_load_type": "intermittent",
            "shock_or_rapid_force": False,
            "static_posture_over_one_minute": False,
            "repeated_more_than_four_per_minute": False,
            "rapid_large_posture_change_or_unstable_base": False,
            "seated": False,
            "legs_weight_bearing": "bilateral",
            "legs_supported_and_balanced": True,
        },
    ),
    "seated_workstation": AssumptionProfile(
        name="seated_workstation",
        label="Seated workstation",
        description=(
            "Seated desk or bench work with the forearms supported and no load in the "
            "hands. Assumes a sustained posture rather than repeated exertion."
        ),
        values={
            **_ARMS_UNSUPPORTED_NEUTRAL,
            **_TRUNK_NECK_NEUTRAL,
            **_bilateral("arm_supported_or_leaning", True),
            **_bilateral("arm_supported_or_gravity_assisted", True),
            "load_kg": 0.0,
            "coupling": "good",
            "rula_load_type": "static_or_repeated",
            "shock_or_rapid_force": False,
            "static_posture_over_one_minute": True,
            "repeated_more_than_four_per_minute": False,
            "rapid_large_posture_change_or_unstable_base": False,
            "seated": True,
            "legs_weight_bearing": "bilateral",
            "legs_supported_and_balanced": True,
        },
    ),
}


def available_profiles() -> list[AssumptionProfile]:
    """Profiles in a stable order, for presentation."""

    return [PROFILES[name] for name in sorted(PROFILES)]


def apply_profile(
    task_inputs: dict[str, Any],
    profile_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill unknown task conditions from a profile.

    Returns the merged inputs and a provenance record. Observed values always
    win: a profile only supplies fields that are absent or explicitly null, so
    entering one condition never causes another to be overwritten.
    """

    observed = {name: value for name, value in task_inputs.items() if value is not None}
    if not profile_name:
        return dict(task_inputs), {
            "profile": None,
            "assumed_fields": [],
            "observed_fields": sorted(observed),
            "confirm_first": [],
        }

    profile = PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"unknown assumption profile {profile_name!r}; available: {sorted(PROFILES)}"
        )

    merged = dict(task_inputs)
    assumed: list[str] = []
    for name, value in profile.values.items():
        if merged.get(name) is None:
            merged[name] = value
            assumed.append(name)

    return merged, {
        "profile": profile.name,
        "profile_label": profile.label,
        "profile_description": profile.description,
        "assumed_fields": sorted(assumed),
        "observed_fields": sorted(observed),
        # Named explicitly so the operator is told what to verify first rather
        # than being handed every field the profile touched.
        "confirm_first": sorted(name for name in HIGH_IMPACT_FIELDS if name in assumed),
    }
