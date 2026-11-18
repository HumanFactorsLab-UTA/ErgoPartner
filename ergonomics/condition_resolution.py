"""Resolve every RULA/REBA task condition from the best evidence available.

A condition can come from four places, in descending order of authority:

1. **observed** -- the operator entered it. They saw the workplace; a camera did
   not. An entered value always wins.
2. **derived** -- measured from this recording, either per frame
   (:mod:`core.posture_conditions`) or over the interval
   (:mod:`temporal.task_dynamics`).
3. **assumed** -- filled from a named profile in :mod:`ergonomics.profiles`
   because nothing else could supply it.
4. **unknown** -- nothing supplied it, and the method will withhold whatever
   depends on it.

Before this module existed, layer 2 did not: measurements the pipeline had
already made were discarded and the operator was asked to type 34 conditions by
hand, with a profile silently answering whichever were left. Adding the derived
layer moves most of those fields onto evidence from the footage and leaves only
what genuinely cannot be seen -- handled load, coupling quality, shock loading,
and whether an arm is resting on something -- to be entered or assumed.

Where an entered value and a measured one disagree, the entered value is used
and the disagreement is recorded, so a reviewer can see that the footage said
something else rather than having the conflict silently resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.angle_calculator import BodyAngles
from core.posture_conditions import (
    DERIVABLE_CONDITIONS as FRAME_CONDITIONS,
)
from core.posture_conditions import (
    ConditionThresholds,
    PostureBaseline,
    PostureConditionDeriver,
)
from ergonomics.profiles import PROFILE_FIELDS, PROFILES
from temporal.task_dynamics import (
    DERIVABLE_CONDITIONS as INTERVAL_CONDITIONS,
)
from temporal.task_dynamics import TaskDynamics

#: Every condition the methods may read. Each one appears in a resolution with
#: an explicit source, so "not present" never has to be distinguished from
#: "present but unknown" downstream.
ALL_CONDITIONS = frozenset(PROFILE_FIELDS | FRAME_CONDITIONS | INTERVAL_CONDITIONS)

OBSERVED = "observed"
DERIVED = "derived"
ASSUMED = "assumed"
UNKNOWN = "unknown"

#: Conditions no pose model can supply, whatever the footage quality. These are
#: the only fields a profile still needs to answer.
NOT_OBSERVABLE_FROM_VIDEO = frozenset(
    {
        # Mass in the hands is not recoverable from geometry.
        "load_kg",
        # Grip quality depends on the handle and the object, not the posture.
        "coupling",
        # Force build-up is a property of the load, not of the movement.
        "shock_or_rapid_force",
        # Whether an arm rests on a surface needs the surface to be modelled.
        "left_arm_supported_or_leaning",
        "right_arm_supported_or_leaning",
        "left_arm_supported_or_gravity_assisted",
        "right_arm_supported_or_gravity_assisted",
        # Forearm rotation is measured without an anatomical zero, so a
        # magnitude threshold on it is meaningless. See the note in
        # core/posture_conditions.py.
        "left_wrist_twist_at_end_range",
        "right_wrist_twist_at_end_range",
        # REBA's combined wrist field inherits that gap. A measured deviation
        # still sets it, so it is derived on those frames; on a straight wrist
        # it stays unknown because the twist half was never measured.
        "left_wrist_deviated_or_twisted",
        "right_wrist_deviated_or_twisted",
    }
)


@dataclass(frozen=True)
class ConditionConflict:
    """An entered condition that the footage measured differently."""

    condition: str
    observed_value: Any
    derived_value: Any
    basis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "observed_value": self.observed_value,
            "derived_value": self.derived_value,
            "measurement_basis": self.basis,
        }


@dataclass
class ResolvedConditions:
    """Task conditions for one frame, with where each value came from."""

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    conflicts: list[ConditionConflict] = field(default_factory=list)
    #: Per-condition measurement evidence: what was measured, the subject's
    #: neutral, and how far past it the frame went. Carried through so a reader
    #: can see why a penalty applied and not merely that it did.
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

    def by_source(self, source: str) -> list[str]:
        return sorted(name for name, value in self.sources.items() if value == source)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "sources": dict(self.sources),
            "observed_fields": self.by_source(OBSERVED),
            "derived_fields": self.by_source(DERIVED),
            "assumed_fields": self.by_source(ASSUMED),
            "unknown_fields": self.by_source(UNKNOWN),
            "conflicts": [item.to_dict() for item in self.conflicts],
            "evidence": dict(self.evidence),
        }


class ConditionResolver:
    """Merge entered, measured, and assumed task conditions in that order."""

    def __init__(
        self,
        manual_inputs: dict[str, Any] | None = None,
        profile_name: str | None = None,
        thresholds: ConditionThresholds | None = None,
        baseline: PostureBaseline | None = None,
    ) -> None:
        manual = dict(manual_inputs or {})
        manual.pop("assumption_profile", None)
        # Only values actually supplied count as observed; a null is an absence.
        self._manual = {name: value for name, value in manual.items() if value is not None}

        if profile_name is not None and profile_name not in PROFILES:
            raise ValueError(
                f"unknown assumption profile {profile_name!r}; available: {sorted(PROFILES)}"
            )
        self._profile = PROFILES[profile_name].values if profile_name else {}
        self._profile_name = profile_name
        self._deriver = PostureConditionDeriver(thresholds, baseline)
        self._baseline = baseline or PostureBaseline()
        self._dynamics: TaskDynamics | None = None

    # ------------------------------------------------------------------

    def set_baseline(self, baseline: PostureBaseline) -> None:
        """Supply the subject's resting levels once the sequence is known."""

        self._baseline = baseline
        self._deriver.baseline = baseline

    def set_dynamics(self, dynamics: TaskDynamics | None) -> None:
        """Supply the interval-level conditions once the sequence is known."""

        self._dynamics = dynamics

    @property
    def profile_name(self) -> str | None:
        return self._profile_name

    # ------------------------------------------------------------------

    def resolve(self, angles: BodyAngles) -> ResolvedConditions:
        """Resolve every condition for one frame's measurements."""

        derived_frame = self._deriver.derive(angles)
        derived: dict[str, Any] = dict(derived_frame.available)
        bases: dict[str, str] = {
            name: evidence.measurement for name, evidence in derived_frame.evidence.items()
        }

        if self._dynamics is not None:
            derived.update(self._dynamics.available)
            bases.update({name: item.basis for name, item in self._dynamics.evidence.items()})

        result = ResolvedConditions()
        result.evidence = {name: item.to_dict() for name, item in derived_frame.evidence.items()}
        names = ALL_CONDITIONS | set(self._manual) | set(derived) | set(self._profile)
        for name in sorted(names):
            manual_value = self._manual.get(name)
            derived_value = derived.get(name)

            if manual_value is not None:
                result.values[name] = manual_value
                result.sources[name] = OBSERVED
                if derived_value is not None and derived_value != manual_value:
                    result.conflicts.append(
                        ConditionConflict(
                            condition=name,
                            observed_value=manual_value,
                            derived_value=derived_value,
                            basis=bases.get(name),
                        )
                    )
            elif derived_value is not None:
                result.values[name] = derived_value
                result.sources[name] = DERIVED
            elif self._profile.get(name) is not None:
                result.values[name] = self._profile[name]
                result.sources[name] = ASSUMED
            else:
                result.values[name] = None
                result.sources[name] = UNKNOWN

        return result

    # ------------------------------------------------------------------

    def summary(self, sample: ResolvedConditions) -> dict[str, Any]:
        """Run-level provenance, built from one representative resolution.

        Which fields are observed or assumed is fixed for the whole run. Which
        are derived can vary frame to frame with landmark visibility, so the
        sample describes a frame rather than a guarantee.
        """

        payload = sample.to_dict()
        payload.pop("values", None)
        payload.pop("sources", None)
        payload["profile"] = self._profile_name
        if self._profile_name:
            profile = PROFILES[self._profile_name]
            payload["profile_label"] = profile.label
            payload["profile_description"] = profile.description
        payload["confirm_first"] = sorted(
            name for name in sample.by_source(ASSUMED) if name in NOT_OBSERVABLE_FROM_VIDEO
        )
        payload["not_observable_from_video"] = sorted(NOT_OBSERVABLE_FROM_VIDEO)
        payload["posture_baseline"] = self._baseline.describe()
        if self._dynamics is not None:
            payload["task_dynamics"] = self._dynamics.to_dict()
        return payload
