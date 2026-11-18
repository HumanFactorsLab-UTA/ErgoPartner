"""Derive RULA/REBA posture conditions from measurements the pipeline already makes.

RULA and REBA phrase several of their penalties as yes/no observations -- "trunk
twisted", "upper arm abducted", "wrist deviated". Historically this project asked
an operator to type all of them, one row per condition per side, which is where
the 34-field entry form came from.

Most of those conditions are not observations at all: they are thresholds on an
angle, and ``core/angle_calculator.py`` already measures every one of those
angles per frame. It computed ``trunk_twist``, ``upper_arm_abduction``,
``wrist_deviation``, ``wrist_pronation`` and ``forearm_midline_cross``, wrote
them to the CSV, and then the adapter asked the operator for the same facts and
ignored the measurement. This module closes that gap.

Three principles carried over from the rest of the scoring layer:

* **Unknown is not false.** When the angle a condition depends on could not be
  measured, the condition is ``None``, never ``False``. A missing measurement
  must not read as a verified-absent penalty.
* **Thresholds are declared, not buried.** Every cut-off lives in
  :class:`ConditionThresholds` with its justification, so a reviewer can see and
  change what "twisted" meant for a given assessment.
* **Provenance survives.** Each derived value records the measurement and margin
  it came from, so the dashboard can show *why* a penalty was applied.

The module is pure: it imports no video, model, or dashboard code, and depends
only on the :class:`~core.angle_calculator.BodyAngles` dataclass and numpy-free
arithmetic. That keeps it testable in isolation like the calculators themselves.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from core.angle_calculator import BodyAngles

# Conditions this module can supply. Anything outside the set is left to manual
# entry or an assumption profile, so a typo cannot silently introduce a field the
# methods never read.
DERIVABLE_CONDITIONS = frozenset(
    {
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


@dataclass(frozen=True)
class ConditionThresholds:
    """How far past the subject's own neutral a posture must go to be penalised.

    These are **departures from a measured baseline**, not absolute angles, and
    that distinction is what makes them work.

    Most of the measurements a condition rests on are rectified: ``trunk_twist``
    and ``trunk_side_bend`` come from an ``arccos``, ``neck_side_bend`` from an
    ``abs``. A rectified quantity cannot average its noise away -- every error
    pushes it up -- so its resting level is not zero. Measured on a subject
    standing at a bench, ``neck_side_bend`` sat at a median of 20 degrees,
    ``trunk_twist`` at 14, and ``trunk_side_bend`` at 12, with the subject
    plainly upright. Absolute thresholds of 10 to 15 degrees fell below those
    floors, so the conditions fired on most frames and pushed REBA to 9-12 on a
    task an assessor scores around 3.

    Comparing against the subject's own quiet level removes that floor, and with
    it the camera placement, the build of the person, and the model's per-joint
    bias. See :class:`PostureBaseline`.
    """

    # Axial rotation between the shoulder line and the hip line, past neutral.
    trunk_twist_deg: float = 20.0
    # Lateral flexion past neutral.
    trunk_side_bend_deg: float = 15.0
    # Head rotation relative to the shoulder line. The ear line is short against
    # landmark noise, so this needs more margin than the trunk.
    neck_rotation_deg: float = 20.0
    neck_side_bend_deg: float = 15.0
    # Upper-arm abduction. RULA's figure shows the arm clearly away from the
    # trunk, which is a large and unambiguous movement.
    upper_arm_abduction_deg: float = 25.0
    # One-sided shoulder elevation. See the limitation note on
    # ``left_shoulder_raised`` below -- this measurement cannot see a bilateral
    # shrug, so the threshold only governs the asymmetric case.
    shoulder_elevation_deg: float = 12.0
    # Radial/ulnar deviation. Functional range is roughly 20 deg radial and
    # 30 deg ulnar; the wrist is the noisiest joint the pose model reports.
    wrist_deviation_deg: float = 15.0
    # Pronation/supination "at or near end of range". Retained for callers that
    # set it, though the twist condition is no longer derived: see the note in
    # PostureConditionDeriver.derive.
    wrist_pronation_end_range_deg: float = 60.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"threshold {name!r} must be a non-negative number")


#: Measurements that are rectified -- an ``abs`` or an ``arccos`` -- so they
#: cannot go below zero and their noise accumulates upward. Their neutral is
#: estimated from a low percentile of the trial, which is where the subject was
#: quietest, rather than from the median, which sits above the noise floor.
RECTIFIED_MEASUREMENTS = frozenset(
    {
        "trunk_twist",
        "trunk_side_bend",
        "neck_rotation",
        "neck_side_bend",
        "left_upper_arm_abduction",
        "right_upper_arm_abduction",
        "left_shoulder_elevation",
        "right_shoulder_elevation",
    }
)


@dataclass(frozen=True)
class BaselinePolicy:
    """How the subject's neutral is estimated from a trial."""

    #: Percentile of a rectified measurement taken as its resting level. Low
    #: enough to sit under any real posture, high enough to clear single-frame
    #: dropouts.
    rectified_percentile: float = 20.0
    #: A signed measurement can go either way, so its habitual value is the
    #: median. Used for wrist deviation.
    signed_percentile: float = 50.0
    #: Frames of median filtering applied before a condition is tested. Landmark
    #: noise is frame-to-frame; a real posture change lasts longer than this.
    #: At the shipped 10 fps this is half a second.
    smoothing_frames: int = 5
    #: A baseline needs this many frames to mean anything. Below it the
    #: measurement is used as-is and the result says so.
    minimum_frames: int = 20

    def __post_init__(self) -> None:
        for name in ("rectified_percentile", "signed_percentile"):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f"{name} must be a percentile in [0, 100]")
        if self.smoothing_frames < 1 or self.smoothing_frames % 2 == 0:
            raise ValueError("smoothing_frames must be a positive odd number")
        if self.minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")


@dataclass(frozen=True)
class PostureBaseline:
    """One subject's resting level for each condition-driving measurement.

    Built from the whole trial, so a condition asks "is this posture unusual for
    this person, in this recording?" rather than "does this number exceed a
    constant?". That is closer to what an assessor does, and it is the only way
    to use a rectified measurement whose zero is not zero.

    A baseline cannot represent a posture held for the entire trial: a subject
    twisted throughout reads as neutral. That is a real limit, recorded in
    ``describe()`` so a reviewer sees the reference that was used.
    """

    offsets: dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    established: bool = False
    source: str = "trial_percentile"
    reference_id: str | None = None

    def neutral(self, measurement: str) -> float:
        return self.offsets.get(measurement, 0.0)

    def describe(self) -> dict[str, Any]:
        return {
            "established": self.established,
            "frame_count": self.frame_count,
            "source": self.source,
            "reference_id": self.reference_id,
            "neutral_deg": {name: round(value, 2) for name, value in sorted(self.offsets.items())},
            "note": {
                "absolute": "Condition thresholds use the measured angle relative to zero.",
                "reviewed_neutral": "The named neutral reference supplies offsets for listed measurements. Other measurements use zero.",
                "trial_percentile": (
                    "Conditions are tested as departures from these per-subject resting levels. "
                    "A posture held for the whole recording is absorbed into the baseline and "
                    "will not be flagged."
                ),
            }[self.source],
        }

    @classmethod
    def from_series(
        cls,
        series: Mapping[str, Sequence[float]],
        policy: BaselinePolicy | None = None,
    ) -> PostureBaseline:
        """Estimate each measurement's resting level from its own trial series."""

        policy = policy or BaselinePolicy()
        offsets: dict[str, float] = {}
        counts: list[int] = []
        for name, values in series.items():
            finite = [
                float(value)
                for value in values
                if value is not None and math.isfinite(float(value))
            ]
            counts.append(len(finite))
            if len(finite) < policy.minimum_frames:
                continue
            percentile = (
                policy.rectified_percentile
                if name in RECTIFIED_MEASUREMENTS
                else policy.signed_percentile
            )
            offsets[name] = float(np.percentile(finite, percentile))
        frames = max(counts) if counts else 0
        return cls(offsets=offsets, frame_count=frames, established=bool(offsets))

    @classmethod
    def for_recording(
        cls,
        series: Mapping[str, Sequence[float]],
        policy: BaselinePolicy,
        config: dict[str, Any] | None,
    ) -> PostureBaseline:
        """Resolve the CV reference independently of worksheet inputs and labels."""
        reference = (config or {}).get("neutral_reference")
        if reference is None:
            return cls.from_series(series, policy)
        if not isinstance(reference, dict):
            raise ValueError("posture_conditions.neutral_reference must be a mapping")
        source = reference.get("source")
        if source in ("absolute", "trial_percentile") and set(reference) == {"source"}:
            return (
                cls(source="absolute") if source == "absolute" else cls.from_series(series, policy)
            )
        if source != "reviewed_neutral" or set(reference) != {
            "source",
            "reference_id",
            "offsets_deg",
        }:
            raise ValueError(
                "Specify an absolute or trial_percentile source alone, or reviewed_neutral with reference_id and offsets_deg"
            )
        reference_id, offsets = reference["reference_id"], reference["offsets_deg"]
        if (
            not isinstance(reference_id, str)
            or not reference_id.strip()
            or not isinstance(offsets, dict)
            or not offsets
        ):
            raise ValueError(
                "Reviewed neutral requires an identified observation and measured offsets"
            )
        allowed = RECTIFIED_MEASUREMENTS | {"left_wrist_deviation", "right_wrist_deviation"}
        for name, value in offsets.items():
            if (
                name not in allowed
                or isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or abs(value) > 180
                or (name in RECTIFIED_MEASUREMENTS and value < 0)
            ):
                raise ValueError(f"Invalid neutral offset for {name!r}")
        return cls(
            offsets=dict(offsets), established=True, source=source, reference_id=reference_id
        )


def smooth_series(values: Sequence[float | None], window: int) -> list[float | None]:
    """Median-filter a measurement series, preserving gaps.

    A median filter is used rather than a mean because landmark noise arrives as
    occasional large spikes -- frame-to-frame jumps of 20 to 30 degrees on the
    wrist -- and a mean carries a spike into its neighbours while a median
    discards it.
    """

    if window <= 1:
        return list(values)
    half = window // 2
    output: list[float | None] = []
    for index, value in enumerate(values):
        if value is None or not math.isfinite(float(value)):
            output.append(None)
            continue
        neighbourhood = [
            float(item)
            for item in values[max(0, index - half) : index + half + 1]
            if item is not None and math.isfinite(float(item))
        ]
        output.append(float(np.median(neighbourhood)) if neighbourhood else None)
    return output


#: Conditions whose supporting measurement is known to be weaker than the
#: others, recorded so the dashboard can mark them rather than presenting every
#: derived value as equally solid.
LIMITED_CONFIDENCE_CONDITIONS = {
    "left_shoulder_raised": (
        "Shoulder elevation is measured as the rise of one shoulder against the other, so a "
        "bilateral shrug cancels to zero and is not detected. A one-sided shrug is detected."
    ),
    "right_shoulder_raised": (
        "Shoulder elevation is measured as the rise of one shoulder against the other, so a "
        "bilateral shrug cancels to zero and is not detected. A one-sided shrug is detected."
    ),
    "left_upper_arm_abducted_or_rotated": (
        "Abduction is measured; humeral rotation is not observable from pose landmarks, so a "
        "rotated-but-adducted arm reads as absent."
    ),
    "right_upper_arm_abducted_or_rotated": (
        "Abduction is measured; humeral rotation is not observable from pose landmarks, so a "
        "rotated-but-adducted arm reads as absent."
    ),
}


@dataclass(frozen=True)
class ConditionEvidence:
    """Why one condition took the value it did."""

    condition: str
    value: bool | None
    measurement: str
    measured_deg: float | None
    threshold_deg: float | None
    #: Signed distance past the threshold, in degrees. Negative means below it.
    margin_deg: float | None
    #: The subject's resting level for this measurement, subtracted before the
    #: threshold is applied. See PostureBaseline.
    neutral_deg: float | None = None
    #: How far past that resting level the measurement went.
    departure_deg: float | None = None
    limitation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "condition": self.condition,
            "value": self.value,
            "measurement": self.measurement,
            "measured_deg": self.measured_deg,
            "neutral_deg": self.neutral_deg,
            "departure_deg": self.departure_deg,
            "threshold_deg": self.threshold_deg,
            "margin_deg": self.margin_deg,
        }
        if self.limitation is not None:
            payload["limitation"] = self.limitation
        return payload


@dataclass
class DerivedConditions:
    """Conditions derived for one frame, with the evidence behind each."""

    values: dict[str, bool | None] = field(default_factory=dict)
    evidence: dict[str, ConditionEvidence] = field(default_factory=dict)

    @property
    def available(self) -> dict[str, bool]:
        """Only the conditions that were actually measurable this frame."""

        return {name: value for name, value in self.values.items() if value is not None}

    @property
    def unavailable(self) -> list[str]:
        return sorted(name for name, value in self.values.items() if value is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "unavailable": self.unavailable,
            "evidence": {name: item.to_dict() for name, item in self.evidence.items()},
        }


def _combine_or(first: bool | None, second: bool | None) -> bool | None:
    """Logical OR where either operand being true is decisive despite unknowns.

    REBA folds two penalties into one field ("twisted **or** side-flexed"). If
    one component is measured true the combined condition is true regardless of
    the other; if one is false and the other unknown the result stays unknown,
    because the unknown component could still trigger it.
    """

    if first is True or second is True:
        return True
    if first is None or second is None:
        return None
    return False


class PostureConditionDeriver:
    """Turn measured angles into the boolean conditions RULA and REBA consume."""

    def __init__(
        self,
        thresholds: ConditionThresholds | None = None,
        baseline: PostureBaseline | None = None,
    ) -> None:
        self.thresholds = thresholds or ConditionThresholds()
        self.baseline = baseline or PostureBaseline()

    # ------------------------------------------------------------------
    # Measurement access
    # ------------------------------------------------------------------

    @staticmethod
    def _measurement(angles: BodyAngles, attribute: str, unavailable_name: str) -> float | None:
        """Read an angle, returning None when the pipeline could not measure it."""

        if unavailable_name in set(angles.unavailable_measurements):
            return None
        value = getattr(angles, attribute, None)
        if value is None:
            return None
        numeric = float(value)
        # A non-finite angle is a failed computation, not a zero-degree posture.
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return None
        return numeric

    def _record(
        self,
        result: DerivedConditions,
        condition: str,
        measured: float | None,
        threshold: float,
        measurement: str,
    ) -> bool | None:
        """Test a measurement against the subject's neutral plus the threshold."""

        neutral = self.baseline.neutral(measurement)
        departure = None
        if measured is not None:
            # A smaller rectified angle is closer to neutral. Signed wrist
            # deviation can cross neutral and continue in the other direction.
            departure = (
                max(0.0, abs(measured) - neutral)
                if measurement in RECTIFIED_MEASUREMENTS
                else abs(measured - neutral)
            )
        value = None if departure is None else departure >= threshold
        result.values[condition] = value
        result.evidence[condition] = ConditionEvidence(
            condition=condition,
            value=value,
            measurement=measurement,
            measured_deg=measured,
            neutral_deg=neutral,
            departure_deg=departure,
            threshold_deg=threshold,
            margin_deg=None if departure is None else departure - threshold,
            limitation=LIMITED_CONFIDENCE_CONDITIONS.get(condition),
        )
        return value

    def _record_boolean(
        self,
        result: DerivedConditions,
        condition: str,
        value: bool | None,
        measurement: str,
    ) -> bool | None:
        result.values[condition] = value
        result.evidence[condition] = ConditionEvidence(
            condition=condition,
            value=value,
            measurement=measurement,
            measured_deg=None,
            threshold_deg=None,
            margin_deg=None,
            limitation=LIMITED_CONFIDENCE_CONDITIONS.get(condition),
        )
        return value

    def _record_combined(
        self,
        result: DerivedConditions,
        condition: str,
        value: bool | None,
        sources: tuple[str, ...],
    ) -> None:
        result.values[condition] = value
        result.evidence[condition] = ConditionEvidence(
            condition=condition,
            value=value,
            measurement=" or ".join(sources),
            measured_deg=None,
            threshold_deg=None,
            margin_deg=None,
            limitation=LIMITED_CONFIDENCE_CONDITIONS.get(condition),
        )

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def derive(self, angles: BodyAngles) -> DerivedConditions:
        """Derive every supportable condition for one frame's measurements."""

        limits = self.thresholds
        result = DerivedConditions()

        # ---- Trunk ----
        trunk_twist = self._measurement(angles, "trunk_twist", "trunk_twist_deg")
        trunk_side = self._measurement(angles, "trunk_side_bend", "trunk_side_bend_deg")
        twisted = self._record(
            result, "trunk_twisted", trunk_twist, limits.trunk_twist_deg, "trunk_twist"
        )
        side_bent = self._record(
            result,
            "trunk_side_bent",
            trunk_side,
            limits.trunk_side_bend_deg,
            "trunk_side_bend",
        )
        self._record_combined(
            result,
            "trunk_twisted_or_side_flexed",
            _combine_or(twisted, side_bent),
            ("trunk_twist", "trunk_side_bend"),
        )

        # ---- Neck ----
        neck_rotation = self._measurement(angles, "neck_rotation", "neck_rotation_deg")
        neck_side = self._measurement(angles, "neck_side_bend", "neck_side_bend_deg")
        neck_twisted = self._record(
            result, "neck_twisted", neck_rotation, limits.neck_rotation_deg, "neck_rotation"
        )
        neck_side_bent = self._record(
            result, "neck_side_bent", neck_side, limits.neck_side_bend_deg, "neck_side_bend"
        )
        self._record_combined(
            result,
            "neck_twisted_or_side_flexed",
            _combine_or(neck_twisted, neck_side_bent),
            ("neck_rotation", "neck_side_bend"),
        )

        # ---- Per-side arm and wrist ----
        for side in ("left", "right"):
            abduction = self._measurement(
                angles, f"{side}_upper_arm_abduction", f"{side}_upper_arm_abduction_deg"
            )
            abducted = self._record(
                result,
                f"{side}_upper_arm_abducted",
                abduction,
                limits.upper_arm_abduction_deg,
                f"{side}_upper_arm_abduction",
            )
            # REBA folds abduction and humeral rotation into one field. Rotation
            # is not observable from pose landmarks, so abduction carries it and
            # the shortfall is recorded in LIMITED_CONFIDENCE_CONDITIONS.
            self._record_boolean(
                result,
                f"{side}_upper_arm_abducted_or_rotated",
                abducted,
                f"{side}_upper_arm_abduction",
            )

            elevation = self._measurement(
                angles, f"{side}_shoulder_elevation", f"{side}_shoulder_elevation_deg"
            )
            self._record(
                result,
                f"{side}_shoulder_raised",
                elevation,
                limits.shoulder_elevation_deg,
                f"{side}_shoulder_elevation",
            )

            # Already a boolean from the angle calculator's midline test.
            midline_available = f"{side}_forearm_midline_cross" not in set(
                angles.unavailable_measurements
            )
            self._record_boolean(
                result,
                f"{side}_lower_arm_across_midline_or_out_to_side",
                bool(getattr(angles, f"{side}_forearm_midline_cross"))
                if midline_available
                else None,
                f"{side}_forearm_midline_cross",
            )

            deviation = self._measurement(
                angles, f"{side}_wrist_deviation", f"{side}_wrist_deviation_deg"
            )
            deviated = self._record(
                result,
                f"{side}_wrist_deviated",
                deviation,
                limits.wrist_deviation_deg,
                f"{side}_wrist_deviation",
            )

            # Wrist twist is deliberately not derived. The pronation measurement
            # is an arctan2 of the palm normal about the forearm axis with no
            # anatomical zero, so it sweeps the full -180..180 circle: measured
            # on real footage it fills that whole range, while physiological
            # forearm rotation spans roughly +/-85 degrees. A magnitude
            # threshold on an uncalibrated circular quantity would report "twist
            # at end range" on a large share of ordinary frames, and that is a
            # real added penalty in RULA. Establishing a neutral reference needs
            # a calibration posture this pipeline does not capture, so the
            # condition is left to observation or a profile.
            self._record_boolean(
                result,
                f"{side}_wrist_twist_at_end_range",
                None,
                f"{side}_wrist_pronation (no anatomical zero; not derived)",
            )

            # The REBA combined field therefore rests on deviation alone. Twist
            # is unknown rather than absent, so a measured deviation still earns
            # the penalty but a neutral wrist stays unknown.
            self._record_combined(
                result,
                f"{side}_wrist_deviated_or_twisted",
                _combine_or(deviated, None),
                (f"{side}_wrist_deviation",),
            )

        return result


def derive_posture_conditions(
    angles: BodyAngles, thresholds: ConditionThresholds | None = None
) -> DerivedConditions:
    """Convenience wrapper for one-off derivation."""

    return PostureConditionDeriver(thresholds).derive(angles)


def apply_condition_smoothing(
    sequence: Sequence[BodyAngles],
    measurements: Sequence[str],
    window: int,
) -> None:
    """Median-filter the condition-driving angles across a sequence, in place.

    Only the angles a condition is thresholded against are touched. Flexion
    angles are left exactly as measured, because they are reported to the
    reviewer and feed the score bands directly; smoothing those would change
    published numbers rather than stabilise a yes/no decision.
    """

    if window <= 1 or len(sequence) < 2:
        return
    for name in measurements:
        raw = [
            None if f"{name}_deg" in item.unavailable_measurements else getattr(item, name, None)
            for item in sequence
        ]
        if all(value is None for value in raw):
            continue
        for item, value in zip(sequence, smooth_series(raw, window), strict=True):
            if value is not None:
                setattr(item, name, value)


def baseline_policy_from_config(config: dict[str, Any] | None) -> BaselinePolicy:
    """Build a baseline policy from the ``posture_conditions.baseline`` block."""

    defaults = BaselinePolicy()
    if not config:
        return defaults
    baseline = config.get("baseline") if isinstance(config, dict) else None
    if not baseline:
        return defaults
    if not isinstance(baseline, dict):
        raise ValueError("posture_conditions.baseline must be a mapping")
    known = set(vars(defaults))
    unknown = sorted(set(baseline) - known)
    if unknown:
        raise ValueError(
            f"unknown posture baseline settings: {unknown}; available: {sorted(known)}"
        )
    values: dict[str, Any] = {
        name: int(value) if name.endswith("_frames") else float(value)
        for name, value in baseline.items()
    }
    return replace(defaults, **values)


def thresholds_from_config(config: dict[str, Any] | None) -> ConditionThresholds:
    """Build thresholds from the ``posture_conditions.thresholds`` config block.

    Unspecified entries keep their documented default, so a config need only
    name the cut-offs it wants to move.
    """

    defaults = ConditionThresholds()
    if not config:
        return defaults
    if "thresholds" in config or "baseline" in config or "neutral_reference" in config:
        thresholds = config.get("thresholds") or {}
    else:
        thresholds = config
    if not isinstance(thresholds, dict):
        raise ValueError("posture_conditions.thresholds must be a mapping")
    known = set(vars(defaults))
    unknown = sorted(set(thresholds) - known)
    if unknown:
        raise ValueError(
            f"unknown posture condition thresholds: {unknown}; available: {sorted(known)}"
        )
    return replace(defaults, **{name: float(value) for name, value in thresholds.items()})
