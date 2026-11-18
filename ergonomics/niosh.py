"""Revised NIOSH Lifting Equation (RNLE), single-task implementation.

Source of truth: NIOSH Applications Manual for the Revised NIOSH Lifting
Equation, DHHS (NIOSH) Publication 94-110 (September 1994).

No physical distance, load, coupling, duration, or task-applicability value is
inferred here. Missing information produces an incomplete result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from ergonomics.common import (
    CALIBRATION_REQUIRED,
    MISSING_REQUIRED_TASK_INPUT,
    UNSUPPORTED_RNLE_TASK,
    VALID,
    VALID_WITH_LIMITATIONS,
    AssessmentContext,
    validate_finite_number,
)

METHOD_VERSION = "Revised NIOSH Lifting Equation, NIOSH 94-110"


FM_TABLE = {
    0.2: {"short": (1.00, 1.00), "moderate": (0.95, 0.95), "long": (0.85, 0.85)},
    0.5: {"short": (0.97, 0.97), "moderate": (0.92, 0.92), "long": (0.81, 0.81)},
    1.0: {"short": (0.94, 0.94), "moderate": (0.88, 0.88), "long": (0.75, 0.75)},
    2.0: {"short": (0.91, 0.91), "moderate": (0.84, 0.84), "long": (0.65, 0.65)},
    3.0: {"short": (0.88, 0.88), "moderate": (0.79, 0.79), "long": (0.55, 0.55)},
    4.0: {"short": (0.84, 0.84), "moderate": (0.72, 0.72), "long": (0.45, 0.45)},
    5.0: {"short": (0.80, 0.80), "moderate": (0.60, 0.60), "long": (0.35, 0.35)},
    6.0: {"short": (0.75, 0.75), "moderate": (0.50, 0.50), "long": (0.27, 0.27)},
    7.0: {"short": (0.70, 0.70), "moderate": (0.42, 0.42), "long": (0.22, 0.22)},
    8.0: {"short": (0.60, 0.60), "moderate": (0.35, 0.35), "long": (0.18, 0.18)},
    9.0: {"short": (0.52, 0.52), "moderate": (0.30, 0.30), "long": (0.00, 0.15)},
    10.0: {"short": (0.45, 0.45), "moderate": (0.26, 0.26), "long": (0.00, 0.13)},
    11.0: {"short": (0.41, 0.41), "moderate": (0.00, 0.23), "long": (0.00, 0.00)},
    12.0: {"short": (0.37, 0.37), "moderate": (0.00, 0.21), "long": (0.00, 0.00)},
    13.0: {"short": (0.00, 0.34), "moderate": (0.00, 0.00), "long": (0.00, 0.00)},
    14.0: {"short": (0.00, 0.31), "moderate": (0.00, 0.00), "long": (0.00, 0.00)},
    15.0: {"short": (0.00, 0.28), "moderate": (0.00, 0.00), "long": (0.00, 0.00)},
}
CM_TABLE = {
    "good": (1.00, 1.00),
    "fair": (0.95, 1.00),
    "poor": (0.90, 0.90),
}


@dataclass(frozen=True)
class NIOSHTaskApplicability:
    """Manual exclusions that video alone normally cannot establish."""

    one_handed: bool | None = None
    lifting_over_eight_hours: bool | None = None
    seated_or_kneeling: bool | None = None
    restricted_workspace: bool | None = None
    unstable_load: bool | None = None
    carrying_pushing_or_pulling: bool | None = None
    wheelbarrow_or_shoveling: bool | None = None
    high_speed_motion: bool | None = None
    insufficient_foot_floor_coupling: bool | None = None
    unfavorable_temperature_or_humidity: bool | None = None
    non_lifting_work_exceeds_ten_percent: bool | None = None
    multiple_task_job: bool | None = None

    @classmethod
    def reviewed_applicable(cls) -> NIOSHTaskApplicability:
        """Construct only after every listed exclusion was reviewed as absent."""

        return cls(**{item.name: False for item in fields(cls)})

    def unsupported_conditions(self) -> list[str]:
        return [name for name, active in asdict(self).items() if active is True]

    def unreviewed_conditions(self) -> list[str]:
        return [name for name, active in asdict(self).items() if active is None]

    def invalid_conditions(self) -> list[str]:
        return [
            name
            for name, active in asdict(self).items()
            if active is not None and not isinstance(active, bool)
        ]


#: Where a physical RNLE distance came from.
#:
#: ``manual_measurement``    -- measured on site with a tape or rule.
#: ``calibrated_video``      -- from video with a calibrated pixel scale; requires
#:                             ``calibration_status == "calibrated"``.
#: ``anthropometric_estimate`` -- scaled from the participant's recorded stature
#:                             against the pose model's body proportions. Needs no
#:                             camera calibration and is accepted, but every result
#:                             carries a warning: it is a screening estimate whose
#:                             error grows with unusual body proportions, loose
#:                             clothing, and oblique camera angles.
DISTANCE_MEASUREMENT_SOURCES = frozenset(
    {"manual_measurement", "calibrated_video", "anthropometric_estimate"}
)


@dataclass(frozen=True)
class NIOSHInput:
    horizontal_cm: float | None = None
    vertical_cm: float | None = None
    travel_distance_cm: float | None = None
    asymmetry_deg: float | None = None
    frequency_lifts_per_min: float | None = None
    duration: str | None = None
    coupling: str | None = None
    load_weight_kg: float | None = None
    applicability: NIOSHTaskApplicability | None = None
    location: str = "origin"
    #: How the physical distances were obtained. See DISTANCE_MEASUREMENT_SOURCES.
    distance_measurement_source: str = "manual_measurement"
    calibration_status: str = "not_applicable"
    context: AssessmentContext = field(default_factory=AssessmentContext)

    @classmethod
    def from_imperial(
        cls,
        *,
        horizontal_in: float | None,
        vertical_in: float | None,
        travel_distance_in: float | None,
        asymmetry_deg: float | None,
        frequency_lifts_per_min: float | None,
        duration: str | None,
        coupling: str | None,
        load_weight_lb: float | None,
        applicability: NIOSHTaskApplicability | None,
        location: str = "origin",
        context: AssessmentContext | None = None,
    ) -> NIOSHInput:
        """Create a canonical metric input using exact SI conversion factors."""

        return cls(
            horizontal_cm=None if horizontal_in is None else horizontal_in * 2.54,
            vertical_cm=None if vertical_in is None else vertical_in * 2.54,
            travel_distance_cm=None if travel_distance_in is None else travel_distance_in * 2.54,
            asymmetry_deg=asymmetry_deg,
            frequency_lifts_per_min=frequency_lifts_per_min,
            duration=duration,
            coupling=coupling,
            load_weight_kg=None if load_weight_lb is None else load_weight_lb * 0.45359237,
            applicability=applicability,
            location=location,
            distance_measurement_source="manual_measurement",
            calibration_status="not_applicable",
            context=context or AssessmentContext(),
        )


@dataclass(frozen=True)
class NIOSHResult:
    assessment_type: str
    method_version: str
    location: str
    recommended_weight_limit_kg: float | None
    lifting_index: float | None
    risk_level: str | None
    action_level: str | None
    validity_status: str
    multipliers: dict[str, float]
    raw_measurements: dict[str, Any]
    units: dict[str, str]
    missing_inputs: list[str]
    invalid_inputs: list[str]
    unsupported_conditions: list[str]
    warnings: list[str]
    context: AssessmentContext

    @property
    def rwl(self) -> float | None:
        return self.recommended_weight_limit_kg

    @property
    def LC(self) -> float | None:
        return self.multipliers.get("LC")

    def __getattr__(self, name: str) -> Any:
        if name == "load_weight_kg":
            return self.raw_measurements.get(name)
        if name in {"HM", "VM", "DM", "AM", "FM", "CM"}:
            return self.multipliers.get(name)
        old_names = {
            "H": "horizontal_cm",
            "V": "vertical_cm",
            "D": "travel_distance_cm",
            "A": "asymmetry_deg",
            "F": "frequency_lifts_per_min",
        }
        if name in old_names:
            return self.raw_measurements.get(old_names[name])
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NIOSHLiftResult:
    """Origin plus optional destination result for a single lift."""

    origin: NIOSHResult
    destination: NIOSHResult | None
    controlling_location: str | None
    recommended_weight_limit_kg: float | None
    lifting_index: float | None
    validity_status: str
    selection_reason: str


class NIOSHCalculator:
    LC = 23.0
    REQUIRED_FIELDS = (
        "horizontal_cm",
        "vertical_cm",
        "travel_distance_cm",
        "asymmetry_deg",
        "frequency_lifts_per_min",
        "duration",
        "coupling",
        "load_weight_kg",
        "applicability",
    )

    @staticmethod
    def _frequency_multiplier(frequency: float, duration: str, vertical_cm: float) -> float:
        keys = sorted(FM_TABLE)
        vertical_index = 0 if vertical_cm < 75 else 1
        if frequency <= keys[0]:
            return FM_TABLE[keys[0]][duration][vertical_index]
        if frequency >= keys[-1]:
            return FM_TABLE[keys[-1]][duration][vertical_index]
        low = max(key for key in keys if key <= frequency)
        high = min(key for key in keys if key >= frequency)
        if low == high:
            return FM_TABLE[low][duration][vertical_index]
        low_value = FM_TABLE[low][duration][vertical_index]
        high_value = FM_TABLE[high][duration][vertical_index]
        fraction = (frequency - low) / (high - low)
        return low_value + fraction * (high_value - low_value)

    @staticmethod
    def _risk(lifting_index: float) -> tuple[str, str]:
        if lifting_index <= 1:
            return (
                "At or below RWL",
                "The load does not exceed the RWL; retain the RNLE applicability and worker-population limitations.",
            )
        return (
            "Exceeds RWL",
            "The load exceeds the RWL; evaluate and redesign the lifting task.",
        )

    def calculate(self, inp: NIOSHInput) -> NIOSHResult:
        raw = asdict(inp)
        raw.pop("context")
        missing = [name for name in self.REQUIRED_FIELDS if getattr(inp, name) is None]
        invalid: list[str] = []
        unsupported = inp.applicability.unsupported_conditions() if inp.applicability else []
        warnings: list[str] = []
        if inp.applicability is not None:
            missing.extend(
                f"applicability.{name}" for name in inp.applicability.unreviewed_conditions()
            )
            invalid.extend(
                f"applicability.{name}" for name in inp.applicability.invalid_conditions()
            )
        numeric_fields = (
            "horizontal_cm",
            "vertical_cm",
            "travel_distance_cm",
            "asymmetry_deg",
            "frequency_lifts_per_min",
            "load_weight_kg",
        )
        for name in numeric_fields:
            value = getattr(inp, name)
            if value is not None:
                validate_finite_number(name, value, invalid)
        if inp.duration is not None and inp.duration not in {"short", "moderate", "long"}:
            invalid.append("duration")
        if inp.coupling is not None and inp.coupling not in CM_TABLE:
            invalid.append("coupling")
        if inp.location not in {"origin", "destination"}:
            invalid.append("location")
        if inp.distance_measurement_source not in DISTANCE_MEASUREMENT_SOURCES:
            invalid.append("distance_measurement_source")
        if inp.distance_measurement_source == "anthropometric_estimate":
            warnings.append(
                "Physical distances were scaled from the participant's stature and the pose "
                "model's body proportions, not measured with a tape or a calibrated camera. "
                "Treat the RWL as a screening estimate and confirm any distance that drives a "
                "decision."
            )

        if (
            inp.distance_measurement_source == "calibrated_video"
            and inp.calibration_status != "calibrated"
        ):
            for field_name in ("horizontal_cm", "vertical_cm", "travel_distance_cm"):
                if field_name not in missing:
                    missing.append(field_name)
            return NIOSHResult(
                assessment_type="RNLE",
                method_version=METHOD_VERSION,
                location=inp.location,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                risk_level=None,
                action_level=None,
                validity_status=CALIBRATION_REQUIRED,
                multipliers={},
                raw_measurements=raw,
                units={
                    "horizontal_cm": "cm",
                    "vertical_cm": "cm",
                    "travel_distance_cm": "cm",
                    "asymmetry_deg": "degrees",
                    "frequency_lifts_per_min": "lifts/min",
                    "load_weight_kg": "kg",
                    "recommended_weight_limit_kg": "kg",
                },
                missing_inputs=sorted(set(missing)),
                invalid_inputs=invalid,
                unsupported_conditions=unsupported,
                warnings=["Physical distances from video require a validated scale calibration."],
                context=inp.context,
            )

        range_checks = {
            "horizontal_cm": lambda value: value > 0 and value <= 63,
            "vertical_cm": lambda value: 0 <= value <= 175,
            "travel_distance_cm": lambda value: value > 0 and value <= 175,
            "asymmetry_deg": lambda value: 0 <= value <= 135,
            "frequency_lifts_per_min": lambda value: value > 0 and value <= 15,
            "load_weight_kg": lambda value: value > 0,
        }
        for name, check in range_checks.items():
            value = getattr(inp, name)
            if value is not None and name not in invalid and not check(value):
                invalid.append(name)

        units = {
            "horizontal_cm": "cm",
            "vertical_cm": "cm",
            "travel_distance_cm": "cm",
            "asymmetry_deg": "degrees",
            "frequency_lifts_per_min": "lifts/min",
            "load_weight_kg": "kg",
            "recommended_weight_limit_kg": "kg",
        }
        if unsupported or invalid:
            return NIOSHResult(
                assessment_type="RNLE",
                method_version=METHOD_VERSION,
                location=inp.location,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                risk_level=None,
                action_level=None,
                validity_status=UNSUPPORTED_RNLE_TASK,
                multipliers={},
                raw_measurements=raw,
                units=units,
                missing_inputs=missing,
                invalid_inputs=sorted(set(invalid)),
                unsupported_conditions=unsupported,
                warnings=warnings,
                context=inp.context,
            )
        if missing:
            return NIOSHResult(
                assessment_type="RNLE",
                method_version=METHOD_VERSION,
                location=inp.location,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                risk_level=None,
                action_level=None,
                validity_status=MISSING_REQUIRED_TASK_INPUT,
                multipliers={},
                raw_measurements=raw,
                units=units,
                missing_inputs=missing,
                invalid_inputs=[],
                unsupported_conditions=[],
                warnings=warnings,
                context=inp.context,
            )

        assert inp.horizontal_cm is not None
        assert inp.vertical_cm is not None
        assert inp.travel_distance_cm is not None
        assert inp.asymmetry_deg is not None
        assert inp.frequency_lifts_per_min is not None
        assert inp.duration is not None
        assert inp.coupling is not None
        assert inp.load_weight_kg is not None
        assert inp.applicability is not None

        horizontal = inp.horizontal_cm
        distance = inp.travel_distance_cm
        frequency = inp.frequency_lifts_per_min
        if horizontal < 25:
            horizontal = 25.0
            warnings.append(
                "Horizontal distance below 25 cm uses the RNLE method-defined 25 cm floor."
            )
        if distance < 25:
            distance = 25.0
            warnings.append("Vertical travel below 25 cm uses the RNLE method-defined 25 cm floor.")
        if frequency < 0.2:
            frequency = 0.2
            warnings.append("Frequency below 0.2 lifts/min uses the lowest RNLE table row.")

        hm = 25.0 / horizontal
        vm = 1.0 - 0.003 * abs(inp.vertical_cm - 75.0)
        dm = 0.82 + 4.5 / distance
        am = 1.0 - 0.0032 * inp.asymmetry_deg
        fm = self._frequency_multiplier(frequency, inp.duration, inp.vertical_cm)
        cm = CM_TABLE[inp.coupling][0 if inp.vertical_cm < 75 else 1]
        multipliers = {"LC": self.LC, "HM": hm, "VM": vm, "DM": dm, "AM": am, "FM": fm, "CM": cm}
        if fm == 0:
            return NIOSHResult(
                assessment_type="RNLE",
                method_version=METHOD_VERSION,
                location=inp.location,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                risk_level=None,
                action_level=None,
                validity_status=UNSUPPORTED_RNLE_TASK,
                multipliers=multipliers,
                raw_measurements=raw,
                units=units,
                missing_inputs=[],
                invalid_inputs=["frequency_lifts_per_min"],
                unsupported_conditions=["frequency_duration_combination_outside_rnle_table"],
                warnings=warnings,
                context=inp.context,
            )
        rwl = self.LC * hm * vm * dm * am * fm * cm
        lifting_index = inp.load_weight_kg / rwl
        risk, action = self._risk(lifting_index)
        status = VALID_WITH_LIMITATIONS if warnings else VALID
        return NIOSHResult(
            assessment_type="RNLE",
            method_version=METHOD_VERSION,
            location=inp.location,
            recommended_weight_limit_kg=rwl,
            lifting_index=lifting_index,
            risk_level=risk,
            action_level=action,
            validity_status=status,
            multipliers=multipliers,
            raw_measurements=raw,
            units=units,
            missing_inputs=[],
            invalid_inputs=[],
            unsupported_conditions=[],
            warnings=warnings,
            context=inp.context,
        )

    def calculate_lift(
        self,
        origin: NIOSHInput,
        destination: NIOSHInput | None = None,
        *,
        significant_destination_control: bool = False,
    ) -> NIOSHLiftResult:
        """Calculate origin and, when required, destination control limits."""

        origin_result = self.calculate(origin)
        if significant_destination_control and destination is None:
            return NIOSHLiftResult(
                origin=origin_result,
                destination=None,
                controlling_location=None,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                validity_status=MISSING_REQUIRED_TASK_INPUT,
                selection_reason="Destination measurements are required when significant control is needed.",
            )
        destination_result = self.calculate(destination) if destination is not None else None
        candidates = [
            result
            for result in (origin_result, destination_result)
            if result is not None and result.recommended_weight_limit_kg is not None
        ]
        expected_count = 2 if significant_destination_control else 1
        if len(candidates) < expected_count:
            return NIOSHLiftResult(
                origin=origin_result,
                destination=destination_result,
                controlling_location=None,
                recommended_weight_limit_kg=None,
                lifting_index=None,
                validity_status=(destination_result or origin_result).validity_status,
                selection_reason="A required RNLE location is incomplete or unsupported.",
            )
        controlling = min(
            candidates,
            key=lambda result: (
                result.recommended_weight_limit_kg
                if result.recommended_weight_limit_kg is not None
                else float("inf")
            ),
        )
        return NIOSHLiftResult(
            origin=origin_result,
            destination=destination_result,
            controlling_location=controlling.location,
            recommended_weight_limit_kg=controlling.recommended_weight_limit_kg,
            lifting_index=controlling.lifting_index,
            validity_status=controlling.validity_status,
            selection_reason="The location with the lower RWL controls the lift.",
        )
