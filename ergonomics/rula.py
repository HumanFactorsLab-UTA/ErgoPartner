"""Deterministic Rapid Upper Limb Assessment (RULA).

Source of truth: McAtamney & Corlett, Applied Ergonomics 24(2), 1993,
figures 1-5 and Tables A-C.  This module accepts assessment measurements; it
does not infer task conditions from pose confidence or load from video.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ergonomics.common import (
    AssessmentContext,
    BilateralAssessmentResult,
    complete_assessment_status,
    incomplete_assessment_status,
    select_worst_side,
    validate_finite_number,
)

METHOD_VERSION = "RULA 1993"


@dataclass(frozen=True)
class RULAInput:
    side: str
    upper_arm_flexion_deg: float | None
    upper_arm_abducted: bool | None
    shoulder_raised: bool | None
    arm_supported_or_leaning: bool | None
    lower_arm_flexion_deg: float | None
    lower_arm_across_midline_or_out_to_side: bool | None
    wrist_flexion_deg: float | None
    wrist_deviated: bool | None
    wrist_twist_at_end_range: bool | None
    neck_flexion_deg: float | None
    neck_twisted: bool | None
    neck_side_bent: bool | None
    trunk_flexion_deg: float | None
    trunk_twisted: bool | None
    trunk_side_bent: bool | None
    legs_supported_and_balanced: bool | None
    static_posture_over_one_minute: bool | None
    repeated_more_than_four_per_minute: bool | None
    load_kg: float | None
    load_type: str | None
    shock_or_rapid_force: bool | None
    context: AssessmentContext = field(default_factory=AssessmentContext)


@dataclass(frozen=True)
class RULAScore:
    assessment_type: str
    method_version: str
    side: str
    final_score: int | None
    risk_level: str | None
    action_level: str | None
    validity_status: str
    intermediate_scores: dict[str, int]
    raw_measurements: dict[str, Any]
    units: dict[str, str]
    missing_inputs: list[str]
    invalid_inputs: list[str]
    warnings: list[str]
    context: AssessmentContext

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # Compatibility properties retained only for output adapters.
    def __getattr__(self, name: str) -> Any:
        aliases = {
            "upper_arm_score": "upper_arm",
            "lower_arm_score": "lower_arm",
            "wrist_score": "wrist",
            "wrist_twist_score": "wrist_twist",
            "posture_score_a": "table_a_posture",
            "muscle_use_a": "muscle_use",
            "force_score_a": "force_load",
            "wrist_arm_score": "score_a",
            "neck_score": "neck",
            "trunk_score": "trunk",
            "legs_score": "legs",
            "posture_score_b": "table_b_posture",
            "muscle_use_b": "muscle_use",
            "force_score_b": "force_load",
            "neck_trunk_score": "score_b",
        }
        if name in aliases:
            return self.intermediate_scores.get(aliases[name])
        raise AttributeError(name)


class RULACalculator:
    """Pure RULA table and body-part scorer."""

    # upper arm x lower arm x wrist x wrist twist
    TABLE_A = (
        (
            ((1, 2), (2, 2), (2, 3), (3, 3)),
            ((2, 2), (2, 2), (3, 3), (3, 3)),
            ((2, 3), (3, 3), (3, 3), (4, 4)),
        ),
        (
            ((2, 3), (3, 3), (3, 4), (4, 4)),
            ((3, 3), (3, 3), (3, 4), (4, 4)),
            ((3, 4), (4, 4), (4, 4), (5, 5)),
        ),
        (
            ((3, 3), (4, 4), (4, 4), (5, 5)),
            ((3, 4), (4, 4), (4, 4), (5, 5)),
            ((4, 4), (4, 4), (4, 5), (5, 5)),
        ),
        (
            ((4, 4), (4, 4), (4, 5), (5, 5)),
            ((4, 4), (4, 4), (4, 5), (5, 5)),
            ((4, 4), (4, 5), (5, 5), (6, 6)),
        ),
        (
            ((5, 5), (5, 5), (5, 6), (6, 7)),
            ((5, 6), (6, 6), (6, 7), (7, 7)),
            ((6, 6), (6, 7), (7, 7), (7, 8)),
        ),
        (
            ((7, 7), (7, 7), (7, 8), (8, 9)),
            ((8, 8), (8, 8), (8, 9), (9, 9)),
            ((9, 9), (9, 9), (9, 9), (9, 9)),
        ),
    )
    TABLE_B = (
        ((1, 3), (2, 3), (3, 4), (5, 5), (6, 6), (7, 7)),
        ((2, 3), (2, 3), (4, 5), (5, 5), (6, 7), (7, 7)),
        ((3, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 7)),
        ((5, 5), (5, 6), (6, 7), (7, 7), (7, 7), (8, 8)),
        ((7, 7), (7, 7), (7, 8), (8, 8), (8, 8), (8, 8)),
        ((8, 8), (8, 8), (8, 8), (8, 9), (9, 9), (9, 9)),
    )
    TABLE_C = (
        (1, 2, 3, 3, 4, 5, 5),
        (2, 2, 3, 4, 4, 5, 5),
        (3, 3, 3, 4, 4, 5, 6),
        (3, 3, 3, 4, 5, 6, 6),
        (4, 4, 4, 5, 6, 7, 7),
        (4, 4, 5, 6, 6, 7, 7),
        (5, 5, 6, 6, 7, 7, 7),
        (5, 5, 6, 7, 7, 7, 7),
    )
    RISK_LEVELS = {
        1: (
            "Acceptable",
            "Posture is acceptable if it is not maintained or repeated for long periods.",
        ),
        2: (
            "Acceptable",
            "Posture is acceptable if it is not maintained or repeated for long periods.",
        ),
        3: ("Investigate", "Further investigation is needed; changes may be required."),
        4: ("Investigate", "Further investigation is needed; changes may be required."),
        5: ("Investigate Soon", "Investigation and changes are required soon."),
        6: ("Investigate Soon", "Investigation and changes are required soon."),
        7: ("Investigate Now", "Investigation and changes are required immediately."),
    }

    @staticmethod
    def score_upper_arm(flexion: float, abducted: bool, raised: bool, supported: bool) -> int:
        if -20 <= flexion <= 20:
            score = 1
        elif flexion < -20 or flexion <= 45:
            score = 2
        elif flexion <= 90:
            score = 3
        else:
            score = 4
        score += int(abducted) + int(raised) - int(supported)
        return min(6, max(1, score))

    @staticmethod
    def score_lower_arm(flexion: float, across_or_out: bool) -> int:
        return min(3, (1 if 60 <= flexion <= 100 else 2) + int(across_or_out))

    @staticmethod
    def score_wrist(flexion: float, deviated: bool) -> int:
        magnitude = abs(flexion)
        if magnitude == 0:
            score = 1
        elif magnitude <= 15:
            score = 2
        else:
            score = 3
        return min(4, score + int(deviated))

    @staticmethod
    def score_neck(flexion: float, twisted: bool, side_bent: bool) -> int:
        if 0 <= flexion <= 10:
            score = 1
        elif flexion <= 20 and flexion > 10:
            score = 2
        elif flexion > 20:
            score = 3
        else:
            score = 4
        return min(6, score + int(twisted) + int(side_bent))

    @staticmethod
    def score_trunk(flexion: float, twisted: bool, side_bent: bool) -> int:
        if flexion == 0:
            score = 1
        elif 0 < flexion <= 20:
            score = 2
        elif flexion <= 60:
            score = 3
        else:
            score = 4
        return min(6, score + int(twisted) + int(side_bent))

    @staticmethod
    def score_force(load_kg: float, load_type: str, shock_or_rapid: bool) -> int:
        if shock_or_rapid:
            return 3
        if load_kg < 2:
            return 0
        if load_kg == 2:
            return 0 if load_type == "intermittent" else 2
        if load_kg <= 10:
            return 1 if load_type == "intermittent" else 2
        return 2 if load_type == "intermittent" else 3

    @staticmethod
    def _require_score(name: str, score: int, low: int, high: int) -> None:
        if isinstance(score, bool) or not isinstance(score, int) or not low <= score <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")

    def lookup_table_a(
        self, upper_arm_score: int, lower_arm_score: int, wrist_score: int, wrist_twist_score: int
    ) -> int:
        self._require_score("upper_arm_score", upper_arm_score, 1, 6)
        self._require_score("lower_arm_score", lower_arm_score, 1, 3)
        self._require_score("wrist_score", wrist_score, 1, 4)
        self._require_score("wrist_twist_score", wrist_twist_score, 1, 2)
        return self.TABLE_A[upper_arm_score - 1][lower_arm_score - 1][wrist_score - 1][
            wrist_twist_score - 1
        ]

    def lookup_table_b(self, neck_score: int, trunk_score: int, legs_score: int) -> int:
        self._require_score("neck_score", neck_score, 1, 6)
        self._require_score("trunk_score", trunk_score, 1, 6)
        self._require_score("legs_score", legs_score, 1, 2)
        return self.TABLE_B[neck_score - 1][trunk_score - 1][legs_score - 1]

    def lookup_table_c(self, score_a: int, score_b: int) -> int:
        if isinstance(score_a, bool) or not isinstance(score_a, int) or score_a < 1:
            raise ValueError("score_a must be a positive integer")
        if isinstance(score_b, bool) or not isinstance(score_b, int) or score_b < 1:
            raise ValueError("score_b must be a positive integer")
        # Official Table C labels its terminal row and column 8+ and 7+.
        return self.TABLE_C[min(score_a, 8) - 1][min(score_b, 7) - 1]

    def calculate(self, inp: RULAInput) -> RULAScore:
        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = list(inp.context.pose_quality.get("warnings", []))
        if inp.load_kg is None:
            missing.append("load_kg")
        else:
            validate_finite_number("load_kg", inp.load_kg, invalid)
            if "load_kg" not in invalid and inp.load_kg < 0:
                invalid.append("load_kg")
        if inp.load_type is None:
            missing.append("load_type")
        elif inp.load_type not in {"intermittent", "static_or_repeated"}:
            invalid.append("load_type")
        if inp.side not in {"left", "right"}:
            invalid.append("side")
        boolean_fields = (
            "upper_arm_abducted",
            "shoulder_raised",
            "arm_supported_or_leaning",
            "lower_arm_across_midline_or_out_to_side",
            "wrist_deviated",
            "wrist_twist_at_end_range",
            "neck_twisted",
            "neck_side_bent",
            "trunk_twisted",
            "trunk_side_bent",
            "legs_supported_and_balanced",
            "static_posture_over_one_minute",
            "repeated_more_than_four_per_minute",
            "shock_or_rapid_force",
        )
        for name in boolean_fields:
            value = getattr(inp, name)
            if value is None:
                missing.append(name)
            elif not isinstance(value, bool):
                invalid.append(name)
        numeric = (
            "upper_arm_flexion_deg",
            "lower_arm_flexion_deg",
            "wrist_flexion_deg",
            "neck_flexion_deg",
            "trunk_flexion_deg",
        )
        for name in numeric:
            value = getattr(inp, name)
            if value is None:
                missing.append(name)
            else:
                validate_finite_number(name, value, invalid)
        # The original RULA trunk figure does not define a trunk-extension bin.
        if (
            inp.trunk_flexion_deg is not None
            and "trunk_flexion_deg" not in invalid
            and inp.trunk_flexion_deg < 0
        ):
            invalid.append("trunk_flexion_deg")
            warnings.append(
                "RULA 1993 does not define a trunk-extension score; no score was guessed."
            )

        raw = asdict(inp)
        raw.pop("context")
        units = {name: "degrees" for name in numeric}
        units["load_kg"] = "kg"
        if missing or invalid:
            return RULAScore(
                assessment_type="RULA",
                method_version=METHOD_VERSION,
                side=inp.side,
                final_score=None,
                risk_level=None,
                action_level=None,
                validity_status=incomplete_assessment_status(inp.context),
                intermediate_scores={},
                raw_measurements=raw,
                units=units,
                missing_inputs=missing,
                invalid_inputs=invalid,
                warnings=warnings,
                context=inp.context,
            )

        assert inp.upper_arm_flexion_deg is not None
        assert inp.upper_arm_abducted is not None
        assert inp.shoulder_raised is not None
        assert inp.arm_supported_or_leaning is not None
        assert inp.lower_arm_flexion_deg is not None
        assert inp.lower_arm_across_midline_or_out_to_side is not None
        assert inp.wrist_flexion_deg is not None
        assert inp.wrist_deviated is not None
        assert inp.wrist_twist_at_end_range is not None
        assert inp.neck_flexion_deg is not None
        assert inp.neck_twisted is not None
        assert inp.neck_side_bent is not None
        assert inp.trunk_flexion_deg is not None
        assert inp.trunk_twisted is not None
        assert inp.trunk_side_bent is not None
        assert inp.legs_supported_and_balanced is not None
        assert inp.static_posture_over_one_minute is not None
        assert inp.repeated_more_than_four_per_minute is not None
        assert inp.load_kg is not None
        assert inp.load_type is not None
        assert inp.shock_or_rapid_force is not None

        upper = self.score_upper_arm(
            inp.upper_arm_flexion_deg,
            inp.upper_arm_abducted,
            inp.shoulder_raised,
            inp.arm_supported_or_leaning,
        )
        lower = self.score_lower_arm(
            inp.lower_arm_flexion_deg, inp.lower_arm_across_midline_or_out_to_side
        )
        wrist = self.score_wrist(inp.wrist_flexion_deg, inp.wrist_deviated)
        twist = 2 if inp.wrist_twist_at_end_range else 1
        table_a = self.lookup_table_a(upper, lower, wrist, twist)
        muscle = int(inp.static_posture_over_one_minute or inp.repeated_more_than_four_per_minute)
        force = self.score_force(inp.load_kg, inp.load_type, inp.shock_or_rapid_force)
        # This input model shares the muscle-use and load adjustments across both posture groups.
        score_a = table_a + muscle + force

        neck = self.score_neck(inp.neck_flexion_deg, inp.neck_twisted, inp.neck_side_bent)
        trunk = self.score_trunk(inp.trunk_flexion_deg, inp.trunk_twisted, inp.trunk_side_bent)
        legs = 1 if inp.legs_supported_and_balanced else 2
        table_b = self.lookup_table_b(neck, trunk, legs)
        score_b = table_b + muscle + force
        final = self.lookup_table_c(score_a, score_b)
        risk, action = self.RISK_LEVELS[final]
        intermediate = {
            "upper_arm": upper,
            "lower_arm": lower,
            "wrist": wrist,
            "wrist_twist": twist,
            "table_a_posture": table_a,
            "muscle_use": muscle,
            "force_load": force,
            "score_a": score_a,
            "neck": neck,
            "trunk": trunk,
            "legs": legs,
            "table_b_posture": table_b,
            "score_b": score_b,
            "table_c": final,
        }
        return RULAScore(
            assessment_type="RULA",
            method_version=METHOD_VERSION,
            side=inp.side,
            final_score=final,
            risk_level=risk,
            action_level=action,
            validity_status=complete_assessment_status(inp.context),
            intermediate_scores=intermediate,
            raw_measurements=raw,
            units=units,
            missing_inputs=[],
            invalid_inputs=[],
            warnings=warnings,
            context=inp.context,
        )

    def calculate_bilateral(
        self, left: RULAInput, right: RULAInput
    ) -> BilateralAssessmentResult[RULAScore]:
        if left.side != "left" or right.side != "right":
            raise ValueError("calculate_bilateral requires left then right anatomical-side inputs")
        return select_worst_side("RULA", self.calculate(left), self.calculate(right))
