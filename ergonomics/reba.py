"""Deterministic Rapid Entire Body Assessment (REBA).

Source of truth: Hignett & McAtamney, Applied Ergonomics 31(2), 2000,
figures 1-5 and Tables A-C.  RULA rules are not reused here.
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

METHOD_VERSION = "REBA 2000"


@dataclass(frozen=True)
class REBAInput:
    side: str
    trunk_flexion_deg: float | None
    trunk_twisted_or_side_flexed: bool | None
    neck_flexion_deg: float | None
    neck_twisted_or_side_flexed: bool | None
    legs_weight_bearing: str | None
    knee_flexion_deg: float | None
    seated: bool | None
    upper_arm_flexion_deg: float | None
    upper_arm_abducted_or_rotated: bool | None
    shoulder_raised: bool | None
    arm_supported_or_gravity_assisted: bool | None
    lower_arm_flexion_deg: float | None
    wrist_flexion_deg: float | None
    wrist_deviated_or_twisted: bool | None
    load_kg: float | None
    shock_or_rapid_force: bool | None
    coupling: str | None
    static_posture_over_one_minute: bool | None
    repeated_more_than_four_per_minute: bool | None
    rapid_large_posture_change_or_unstable_base: bool | None
    context: AssessmentContext = field(default_factory=AssessmentContext)


@dataclass(frozen=True)
class REBAScore:
    assessment_type: str
    method_version: str
    side: str
    final_score: int | None
    risk_level: str | None
    action_level: str | None
    action_category: int | None
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

    def __getattr__(self, name: str) -> Any:
        aliases = {
            "neck_score": "neck",
            "trunk_score": "trunk",
            "legs_score": "legs",
            "table_a_score": "table_a",
            "force_score": "load_force",
            "score_a": "score_a",
            "upper_arm_score": "upper_arm",
            "lower_arm_score": "lower_arm",
            "wrist_score": "wrist",
            "table_b_score": "table_b",
            "coupling_score": "coupling",
            "score_b": "score_b",
            "table_c_score": "table_c",
            "activity_score": "activity",
        }
        if name in aliases:
            return self.intermediate_scores.get(aliases[name])
        raise AttributeError(name)


class REBACalculator:
    TABLE_A = (
        ((1, 2, 3, 4), (1, 2, 3, 4), (3, 3, 5, 6)),
        ((2, 3, 4, 5), (3, 4, 5, 6), (4, 5, 6, 7)),
        ((2, 4, 5, 6), (4, 5, 6, 7), (5, 6, 7, 8)),
        ((3, 5, 6, 7), (5, 6, 7, 8), (6, 7, 8, 9)),
        ((4, 6, 7, 8), (6, 7, 8, 9), (7, 8, 9, 9)),
    )
    TABLE_B = (
        ((1, 2, 2), (1, 2, 3)),
        ((1, 2, 3), (2, 3, 4)),
        ((3, 4, 5), (4, 5, 5)),
        ((4, 5, 5), (5, 6, 7)),
        ((6, 7, 8), (7, 8, 8)),
        ((7, 8, 8), (8, 9, 9)),
    )
    TABLE_C = (
        (1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7),
        (1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8),
        (2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8),
        (3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9),
        (4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9),
        (6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10),
        (7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11),
        (8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11),
        (9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12),
        (10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12),
        (11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12),
        (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
    )
    RISK_LEVELS = {
        1: ("Negligible", "No action necessary", 0),
        2: ("Low", "Action may be necessary", 1),
        3: ("Low", "Action may be necessary", 1),
        4: ("Medium", "Action is necessary", 2),
        5: ("Medium", "Action is necessary", 2),
        6: ("Medium", "Action is necessary", 2),
        7: ("Medium", "Action is necessary", 2),
        8: ("High", "Action is necessary soon", 3),
        9: ("High", "Action is necessary soon", 3),
        10: ("High", "Action is necessary soon", 3),
        11: ("Very High", "Action is necessary now", 4),
        12: ("Very High", "Action is necessary now", 4),
        13: ("Very High", "Action is necessary now", 4),
        14: ("Very High", "Action is necessary now", 4),
        15: ("Very High", "Action is necessary now", 4),
    }

    @staticmethod
    def score_trunk(flexion: float, twisted_or_side_flexed: bool) -> int:
        if flexion == 0:
            score = 1
        elif -20 <= flexion <= 20:
            score = 2
        elif flexion < -20 or flexion <= 60:
            score = 3
        else:
            score = 4
        return min(5, score + int(twisted_or_side_flexed))

    @staticmethod
    def score_neck(flexion: float, twisted_or_side_flexed: bool) -> int:
        score = 1 if 0 <= flexion <= 20 else 2
        return min(3, score + int(twisted_or_side_flexed))

    @staticmethod
    def score_legs(weight_bearing: str, knee_flexion: float, seated: bool) -> int:
        if weight_bearing not in {"bilateral", "walking", "unilateral", "unstable"}:
            raise ValueError(
                "legs_weight_bearing must be bilateral, walking, unilateral, or unstable"
            )
        score = 1 if weight_bearing in {"bilateral", "walking"} else 2
        if not seated:
            if 30 < knee_flexion <= 60:
                score += 1
            elif knee_flexion > 60:
                score += 2
        return min(4, score)

    @staticmethod
    def score_upper_arm(
        flexion: float,
        abducted_or_rotated: bool,
        shoulder_raised: bool,
        supported_or_assisted: bool,
    ) -> int:
        if -20 <= flexion <= 20:
            score = 1
        elif flexion < -20 or flexion <= 45:
            score = 2
        elif flexion <= 90:
            score = 3
        else:
            score = 4
        score += int(abducted_or_rotated) + int(shoulder_raised) - int(supported_or_assisted)
        return min(6, max(1, score))

    @staticmethod
    def score_lower_arm(flexion: float) -> int:
        return 1 if 60 <= flexion <= 100 else 2

    @staticmethod
    def score_wrist(flexion: float, deviated_or_twisted: bool) -> int:
        return min(3, (1 if abs(flexion) <= 15 else 2) + int(deviated_or_twisted))

    @staticmethod
    def score_load(load_kg: float, shock_or_rapid: bool) -> int:
        if load_kg < 5:
            score = 0
        elif load_kg <= 10:
            score = 1
        else:
            score = 2
        return min(3, score + int(shock_or_rapid))

    @staticmethod
    def score_coupling(coupling: str) -> int:
        try:
            return {"good": 0, "fair": 1, "poor": 2, "unacceptable": 3}[coupling]
        except KeyError as exc:
            raise ValueError("coupling must be good, fair, poor, or unacceptable") from exc

    @staticmethod
    def _require_score(name: str, score: int, low: int, high: int) -> None:
        if isinstance(score, bool) or not isinstance(score, int) or not low <= score <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")

    def lookup_table_a(self, trunk: int, neck: int, legs: int) -> int:
        self._require_score("trunk_score", trunk, 1, 5)
        self._require_score("neck_score", neck, 1, 3)
        self._require_score("legs_score", legs, 1, 4)
        return self.TABLE_A[trunk - 1][neck - 1][legs - 1]

    def lookup_table_b(self, upper_arm: int, lower_arm: int, wrist: int) -> int:
        self._require_score("upper_arm_score", upper_arm, 1, 6)
        self._require_score("lower_arm_score", lower_arm, 1, 2)
        self._require_score("wrist_score", wrist, 1, 3)
        return self.TABLE_B[upper_arm - 1][lower_arm - 1][wrist - 1]

    def lookup_table_c(self, score_a: int, score_b: int) -> int:
        self._require_score("score_a", score_a, 1, 12)
        self._require_score("score_b", score_b, 1, 12)
        return self.TABLE_C[score_a - 1][score_b - 1]

    def calculate(self, inp: REBAInput) -> REBAScore:
        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = list(inp.context.pose_quality.get("warnings", []))
        if inp.load_kg is None:
            missing.append("load_kg")
        else:
            validate_finite_number("load_kg", inp.load_kg, invalid)
            if "load_kg" not in invalid and inp.load_kg < 0:
                invalid.append("load_kg")
        if inp.coupling is None:
            missing.append("coupling")
        elif inp.coupling not in {"good", "fair", "poor", "unacceptable"}:
            invalid.append("coupling")
        if inp.side not in {"left", "right"}:
            invalid.append("side")
        if inp.legs_weight_bearing is None:
            missing.append("legs_weight_bearing")
        elif inp.legs_weight_bearing not in {"bilateral", "walking", "unilateral", "unstable"}:
            invalid.append("legs_weight_bearing")
        boolean_fields = (
            "trunk_twisted_or_side_flexed",
            "neck_twisted_or_side_flexed",
            "seated",
            "upper_arm_abducted_or_rotated",
            "shoulder_raised",
            "arm_supported_or_gravity_assisted",
            "wrist_deviated_or_twisted",
            "shock_or_rapid_force",
            "static_posture_over_one_minute",
            "repeated_more_than_four_per_minute",
            "rapid_large_posture_change_or_unstable_base",
        )
        for name in boolean_fields:
            value = getattr(inp, name)
            if value is None:
                missing.append(name)
            elif not isinstance(value, bool):
                invalid.append(name)
        numeric = (
            "trunk_flexion_deg",
            "neck_flexion_deg",
            "knee_flexion_deg",
            "upper_arm_flexion_deg",
            "lower_arm_flexion_deg",
            "wrist_flexion_deg",
        )
        for name in numeric:
            value = getattr(inp, name)
            if value is None:
                missing.append(name)
            else:
                validate_finite_number(name, value, invalid)
        if (
            inp.knee_flexion_deg is not None
            and "knee_flexion_deg" not in invalid
            and not 0 <= inp.knee_flexion_deg <= 180
        ):
            invalid.append("knee_flexion_deg")

        raw = asdict(inp)
        raw.pop("context")
        units = {name: "degrees" for name in numeric}
        units["load_kg"] = "kg"
        if missing or invalid:
            return REBAScore(
                assessment_type="REBA",
                method_version=METHOD_VERSION,
                side=inp.side,
                final_score=None,
                risk_level=None,
                action_level=None,
                action_category=None,
                validity_status=incomplete_assessment_status(inp.context),
                intermediate_scores={},
                raw_measurements=raw,
                units=units,
                missing_inputs=missing,
                invalid_inputs=invalid,
                warnings=warnings,
                context=inp.context,
            )

        assert inp.trunk_flexion_deg is not None
        assert inp.trunk_twisted_or_side_flexed is not None
        assert inp.neck_flexion_deg is not None
        assert inp.neck_twisted_or_side_flexed is not None
        assert inp.legs_weight_bearing is not None
        assert inp.knee_flexion_deg is not None
        assert inp.seated is not None
        assert inp.upper_arm_flexion_deg is not None
        assert inp.upper_arm_abducted_or_rotated is not None
        assert inp.shoulder_raised is not None
        assert inp.arm_supported_or_gravity_assisted is not None
        assert inp.lower_arm_flexion_deg is not None
        assert inp.wrist_flexion_deg is not None
        assert inp.wrist_deviated_or_twisted is not None
        assert inp.load_kg is not None
        assert inp.shock_or_rapid_force is not None
        assert inp.coupling is not None
        assert inp.static_posture_over_one_minute is not None
        assert inp.repeated_more_than_four_per_minute is not None
        assert inp.rapid_large_posture_change_or_unstable_base is not None

        trunk = self.score_trunk(inp.trunk_flexion_deg, inp.trunk_twisted_or_side_flexed)
        neck = self.score_neck(inp.neck_flexion_deg, inp.neck_twisted_or_side_flexed)
        legs = self.score_legs(inp.legs_weight_bearing, inp.knee_flexion_deg, inp.seated)
        table_a = self.lookup_table_a(trunk, neck, legs)
        load_force = self.score_load(inp.load_kg, inp.shock_or_rapid_force)
        score_a = table_a + load_force

        upper = self.score_upper_arm(
            inp.upper_arm_flexion_deg,
            inp.upper_arm_abducted_or_rotated,
            inp.shoulder_raised,
            inp.arm_supported_or_gravity_assisted,
        )
        lower = self.score_lower_arm(inp.lower_arm_flexion_deg)
        wrist = self.score_wrist(inp.wrist_flexion_deg, inp.wrist_deviated_or_twisted)
        table_b = self.lookup_table_b(upper, lower, wrist)
        coupling = self.score_coupling(inp.coupling)
        score_b = table_b + coupling
        table_c = self.lookup_table_c(score_a, score_b)
        activity = sum(
            (
                inp.static_posture_over_one_minute,
                inp.repeated_more_than_four_per_minute,
                inp.rapid_large_posture_change_or_unstable_base,
            )
        )
        final = table_c + activity
        risk, action, action_category = self.RISK_LEVELS[final]
        intermediate = {
            "trunk": trunk,
            "neck": neck,
            "legs": legs,
            "table_a": table_a,
            "load_force": load_force,
            "score_a": score_a,
            "upper_arm": upper,
            "lower_arm": lower,
            "wrist": wrist,
            "table_b": table_b,
            "coupling": coupling,
            "score_b": score_b,
            "table_c": table_c,
            "activity": activity,
        }
        return REBAScore(
            assessment_type="REBA",
            method_version=METHOD_VERSION,
            side=inp.side,
            final_score=final,
            risk_level=risk,
            action_level=action,
            action_category=action_category,
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
        self, left: REBAInput, right: REBAInput
    ) -> BilateralAssessmentResult[REBAScore]:
        if left.side != "left" or right.side != "right":
            raise ValueError("calculate_bilateral requires left then right anatomical-side inputs")
        return select_worst_side("REBA", self.calculate(left), self.calculate(right))
