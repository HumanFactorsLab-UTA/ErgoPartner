"""Shared, model-independent assessment result types.

The ergonomic calculators deliberately do not import pose, video, OpenCV, or
MediaPipe modules.  Callers must provide validated measurements and may attach
acquisition context separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VALID = "valid"
VALID_WITH_LIMITATIONS = "valid_with_limitations"
INSUFFICIENT_POSE_QUALITY = "insufficient_pose_quality"
MISSING_REQUIRED_TASK_INPUT = "missing_required_task_input"
UNSUPPORTED_CAMERA_VIEW = "unsupported_camera_view"
UNSUPPORTED_RNLE_TASK = "unsupported_rnle_task"
CALIBRATION_REQUIRED = "calibration_required"


@dataclass(frozen=True)
class AssessmentContext:
    """Optional provenance supplied by the acquisition pipeline."""

    frame_number: int | None = None
    timestamp_sec: float | None = None
    person_identifier: str | None = None
    input_source: str | None = None
    calibration_status: str = "not_applicable"
    pose_quality: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BilateralAssessmentResult[T]:
    """Two independently calculated sides and deterministic selection."""

    assessment_type: str
    left: T
    right: T
    selected_side: str | None
    selected: T | None
    selection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_worst_side[T](assessment_type: str, left: T, right: T) -> BilateralAssessmentResult[T]:
    """Select the higher valid score without disguising incomplete sides."""

    left_score = getattr(left, "final_score", None)
    right_score = getattr(right, "final_score", None)
    if left_score is None or right_score is None:
        return BilateralAssessmentResult(
            assessment_type=assessment_type,
            left=left,
            right=right,
            selected_side=None,
            selected=None,
            selection_reason="Both anatomical sides must be valid before worst-side selection.",
        )
    if right_score > left_score:
        selected_side, selected = "right", right
        reason = f"Right score {right_score} exceeds left score {left_score}."
    elif left_score > right_score:
        selected_side, selected = "left", left
        reason = f"Left score {left_score} exceeds right score {right_score}."
    else:
        # A tie is not silently presented as evidence that one side was worse.
        selected_side, selected = "both", left
        reason = f"Left and right scores are tied at {left_score}."
    return BilateralAssessmentResult(
        assessment_type=assessment_type,
        left=left,
        right=right,
        selected_side=selected_side,
        selected=selected,
        selection_reason=reason,
    )


def validate_finite_number(name: str, value: Any, invalid: list[str]) -> None:
    """Append a field name when a supplied numeric value is not finite."""

    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        invalid.append(name)


def incomplete_assessment_status(context: AssessmentContext) -> str:
    """Preserve acquisition failures instead of relabelling them as missing task data."""

    status = context.pose_quality.get("status")
    if status in {INSUFFICIENT_POSE_QUALITY, UNSUPPORTED_CAMERA_VIEW, CALIBRATION_REQUIRED}:
        return str(status)
    return MISSING_REQUIRED_TASK_INPUT


def complete_assessment_status(context: AssessmentContext) -> str:
    """A pose-derived score remains limited even when all method inputs are present."""

    return (
        VALID_WITH_LIMITATIONS
        if context.pose_quality.get("status") == VALID_WITH_LIMITATIONS
        else VALID
    )
