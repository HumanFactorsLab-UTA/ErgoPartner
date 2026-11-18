"""Explicit boundary between pose-derived measurements and ergonomic methods."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Any

from core.angle_calculator import BodyAngles, neck_flexion_landmarks
from ergonomics.common import (
    INSUFFICIENT_POSE_QUALITY,
    UNSUPPORTED_CAMERA_VIEW,
    VALID_WITH_LIMITATIONS,
    AssessmentContext,
)
from ergonomics.reba import REBAInput
from ergonomics.rula import RULAInput

LEFT_IDS = {
    "shoulder": 11,
    "elbow": 13,
    "wrist": 15,
    "index": 19,
    "pinky": 17,
    "hip": 23,
    "knee": 25,
    "ankle": 27,
}
RIGHT_IDS = {
    "shoulder": 12,
    "elbow": 14,
    "wrist": 16,
    "index": 20,
    "pinky": 18,
    "hip": 24,
    "knee": 26,
    "ankle": 28,
}


# RULA and REBA define their lowest posture band at exactly 0 degrees, which is
# faithful to the published figures and is deliberately preserved in the
# calculators. Pose-derived angles are continuous and never land on exactly zero,
# so that band is unreachable from video and a genuinely upright subject scores
# one band high on trunk, neck and wrist on every frame. Worse, RULA rejects
# negative trunk flexion outright, so a subject 0.3 degrees behind vertical
# invalidates the whole assessment.
#
# Resolving this in the calculators would break fidelity to the source method.
# It belongs here instead, at the CV/method boundary: a measurement whose
# magnitude is below the pipeline's angular resolution is reported as the neutral
# posture it represents, and the applied tolerance is recorded in the assessment
# context so the decision stays auditable.
DEFAULT_NEUTRAL_TOLERANCE_DEG = 2.0

# Angles whose lowest band is anchored at exactly zero.
NEUTRAL_SNAPPED_MEASUREMENTS = ("trunk_flexion", "neck_flexion", "wrist_flexion")


def _task_value(task: dict[str, Any], name: str) -> Any:
    """Return only an explicitly supplied task value; never manufacture one."""

    return task[name] if name in task else None


def _snap_neutral(value: float | None, tolerance_deg: float) -> float | None:
    """Report an angle within measurement resolution of neutral as exactly zero."""

    if value is None or tolerance_deg <= 0:
        return value
    return 0.0 if abs(value) <= tolerance_deg else value


def _quality_context(
    angles: BodyAngles,
    side: str,
    context: AssessmentContext,
    visibility_threshold: float,
    allow_pose_only_wrist: bool,
    admitted_landmarks: frozenset[int],
    *,
    requires_knee: bool,
) -> tuple[AssessmentContext, set[str]]:
    ids = LEFT_IDS if side == "left" else RIGHT_IDS
    wrist_source = getattr(angles, f"{side}_wrist_data_source", angles.wrist_data_source)
    wrist_landmarks = [ids["elbow"], ids["wrist"]]
    if wrist_source != "hand_mesh":
        wrist_landmarks.extend((ids["index"], ids["pinky"]))
    groups = {
        "trunk_flexion_deg": [11, 12, 23, 24],
        "neck_flexion_deg": neck_flexion_landmarks(angles.landmark_visibility),
        "upper_arm_flexion_deg": [ids["shoulder"], ids["elbow"]],
        "lower_arm_flexion_deg": [ids["shoulder"], ids["elbow"], ids["wrist"]],
        "wrist_flexion_deg": wrist_landmarks,
    }
    if requires_knee:
        groups["knee_flexion_deg"] = [ids["hip"], ids["knee"], ids["ankle"]]
    unavailable: set[str] = set()
    #: Measurements admitted on sustained-tracking evidence rather than on the
    #: per-frame confidence gate. Recorded so a score never hides which of the
    #: two supported it.
    sustained: set[str] = set()
    for name in getattr(angles, "unavailable_measurements", []):
        prefix = f"{side}_"
        if name.startswith(prefix):
            field = name[len(prefix) :].replace("forearm_flexion_deg", "lower_arm_flexion_deg")
            if field != "knee_flexion_deg" or requires_knee:
                unavailable.add(field)
        elif not name.startswith(("left_", "right_")):
            unavailable.add(name)
    visibility = angles.landmark_visibility
    per_measurement: dict[str, float] = {}
    for measurement, landmark_ids in groups.items():
        values = [float(visibility.get(index, 0.0)) for index in landmark_ids]
        minimum = min(values) if all(isfinite(value) for value in values) else 0.0
        source_name = measurement
        if measurement == "lower_arm_flexion_deg":
            source_name = f"{side}_forearm_flexion_deg"
        elif measurement in {
            "upper_arm_flexion_deg",
            "wrist_flexion_deg",
            "knee_flexion_deg",
        }:
            source_name = f"{side}_{measurement}"
        minimum = float(angles.measurement_visibility.get(source_name, minimum))
        if not isfinite(minimum):
            minimum = 0.0
        per_measurement[measurement] = minimum
        if minimum < visibility_threshold:
            # A landmark the whole sequence shows was tracked continuously and
            # smoothly is observed, even when its per-frame confidence sits under
            # the gate. Without this the knee, which reports 0.40-0.50 for an
            # entire single-camera trial while plainly visible, withheld every
            # REBA observation. See core/landmark_reliability.py.
            if landmark_ids and all(index in admitted_landmarks for index in landmark_ids):
                sustained.add(measurement)
            else:
                unavailable.add(measurement)
    warnings: list[str] = [
        "Angles are derived from monocular model-estimated 3D landmarks; anatomical-angle "
        "validity has not been established for this recording."
    ]
    camera_view_status = context.pose_quality.get("camera_view_status", "unknown")
    mirrored = context.pose_quality.get("mirrored")
    if wrist_source == "pose_only" and not allow_pose_only_wrist:
        unavailable.add("wrist_flexion_deg")
        warnings.append(
            "Wrist measurements are unavailable: pose-only hand points are not accepted "
            "for auditable wrist scoring."
        )
    if camera_view_status == "unsupported" or mirrored is True:
        unavailable.update(groups)
        status = UNSUPPORTED_CAMERA_VIEW
        warnings.append(
            "The configured camera view is unsupported, or mirrored footage was not "
            "anatomically remapped and validated."
        )
    elif unavailable.intersection(groups):
        status = INSUFFICIENT_POSE_QUALITY
        warnings.append("One or more measurements failed landmark-quality requirements.")
    else:
        status = VALID_WITH_LIMITATIONS
        if camera_view_status != "suitable":
            warnings.append("Camera viewpoint suitability was not confirmed.")
        if mirrored is None:
            warnings.append("Whether the footage is mirrored was not declared.")
    if sustained:
        warnings.append(
            "These measurements rest on landmarks admitted because the sequence shows them "
            "tracked continuously and smoothly, not because their per-frame confidence "
            f"cleared the threshold: {', '.join(sorted(sustained))}."
        )
    pose_quality = dict(context.pose_quality)
    pose_quality.update(
        {
            "status": status,
            "minimum_visibility_by_measurement": per_measurement,
            "unavailable_measurements": sorted(unavailable),
            "sustained_tracking_measurements": sorted(sustained),
            "wrist_data_source": wrist_source,
            "warnings": warnings,
        }
    )
    return replace(context, pose_quality=pose_quality), unavailable


def _record_tolerance(context: AssessmentContext, tolerance_deg: float) -> AssessmentContext:
    """Record the neutral tolerance that was applied to this observation."""

    pose_quality = dict(context.pose_quality)
    pose_quality["neutral_tolerance_deg"] = float(tolerance_deg)
    pose_quality["neutral_snapped_measurements"] = list(NEUTRAL_SNAPPED_MEASUREMENTS)
    return replace(context, pose_quality=pose_quality)


def _angle_value(
    angles: BodyAngles, side: str, base: str, unavailable: set[str], measurement_name: str
) -> float | None:
    if measurement_name in unavailable:
        return None
    return float(getattr(angles, f"{side}_{base}"))


def build_rula_input(
    angles: BodyAngles,
    side: str,
    task: dict[str, Any],
    context: AssessmentContext,
    *,
    visibility_threshold: float,
    allow_pose_only_wrist: bool = False,
    neutral_tolerance_deg: float = DEFAULT_NEUTRAL_TOLERANCE_DEG,
    admitted_landmarks: frozenset[int] = frozenset(),
) -> RULAInput:
    quality_context, unavailable = _quality_context(
        angles,
        side,
        context,
        visibility_threshold,
        allow_pose_only_wrist,
        admitted_landmarks,
        requires_knee=False,
    )
    quality_context = _record_tolerance(quality_context, neutral_tolerance_deg)
    return RULAInput(
        side=side,
        upper_arm_flexion_deg=_angle_value(
            angles, side, "upper_arm_flexion", unavailable, "upper_arm_flexion_deg"
        ),
        upper_arm_abducted=_task_value(task, f"{side}_upper_arm_abducted"),
        shoulder_raised=_task_value(task, f"{side}_shoulder_raised"),
        arm_supported_or_leaning=_task_value(task, f"{side}_arm_supported_or_leaning"),
        lower_arm_flexion_deg=_angle_value(
            angles, side, "forearm_flexion", unavailable, "lower_arm_flexion_deg"
        ),
        lower_arm_across_midline_or_out_to_side=_task_value(
            task, f"{side}_lower_arm_across_midline_or_out_to_side"
        ),
        wrist_flexion_deg=_snap_neutral(
            _angle_value(angles, side, "wrist_flexion", unavailable, "wrist_flexion_deg"),
            neutral_tolerance_deg,
        ),
        wrist_deviated=_task_value(task, f"{side}_wrist_deviated"),
        wrist_twist_at_end_range=_task_value(task, f"{side}_wrist_twist_at_end_range"),
        neck_flexion_deg=None
        if "neck_flexion_deg" in unavailable
        else _snap_neutral(float(angles.neck_flexion), neutral_tolerance_deg),
        neck_twisted=_task_value(task, "neck_twisted"),
        neck_side_bent=_task_value(task, "neck_side_bent"),
        trunk_flexion_deg=None
        if "trunk_flexion_deg" in unavailable
        else _snap_neutral(float(angles.trunk_flexion), neutral_tolerance_deg),
        trunk_twisted=_task_value(task, "trunk_twisted"),
        trunk_side_bent=_task_value(task, "trunk_side_bent"),
        legs_supported_and_balanced=_task_value(task, "legs_supported_and_balanced"),
        static_posture_over_one_minute=_task_value(task, "static_posture_over_one_minute"),
        repeated_more_than_four_per_minute=_task_value(task, "repeated_more_than_four_per_minute"),
        load_kg=_task_value(task, "load_kg"),
        load_type=_task_value(task, "rula_load_type"),
        shock_or_rapid_force=_task_value(task, "shock_or_rapid_force"),
        context=quality_context,
    )


def build_reba_input(
    angles: BodyAngles,
    side: str,
    task: dict[str, Any],
    context: AssessmentContext,
    *,
    visibility_threshold: float,
    allow_pose_only_wrist: bool = False,
    neutral_tolerance_deg: float = DEFAULT_NEUTRAL_TOLERANCE_DEG,
    admitted_landmarks: frozenset[int] = frozenset(),
) -> REBAInput:
    quality_context, unavailable = _quality_context(
        angles,
        side,
        context,
        visibility_threshold,
        allow_pose_only_wrist,
        admitted_landmarks,
        requires_knee=True,
    )
    quality_context = _record_tolerance(quality_context, neutral_tolerance_deg)
    return REBAInput(
        side=side,
        trunk_flexion_deg=None
        if "trunk_flexion_deg" in unavailable
        else _snap_neutral(float(angles.trunk_flexion), neutral_tolerance_deg),
        trunk_twisted_or_side_flexed=_task_value(task, "trunk_twisted_or_side_flexed"),
        neck_flexion_deg=None
        if "neck_flexion_deg" in unavailable
        else _snap_neutral(float(angles.neck_flexion), neutral_tolerance_deg),
        neck_twisted_or_side_flexed=_task_value(task, "neck_twisted_or_side_flexed"),
        legs_weight_bearing=_task_value(task, "legs_weight_bearing"),
        # Each side is evaluated with its own knee angle. Review asymmetric postures against the worksheet.
        knee_flexion_deg=_angle_value(
            angles, side, "knee_flexion", unavailable, "knee_flexion_deg"
        ),
        seated=_task_value(task, "seated"),
        upper_arm_flexion_deg=_angle_value(
            angles, side, "upper_arm_flexion", unavailable, "upper_arm_flexion_deg"
        ),
        upper_arm_abducted_or_rotated=_task_value(task, f"{side}_upper_arm_abducted_or_rotated"),
        shoulder_raised=_task_value(task, f"{side}_shoulder_raised"),
        arm_supported_or_gravity_assisted=_task_value(
            task, f"{side}_arm_supported_or_gravity_assisted"
        ),
        lower_arm_flexion_deg=_angle_value(
            angles, side, "forearm_flexion", unavailable, "lower_arm_flexion_deg"
        ),
        wrist_flexion_deg=_snap_neutral(
            _angle_value(angles, side, "wrist_flexion", unavailable, "wrist_flexion_deg"),
            neutral_tolerance_deg,
        ),
        wrist_deviated_or_twisted=_task_value(task, f"{side}_wrist_deviated_or_twisted"),
        load_kg=_task_value(task, "load_kg"),
        shock_or_rapid_force=_task_value(task, "shock_or_rapid_force"),
        coupling=_task_value(task, "coupling"),
        static_posture_over_one_minute=_task_value(task, "static_posture_over_one_minute"),
        repeated_more_than_four_per_minute=_task_value(task, "repeated_more_than_four_per_minute"),
        rapid_large_posture_change_or_unstable_base=_task_value(
            task, "rapid_large_posture_change_or_unstable_base"
        ),
        context=quality_context,
    )
