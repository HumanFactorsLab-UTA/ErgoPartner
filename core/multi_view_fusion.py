"""
Multi-View Fusion - Per-Angle Best-View Selection.

When multiple cameras are available, selects the best angle estimate
from the camera whose view geometry is optimal for that measurement
AND whose landmark visibility is highest.

Strategy:
    - Sagittal angles (flexion/extension) -> prefer "side" camera
    - Frontal angles (abduction, side-bend) -> prefer "front" camera
    - Transverse angles (twist) -> any camera with good visibility
    - Fallback: visibility-weighted selection
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .angle_calculator import (
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_INDEX,
    LEFT_KNEE,
    LEFT_PINKY,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_INDEX,
    RIGHT_KNEE,
    RIGHT_PINKY,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    BodyAngles,
    neck_flexion_landmarks,
)


@dataclass
class AngleViewPreference:
    """Mapping of an angle to preferred camera view and required landmarks."""

    preferred_view: str  # "side", "front", or "any"
    required_landmarks: list[int]


# Which camera view is geometrically best for each angle
ANGLE_VIEW_MAP: dict[str, AngleViewPreference] = {
    # Sagittal plane (best seen from side)
    "trunk_flexion": AngleViewPreference(
        "side", [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    ),
    "neck_flexion": AngleViewPreference("side", neck_flexion_landmarks({})),
    "left_upper_arm_flexion": AngleViewPreference("side", [LEFT_SHOULDER, LEFT_ELBOW]),
    "right_upper_arm_flexion": AngleViewPreference("side", [RIGHT_SHOULDER, RIGHT_ELBOW]),
    "left_forearm_flexion": AngleViewPreference("side", [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]),
    "right_forearm_flexion": AngleViewPreference(
        "side", [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
    ),
    "left_knee_flexion": AngleViewPreference("side", [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]),
    "right_knee_flexion": AngleViewPreference("side", [RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]),
    "left_wrist_flexion": AngleViewPreference(
        "side", [LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX, LEFT_PINKY]
    ),
    "right_wrist_flexion": AngleViewPreference(
        "side", [RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY]
    ),
    # Frontal plane (best seen from front)
    "left_upper_arm_abduction": AngleViewPreference("front", [LEFT_SHOULDER, LEFT_ELBOW]),
    "right_upper_arm_abduction": AngleViewPreference("front", [RIGHT_SHOULDER, RIGHT_ELBOW]),
    "trunk_side_bend": AngleViewPreference(
        "front", [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    ),
    "neck_side_bend": AngleViewPreference(
        "front", [LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    ),
    "left_wrist_deviation": AngleViewPreference(
        "front", [LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX, LEFT_PINKY]
    ),
    "right_wrist_deviation": AngleViewPreference(
        "front", [RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY]
    ),
    # Transverse plane (any view)
    "trunk_twist": AngleViewPreference("any", [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]),
    "neck_rotation": AngleViewPreference(
        "any", [LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER]
    ),
    # Wrist pronation (best from front — palm orientation visible)
    "left_wrist_pronation": AngleViewPreference(
        "front", [LEFT_ELBOW, LEFT_WRIST, LEFT_INDEX, LEFT_PINKY]
    ),
    "right_wrist_pronation": AngleViewPreference(
        "front", [RIGHT_ELBOW, RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY]
    ),
    # Shoulder elevation (best from front — Y-difference between shoulders)
    "left_shoulder_elevation": AngleViewPreference(
        "front", [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    ),
    "right_shoulder_elevation": AngleViewPreference(
        "front", [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    ),
    # Midline crossing (any view — uses local anatomical frame)
    "left_forearm_midline_cross": AngleViewPreference(
        "any", [LEFT_ELBOW, LEFT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER]
    ),
    "right_forearm_midline_cross": AngleViewPreference(
        "any", [RIGHT_ELBOW, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER]
    ),
}

# Entries in ANGLE_VIEW_MAP that are boolean rather than angular. Selection
# treats every field numerically, so these are restored to bool afterwards.
BOOLEAN_ANGLES = frozenset({"left_forearm_midline_cross", "right_forearm_midline_cross"})


def _measurement_name(angle_name: str) -> str:
    """Method-side missingness key for a BodyAngles attribute."""
    if angle_name in BOOLEAN_ANGLES:
        return angle_name
    return angle_name.replace("forearm_flexion", "lower_arm_flexion") + "_deg"


def _is_missing(angles: BodyAngles, angle_name: str) -> bool:
    # Old fused outputs tagged every attribute with "_deg", including
    # booleans and the elbow attribute named forearm_flexion in the file.
    names = angles.unavailable_measurements
    return _measurement_name(angle_name) in names or f"{angle_name}_deg" in names


#: Landmarks the sequence-level analyses depend on: hands and knuckles for the
#: RNLE lifting geometry, hips through feet for seated detection, leg weight
#: bearing, and the floor reference.
SEQUENCE_LANDMARKS: tuple[int, ...] = (
    15,
    16,
    19,
    20,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
)


def select_landmark_source(
    per_camera_landmarks: Sequence[dict[int, np.ndarray]],
    required_ids: Sequence[int] = SEQUENCE_LANDMARKS,
    visibility_threshold: float = 0.5,
) -> tuple[int, dict[int, np.ndarray]]:
    """Pick the camera whose landmarks best support the sequence analyses.

    The angle path already chooses a view per measurement. The sequence-level
    analyses -- lift geometry, seated detection, leg weight bearing -- consume
    whole landmark positions instead, and they were taking camera zero
    unconditionally. A leg occluded in that one view disappeared from the
    analysis even when another camera had it in full sight.

    Landmarks are **not** merged across cameras. Each camera's world landmarks
    are hip-centred in that camera's own orientation, so combining them without
    extrinsic calibration yields a skeleton that is internally inconsistent --
    the reason this project selects a view per angle rather than triangulating.
    Choosing one camera per frame keeps every frame coherent while letting the
    choice follow the evidence.

    Returns:
        The chosen camera index and its landmarks. Index 0 when nothing is
        usable, so the caller always has a defined source.
    """

    best_index = 0
    best_key = (-1, -1.0)
    for index, landmarks in enumerate(per_camera_landmarks):
        visible = 0
        total = 0.0
        for landmark_id in required_ids:
            point = landmarks.get(landmark_id)
            if point is None:
                continue
            array = np.asarray(point, dtype=float)
            if array.size < 3 or not np.all(np.isfinite(array[:3])):
                continue
            visibility = float(array[3]) if array.size > 3 else 1.0
            total += visibility
            if visibility >= visibility_threshold:
                visible += 1
        key = (visible, total)
        if key > best_key:
            best_key, best_index = key, index
    return best_index, (dict(per_camera_landmarks[best_index]) if per_camera_landmarks else {})


class MultiViewFusion:
    """
    Fuse angle estimates from multiple cameras using per-angle best-view selection.

    For each body angle:
        1. Prefer the camera with the correct geometric view (side for sagittal, etc.)
        2. Among cameras with the preferred view, pick highest landmark visibility
        3. Fall back to any camera with highest visibility if preferred view unavailable
    """

    def __init__(
        self,
        camera_roles: dict[str, int] | None = None,
        visibility_threshold: float = 0.5,
        outlier_margin_deg: float = 25.0,
        *,
        prefer_hand_mesh: bool = False,
    ):
        """
        Args:
            camera_roles: {"front": 0, "side": 1, "iso": 2} mapping role to video index
            visibility_threshold: Minimum avg visibility to consider a camera's estimate
            outlier_margin_deg: How far the preferred view may sit from the median
                of all views before the median is used instead. Needs three or
                more cameras to apply. Set 0 to always trust the preferred view.
                25 degrees is wider than the disagreement between two sound views
                of the same joint and narrower than a view that has the posture
                wrong.
        """
        if not isinstance(prefer_hand_mesh, bool):
            raise ValueError("prefer_hand_mesh must be a boolean")
        if not np.isfinite(visibility_threshold) or not 0 <= visibility_threshold <= 1:
            raise ValueError("visibility_threshold must be between 0 and 1")
        if not np.isfinite(outlier_margin_deg) or outlier_margin_deg < 0:
            raise ValueError("outlier_margin_deg must be a finite nonnegative angle")
        self.camera_roles = camera_roles or {}
        self.vis_threshold = visibility_threshold
        self.outlier_margin_deg = outlier_margin_deg
        self.prefer_hand_mesh = prefer_hand_mesh
        # Reverse map: camera_index -> role
        self.index_to_role: dict[int, str] = {v: k for k, v in self.camera_roles.items()}

    def fuse_angles(
        self,
        per_camera_angles: list[BodyAngles],
        camera_indices: list[int],
    ) -> tuple[BodyAngles, dict[str, str]]:
        """
        Fuse angle estimates from multiple cameras into one BodyAngles.

        Args:
            per_camera_angles: BodyAngles from each camera
            camera_indices: Camera index for each entry

        Returns:
            Tuple of (fused BodyAngles, selection_map).
            selection_map: {angle_name: camera_role} showing which camera
            was selected as best view for each angle.
        """
        selection_map: dict[str, str] = {}

        if len(per_camera_angles) == 1:
            role = self.index_to_role.get(camera_indices[0], f"cam_{camera_indices[0]}")
            for angle_name in ANGLE_VIEW_MAP:
                selection_map[angle_name] = (
                    "unavailable" if _is_missing(per_camera_angles[0], angle_name) else role
                )
            return per_camera_angles[0], selection_map

        result = BodyAngles()

        # Merge all visibility data (keep highest per landmark)
        for angles in per_camera_angles:
            for lm_id, vis in angles.landmark_visibility.items():
                current = result.landmark_visibility.get(lm_id, 0.0)
                result.landmark_visibility[lm_id] = max(current, vis)

        # Select best value for each angle, tracking which camera was chosen
        for angle_name, pref in ANGLE_VIEW_MAP.items():
            best_value, best_role, selected_visibility = self._select_best(
                angle_name, pref, per_camera_angles, camera_indices
            )
            # Midline crossing is a boolean. Storing the float from _select_best
            # made multi-camera runs emit 1.0/0.0 where single-camera runs emit
            # true/false, so the same field had two types across runs.
            if angle_name in BOOLEAN_ANGLES:
                setattr(
                    result, angle_name, bool(best_value) if best_role != "unavailable" else False
                )
            else:
                setattr(result, angle_name, best_value)
            selection_map[angle_name] = best_role
            result.measurement_visibility[f"{angle_name}_deg"] = selected_visibility
            if best_role == "unavailable":
                result.unavailable_measurements.append(_measurement_name(angle_name))

        # Set derived values
        result.knee_flexion = max(result.left_knee_flexion, result.right_knee_flexion)

        # Propagate the source from the camera actually selected for each wrist.
        role_to_angles = {
            self.index_to_role.get(index, f"cam_{index}"): angles
            for angles, index in zip(per_camera_angles, camera_indices, strict=True)
        }
        for side in ("left", "right"):
            role = selection_map.get(f"{side}_wrist_flexion", "unavailable")
            source_angles = role_to_angles.get(role.removesuffix("+consensus"))
            source = (
                getattr(source_angles, f"{side}_wrist_data_source", "pose_only")
                if source_angles is not None
                else "pose_only"
            )
            setattr(result, f"{side}_wrist_data_source", source)

        return result, selection_map

    def _select_best(
        self,
        angle_name: str,
        pref: AngleViewPreference,
        angles_list: list[BodyAngles],
        cam_indices: list[int],
    ) -> tuple[float, str, float]:
        """Return value, camera role, and minimum required-landmark visibility."""
        candidates: list[tuple[float, float, bool, str]] = []

        mesh_candidates: list[tuple[float, float, bool, str]] = []
        for angles, cam_idx in zip(angles_list, cam_indices, strict=True):
            if _is_missing(angles, angle_name):
                continue
            hand_mesh = (
                "_wrist_" in angle_name
                and getattr(angles, f"{angle_name.split('_', 1)[0]}_wrist_data_source")
                == "hand_mesh"
            )
            required = pref.required_landmarks
            if angle_name == "neck_flexion":
                required = neck_flexion_landmarks(angles.landmark_visibility)
            elif hand_mesh:
                # The validated hand mesh supplies the hand axis. Only the
                # forearm direction still comes from the pose model.
                required = required[:2]
            visibility = [float(angles.landmark_visibility.get(i, 0.0)) for i in required]
            if not all(map(math.isfinite, visibility)):
                continue
            minimum_visibility = min(visibility)
            if minimum_visibility < self.vis_threshold:
                continue

            # Check if this camera has the preferred role
            role = self.index_to_role.get(cam_idx, f"cam_{cam_idx}")
            role_match = (role == pref.preferred_view) or (pref.preferred_view == "any")

            value = float(getattr(angles, angle_name, float("nan")))
            if not math.isfinite(value):
                continue
            candidate = (value, minimum_visibility, role_match, role)
            candidates.append(candidate)
            if hand_mesh:
                mesh_candidates.append(candidate)

        if self.prefer_hand_mesh and mesh_candidates:
            candidates = mesh_candidates
        if not candidates:
            return float("nan"), "unavailable", 0.0

        # Sort by: role_match (True first), then visibility (descending)
        candidates.sort(key=lambda c: (-int(c[2]), -c[1]))
        preferred_value, preferred_visibility, _, preferred_role = candidates[0]

        # Three cameras give three independent estimates of the same anatomical
        # angle. Taking only the preferred view throws two of them away, and if
        # that one view is wrong the error goes straight into the score with
        # nothing to catch it: on a bench trial the side camera read the neck at
        # -14 degrees while the other two read +60, and the outlier won, scoring
        # the neck at RULA's maximum on every frame.
        #
        # Where the preferred view disagrees with the consensus of the others by
        # more than the outlier margin, the median is used instead. This is not
        # triangulation and makes no claim to calibrated 3-D: it is the median of
        # repeated measurements of one quantity, which is the standard defence
        # against a single bad reading.
        if len(candidates) < 3 or self.outlier_margin_deg <= 0:
            return preferred_value, preferred_role, preferred_visibility
        if angle_name in BOOLEAN_ANGLES:
            return preferred_value, preferred_role, preferred_visibility

        values = [item[0] for item in candidates]
        median = statistics.median(values)
        if abs(preferred_value - median) <= self.outlier_margin_deg:
            return preferred_value, preferred_role, preferred_visibility

        # For an even count the median can lie between cameras. Use the
        # closest observed value so the reported source supplied the result.
        value, selected_visibility, _, role = min(
            candidates, key=lambda item: abs(item[0] - median)
        )
        return value, f"{role}+consensus", selected_visibility

    def set_active_side(self, result: BodyAngles, side: str):
        """Set active-side shortcuts on fused result."""
        if side == "left":
            result.upper_arm_flexion = result.left_upper_arm_flexion
            result.upper_arm_abduction = result.left_upper_arm_abduction
            result.forearm_flexion = result.left_forearm_flexion
            result.wrist_flexion = result.left_wrist_flexion
            result.wrist_deviation = result.left_wrist_deviation
            result.wrist_pronation = result.left_wrist_pronation
        else:
            result.upper_arm_flexion = result.right_upper_arm_flexion
            result.upper_arm_abduction = result.right_upper_arm_abduction
            result.forearm_flexion = result.right_forearm_flexion
            result.wrist_flexion = result.right_wrist_flexion
            result.wrist_deviation = result.right_wrist_deviation
            result.wrist_pronation = result.right_wrist_pronation
        result.wrist_data_source = getattr(result, f"{side}_wrist_data_source")
