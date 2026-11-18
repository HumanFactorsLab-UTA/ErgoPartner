"""
Angle Calculator - Y-UP Convention for MediaPipe World Landmarks.

Computes all joint angles needed for REBA/RULA assessment from 3D pose data.

Coordinate System (MediaPipe pose_world_landmarks):
    X: positive = camera/image right
    Y: positive = UP
    Z: positive = toward camera

    Vertical vector: [0, 1, 0] (upward)

Angle Definitions:
    Trunk flexion:      Trunk axis deviation from GLOBAL vertical in sagittal (YZ) plane
                        (REBA-compliant: captures hip-hinge postures)
    Trunk side bend:    Trunk axis deviation from global vertical in frontal (XY) plane
    Trunk twist:        Shoulder-line vs hip-line rotation in horizontal (XZ) plane
    Neck flexion:       Head direction relative to trunk axis (NOT vertical)
    Upper arm flexion:  Arm vs trunk-down in sagittal (YZ) plane (local shoulder frame)
    Upper arm abduction: Arm vs trunk-down in frontal (XY) plane (local shoulder frame)
    Forearm flexion:    Angle at elbow joint
    Wrist flexion:      Forearm-to-hand angle via palm plane method
    Wrist deviation:    Ulnar/radial deviation via palm plane method
    Wrist pronation:    Palm normal rotation around forearm axis (pronation/supination)
    Shoulder elevation: Clavicle elevation angle relative to mid-shoulder baseline
    Midline crossing:   Boolean — whether forearm crosses the torso sagittal midline
    Knee flexion:       Deviation from straight leg
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def neck_flexion_landmarks(visibility: dict[int, float]) -> list[int]:
    """Pose landmarks used by the head-centroid and thorax-axis calculation."""
    required = [LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if LEFT_EYE in visibility and RIGHT_EYE in visibility:
        required.extend((LEFT_EYE, RIGHT_EYE))
    return required


@dataclass
class BodyAngles:
    """All angles needed for REBA/RULA calculation."""

    # Trunk
    trunk_flexion: float = 0.0  # positive = forward flexion
    trunk_side_bend: float = 0.0  # absolute degrees
    trunk_twist: float = 0.0  # absolute degrees

    # Neck (relative to trunk axis)
    neck_flexion: float = 0.0  # positive = forward (looking down)
    neck_side_bend: float = 0.0  # absolute degrees
    neck_rotation: float = 0.0  # absolute degrees: head yaw against the shoulder line

    # Left arm
    left_upper_arm_flexion: float = 0.0
    left_upper_arm_abduction: float = 0.0
    left_forearm_flexion: float = 90.0
    left_wrist_flexion: float = 0.0
    left_wrist_deviation: float = 0.0
    left_wrist_pronation: float = 0.0  # degrees: positive=pronation, negative=supination

    # Right arm
    right_upper_arm_flexion: float = 0.0
    right_upper_arm_abduction: float = 0.0
    right_forearm_flexion: float = 90.0
    right_wrist_flexion: float = 0.0
    right_wrist_deviation: float = 0.0
    right_wrist_pronation: float = 0.0  # degrees: positive=pronation, negative=supination

    # Legs
    left_knee_flexion: float = 0.0
    right_knee_flexion: float = 0.0

    # Shoulder elevation (RULA +1 penalty if raised)
    left_shoulder_elevation: float = 0.0  # degrees of clavicle elevation
    right_shoulder_elevation: float = 0.0

    # Midline crossing (RULA Step 2 +1 penalty)
    left_forearm_midline_cross: bool = False
    right_forearm_midline_cross: bool = False

    # Active-side shortcuts (set by _set_active_side)
    upper_arm_flexion: float = 0.0
    upper_arm_abduction: float = 0.0
    forearm_flexion: float = 90.0
    wrist_flexion: float = 0.0
    wrist_deviation: float = 0.0
    wrist_pronation: float = 0.0  # degrees: positive=pronation, negative=supination
    knee_flexion: float = 0.0
    shoulder_elevation: float = 0.0  # degrees of clavicle elevation
    forearm_midline_cross: bool = False

    # Per-landmark visibility for multi-view fusion
    landmark_visibility: dict[int, float] = field(default_factory=dict)
    # Per-measurement visibility from the camera actually selected by fusion.
    measurement_visibility: dict[str, float] = field(default_factory=dict)

    # Provenance tracking for wrist angle data source
    wrist_data_source: str = "pose_only"  # "pose_only" or "hand_mesh"
    left_wrist_data_source: str = "pose_only"
    right_wrist_data_source: str = "pose_only"

    # Names of measurements that could not be computed from valid geometry.
    unavailable_measurements: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        payload: dict[str, Any] = {
            "trunk_flexion": self.trunk_flexion,
            "trunk_side_bend": self.trunk_side_bend,
            "trunk_twist": self.trunk_twist,
            "neck_flexion": self.neck_flexion,
            "neck_side_bend": self.neck_side_bend,
            "neck_rotation": self.neck_rotation,
            "left_upper_arm_flexion": self.left_upper_arm_flexion,
            "left_upper_arm_abduction": self.left_upper_arm_abduction,
            "left_forearm_flexion": self.left_forearm_flexion,
            "left_wrist_flexion": self.left_wrist_flexion,
            "left_wrist_deviation": self.left_wrist_deviation,
            "left_wrist_pronation": self.left_wrist_pronation,
            "right_upper_arm_flexion": self.right_upper_arm_flexion,
            "right_upper_arm_abduction": self.right_upper_arm_abduction,
            "right_forearm_flexion": self.right_forearm_flexion,
            "right_wrist_flexion": self.right_wrist_flexion,
            "right_wrist_deviation": self.right_wrist_deviation,
            "right_wrist_pronation": self.right_wrist_pronation,
            "left_knee_flexion": self.left_knee_flexion,
            "right_knee_flexion": self.right_knee_flexion,
            "left_shoulder_elevation": self.left_shoulder_elevation,
            "right_shoulder_elevation": self.right_shoulder_elevation,
            "left_forearm_midline_cross": self.left_forearm_midline_cross,
            "right_forearm_midline_cross": self.right_forearm_midline_cross,
            "left_wrist_data_source": self.left_wrist_data_source,
            "right_wrist_data_source": self.right_wrist_data_source,
            "measurement_visibility": self.measurement_visibility,
            "unavailable_measurements": sorted(set(self.unavailable_measurements)),
        }
        output_aliases = {
            "left_lower_arm_flexion_deg": "left_forearm_flexion",
            "right_lower_arm_flexion_deg": "right_forearm_flexion",
        }
        for measurement in self.unavailable_measurements:
            key = output_aliases.get(measurement, measurement.removesuffix("_deg"))
            if key in payload:
                payload[key] = None
        return payload


class AngleCalculator:
    """
    Y-UP coordinate system angle calculator for ergonomic assessment.

    All methods expect world_landmarks from MediaPipe:
        X = right, Y = UP, Z = toward camera
        Unit: meters, origin: hip midpoint

    Vertical vector: [0, 1, 0]
    """

    def __init__(
        self,
        *,
        neck_flexion_method: str = "legacy_cone",
        arm_abduction_method: str = "frontal_projection",
    ):
        if neck_flexion_method not in {"sagittal_projection", "legacy_cone"}:
            raise ValueError("neck_flexion_method must be sagittal_projection or legacy_cone")
        self.neck_flexion_method = neck_flexion_method
        if arm_abduction_method not in {"frontal_projection", "lateral_excursion"}:
            raise ValueError("arm_abduction_method must be frontal_projection or lateral_excursion")
        self.arm_abduction_method = arm_abduction_method

    # ------------------------------------------------------------------
    # Local anatomical coordinate frame construction
    # ------------------------------------------------------------------

    def _build_pelvis_frame(
        self, joints: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Construct the local pelvis coordinate system.

        Returns (X_local, Y_local, Z_local) unit vectors or None if insufficient landmarks.

        Convention (ISB-recommended for pelvis):
            X_local = LEFT_HIP -> RIGHT_HIP  (lateral, left-to-right)
            Y_local = mid_hip -> mid_shoulder (superior, along spine)
            Z_local = cross(Y_local, X_local) (anterior, out of torso front)

        This frame is invariant to the subject's orientation relative to the camera.
        """
        lh = joints.get(LEFT_HIP)
        rh = joints.get(RIGHT_HIP)
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        if lh is None or rh is None or ls is None or rs is None:
            return None

        mid_hip = (lh + rh) / 2.0
        mid_shoulder = (ls + rs) / 2.0

        # X: lateral axis (left hip to right hip)
        x_local = rh - lh
        xn = np.linalg.norm(x_local)
        if xn < 1e-6:
            return None
        x_local /= xn

        # Y: superior axis (hip midpoint to shoulder midpoint, along spine)
        y_local = mid_shoulder - mid_hip
        yn = np.linalg.norm(y_local)
        if yn < 1e-6:
            return None
        y_local /= yn

        # Z: anterior axis (complete right-handed frame)
        z_local = np.cross(y_local, x_local)
        zn = np.linalg.norm(z_local)
        if zn < 1e-6:
            return None  # degenerate: X and Y are collinear
        z_local /= zn

        # Re-orthogonalize Y to ensure perfect orthogonality after Z is fixed
        y_local = np.cross(x_local, z_local)

        return x_local, y_local, z_local

    def _build_shoulder_frame(
        self, joints: dict[int, np.ndarray], pelvis_y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Construct the local shoulder coordinate system.

        Returns (X_local, Y_local, Z_local) unit vectors or None.

        Convention:
            X_local = LEFT_SHOULDER -> RIGHT_SHOULDER (lateral)
            Y_local = inherited from pelvis superior axis (along spine)
            Z_local = cross(Y_local, X_local) (anterior)

        Args:
            pelvis_y: The superior axis from the pelvis frame. If None, falls back
                      to global vertical [0, 1, 0].
        """
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        if ls is None or rs is None:
            return None

        # X: lateral axis (left shoulder to right shoulder)
        x_local = rs - ls
        xn = np.linalg.norm(x_local)
        if xn < 1e-6:
            return None
        x_local /= xn

        # Y: superior axis (inherited from pelvis, or global vertical fallback)
        if pelvis_y is not None:
            y_local = pelvis_y.copy()
        else:
            y_local = np.array([0.0, 1.0, 0.0])

        # Z: anterior axis
        z_local = np.cross(y_local, x_local)
        zn = np.linalg.norm(z_local)
        if zn < 1e-6:
            return None
        z_local /= zn

        # Re-orthogonalize Y
        y_local = np.cross(x_local, z_local)

        return x_local, y_local, z_local

    def _project_to_local(
        self,
        vec: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
    ) -> np.ndarray:
        """
        Project a global-space vector into a local coordinate frame.

        Args:
            vec: Vector in global (camera) coordinates.
            x_axis, y_axis, z_axis: Orthonormal basis of the local frame.

        Returns:
            Vector expressed in local coordinates [x_local, y_local, z_local].
        """
        return np.array(
            [
                np.dot(vec, x_axis),
                np.dot(vec, y_axis),
                np.dot(vec, z_axis),
            ]
        )

    def calculate_all_angles(
        self,
        joints: dict[int, np.ndarray],
        side: str = "right",
        hand_landmarks: dict[str, dict[int, np.ndarray]] | None = None,
    ) -> BodyAngles:
        """
        Calculate all ergonomic angles from 3D joint positions.

        Args:
            joints: {landmark_id: np.array([x, y, z])} or [x, y, z, visibility]
                    in world coordinates (Y-UP, meters).
            side: "left" or "right" for active-side shortcuts.
            hand_landmarks: Optional dict with keys "left" and/or "right", each mapping
                           to {landmark_id: np.array([x, y, z])} in world coordinates
                           from the 21-point hand mesh. When provided, wrist angles use
                           the full hand mesh for superior accuracy.

        Returns:
            BodyAngles with all computed angles and visibility data.
        """
        result = BodyAngles()

        # Strip visibility component, store separately
        cleaned: dict[int, np.ndarray] = {}
        for k, v in joints.items():
            arr = np.asarray(v, dtype=np.float64)
            result.landmark_visibility[k] = float(arr[3]) if len(arr) > 3 else 1.0
            cleaned[k] = arr[:3]
        joints = cleaned

        def valid_segment(first_id: int, second_id: int) -> bool:
            first = joints.get(first_id)
            second = joints.get(second_id)
            return bool(
                first is not None
                and second is not None
                and np.all(np.isfinite(first))
                and np.all(np.isfinite(second))
                and np.linalg.norm(second - first) >= 1e-6
            )

        trunk_available = (
            valid_segment(LEFT_SHOULDER, LEFT_HIP)
            and valid_segment(RIGHT_SHOULDER, RIGHT_HIP)
            and valid_segment(LEFT_HIP, RIGHT_HIP)
            and valid_segment(LEFT_SHOULDER, RIGHT_SHOULDER)
        )
        if not trunk_available:
            result.unavailable_measurements.extend(
                ["trunk_flexion_deg", "trunk_side_bend_deg", "trunk_twist_deg"]
            )
        neck_available = (
            valid_segment(LEFT_EAR, RIGHT_EAR)
            and valid_segment(LEFT_HIP, RIGHT_HIP)
            and valid_segment(LEFT_SHOULDER, RIGHT_SHOULDER)
        )
        if not neck_available:
            result.unavailable_measurements.extend(["neck_flexion_deg", "neck_side_bend_deg"])
        # Neck rotation needs the ear line and the shoulder line; it does not
        # need the nose, so it has its own availability test.
        neck_rotation_available = valid_segment(LEFT_EAR, RIGHT_EAR) and valid_segment(
            LEFT_SHOULDER, RIGHT_SHOULDER
        )
        if not neck_rotation_available:
            result.unavailable_measurements.append("neck_rotation_deg")
        for (
            side_name,
            shoulder_id,
            elbow_id,
            wrist_id,
            index_id,
            pinky_id,
            hip_id,
            knee_id,
            ankle_id,
        ) in (
            (
                "left",
                LEFT_SHOULDER,
                LEFT_ELBOW,
                LEFT_WRIST,
                LEFT_INDEX,
                LEFT_PINKY,
                LEFT_HIP,
                LEFT_KNEE,
                LEFT_ANKLE,
            ),
            (
                "right",
                RIGHT_SHOULDER,
                RIGHT_ELBOW,
                RIGHT_WRIST,
                RIGHT_INDEX,
                RIGHT_PINKY,
                RIGHT_HIP,
                RIGHT_KNEE,
                RIGHT_ANKLE,
            ),
        ):
            if not valid_segment(shoulder_id, elbow_id):
                result.unavailable_measurements.extend(
                    [
                        f"{side_name}_upper_arm_flexion_deg",
                        f"{side_name}_upper_arm_abduction_deg",
                    ]
                )
            if not (valid_segment(shoulder_id, elbow_id) and valid_segment(elbow_id, wrist_id)):
                result.unavailable_measurements.append(f"{side_name}_lower_arm_flexion_deg")
            if not (
                valid_segment(elbow_id, wrist_id)
                and (valid_segment(wrist_id, index_id) or valid_segment(wrist_id, pinky_id))
            ):
                result.unavailable_measurements.extend(
                    [
                        f"{side_name}_wrist_flexion_deg",
                        f"{side_name}_wrist_deviation_deg",
                        f"{side_name}_wrist_pronation_deg",
                    ]
                )
            if not (valid_segment(hip_id, knee_id) and valid_segment(knee_id, ankle_id)):
                result.unavailable_measurements.append(f"{side_name}_knee_flexion_deg")
            if not (
                valid_segment(elbow_id, wrist_id)
                and valid_segment(LEFT_SHOULDER, RIGHT_SHOULDER)
                and valid_segment(LEFT_HIP, RIGHT_HIP)
            ):
                result.unavailable_measurements.append(f"{side_name}_forearm_midline_cross")

        if not valid_segment(LEFT_SHOULDER, RIGHT_SHOULDER):
            result.unavailable_measurements.extend(
                ["left_shoulder_elevation_deg", "right_shoulder_elevation_deg"]
            )

        # Compute virtual midpoints
        mid_shoulder = self._midpoint(joints, LEFT_SHOULDER, RIGHT_SHOULDER)
        mid_hip = self._midpoint(joints, LEFT_HIP, RIGHT_HIP)

        # ---- BUILD LOCAL ANATOMICAL FRAMES ----
        pelvis_frame = self._build_pelvis_frame(joints)
        shoulder_frame = None
        pelvis_y = None
        if pelvis_frame is not None:
            _, pelvis_y, _ = pelvis_frame
            shoulder_frame = self._build_shoulder_frame(joints, pelvis_y)

        # ---- TRUNK (computed in pelvis local frame) ----
        trunk_axis_unit = None
        if mid_shoulder is not None and mid_hip is not None:
            trunk_axis = mid_shoulder - mid_hip  # points UPWARD (shoulder above hip)
            trunk_norm = np.linalg.norm(trunk_axis)
            if trunk_norm > 1e-6:
                trunk_axis_unit = trunk_axis / trunk_norm

        # Use gravity plus the labelled hip axis rather than camera X/Z as
        # anatomical frontal/sagittal axes.
        trunk_axis_fallback = (
            mid_shoulder - mid_hip if (mid_shoulder is not None and mid_hip is not None) else None
        )
        result.trunk_flexion, result.trunk_side_bend = self._trunk_angles(
            joints, trunk_axis_fallback
        )
        result.trunk_twist = self._trunk_twist(joints)

        # ---- NECK (computed relative to trunk axis, side bend in local frame) ----
        nf, nsb = self._neck_relative_to_trunk(joints, trunk_axis_unit, pelvis_frame)
        if nf is None:
            if "neck_flexion_deg" not in result.unavailable_measurements:
                result.unavailable_measurements.append("neck_flexion_deg")
        else:
            result.neck_flexion = nf
        if nsb is None:
            if "neck_side_bend_deg" not in result.unavailable_measurements:
                result.unavailable_measurements.append("neck_side_bend_deg")
        else:
            result.neck_side_bend = nsb
        result.neck_rotation = self._neck_rotation(joints, pelvis_frame)

        # ---- ARMS (both sides, computed in shoulder local frame) ----
        for s in ("left", "right"):
            if shoulder_frame is not None:
                sx, sy, sz = shoulder_frame
                flex, abd = self._arm_angles_local(joints, s, sx, sy, sz)
            else:
                flex, abd = self._arm_angles(joints, s, trunk_axis_unit)
                if self.arm_abduction_method == "lateral_excursion":
                    result.unavailable_measurements.append(f"{s}_upper_arm_abduction_deg")
            elbow = self._elbow_angle(joints, s)

            # Wrist angles use the hand mesh only when its geometry is complete
            # and non-degenerate. Otherwise provenance remains pose_only, which
            # the assessment adapter rejects by default.
            if hand_landmarks is not None and s in hand_landmarks:
                hand_mesh = hand_landmarks[s]
                shoulder_jt = (
                    joints.get(RIGHT_SHOULDER) if s == "right" else joints.get(LEFT_SHOULDER)
                )
                elbow_jt = joints.get(RIGHT_ELBOW) if s == "right" else joints.get(LEFT_ELBOW)
                wrist_jt = joints.get(RIGHT_WRIST) if s == "right" else joints.get(LEFT_WRIST)
                forearm_vector = None
                if elbow_jt is not None and wrist_jt is not None:
                    forearm_vector = wrist_jt - elbow_jt
                upper_arm_vector = None
                if shoulder_jt is not None and elbow_jt is not None:
                    upper_arm_vector = elbow_jt - shoulder_jt
                if self._hand_mesh_usable(hand_mesh, forearm_vector):
                    wf, wd, wp = self._wrist_angles_from_hand_mesh(
                        hand_mesh, forearm_vector, upper_arm_vector
                    )
                    setattr(result, f"{s}_wrist_data_source", "hand_mesh")
                    for measurement in (
                        f"{s}_wrist_flexion_deg",
                        f"{s}_wrist_deviation_deg",
                        f"{s}_wrist_pronation_deg",
                    ):
                        while measurement in result.unavailable_measurements:
                            result.unavailable_measurements.remove(measurement)
                else:
                    wf, wd, wp = self._wrist_angles(joints, s)
            else:
                wf, wd, wp = self._wrist_angles(joints, s)

            setattr(result, f"{s}_upper_arm_flexion", flex)
            setattr(result, f"{s}_upper_arm_abduction", abd)
            setattr(result, f"{s}_forearm_flexion", elbow)
            setattr(result, f"{s}_wrist_flexion", wf)
            setattr(result, f"{s}_wrist_deviation", wd)
            setattr(result, f"{s}_wrist_pronation", wp)

        # ---- SHOULDER ELEVATION (RULA +1 penalty if raised) ----
        result.left_shoulder_elevation = self._shoulder_elevation(joints, "left")
        result.right_shoulder_elevation = self._shoulder_elevation(joints, "right")

        # ---- FOREARM MIDLINE CROSSING (RULA Step 2 +1 penalty) ----
        result.left_forearm_midline_cross = self._forearm_midline_cross(
            joints, "left", pelvis_frame
        )
        result.right_forearm_midline_cross = self._forearm_midline_cross(
            joints, "right", pelvis_frame
        )

        # ---- LEGS ----
        result.left_knee_flexion = self._knee_angle(joints, "left")
        result.right_knee_flexion = self._knee_angle(joints, "right")
        result.knee_flexion = max(result.left_knee_flexion, result.right_knee_flexion)

        # ---- Active-side shortcuts ----
        self._set_active_side(result, side)
        result.wrist_data_source = getattr(result, f"{side}_wrist_data_source")

        return result

    # ------------------------------------------------------------------
    # Trunk calculations
    # ------------------------------------------------------------------

    def _trunk_angles(
        self, joints: dict[int, np.ndarray], trunk_axis: np.ndarray | None
    ) -> tuple[float, float]:
        """Return signed flexion and unsigned side bend relative to gravity.

        Anatomical right is obtained from labelled hips and projected onto the
        horizontal plane. Anterior is ``vertical × anatomical-right``. This
        avoids treating camera X/Z as anatomical sagittal/frontal axes.
        """

        left_hip = joints.get(LEFT_HIP)
        right_hip = joints.get(RIGHT_HIP)
        if trunk_axis is None or left_hip is None or right_hip is None:
            return 0.0, 0.0
        vertical = np.array([0.0, 1.0, 0.0])
        lateral = right_hip - left_hip
        lateral = lateral - np.dot(lateral, vertical) * vertical
        lateral_norm = np.linalg.norm(lateral)
        trunk_norm = np.linalg.norm(trunk_axis)
        if lateral_norm < 1e-6 or trunk_norm < 1e-6:
            return 0.0, 0.0
        lateral /= lateral_norm
        anterior = np.cross(vertical, lateral)
        anterior /= np.linalg.norm(anterior)

        sagittal = trunk_axis - np.dot(trunk_axis, lateral) * lateral
        frontal = trunk_axis - np.dot(trunk_axis, anterior) * anterior
        flexion = np.degrees(np.arctan2(np.dot(sagittal, anterior), np.dot(sagittal, vertical)))
        side_bend = abs(np.degrees(np.arctan2(np.dot(frontal, lateral), np.dot(frontal, vertical))))
        return float(flexion), float(side_bend)

    def _trunk_twist(self, joints: dict[int, np.ndarray]) -> float:
        """
        Trunk twist: rotation offset between shoulder line and hip line
        projected onto the horizontal (XZ) plane.

        Coordinate-agnostic: works in both Y-UP and Y-DOWN since we
        project to XZ plane (horizontal in both systems).
        """
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        lh = joints.get(LEFT_HIP)
        rh = joints.get(RIGHT_HIP)
        if ls is None or rs is None or lh is None or rh is None:
            return 0.0

        shoulder_vec = rs - ls
        hip_vec = rh - lh

        # Project to XZ plane (horizontal)
        s_xz = np.array([shoulder_vec[0], shoulder_vec[2]])
        h_xz = np.array([hip_vec[0], hip_vec[2]])
        sn = np.linalg.norm(s_xz)
        hn = np.linalg.norm(h_xz)
        if sn < 1e-6 or hn < 1e-6:
            return 0.0
        cos_a = np.dot(s_xz / sn, h_xz / hn)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    # ------------------------------------------------------------------
    # Neck calculations
    # ------------------------------------------------------------------

    def _neck_rotation(
        self,
        joints: dict[int, np.ndarray],
        pelvis_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> float:
        """
        Neck rotation: head yaw relative to the shoulder line.

        Both RULA and REBA penalise a twisted neck. The ear line and the
        shoulder line are close to parallel when the subject looks straight
        ahead, so the angle between them in the transverse plane is the head's
        rotation relative to the torso. This is the same construction
        ``_trunk_twist`` uses for shoulders against hips.

        When a pelvis frame is available both lines are expressed in it first,
        which makes the result invariant to where the camera stands. Otherwise
        the world XZ plane is used, which is correct while the subject is
        upright and degrades as the trunk pitches.

        Returns:
            Absolute rotation in degrees, 0 when the head faces the same way as
            the shoulders.

        ACCURACY NOTE. The ear line is short relative to landmark noise, so this
        is a coarser measurement than trunk twist: expect a few degrees of
        jitter on a stationary head. It resolves the difference between a
        forward-facing and a clearly turned head, which is what the methods ask
        for, but it should not be read as a precise cervical rotation angle.
        A face-mesh head pose would supply that; see ``core/face_landmarker``.
        """
        l_ear = joints.get(LEFT_EAR)
        r_ear = joints.get(RIGHT_EAR)
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        if l_ear is None or r_ear is None or ls is None or rs is None:
            return 0.0

        ear_vec = r_ear - l_ear
        shoulder_vec = rs - ls

        if pelvis_frame is not None:
            px, py, pz = pelvis_frame
            ear_local = self._project_to_local(ear_vec, px, py, pz)
            shoulder_local = self._project_to_local(shoulder_vec, px, py, pz)
            # X is lateral and Z is anterior in the pelvis frame, so dropping Y
            # leaves the transverse plane the rotation happens in.
            ear_flat = np.array([ear_local[0], ear_local[2]])
            shoulder_flat = np.array([shoulder_local[0], shoulder_local[2]])
        else:
            ear_flat = np.array([ear_vec[0], ear_vec[2]])
            shoulder_flat = np.array([shoulder_vec[0], shoulder_vec[2]])

        en = np.linalg.norm(ear_flat)
        sn = np.linalg.norm(shoulder_flat)
        if en < 1e-6 or sn < 1e-6:
            return 0.0
        cos_a = np.dot(ear_flat / en, shoulder_flat / sn)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def _neck_relative_to_trunk(
        self,
        joints: dict[int, np.ndarray],
        trunk_axis_unit: np.ndarray | None,
        pelvis_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> tuple[float | None, float | None]:
        """Signed sagittal inclination of the head centroid relative to the trunk.

        The ear-line inclination supplies the separate side-bend measurement.
        Both are landmark proxies. The legacy method keeps the historical 3-D
        cone magnitude for comparison on archived recordings.
        """
        if trunk_axis_unit is None or not np.all(np.isfinite(trunk_axis_unit)):
            return None, None
        mid_shoulder = self._midpoint(joints, LEFT_SHOULDER, RIGHT_SHOULDER)
        left_ear, right_ear = joints.get(LEFT_EAR), joints.get(RIGHT_EAR)
        if mid_shoulder is None or left_ear is None or right_ear is None:
            return None, None

        ear_line = right_ear - left_ear
        ear_width = np.linalg.norm(ear_line)
        if not np.isfinite(ear_width) or ear_width < 1e-6:
            return None, None
        superior = pelvis_frame[1] if pelvis_frame is not None else trunk_axis_unit
        side_bend = float(
            np.degrees(np.arcsin(np.clip(abs(np.dot(ear_line, superior)) / ear_width, 0, 1)))
        )

        head_points = [left_ear, right_ear]
        if LEFT_EYE in joints and RIGHT_EYE in joints:
            head_points.extend((joints[LEFT_EYE], joints[RIGHT_EYE]))
        head = np.mean(head_points, axis=0) - mid_shoulder
        head_length = np.linalg.norm(head)
        if not np.isfinite(head_length) or head_length < 1e-6:
            return None, side_bend
        head = head / head_length

        if self.neck_flexion_method == "legacy_cone":
            cosine = np.clip(np.dot(head, trunk_axis_unit), -1, 1)
            magnitude = float(np.degrees(np.arccos(cosine)))
            sign = (
                np.dot(head, pelvis_frame[2])
                if pelvis_frame is not None
                else np.cross(trunk_axis_unit, head)[0]
            )
            return (-magnitude if sign < 0 else magnitude), side_bend

        shoulder_line = joints[RIGHT_SHOULDER] - joints[LEFT_SHOULDER]
        anterior = np.cross(trunk_axis_unit, shoulder_line)
        anterior_length = np.linalg.norm(anterior)
        if not np.isfinite(anterior_length) or anterior_length < 1e-6:
            return None, side_bend
        forward = np.dot(head, anterior / anterior_length)
        up = np.dot(head, trunk_axis_unit)
        if np.hypot(forward, up) < 1e-6:
            return None, side_bend
        return float(np.degrees(np.arctan2(forward, up))), side_bend

    # ------------------------------------------------------------------
    # Arm calculations (both sides)
    # ------------------------------------------------------------------

    def _arm_angles(
        self,
        joints: dict[int, np.ndarray],
        side: str,
        trunk_axis_unit: np.ndarray | None,
    ) -> tuple[float, float]:
        """
        Upper arm flexion (sagittal, signed) and abduction (frontal, unsigned).

        Measured relative to trunk-down direction.

        Returns:
            flexion:  Positive = forward flexion (arm in front of torso),
                      Negative = extension (arm behind torso).
            abduction: Always non-negative lateral deviation from sagittal plane.
        """
        if side == "right":
            shoulder = joints.get(RIGHT_SHOULDER)
            elbow = joints.get(RIGHT_ELBOW)
        else:
            shoulder = joints.get(LEFT_SHOULDER)
            elbow = joints.get(LEFT_ELBOW)

        if shoulder is None or elbow is None:
            return 0.0, 0.0

        arm = elbow - shoulder
        an = np.linalg.norm(arm)
        if an < 1e-6:
            return 0.0, 0.0

        # Trunk-down direction: opposite of trunk axis (from shoulder toward hip)
        if trunk_axis_unit is not None:
            trunk_down = -trunk_axis_unit
        else:
            trunk_down = np.array([0.0, -1.0, 0.0])  # default: straight down

        # --- Flexion in SAGITTAL plane (YZ) — SIGNED ---
        # Zero out X to isolate forward/backward movement
        arm_sag = np.array([0.0, arm[1], arm[2]])
        trunk_sag = np.array([0.0, trunk_down[1], trunk_down[2]])
        asn = np.linalg.norm(arm_sag)
        tsn = np.linalg.norm(trunk_sag)
        flexion = 0.0
        if asn > 1e-6 and tsn > 1e-6:
            cos_a = np.dot(arm_sag / asn, trunk_sag / tsn)
            flexion = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        # Sign: Z > 0 in world coords = elbow forward of shoulder = flexion
        #       Z < 0 = elbow behind shoulder = extension (negative)
        flexion = flexion if arm[2] >= 0 else -flexion

        # --- Abduction in FRONTAL plane (XY) — UNSIGNED ---
        # Zero out Z to isolate lateral movement
        arm_fron = np.array([arm[0], arm[1], 0.0])
        trunk_fron = np.array([trunk_down[0], trunk_down[1], 0.0])
        afn = np.linalg.norm(arm_fron)
        tfn = np.linalg.norm(trunk_fron)
        abduction = 0.0
        if afn > 1e-6 and tfn > 1e-6:
            cos_a = np.dot(arm_fron / afn, trunk_fron / tfn)
            abduction = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        return flexion, abs(abduction)

    def _elbow_angle(self, joints: dict[int, np.ndarray], side: str) -> float:
        """
        Lower-arm flexion at the elbow: 0 = straight arm, 90 = right angle.

        Anatomical flexion, measured from full extension, which is the
        convention RULA and REBA band their lower-arm scores on: both give 1 for
        60-100 degrees and 2 outside it.

        This used to return the raw vertex angle, where a straight arm reads 180
        rather than 0. The two conventions are mirror images about 90, so the
        band happened to coincide at exactly 90 degrees and diverged either side
        of it: an arm at 70 degrees of flexion scored 2 when the table says 1,
        and one at 110 scored 1 when the table says 2. That is a common working
        range, and the error was invisible because the tests exercised the band
        edges in the same wrong convention. The knee has always converted this
        way -- see ``_knee_angle`` -- so the elbow was the odd one out.
        """
        if side == "right":
            s, e, w = joints.get(RIGHT_SHOULDER), joints.get(RIGHT_ELBOW), joints.get(RIGHT_WRIST)
        else:
            s, e, w = joints.get(LEFT_SHOULDER), joints.get(LEFT_ELBOW), joints.get(LEFT_WRIST)

        if s is None or e is None or w is None:
            return 90.0
        return 180.0 - self._angle_at_vertex(s, e, w)

    def _knee_angle(self, joints: dict[int, np.ndarray], side: str) -> float:
        """
        Knee flexion: 0 = straight leg, 90 = right-angle bend.
        """
        if side == "right":
            h, k, a = joints.get(RIGHT_HIP), joints.get(RIGHT_KNEE), joints.get(RIGHT_ANKLE)
        else:
            h, k, a = joints.get(LEFT_HIP), joints.get(LEFT_KNEE), joints.get(LEFT_ANKLE)

        if h is None or k is None or a is None:
            return 0.0
        return 180.0 - self._angle_at_vertex(h, k, a)

    # ------------------------------------------------------------------
    # Wrist calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _wrist_angles_from_geometry(
        forearm: np.ndarray,
        radial: np.ndarray,
        ulnar: np.ndarray,
        distal: np.ndarray,
        upper_arm: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """Wrist flexion, deviation and pronation from anatomically named vectors.

        One implementation serves both the pose-only and hand-mesh paths, which
        previously carried separate copies of the same geometry and the same two
        errors. Every input is defined anatomically rather than by coordinate
        handedness, so the signs hold for both hands without a per-side special
        case.

        Args:
            forearm: elbow -> wrist (distal along the forearm).
            radial:  wrist -> a radial (thumb-side) hand landmark.
            ulnar:   wrist -> an ulnar (little-finger-side) hand landmark.
            distal:  wrist -> a point along the hand's long axis.
            upper_arm: shoulder -> elbow. Required for pronation, which is
                referenced to the elbow flexion plane; pronation is reported as
                0.0 when it is absent or the elbow is straight.

        Returns:
            (flexion, deviation, pronation) in degrees.
                flexion   positive = palmar flexion, negative = dorsal extension
                deviation positive = ulnar, negative = radial
                pronation positive = palm rotating toward the elbow flexion
                          plane normal; see the note on validity below.

        A neutral wrist -- hand collinear with the forearm and in the palm plane
        -- returns 0 for flexion and deviation. The previous implementation
        returned 180 for deviation in exactly that posture, because it measured
        against the radial-ulnar axis rather than the hand's long axis and then
        subtracted from 180.

        Pronation is reported for completeness only. No ergonomic method in this
        project consumes it: RULA takes wrist twist and REBA takes wrist
        deviation/twist as observed task inputs, not from pose. Its zero is the
        elbow flexion plane, which is a real anatomical reference but has not
        been validated against instrumented measurement.
        """

        forearm_norm = np.linalg.norm(forearm)
        if forearm_norm < 1e-6:
            return 0.0, 0.0, 0.0
        forearm_unit = forearm / forearm_norm

        # Palm plane from two anatomically defined in-plane directions. The
        # operand order matters: cross(distal, radial->ulnar) points out of the
        # palm, and reversing it points out of the back of the hand, which
        # silently inverts the sign of flexion.
        ulnar_axis = ulnar - radial
        normal = np.cross(distal, ulnar_axis)
        normal_norm = np.linalg.norm(normal)
        distal_norm = np.linalg.norm(distal)
        ulnar_norm = np.linalg.norm(ulnar_axis)
        if normal_norm < 1e-6 or distal_norm < 1e-6 or ulnar_norm < 1e-6:
            return 0.0, 0.0, 0.0
        palm_normal = normal / normal_norm
        hand_axis = distal / distal_norm
        ulnar_unit = ulnar_axis / ulnar_norm

        # Flexion: how far the forearm lies out of the palm plane. Zero when the
        # forearm lies in that plane, which is the neutral wrist.
        flexion = float(
            np.degrees(np.arcsin(np.clip(np.dot(forearm_unit, palm_normal), -1.0, 1.0)))
        )

        # Deviation: angle between the forearm and the hand's long axis, both
        # projected into the palm plane, signed toward the ulnar side.
        forearm_in_plane = forearm_unit - np.dot(forearm_unit, palm_normal) * palm_normal
        in_plane_norm = np.linalg.norm(forearm_in_plane)
        deviation = 0.0
        if in_plane_norm > 1e-6:
            forearm_in_plane = forearm_in_plane / in_plane_norm
            magnitude = np.degrees(
                np.arccos(np.clip(np.dot(forearm_in_plane, hand_axis), -1.0, 1.0))
            )
            perpendicular = hand_axis - np.dot(hand_axis, forearm_in_plane) * forearm_in_plane
            direction = float(np.dot(perpendicular, ulnar_unit))
            deviation = float(magnitude if direction >= 0 else -magnitude)

        # Pronation: signed rotation of the palm normal about the forearm axis,
        # measured from the elbow flexion plane. arctan2 keeps it monotonic and
        # able to separate pronation from supination; the previous code applied
        # arcsin to a cosine, which folded +45 and -45 onto the same value.
        pronation = 0.0
        if upper_arm is not None:
            reference = np.cross(upper_arm, forearm)
            reference_norm = np.linalg.norm(reference)
            if reference_norm > 1e-6:
                reference = reference / reference_norm
                pronation = float(
                    np.degrees(
                        np.arctan2(
                            np.dot(np.cross(reference, palm_normal), forearm_unit),
                            np.dot(reference, palm_normal),
                        )
                    )
                )
        return flexion, deviation, pronation

    @staticmethod
    def _hand_mesh_usable(
        hand_landmarks: dict[int, np.ndarray], forearm_vector: np.ndarray | None
    ) -> bool:
        """Require finite, non-degenerate geometry for the hand-mesh path."""

        required = (0, 5, 9, 13, 17)
        if forearm_vector is None or np.linalg.norm(forearm_vector) < 1e-6:
            return False
        if not np.all(np.isfinite(forearm_vector)):
            return False
        if any(index not in hand_landmarks for index in required):
            return False
        points = {index: np.asarray(hand_landmarks[index], dtype=float)[:3] for index in required}
        if any(point.shape != (3,) or not np.all(np.isfinite(point)) for point in points.values()):
            return False
        wrist = points[0]
        if any(np.linalg.norm(points[index] - wrist) < 1e-6 for index in required[1:]):
            return False
        radial_ulnar = points[17] - points[5]
        palm_center = (points[5] + points[17]) / 2.0 - wrist
        return bool(np.linalg.norm(np.cross(radial_ulnar, palm_center)) >= 1e-6)

    def _wrist_angles(self, joints: dict[int, np.ndarray], side: str) -> tuple[float, float, float]:
        """
        Wrist flexion/extension, ulnar/radial deviation, and pronation/supination.

        Returns signed values:
            - Positive flexion  = palmar flexion (bending toward palm)
            - Negative flexion  = dorsal extension (bending backward)
            - Positive deviation = ulnar deviation
            - Negative deviation = radial deviation
            - Positive pronation = palm-down rotation, negative = palm-up (supination)

        Uses 3D palm-plane method when index + pinky landmarks available.
        Falls back to forearm-to-hand angle when only index is available.
        Pronation is estimated from the palm normal vs. global vertical in the
        forearm's transverse plane.
        """
        if side == "right":
            shoulder = joints.get(RIGHT_SHOULDER)
            elbow = joints.get(RIGHT_ELBOW)
            wrist = joints.get(RIGHT_WRIST)
            index = joints.get(RIGHT_INDEX)
            pinky = joints.get(RIGHT_PINKY)
        else:
            shoulder = joints.get(LEFT_SHOULDER)
            elbow = joints.get(LEFT_ELBOW)
            wrist = joints.get(LEFT_WRIST)
            index = joints.get(LEFT_INDEX)
            pinky = joints.get(LEFT_PINKY)

        if elbow is None or wrist is None:
            return 0.0, 0.0, 0.0

        forearm = wrist - elbow
        if np.linalg.norm(forearm) < 1e-6:
            return 0.0, 0.0, 0.0

        # Full palm plane: index is radial, pinky is ulnar, and their midpoint
        # gives the hand's long axis.
        if index is not None and pinky is not None:
            return self._wrist_angles_from_geometry(
                forearm=forearm,
                radial=index - wrist,
                ulnar=pinky - wrist,
                distal=((index - wrist) + (pinky - wrist)) / 2.0,
                upper_arm=(elbow - shoulder) if shoulder is not None else None,
            )

        # Fallback: one hand point gives an unsigned deflection only. Direction
        # is genuinely unknown without the palm plane, so it is not invented.
        if index is not None:
            hand_vec = index - wrist
            hand_norm = np.linalg.norm(hand_vec)
            if hand_norm > 1e-6:
                cos_a = np.dot(forearm / np.linalg.norm(forearm), hand_vec / hand_norm)
                straight_angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                return abs(180.0 - straight_angle), 0.0, 0.0

        return 0.0, 0.0, 0.0

    def _wrist_angles_from_hand_mesh(
        self,
        hand_landmarks: dict[int, np.ndarray],
        forearm_vector: np.ndarray | None,
        upper_arm_vector: np.ndarray | None = None,
    ) -> tuple[float, float, float]:
        """
        Wrist flexion/extension, ulnar/radial deviation, and pronation/supination
        from full 21-point hand mesh.

        Uses MCP joints (indices 5, 9, 13, 17) to construct a stable palm plane that
        is invariant to finger flexion. Pronation/supination is computed from the
        palm normal relative to the forearm's coronal plane.

        Hand landmark indices (MediaPipe Hand):
            0   = Wrist center
            1   = Thumb CMC
            2   = Thumb MCP
            3   = Thumb IP
            4   = Thumb tip
            5   = Index MCP (radial side of palm)
            9   = Middle MCP (palm center proxy)
            13  = Ring MCP
            17  = Pinky MCP (ulnar side of palm)

        Args:
            hand_landmarks: {landmark_id: np.array([x, y, z])} in world coordinates.
            forearm_vector: Elbow-to-wrist vector from the pose model.  Only its
                direction is combined with hand vectors because the two models
                do not share a coordinate origin.

        Returns:
            (flexion, deviation, pronation) in degrees.
                - Positive flexion  = palmar flexion
                - Negative flexion  = dorsal extension
                - Positive deviation = ulnar deviation
                - Negative deviation = radial deviation
                - Positive pronation = palm-down rotation
                - Negative pronation = palm-up (supination)
        """
        wrist = hand_landmarks.get(0)
        index_mcp = hand_landmarks.get(5)
        middle_mcp = hand_landmarks.get(9)
        pinky_mcp = hand_landmarks.get(17)

        if wrist is None or forearm_vector is None:
            return 0.0, 0.0, 0.0
        forearm = np.asarray(forearm_vector, dtype=np.float64)
        if np.linalg.norm(forearm) < 1e-6:
            return 0.0, 0.0, 0.0

        if index_mcp is not None and pinky_mcp is not None:
            # The third metacarpal is the standard hand long axis; fall back to
            # the index/pinky midpoint when the middle MCP is missing.
            distal = (
                middle_mcp - wrist
                if middle_mcp is not None
                else ((index_mcp + pinky_mcp) / 2.0 - wrist)
            )
            return self._wrist_angles_from_geometry(
                forearm=forearm,
                radial=index_mcp - wrist,
                ulnar=pinky_mcp - wrist,
                distal=distal,
                upper_arm=upper_arm_vector,
            )

        # Fallback: wrist and middle MCP give an unsigned deflection only.
        if middle_mcp is not None:
            palm_vec = middle_mcp - wrist
            palm_norm = np.linalg.norm(palm_vec)
            if palm_norm > 1e-6:
                cos_a = np.dot(forearm / np.linalg.norm(forearm), palm_vec / palm_norm)
                straight_angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                return abs(180.0 - straight_angle), 0.0, 0.0

        return 0.0, 0.0, 0.0

    # ------------------------------------------------------------------
    # Arm calculations in local anatomical frame (camera-invariant)
    # ------------------------------------------------------------------

    def _arm_angles_local(
        self,
        joints: dict[int, np.ndarray],
        side: str,
        sx: np.ndarray,
        sy: np.ndarray,
        sz: np.ndarray,
    ) -> tuple[float, float]:
        """
        Upper arm flexion (sagittal, signed) and abduction (frontal, unsigned)
        computed in the shoulder local frame.

        Measured relative to trunk-down direction expressed in local coords.
        This is camera-invariant because it uses the anatomical shoulder frame.

        Args:
            joints: Cleaned joint positions.
            side: "left" or "right".
            sx, sy, sz: Orthonormal basis of the shoulder local frame.

        Returns:
            flexion:  Positive = forward flexion (arm in front of torso),
                      Negative = extension (arm behind torso).
            abduction: Always non-negative lateral deviation from sagittal plane.
        """
        if side == "right":
            shoulder = joints.get(RIGHT_SHOULDER)
            elbow = joints.get(RIGHT_ELBOW)
        else:
            shoulder = joints.get(LEFT_SHOULDER)
            elbow = joints.get(LEFT_ELBOW)

        if shoulder is None or elbow is None:
            return 0.0, 0.0

        arm = elbow - shoulder
        an = np.linalg.norm(arm)
        if an < 1e-6:
            return 0.0, 0.0

        # Express arm vector in local shoulder frame
        arm_local = self._project_to_local(arm, sx, sy, sz)

        # Trunk-down in local frame: trunk axis is +Y (superior), so down is -Y
        trunk_down_local = np.array([0.0, -1.0, 0.0])

        # --- Flexion in SAGITTAL plane (local YZ) — SIGNED ---
        # Zero out X to isolate forward/backward movement
        arm_sag = np.array([0.0, arm_local[1], arm_local[2]])
        asn = np.linalg.norm(arm_sag)
        flexion = 0.0
        if asn > 1e-6:
            cos_a = np.dot(arm_sag / asn, trunk_down_local)
            flexion = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        # Sign: positive Z in local frame = anterior = forward flexion
        flexion = flexion if arm_local[2] >= 0 else -flexion

        if self.arm_abduction_method == "lateral_excursion":
            # The shoulder X axis points from anatomical left to right. The
            # angle out of the sagittal plane remains defined for a forward
            # overhead reach, where a frontal projection can flip to 180°.
            outward = arm_local[0] if side == "right" else -arm_local[0]
            excursion = np.degrees(np.arctan2(max(0.0, outward), np.hypot(arm_local[1], arm_local[2])))
            return float(flexion), float(excursion)

        # --- Abduction in FRONTAL plane (local XY) — UNSIGNED ---
        # Zero out Z to isolate lateral movement
        arm_fron = np.array([arm_local[0], arm_local[1], 0.0])
        afn = np.linalg.norm(arm_fron)
        abduction = 0.0
        if afn > 1e-6:
            cos_a = np.dot(arm_fron / afn, trunk_down_local)
            abduction = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        return flexion, abs(abduction)

    # ------------------------------------------------------------------
    # Shoulder elevation (RULA +1 penalty if raised)
    # ------------------------------------------------------------------

    def _shoulder_elevation(
        self,
        joints: dict[int, np.ndarray],
        side: str,
    ) -> float:
        """
        Measure shoulder elevation (shrugging) relative to the subject's own torso.

        RULA adds +1 if the shoulder is raised. Elevation is the angle between the
        hip-to-shoulder vector on this side and the trunk's superior axis: as the
        shoulder rises toward the ear it swings closer to the spine axis, and that
        change is what this reports.

        The previous implementation compared each shoulder against the opposite
        shoulder's height, which measures asymmetry rather than elevation. A
        bilateral shrug -- both shoulders raised, which is the common ergonomic
        case and exactly the posture RULA penalises -- cancelled to zero and was
        invisible.

        Args:
            joints: Cleaned joint positions.
            side: "left" or "right".

        Returns:
            Degrees by which this shoulder sits above the opposite one.

        KNOWN LIMITATION -- asymmetry only. This detects a one-sided shrug and is
        structurally blind to a bilateral one: raising both shoulders together
        cancels out and reports exactly zero, which is the common ergonomic case
        and the posture RULA penalises.

        An ear-referenced measure was trialled to close that gap. It does detect
        a bilateral shrug, but its neutral reference has to be anthropometric
        rather than subject-specific, and on real footage it read 14-27 degrees
        of elevation on every frame including upright ones. A number that is
        always non-zero is more misleading to an ergonomist than one that is
        honestly insensitive to a known case, so it was not adopted.

        Measuring bilateral elevation properly needs a recorded neutral posture
        per subject, which this pipeline does not capture. Nothing is scored from
        this value: RULA and REBA both take ``shoulder_raised`` as an observed
        task input.
        """
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        if ls is None or rs is None:
            return 0.0

        target_shoulder = rs if side == "right" else ls
        other_shoulder = ls if side == "right" else rs
        shoulder_width = np.linalg.norm(rs - ls)
        if shoulder_width < 1e-6:
            return 0.0

        rise = max(float(target_shoulder[1] - other_shoulder[1]), 0.0)
        ratio = min(rise / float(shoulder_width), 1.0)
        return float(np.degrees(np.arcsin(ratio)))

    # ------------------------------------------------------------------
    # Forearm midline crossing (RULA Step 2 +1 penalty)
    # ------------------------------------------------------------------

    def _forearm_midline_cross(
        self,
        joints: dict[int, np.ndarray],
        side: str,
        pelvis_frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> bool:
        """
        Determine if the forearm crosses the midline of the body.

        RULA Step 2 adds +1 if the lower arm is working across the midline
        of the body or out to the side. We project the forearm vector into
        the body's local transverse (XZ) plane and check if it crosses
        the torso's sagittal midline.

        Args:
            joints: Cleaned joint positions.
            side: "left" or "right".
            pelvis_frame: Optional local anatomical frame for camera-invariance.

        Returns:
            True if forearm crosses the body midline.
        """
        if side == "right":
            elbow = joints.get(RIGHT_ELBOW)
            wrist = joints.get(RIGHT_WRIST)
        else:
            elbow = joints.get(LEFT_ELBOW)
            wrist = joints.get(LEFT_WRIST)

        if elbow is None or wrist is None:
            return False

        forearm_vec = wrist - elbow
        fn = np.linalg.norm(forearm_vec)
        if fn < 1e-6:
            return False

        # Get midline reference
        ls = joints.get(LEFT_SHOULDER)
        rs = joints.get(RIGHT_SHOULDER)
        lh = joints.get(LEFT_HIP)
        rh = joints.get(RIGHT_HIP)

        if ls is None or rs is None or lh is None or rh is None:
            return False

        mid_shoulder = (ls + rs) / 2.0
        mid_hip = (lh + rh) / 2.0

        if pelvis_frame is not None:
            px, py, pz = pelvis_frame
            # Express forearm and elbow in local pelvis frame
            forearm_local = self._project_to_local(forearm_vec, px, py, pz)
            elbow_global = elbow
            elbow_local = self._project_to_local(elbow_global - mid_hip, px, py, pz)

            # In local frame, X-axis is lateral (left-to-right).
            # Midline is at X=0. Right arm crossing midline means wrist X < 0.
            # Left arm crossing midline means wrist X > 0.
            wrist_local = elbow_local + forearm_local

            if side == "right":
                return wrist_local[0] < 0.0
            else:
                return wrist_local[0] > 0.0
        else:
            # Fallback: use global X-axis
            # Right arm crosses midline if wrist is to the left of mid-shoulder (smaller X)
            # Left arm crosses midline if wrist is to the right of mid-shoulder (larger X)
            wrist_x = wrist[0]
            mid_x = mid_shoulder[0]

            if side == "right":
                return wrist_x < mid_x
            else:
                return wrist_x > mid_x

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _midpoint(joints: dict[int, np.ndarray], left_id: int, right_id: int) -> np.ndarray | None:
        """Compute midpoint of two landmarks, or None if either is missing."""
        left = joints.get(left_id)
        right = joints.get(right_id)
        if left is not None and right is not None:
            return (left + right) / 2.0
        return None

    @staticmethod
    def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at vertex b formed by rays BA and BC, in degrees."""
        ba = a - b
        bc = c - b
        ban = np.linalg.norm(ba)
        bcn = np.linalg.norm(bc)
        if ban < 1e-6 or bcn < 1e-6:
            return 0.0
        cos_a = np.dot(ba, bc) / (ban * bcn)
        return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    @staticmethod
    def _set_active_side(result: BodyAngles, side: str):
        """Copy the specified side's values to the active-side shortcuts."""
        if side == "left":
            result.upper_arm_flexion = result.left_upper_arm_flexion
            result.upper_arm_abduction = result.left_upper_arm_abduction
            result.forearm_flexion = result.left_forearm_flexion
            result.wrist_flexion = result.left_wrist_flexion
            result.wrist_deviation = result.left_wrist_deviation
            result.wrist_pronation = result.left_wrist_pronation
            result.shoulder_elevation = result.left_shoulder_elevation
            result.forearm_midline_cross = result.left_forearm_midline_cross
        else:
            result.upper_arm_flexion = result.right_upper_arm_flexion
            result.upper_arm_abduction = result.right_upper_arm_abduction
            result.forearm_flexion = result.right_forearm_flexion
            result.wrist_flexion = result.right_wrist_flexion
            result.wrist_deviation = result.right_wrist_deviation
            result.wrist_pronation = result.right_wrist_pronation
            result.shoulder_elevation = result.right_shoulder_elevation
            result.forearm_midline_cross = result.right_forearm_midline_cross
