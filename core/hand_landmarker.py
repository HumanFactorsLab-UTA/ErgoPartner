"""
MediaPipe Hand Landmarker - Wrapper for 21-point hand mesh detection.

Runs with PoseLandmarker to provide model-estimated 3-D hand landmarks for
wrist angle computation. Monocular model output is not calibrated motion
capture and is reported with that limitation.

Coordinate System (MediaPipe hand_world_landmarks):
    Origin: Hand geometric center (separate from the pose-landmark origin)
    Units: Meters
    X: image/camera right before canonical conversion; anatomical laterality is
       determined only by the model's labelled left/right landmarks
    Y: DOWN (negated to UP in this wrapper for consistency with pose model)
    Z: toward camera (negated for consistency with pose model)

Hand Landmark Indices (MediaPipe Hand):
    0   = Wrist
    1-4 = Thumb (CMC, MCP, IP, TIP)
    5-8 = Index finger (MCP, PIP, DIP, TIP)
    9-12 = Middle finger (MCP, PIP, DIP, TIP)
    13-16 = Ring finger (MCP, PIP, DIP, TIP)
    17-20 = Pinky (MCP, PIP, DIP, TIP)

Key landmarks for wrist angle computation:
    0   = Wrist center (stable anchor point)
    5   = Index MCP (radial side of palm)
    9   = Middle MCP (palm center proxy)
    13  = Ring MCP
    17  = Pinky MCP (ulnar side of palm)
"""

import math
import os
import urllib.request
from dataclasses import dataclass, field

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np

from core.coordinates import mediapipe_world_to_canonical
from core.model_assets import ModelIntegrityError, verify_sha256

# New tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model download URLs by variant.
#
# Unlike the pose landmarker, MediaPipe publishes only one hand bundle and it
# lives at the unqualified path. The per-variant URLs this used to list
# (.../hand_landmarker/full/... and .../lite/...) return a 251-byte XML error
# rather than a model, so an auto-download on a machine without the artifact
# fetched an error page and failed the checksum with a misleading message.
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_URLS = {
    "full": _HAND_MODEL_URL,
    "lite": _HAND_MODEL_URL,
}


@dataclass
class HandLandmarks:
    """Hand detection result for one frame from one camera."""

    # {landmark_id: np.array([x, y, z, visibility])} in meters, Y-UP, wrist-centered
    left_hand: dict[int, np.ndarray] = field(default_factory=dict)
    right_hand: dict[int, np.ndarray] = field(default_factory=dict)

    timestamp: float = 0.0
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    detection_confidence: float = 0.0
    #: Landmarks the model returned as non-finite and this wrapper dropped.
    #: Retained so a run can report how often the hand mesh was incomplete
    #: rather than presenting a partial hand as a whole one.
    discarded_non_finite_landmarks: int = 0


def _ensure_model(
    model_path: str, variant: str = "full", expected_sha256: str | None = None
) -> str:
    """Download the hand landmarker model if not present."""
    if not expected_sha256:
        raise ModelIntegrityError(f"No pinned checksum is configured for hand model {variant!r}")
    if os.path.isfile(model_path):
        verify_sha256(model_path, expected_sha256)
        return model_path

    url = _MODEL_URLS.get(variant, _MODEL_URLS["full"])
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"  Downloading HandLandmarker model ({variant})...")
    print(f"    URL: {url}")
    print(f"    Destination: {model_path}")
    temporary_path = model_path + ".download"
    try:
        urllib.request.urlretrieve(url, temporary_path)
        verify_sha256(temporary_path, expected_sha256)
        os.replace(temporary_path, model_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"    Downloaded: {size_mb:.1f} MB")
    return model_path


class HandLandmarkerWrapper:
    """
    MediaPipe HandLandmarker wrapper (tasks API).

    Uses mp.tasks.vision.HandLandmarker for 21-point hand mesh detection.
    Extracts normalized landmarks per frame, separated by left/right hand.
    """

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        model_variant: str = "full",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = False,
        expected_model_sha256: str | None = None,
    ):
        # Ensure model file exists (auto-download if needed)
        model_path = _ensure_model(model_path, model_variant, expected_model_sha256)

        # Try GPU delegate if requested, fall back to CPU on failure
        delegate = BaseOptions.Delegate.CPU
        if use_gpu:
            try:
                test_options = HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=model_path,
                        delegate=BaseOptions.Delegate.GPU,
                    ),
                    running_mode=VisionRunningMode.IMAGE,
                    num_hands=2,  # FIXED: HandLandmarker uses num_hands, not num_poses
                    min_hand_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                test_lm = HandLandmarker.create_from_options(test_options)
                test_lm.close()
                delegate = BaseOptions.Delegate.GPU
                print("  Hand GPU delegate: enabled")
            except Exception as e:
                print(f"  Hand GPU delegate unavailable ({e}), using CPU")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=delegate,
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,  # FIXED: HandLandmarker uses num_hands, not num_poses
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0
        # Detection confidence gates model output only. It is not a claim of
        # anatomical-angle validity; the assessment adapter reports that separately.
        self._min_confidence = min_detection_confidence
        self._ergonomic_confidence_threshold = 0.5

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_number: int = 0,
        pose_norm_landmarks: dict[int, np.ndarray] | None = None,
    ) -> HandLandmarks | None:
        """
        Process one BGR frame and return hand landmarks.

        Args:
            frame: BGR image from cv2.VideoCapture
            timestamp: time in seconds from video start
            frame_number: 0-based frame index
            pose_norm_landmarks: Optional normalized pose landmarks from PoseEstimator.
                                Used to spatially lock hand meshes to the correct arm
                                via 2D Euclidean distance (overrides MediaPipe handedness).

        Returns:
            HandLandmarks with left/right hand world coordinates (meters, Y-UP),
            or None if no hands detected or confidence too low for ergonomic use.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create mp.Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must be in milliseconds, monotonically increasing
        timestamp_ms = int(timestamp * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # Check if any hands were detected
        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        left_hand: dict[int, np.ndarray] = {}
        right_hand: dict[int, np.ndarray] = {}
        total_confidence = []
        non_finite_landmarks = 0

        for idx, (hand_norm_list, hand_classification) in enumerate(
            zip(result.hand_landmarks, result.handedness, strict=True)
        ):
            conf_score = hand_classification[0].score

            # Occlusion safety: skip hands with low confidence (power grips, tool occlusion)
            if conf_score < self._ergonomic_confidence_threshold:
                continue

            total_confidence.append(conf_score)

            # 1. Start with MediaPipe's native guess
            assigned_side = hand_classification[0].category_name  # "Left" or "Right"

            # 2. OVERRIDE: Spatial lock using 2D Normalized Euclidean Distance
            if pose_norm_landmarks is not None:
                hand_wrist_norm = hand_norm_list[0]
                hx, hy = hand_wrist_norm.x, hand_wrist_norm.y

                left_pose_wrist = pose_norm_landmarks.get(15)  # MediaPipe LEFT_WRIST
                right_pose_wrist = pose_norm_landmarks.get(16)  # MediaPipe RIGHT_WRIST

                dist_left = float("inf")
                dist_right = float("inf")

                # Calculate 2D Euclidean distance (screen space, normalized [0,1])
                if left_pose_wrist is not None:
                    dist_left = math.hypot(hx - left_pose_wrist[0], hy - left_pose_wrist[1])
                if right_pose_wrist is not None:
                    dist_right = math.hypot(hx - right_pose_wrist[0], hy - right_pose_wrist[1])

                # Max threshold (15% of screen dimension) to reject background hands
                MAX_DIST = 0.15
                if min(dist_left, dist_right) > MAX_DIST:
                    continue  # Hand too far from worker's wrists; likely background person

                # Reassign based strictly on spatial proximity
                assigned_side = "Left" if dist_left < dist_right else "Right"

            target_dict = left_hand if assigned_side == "Left" else right_hand

            # Require world landmarks. Normalized image coordinates have
            # different units and must never be mixed with pose world vectors.
            has_world = (
                result.hand_world_landmarks is not None
                and len(result.hand_world_landmarks) > idx
                and len(result.hand_world_landmarks[idx]) > 0
            )

            if has_world:
                world_list = result.hand_world_landmarks[idx]
                for i, lm in enumerate(world_list):
                    # MediaPipe world coords: Y-DOWN, Z-away-from-camera.
                    # Negate Y and Z to match our pose model convention (Y-UP, Z-toward-camera).
                    #
                    # The hand model populates visibility as an explicit None
                    # rather than omitting it, so a getattr default never fires
                    # and None reached the finite check. That made every hand
                    # landmark unusable and silently pinned wrist_data_source to
                    # pose_only whenever the hand model was switched on. The
                    # hand model reports no per-landmark visibility at all, so
                    # full confidence is the correct reading; the detection
                    # confidence for the hand as a whole is recorded separately.
                    raw_visibility = getattr(lm, "visibility", None)
                    vis = 1.0 if raw_visibility is None else float(raw_visibility)
                    try:
                        target_dict[i] = mediapipe_world_to_canonical(lm.x, lm.y, lm.z, vis)
                    except ValueError:
                        # The hand model occasionally emits a non-finite
                        # coordinate on a partly occluded hand. That landmark is
                        # missing evidence, and the wrist angle simply goes
                        # unavailable for this frame -- the same outcome as a
                        # hand the model did not find at all. Aborting the run
                        # over one bad landmark would discard an entire trial.
                        non_finite_landmarks += 1
                        continue
            else:
                continue

        avg_conf = float(np.mean(total_confidence)) if total_confidence else 0.0

        # If all hands were filtered out by confidence threshold, or every
        # landmark of the ones that survived was non-finite, return None.
        if not left_hand and not right_hand:
            return None

        return HandLandmarks(
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp=timestamp,
            frame_number=frame_number,
            image_size=(w, h),
            detection_confidence=avg_conf,
            discarded_non_finite_landmarks=non_finite_landmarks,
        )

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
