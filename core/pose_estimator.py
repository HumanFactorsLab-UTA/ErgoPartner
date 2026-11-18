"""
MediaPipe Pose Estimator - Wrapper for pose landmark detection.

Uses MediaPipe PoseLandmarker (tasks API, 33 body landmarks) to detect
human pose from video frames.
Returns both normalized (pixel-space) and world (3D meters) landmark coordinates.

Coordinate Systems:
  normalized: X right [0,1], Y DOWN [0,1], Z depth
  world:      X right (m),   Y UP (m),     Z toward camera (m), hip-centered
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cv2
import mediapipe as mp  # type: ignore[import-untyped]
import numpy as np

from core.coordinates import mediapipe_world_to_canonical
from core.model_assets import ModelIntegrityError, verify_sha256

# New tasks API imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model download URLs by variant
_MODEL_URLS = {
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}
_MODEL_SHA256 = {
    "heavy": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b",
}


@dataclass
class PoseLandmarks:
    """Pose detection result for one frame from one camera."""

    # {landmark_id: np.array([x, y, z, visibility])}
    normalized: dict[int, np.ndarray] = field(default_factory=dict)
    # {landmark_id: np.array([x, y, z, visibility])} in meters, Y-UP, hip-centered
    world: dict[int, np.ndarray] = field(default_factory=dict)
    timestamp: float = 0.0
    frame_number: int = 0
    image_size: tuple[int, int] = (0, 0)  # (width, height)
    mean_keypoint_visibility: float = 0.0

    @property
    def detection_confidence(self) -> float:
        """Compatibility alias; this is not detector confidence."""
        return self.mean_keypoint_visibility


class MultiplePeopleDetectedError(RuntimeError):
    """Raised by the single-person compatibility API on an ambiguous frame."""


class GPUDelegateUnavailableError(RuntimeError):
    """Raised when GPU inference was requested but the delegate cannot be created.

    The MediaPipe Python wheels for Windows are built without GPU support, so
    ``use_gpu=True`` would otherwise degrade silently to CPU and roughly halve
    expected throughput while reporting success. Callers that genuinely accept
    CPU must opt in through ``allow_cpu_fallback``.
    """


def _ensure_model(
    model_path: str, variant: str = "heavy", expected_sha256: str | None = None
) -> str:
    """Download the pose landmarker model if not present."""
    expected_sha256 = expected_sha256 or _MODEL_SHA256.get(variant)
    if not expected_sha256:
        raise ModelIntegrityError(f"No pinned checksum is configured for pose model {variant!r}")
    if os.path.isfile(model_path):
        verify_sha256(model_path, expected_sha256)
        return model_path

    url = _MODEL_URLS.get(variant, _MODEL_URLS["heavy"])
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"  Downloading PoseLandmarker model ({variant})...")
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


class PoseEstimator:
    """
    MediaPipe PoseLandmarker wrapper (tasks API).

    Uses mp.tasks.vision.PoseLandmarker (33 landmarks) for pose estimation.
    Extracts both normalized and world_landmarks per frame.
    """

    # Landmark indices for reference
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

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_heavy.task",
        model_variant: str = "heavy",  # Controls which .task file is loaded ("lite"/"full"/"heavy")
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_gpu: bool = False,
        max_poses: int = 4,
        expected_model_sha256: str | None = None,
        allow_cpu_fallback: bool = False,
        independent_images: bool = False,
    ):
        if max_poses < 1:
            raise ValueError("max_poses must be at least 1")
        # Ensure model file exists (auto-download if needed)
        model_path = _ensure_model(model_path, model_variant, expected_model_sha256)
        self._independent_images = independent_images

        def build(delegate) -> Any:
            return PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=model_path,
                        delegate=delegate,
                    ),
                    running_mode=VisionRunningMode.IMAGE if independent_images else VisionRunningMode.VIDEO,
                    num_poses=max_poses,
                    min_pose_detection_confidence=min_detection_confidence,
                    min_pose_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    output_segmentation_masks=False,
                )
            )

        # Build the real landmarker directly on the requested delegate. The
        # previous implementation created a throwaway probe landmarker first,
        # which paid the model-load cost twice on every camera.
        landmarker: Any = None
        self._using_gpu = False
        if use_gpu:
            try:
                landmarker = build(BaseOptions.Delegate.GPU)
                self._using_gpu = True
            except Exception as exc:
                if not allow_cpu_fallback:
                    raise GPUDelegateUnavailableError(
                        "GPU inference was requested but the delegate could not be "
                        f"created: {exc}. The MediaPipe Python wheel for this platform "
                        "is commonly built without GPU support. Set "
                        "pose.allow_cpu_fallback: true to accept CPU inference, or "
                        "pose.use_gpu: false if CPU is intended."
                    ) from exc
                print(f"  GPU delegate unavailable ({exc}); falling back to CPU as configured")
        self._landmarker: Any = (
            landmarker if landmarker is not None else build(BaseOptions.Delegate.CPU)
        )
        self._last_timestamp_ms = -1

    def process_frame_all(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_number: int = 0,
    ) -> list[PoseLandmarks]:
        """
        Process one BGR frame and return pose landmarks.

        Args:
            frame: BGR image from cv2.VideoCapture
            timestamp: time in seconds from video start
            frame_number: 0-based frame index

        Returns:
            Every detected pose. Ordering is detector output ordering and must not
            be used as identity; pass the list to a tracker.
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

        if self._independent_images:
            result = self._landmarker.detect(mp_image)
        else:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        # Check if any pose was detected
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return []

        poses: list[PoseLandmarks] = []
        for pose_index, pose_landmarks in enumerate(result.pose_landmarks):
            normalized = {
                i: np.array([lm.x, lm.y, lm.z, lm.visibility], dtype=np.float64)
                for i, lm in enumerate(pose_landmarks)
            }
            world = {}
            if result.pose_world_landmarks and pose_index < len(result.pose_world_landmarks):
                for i, lm in enumerate(result.pose_world_landmarks[pose_index]):
                    # Canonical project convention: camera-aligned X right,
                    # Y up, Z toward camera; origin is the model's hip midpoint.
                    world[i] = mediapipe_world_to_canonical(lm.x, lm.y, lm.z, lm.visibility)
            key_landmarks = [11, 12, 23, 24, 0]
            values = [normalized[i][3] for i in key_landmarks if i in normalized]
            mean_visibility = float(np.mean(values)) if values else 0.0
            poses.append(
                PoseLandmarks(
                    normalized=normalized,
                    world=world,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    image_size=(w, h),
                    mean_keypoint_visibility=mean_visibility,
                )
            )
        return poses

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_number: int = 0,
    ) -> PoseLandmarks | None:
        """Compatibility single-person API that refuses ambiguous frames."""

        poses = self.process_frame_all(frame, timestamp, frame_number)
        if len(poses) > 1:
            raise MultiplePeopleDetectedError(
                f"Detected {len(poses)} people; use process_frame_all with identity tracking"
            )
        return poses[0] if poses else None

    def reset(self):
        """Reset is not directly supported in the new API. No-op for compatibility."""
        pass

    def close(self):
        """Release MediaPipe resources."""
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
