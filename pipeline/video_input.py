"""Timestamp-preserving OpenCV video input with explicit fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedFrame:
    frame_number: int
    timestamp_sec: float
    image_bgr: np.ndarray
    timestamp_source: str


class VideoReader:
    """Decode frames and prefer decoder timestamps over frame/FPS arithmetic."""

    def __init__(self, path: str | Path, *, fps_override: float | None = None):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise VideoInputError(f"video does not exist: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise VideoInputError(f"OpenCV could not open video: {self.path}")
        native_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = float(fps_override) if fps_override is not None else native_fps
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.capture.release()
            raise VideoInputError("video FPS is invalid; provide a positive fps_override")
        self.native_fps = native_fps
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rotation_deg = 0.0
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            self.rotation_deg = float(self.capture.get(cv2.CAP_PROP_ORIENTATION_META))
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        self._index = 0
        self._last_timestamp = -1.0
        self.timestamp_fallback_count = 0

    def read(self) -> DecodedFrame | None:
        ok, image = self.capture.read()
        if not ok:
            return None
        decoder_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
        decoder_timestamp = decoder_ms / 1000.0
        if (
            np.isfinite(decoder_timestamp)
            and decoder_timestamp >= 0
            and decoder_timestamp > self._last_timestamp
        ):
            timestamp = decoder_timestamp
            source = "decoder_pts"
        elif self._index == 0 and decoder_timestamp == 0:
            timestamp = 0.0
            source = "decoder_pts"
        else:
            timestamp = self._index / self.fps
            source = "frame_index_over_fps_fallback"
            self.timestamp_fallback_count += 1
            if timestamp <= self._last_timestamp:
                timestamp = self._last_timestamp + 1.0 / self.fps
        result = DecodedFrame(self._index, timestamp, image, source)
        self._index += 1
        self._last_timestamp = timestamp
        return result

    def close(self) -> None:
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
