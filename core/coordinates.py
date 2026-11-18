"""Canonical coordinate definitions and transformations.

MediaPipe normalized image landmarks remain x-right/y-down normalized image
coordinates.  MediaPipe model world outputs are transformed once at their
acquisition boundary into the project convention below.
"""

from __future__ import annotations

import numpy as np

CANONICAL_COORDINATE_SYSTEM = {
    "origin": "model-estimated midpoint between hips",
    "x_positive": "camera/image right",
    "y_positive": "up",
    "z_positive": "toward camera",
    "units": "model-estimated metres",
    "anatomical_side": "MediaPipe landmark label, never image side",
}


def mediapipe_world_to_canonical(
    x: float, y: float, z: float, visibility: float = 1.0
) -> np.ndarray:
    """Convert MediaPipe x-right/y-down/z-away output to canonical axes."""

    values = np.asarray([x, -y, -z, visibility], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("world landmark coordinates and visibility must be finite")
    return values
