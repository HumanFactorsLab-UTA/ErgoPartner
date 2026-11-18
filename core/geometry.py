"""Coordinate-agnostic vector primitives used by biomechanical calculations.

Functions accept 2-D or 3-D vectors, return degrees, clamp roundoff before
inverse trigonometric functions, and reject undefined zero-length geometry.
"""

from __future__ import annotations

import numpy as np


class GeometryError(ValueError):
    """Raised when a requested geometric quantity is undefined."""


def _vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size not in (2, 3):
        raise GeometryError(f"{name} must be a finite 2-D or 3-D vector")
    if not np.all(np.isfinite(vector)):
        raise GeometryError(f"{name} must contain only finite values")
    return vector


def _unit(value, name: str) -> np.ndarray:
    vector = _vector(value, name)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise GeometryError(f"{name} is a zero-length vector")
    return vector / norm


def angle_between(first, second) -> float:
    """Return the unsigned angle in [0, 180] degrees."""

    first_unit = _unit(first, "first vector")
    second_unit = _unit(second, "second vector")
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def signed_angle(first, second, positive_normal) -> float:
    """Return the signed 3-D angle in [-180, 180] about ``positive_normal``.

    The right-hand rule defines positive rotation.  All inputs must use the
    same coordinate frame and units.
    """

    first_unit = _unit(first, "first vector")
    second_unit = _unit(second, "second vector")
    normal_unit = _unit(positive_normal, "positive normal")
    if first_unit.size != 3 or second_unit.size != 3 or normal_unit.size != 3:
        raise GeometryError("signed_angle requires three 3-D vectors")
    sine = float(np.dot(normal_unit, np.cross(first_unit, second_unit)))
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))
