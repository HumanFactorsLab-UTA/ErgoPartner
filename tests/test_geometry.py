"""Synthetic geometry tests isolate mathematics from pose-model accuracy."""

import numpy as np
import pytest

from core.geometry import GeometryError, angle_between, signed_angle


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], 90.0),
        ([1, 0, 0], [-1, 0, 0], 180.0),
    ],
)
def test_known_unsigned_angles(first, second, expected):
    assert angle_between(first, second) == pytest.approx(expected)


def test_signed_angle_direction():
    assert signed_angle([1, 0, 0], [0, 1, 0], [0, 0, 1]) == pytest.approx(90)
    assert signed_angle([1, 0, 0], [0, -1, 0], [0, 0, 1]) == pytest.approx(-90)


def test_zero_length_vector_is_explicit_error_not_zero_degrees():
    with pytest.raises(GeometryError, match="zero-length"):
        angle_between([0, 0, 0], [1, 0, 0])


def test_cosine_is_clamped_before_arccos():
    # Nearly collinear large values exercise floating-point roundoff.
    result = angle_between(np.array([1e150, 1.0, 0.0]), np.array([1e150, 1.000001, 0.0]))
    assert np.isfinite(result)
    assert result == pytest.approx(0.0, abs=1e-9)
