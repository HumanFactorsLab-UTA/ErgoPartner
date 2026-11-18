"""The convention each joint angle is reported in, pinned against known geometry.

RULA and REBA band their scores on specific conventions, and a joint measured in
the mirror convention scores correctly at the band centre while being wrong
either side of it. That is how the elbow shipped reporting a vertex angle
(180 = straight) into tables written for anatomical flexion (0 = straight): every
existing test happened to exercise the band edges in the same wrong convention,
so nothing caught it.

These tests build a limb from explicit coordinates, so the expected angle is
known independently of the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.angle_calculator import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    AngleCalculator,
)
from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator


def _arm(elbow_flexion_deg: float) -> dict[int, np.ndarray]:
    """A left arm bent by a known anatomical flexion.

    Upper arm points straight down from the shoulder. At zero flexion the
    forearm continues straight down; increasing flexion swings the wrist
    forward, which is what bending an elbow does.
    """

    shoulder = np.array([0.0, 1.40, 0.0])
    elbow = np.array([0.0, 1.10, 0.0])
    radians = np.radians(elbow_flexion_deg)
    forearm = np.array([0.0, -np.cos(radians), np.sin(radians)]) * 0.28
    return {
        LEFT_SHOULDER: shoulder,
        LEFT_ELBOW: elbow,
        LEFT_WRIST: elbow + forearm,
    }


def _leg(knee_flexion_deg: float) -> dict[int, np.ndarray]:
    hip = np.array([0.0, 0.95, 0.0])
    knee = np.array([0.0, 0.50, 0.0])
    radians = np.radians(knee_flexion_deg)
    shank = np.array([0.0, -np.cos(radians), np.sin(radians)]) * 0.42
    return {LEFT_HIP: hip, LEFT_KNEE: knee, LEFT_ANKLE: knee + shank}


@pytest.mark.parametrize("flexion", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_elbow_reports_anatomical_flexion(flexion: float):
    """0 degrees is a straight arm, not 180."""

    measured = AngleCalculator()._elbow_angle(_arm(flexion), "left")

    assert measured == pytest.approx(flexion, abs=0.5)


def test_a_straight_arm_reads_zero_not_one_hundred_and_eighty():
    assert AngleCalculator()._elbow_angle(_arm(0.0), "left") == pytest.approx(0.0, abs=0.5)


@pytest.mark.parametrize("flexion", [0.0, 30.0, 60.0, 90.0, 120.0])
def test_knee_reports_anatomical_flexion(flexion: float):
    measured = AngleCalculator()._knee_angle(_leg(flexion), "left")

    assert measured == pytest.approx(flexion, abs=0.5)


def test_elbow_and_knee_share_one_convention():
    """They are the same kind of joint and must not disagree about zero."""

    calculator = AngleCalculator()

    assert calculator._elbow_angle(_arm(0.0), "left") == pytest.approx(
        calculator._knee_angle(_leg(0.0), "left"), abs=0.5
    )
    assert calculator._elbow_angle(_arm(90.0), "left") == pytest.approx(
        calculator._knee_angle(_leg(90.0), "left"), abs=0.5
    )


# ----------------------------------------------------------------------
# The scores the convention feeds
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flexion", "expected"),
    [
        (0.0, 2),  # straight arm, outside the comfortable band
        (45.0, 2),
        (60.0, 1),  # band opens
        (80.0, 1),
        (100.0, 1),  # band closes
        (110.0, 2),  # over-flexed, outside again
        (140.0, 2),
    ],
)
def test_rula_lower_arm_bands_match_the_published_table(flexion: float, expected: int):
    """RULA gives 1 for 60-100 degrees of lower-arm flexion and 2 outside it."""

    assert RULACalculator.score_lower_arm(flexion, False) == expected


@pytest.mark.parametrize(
    ("flexion", "expected"),
    [(0.0, 2), (59.0, 2), (60.0, 1), (100.0, 1), (101.0, 2), (140.0, 2)],
)
def test_reba_lower_arm_bands_match_the_published_table(flexion: float, expected: int):
    assert REBACalculator.score_lower_arm(flexion) == expected


@pytest.mark.parametrize(("flexion", "expected"), [(70.0, 1), (110.0, 2)])
def test_the_cases_the_old_convention_got_backwards(flexion: float, expected: int):
    """The two ranges where vertex and anatomical conventions disagree.

    Under the old vertex convention a 70 degree elbow was passed to the table as
    110 and scored 2, and a 110 degree elbow was passed as 70 and scored 1 --
    each the opposite of the published answer. Measuring the geometry and
    scoring it end to end pins both.
    """

    measured = AngleCalculator()._elbow_angle(_arm(flexion), "left")

    assert measured == pytest.approx(flexion, abs=0.5)
    assert RULACalculator.score_lower_arm(measured, False) == expected
    assert REBACalculator.score_lower_arm(measured) == expected
