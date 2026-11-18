"""The CV/method boundary reports near-neutral angles as neutral.

RULA and REBA anchor their lowest trunk, neck and wrist bands at exactly 0
degrees, faithfully to the published figures. Continuous pose data never lands
on exactly zero, so without a tolerance those bands are unreachable from video
and an upright subject scores one band high on every frame.
"""

import pytest

from core.angle_calculator import BodyAngles
from ergonomics.adapters import (
    DEFAULT_NEUTRAL_TOLERANCE_DEG,
    build_reba_input,
    build_rula_input,
)
from ergonomics.common import AssessmentContext
from ergonomics.rula import RULACalculator


def _angles(**overrides) -> BodyAngles:
    angles = BodyAngles(
        wrist_data_source="hand_mesh",
        left_wrist_data_source="hand_mesh",
        right_wrist_data_source="hand_mesh",
        **overrides,
    )
    angles.landmark_visibility = {index: 0.99 for index in range(33)}
    return angles


def _rula(angles: BodyAngles, tolerance: float = DEFAULT_NEUTRAL_TOLERANCE_DEG):
    return build_rula_input(
        angles,
        "left",
        {},
        AssessmentContext(),
        visibility_threshold=0.5,
        neutral_tolerance_deg=tolerance,
    )


def _reba(angles: BodyAngles, tolerance: float = DEFAULT_NEUTRAL_TOLERANCE_DEG):
    return build_reba_input(
        angles,
        "left",
        {},
        AssessmentContext(),
        visibility_threshold=0.5,
        neutral_tolerance_deg=tolerance,
    )


@pytest.mark.parametrize("angle", [0.0, 0.0001, 0.9, 1.999, -0.5, -1.999])
def test_near_neutral_trunk_is_reported_as_neutral(angle):
    assert _rula(_angles(trunk_flexion=angle)).trunk_flexion_deg == 0.0
    assert _reba(_angles(trunk_flexion=angle)).trunk_flexion_deg == 0.0


@pytest.mark.parametrize("angle", [2.001, 5.0, -2.001, -30.0])
def test_angles_beyond_tolerance_pass_through_unchanged(angle):
    assert _rula(_angles(trunk_flexion=angle)).trunk_flexion_deg == pytest.approx(angle)


def test_neck_and_wrist_are_snapped_too():
    angles = _angles(neck_flexion=-1.2, left_wrist_flexion=0.8)
    rula = _rula(angles)
    assert rula.neck_flexion_deg == 0.0
    assert rula.wrist_flexion_deg == 0.0


def test_upper_arm_and_knee_are_not_snapped():
    """Their lowest bands are ranges, not a single point, so they need no tolerance."""

    angles = _angles(left_upper_arm_flexion=1.0, left_knee_flexion=1.0)
    assert _rula(angles).upper_arm_flexion_deg == pytest.approx(1.0)
    assert _reba(angles).knee_flexion_deg == pytest.approx(1.0)


def test_tolerance_of_zero_disables_snapping():
    angles = _angles(trunk_flexion=0.5)
    assert _rula(angles, tolerance=0.0).trunk_flexion_deg == pytest.approx(0.5)


def test_upright_subject_can_now_reach_the_neutral_band():
    """The behaviour this exists for: 0.4 degrees is upright, not mild flexion."""

    measured = 0.4
    without = RULACalculator.score_trunk(measured, False, False)
    with_tolerance = RULACalculator.score_trunk(
        _rula(_angles(trunk_flexion=measured)).trunk_flexion_deg, False, False
    )
    assert without == 2
    assert with_tolerance == 1


def test_small_backward_lean_no_longer_invalidates_rula():
    """RULA rejects negative trunk flexion, so noise behind vertical voided the frame."""

    angles = _angles(trunk_flexion=-0.3)
    calculator = RULACalculator()

    untolerated = calculator.calculate(_rula(angles, tolerance=0.0))
    assert "trunk_flexion_deg" in untolerated.invalid_inputs

    tolerated = calculator.calculate(_rula(angles))
    assert "trunk_flexion_deg" not in tolerated.invalid_inputs


def test_applied_tolerance_is_recorded_for_audit():
    context = _rula(_angles(trunk_flexion=0.5)).context
    assert context.pose_quality["neutral_tolerance_deg"] == DEFAULT_NEUTRAL_TOLERANCE_DEG
    assert "trunk_flexion" in context.pose_quality["neutral_snapped_measurements"]
