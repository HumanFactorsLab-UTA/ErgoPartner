"""Skeleton topology: a drawn neck and spine, synthesised from real landmarks."""

from __future__ import annotations

import pytest

from core.skeleton import (
    C7,
    HEAD_CENTRE,
    LEFT_EAR,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    PELVIS,
    POSE_CONNECTIONS,
    RIGHT_EAR,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    VIRTUAL_JOINTS,
    augment_with_virtual_joints,
)


def _upright() -> dict[int, list[float]]:
    return {
        NOSE: [0.0, 1.70, 0.10],
        LEFT_EYE: [-0.03, 1.72, 0.08],
        RIGHT_EYE: [0.03, 1.72, 0.08],
        LEFT_EAR: [-0.07, 1.71, 0.0],
        RIGHT_EAR: [0.07, 1.71, 0.0],
        LEFT_SHOULDER: [-0.20, 1.45, 0.0],
        RIGHT_SHOULDER: [0.20, 1.45, 0.0],
        LEFT_HIP: [-0.12, 0.95, 0.0],
        RIGHT_HIP: [0.12, 0.95, 0.0],
    }


def test_c7_sits_at_the_shoulder_midpoint():
    joints = augment_with_virtual_joints(_upright())

    assert joints[C7] == pytest.approx([0.0, 1.45, 0.0])


def test_pelvis_sits_at_the_hip_midpoint():
    joints = augment_with_virtual_joints(_upright())

    assert joints[PELVIS] == pytest.approx([0.0, 0.95, 0.0])


def test_head_centre_is_the_ear_and_eye_centroid_not_the_nose():
    """The nose swings with head rotation; a symmetric centroid does not."""

    joints = augment_with_virtual_joints(_upright())

    assert joints[HEAD_CENTRE][0] == pytest.approx(0.0)
    # The nose sits forward of the centroid, which is why it is not used.
    assert joints[HEAD_CENTRE][2] < _upright()[NOSE][2]


def test_head_centre_does_not_move_when_the_head_turns():
    facing = _upright()
    turned = dict(facing)
    # Yaw the head: the nose swings sideways, the ears swap depth.
    turned[NOSE] = [0.09, 1.70, 0.0]
    turned[LEFT_EAR] = [-0.02, 1.71, -0.06]
    turned[RIGHT_EAR] = [0.02, 1.71, 0.06]

    before = augment_with_virtual_joints(facing)[HEAD_CENTRE]
    after = augment_with_virtual_joints(turned)[HEAD_CENTRE]

    assert after[0] == pytest.approx(before[0], abs=0.02)


def test_the_neck_and_spine_are_drawn():
    assert (C7, HEAD_CENTRE) in POSE_CONNECTIONS
    assert (C7, PELVIS) in POSE_CONNECTIONS


def test_the_ear_to_shoulder_diagonals_are_gone():
    """Those two lines were the old stand-in for a neck; they are not anatomy."""

    assert (LEFT_EAR, LEFT_SHOULDER) not in POSE_CONNECTIONS
    assert (RIGHT_EAR, RIGHT_SHOULDER) not in POSE_CONNECTIONS
    assert (NOSE, LEFT_SHOULDER) not in POSE_CONNECTIONS
    assert (NOSE, RIGHT_SHOULDER) not in POSE_CONNECTIONS


def test_no_edge_references_a_landmark_that_cannot_exist():
    for start, end in POSE_CONNECTIONS:
        for index in (start, end):
            assert 0 <= index < 33 or index in VIRTUAL_JOINTS


def test_visibility_of_a_synthesised_joint_is_the_mean_of_its_sources():
    landmarks = {
        LEFT_SHOULDER: [-0.20, 1.45, 0.0, 0.9],
        RIGHT_SHOULDER: [0.20, 1.45, 0.0, 0.5],
    }
    joints = augment_with_virtual_joints(landmarks)

    assert joints[C7][3] == pytest.approx(0.7)


def test_a_joint_is_omitted_when_its_sources_are_missing():
    """A partially visible pose loses a segment; it never gains an invented one."""

    joints = augment_with_virtual_joints({LEFT_SHOULDER: [-0.20, 1.45, 0.0]})

    assert C7 not in joints
    assert PELVIS not in joints
    assert HEAD_CENTRE not in joints


def test_head_centre_falls_back_through_available_pairs():
    eyes_only = {LEFT_EYE: [-0.03, 1.72, 0.08], RIGHT_EYE: [0.03, 1.72, 0.08]}
    assert augment_with_virtual_joints(eyes_only)[HEAD_CENTRE] == pytest.approx([0.0, 1.72, 0.08])

    nose_only = {NOSE: [0.0, 1.70, 0.10]}
    assert augment_with_virtual_joints(nose_only)[HEAD_CENTRE] == pytest.approx([0.0, 1.70, 0.10])


def test_real_landmarks_are_preserved_untouched():
    original = _upright()
    joints = augment_with_virtual_joints(original)

    for index, value in original.items():
        assert joints[index] == value


def test_augmentation_does_not_mutate_the_input():
    original = _upright()
    augment_with_virtual_joints(original)

    assert set(original) == set(_upright())


def test_an_existing_joint_is_not_overwritten():
    landmarks = _upright()
    landmarks[C7] = [9.0, 9.0, 9.0]

    assert augment_with_virtual_joints(landmarks)[C7] == [9.0, 9.0, 9.0]
