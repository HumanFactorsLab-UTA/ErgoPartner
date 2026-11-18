"""Choosing which camera supplies whole landmark positions for the sequence analyses."""

from __future__ import annotations

import numpy as np

from core.multi_view_fusion import SEQUENCE_LANDMARKS, select_landmark_source


def _camera(visibility: float, *, missing: tuple[int, ...] = ()) -> dict[int, np.ndarray]:
    return {
        index: np.array([0.1, 0.2, 0.3, visibility])
        for index in SEQUENCE_LANDMARKS
        if index not in missing
    }


def test_the_camera_that_sees_more_joints_is_chosen():
    occluded = _camera(0.9, missing=(25, 26, 27, 28))
    complete = _camera(0.8)

    index, landmarks = select_landmark_source([occluded, complete])

    assert index == 1
    assert 27 in landmarks


def test_visibility_breaks_a_tie_on_joint_count():
    dim = _camera(0.55)
    clear = _camera(0.95)

    assert select_landmark_source([dim, clear])[0] == 1


def test_camera_zero_is_not_privileged():
    """The defect this fixes: camera zero was used unconditionally."""

    blind = _camera(0.1)
    seeing = _camera(0.9)

    assert select_landmark_source([blind, seeing])[0] == 1


def test_the_first_camera_wins_when_views_are_equal():
    same = _camera(0.9)

    assert select_landmark_source([same, dict(same), dict(same)])[0] == 0


def test_low_visibility_landmarks_do_not_count_as_seen():
    below = _camera(0.4)
    above = _camera(0.6)

    assert select_landmark_source([below, above], visibility_threshold=0.5)[0] == 1


def test_non_finite_landmarks_are_ignored():
    broken = {index: np.array([np.nan, 0.0, 0.0, 0.99]) for index in SEQUENCE_LANDMARKS}
    usable = _camera(0.6)

    assert select_landmark_source([broken, usable])[0] == 1


def test_landmarks_are_never_merged_across_cameras():
    """Each camera's world landmarks sit in its own frame; mixing them is unsound."""

    left_only = _camera(0.9, missing=(27, 28))
    right_only = {27: np.array([9.0, 9.0, 9.0, 0.99]), 28: np.array([9.0, 9.0, 9.0, 0.99])}

    _, landmarks = select_landmark_source([left_only, right_only])

    # The chosen camera's landmarks are returned whole; the other camera's
    # values for the joints it was missing are not spliced in.
    assert 27 not in landmarks


def test_returned_landmarks_are_a_copy():
    original = _camera(0.9)
    _, landmarks = select_landmark_source([original])
    landmarks[999] = np.zeros(4)

    assert 999 not in original


def test_no_cameras_yields_camera_zero_and_nothing():
    index, landmarks = select_landmark_source([])

    assert index == 0
    assert landmarks == {}


def test_empty_camera_still_returns_a_defined_source():
    index, landmarks = select_landmark_source([{}])

    assert index == 0
    assert landmarks == {}


def test_landmarks_without_a_visibility_component_are_treated_as_visible():
    three_component = {index: np.array([0.1, 0.2, 0.3]) for index in SEQUENCE_LANDMARKS}
    dim = _camera(0.2)

    assert select_landmark_source([dim, three_component])[0] == 1
