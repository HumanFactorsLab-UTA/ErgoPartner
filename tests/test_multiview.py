"""Multi-view selection and confidence provenance regressions."""

from core.angle_calculator import BodyAngles
from core.multi_view_fusion import MultiViewFusion


def _angles(trunk_flexion: float, visibility: dict[int, float]) -> BodyAngles:
    return BodyAngles(trunk_flexion=trunk_flexion, landmark_visibility=visibility)


def test_occluded_required_landmark_cannot_hide_behind_high_average_visibility():
    required = (11, 12, 23, 24)
    front = _angles(10.0, {index: 0.80 for index in required})
    side = _angles(40.0, {11: 0.10, 12: 0.99, 23: 0.99, 24: 0.99})
    fusion = MultiViewFusion(
        camera_roles={"front": 0, "side": 1},
        visibility_threshold=0.5,
    )
    result, selected = fusion.fuse_angles([front, side], [0, 1])
    assert result.trunk_flexion == 10.0
    assert selected["trunk_flexion"] == "front"
    assert result.measurement_visibility["trunk_flexion_deg"] == 0.80


def test_preferred_view_wins_only_when_every_required_landmark_is_usable():
    required = (11, 12, 23, 24)
    front = _angles(10.0, {index: 0.99 for index in required})
    side = _angles(40.0, {index: 0.60 for index in required})
    fusion = MultiViewFusion(
        camera_roles={"front": 0, "side": 1},
        visibility_threshold=0.5,
    )
    result, selected = fusion.fuse_angles([front, side], [0, 1])
    assert result.trunk_flexion == 40.0
    assert selected["trunk_flexion"] == "side"
    assert result.measurement_visibility["trunk_flexion_deg"] == 0.60


def test_unavailable_elbow_and_midline_never_turn_into_neutral_measurements():
    cameras = [
        BodyAngles(
            landmark_visibility=dict.fromkeys(range(33), 1.0),
            unavailable_measurements=["right_lower_arm_flexion_deg", "right_forearm_midline_cross"],
        )
        for _ in range(2)
    ]
    result, selected = MultiViewFusion().fuse_angles(cameras, [0, 1])
    assert selected["right_forearm_flexion"] == "unavailable"
    assert selected["right_forearm_midline_cross"] == "unavailable"
    assert result.to_dict()["right_forearm_flexion"] is None
    assert result.to_dict()["right_forearm_midline_cross"] is None


def test_consensus_wrist_uses_real_mesh_evidence_even_without_pose_knuckles():
    cameras = [
        BodyAngles(
            right_wrist_flexion=value,
            right_wrist_data_source="hand_mesh",
            landmark_visibility={14: 0.9, 16: 0.8},
        )
        for value in (1, 80, 5)
    ]
    fusion = MultiViewFusion({"front": 0, "side": 1, "iso": 2})
    result, selected = fusion.fuse_angles(cameras, [0, 1, 2])
    assert result.right_wrist_flexion == 5
    assert selected["right_wrist_flexion"] == "iso+consensus"
    assert result.right_wrist_data_source == "hand_mesh"
    assert result.measurement_visibility["right_wrist_flexion_deg"] == 0.8


def test_usable_hand_mesh_takes_priority_over_sparse_pose_fingers():
    pose = BodyAngles(right_wrist_flexion=60, landmark_visibility=dict.fromkeys(range(33), 1.0))
    mesh = BodyAngles(
        right_wrist_flexion=9,
        right_wrist_data_source="hand_mesh",
        landmark_visibility={14: 0.95, 16: 0.95},
    )
    views = {"side": 0, "front": 1}
    normal, sources = MultiViewFusion(views).fuse_angles([pose, mesh], [0, 1])
    assert normal.right_wrist_flexion == 60
    assert sources["right_wrist_flexion"] == "side"
    result, sources = MultiViewFusion(views, prefer_hand_mesh=True).fuse_angles(
        [pose, mesh], [0, 1]
    )
    assert result.right_wrist_flexion == 9
    assert sources["right_wrist_flexion"] == "front"
    assert result.right_wrist_data_source == "hand_mesh"
