"""Visual evidence must identify the same person, camera and recording as the review."""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from assistant import backend, scene, tools, visuals
from assistant.config import AssistantConfig


def test_visual_and_numeric_tools_use_the_selected_worker_and_method(monkeypatch):
    def detail(session, frame, method, person):
        assert (session, frame, method, person) == ("test", 48, "rula", "worker-2")
        return {
            "frame": frame,
            "clock": "0:02",
            "timestampSec": 2,
            "person": person,
            "method": "RULA",
            "score": 6,
            "angles": {},
            "band": None,
            "selectedSide": "left",
            "components": {},
            "quality": {},
            "drivers": [],
            "offset": 0,
            "sides": {"left": {"score": 6}, "right": {"score": 4}},
        }

    monkeypatch.setattr(tools.analysis_api, "frame_detail", detail)

    def render(session, detail, method, camera, region, overlay, width, budget):
        assert (camera, region, overlay, budget) == (1, "right_hand", False, 1)
        return [("raw", b"image")], [
            {
                "frame": 48,
                "sec": 2,
                "method": method,
                "camera": 1,
                "person": detail["person"],
                "view": "raw",
            }
        ]

    monkeypatch.setattr(visuals, "render", render)
    selection = {
        "method": "rula",
        "frame": 48,
        "person": "worker-2",
        "camera": 1,
        "region": "right_hand",
    }
    numeric = tools.run_tool("frame", "test", {}, selection=selection)
    assert numeric.content["sides"]["right"]["score"] == 4
    assert not numeric.images and not numeric.image_refs
    visual = tools.run_tool("look_at", "test", {}, remaining_images=1, selection=selection)
    assert visual.frames == [48] and visual.image_refs[0]["camera"] == 1
    assert "score" not in visual.content and "score" not in visual.image_refs[0]


def test_raw_crop_uses_specific_camera_and_refuses_invisible_joint(monkeypatch):
    from api import main

    left = {"16": [0.15, 0.2, 0, 0.99], "18": [0.2, 0.22, 0, 0.99]}
    hidden = {key: [*value[:3], 0.2] for key, value in left.items()}
    record = {"pose_landmarks_normalized_by_camera": {"0": hidden, "1": left}}
    monkeypatch.setattr(visuals, "cameras", lambda s: [{"camera_index": 0}, {"camera_index": 1}])
    monkeypatch.setattr(
        visuals.analysis_api,
        "_store",
        lambda s: SimpleNamespace(load_record_at_offset=lambda o: record),
    )
    monkeypatch.setattr(visuals.analysis_api, "people", lambda s: ["worker"])
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    image[65:160, 72:210, 2] = 255
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    monkeypatch.setattr(main, "_frame_jpeg", lambda *args: jpeg.tobytes())
    data = {
        "frame": 10,
        "timestampSec": 1.23,
        "person": "worker",
        "offset": 123,
        "cameras": [{"cameraIndex": 1, "sourceFrame": 11}],
    }
    views, refs = visuals.render("s", data, "rula", None, "right_hand", False, 640, 2)
    assert [r["view"] for r in refs] == ["raw", "crop"]
    assert all(r["camera"] == 1 and r["sourceFrame"] == 11 for r in refs)
    crop = cv2.imdecode(np.frombuffer(views[1][1], np.uint8), cv2.IMREAD_COLOR)
    assert crop.shape[0] < 250 and crop.shape[1] < 300
    assert crop[:, :, 2].mean() > 100
    _, refs = visuals.render("s", data, "rula", 0, "right_hand", False, 640, 2)
    assert len(refs) == 1 and refs[0]["view"] == "raw"
    with pytest.raises(ValueError, match="unavailable"):
        visuals.render("s", data, "rula", 2, "", False, 640, 1)


def test_motion_sampling_includes_ends_even_when_scores_were_withheld(monkeypatch):
    entries = [
        SimpleNamespace(frame_number=i, timestamp_sec=i / 10, person_identifier="p")
        for i in range(11)
    ]
    monkeypatch.setattr(tools.analysis_api, "_store", lambda _: SimpleNamespace(entries=entries))
    monkeypatch.setattr(tools, "_exact_frame", lambda s, f, m, p: {"frame": f, "person": p})

    def render(s, d, m, c, *a):
        return [("caption", b"jpeg")], [
            {"frame": d["frame"], "sec": d["frame"] / 10, "method": m, "camera": 1}
        ]

    monkeypatch.setattr(visuals, "render", render)
    result = tools.run_tool(
        "watch",
        "s",
        {"max_frames": 3},
        selection={"method": "rula", "person": "p", "fromSec": 0, "toSec": 1},
    )
    assert result.frames == [0, 5, 10]
    assert all(r["method"] == "rula" for r in result.image_refs)


def test_scene_grounding_refuses_unknown_ids_and_occluded_proposals():
    images = [{"frame": 35, "sec": 1.2, "camera": 0}, {"frame": 35, "sec": 1.2, "camera": 2}]
    record = {
        "task_inference": {"activity_class": "manual_lifting", "evidence_images": [1]},
        "workstation": {
            "objects": [{"fact": "box", "visibility": "clear", "evidence_images": [2]}],
            "tools": [],
            "surfaces": [],
        },
        "observations": [
            {
                "kind": "phase",
                "fact": "transfers between two carts",
                "visibility": "clear",
                "evidence_images": [1, 8],
            }
        ],
        "condition_proposals": [
            {
                "condition": "coupling",
                "value": "good",
                "confidence": "high",
                "rationale": "Hidden fingers",
                "visibility": "occluded",
                "evidence_images": [2],
            }
        ],
    }
    parsed = scene._ground_reading(json.dumps(record), images)
    assert [f["fact"] for f in parsed["findings"]] == ["box"]
    assert parsed["findings"][0]["evidence"] == [images[1]]
    assert parsed["condition_proposals"] == []
    record["workstation"]["objects"][0]["visibility"] = "certain"
    with pytest.raises(backend.BackendError, match="schema"):
        scene._ground_reading(json.dumps(record), images)


def test_scene_cache_expires_when_recording_changes_or_model_changes(tmp_path, monkeypatch):
    config = AssistantConfig(scene_model="one")
    monkeypatch.setattr(scene, "effective_config", lambda _: config)
    monkeypatch.setattr(scene, "resolve_result_dir", lambda _: str(tmp_path))
    source = tmp_path / "recording.mp4"
    monkeypatch.setattr(scene, "cameras", lambda _: [{"camera_index": 0, "path": str(source)}])
    for name in ("frame_results.jsonl", "run_config.json", "video_summary.json"):
        (tmp_path / name).write_text("data")
    source.write_bytes(b"initial")
    fp = scene._fingerprint("s", config)
    scene.scene_path("s").write_text(json.dumps({"schema": scene.SCENE_SCHEMA, "fingerprint": fp}))
    assert scene.load_scene("s")
    monkeypatch.setattr(scene, "effective_config", lambda _: AssistantConfig(scene_model="two"))
    assert scene.load_scene("s") is None
    monkeypatch.setattr(scene, "effective_config", lambda _: config)
    source.write_bytes(b"changed bytes")
    assert scene.load_scene("s") is None


def test_decisions_apply_only_to_the_reviewed_value_and_evidence(tmp_path, monkeypatch):
    from assistant import proposals

    suggestion = {
        "condition": "coupling",
        "value": "good",
        "evidence_frames": [10],
        "evidence": [{"frame": 10, "camera": 0}],
    }
    card = {"fingerprint": "session-A", "condition_proposals": [suggestion]}
    monkeypatch.setattr(proposals, "load_scene", lambda _: card)
    monkeypatch.setattr(proposals, "proposals_path", lambda _: tmp_path / "decisions.json")
    proposal_id = proposals.load_proposals("s")["proposals"][0]["proposalId"]
    proposals.accept_proposal("s", "coupling", value="fair", proposal_id=proposal_id)
    assert proposals.load_proposals("s")["proposals"][0]["status"] == "accepted"
    assert proposals.entered_conditions("s") == {"coupling": "fair"}
    suggestion["value"] = "poor"
    assert proposals.load_proposals("s")["proposals"][0]["status"] == "pending"
    with pytest.raises(ValueError, match="proposal changed"):
        proposals.accept_proposal("s", "coupling", proposal_id=proposal_id)
    suggestion["value"] = "good"
    suggestion["evidence"][0]["camera"] = 1
    new_id = proposals.load_proposals("s")["proposals"][0]["proposalId"]
    assert new_id != proposal_id
    for _ in range(2):
        proposals.reject_proposal("s", "coupling", proposal_id=new_id)
    assert proposals.entered_conditions("s") == {"coupling": "fair"}


def test_scene_falls_back_to_real_frames_without_scores(monkeypatch):
    from assistant import sampling

    monkeypatch.setattr(
        sampling.analysis_api, "timeline", lambda *args: {"series": {"points": []}, "person": "p"}
    )
    monkeypatch.setattr(sampling, "posture_cloud", lambda s: ([], [], np.empty((0, 0))))
    monkeypatch.setattr(
        sampling.analysis_api,
        "_records",
        lambda s, p: [{"frame_number": f, "timestamp_sec": f / 10} for f in [0, 5, 8, 13, 15, 16]],
    )
    samples = sampling.representative_frames("s", 2)
    assert [s.frame for s in samples] == [0, 16]
    assert all(s.score is None for s in samples)
