import json
import time
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from dashboard.camera_placement import CameraPlacement, PlacementError, Sample, measure, summarize


def samples(metrics, count=5, now=100, start=0, pixel=120):
    return [
        Sample(
            start + i,
            now - (count - i - 1) * 0.9,
            metrics,
            b"image",
            np.full((96, 3), pixel, dtype=np.int16),
        )
        for i in range(count)
    ]


def clear_pose():
    return {i: np.array([0.5, 0.1 + 0.8 * i / 33, 0, 0.9]) for i in range(33)}


def test_coverage_is_camera_specific_and_distinguishes_partial_edge_and_ambiguous_workers():
    image = np.full((48, 48, 3), 120, dtype=np.uint8)
    points = clear_pose()
    detector = SimpleNamespace(process_frame_all=lambda _: [SimpleNamespace(normalized=points)])
    healthy = measure(image, detector)
    assert summarize(samples(healthy), 100)["state"] == "clear"
    points[15][0] = points[19][0] = 1.2
    points[13][3] = 0.05
    cut = summarize(samples(measure(image, detector)), 100)
    assert dict((r["name"], r["status"]) for r in cut["regions"])["Hands"] == "near edge"
    assert any("arms" in issue for issue in cut["issues"])
    assert summarize(samples(healthy, 2), 100)["state"] == "warming"
    assert summarize(samples(healthy), 110)["state"] == "unavailable"
    assert summarize(samples({"people": 2, "lighting": None}), 100)["issues"][0].startswith(
        "Multiple"
    )
    offline = summarize(samples(measure(image, None)), 100)
    assert offline["state"] == "unavailable"
    assert offline["regions"] == []


def test_decision_and_visual_images_are_scoped_to_an_active_camera_and_preview():
    started, finish = Event(), Event()

    def visual_reader(device, shown):
        assert device == 9 and [i for i, _ in shown] == [0, 2, 4]
        started.set()
        finish.wait(2)
        return {"findings": []}

    monitor = CameraPlacement([4, 9], visual_reader=visual_reader)
    now = time.perf_counter()
    metrics = {"people": 1, "lighting": None, "regions": {}}
    monitor._samples[9].extend(samples(metrics, now=now))
    review = monitor.status()["cameras"][9]
    assert monitor.status()["cameras"][4]["samples"] == 0
    monitor.inspect(9, review["id"])
    try:
        assert started.wait(2)
        state = monitor.status()["cameras"][9]
        with pytest.raises(PlacementError):
            monitor.decide(9, review["id"], "ready")
        assert monitor.evidence(9, state["visual"]["checkId"], 2) == b"image"
        with pytest.raises(PlacementError):
            monitor.evidence(4, state["visual"]["checkId"], 2)
        finish.set()
        for _ in range(100):
            if monitor.status()["cameras"][9]["visual"]["state"] == "ready":
                break
            time.sleep(0.01)
        state = monitor.status()["cameras"][9]
        assert state["visual"]["state"] == "ready"
        monitor.decide(9, state["id"], "adjust")
        assert monitor.status()["cameras"][9]["decision"]["value"] == "adjust"
        monitor._samples[9].extend(samples(metrics, count=2, start=10, now=now + 1, pixel=0))
        new_state = monitor.status()["cameras"][9]
        assert new_state["visual"]["state"] == "expired" and new_state["decision"] is None
        with pytest.raises(PlacementError):
            monitor.decide(9, state["id"], "ready")
    finally:
        finish.set()
        monitor.close()
    with pytest.raises(PlacementError):
        monitor.inspect(9, review["id"])


def test_vlm_must_use_the_actual_preview_samples(monkeypatch):
    from assistant import placement
    from assistant.config import AssistantConfig

    monkeypatch.setattr(placement, "effective_config", lambda **kwargs: AssistantConfig())

    finding = {"visibility": "unclear", "cue": "The arm hides the contact area.", "frames": [12]}
    reading = dict.fromkeys(("framing", "grip", "support"), finding)
    monkeypatch.setattr(placement, "complete", lambda *args, **kwargs: json.dumps(reading))
    assert (
        placement.inspect_placement(4, [(12, b"image")])["findings"][1]["visibility"] == "unclear"
    )
    finding["frames"] = [20]
    with pytest.raises(placement.BackendError, match="outside this preview"):
        placement.inspect_placement(4, [(12, b"image")])


def test_api_rejects_another_assessment_and_reports_coverage_across_views(monkeypatch):
    from fastapi.testclient import TestClient

    from api import assessment_api, main

    monitor = CameraPlacement([3, 8])
    now = time.perf_counter()
    for device, regions in [(3, {"Head": "visible", "Trunk": "visible"}), (8, {"Legs": "visible"})]:
        monitor._samples[device].extend(
            samples({"people": 1, "lighting": None, "regions": regions}, now=now)
        )
    monkeypatch.setattr(assessment_api, "CAPTURE_OWNER", "current")
    monkeypatch.setattr(
        assessment_api, "get_recording_service", lambda: SimpleNamespace(placement=lambda: monitor)
    )
    with TestClient(
        main.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50403)
    ) as client:
        url = "/api/assessments/current/capture/placement"
        assert client.get(url.replace("current", "other")).status_code == 409
        result = client.get(url).json()
        assert result["noClearView"] == ["Arms", "Hands"]
        body = {"reviewId": result["cameras"]["3"]["id"], "choice": "adjust"}
        assert client.post(url + "/8/decision", json=body).status_code == 409
        assert (
            client.post(url + "/3/decision", json=body).json()["cameras"]["3"]["decision"]["value"]
            == "adjust"
        )
    monitor.close()
