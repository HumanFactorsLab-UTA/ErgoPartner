import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api import focused_review_api, main
from assistant import focused_benchmark as bench
from assistant import focused_review as review
from assistant import visuals
from assistant.backend import BackendError
from assistant.config import AssistantConfig


def _case(session="s", frame=12, camera=0, mode="single"):
    evidence = tuple(
        {
            "imageIndex": i + 1,
            "frame": f,
            "sourceFrame": f,
            "sec": f / 10,
            "view": v,
            "camera": camera,
            "person": "worker",
        }
        for i, (f, v) in enumerate(((frame - 1, "raw"), (frame, "raw"), (frame, "crop")))
    )
    return review.Case(
        review.digest([session, frame, camera]),
        "run",
        session,
        frame,
        "worker",
        camera,
        mode,
        "rula",
        evidence,
        (b"first", b"target", b"crop"),
    )


def _response(grip="object_grasp", support="clear_gap", visibility="partial", ids=(2,)):
    return {
        k: {"value": v, "reason": "Visible cue in this fixture.", "images": list(ids)}
        for k, v in (("grip", grip), ("armSupport", support), ("visibility", visibility))
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    config = AssistantConfig()
    monkeypatch.setattr(bench, "effective_config", lambda *args: config)
    monkeypatch.setattr(focused_review_api, "effective_config", lambda _: config)
    monkeypatch.setattr(bench, "run_key", lambda _: "run")
    monkeypatch.setattr(bench, "data_path", lambda session: tmp_path / (session + ".db"))
    monkeypatch.setattr(review, "complete", lambda *a, **kw: json.dumps(_response()))
    monkeypatch.setattr(focused_review_api, "get_case", lambda s, f, p, m, c: _case(s, f, c))
    return config


def test_exact_images_and_decision_cycle_through_the_api(setup):
    url = "/api/sessions/s/focused-review"
    ctx = {"frame": 12, "person": "worker", "method": "rula", "camera": 0}
    reference = {"grip": "unclear", "armSupport": "clear_gap", "visibility": "partial"}
    with TestClient(main.app) as api:
        case = api.get(url, params=ctx).json()
        payload = {**ctx, "caseId": case["caseId"]}
        assert api.get(url + "/images/2", params=payload).content == b"target"
        assert api.get(url + "/images/0", params=payload).status_code == 404
        assert api.get(url + "/images/2", params={**payload, "caseId": "a" * 64}).status_code == 409
        assert case["blindEligible"] and case["prediction"] is None
        assert api.post(url + "/reference", json={**payload, "answers": reference}).is_success
        assert api.post(url + "/reference", json={**payload, "answers": reference}).is_success
        pred = api.post(url + "/predict", json=payload).json()["prediction"]
        assert pred["answers"]["grip"]["evidence"][0]["imageIndex"] == 2
        other = {**reference, "grip": "no_grasp"}
        assert api.post(url + "/reference", json={**payload, "answers": other}).status_code == 409
        assert (
            api.post(
                url + "/decision", json={**payload, "answers": other, "predictionId": "a" * 64}
            ).status_code
            == 409
        )
        assert api.post(
            url + "/decision", json={**payload, "answers": other, "predictionId": pred["id"]}
        ).is_success
        assert api.get(url, params=ctx).json()["reference"]["answers"] == reference
        report = api.get(url + "/evaluation").json()
        assert report["pairedCases"] == 1 and report["metrics"]["grip"]["falseDefinites"] == 1
        assert report["metrics"]["armSupport"]["accuracy"] == 1
        assert report["byCameraMode"]["multi"]["metrics"]["grip"]["accuracy"] is None


def test_revealed_overlapping_camera_view_cannot_become_independent_reference(setup):
    first = _case(mode="multi")
    bench.predict(first)
    other = _case(camera=1, mode="multi")
    assert not bench.describe(other)["blindEligible"]
    answers = review.Answers(grip="unclear", armSupport="unclear", visibility="unclear")
    with pytest.raises(review.StaleVisualReview):
        bench.save_reference(other, answers)
    assert bench.describe(_case(frame=50))["blindEligible"]
    assert bench.evaluation(bench.records("s"))["pairedCases"] == 0


def test_frozen_readings_and_locked_references_across_versions_and_races(setup, monkeypatch):
    case = _case()
    answers = review.Answers(grip="unclear", armSupport="unclear", visibility="unclear")
    bench.save_reference(case, answers)
    with ThreadPoolExecutor(max_workers=3) as pool:
        ids = list(pool.map(lambda _: bench.predict(case)["prediction"]["id"], range(3)))
    assert len(set(ids)) == 1
    monkeypatch.setattr(bench, "effective_config", lambda *args: replace(setup, scene_model="other-model"))
    assert bench.describe(case)["prediction"] is None
    assert bench.describe(case)["reference"]["answers"] == answers.model_dump()
    assert bench.evaluation(bench.records("s"))["pairedCases"] == 0
    monkeypatch.setattr(bench, "run_key", lambda _: "changed")
    with pytest.raises(review.StaleVisualReview):
        bench.predict(case)
    assert bench.records("s") == []


@pytest.mark.parametrize("ids", [[2, 3], [1], [1, 999]])
def test_invalid_or_single_time_support_evidence_requires_another_reading(monkeypatch, ids):
    replies = iter(
        [
            json.dumps(_response(support="contact", ids=ids)),
            json.dumps(_response(support="unclear")),
        ]
    )
    monkeypatch.setattr(review, "complete", lambda *a, **kw: next(replies))
    assert (
        review.read_images(_case(), AssistantConfig())["answers"]["armSupport"]["value"]
        == "unclear"
    )


def test_repeated_camera_source_frame_does_not_establish_resting_contact(monkeypatch):
    case = _case()
    case = replace(case, evidence=tuple({**r, "sourceFrame": 99} for r in case.evidence))
    monkeypatch.setattr(
        review, "complete", lambda *a, **kw: json.dumps(_response(support="contact", ids=[1, 2]))
    )
    with pytest.raises(BackendError):
        review.read_images(case, AssistantConfig())


def test_multiworker_images_identify_selected_person_or_withhold(monkeypatch):
    record = {
        "pose_landmarks_normalized_by_camera": {
            "1": {str(i): [0.5, 0.5, 0.0, 0.9] for i in range(33)}
        }
    }
    monkeypatch.setattr(visuals, "cameras", lambda _: [{"camera_index": 1}])
    monkeypatch.setattr(visuals.analysis_api, "people", lambda _: ["worker", "other"])
    monkeypatch.setattr(
        visuals.analysis_api,
        "_store",
        lambda _: SimpleNamespace(load_record_at_offset=lambda _: record),
    )
    calls = []

    def jpeg(session, frame, camera, region, overlay, width, person):
        calls.append((camera, overlay, person))
        return b"image"

    monkeypatch.setattr(main, "_frame_jpeg", jpeg)
    monkeypatch.setattr(visuals, "_crop", lambda *_: b"crop")
    detail = {"frame": 12, "person": "worker", "timestampSec": 1.2, "offset": 5}
    _, refs = visuals.render("s", detail, "rula", 1, "hand_and_forearm", False, 960, 2)
    assert [r["view"] for r in refs] == ["overlay", "crop"]
    assert calls == [(1, True, "worker"), (1, False, "worker")]
    record.clear()
    with pytest.raises(visuals.SessionError, match="could not be located"):
        visuals.render("s", detail, "rula", 1, "hand_and_forearm", False, 960, 2)


def test_library_counts_only_current_published_reviews_in_active_studies(tmp_path, monkeypatch):
    from api import sessions
    from dashboard import data_loader

    def path(session):
        study, participant, trial, rate = session.split("__")
        return tmp_path / (rate + "fps") / study / participant / trial / "focused_review.sqlite3"

    monkeypatch.setattr(data_loader, "get_results_root", lambda: str(tmp_path))
    monkeypatch.setattr(bench, "effective_config", lambda *args: AssistantConfig())
    monkeypatch.setattr(bench, "data_path", path)
    monkeypatch.setattr(bench, "run_key", lambda _: "run")
    monkeypatch.setattr(
        sessions,
        "catalog",
        lambda _: [
            {
                "study": {"folder_name": "selected"},
                "participant_id": "P",
                "tasks": [{"folder_name": "current"}, {"folder_name": "stale"}],
            }
        ],
    )
    for session in (
        "selected__P__current__10",
        "selected__P__stale__10",
        "unopened__P__current__10",
    ):
        path(session).parent.mkdir(parents=True)
        bench.save_reference(
            _case(session),
            review.Answers(grip="unclear", armSupport="unclear", visibility="unclear"),
        )
    monkeypatch.setattr(bench, "run_key", lambda s: "changed" if "stale" in s else "run")
    result = bench.library_evaluation("root")
    assert result["sessionCount"] == 1 and result["labelledCases"] == 1
    assert result["skippedSessions"] == 1 and result["pairedSessions"] == 0
