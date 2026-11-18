"""User workflows preserve source recordings and bind human review to one result."""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.assessment_api as routes
import api.assessments as store
import api.jobs as jobs
import api.main as main
from api import sessions
from dashboard.cache import clear_all_caches
from dashboard.data_loader import participant_catalog


@pytest.fixture
def workflow_env(tmp_path, monkeypatch):
    root = tmp_path / "studies"
    root.mkdir()
    monkeypatch.setattr(store, "WORKFLOW_ROOT", tmp_path / "workflow")
    monkeypatch.setattr(main, "STUDIES_ROOT", str(root))
    monkeypatch.setattr(routes, "CAPTURE_OWNER", None)
    monkeypatch.setattr(routes, "CAPTURE_TAKE", None)
    registry = jobs.AnalysisJobRegistry(studies_root=root)
    monkeypatch.setattr(jobs, "analysis_jobs", registry)
    monkeypatch.setattr(main, "analysis_jobs", registry)
    source = tmp_path / "example.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    writer.release()
    clear_all_caches()
    with TestClient(
        main.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 51001)
    ) as client:
        yield client, root, source, registry
    store.UPLOADING.clear()
    clear_all_caches()


def make_draft(client, source="video"):
    response = client.post("/api/assessments", json={"source": source})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_atomic_json_retries_a_temporary_windows_sharing_error(tmp_path, monkeypatch):
    path = tmp_path / "draft.json"
    path.write_text('{"step": 1}')
    replace = Path.replace
    attempts = []

    def temporarily_blocked(staged, target):
        attempts.append(staged)
        if len(attempts) < 3:
            assert json.loads(target.read_text()) == {"step": 1}
            error = PermissionError("test sharing violation")
            error.winerror = 5
            raise error
        return replace(staged, target)

    monkeypatch.setattr(Path, "replace", temporarily_blocked)
    store.atomic_json(path, {"step": 2})
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert json.loads(path.read_text()) == {"step": 2}
    assert list(tmp_path.iterdir()) == [path]


def test_named_tasks_share_a_participant_before_publication(workflow_env):
    client, root, source, _ = workflow_env

    def grouped(**body):
        response = client.post("/api/assessments", json={"source": "video", **body})
        assert response.status_code == 200, response.text
        return response.json()

    first = grouped(group={"studyName": "Assembly", "taskName": "Reach"})
    person = client.get("/api/participants").json()["participants"][0]
    assert first["grouped"] and person["participantId"] == "001" and person["trials"] == []
    second = grouped(participantKey=person["key"], group={"taskName": "Fasten"})
    following = grouped(group={"studyFolder": first["setup"]["studyFolder"], "taskName": "Lift"})
    assert following["setup"]["participantId"] == "002"
    for draft in (first, second):
        task = add_video(client, draft, source)
        setup = {
            **task["setup"],
            "participantId": "override",
            "conditionsReviewed": True,
            "footageReviewed": True,
        }
        saved = client.post(f"/api/assessments/{task['id']}", json={"setup": setup}).json()
        assert saved["setup"]["participantId"] == "001"
        store.publish(saved, store.validate_ready(saved))
    tasks = sessions.find_context(str(root), person["key"])["tasks"]
    assert {t["name"] for t in tasks} == {"Reach", "Fasten"}
    assert len({t["folder_name"] for t in tasks}) == 2


def test_trash_is_reversible_for_draft_task_and_participant(workflow_env, monkeypatch):
    client, root, source, registry = workflow_env
    blank = make_draft(client)

    def move(**selection):
        response = client.post("/api/assessment-trash", json=selection)
        assert response.status_code == 200, response.text
        return response.json()["id"]

    def restore(item):
        return client.post(f"/api/assessment-trash/{item}/restore")

    store.UPLOADING.add(blank["id"])
    selection = {"kind": "draft", "draftId": blank["id"]}
    assert client.post("/api/assessment-trash", json=selection).status_code == 409
    store.UPLOADING.clear()
    item = move(**selection)
    assert client.get(f"/api/assessments/{blank['id']}").status_code == 404
    assert client.get("/api/assessments").json()["assessments"] == []
    assert restore(item).status_code == 200
    assert client.get(f"/api/assessments/{blank['id']}").status_code == 200

    active = confirm(add_video(client, blank, source))
    published = store.publish(active, store.validate_ready(active))
    key, folder = published["participantKey"], published["taskFolder"]
    context = sessions.find_context(str(root), key)
    video = Path(context["tasks"][0]["path"]) / "raw" / "cam_0.avi"
    session = sessions.SessionRef(context["study"]["folder_name"], "P001", folder, 10).session_id
    sibling = client.post(
        "/api/assessments", json={"participantKey": key, "group": {"taskName": "Take two"}}
    ).json()
    with monkeypatch.context() as patch:
        patch.setattr(registry, "working_on", lambda k, f: k == key and f in (None, folder))
        for selection in (
            {"kind": "task", "participantKey": key, "taskFolder": folder},
            {"kind": "participant", "participantKey": key},
        ):
            assert client.post("/api/assessment-trash", json=selection).status_code == 409
    with monkeypatch.context() as patch:
        patch.setattr(routes, "CAPTURE_OWNER", sibling["id"])
        assert (
            client.post(
                "/api/assessment-trash", json={"kind": "participant", "participantKey": key}
            ).status_code
            == 409
        )
    child = move(kind="task", participantKey=key, taskFolder=folder)
    assert client.get(f"/api/sessions/{session}/review").status_code == 404
    assert client.get(f"/api/assessments/{active['id']}").status_code == 404
    assert (
        registry._contexts_for_scope("all", participant_key=None, study_folder=None)[0]["tasks"]
        == []
    )
    assert client.get(f"/api/assessments/{sibling['id']}").status_code == 200
    parent = move(kind="participant", participantKey=key)
    assert client.get("/api/participants").json()["count"] == 0
    assert client.get(f"/api/assessments/{sibling['id']}").status_code == 404
    with monkeypatch.context() as patch:
        patch.setattr(main, "STUDIES_ROOT", str(root.parent / "other_library"))
        assert client.get("/api/assessment-trash").json()["items"] == []
        assert restore(parent).status_code == 404
    assert restore(child).status_code == 409
    assert restore(parent).status_code == 200
    assert client.get(f"/api/assessments/{sibling['id']}").status_code == 200
    assert client.get(f"/api/sessions/{session}/review").status_code == 404
    assert restore(child).status_code == 200
    assert client.get(f"/api/sessions/{session}/review").status_code == 200
    assert client.get("/api/assessment-trash").json()["items"] == []
    assert video.read_bytes() == source.read_bytes()


def test_imported_task_setup_trash_preserves_completed_source_and_other_participants(
    workflow_env, monkeypatch
):
    from dashboard import data_loader

    client, root, source, _ = workflow_env
    results = root.parent / "results"
    monkeypatch.setattr(data_loader, "RESULTS_DIR_STR", str(results))
    folder = "stocking"
    people = [("floor_a", "001"), ("floor_a", "002"), ("floor_b", "001")]
    for study, person in people:
        person_path = root / study / f"participant_{person}"
        raw = person_path / folder / "raw"
        raw.mkdir(parents=True)
        shutil.copy2(source, raw / "front.avi")
        (person_path / "participant_info.json").write_text(
            json.dumps({"participant_id": person, "custom": "retain me"})
        )
        (person_path.parent / "study_config.json").write_text(
            json.dumps(
                {
                    "study_name": "Assembly",
                    "tasks": [{"folder_name": folder, "name": "Stock shelves"}],
                }
            )
        )
    result = results / "10fps" / "floor_a" / "001" / folder
    result.mkdir(parents=True)
    for name, content in {
        "run_config.json": "{}",
        "video_summary.json": "{}",
        "frame_results.csv": "frame\n",
    }.items():
        (result / name).write_text(content)
    partial = results / "10fps" / "floor_a" / "002" / folder
    partial.mkdir(parents=True)
    (partial / "run_config.json").write_text("{}")
    key = "floor_a::001"
    sid = sessions.SessionRef("floor_a", "001", folder, 10).session_id
    scoped = {"participantKey": key, "taskFolder": folder}

    def cards():
        return {p["key"]: p for p in client.get("/api/participants").json()["participants"]}

    assert cards()[key]["trials"][0]["sessionId"] == sid
    assert cards()["floor_a::002"]["trials"][0]["status"] == "incomplete"
    partial_item = client.post("/api/assessment-trash", json={
        "kind": "task", "participantKey": "floor_a::002", "taskFolder": folder,
    }).json()
    assert cards()["floor_a::002"]["trials"] == []
    assert client.post(f"/api/assessment-trash/{partial_item['id']}/restore").status_code == 200
    assert cards()["floor_a::002"]["trials"][0]["status"] == "incomplete"
    assert (partial / "run_config.json").read_text() == "{}"
    source_meta = root / "floor_a" / "participant_001" / "participant_info.json"
    original_meta = source_meta.read_bytes()
    review = {"findings": "Original observation", "reviewer": "A", "reviewed": True}
    assert client.post(f"/api/sessions/{sid}/review", json=review).status_code == 200
    queued = client.post("/api/assessments", json={"source": "existing", **scoped}).json()
    entry = client.post(
        "/api/assessment-trash", json={"kind": "draft", "draftId": queued["id"]}
    ).json()
    assert cards()[key]["trials"][0]["reviewStatus"] == "reviewed"
    hidden = client.post("/api/assessment-trash", json={"kind": "task", **scoped}).json()
    assert cards()[key]["trials"] == []
    assert cards()["floor_b::001"]["trialCount"] == cards()["floor_a::002"]["trialCount"] == 1
    assert client.post(f"/api/assessment-trash/{entry['id']}/restore").status_code == 409
    assert client.post("/api/assessments", json={"source": "existing", **scoped}).status_code == 404
    assert client.post(f"/api/assessments/{queued['id']}/analyze").status_code == 404
    with pytest.raises(HTTPException, match="Restore"):
        store.publish(queued, store.Setup(**queued["setup"]))
    assert client.post(f"/api/assessment-trash/{hidden['id']}/restore").status_code == 200
    assert cards()[key]["trials"][0]["sessionId"] == sid
    assert client.get(f"/api/sessions/{sid}/review").json()["findings"] == review["findings"]
    assert client.get(f"/api/assessments/{queued['id']}").status_code == 404
    assert client.post(f"/api/assessment-trash/{entry['id']}/restore").status_code == 200
    assert client.get(f"/api/assessments/{queued['id']}").status_code == 200

    sibling = client.post(
        "/api/assessments", json={"participantKey": key, "group": {"taskName": "Pack box"}}
    ).json()
    sibling = add_video(client, sibling, source)
    sibling["setup"].update(heightCm=175, footageReviewed=True, conditionsReviewed=True)
    assert store.publish(sibling, store.validate_ready(sibling))["participantKey"] == key
    assert cards()[key]["trialCount"] == 2
    assert cards()["floor_b::001"]["trialCount"] == cards()["floor_a::002"]["trialCount"] == 1
    assert source_meta.read_bytes() == original_meta
    assert (source_meta.parent / folder / "raw" / "front.avi").read_bytes() == source.read_bytes()


def add_video(client, draft, source):
    response = client.post(
        f"/api/assessments/{draft['id']}/videos?name=Original%20video.avi",
        content=source.read_bytes(),
    )
    assert response.status_code == 200, response.text
    return response.json()


def confirm(draft):
    draft["setup"].update(
        studyName="Study A",
        participantId="P001",
        taskName="Lift the box",
        loadKg=4.5,
        footageReviewed=True,
        conditionsReviewed=True,
        synchronizationConfirmed=True,
    )
    return draft


def test_upload_resume_and_publish_leave_original_unchanged(workflow_env):
    client, root, source, _ = workflow_env
    original = source.read_bytes()
    draft = add_video(client, make_draft(client), source)
    draft = confirm(draft)
    response = client.post(
        f"/api/assessments/{draft['id']}",
        json={"setup": draft["setup"], "step": 3, "roles": ["camera"]},
    )
    assert response.status_code == 200
    resumed = client.get(f"/api/assessments/{draft['id']}").json()
    assert resumed["step"] == 3
    assert resumed["videos"][0]["role"] == "camera"
    assert resumed["setup"]["loadKg"] == 4.5
    identity = store.publish(resumed, store.validate_ready(resumed))
    contexts = participant_catalog(str(root))
    assert contexts[0]["key"] == identity["participantKey"]
    raw = Path(contexts[0]["tasks"][0]["path"]) / "raw"
    assert (raw / "cam_0.avi").read_bytes() == original == source.read_bytes()
    assert store.publish(resumed, store.Setup(**resumed["setup"])) == identity
    assert len(list(raw.iterdir())) == 1


def test_failed_and_oversize_uploads_do_not_enter_draft(workflow_env, monkeypatch):
    client, _, source, _ = workflow_env
    draft = make_draft(client)
    url = f"/api/assessments/{draft['id']}/videos"
    assert client.post(url + "?name=file.exe", content=b"bad").status_code == 422
    assert client.post(url + "?name=corrupt.mp4", content=b"bad").status_code == 422
    monkeypatch.setattr(store, "MAX_VIDEO_BYTES", 8)
    assert client.post(url + "?name=large.avi", content=source.read_bytes()).status_code == 413
    assert store.load(draft["id"])["videos"] == []
    assert not list((store.draft_dir(draft["id"]) / "uploads").glob("*"))
    assert draft["id"] not in store.UPLOADING


def test_confirmation_and_synchronization_are_required(workflow_env):
    client, _, source, _ = workflow_env
    draft = confirm(add_video(client, make_draft(client), source))
    draft["setup"]["conditionsReviewed"] = False
    with pytest.raises(Exception, match="Review the footage and task conditions"):
        store.validate_ready(draft)
    draft["setup"]["conditionsReviewed"] = True
    draft["videos"].append({**draft["videos"][0], "role": "side"})
    draft["setup"]["cameraMode"] = "multi"
    draft["videos"][0]["role"] = "front"
    draft["setup"]["synchronizationConfirmed"] = False
    with pytest.raises(Exception, match="aligned start times"):
        store.validate_ready(draft)
    draft["setup"]["synchronizationConfirmed"] = True
    draft["videos"][1]["durationSec"] += 20
    with pytest.raises(Exception, match="durations differ"):
        store.validate_ready(draft)


def test_capture_takes_are_distinct_and_folder_changes_are_blocked(workflow_env, monkeypatch):
    client, root, source, _ = workflow_env

    class CameraService:
        preview_active = False
        is_recording = False

        def start_preview(self, specs):
            self.specs = specs
            self.preview_active = True

        def start_recording(self, path):
            self.path = Path(path)
            self.is_recording = True

        def snapshot(self):
            return {
                "preview_active": self.preview_active,
                "recording": self.is_recording,
                "cameras": {},
            }

        def stop_recording(self):
            self.path.mkdir(parents=True)
            raw = self.path / "take.avi"
            shutil.copy2(source, raw)
            self.is_recording = False
            return {
                "duration_sec": 2,
                "raw_videos": {0: str(raw)},
                "camera_specs": [{"device_id": 0, "label": "front"}],
            }

        def stop_preview(self):
            self.preview_active = False

    monkeypatch.setattr(routes, "get_recording_service", lambda: service)
    service = CameraService()
    draft = make_draft(client, "live")
    second = make_draft(client, "live")
    base = f"/api/assessments/{draft['id']}"
    preview = {"cameras": [{"deviceId": 0, "role": "front"}]}
    assert client.post(base + "/capture/preview", json=preview).status_code == 200
    assert (
        client.post(f"/api/assessments/{second['id']}/capture/preview", json=preview).status_code
        == 409
    )
    (root / "study_a").mkdir()
    (root / "study_a" / "study_config.json").write_text("{}")
    assert client.post("/api/studies-folder", json={"path": str(root)}).status_code == 409
    assert client.post(base, json={"setup": draft["setup"]}).status_code == 409
    for _ in range(2):
        assert client.post(base + "/capture/start").status_code == 200
        assert client.post(base + "/capture/close").status_code == 409
        stopped = client.post(base + "/capture/stop")
        assert stopped.status_code == 200, stopped.text
    result = stopped.json()
    assert len(result["takes"]) == 2 and len(result["videos"]) == 1
    assert result["takes"][0]["id"] != result["takes"][1]["id"]
    assert result["takes"][0]["videos"][0]["path"] != result["takes"][1]["videos"][0]["path"]
    assert client.post(base + "/capture/close").status_code == 200
    first = result["takes"][0]
    chosen = client.post(base + f"/takes/{first['id']}").json()
    assert chosen["videos"] == first["videos"]


def test_preferences_apply_only_to_new_drafts(workflow_env):
    client, _, _, _ = workflow_env
    first = make_draft(client)
    response = client.post(
        "/api/assessment-preferences",
        json={
            "primaryMethod": "rula",
            "samplingFps": 15,
            "assumptionProfile": None,
            "cameraFps": 24,
        },
    )
    assert response.status_code == 200
    assert make_draft(client)["setup"]["primaryMethod"] == "rula"
    assert store.load(first["id"])["setup"]["primaryMethod"] == "reba"


def test_selected_library_camera_stays_bound_to_analysis_replay_and_rerun(
    workflow_env, monkeypatch
):
    client, root, source, _ = workflow_env
    draft = client.post("/api/assessments", json={"source": "video", "cameraMode": "multi"}).json()
    other = source.with_name("side.avi")
    other.write_bytes(source.read_bytes() + b"side-camera recording")
    draft = confirm(add_video(client, add_video(client, draft, source), other))
    original = store.publish(draft, store.validate_ready(draft))
    single = client.post(
        "/api/assessments", json={"source": "existing", "cameraMode": "single", **original}
    ).json()
    assert single["videos"] == []
    side = single["availableVideos"][1]
    response = client.post(
        f"/api/assessments/{single['id']}",
        json={"setup": single["setup"], "videoIds": [side["id"]]},
    )
    assert response.status_code == 200
    selected = confirm(response.json())
    assert [(v["id"], v["role"]) for v in selected["videos"]] == [(side["id"], "camera")]
    identity = store.publish(selected, store.validate_ready(selected))
    assert identity["taskFolder"] != original["taskFolder"]
    ctx = sessions.find_context(str(root), identity["participantKey"])
    task = sessions.find_trial(ctx, identity["taskFolder"])
    raw = Path(task["path"]) / "raw"
    (raw / "aaa-unrelated.avi").write_bytes(b"unrelated later recording")
    cfg = store.overrides(selected, store.Setup(**selected["setup"]))
    ref = sessions.SessionRef(
        ctx["study"]["folder_name"], ctx["participant_id"], task["folder_name"], 10
    )
    monkeypatch.setattr(sessions, "load_run_config", lambda _: cfg)
    videos = sessions.session_videos(str(root), ref)
    assert [(v["camera_index"], v["name"]) for v in videos] == [(0, "cam_0.avi")]
    assert Path(videos[0]["path"]).read_bytes() == Path(side["path"]).read_bytes()
    registry = jobs.analysis_jobs
    monkeypatch.setattr(registry, "_existing_session", lambda *_: ref.session_id)
    monkeypatch.setattr(jobs, "load_run_config", lambda _: cfg)
    targets, _ = registry._prepare_targets([(ctx, task)], sampling=10, rerun=True, trial_scope=True)
    assert targets[0].videos == (str(raw / "cam_0.avi"),)
    (raw / "cam_0.avi").write_bytes(b"file replaced with different footage")
    assert sessions.session_videos(str(root), ref) == []
    with pytest.raises(jobs.AnalysisJobError, match="changed"):
        registry._prepare_targets([(ctx, task)], sampling=10, rerun=True, trial_scope=True)


def test_mode_switch_requires_a_retained_view_and_keeps_the_other_upload(workflow_env):
    client, _, source, _ = workflow_env
    draft = add_video(client, make_draft(client), source)
    base = f"/api/assessments/{draft['id']}"
    assert (
        client.post(base + "/videos?name=extra.avi", content=source.read_bytes()).status_code == 422
    )
    setup = {**draft["setup"], "cameraMode": "multi"}
    draft = client.post(base, json={"setup": setup}).json()
    with pytest.raises(Exception, match="two or three"):
        store.validate_ready(confirm(draft))
    draft = add_video(client, draft, source)
    selected_ids = [v["id"] for v in draft["videos"]]
    setup = {**draft["setup"], "cameraMode": "single"}
    assert client.post(base, json={"setup": setup}).status_code == 422
    result = client.post(base, json={"setup": setup, "videoIds": selected_ids[1:]}).json()
    assert [v["id"] for v in result["availableVideos"]] == selected_ids
    assert [v["id"] for v in result["videos"]] == selected_ids[1:]
    assert result["setup"]["footageReviewed"] is False
    result = client.post(
        base, json={"setup": {**setup, "cameraMode": "multi"}, "videoIds": selected_ids}
    ).json()
    assert [v["role"] for v in result["videos"]] == ["front", "side"]


def test_legacy_result_keeps_camera_indexes_when_original_file_is_missing(
    workflow_env, monkeypatch
):
    client, root, source, _ = workflow_env
    draft = confirm(add_video(client, make_draft(client), source))
    identity = store.publish(draft, store.validate_ready(draft))
    context = sessions.find_context(str(root), identity["participantKey"])
    task = sessions.find_trial(context, identity["taskFolder"])
    raw = Path(task["path"]) / "raw"
    (raw / "000-new-view.avi").write_bytes(source.read_bytes())
    ref = sessions.SessionRef(context["study"]["folder_name"], "P001", task["folder_name"], 10)
    monkeypatch.setattr(sessions, "load_run_config", lambda _: {})
    monkeypatch.setattr(
        sessions,
        "load_summary",
        lambda _: {
            "video_metadata": [
                {"camera_index": 1, "path": "/former/mount/cam_0.avi", "camera_role": "side"},
                {"camera_index": 0, "path": "/former/mount/gone.avi", "camera_role": "front"},
            ]
        },
    )
    assert sessions.session_videos(str(root), ref) == [
        {"camera_index": 1, "name": "cam_0.avi", "role": "side", "path": str(raw / "cam_0.avi")},
    ]


def test_unprocessed_complete_library_selection_retains_identity(workflow_env):
    client, root, source, _ = workflow_env
    original = confirm(add_video(client, make_draft(client), source))
    identity = store.publish(original, store.validate_ready(original))
    existing = client.post("/api/assessments", json={"source": "existing", **identity}).json()
    assert existing["setup"]["cameraMode"] == "single"
    draft = confirm(existing)
    assert store.publish(draft, store.validate_ready(draft)) == identity
    assert draft["recordings"] == original["recordings"]
    assert len(sessions.find_context(str(root), identity["participantKey"])["tasks"]) == 1


def test_batch_and_worker_use_saved_camera_identity_and_role(tmp_path, monkeypatch):
    import batch_process
    from pipeline.recording_sources import recording_entry, verify_run_recordings

    task = tmp_path / "trial"
    raw = task / "raw"
    raw.mkdir(parents=True)
    saved = raw / "side.mov"
    saved.write_bytes(b"selected camera")
    extra = raw / "000-unrelated.mp4"
    extra.write_bytes(b"late addition")
    original = recording_entry(saved, "camera")
    (task / "source_manifest.json").write_text(json.dumps({"recordings": [original]}))
    configs = []

    class Processor:
        def __init__(self, config):
            configs.append(config)

        def process(self, paths, output):
            assert paths == [str(saved)]
            verify_run_recordings(paths, configs[-1])

    monkeypatch.setattr(batch_process, "VideoProcessor", Processor)
    job = {
        "participant_id": "P",
        "task_name": "Test",
        "task_path": task,
        "video_paths": [str(extra), str(saved)],
        "assessment_override": None,
    }
    default = {"fusion": {"camera_roles": {"front": 0, "side": 1, "iso": 2}}}
    assert batch_process.run_job(job, default, str(tmp_path / "out"))["ok"]
    assert configs[-1]["fusion"]["camera_roles"] == {"camera": 0}
    assert configs[-1]["assessment_workflow"]["recordings"] == [original]
    saved.write_bytes(b"replaced camera")
    with pytest.raises(ValueError, match="changed"):
        verify_run_recordings([str(saved)], configs[-1])
    failed = batch_process.run_job(job, default, str(tmp_path / "out"))
    assert not failed["ok"] and "changed" in failed["error"]


def test_review_is_explicit_and_invalidated_when_result_changes(
    workflow_env, monkeypatch, tmp_path
):
    client, _, _, _ = workflow_env
    result = tmp_path / "result"
    result.mkdir()
    (result / "video_summary.json").write_text('{"maximum":4}')
    (result / "run_config.json").write_text("{}")
    monkeypatch.setattr(store, "resolve_result_dir", lambda _: str(result))
    url = "/api/sessions/study_a__001__task_a__10/review"
    assert client.get(url).json()["reviewed"] is False
    assert client.post(url, json={"reviewed": True}).status_code == 422
    saved = client.post(
        url,
        json={
            "reviewer": "Reviewer A",
            "findings": "Reach observed",
            "actions": "Adjust shelf",
            "reviewed": True,
        },
    )
    assert saved.status_code == 200
    assert client.get(url).json()["reviewed"] is True
    (result / "video_summary.json").write_text('{"maximum":6}')
    stale = client.get(url).json()
    assert stale["reviewed"] is False and stale["stale"] is True
    assert stale["findings"] == "Reach observed"


def test_workflow_handoff_records_inputs_without_editing_source_config(workflow_env, monkeypatch):
    client, _, source, registry = workflow_env
    draft = confirm(add_video(client, make_draft(client), source))
    store.save(draft)
    received = []

    def start(**kwargs):
        received.append(kwargs)
        return {"jobId": "job-a"}

    monkeypatch.setattr(jobs, "start_analysis_job", start)
    monkeypatch.setattr(jobs, "get_analysis_job", lambda _: {"status": "running", "jobId": "job-a"})
    response = client.post(f"/api/assessments/{draft['id']}/analyze")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "processing"
    override = received[0]["assessment_overrides"]
    assert override["task_inputs"]["load_kg"] == 4.5
    assert override["assessment_workflow"]["draft_id"] == draft["id"]
    assert override["fusion"]["camera_roles"] == {"camera": 0}
    saved_inputs = list(Path(registry.studies_root).rglob("assessment_inputs.yaml"))
    assert len(saved_inputs) == 1
    assert json.loads(saved_inputs[0].read_text())["task_inputs"]["load_kg"] == 4.5
    assert (
        client.post(f"/api/assessments/{draft['id']}/analyze").json()["analysis"]
        == response.json()["analysis"]
    )
    assert len(received) == 1


def test_draft_completion_recovers_after_server_restart(workflow_env, monkeypatch, tmp_path):
    client, _, _, _ = workflow_env
    draft = make_draft(client)
    draft["analysis"] = {"jobId": "gone", "sessionId": "s__001__t__10"}
    result = tmp_path / "results"
    result.mkdir()
    monkeypatch.setattr(store, "resolve_result_dir", lambda _: str(result))
    (result / "run_config.json").write_text(
        json.dumps({"assessment_workflow": {"draft_id": draft["id"]}})
    )
    assert store.describe(draft)["status"] == "ready"
    (result / "run_config.json").write_text("{}")
    assert store.describe(draft)["status"] == "interrupted"


def test_local_access_and_draft_identifiers_are_checked(workflow_env):
    client, _, _, _ = workflow_env
    assert client.get("/api/assessments/not-an-id").status_code == 404
    assert (
        client.post(
            "/api/capture/devices", headers={"Origin": "https://unrelated.example"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/assessments",
            json={"source": "video"},
            headers={"Origin": "https://unrelated.example"},
        ).status_code
        == 403
    )


def test_new_routes_serve_the_frontend_and_reject_malformed_paths(tmp_path):
    from fastapi import FastAPI

    (tmp_path / "index.html").write_text("<html>entry</html>")
    app = FastAPI()
    app.mount("/", main.WebClientFiles(directory=tmp_path, html=True))
    client = TestClient(app)
    for path in (
        "/",
        "/overview",
        "/new",
        "/settings",
        "/assessments",
        "/assessments/" + "a" * 32,
        "/library",
        "/record",
    ):
        assert client.get(path).status_code == 200
    assert client.get("/assessments/invalid").status_code == 404


def test_existing_recording_prefills_setup_without_changing_its_inputs(workflow_env):
    client, root, source, _ = workflow_env
    original = confirm(add_video(client, make_draft(client), source))
    original["setup"]["primaryMethod"] = "rula"
    identity = store.publish(original, store.validate_ready(original))
    source_inputs = next(root.rglob("assessment_inputs.yaml"))
    before = source_inputs.read_bytes()
    response = client.post("/api/assessments", json={"source": "existing", **identity})
    assert response.status_code == 200, response.text
    draft = response.json()
    assert draft["setup"]["loadKg"] == 4.5
    assert draft["setup"]["primaryMethod"] == "rula"
    assert draft["setup"]["heightCm"] is None
    draft["setup"]["loadKg"] = 7.5
    assert client.post(f"/api/assessments/{draft['id']}", json={"setup": draft["setup"]}).status_code == 200
    assert source_inputs.read_bytes() == before
