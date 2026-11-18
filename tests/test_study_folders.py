"""Study source selection persists and keeps the API and job runner in agreement."""

import json

import pytest
from fastapi.testclient import TestClient

import api.jobs as jobs
import api.main as main
import paths
from dashboard.cache import clear_all_caches


@pytest.fixture
def folders(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    root = tmp_path / "Studies with spaces"
    study = root / "study_a"
    participant = study / "participant_001"
    participant.mkdir(parents=True)
    (study / "study_config.json").write_text(
        '{"study_name":"Study A","tasks":[]}', encoding="utf-8"
    )
    (participant / "participant_info.json").write_text('{"participant_id":"001"}', encoding="utf-8")
    (root / "Notes.txt").write_text("File contents must not appear in the folder listing.")
    (root / "Empty folder").mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.setattr(main, "STUDIES_ROOT", str(tmp_path / "old"))
    registry = jobs.AnalysisJobRegistry(studies_root=main.STUDIES_ROOT)
    monkeypatch.setattr(main, "analysis_jobs", registry)
    monkeypatch.setattr(jobs, "analysis_jobs", registry)
    clear_all_caches()
    with TestClient(
        main.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 51000)
    ) as client:
        yield client, project, root, registry
    clear_all_caches()


def test_browse_then_select_refreshes_catalog_and_survives_restart(folders, monkeypatch):
    client, project, root, registry = folders
    assert client.get("/api/participants").json()["count"] == 0
    listed = client.get("/api/studies-folder/browse", params={"path": str(root)}).json()
    assert listed["studyCount"] == 1
    assert [f["name"] for f in listed["folders"]] == ["Empty folder", "study_a"]
    assert listed["parent"] == str(root.parent)
    assert listed["folders"][1]["isStudy"] is True
    assert (
        client.get("/api/studies-folder/browse", params={"path": listed["parent"]}).status_code
        == 200
    )
    selected = client.post("/api/studies-folder", json={"path": f'"{root}"'})
    assert selected.status_code == 200
    assert selected.json() == {"path": str(root), "exists": True}
    assert registry.studies_root == str(root)
    assert client.get("/api/health").json()["studiesRoot"] == str(root)
    assert client.get("/api/participants").json()["participants"][0]["participantId"] == "001"
    assert json.loads((project / ".posture-settings.json").read_text())["studies_root"] == str(root)
    monkeypatch.setenv("POSTURE_STUDIES_ROOT", str(root.parent / "stale-env"))
    assert paths.resolve_studies_root(project) == root


@pytest.mark.parametrize("choice", ["empty", "missing", "file", "relative"])
def test_invalid_choice_keeps_existing_root_and_does_not_write_settings(folders, choice):
    client, project, root, registry = folders
    original = registry.studies_root
    values = {
        "empty": str(root / "Empty folder"),
        "missing": str(root / "missing"),
        "file": str(root / "Notes.txt"),
        "relative": "relative/path",
    }
    response = client.post("/api/studies-folder", json={"path": values[choice]})
    assert response.status_code == 422
    assert registry.studies_root == original == main.STUDIES_ROOT
    assert not (project / ".posture-settings.json").exists()


def test_save_failure_leaves_runtime_folder_unchanged(folders, monkeypatch):
    client, project, root, registry = folders
    original = registry.studies_root

    def fail(_root):
        raise PermissionError("read-only settings")

    monkeypatch.setattr(jobs, "save_studies_root", fail)
    assert client.post("/api/studies-folder", json={"path": str(root)}).status_code == 500
    assert registry.studies_root == original == main.STUDIES_ROOT
    assert not (project / ".posture-settings.json").exists()


def test_remote_pages_cannot_browse_or_change_local_folders(folders):
    client, _, root, _ = folders
    headers = {"Origin": "https://unrelated.example"}
    assert (
        client.get(
            "/api/studies-folder/browse", params={"path": str(root)}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.post("/api/studies-folder", json={"path": str(root)}, headers=headers).status_code
        == 403
    )
    assert (
        client.get(
            "/api/studies-folder/browse",
            params={"path": str(root)},
            headers={"Host": "rebound.example"},
        ).status_code
        == 403
    )
    with TestClient(
        main.app, base_url="http://127.0.0.1:8000", client=("192.168.1.5", 51000)
    ) as remote:
        assert (
            remote.get("/api/studies-folder/browse", params={"path": str(root)}).status_code == 403
        )


def test_corrupt_preference_falls_back_to_environment(tmp_path, monkeypatch):
    (tmp_path / ".posture-settings.json").write_text("broken json", encoding="utf-8")
    monkeypatch.setenv("POSTURE_STUDIES_ROOT", str(tmp_path / "configured"))
    assert paths.resolve_studies_root(tmp_path) == tmp_path / "configured"
