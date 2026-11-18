from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import WebClientFiles
from paths import resolve_studies_root


def test_study_root_precedence_and_manifest_requirement(tmp_path, monkeypatch):
    monkeypatch.delenv("POSTURE_STUDIES_ROOT", raising=False)
    project = tmp_path / "projects" / "checkout"
    project.mkdir(parents=True)
    sibling = project.parent / "studies"
    ancestor = tmp_path / "studies"
    manifest = ancestor / "study_01" / "study_config.json"
    manifest.parent.mkdir(parents=True)
    assert resolve_studies_root(project) == sibling
    manifest.write_text("{}")
    assert resolve_studies_root(project) == ancestor
    sibling.mkdir()
    assert resolve_studies_root(project) == sibling
    override = tmp_path / "explicit"  # Honor an explicitly selected, unmounted volume.
    monkeypatch.setenv("POSTURE_STUDIES_ROOT", str(override))
    assert resolve_studies_root(project) == override


def test_production_static_mount_supports_real_workspace_links_only(tmp_path):
    (tmp_path / "index.html").write_text("<html>client entry</html>", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "styles.css").write_text("body{}", encoding="utf-8")
    app = FastAPI()
    app.mount("/", WebClientFiles(directory=tmp_path, html=True))
    client = TestClient(app)
    session = "/s/study_01__001__task1__10"
    for path in ("/", "/record", "/library/", session, session + "/reba", session + "/rula?t=3"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.text == "<html>client entry</html>"
        assert client.head(path).content == b""
    assert client.get("/assets/styles.css").text == "body{}"
    for path in ("/api/missing", "/assets/missing.js", "/s/logo.svg", "/library/child", session + "/typo"):
        assert client.get(path).status_code == 404
