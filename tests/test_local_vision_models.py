from dataclasses import replace

import pytest
import requests
from fastapi.testclient import TestClient

from api import assessments, main, sessions
from assistant import backend, models
from assistant.config import AssistantConfig
from assistant.focused_review import predictor


@pytest.fixture
def workstation(tmp_path, monkeypatch):
    installed = {
        "small:4b": {"capabilities": ["completion", "vision"]},
        "large:26b": {"capabilities": ["completion", "vision", "tools"]},
        "writer:7b": {"capabilities": ["completion", "tools"]},
        "alias:latest": {
            "capabilities": ["completion", "vision"],
            "remote_host": "https://ollama.com",
        },
        "unknown:latest": {},
    }

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def get(url, **kwargs):
        assert url.endswith("/api/tags")
        return Response(
            {
                "models": [
                    {"name": tag, "digest": f"id-{tag}", "size": index + 10}
                    for index, tag in enumerate(installed)
                ]
            }
        )

    def post(url, *, json, **kwargs):
        assert url.endswith("/api/show")
        return Response(installed[json["model"]])

    config = AssistantConfig(scene_model="large:26b", chat_model="large:26b")
    monkeypatch.setattr(models, "load_config", lambda: config)
    monkeypatch.setattr("api.assistant_api.load_config", lambda: config)
    monkeypatch.setattr(models.requests, "get", get)
    monkeypatch.setattr(models.requests, "post", post)
    monkeypatch.setattr(assessments, "WORKFLOW_ROOT", tmp_path / "workflow")
    monkeypatch.setattr(sessions, "resolve_result_dir", lambda session: str(tmp_path / session))
    monkeypatch.setattr(
        sessions,
        "load_run_config",
        lambda _: {"assessment_workflow": {"vision_model": "large:26b"}},
    )
    models._catalog.cache_clear()
    models._details.cache_clear()
    yield installed
    models._catalog.cache_clear()
    models._details.cache_clear()


def test_discovery_default_override_and_removal(workstation):
    with TestClient(
        main.app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 1234)
    ) as client:
        listed = client.get("/api/assistant/models").json()
        assert [model["tag"] for model in listed["models"]] == ["small:4b", "large:26b"]
        assert (
            client.post(
                "/api/assessment-preferences", json={"visionModel": "writer:7b"}
            ).status_code
            == 422
        )
        assert client.post(
            "/api/assessment-preferences", json={"visionModel": "small:4b"}
        ).is_success
        assert models.effective_config().scene_model == "small:4b"
        assert models.effective_config("task").chat_model == "large:26b"
        path = "/api/sessions/task/assistant/model"
        assert client.post(path, json={"model": "small:4b"}).is_success
        selected = models.effective_config("task")
        assert selected.chat_model == selected.scene_model == "small:4b"
        assert selected.describe()["modelDigest"] == "id-small:4b"
        assert not client.get("/api/assistant/health?sessionId=task").json()["ok"]
        assert (
            models.effective_config("task", requested_model="large:26b").scene_model == "large:26b"
        )
        assert predictor(selected) != predictor(replace(selected, model_digest="new-weights"))
        with pytest.raises(backend.BackendError, match="tool support"):
            list(backend.stream_completion([], tools=[{}], config=selected))
        del workstation["small:4b"]
        with pytest.raises(backend.BackendError, match="changed or was removed"):
            backend.complete([], config=selected)
        client.get("/api/assistant/models?refresh=true").raise_for_status()
        missing = models.effective_config("task")
        assert missing.model_digest == selected.model_digest
        with pytest.raises(backend.BackendError, match="unavailable"):
            backend.complete([], config=missing)
        assert client.post(path, json={"model": None}).is_success
        assert models.effective_config("task").scene_model == "large:26b"


def test_offline_local_vision_and_external_configuration(workstation, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(models.requests, "get", offline)
    assert models.inventory(AssistantConfig(), refresh=True)["state"] == "unreachable"
    with pytest.raises(backend.BackendError, match="Start Ollama"):
        models.require_vision(models.effective_config())
    configured = AssistantConfig(
        endpoint="http://localhost:9000/v1",
        dialect="openai",
        scene_model="server-vision",
        chat_model="server-chat",
    )
    monkeypatch.setattr(models, "load_config", lambda: configured)
    assert models.effective_config() == configured
    assert models.inventory(configured)["models"] == []
