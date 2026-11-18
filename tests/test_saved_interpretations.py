"""Persistence of selected visual evidence across model changes and reanalysis."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import assessments, interpretations
from assistant import interpretation, models, scene


def test_library_reopens_without_ollama_and_exports_only_selected_observations(
    tmp_path, monkeypatch
):
    result = tmp_path / "result"
    result.mkdir()
    for name in ("run_config.json", "video_summary.json"):
        (result / name).write_text("{}")
    monkeypatch.setattr(assessments, "WORKFLOW_ROOT", tmp_path / "notes")
    monkeypatch.setattr(assessments, "resolve_result_dir", lambda _: result)
    monkeypatch.setattr(interpretations, "resolve_result_dir", lambda _: result)
    monkeypatch.setattr(
        models,
        "effective_config",
        lambda _, requested_model: SimpleNamespace(
            scene_model=requested_model, model_digest=f"digest-{requested_model}"
        ),
    )
    monkeypatch.setattr(scene, "load_scene", lambda *args, **kwargs: None)
    evidence = [{"frame": 12, "sec": 1.2, "camera": 0, "view": "raw", "person": "worker"}]

    def build(_, *, config):
        return {
            "model": config.scene_model,
            "fingerprint": config.model_digest,
            "builtAt": "test",
            "frames_sampled": [12],
            "findings": [
                {
                    "fact": "Reaches forward — with the right hand.",
                    "category": "phase",
                    "visibility": "clear",
                    "evidence": evidence,
                },
                {
                    "fact": "Other observation",
                    "category": "contact",
                    "visibility": "partial",
                    "evidence": evidence,
                },
                {"fact": "Uncited observation", "evidence": []},
            ],
        }

    monkeypatch.setattr(scene, "build_scene", build)
    assert interpretations.saved("session") == []
    (first,) = interpretation.generate("session", "small")
    assert len(first["observations"]) == 2 and first["selected"] == []
    assert "—" not in first["observations"][0]["text"]
    interpretations.select("session", first["id"], "1", True)
    second, earlier = interpretation.generate("session", "large")
    assert earlier["selected"] == ["1"] and second["modelDigest"] == "digest-large"
    assert len(interpretation.generate("session", "small")) == 2
    monkeypatch.setattr(models, "effective_config", lambda *a, **k: pytest.fail("Contacted Ollama"))
    assert len(interpretations.saved("session")) == 2
    (report,) = interpretations.for_report("session")
    assert report["observations"] == [first["observations"][0]]
    assert report["observations"][0]["evidence"] == evidence
    with pytest.raises(HTTPException) as invalid:
        interpretations.select("session", first["id"], "unknown", True)
    assert invalid.value.status_code == 404
    (result / "run_config.json").write_text('{"updated": true}')
    assert all(reading["stale"] for reading in interpretations.saved("session"))
    assert interpretations.for_report("session") == []
    with pytest.raises(HTTPException) as stale:
        interpretations.select("session", first["id"], "2", True)
    assert stale.value.status_code == 409
