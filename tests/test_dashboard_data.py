import json

import dashboard.data_loader as data_loader


def test_native_sampling_is_discovered(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "RESULTS_DIR_STR", str(tmp_path))
    result = tmp_path / "native" / "study" / "001" / "task"
    result.mkdir(parents=True)
    (result / "run_config.json").write_text(
        json.dumps({"video": {"target_fps": None}}), encoding="utf-8"
    )

    data_loader.get_processed_fps.clear()
    assert data_loader.get_processed_fps("study", "001", "task") == [None]


def test_current_result_detection_rejects_archived_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "RESULTS_DIR_STR", str(tmp_path))
    result = tmp_path / "native" / "study" / "001" / "task"
    (result / "scores").mkdir(parents=True)
    (result / "scores" / "frame_scores.csv").write_text(
        "frame_number,reba_score\n0,1\n", encoding="utf-8"
    )

    assert data_loader.detect_current_results("study", "001", "task", None) is None


def test_participant_catalog_uses_participant_label_not_internal_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(data_loader, "RESULTS_DIR_STR", str(tmp_path / "results"))
    study = tmp_path / "study_internal_20260715"
    participant = study / "participant_internal_001"
    task = participant / "task_internal"
    task.mkdir(parents=True)
    (study / "study_config.json").write_text(
        json.dumps(
            {
                "study_name": "study_internal_20260715",
                "created_at": "2026-07-15T10:00:00",
                "tasks": [
                    {
                        "task_id": 1,
                        "name": "Controlled lift",
                        "folder_name": "task_internal",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (participant / "participant_info.json").write_text(
        json.dumps({"participant_id": "P-001"}), encoding="utf-8"
    )
    data_loader.participant_catalog.clear()
    data_loader.discover_studies.clear()
    data_loader.discover_participants.clear()
    data_loader.discover_tasks.clear()

    catalog = data_loader.participant_catalog(str(tmp_path))

    assert catalog[0]["label"] == "Participant P-001 · 2026-07-15"
    assert "study_internal" not in catalog[0]["label"]
    assert catalog[0]["tasks"][0]["name"] == "Controlled lift"
