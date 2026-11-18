"""Server-side analysis job orchestration for the React dashboard."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import api.jobs as jobs


def _context(
    root: Path,
    *,
    study: str = "study_a",
    participant: str = "001",
    height: Any = 180,
    task_names: tuple[str, ...] = ("task_a",),
) -> dict[str, Any]:
    participant_path = root / study / f"participant_{participant}"
    tasks = []
    for index, name in enumerate(task_names, start=1):
        task_path = participant_path / name
        (task_path / "raw").mkdir(parents=True, exist_ok=True)
        (task_path / "raw" / "cam_0.mp4").write_bytes(b"source for the simulated worker")
        tasks.append(
            {
                "task_id": index,
                "name": f"Task {name[-1].upper()}",
                "folder_name": name,
                "path": str(task_path),
            }
        )
    return {
        "key": f"{study}::{participant}",
        "participant_id": participant,
        "participant": {"info": {"height_cm": height}},
        "study": {"folder_name": study},
        "tasks": tasks,
    }


def _config(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "schema_version: 4",
                "video:",
                "  target_fps: 5",
                "pose:",
                "  use_gpu: false",
                "task_inputs:",
                "  load_kg: null",
                "rnle:",
                "  participant_height_cm: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class _ImmediateBatch:
    def __init__(self, trials: list[dict[str, Any]], *, fail: bool = False):
        self.trials = trials
        self.fail = fail

    def describe(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "total": len(self.trials),
            "completed": len(self.trials),
            "failed": len(self.trials) if self.fail else 0,
            "current_index": -1,
            "current_label": None,
            "current": None,
            "outcomes": [
                {
                    "label": trial["label"],
                    "ok": not self.fail,
                    "error": "simulated failure" if self.fail else None,
                    "output_dir": trial["output_dir"] if not self.fail else None,
                }
                for trial in self.trials
            ],
        }


class _JoinedThread:
    @staticmethod
    def join() -> None:
        return None


class _RunningBatch:
    def __init__(self, trials: list[dict[str, Any]]):
        self.trials = trials
        self.finished = threading.Event()

    def describe(self) -> dict[str, Any]:
        if self.finished.is_set():
            return _ImmediateBatch(self.trials).describe()
        return {
            "status": "running",
            "total": len(self.trials),
            "completed": 0,
            "failed": 0,
            "current_index": 0,
            "current_label": self.trials[0]["label"],
            "current": {
                "status": "running",
                "progress_frame": 25,
                "total_frames": 100,
                "stage": "processing",
                "message": "Processing source frames.",
                "elapsed_sec": 2.0,
                "eta_sec": 6.0,
            },
            "outcomes": [],
        }


class _BlockingThread:
    def __init__(self, event: threading.Event):
        self.event = event

    def join(self) -> None:
        self.event.wait(timeout=5)


@pytest.fixture
def prepared(monkeypatch, tmp_path):
    config_path = _config(tmp_path / "config.yaml")
    results = tmp_path / "results"
    contexts: list[dict[str, Any]] = []
    launched: list[list[dict[str, Any]]] = []
    cache_clears: list[bool] = []

    monkeypatch.setattr(jobs, "participant_catalog", lambda _root: contexts)
    monkeypatch.setattr(
        jobs,
        "get_task_results_path",
        lambda study, participant, task, sampling: str(
            results
            / ("native" if sampling is None else f"{sampling}fps")
            / study
            / participant
            / task
        ),
    )
    monkeypatch.setattr(jobs, "get_processed_fps", lambda *_args: [])
    monkeypatch.setattr(jobs, "detect_current_results", lambda *_args: None)
    monkeypatch.setattr(
        jobs,
        "find_videos",
        lambda task_path: {
            "raw": [str(Path(task_path) / "raw" / "cam_0.mp4")],
            "overlay": [],
        },
    )

    def launch(trials):
        launched.append(trials)
        return _ImmediateBatch(trials), _JoinedThread()

    monkeypatch.setattr(jobs, "launch_batch_thread", launch)
    monkeypatch.setattr(jobs, "clear_all_caches", lambda: cache_clears.append(True))
    registry = jobs.AnalysisJobRegistry(
        studies_root=tmp_path / "studies",
        config_path=config_path,
    )
    return registry, contexts, launched, cache_clears, results


def test_trial_uses_server_paths_explicit_sampling_override_and_recorded_height(
    prepared,
):
    registry, contexts, launched, _, results = prepared
    context = _context(Path(registry.studies_root))
    contexts.append(context)
    task_path = Path(context["tasks"][0]["path"])
    (task_path / "assessment_inputs.yaml").write_text(
        "\n".join(
            [
                "video:",
                "  target_fps: 30",
                "task_inputs:",
                "  load_kg: 12",
                "rnle:",
                "  participant_height_cm: 999",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    state = registry.start(
        scope="trial",
        participant_key="study_a::001",
        task_folder="task_a",
    )

    assert state["sampling"] == 10
    assert state["targetCount"] == 1
    assert len(launched) == 1
    trial = launched[0][0]
    assert trial["videos"] == [str(task_path / "raw" / "cam_0.mp4")]
    assert trial["output_dir"] == str(results / "10fps" / "study_a" / "001" / "task_a")
    assert trial["config"]["video"]["target_fps"] == 10
    assert trial["config"]["task_inputs"]["load_kg"] == 12
    assert trial["config"]["rnle"]["participant_height_cm"] == 180.0
    assert state["outcomes"][0]["sessionId"] == "study_a__001__task_a__10"


def test_participant_scope_is_missing_only_and_reports_skips(prepared, monkeypatch):
    registry, contexts, launched, _, _ = prepared
    context = _context(
        Path(registry.studies_root),
        task_names=("task_existing", "task_missing", "task_new"),
    )
    contexts.append(context)

    monkeypatch.setattr(
        jobs,
        "find_videos",
        lambda task_path: {
            "raw": (
                []
                if str(task_path).endswith("task_missing")
                else [str(Path(task_path) / "raw" / "cam_0.mp4")]
            ),
            "overlay": [],
        },
    )
    monkeypatch.setattr(
        jobs,
        "get_processed_fps",
        lambda _study, _participant, task: [10] if task == "task_existing" else [],
    )
    monkeypatch.setattr(
        jobs,
        "detect_current_results",
        lambda _study, _participant, task, _sampling: (
            {"dir": "complete"} if task == "task_existing" else None
        ),
    )

    state = registry.start(scope="participant", participant_key="study_a::001")

    assert state["candidateCount"] == 3
    assert state["targetCount"] == 1
    assert state["skippedCount"] == 2
    assert launched[0][0]["label"].endswith("Task W")
    assert {item["status"] for item in state["skipped"]} == {
        "already_analysed",
        "no_recording",
    }
    existing = next(item for item in state["skipped"] if item["status"] == "already_analysed")
    assert existing["sessionId"] == "study_a__001__task_existing__10"


def test_existing_trial_returns_reviewable_conflict(prepared, monkeypatch):
    registry, contexts, launched, _, _ = prepared
    contexts.append(_context(Path(registry.studies_root)))
    monkeypatch.setattr(jobs, "get_processed_fps", lambda *_args: [10])
    monkeypatch.setattr(
        jobs,
        "detect_current_results",
        lambda *_args: {"dir": "complete"},
    )

    with pytest.raises(jobs.AnalysisJobError) as caught:
        registry.start(
            scope="trial",
            participant_key="study_a::001",
            task_folder="task_a",
        )

    assert caught.value.status_code == 409
    assert caught.value.payload["existingSessionId"] == "study_a__001__task_a__10"
    assert launched == []


def test_missing_only_reuses_a_complete_result_at_another_sampling(prepared, monkeypatch):
    registry, contexts, launched, _, _ = prepared
    contexts.append(_context(Path(registry.studies_root)))
    monkeypatch.setattr(jobs, "get_processed_fps", lambda *_args: [None, 15])
    monkeypatch.setattr(
        jobs,
        "detect_current_results",
        lambda _study, _participant, _task, sampling: (
            {"dir": "complete"} if sampling is None else None
        ),
    )

    state = registry.start(scope="all")

    assert state["targetCount"] == 0
    assert state["skippedCount"] == 1
    assert state["skipped"][0]["sessionId"] == "study_a__001__task_a__native"
    assert launched == []


def test_all_scope_spans_studies_while_study_scope_filters(prepared):
    registry, contexts, launched, _, _ = prepared
    contexts.extend(
        [
            _context(Path(registry.studies_root), study="study_a", participant="001"),
            _context(Path(registry.studies_root), study="study_b", participant="002"),
        ]
    )

    study_state = registry.start(scope="study", study_folder="study_b")
    assert study_state["targetCount"] == 1
    assert launched[0][0]["label"].startswith("002")

    all_state = registry.start(scope="all")
    assert all_state["targetCount"] == 2
    assert {trial["label"].split(" \u00b7 ")[0] for trial in launched[1]} == {"001", "002"}


def test_only_one_top_level_job_can_run_and_active_status_has_frame_progress(
    prepared,
    monkeypatch,
):
    registry, contexts, _, cache_clears, _ = prepared
    contexts.append(_context(Path(registry.studies_root)))
    holder: dict[str, _RunningBatch] = {}

    def launch(trials):
        batch = _RunningBatch(trials)
        holder["batch"] = batch
        return batch, _BlockingThread(batch.finished)

    monkeypatch.setattr(jobs, "launch_batch_thread", launch)
    first = registry.start(
        scope="trial",
        participant_key="study_a::001",
        task_folder="task_a",
    )

    assert first["status"] == "running"
    assert first["current"]["percent"] == 25.0
    assert registry.active()["jobId"] == first["jobId"]
    with pytest.raises(jobs.AnalysisJobError) as caught:
        registry.start(
            scope="trial",
            participant_key="study_a::001",
            task_folder="task_a",
            rerun=True,
        )
    assert caught.value.status_code == 409
    assert caught.value.payload["activeJobId"] == first["jobId"]

    original_root = registry.studies_root
    with pytest.raises(jobs.AnalysisJobError) as folder_error:
        registry.set_studies_root(Path(original_root) / "another")
    assert folder_error.value.status_code == 409
    assert registry.studies_root == original_root

    holder["batch"].finished.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not cache_clears:
        time.sleep(0.01)
    completed = registry.get(first["jobId"])
    assert completed["status"] == "completed"
    assert completed["percent"] == 100.0
    assert cache_clears == [True]
    assert registry.active() is None


def test_failed_single_trial_has_failed_status_and_keeps_review_identity(
    prepared,
    monkeypatch,
):
    registry, contexts, _, _, _ = prepared
    contexts.append(_context(Path(registry.studies_root)))
    monkeypatch.setattr(
        jobs,
        "launch_batch_thread",
        lambda trials: (_ImmediateBatch(trials, fail=True), _JoinedThread()),
    )

    state = registry.start(
        scope="trial",
        participant_key="study_a::001",
        task_folder="task_a",
        rerun=True,
    )

    assert state["status"] == "failed"
    assert state["failedCount"] == 1
    assert state["outcomes"][0]["taskFolder"] == "task_a"
    assert state["outcomes"][0]["error"] == "simulated failure"


def test_all_existing_scope_completes_without_launching(prepared, monkeypatch):
    registry, contexts, launched, cache_clears, _ = prepared
    contexts.append(_context(Path(registry.studies_root)))
    monkeypatch.setattr(jobs, "get_processed_fps", lambda *_args: [10])
    monkeypatch.setattr(
        jobs,
        "detect_current_results",
        lambda *_args: {"dir": "complete"},
    )

    state = registry.start(scope="all")

    assert state["status"] == "completed"
    assert state["targetCount"] == 0
    assert state["skippedCount"] == 1
    assert state["percent"] == 100.0
    assert launched == []
    assert cache_clears == []


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"scope": "folder"}, 422),
        ({"scope": "trial", "participant_key": "study_a::001"}, 422),
        ({"scope": "study"}, 422),
        ({"scope": "all", "sampling": 2}, 422),
    ],
)
def test_invalid_requests_fail_before_catalog_or_worker(prepared, kwargs, status):
    registry, _, launched, _, _ = prepared
    with pytest.raises(jobs.AnalysisJobError) as caught:
        registry.start(**kwargs)
    assert caught.value.status_code == status
    assert launched == []


def test_saved_workflow_conditions_reach_worker_and_survive_rerun(prepared, monkeypatch):
    registry, contexts, launched, _, _ = prepared
    with registry.config_path.open("a", encoding="utf-8") as config:
        config.write("fusion:\n  camera_roles:\n    front: 0\n    side: 1\n    iso: 2\n")
    contexts.append(_context(Path(registry.studies_root)))
    task_inputs = Path(contexts[0]["tasks"][0]["path"]) / "assessment_inputs.yaml"
    task_inputs.write_text("task_inputs: {load_kg: 2.0, coupling: null}\n", encoding="utf-8")
    override = {
        "task_inputs": {"load_kg": 4.5},
        "fusion": {"camera_roles": {"side": 0}},
        "assessment_workflow": {"draft_id": "a" * 32},
    }
    registry.start(
        scope="trial",
        participant_key="study_a::001",
        task_folder="task_a",
        assessment_overrides=override,
    )
    assert launched[0][0]["config"]["task_inputs"]["load_kg"] == 4.5
    assert launched[0][0]["config"]["fusion"]["camera_roles"] == {"side": 0}
    monkeypatch.setattr(jobs, "get_processed_fps", lambda *_: [10])
    monkeypatch.setattr(jobs, "detect_current_results", lambda *_: {"dir": "complete"})
    previous = launched[0][0]["config"]
    monkeypatch.setattr(jobs, "load_run_config", lambda _: previous)
    monkeypatch.setattr(jobs, "confirmed_conditions", lambda _: {"coupling": "fair"})
    registry.start(scope="trial", participant_key="study_a::001", task_folder="task_a", rerun=True)
    assert launched[1][0]["config"]["task_inputs"]["load_kg"] == 4.5
    assert launched[1][0]["config"]["fusion"]["camera_roles"] == {"side": 0}
    assert launched[1][0]["config"]["task_inputs"]["coupling"] == "fair"
    task_inputs.write_text("task_inputs: {load_kg: 7.0}\n", encoding="utf-8")
    registry.start(scope="trial", participant_key="study_a::001", task_folder="task_a", rerun=True)
    assert launched[2][0]["config"]["task_inputs"]["load_kg"] == 7.0
