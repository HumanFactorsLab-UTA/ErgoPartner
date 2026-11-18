import json
from pathlib import Path

from dashboard.processor_wrapper import (
    ProcessingJob,
    _OutputProcessLock,
    launch_processing_thread,
    reconcile_processing_thread,
    run_processing,
)
from pipeline.recording_sources import recording_entry


def _write_processor_artifacts(
    result_dir: str | Path,
    marker: str,
    omit: set[str] | None = None,
) -> None:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    omitted = omit or set()
    artifacts = {
        "ASSESSMENT_REPORT.md": f"# Assessment report\n\n{marker}\n",
        "frame_results.csv": f"frame,marker\n0,{marker}\n",
        "frame_results.jsonl": json.dumps({"frame": 0, "marker": marker}) + "\n",
        "run_config.json": json.dumps({"schema_version": 4, "marker": marker}),
        "video_summary.json": json.dumps({"marker": marker, "processed_frames": 1}),
    }
    for filename, payload in artifacts.items():
        if filename not in omitted:
            (result_path / filename).write_text(payload, encoding="utf-8")


def _write_complete_result(result_dir: Path, marker: str) -> None:
    _write_processor_artifacts(result_dir, marker)
    (result_dir / "run_metadata.json").write_text(
        json.dumps({"marker": marker, "output_dir": str(result_dir)}),
        encoding="utf-8",
    )


def _snapshot_files(result_dir: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(result_dir.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }


def _managed_siblings(final_output: Path, kind: str) -> list[Path]:
    prefix = f".{final_output.name}.{kind}-"
    return sorted(path for path in final_output.parent.iterdir() if path.name.startswith(prefix))


def test_dashboard_wrapper_does_not_create_processor_rejected_sentinel(monkeypatch, tmp_path):
    observed_entries = []

    class FakeProcessor:
        def __init__(self, config):
            assert config["schema_version"] == 4

        def process(self, video_paths, output_dir, progress_callback):
            observed_entries.extend(path.name for path in Path(output_dir).iterdir())
            _write_processor_artifacts(output_dir, "first-run")
            progress_callback(1, 1)
            return output_dir

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        lambda: FakeProcessor,
    )
    job = ProcessingJob()
    final_output = tmp_path / "result"

    run_processing(
        job,
        ["recording.mp4"],
        str(final_output),
        {"schema_version": 4},
    )

    state = job.get_state()
    assert ".processing_in_progress" not in observed_entries
    assert state["status"] == "completed", state["error"]
    assert state["stage"] == "completed"
    assert state["progress_frame"] == 1
    assert state["output_dir"] == str(final_output)
    assert (final_output / "run_metadata.json").is_file()
    assert _managed_siblings(final_output, "staging") == []


def test_import_failure_becomes_visible_retryable_error(monkeypatch, tmp_path):
    job = ProcessingJob()
    observed_statuses = []

    def fail_loader():
        observed_statuses.append(job.get_state()["status"])
        raise ImportError("simulated MediaPipe startup failure")

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        fail_loader,
    )
    final_output = tmp_path / "result"

    run_processing(job, ["recording.mp4"], str(final_output), {"schema_version": 4})

    state = job.get_state()
    assert observed_statuses == ["running"]
    assert state["status"] == "error"
    assert state["stage"] == "error"
    assert "simulated MediaPipe startup failure" in state["error"]
    assert not final_output.exists()
    assert _managed_siblings(final_output, "staging") == []


def test_source_replacement_during_run_preserves_previous_result(monkeypatch, tmp_path):
    source = tmp_path / "camera.avi"
    source.write_bytes(b"reviewed recording")
    config = {"assessment_workflow": {"recordings": [recording_entry(source, "camera")]}}
    output = tmp_path / "result"
    _write_complete_result(output, "previous")
    before = _snapshot_files(output)

    class Processor:
        def __init__(self, config):
            pass

        def process(self, video_paths, output_dir, progress_callback):
            _write_processor_artifacts(output_dir, "staged")
            source.write_bytes(b"replacement during processing")
            return output_dir

    monkeypatch.setattr("dashboard.processor_wrapper._load_video_processor_class", lambda: Processor)
    job = ProcessingJob()
    run_processing(job, [str(source)], str(output), config)
    assert job.get_state()["status"] == "error"
    assert "changed since setup" in job.get_state()["error"]
    assert _snapshot_files(output) == before
    assert _managed_siblings(output, "staging") == []


def test_dead_processing_thread_is_reconciled_to_error():
    class DeadThread:
        @staticmethod
        def is_alive():
            return False

    job = ProcessingJob()
    state = reconcile_processing_thread(job, DeadThread())

    assert state["status"] == "error"
    assert "stopped before reporting completion" in state["error"]


def test_reconciliation_preserves_completed_job():
    job = ProcessingJob()
    job._set_running()
    job._set_completed("results")

    state = reconcile_processing_thread(job, None)

    assert state["status"] == "completed", state["error"]
    assert state["output_dir"] == "results"


def test_thread_start_failure_becomes_visible_error(monkeypatch, tmp_path):
    def fail_start(_thread):
        raise RuntimeError("simulated thread start failure")

    monkeypatch.setattr("dashboard.processor_wrapper.threading.Thread.start", fail_start)

    job, thread = launch_processing_thread(
        ["recording.mp4"],
        str(tmp_path / "result"),
        {"schema_version": 4},
    )

    state = job.get_state()
    assert not thread.is_alive()
    assert state["status"] == "error"
    assert "processing worker could not start" in state["error"]
    assert "simulated thread start failure" in state["error"]


def test_failed_rerun_preserves_prior_result_bytes(monkeypatch, tmp_path):
    final_output = tmp_path / "result"
    _write_complete_result(final_output, "prior-complete-run")
    prior_snapshot = _snapshot_files(final_output)

    class FailingProcessor:
        def __init__(self, config):
            pass

        def process(self, video_paths, output_dir, progress_callback):
            (Path(output_dir) / "run_config.json").write_text(
                json.dumps({"partial": True}),
                encoding="utf-8",
            )
            progress_callback(1, 4)
            raise RuntimeError("simulated processing failure")

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        lambda: FailingProcessor,
    )
    job = ProcessingJob()

    run_processing(job, ["recording.mp4"], str(final_output), {"schema_version": 4})

    state = job.get_state()
    assert state["status"] == "error"
    assert "simulated processing failure" in state["error"]
    assert _snapshot_files(final_output) == prior_snapshot
    assert _managed_siblings(final_output, "archive") == []
    assert _managed_siblings(final_output, "staging") == []


def test_successful_reruns_archive_and_promote_deterministically(monkeypatch, tmp_path):
    final_output = tmp_path / "result"
    _write_complete_result(final_output, "original")
    original_snapshot = _snapshot_files(final_output)
    markers = iter(("replacement-one", "replacement-two"))

    class SuccessfulProcessor:
        def __init__(self, config):
            self.marker = next(markers)

        def process(self, video_paths, output_dir, progress_callback):
            _write_processor_artifacts(output_dir, self.marker)
            progress_callback(3, 3, "pose_inference", "Pose inference complete")
            return output_dir

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        lambda: SuccessfulProcessor,
    )

    first_job = ProcessingJob()
    run_processing(first_job, ["recording.mp4"], str(final_output), {"schema_version": 4})
    first_snapshot = _snapshot_files(final_output)

    first_state = first_job.get_state()
    assert first_state["status"] == "completed", first_state["error"]
    assert _snapshot_files(tmp_path / ".result.archive-0001") == original_snapshot
    assert (
        json.loads((final_output / "video_summary.json").read_text(encoding="utf-8"))["marker"]
        == "replacement-one"
    )

    second_job = ProcessingJob()
    run_processing(second_job, ["recording.mp4"], str(final_output), {"schema_version": 4})

    second_state = second_job.get_state()
    assert second_state["status"] == "completed", second_state["error"]
    assert _snapshot_files(tmp_path / ".result.archive-0002") == first_snapshot
    assert (
        json.loads((final_output / "video_summary.json").read_text(encoding="utf-8"))["marker"]
        == "replacement-two"
    )
    assert _managed_siblings(final_output, "archive") == [
        tmp_path / ".result.archive-0001",
        tmp_path / ".result.archive-0002",
    ]
    assert _managed_siblings(final_output, "staging") == []


def test_missing_required_artifact_never_promotes(monkeypatch, tmp_path):
    final_output = tmp_path / "result"
    _write_complete_result(final_output, "prior-complete-run")
    prior_snapshot = _snapshot_files(final_output)

    class IncompleteProcessor:
        def __init__(self, config):
            pass

        def process(self, video_paths, output_dir, progress_callback):
            _write_processor_artifacts(output_dir, "incomplete", omit={"video_summary.json"})
            progress_callback(1, 1)
            return output_dir

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        lambda: IncompleteProcessor,
    )
    job = ProcessingJob()

    run_processing(job, ["recording.mp4"], str(final_output), {"schema_version": 4})

    state = job.get_state()
    assert state["status"] == "error"
    assert "video_summary.json" in state["error"]
    assert _snapshot_files(final_output) == prior_snapshot
    assert _managed_siblings(final_output, "archive") == []
    assert _managed_siblings(final_output, "staging") == []


def test_concurrent_output_lock_fails_clearly(monkeypatch, tmp_path):
    final_output = tmp_path / "result"
    processor_called = False

    class ProcessorThatMustNotRun:
        def __init__(self, config):
            nonlocal processor_called
            processor_called = True

    monkeypatch.setattr(
        "dashboard.processor_wrapper._load_video_processor_class",
        lambda: ProcessorThatMustNotRun,
    )
    job = ProcessingJob()

    with _OutputProcessLock(final_output):
        run_processing(job, ["recording.mp4"], str(final_output), {"schema_version": 4})

    state = job.get_state()
    assert state["status"] == "error"
    assert "already running for this output" in state["error"].lower()
    assert processor_called is False
    assert not final_output.exists()
    assert _managed_siblings(final_output, "staging") == []
    assert not (tmp_path / ".result.processing.lock").exists()
