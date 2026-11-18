"""
Processor Wrapper - Thread-safe wrapper for running VideoProcessor from the dashboard.
"""

import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import yaml

# Add project root to path using paths module (centralized)
from paths import PROJECT_ROOT_STR
from pipeline.recording_sources import verify_run_recordings

_REQUIRED_RESULT_FILES = frozenset(
    {
        "ASSESSMENT_REPORT.md",
        "frame_results.csv",
        "frame_results.jsonl",
        "run_config.json",
        "run_metadata.json",
        "video_summary.json",
    }
)
_JSON_RESULT_FILES = frozenset(
    {
        "run_config.json",
        "run_metadata.json",
        "video_summary.json",
    }
)


class ProcessingOutputError(RuntimeError):
    """Raised when an output cannot be safely validated or promoted."""


def _path_exists(path: Path) -> bool:
    """Return True for normal paths and broken symbolic links."""
    return os.path.lexists(path)


def _resolve_final_output(output_dir: str) -> Path:
    """Resolve the output parent while preserving the final path entry."""
    raw_path = Path(output_dir).expanduser()
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    if not raw_path.name:
        raise ProcessingOutputError(
            "The analysis output must be a named directory, not a root path"
        )
    return raw_path.parent.resolve(strict=False) / raw_path.name


def _staging_prefix(final_output: Path) -> str:
    return f".{final_output.name}.staging-"


def _is_verified_staging_path(staging: Path, final_output: Path) -> bool:
    """Only recognize UUID-named staging siblings created by this wrapper."""
    staging_parent = staging.parent.resolve(strict=False)
    final_parent = final_output.parent.resolve(strict=False)
    prefix = _staging_prefix(final_output)
    if staging_parent != final_parent or not staging.name.startswith(prefix):
        return False
    suffix = staging.name[len(prefix) :]
    return len(suffix) == 32 and all(character in "0123456789abcdef" for character in suffix)


def _create_staging_directory(final_output: Path) -> Path:
    staging = final_output.parent / f"{_staging_prefix(final_output)}{uuid.uuid4().hex}"
    if not _is_verified_staging_path(staging, final_output):
        raise ProcessingOutputError(f"Refusing unsafe staging path: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def _cleanup_staging_directory(staging: Path, final_output: Path) -> None:
    """Remove only a verified, non-symlink staging sibling."""
    if not _is_verified_staging_path(staging, final_output):
        raise ProcessingOutputError(f"Refusing to clean an unverified staging path: {staging}")
    if not _path_exists(staging):
        return
    if staging.is_symlink() or not staging.is_dir():
        raise ProcessingOutputError(f"Refusing to clean a non-directory staging path: {staging}")
    shutil.rmtree(staging)


def _validate_complete_result(result_dir: Path) -> Path:
    """Validate the minimum deterministic artifact contract for a complete run."""
    result_dir = Path(result_dir)
    if result_dir.is_symlink() or not result_dir.is_dir():
        raise ProcessingOutputError(f"Analysis result is not a regular directory: {result_dir}")

    resolved_root = result_dir.resolve(strict=True)
    invalid_artifacts: list[str] = []
    for filename in sorted(_REQUIRED_RESULT_FILES):
        artifact = result_dir / filename
        if artifact.is_symlink() or not artifact.is_file():
            invalid_artifacts.append(filename)
            continue
        try:
            if artifact.resolve(strict=True).parent != resolved_root:
                invalid_artifacts.append(filename)
        except OSError:
            invalid_artifacts.append(filename)

    if invalid_artifacts:
        invalid = ", ".join(invalid_artifacts)
        raise ProcessingOutputError(
            f"Analysis result is incomplete; required artifacts are missing or invalid: {invalid}"
        )

    for filename in sorted(_JSON_RESULT_FILES):
        artifact = result_dir / filename
        try:
            with artifact.open(encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProcessingOutputError(f"Invalid JSON artifact {filename}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProcessingOutputError(f"JSON artifact {filename} must contain an object")

    return resolved_root


def _next_archive_path(final_output: Path) -> Path:
    for archive_number in range(1, 10_000):
        archive = final_output.parent / f".{final_output.name}.archive-{archive_number:04d}"
        if not _path_exists(archive):
            return archive
    raise ProcessingOutputError(f"No archive slot is available for {final_output}")


def _promote_staging_directory(staging: Path, final_output: Path) -> Path:
    """Promote a validated sibling directory, rolling back the prior result on failure."""
    if not _is_verified_staging_path(staging, final_output):
        raise ProcessingOutputError(f"Refusing to promote an unverified staging path: {staging}")
    _validate_complete_result(staging)

    archive: Path | None = None
    if _path_exists(final_output):
        _validate_complete_result(final_output)
        archive = _next_archive_path(final_output)
        final_output.replace(archive)

    try:
        staging.replace(final_output)
    except Exception as promotion_error:
        if archive is not None and not _path_exists(final_output):
            try:
                archive.replace(final_output)
            except Exception as rollback_error:
                raise ProcessingOutputError(
                    "Output promotion and prior-result rollback both failed: "
                    f"promotion={promotion_error}; rollback={rollback_error}"
                ) from promotion_error
        raise ProcessingOutputError(
            f"Could not promote completed analysis: {promotion_error}"
        ) from promotion_error

    return final_output


class _OutputProcessLock:
    """Cross-process, per-output lock implemented with atomic file creation."""

    def __init__(self, final_output: Path):
        self.final_output = _resolve_final_output(str(final_output))
        self.path = self.final_output.parent / f".{self.final_output.name}.processing.lock"
        self.token = uuid.uuid4().hex
        self._acquired = False

    def __enter__(self):
        self.final_output.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as exc:
            raise ProcessingOutputError(
                f"Another analysis is already running for this output: {self.final_output}"
            ) from exc

        try:
            payload = json.dumps(
                {
                    "token": self.token,
                    "pid": os.getpid(),
                    "output_dir": str(self.final_output),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                sort_keys=True,
            ).encode("utf-8")
            bytes_written = os.write(descriptor, payload)
            if bytes_written != len(payload):
                raise OSError("Processing lock write was incomplete")
            os.fsync(descriptor)
            self._acquired = True
        except Exception:
            os.close(descriptor)
            self.path.unlink(missing_ok=True)
            raise
        else:
            os.close(descriptor)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if not self._acquired:
            return False
        try:
            try:
                lock_payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as lock_error:
                raise ProcessingOutputError(
                    f"Could not verify processing lock ownership: {lock_error}"
                ) from lock_error
            if lock_payload.get("token") != self.token:
                raise ProcessingOutputError("Processing lock ownership changed unexpectedly")
            self.path.unlink()
        finally:
            self._acquired = False
        return False


def _load_video_processor_class():
    """Load the heavyweight video pipeline inside the guarded worker lifecycle."""
    if PROJECT_ROOT_STR not in sys.path:
        sys.path.insert(0, PROJECT_ROOT_STR)

    from pipeline.video_processor import VideoProcessor

    return VideoProcessor


class ProcessingJob:
    """Thread-safe container for tracking processing progress and results."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "pending"  # pending, running, completed, error
        self._progress_frame = 0
        self._total_frames = 0
        self._stage = "queued"
        self._message = "Waiting to start."
        self._last_update_time = time.time()
        self._error = None
        self._output_dir = None
        self._start_time = None
        self._end_time = None

    def get_state(self) -> dict:
        with self._lock:
            elapsed = 0
            if self._start_time:
                end = self._end_time or time.time()
                elapsed = end - self._start_time
            fps = self._progress_frame / elapsed if elapsed > 0 else 0
            eta = 0
            if fps > 0 and self._total_frames > self._progress_frame:
                eta = (self._total_frames - self._progress_frame) / fps
            return {
                "status": self._status,
                "progress_frame": self._progress_frame,
                "total_frames": self._total_frames,
                "stage": self._stage,
                "message": self._message,
                "last_update_age_sec": max(0.0, time.time() - self._last_update_time),
                "error": self._error,
                "output_dir": self._output_dir,
                "elapsed_sec": elapsed,
                "fps": fps,
                "eta_sec": eta,
            }

    def _update_progress(
        self,
        frame: int,
        total: int,
        stage: str = "processing",
        message: str | None = None,
    ):
        with self._lock:
            self._progress_frame = frame
            self._total_frames = total
            self._stage = stage
            self._message = message or stage.replace("_", " ").title()
            self._last_update_time = time.time()

    def _set_running(self):
        with self._lock:
            self._status = "running"
            self._stage = "validating"
            self._message = "Validating configuration and preparing output."
            self._start_time = time.time()
            self._last_update_time = self._start_time

    def _set_completed(self, output_dir: str):
        with self._lock:
            self._status = "completed"
            self._stage = "completed"
            self._message = "Analysis completed."
            if self._total_frames > 0:
                self._progress_frame = self._total_frames
            self._output_dir = output_dir
            self._end_time = time.time()
            self._last_update_time = self._end_time

    def _set_error(self, error: str):
        with self._lock:
            self._status = "error"
            self._stage = "error"
            self._message = "Analysis failed."
            self._error = error
            self._end_time = time.time()
            self._last_update_time = self._end_time


def build_processing_config(
    base_config_path: str,
    side: str = "right",
    load_kg: float | None = None,
    coupling: str | None = None,
    niosh_params: dict | None = None,
    enable_llm: bool = False,
    enable_vlm: bool = False,
    reba_enabled: bool = True,
    rula_enabled: bool = True,
    niosh_enabled: bool = True,
    use_gpu: bool = False,
    target_fps: int | None = None,  # NEW PARAMETER FOR FPS SELECTION
    start_frame: int | None = None,
    end_frame: int | None = None,
    task_inputs: dict | None = None,
    camera_view_status: str | None = None,
    calibration_status: str | None = None,
    mirrored: bool | None = None,
    camera_roles: dict[str, int] | None = None,
) -> dict:
    """
    Build processing config by merging base config.yaml with UI overrides.
    """
    config = {}
    if os.path.isfile(base_config_path):
        with open(base_config_path) as f:
            config = yaml.safe_load(f) or {}

    # v4 always calculates both anatomical sides. ``side`` remains in this
    # wrapper signature only so old dashboard calls fail safely rather than
    # changing scientific behavior.
    merged_task_inputs = config.setdefault("task_inputs", {})
    if task_inputs:
        merged_task_inputs.update(task_inputs)
    if load_kg is not None:
        merged_task_inputs["load_kg"] = load_kg
    if coupling is not None:
        merged_task_inputs["coupling"] = coupling

    if niosh_params:
        niosh_cfg = config.setdefault("rnle", {})
        key_map = {
            "H": "horizontal_cm",
            "V": "vertical_cm",
            "D": "travel_distance_cm",
            "A": "asymmetry_deg",
            "F": "frequency_lifts_per_min",
            "duration": "duration",
            "coupling": "coupling",
            "load_weight_kg": "load_weight_kg",
            "applicability": "applicability",
        }
        for old_key, new_key in key_map.items():
            if old_key in niosh_params and niosh_params[old_key] is not None:
                niosh_cfg[new_key] = niosh_params[old_key]

    # LLM/VLM output is not part of the validated scoring pipeline.
    config.setdefault("pose", {})["use_gpu"] = use_gpu

    # Set target FPS if specified (overrides config.yaml)
    video_cfg = config.setdefault("video", {})
    video_cfg["target_fps"] = int(target_fps) if target_fps is not None and target_fps > 0 else None
    video_cfg["start_frame"] = start_frame
    video_cfg["end_frame"] = end_frame
    video_cfg["mirrored"] = mirrored
    if camera_view_status is not None:
        config.setdefault("quality", {})["camera_view_status"] = camera_view_status
    if calibration_status is not None:
        config.setdefault("calibration", {})["status"] = calibration_status
    if camera_roles is not None:
        if len(set(camera_roles.values())) != len(camera_roles):
            raise ValueError("Each selected recording must have one unique camera role")
        config.setdefault("fusion", {})["camera_roles"] = dict(camera_roles)

    return config


def run_processing(
    job: ProcessingJob,
    video_paths: list[str],
    output_dir: str,
    config: dict,
):
    """Run an analysis transaction and expose progress through ``job``."""
    job._set_running()
    staging: Path | None = None

    try:
        final_output = _resolve_final_output(output_dir)
        final_output.parent.mkdir(parents=True, exist_ok=True)

        with _OutputProcessLock(final_output):
            try:
                verify_run_recordings(video_paths, config)
                if _path_exists(final_output):
                    _validate_complete_result(final_output)

                staging = _create_staging_directory(final_output)
                VideoProcessor = _load_video_processor_class()
                processor = VideoProcessor(config)
                result_dir = processor.process(
                    video_paths,
                    str(staging),
                    progress_callback=job._update_progress,
                )

                if result_dir is None:
                    raise ProcessingOutputError(
                        "Video processor did not report an output directory"
                    )
                reported_output = _resolve_final_output(str(result_dir))
                if reported_output != staging:
                    raise ProcessingOutputError(
                        "Video processor redirected output outside its assigned staging directory: "
                        f"expected {staging}, received {reported_output}"
                    )

                state = job.get_state()
                job._update_progress(
                    state["progress_frame"],
                    state["total_frames"],
                    stage="validating_output",
                    message="Validating completed analysis before publication.",
                )
                metadata = {
                    "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "video_paths": video_paths,
                    "output_dir": str(final_output),
                    "duration_sec": state["elapsed_sec"],
                    "total_frames": state["total_frames"],
                }
                metadata_path = staging / "run_metadata.json"
                if _path_exists(metadata_path) and (
                    metadata_path.is_symlink() or not metadata_path.is_file()
                ):
                    raise ProcessingOutputError(
                        f"Cannot safely create run metadata at {metadata_path}"
                    )
                with metadata_path.open("w", encoding="utf-8") as file_handle:
                    json.dump(metadata, file_handle, indent=2)
                    file_handle.write("\n")

                _validate_complete_result(staging)
                verify_run_recordings(video_paths, config)
                promoted_output = _promote_staging_directory(staging, final_output)
                staging = None
            except Exception as processing_error:
                if staging is not None and _path_exists(staging):
                    try:
                        _cleanup_staging_directory(staging, final_output)
                    except Exception as cleanup_error:
                        raise ProcessingOutputError(
                            "Analysis failed and its verified staging directory could not be cleaned: "
                            f"{cleanup_error}"
                        ) from processing_error
                raise

        job._set_completed(str(promoted_output))

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        job._set_error(error_msg)


def launch_processing_thread(
    video_paths: list[str],
    output_dir: str,
    config: dict,
) -> tuple:
    """
    Launch processing in a background thread (non-daemon).

    Uses non-daemon threads to ensure proper cleanup and sentinel file handling.
    Returns (ProcessingJob, Thread).
    """
    job = ProcessingJob()
    thread = threading.Thread(
        target=run_processing,
        args=(job, video_paths, output_dir, config),
        daemon=False,  # Non-daemon: allows proper cleanup and sentinel handling
    )
    try:
        thread.start()
    except Exception as exc:
        error_msg = (
            f"{type(exc).__name__}: processing worker could not start: {exc}\n"
            f"{traceback.format_exc()}"
        )
        job._set_error(error_msg)
    return job, thread


def reconcile_processing_thread(
    job: ProcessingJob,
    thread: threading.Thread | None,
) -> dict:
    """Convert an orphaned active job into an explicit, retryable error state."""
    state = job.get_state()
    if state["status"] in {"pending", "running"} and (thread is None or not thread.is_alive()):
        job._set_error(
            "Processing worker stopped before reporting completion. The analysis did not "
            "finish; review the runtime error and start it again."
        )
        state = job.get_state()
    return state


class BatchJob:
    """Track a queue of independent trial analyses run by one worker thread.

    Each trial goes through the same run_processing transaction as a single
    dashboard run -- staging, validation, archive and promotion -- so a batch
    cannot produce a partially written result. Trials are sequential; each one
    already parallelises its own per-camera inference.
    """

    def __init__(self, trials: list[dict]):
        self._lock = threading.Lock()
        self._trials = trials
        self._current_index = -1
        self._current_job: ProcessingJob | None = None
        self._outcomes: list[dict] = []
        self._status = "pending"  # pending, running, completed

    def describe(self) -> dict:
        with self._lock:
            current_state = self._current_job.get_state() if self._current_job else None
            return {
                "status": self._status,
                "total": len(self._trials),
                "completed": len(self._outcomes),
                "failed": sum(1 for item in self._outcomes if not item["ok"]),
                "current_index": self._current_index,
                "current_label": (
                    self._trials[self._current_index]["label"]
                    if 0 <= self._current_index < len(self._trials)
                    else None
                ),
                "current": current_state,
                "outcomes": list(self._outcomes),
            }

    def _run(self) -> None:
        with self._lock:
            self._status = "running"
        for index, trial in enumerate(self._trials):
            job = ProcessingJob()
            with self._lock:
                self._current_index = index
                self._current_job = job
            try:
                run_processing(job, trial["videos"], trial["output_dir"], trial["config"])
            except Exception as exc:  # run_processing reports via the job; belt and braces
                job._set_error(f"{type(exc).__name__}: {exc}")
            state = job.get_state()
            with self._lock:
                self._outcomes.append(
                    {
                        "label": trial["label"],
                        "ok": state["status"] == "completed",
                        "error": state["error"],
                        "output_dir": state["output_dir"],
                    }
                )
        with self._lock:
            self._status = "completed"
            self._current_job = None
            self._current_index = -1


def launch_batch_thread(trials: list[dict]) -> tuple[BatchJob, threading.Thread]:
    """Start a background batch. Each trial dict needs label/videos/output_dir/config."""

    batch = BatchJob(trials)
    thread = threading.Thread(target=batch._run, daemon=False)
    thread.start()
    return batch, thread
