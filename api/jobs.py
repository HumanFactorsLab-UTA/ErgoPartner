"""Background analysis jobs for the browser dashboard.

This module is deliberately independent of FastAPI routing.  It resolves every
participant, task, recording, configuration, and output path on the server, then
hands the work to the transactional dashboard processor.  The HTTP layer only
needs to translate :class:`AnalysisJobError` and return the dictionaries exposed
here.

The shipped desktop application runs one Uvicorn process.  Job state therefore
lives in memory, while result integrity remains cross-process safe because every
trial still uses ``dashboard.processor_wrapper.run_processing`` and its output
lock/staging/promotion transaction.
"""

from __future__ import annotations

import copy
import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from api.assessment_trash import filter_catalog
from api.sessions import SessionError, SessionRef, load_run_config, saved_session_recordings
from dashboard.cache import clear_all_caches
from dashboard.data_loader import (
    detect_current_results,
    find_videos,
    get_processed_fps,
    get_task_results_path,
    participant_catalog,
)
from dashboard.processor_wrapper import build_processing_config, launch_batch_thread
from paths import CONFIG_PATH, STUDIES_DIR_STR, save_studies_root
from pipeline.recording_sources import bind_recordings, recording_hash

DEFAULT_SAMPLING_FPS = 10
SUPPORTED_SCOPES = frozenset({"trial", "participant", "study", "all"})
ACTIVE_STATES = frozenset({"pending", "running"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class AnalysisJobError(RuntimeError):
    """A validation or conflict error suitable for translation by an API route."""

    def __init__(self, status_code: int, detail: str, **context: Any):
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)
        self.context = context

    @property
    def payload(self) -> dict[str, Any]:
        return {"detail": self.detail, **self.context}


@dataclass(frozen=True, slots=True)
class _Target:
    label: str
    participant_key: str
    study_folder: str
    participant_id: str
    task_folder: str
    task_name: str
    sampling: int | None
    session_id: str
    videos: tuple[str, ...]
    output_dir: str
    config: dict[str, Any]

    def processor_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "videos": list(self.videos),
            "output_dir": self.output_dir,
            "config": copy.deepcopy(self.config),
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "participantKey": self.participant_key,
            "study": self.study_folder,
            "participantId": self.participant_id,
            "taskFolder": self.task_folder,
            "taskName": self.task_name,
            "sampling": self.sampling,
            "sessionId": self.session_id,
        }


@dataclass(slots=True)
class _JobHandle:
    job_id: str
    scope: str
    sampling: int | None
    rerun: bool
    candidate_count: int
    targets: list[_Target]
    skipped: list[dict[str, Any]]
    batch: Any
    worker_thread: threading.Thread | None
    created_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    cache_cleared: bool = False
    monitor_thread: threading.Thread | None = None


class _CompletedBatch:
    """Batch-shaped state for a request whose every target was skipped."""

    @staticmethod
    def describe() -> dict[str, Any]:
        return {
            "status": "completed",
            "total": 0,
            "completed": 0,
            "failed": 0,
            "current_index": -1,
            "current_label": None,
            "current": None,
            "outcomes": [],
        }


def _merge_mapping(target: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge an assessment override without retaining shared values."""

    for key, value in override.items():
        if key != "camera_roles" and isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _load_override(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise AnalysisJobError(
            422,
            f"Could not read assessment inputs for {path.parent.name}: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise AnalysisJobError(
            422,
            f"Assessment inputs must contain a YAML mapping: {path}",
        )
    return value


def restore_reviewed_setup(config: dict[str, Any], previous: dict[str, Any], task: Path) -> None:
    """Restore the reviewed setup, including choices saved only with the previous run."""
    workflow = previous.get("assessment_workflow", {})
    if not workflow.get("draft_id"):
        return
    for section in ("task_inputs", "rnle", "fusion", "assessment_workflow"):
        if section in previous:
            _merge_mapping(config, {section: previous[section]})
    override = task / "assessment_inputs.yaml"
    if override.is_file() and recording_hash(override) != workflow.get("task_inputs_sha256"):
        _merge_mapping(config, _load_override(override))


def confirmed_conditions(session_id: str) -> dict[str, Any]:
    """Task conditions an ergonomist accepted from the assistant's visual reading.

    Empty when the assistant was never used, when nothing was accepted, or when
    its package is unavailable. Nothing in the scoring path depends on it.
    """

    try:
        from assistant.proposals import entered_conditions

        return dict(entered_conditions(session_id))
    except Exception:  # noqa: BLE001 - an optional layer must never block a run
        return {}


def _apply_confirmed_conditions(config: dict[str, Any], session_id: str) -> None:
    """Fill still-unanswered task inputs from what the operator confirmed.

    These only ever fill a gap. A value written into ``assessment_inputs.yaml``
    was typed deliberately and keeps precedence; an accepted proposal outranks
    the assumption profile that would otherwise have answered, and nothing else.
    That is the same authority order ``ergonomics/condition_resolution.py``
    documents, with the confirmed reading sitting where it belongs in it.
    """

    confirmed = confirmed_conditions(session_id)
    if not confirmed:
        return
    inputs = config.setdefault("task_inputs", {})
    if not isinstance(inputs, dict):
        return
    for name, value in confirmed.items():
        if inputs.get(name) is None:
            inputs[name] = value


def _recorded_height(context: dict[str, Any]) -> float | None:
    info = (context.get("participant") or {}).get("info") or {}
    value = info.get("height_cm")
    if value is None or value == "" or isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str):
        return None
    try:
        height = float(value)
    except (TypeError, ValueError):
        return None
    return height if math.isfinite(height) and height > 0 else None


class AnalysisJobRegistry:
    """Resolve, launch, and report one top-level analysis job at a time."""

    def __init__(
        self,
        *,
        studies_root: str | Path | None = None,
        config_path: str | Path = CONFIG_PATH,
        max_terminal_jobs: int = 50,
    ):
        self.studies_root = str(Path(studies_root if studies_root is not None else STUDIES_DIR_STR))
        self.config_path = Path(config_path)
        self.max_terminal_jobs = max(1, int(max_terminal_jobs))
        self._lock = threading.RLock()
        self._jobs: dict[str, _JobHandle] = {}

    def set_studies_root(self, root: Path) -> None:
        """Switch sources only between jobs, saving before changing runtime state."""
        with self._lock:
            self._refresh_all_locked()
            if self._active_handles_locked():
                raise AnalysisJobError(409, "Wait for the current analysis to finish before changing the studies folder.")
            save_studies_root(root)
            self.studies_root = str(root)
            clear_all_caches()

    def start(
        self,
        *,
        scope: str,
        participant_key: str | None = None,
        study_folder: str | None = None,
        task_folder: str | None = None,
        sampling: int | None = DEFAULT_SAMPLING_FPS,
        rerun: bool = False,
        assessment_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a trial, participant, study, or whole-folder analysis.

        ``rerun=False`` is deliberately missing-only.  A trial request reports a
        conflict with its existing session, while wider scopes skip complete
        results and process only the remaining recordings.
        """

        normalized_scope = str(scope).strip().lower()
        self._validate_request(
            normalized_scope,
            participant_key=participant_key,
            study_folder=study_folder,
            task_folder=task_folder,
            sampling=sampling,
            rerun=rerun,
        )

        with self._lock:
            self._refresh_all_locked()
            active = self._active_handles_locked()
            if active:
                current = active[0]
                raise AnalysisJobError(
                    409,
                    "Another analysis job is already running.",
                    activeJobId=current.job_id,
                )

            contexts = self._contexts_for_scope(
                normalized_scope,
                participant_key=participant_key,
                study_folder=study_folder,
            )
            tasks = self._tasks_for_scope(
                normalized_scope,
                contexts,
                task_folder=task_folder,
            )
            targets, skipped = self._prepare_targets(
                tasks,
                sampling=sampling,
                rerun=rerun,
                trial_scope=normalized_scope == "trial",
                assessment_overrides=assessment_overrides,
            )

            job_id = uuid.uuid4().hex
            if targets:
                try:
                    batch, worker = launch_batch_thread(
                        [target.processor_payload() for target in targets]
                    )
                except Exception as exc:
                    raise AnalysisJobError(
                        500,
                        f"Could not start the analysis worker: {type(exc).__name__}: {exc}",
                    ) from exc
                handle = _JobHandle(
                    job_id=job_id,
                    scope=normalized_scope,
                    sampling=sampling,
                    rerun=rerun,
                    candidate_count=len(tasks),
                    targets=targets,
                    skipped=skipped,
                    batch=batch,
                    worker_thread=worker,
                )
            else:
                handle = _JobHandle(
                    job_id=job_id,
                    scope=normalized_scope,
                    sampling=sampling,
                    rerun=rerun,
                    candidate_count=len(tasks),
                    targets=[],
                    skipped=skipped,
                    batch=_CompletedBatch(),
                    worker_thread=None,
                    finished_at=_utc_now(),
                    cache_cleared=True,
                )

            self._jobs[job_id] = handle
            self._prune_locked()
            if handle.worker_thread is not None:
                monitor = threading.Thread(
                    target=self._watch_worker,
                    args=(job_id, handle.worker_thread),
                    name=f"analysis-job-monitor-{job_id[:8]}",
                    daemon=True,
                )
                handle.monitor_thread = monitor
                monitor.start()
            return self._snapshot_locked(handle)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            handle = self._jobs.get(str(job_id))
            if handle is None:
                raise AnalysisJobError(404, f"No analysis job matches {job_id!r}.")
            self._refresh_locked(handle)
            return self._snapshot_locked(handle)

    def active(self) -> dict[str, Any] | None:
        """Return the sole active job, if any."""

        with self._lock:
            self._refresh_all_locked()
            active = self._active_handles_locked()
            return self._snapshot_locked(active[0]) if active else None

    def list_active(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_all_locked()
            return [self._snapshot_locked(handle) for handle in self._active_handles_locked()]

    def working_on(self, participant_key: str, task_folder: str | None) -> bool:
        """Include queued targets in a batch as well as its current task."""
        with self._lock:
            self._refresh_all_locked()
            return any(
                target.participant_key == participant_key
                and (task_folder is None or target.task_folder == task_folder)
                for handle in self._active_handles_locked()
                for target in handle.targets
            )

    def _validate_request(
        self,
        scope: str,
        *,
        participant_key: str | None,
        study_folder: str | None,
        task_folder: str | None,
        sampling: int | None,
        rerun: bool,
    ) -> None:
        if scope not in SUPPORTED_SCOPES:
            offered = ", ".join(sorted(SUPPORTED_SCOPES))
            raise AnalysisJobError(422, f"Analysis scope must be one of: {offered}.")
        if sampling is not None and (
            isinstance(sampling, bool) or not isinstance(sampling, int) or not 5 <= sampling <= 60
        ):
            raise AnalysisJobError(422, "Sampling must be null or an integer from 5 to 60 fps.")
        if not isinstance(rerun, bool):
            raise AnalysisJobError(422, "rerun must be true or false.")
        if scope in {"trial", "participant"} and not participant_key:
            raise AnalysisJobError(422, f"{scope} scope requires participantKey.")
        if scope == "trial" and not task_folder:
            raise AnalysisJobError(422, "trial scope requires taskFolder.")
        if scope == "study" and not study_folder:
            raise AnalysisJobError(422, "study scope requires studyFolder.")

    def _contexts_for_scope(
        self,
        scope: str,
        *,
        participant_key: str | None,
        study_folder: str | None,
    ) -> list[dict[str, Any]]:
        contexts = filter_catalog(self.studies_root, list(participant_catalog(self.studies_root)))
        if scope in {"trial", "participant"}:
            matched = [
                context for context in contexts if str(context.get("key")) == str(participant_key)
            ]
            if not matched:
                raise AnalysisJobError(
                    404,
                    f"No participant matches {participant_key!r}.",
                )
            return matched
        if scope == "study":
            matched = [
                context
                for context in contexts
                if str((context.get("study") or {}).get("folder_name")) == str(study_folder)
            ]
            if not matched:
                raise AnalysisJobError(404, f"No study matches {study_folder!r}.")
            return matched
        if not contexts:
            raise AnalysisJobError(
                404,
                f"No study participants were found under {self.studies_root}.",
            )
        return contexts

    @staticmethod
    def _tasks_for_scope(
        scope: str,
        contexts: list[dict[str, Any]],
        *,
        task_folder: str | None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for context in contexts:
            available = [task for task in context.get("tasks", []) if isinstance(task, dict)]
            if scope == "trial":
                available = [
                    task for task in available if str(task.get("folder_name")) == str(task_folder)
                ]
                if not available:
                    raise AnalysisJobError(
                        404,
                        f"No trial {task_folder!r} exists for this participant.",
                    )
            tasks.extend((context, task) for task in available)
        if not tasks:
            raise AnalysisJobError(409, "The selected scope contains no trials.")
        return tasks

    def _prepare_targets(
        self,
        tasks: list[tuple[dict[str, Any], dict[str, Any]]],
        *,
        sampling: int | None,
        rerun: bool,
        trial_scope: bool,
        assessment_overrides: dict[str, Any] | None = None,
    ) -> tuple[list[_Target], list[dict[str, Any]]]:
        targets: list[_Target] = []
        skipped: list[dict[str, Any]] = []
        for context, task in tasks:
            identity = self._identity(context, task, sampling)
            existing_session = self._existing_session(context, task, sampling)
            if existing_session is not None and not rerun:
                if trial_scope:
                    raise AnalysisJobError(
                        409,
                        "This trial has already been analysed.",
                        existingSessionId=existing_session,
                        **identity,
                    )
                skipped.append(
                    {
                        **identity,
                        "status": "already_analysed",
                        "reason": "A complete result already exists at this sampling.",
                        "sessionId": existing_session,
                    }
                )
                continue

            config = self._config_for(context, task, sampling)
            saved = None
            if rerun and existing_session and assessment_overrides is None:
                previous = load_run_config(existing_session)
                try:
                    saved = saved_session_recordings(existing_session, previous)
                except SessionError as exc:
                    raise AnalysisJobError(
                        409, "Cannot verify the original camera sources.", **identity
                    ) from exc
                if saved is None:
                    raise AnalysisJobError(
                        409,
                        "This archived run does not identify its original camera sources.",
                        **identity,
                    )
                restore_reviewed_setup(config, previous, Path(task["path"]))
            if assessment_overrides is not None:
                _merge_mapping(config, assessment_overrides)
            _apply_confirmed_conditions(config, identity["sessionId"])
            task_path = Path(task["path"])
            try:
                videos = tuple(
                    bind_recordings(task_path, find_videos(str(task_path))["raw"], config, saved)
                )
                if not videos:
                    raise ValueError("This trial has no raw recording to analyse.")
            except (ValueError, OSError) as exc:
                if trial_scope:
                    raise AnalysisJobError(409, str(exc), **identity) from exc
                skipped.append({**identity, "status": "no_recording", "reason": str(exc)})
                continue
            output_dir = get_task_results_path(
                identity["study"],
                identity["participantId"],
                identity["taskFolder"],
                sampling,
            )
            targets.append(
                _Target(
                    label=f"{identity['participantId']} \u00b7 {identity['taskName']}",
                    participant_key=identity["participantKey"],
                    study_folder=identity["study"],
                    participant_id=identity["participantId"],
                    task_folder=identity["taskFolder"],
                    task_name=identity["taskName"],
                    sampling=sampling,
                    session_id=identity["sessionId"],
                    videos=videos,
                    output_dir=output_dir,
                    config=config,
                )
            )
        return targets, skipped

    @staticmethod
    def _identity(
        context: dict[str, Any],
        task: dict[str, Any],
        sampling: int | None,
    ) -> dict[str, Any]:
        study = str((context.get("study") or {}).get("folder_name") or "")
        participant = str(context.get("participant_id") or "")
        participant_key = str(context.get("key") or f"{study}::{participant}")
        task_folder = str(task.get("folder_name") or "")
        task_name = str(task.get("name") or task_folder)
        if not study or not participant or not task_folder:
            raise AnalysisJobError(
                422,
                "The selected catalog entry is missing its study, participant, or task identifier.",
            )
        session_id = SessionRef(study, participant, task_folder, sampling).session_id
        return {
            "participantKey": participant_key,
            "study": study,
            "participantId": participant,
            "taskFolder": task_folder,
            "taskName": task_name,
            "sampling": sampling,
            "sessionId": session_id,
        }

    @staticmethod
    def _existing_session(
        context: dict[str, Any],
        task: dict[str, Any],
        sampling: int | None,
    ) -> str | None:
        identity = AnalysisJobRegistry._identity(context, task, sampling)
        available = get_processed_fps(
            identity["study"],
            identity["participantId"],
            identity["taskFolder"],
        )
        # "Missing" in the browser means no complete analysis exists at any
        # sampling. Prefer the requested sampling when it is present, then reuse
        # any other complete result instead of unexpectedly reprocessing a trial
        # that the library already labels ready to review.
        ordered = ([sampling] if sampling in available else []) + [
            candidate for candidate in available if candidate != sampling
        ]
        for candidate in ordered:
            result = detect_current_results(
                identity["study"],
                identity["participantId"],
                identity["taskFolder"],
                candidate,
            )
            if result and result.get("dir"):
                return SessionRef(
                    identity["study"],
                    identity["participantId"],
                    identity["taskFolder"],
                    candidate,
                ).session_id
        return None

    def _config_for(
        self,
        context: dict[str, Any],
        task: dict[str, Any],
        sampling: int | None,
    ) -> dict[str, Any]:
        if not self.config_path.is_file():
            raise AnalysisJobError(
                500,
                f"The analysis configuration is missing: {self.config_path}",
            )
        try:
            config = build_processing_config(
                str(self.config_path),
                target_fps=sampling,
            )
        except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError) as exc:
            raise AnalysisJobError(
                422,
                f"Could not build the analysis configuration: {exc}",
            ) from exc

        override_path = Path(str(task.get("path"))) / "assessment_inputs.yaml"
        if override_path.is_file():
            _merge_mapping(config, _load_override(override_path))

        # The API sampling selects both the processor behavior and the published
        # result path.  A task override must not silently split those identities.
        config.setdefault("video", {})["target_fps"] = sampling

        height = _recorded_height(context)
        if height is not None:
            config.setdefault("rnle", {})["participant_height_cm"] = height
        return config

    def _watch_worker(self, job_id: str, worker: threading.Thread) -> None:
        worker.join()
        with self._lock:
            handle = self._jobs.get(job_id)
            if handle is not None:
                self._refresh_locked(handle)

    def _refresh_all_locked(self) -> None:
        for handle in list(self._jobs.values()):
            self._refresh_locked(handle)

    def _refresh_locked(self, handle: _JobHandle) -> dict[str, Any]:
        state = handle.batch.describe()
        if state.get("status") not in ACTIVE_STATES:
            if handle.finished_at is None:
                handle.finished_at = _utc_now()
            if not handle.cache_cleared:
                clear_all_caches()
                handle.cache_cleared = True
        return state

    def _active_handles_locked(self) -> list[_JobHandle]:
        active = []
        for handle in self._jobs.values():
            state = handle.batch.describe()
            if state.get("status") in ACTIVE_STATES:
                active.append(handle)
        return sorted(active, key=lambda item: item.created_at)

    def _prune_locked(self) -> None:
        terminal = [
            handle
            for handle in self._jobs.values()
            if handle.batch.describe().get("status") not in ACTIVE_STATES
        ]
        terminal.sort(key=lambda item: item.finished_at or item.created_at, reverse=True)
        for handle in terminal[self.max_terminal_jobs :]:
            self._jobs.pop(handle.job_id, None)

    def _snapshot_locked(self, handle: _JobHandle) -> dict[str, Any]:
        state = self._refresh_locked(handle)
        raw_outcomes = list(state.get("outcomes") or [])
        outcomes: list[dict[str, Any]] = []
        for index, outcome in enumerate(raw_outcomes):
            identity = (
                handle.targets[index].identity_payload()
                if index < len(handle.targets)
                else {"taskName": outcome.get("label")}
            )
            ok = bool(outcome.get("ok"))
            outcomes.append(
                {
                    **identity,
                    "label": str(outcome.get("label") or identity.get("taskName") or "Trial"),
                    "ok": ok,
                    "status": "completed" if ok else "failed",
                    "error": None if ok else outcome.get("error"),
                    "outputDir": outcome.get("output_dir"),
                }
            )

        batch_status = str(state.get("status") or "pending")
        failed = sum(1 for outcome in outcomes if outcome["status"] == "failed")
        if batch_status in ACTIVE_STATES:
            status = batch_status
        elif failed and len(handle.targets) == 1:
            status = "failed"
        elif failed:
            status = "completed_with_errors"
        else:
            status = "completed"

        current_payload = None
        current = state.get("current")
        current_index = int(state.get("current_index", -1))
        if isinstance(current, dict):
            total_frames = int(current.get("total_frames") or 0)
            progress_frame = int(current.get("progress_frame") or 0)
            frame_percent = (
                min(100.0, max(0.0, progress_frame / total_frames * 100.0))
                if total_frames > 0
                else 0.0
            )
            identity = (
                handle.targets[current_index].identity_payload()
                if 0 <= current_index < len(handle.targets)
                else {}
            )
            current_payload = {
                **identity,
                "index": current_index,
                "label": state.get("current_label"),
                "status": current.get("status"),
                "stage": current.get("stage"),
                "message": current.get("message"),
                "progressFrame": progress_frame,
                "totalFrames": total_frames,
                "percent": frame_percent,
                "elapsedSec": float(current.get("elapsed_sec") or 0.0),
                "etaSec": float(current.get("eta_sec") or 0.0),
            }

        completed_count = len(raw_outcomes)
        running_fraction = (
            (current_payload["percent"] / 100.0)
            if current_payload is not None and status in ACTIVE_STATES
            else 0.0
        )
        overall_percent = (
            min(100.0, (completed_count + running_fraction) / len(handle.targets) * 100.0)
            if handle.targets
            else 100.0
        )
        done = status not in ACTIVE_STATES
        successful_session_ids = [
            str(outcome["sessionId"])
            for outcome in outcomes
            if outcome["status"] == "completed" and outcome.get("sessionId")
        ]
        return {
            "jobId": handle.job_id,
            "scope": handle.scope,
            "status": status,
            "done": done,
            "sampling": handle.sampling,
            "samplingLabel": (
                "Native frame rate" if handle.sampling is None else f"{handle.sampling} fps"
            ),
            "rerun": handle.rerun,
            "createdAt": handle.created_at,
            "finishedAt": handle.finished_at,
            "candidateCount": handle.candidate_count,
            "targetCount": len(handle.targets),
            "total": len(handle.targets),
            "completedCount": completed_count,
            "completed": completed_count,
            "failedCount": failed,
            "failed": failed,
            "skippedCount": len(handle.skipped),
            "percent": overall_percent,
            "progressPercent": overall_percent,
            "currentLabel": (
                current_payload.get("label") if current_payload is not None else None
            ),
            "current": current_payload,
            "outcomes": outcomes,
            "skipped": copy.deepcopy(handle.skipped),
            "sessionIds": successful_session_ids,
        }


analysis_jobs = AnalysisJobRegistry()


def start_analysis_job(**kwargs: Any) -> dict[str, Any]:
    return analysis_jobs.start(**kwargs)


def get_analysis_job(job_id: str) -> dict[str, Any]:
    return analysis_jobs.get(job_id)


def get_active_analysis_job() -> dict[str, Any] | None:
    return analysis_jobs.active()


def list_active_analysis_jobs() -> list[dict[str, Any]]:
    return analysis_jobs.list_active()


__all__ = [
    "ACTIVE_STATES",
    "DEFAULT_SAMPLING_FPS",
    "SUPPORTED_SCOPES",
    "AnalysisJobError",
    "AnalysisJobRegistry",
    "analysis_jobs",
    "get_active_analysis_job",
    "get_analysis_job",
    "list_active_analysis_jobs",
    "start_analysis_job",
]
