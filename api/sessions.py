"""Resolving a session identifier to a result bundle on disk.

A session is one participant's one trial at one result sampling. The browser
needs to name that in a URL, and the name has to survive a page reload and a
copied link, so it is derived from the four things that identify it rather than
handed out by a server-side registry that would not outlive a restart.

Nothing here reads frame data. It answers "which directory" and "does it exist",
and the routers layer the analysis on top.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from api.assessment_trash import filter_catalog
from dashboard.cache import ttl_cache
from dashboard.data_loader import (
    detect_current_results,
    get_processed_fps,
    participant_catalog,
)
from pipeline.recording_sources import recorded_path, summary_recordings, trial_manifest

#: ``study__participant__task__sampling``. Study and task folder names are
#: filesystem-safe already; a double underscore separates them because a single
#: one occurs inside folder names like ``task1_niosh_lifting``.
SEPARATOR = "__"
NATIVE = "native"


class SessionError(ValueError):
    """Raised when a session identifier does not resolve to a usable result."""


@dataclass(frozen=True, slots=True)
class SessionRef:
    study_folder: str
    participant_id: str
    task_folder: str
    sampling: int | None

    @property
    def sampling_token(self) -> str:
        return NATIVE if self.sampling is None else str(self.sampling)

    @property
    def session_id(self) -> str:
        return SEPARATOR.join(
            (self.study_folder, self.participant_id, self.task_folder, self.sampling_token)
        )

    @property
    def sampling_label(self) -> str:
        return "Native frame rate" if self.sampling is None else f"{self.sampling} fps"


def parse_session_id(session_id: str) -> SessionRef:
    parts = str(session_id).split(SEPARATOR)
    if len(parts) != 4:
        raise SessionError(
            f"A session id must be study__participant__task__sampling; received {session_id!r}"
        )
    study, participant, task, sampling = parts
    if not all((study, participant, task, sampling)):
        raise SessionError(f"Session id has an empty segment: {session_id!r}")
    if sampling == NATIVE:
        rate: int | None = None
    else:
        try:
            rate = int(sampling)
        except ValueError as exc:
            raise SessionError(f"Unknown result sampling {sampling!r}") from exc
    return SessionRef(study, participant, task, rate)


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@ttl_cache(ttl=20, copy_result=False)
def resolve_result_dir(session_id: str) -> str:
    """Return the published result directory for a session, or raise."""

    ref = parse_session_id(session_id)
    available = list(get_processed_fps(ref.study_folder, ref.participant_id, ref.task_folder))
    if ref.sampling not in available:
        offered = ", ".join(NATIVE if item is None else str(item) for item in available) or "none"
        raise SessionError(
            f"No analysis at {ref.sampling_label} for this trial. Available: {offered}"
        )
    result = detect_current_results(
        ref.study_folder, ref.participant_id, ref.task_folder, ref.sampling
    )
    if not result or not result.get("dir"):
        raise SessionError("The analysis for this trial is incomplete or not traceable")
    return str(result["dir"])


@ttl_cache(ttl=20)
def load_summary(session_id: str) -> dict[str, Any]:
    path = Path(resolve_result_dir(session_id)) / "video_summary.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"Could not read the analysis summary: {exc}") from exc
    if not isinstance(value, dict):
        raise SessionError("The analysis summary is not a JSON object")
    return value


@ttl_cache(ttl=20)
def load_run_config(session_id: str) -> dict[str, Any]:
    path = Path(resolve_result_dir(session_id)) / "run_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def catalog(studies_root: str) -> list[dict[str, Any]]:
    # Discovery is cached by participant_catalog; apply current Trash membership on every lookup.
    return filter_catalog(studies_root, list(participant_catalog(studies_root)))


def find_context(studies_root: str, participant_key: str) -> dict[str, Any]:
    entry = next(
        (item for item in catalog(studies_root) if str(item.get("key")) == participant_key),
        None,
    )
    if entry is None:
        raise SessionError(f"No participant matches {participant_key!r}")
    return entry


def find_trial(context: dict[str, Any], task_folder: str) -> dict[str, Any]:
    trial = next(
        (item for item in context.get("tasks", []) if str(item.get("folder_name")) == task_folder),
        None,
    )
    if trial is None:
        raise SessionError(f"No trial {task_folder!r} for this participant")
    return trial


def saved_session_recordings(
    session_id: str, config: dict[str, Any] | None = None
) -> list[dict[str, Any]] | None:
    if config is None:
        config = load_run_config(session_id)
    recordings = config.get("assessment_workflow", {}).get("recordings")
    if recordings is None:
        recordings = summary_recordings(load_summary(session_id))
    if recordings is not None:
        return recordings
    # Some earlier dashboard bundles kept the original ordering in run metadata.
    try:
        metadata = json.loads(
            (Path(resolve_result_dir(session_id)) / "run_metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    paths = metadata.get("video_paths")
    if isinstance(paths, list) and paths and all(isinstance(p, str) for p in paths):
        return [{"file": PureWindowsPath(path).name} for path in paths]
    return None


def session_videos(studies_root: str, ref: SessionRef) -> list[dict[str, Any]]:
    """Return the exact playable source at each original camera index."""
    context = find_context(studies_root, f"{ref.study_folder}::{ref.participant_id}")
    trial = find_trial(context, ref.task_folder)
    task = Path(trial["path"])
    recordings = saved_session_recordings(ref.session_id)
    if recordings is None:
        recordings = trial_manifest(task) or []
    available = []
    for index, entry in enumerate(recordings):
        try:
            path = recorded_path(task, entry)
        except (ValueError, OSError):
            # Keep camera indexes stable even when a source has disappeared or changed.
            continue
        role = "camera" if len(recordings) == 1 else entry.get("role")
        available.append(
            {"camera_index": index, "path": str(path), "name": path.name, "role": role}
        )
    return available


__all__ = [
    "NATIVE",
    "SEPARATOR",
    "SessionError",
    "SessionRef",
    "catalog",
    "finite",
    "find_context",
    "find_trial",
    "load_run_config",
    "load_summary",
    "parse_session_id",
    "resolve_result_dir",
    "session_videos",
]
