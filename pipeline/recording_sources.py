"""Resolve the ordered recordings that belong to a saved assessment."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any

VIDEO_EXTENSIONS = frozenset({".avi", ".mov", ".mp4", ".mkv", ".webm"})


@lru_cache(maxsize=256)
def _hash_file(path: Path, size: int, modified: int, changed: int, inode: int) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def recording_hash(path: Path) -> str:
    stat = path.stat()
    return _hash_file(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def recording_entry(path: Path, role: str | None = None) -> dict[str, Any]:
    return {"file": path.name, "role": role, "sha256": recording_hash(path)}


def verify_run_recordings(paths: list[str], config: dict[str, Any]) -> None:
    entries = config.get("assessment_workflow", {}).get("recordings")
    if entries is None:
        return
    if len(entries) != len(paths):
        raise ValueError("The number of source recordings changed after setup.")
    for source, entry in zip(paths, entries, strict=True):
        path = Path(source)
        if path.name != entry["file"] or recording_hash(path) != entry["sha256"]:
            raise ValueError(f"The recorded video {path.name} has changed since setup.")


def recorded_path(task: Path, entry: dict[str, Any]) -> Path:
    """Resolve within this trial even if the original study has moved."""
    name = entry.get("file") if isinstance(entry, dict) else None
    if (
        not isinstance(name, str)
        or PureWindowsPath(name).name != name
        or Path(name).suffix.lower() not in VIDEO_EXTENSIONS
    ):
        raise ValueError("Invalid recording name in the saved source list.")
    raw = (task / "raw").resolve()
    path = (raw / name).resolve()
    if path.parent != raw or not path.is_file():
        raise ValueError(f"The recorded video {name} is unavailable.")
    if entry.get("sha256") and recording_hash(path) != entry["sha256"]:
        raise ValueError(f"The recorded video {name} has changed since it was selected.")
    return path


def trial_manifest(task: Path) -> list[dict[str, Any]] | None:
    manifest = task / "source_manifest.json"
    if not manifest.exists():
        return None
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("The saved trial source manifest is invalid.")
    if "recordings" in value:
        if not isinstance(value["recordings"], list):
            raise ValueError("The saved trial recording list is invalid.")
        return value["recordings"]
    # Older workflow publications copied each draft input to raw/cam_INDEX.EXT.
    if isinstance(value.get("videos"), list) and all(
        isinstance(video, dict) and isinstance(video.get("path"), str) for video in value["videos"]
    ):
        return [
            {"file": f"cam_{i}{Path(video['path']).suffix.lower()}", "role": video.get("role")}
            for i, video in enumerate(value["videos"])
        ]
    raise ValueError("The saved trial source manifest is incomplete.")


def summary_recordings(summary: dict[str, Any]) -> list[dict[str, Any]] | None:
    metadata = summary.get("video_metadata") or []
    if not metadata or not all(isinstance(item, dict) and item.get("path") for item in metadata):
        paths = summary.get("input_source")
        if isinstance(paths, str) and paths.strip():
            return [{"file": PureWindowsPath(path).name} for path in paths.split(" | ")]
        return None
    entries = sorted(enumerate(metadata), key=lambda pair: pair[1].get("camera_index", pair[0]))
    return [
        {"file": PureWindowsPath(item["path"]).name, "role": item.get("camera_role")}
        for _, item in entries
    ]


def bind_recordings(
    task: Path,
    discovered: list[str],
    config: dict[str, Any],
    saved: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Verify exact source identities before starting. Persist them in run_config."""
    workflow = config.setdefault("assessment_workflow", {})
    if saved is None:
        saved = workflow.get("recordings")
    if saved is None:
        saved = trial_manifest(task)
    if saved is not None:
        if not isinstance(saved, list):
            raise ValueError("The saved recording list is invalid.")
        paths = [recorded_path(task, entry) for entry in saved]
    else:
        paths = [Path(path) for path in discovered]
    if not paths:
        return []
    if len(paths) > 3 or len(set(paths)) != len(paths):
        raise ValueError("Choose one to three distinct recordings from the same take.")
    mode = "single" if len(paths) == 1 else "multi"
    if workflow.get("camera_mode", mode) != mode:
        raise ValueError("The selected camera mode differs from the saved recording count.")
    workflow["camera_mode"] = mode
    fusion = config.setdefault("fusion", {})
    by_index = {index: role for role, index in fusion.get("camera_roles", {}).items()}
    if mode == "single" and len(by_index) != 1:
        by_index = {0: "camera"}
    roles = [
        (saved[i].get("role") if saved else None) or by_index.get(i) for i in range(len(paths))
    ]
    fusion["camera_roles"] = {role: i for i, role in enumerate(roles) if role}
    workflow["recordings"] = [
        recording_entry(path, role) for path, role in zip(paths, roles, strict=True)
    ]
    inputs_path = task / "assessment_inputs.yaml"
    if workflow.get("draft_id") and inputs_path.is_file():
        workflow["task_inputs_sha256"] = recording_hash(inputs_path)
    return [str(path) for path in paths]
