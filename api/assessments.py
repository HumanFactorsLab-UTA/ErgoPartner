"""Durable assessment setup and reviewer notes around the existing analysis pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import cv2
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from api import jobs
from api.assessment_trash import draft_hidden, task_hidden
from api.assessment_trash import items as trash_items
from api.sessions import (
    SessionError,
    SessionRef,
    find_context,
    find_trial,
    load_run_config,
    resolve_result_dir,
    saved_session_recordings,
)
from dashboard.cache import clear_all_caches
from dashboard.data_loader import discover_participants, discover_studies, find_videos
from paths import PROJECT_ROOT
from pipeline.recording_sources import recorded_path, recording_hash, trial_manifest

WORKFLOW_ROOT = PROJECT_ROOT / ".posture" / "workflow"
LOCK = threading.RLock()
UPLOADING: set[str] = set()
MAX_VIDEO_BYTES = 10 * 1024**3
CAMERA_ROLES = ("front", "side", "iso")
CameraMode = Literal["single", "multi"]
CameraRole = Literal["camera", "front", "side", "iso"]


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # A short-lived file reader on Windows can hold a handle without delete sharing.
        for attempt in range(6):
            try:
                temporary.replace(path)
                break
            except PermissionError as exc:
                if (
                    os.name != "nt"
                    or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == 5
                ):
                    raise
                time.sleep(0.01 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)


def draft_dir(identifier: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise HTTPException(404, "Assessment draft not found.")
    return WORKFLOW_ROOT / "drafts" / identifier


class Setup(BaseModel):
    visionModel: str | None = Field(default=None, min_length=1, max_length=200)
    cameraMode: CameraMode = "single"
    studyName: str = Field(default="", max_length=160)
    studyFolder: str | None = None
    participantId: str = Field(default="", max_length=64)
    taskName: str = Field(default="", max_length=160)
    heightCm: float | None = Field(default=None, ge=50, le=260, allow_inf_nan=False)
    primaryMethod: Literal["reba", "rula"] = "reba"
    samplingFps: int = Field(default=10, ge=5, le=60)
    loadKg: float | None = Field(default=None, ge=0, le=1000, allow_inf_nan=False)
    coupling: Literal["good", "fair", "poor", "unacceptable"] | None = None
    assumptionProfile: (
        Literal["standing_light_work", "standing_manual_handling", "seated_workstation"] | None
    ) = "standing_light_work"
    shockForce: bool | None = None
    leftArmSupport: bool | None = None
    rightArmSupport: bool | None = None
    leftWristEndRange: bool | None = None
    rightWristEndRange: bool | None = None
    rulaLoadType: Literal["intermittent", "static_or_repeated"] | None = None
    footageReviewed: bool = False
    synchronizationConfirmed: bool = False
    conditionsReviewed: bool = False


class DraftUpdate(BaseModel):
    setup: Setup
    step: int = Field(default=1, ge=1, le=4)
    roles: list[CameraRole] | None = None
    videoIds: list[str] | None = Field(default=None, max_length=3)


class TaskGroup(BaseModel):
    studyFolder: str | None = None
    studyName: str = Field(default="", max_length=160)
    participantId: str | None = Field(default=None, max_length=64)
    taskName: str = Field(min_length=1, max_length=160)


class CreateDraft(BaseModel):
    source: Literal["video", "live", "existing"] = "video"
    cameraMode: CameraMode | None = None
    participantKey: str | None = None
    taskFolder: str | None = None
    group: TaskGroup | None = None


class Preferences(BaseModel):
    visionModel: str | None = Field(default=None, min_length=1, max_length=200)
    primaryMethod: Literal["reba", "rula"] = "reba"
    samplingFps: int = Field(default=10, ge=5, le=60)
    assumptionProfile: (
        Literal["standing_light_work", "standing_manual_handling", "seated_workstation"] | None
    ) = "standing_light_work"
    cameraFps: Literal[24, 30, 60] = 30


def preferences() -> Preferences:
    try:
        return Preferences(
            **json.loads((WORKFLOW_ROOT / "preferences.json").read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, TypeError):
        return Preferences()


def load(identifier: str) -> dict[str, Any]:
    try:
        draft = json.loads((draft_dir(identifier) / "draft.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(404, "Assessment draft not found.") from exc
    if draft_hidden(draft):
        raise HTTPException(404, "This assessment is in Trash. Restore it from Assessments.")
    videos = draft["videos"]
    setup = draft["setup"]
    setup.setdefault("cameraMode", "multi" if len(videos) > 1 else "single")
    draft.setdefault("availableVideos", [dict(v) for v in videos])
    if setup["cameraMode"] == "single":
        for video in videos:
            video["role"] = "camera"
    return draft


def save(draft: dict[str, Any]) -> None:
    draft["updatedAt"] = now()
    atomic_json(draft_dir(draft["id"]) / "draft.json", draft)


def describe_video(path: Path, *, name: str | None = None, role: str = "camera") -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise HTTPException(422, f"Could not decode {name or path.name}.")
        ok, _ = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count < 1:
            raise HTTPException(422, "This video has no readable frames or usable timing.")
        frames = int(count)
        return {
            "id": uuid.uuid4().hex,
            "name": name or path.name,
            "path": str(path),
            "role": role,
            "sha256": recording_hash(path),
            "fps": fps,
            "frames": frames,
            "durationSec": frames / fps,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def validate_participant_id(value: str) -> None:
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)
        or "__" in value
        or value.endswith(".")
    ):
        raise HTTPException(
            422, "Participant ID must use letters, numbers, dots, dashes, or single underscores."
        )


def set_task_group(
    setup: Setup, group: TaskGroup, key: str | None, root: str, identifier: str
) -> Path | None:
    setup.taskName = group.taskName.strip()
    if not setup.taskName:
        raise HTTPException(422, "Enter a task name.")
    if key:
        context = find_context(root, key)
        setup.studyFolder = context["study"]["folder_name"]
        setup.studyName = context["study"]["name"]
        setup.participantId = str(context["participant_id"])
        setup.heightCm = jobs._recorded_height(context)
        return None

    base = Path(root).resolve()
    if not base.is_dir():
        raise HTTPException(422, "Choose an available studies folder.")
    studies = discover_studies(root)
    if group.studyFolder:
        study = next((s for s in studies if s["folder_name"] == group.studyFolder), None)
        if not study:
            raise HTTPException(422, "The selected study is no longer available.")
        setup.studyFolder, setup.studyName = study["folder_name"], study["name"]
    else:
        setup.studyName = group.studyName.strip()
        if not setup.studyName:
            raise HTTPException(422, "Enter a study name.")
        setup.studyFolder = f"study_assessment_{identifier}"

    people = [
        (study["folder_name"], str(person["id"]))
        for study in studies
        for person in discover_participants(study["path"])
    ]
    requested_id = (group.participantId or "").strip()
    if not requested_id:
        numeric = (int(pid) for _, pid in people if pid.isascii() and pid.isdigit())
        requested_id = f"{max(numeric, default=0) + 1:03d}"
    validate_participant_id(requested_id)
    setup.participantId = requested_id
    if any(
        study == setup.studyFolder and pid.casefold() == requested_id.casefold()
        for study, pid in people
    ):
        raise HTTPException(
            409,
            "That participant already exists in this study. Select the existing participant or choose another ID.",
        )
    study_path = base / setup.studyFolder
    person_path = study_path / f"participant_{requested_id}"
    if not person_path.resolve().is_relative_to(base) or person_path.exists():
        raise HTTPException(409, "The participant folder is unavailable. Choose another ID.")
    return person_path


def publish_participant(person: Path, setup: Setup, new_study: bool) -> None:
    destination = person.parent if new_study else person
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".participant-") as temp:
        staged = Path(temp) / destination.name
        staged_person = staged / person.name if new_study else staged
        atomic_json(
            staged_person / "participant_info.json",
            {"participant_id": setup.participantId, "assessment_group": True},
        )
        if new_study:
            atomic_json(
                staged / "study_config.json",
                {"study_name": setup.studyName, "created_at": now(), "tasks": []},
            )
        staged.rename(destination)
    clear_all_caches()


def create(payload: CreateDraft, root: str) -> dict[str, Any]:
    identifier = uuid.uuid4().hex
    defaults = preferences()
    setup = Setup(
        cameraMode=payload.cameraMode or "single",
        **defaults.model_dump(exclude={"cameraFps", "visionModel"}),
    )
    new_participant = None
    if payload.group:
        if payload.source == "existing":
            raise HTTPException(
                422, "Choose the existing participant and recording for this source."
            )
        new_participant = set_task_group(
            setup, payload.group, payload.participantKey, root, identifier
        )
    draft: dict[str, Any] = {
        "id": identifier,
        "source": payload.source,
        "studiesRoot": root,
        "createdAt": now(),
        "updatedAt": now(),
        "step": 1,
        "setup": setup.model_dump(),
        "videos": [],
        "availableVideos": [],
        "takes": [],
        "selectedTake": None,
        "analysis": None,
        "published": None,
    }
    if payload.group:
        draft["participantKey"] = f"{setup.studyFolder}::{setup.participantId}"
    if payload.source == "existing":
        if not payload.participantKey or not payload.taskFolder:
            raise HTTPException(422, "Choose a participant and recording.")
        context = find_context(root, payload.participantKey)
        trial = find_trial(context, payload.taskFolder)
        task = Path(trial["path"])
        try:
            previous_id = jobs.analysis_jobs._existing_session(context, trial, setup.samplingFps)
            recordings = saved_session_recordings(previous_id) if previous_id else None
            if recordings is None:
                recordings = trial_manifest(task)
            videos = (
                [str(recorded_path(task, entry)) for entry in recordings]
                if recordings is not None
                else find_videos(str(task))["raw"]
            )
        except (OSError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if not videos:
            raise HTTPException(422, "This trial has no available recordings.")
        config = jobs.analysis_jobs._config_for(context, trial, setup.samplingFps)
        if previous_id:
            jobs.restore_reviewed_setup(config, load_run_config(previous_id), task)
        inputs = config.get("task_inputs", {})
        info = context.get("participant", {}).get("info", {})
        height: float | None
        try:
            entered_height = config.get("rnle", {}).get("participant_height_cm")
            raw_height = float(
                entered_height if entered_height is not None else info.get("height_cm")
            )
            height = raw_height if math.isfinite(raw_height) and 50 <= raw_height <= 260 else None
        except (TypeError, ValueError):
            height = None
        primary_method = defaults.primaryMethod
        if trial.get("primary_tool") == "rula":
            primary_method = "rula"
        elif trial.get("primary_tool") == "reba":
            primary_method = "reba"
        setup = Setup(
            cameraMode=payload.cameraMode or ("multi" if len(videos) > 1 else "single"),
            studyName=context["study"]["name"],
            visionModel=config.get("assessment_workflow", {}).get("vision_model_override"),
            studyFolder=context["study"]["folder_name"],
            participantId=context["participant_id"],
            taskName=trial["name"],
            heightCm=height,
            primaryMethod=primary_method,
            loadKg=inputs.get("load_kg"),
            coupling=inputs.get("coupling"),
            assumptionProfile=inputs.get("assumption_profile"),
            shockForce=inputs.get("shock_or_rapid_force"),
            leftArmSupport=inputs.get("left_arm_supported_or_leaning"),
            rightArmSupport=inputs.get("right_arm_supported_or_leaning"),
            leftWristEndRange=inputs.get("left_wrist_twist_at_end_range"),
            rightWristEndRange=inputs.get("right_wrist_twist_at_end_range"),
            rulaLoadType=inputs.get("rula_load_type"),
        )
        configured_roles = config.get("fusion", {}).get("camera_roles", {})
        roles = {index: role for role, index in configured_roles.items()}
        if recordings:
            roles.update({i: e["role"] for i, e in enumerate(recordings) if e.get("role")})
        draft["availableVideos"] = [
            describe_video(Path(video), role=roles.get(i, CAMERA_ROLES[i] if i < 3 else "camera"))
            for i, video in enumerate(videos)
        ]
        if len(videos) == 1 or (setup.cameraMode == "multi" and len(videos) <= 3):
            draft["videos"] = assign_roles(draft["availableVideos"], setup.cameraMode)
        draft["existing"] = {
            "participantKey": payload.participantKey,
            "taskFolder": payload.taskFolder,
        }
        draft["setup"] = setup.model_dump()
    save(draft)
    if new_participant is not None and payload.group:
        try:
            publish_participant(new_participant, setup, new_study=not payload.group.studyFolder)
        except OSError as exc:
            (draft_dir(identifier) / "draft.json").unlink(missing_ok=True)
            raise HTTPException(
                409, "Could not create the participant. Check the study folder and try again."
            ) from exc
    return describe(draft)


def describe(draft: dict[str, Any]) -> dict[str, Any]:
    value = dict(draft)
    value["grouped"] = bool(draft.get("participantKey"))
    value["status"] = "draft"
    value["job"] = None
    analysis = draft.get("analysis")
    if analysis:
        try:
            value["job"] = jobs.get_analysis_job(analysis["jobId"])
            state = value["job"]["status"]
            value["status"] = (
                "processing"
                if state in jobs.ACTIVE_STATES
                else "failed"
                if state in {"failed", "completed_with_errors"}
                else "ready"
            )
        except jobs.AnalysisJobError:
            value["status"] = "interrupted"
            try:
                result = Path(resolve_result_dir(analysis["sessionId"]))
                config = json.loads((result / "run_config.json").read_text(encoding="utf-8"))
                if config.get("assessment_workflow", {}).get("draft_id") == draft["id"]:
                    value["status"] = "ready"
            except (SessionError, OSError, ValueError):
                pass
    return value


def list_drafts(root: str) -> list[dict[str, Any]]:
    entries = []
    for path in (WORKFLOW_ROOT / "drafts").glob("*/draft.json"):
        try:
            draft = load(path.parent.name)
            if Path(draft["studiesRoot"]) == Path(root):
                entries.append(describe(draft))
        except (HTTPException, KeyError):
            continue
    return sorted(entries, key=lambda item: item["updatedAt"], reverse=True)


def ensure_editable(draft: dict[str, Any]) -> None:
    if draft["id"] in UPLOADING:
        raise HTTPException(409, "Wait for the video upload to finish.")
    if draft.get("published") or draft.get("analysis"):
        raise HTTPException(
            409,
            "This assessment has already been submitted. Create a new assessment to change its setup.",
        )


def update(identifier: str, payload: DraftUpdate) -> dict[str, Any]:
    from assistant.models import validate_choice

    with LOCK:
        draft = load(identifier)
        ensure_editable(draft)
        setup = payload.setup.model_dump()
        if "visionModel" not in payload.setup.model_fields_set:
            setup["visionModel"] = draft["setup"].get("visionModel")
        if setup["visionModel"] != draft["setup"].get("visionModel"):
            validate_choice(setup["visionModel"])
        if "cameraMode" not in payload.setup.model_fields_set:
            setup["cameraMode"] = draft["setup"]["cameraMode"]
        if draft.get("participantKey"):
            for field in ("studyName", "studyFolder", "participantId"):
                setup[field] = draft["setup"][field]
        if draft["source"] == "existing":
            for field in ("studyName", "studyFolder", "participantId", "taskName"):
                setup[field] = draft["setup"][field]
        if setup.get("studyFolder") and task_hidden(
            trash_items(draft["studiesRoot"]), f"{setup['studyFolder']}::{setup['participantId']}"
        ):
            raise HTTPException(409, "Restore this participant before adding a task.")
        selected = draft["videos"]
        choices = {v["id"]: v for v in draft["availableVideos"]}
        if payload.videoIds is not None:
            if len(set(payload.videoIds)) != len(payload.videoIds) or any(
                v not in choices for v in payload.videoIds
            ):
                raise HTTPException(422, "Choose recordings from this assessment.")
            selected = [choices[identifier] for identifier in payload.videoIds]
        elif setup["cameraMode"] != draft["setup"]["cameraMode"]:
            selected = [choices[v["id"]] for v in selected]
        videos = assign_roles(selected, setup["cameraMode"], payload.roles)
        if videos != draft["videos"] or setup["cameraMode"] != draft["setup"]["cameraMode"]:
            setup["footageReviewed"] = setup["synchronizationConfirmed"] = False
        draft["videos"] = videos
        if setup["cameraMode"] == "multi":
            for video in videos:
                choices[video["id"]]["role"] = video["role"]
        draft.update(setup=setup, step=payload.step)
        save(draft)
        return describe(draft)


def assign_roles(
    videos: list[dict[str, Any]], mode: CameraMode, roles: list[CameraRole] | None = None
) -> list[dict[str, Any]]:
    if roles is not None and len(roles) != len(videos):
        raise HTTPException(422, "Assign a role to each selected view.")
    if mode == "single":
        if len(videos) > 1:
            raise HTTPException(
                422, "Choose the one recording to retain for single-camera analysis."
            )
        return [dict(video, role="camera") for video in videos]
    if len(videos) > 3:
        raise HTTPException(422, "Choose at most three camera views.")
    assigned: list[str] = list(roles) if roles is not None else [v["role"] for v in videos]
    if roles is None:
        unused = iter(role for role in CAMERA_ROLES if role not in assigned)
        seen = set()
        for index, role in enumerate(assigned):
            if role not in CAMERA_ROLES or role in seen:
                assigned[index] = next(unused)
            seen.add(assigned[index])
    if any(role not in CAMERA_ROLES for role in assigned) or len(set(assigned)) != len(videos):
        raise HTTPException(422, "Assign a different camera role to each view.")
    return [dict(video, role=role) for video, role in zip(videos, assigned, strict=True)]


def reset_footage_confirmation(draft: dict[str, Any]) -> None:
    draft["setup"]["footageReviewed"] = False
    draft["setup"]["synchronizationConfirmed"] = False


def validate_ready(draft: dict[str, Any]) -> Setup:
    setup = Setup(**draft["setup"])
    if not setup.taskName.strip() or not setup.studyName.strip():
        raise HTTPException(422, "Enter the study and task names.")
    validate_participant_id(setup.participantId)
    videos = draft["videos"]
    validate_camera_count(setup.cameraMode, len(videos))
    if not 1 <= len(videos) <= 3 or any(not Path(video["path"]).is_file() for video in videos):
        raise HTTPException(422, "Select one to three available video files.")
    if len({video["role"] for video in videos}) != len(videos):
        raise HTTPException(422, "Assign a different role to each camera.")
    if not setup.footageReviewed or not setup.conditionsReviewed:
        raise HTTPException(422, "Review the footage and task conditions before starting analysis.")
    if len(videos) > 1:
        if not setup.synchronizationConfirmed:
            raise HTTPException(
                422, "Confirm that the camera views show the same take with aligned start times."
            )
        durations = [video["durationSec"] for video in videos]
        if max(durations) - min(durations) > max(1.0, max(durations) * 0.02):
            raise HTTPException(
                422,
                "Camera durations differ. Align the videos to the same interval before analysis.",
            )
    return setup


def validate_camera_count(mode: CameraMode, count: int) -> None:
    if mode == "single" and count != 1:
        raise HTTPException(422, "Single-camera analysis requires exactly one camera.")
    if mode == "multi" and count not in (2, 3):
        raise HTTPException(422, "Multi-camera analysis requires two or three cameras.")


def overrides(draft: dict[str, Any], setup: Setup) -> dict[str, Any]:
    from assistant.config import load_config
    from assistant.models import is_local

    # Freeze the preferred tag without contacting Ollama during posture processing.
    config = load_config()
    vision_model = setup.visionModel or preferences().visionModel
    if not vision_model and is_local(config.endpoint) and config.dialect != "openai":
        vision_model = config.scene_model
    return {
        "task_inputs": {
            "assumption_profile": setup.assumptionProfile,
            "load_kg": setup.loadKg,
            "coupling": setup.coupling,
            "shock_or_rapid_force": setup.shockForce,
            "left_arm_supported_or_leaning": setup.leftArmSupport,
            "right_arm_supported_or_leaning": setup.rightArmSupport,
            "left_arm_supported_or_gravity_assisted": setup.leftArmSupport,
            "right_arm_supported_or_gravity_assisted": setup.rightArmSupport,
            "left_wrist_twist_at_end_range": setup.leftWristEndRange,
            "right_wrist_twist_at_end_range": setup.rightWristEndRange,
            "rula_load_type": setup.rulaLoadType,
        },
        "rnle": {"participant_height_cm": setup.heightCm, "load_weight_kg": setup.loadKg},
        "fusion": {"camera_roles": {video["role"]: i for i, video in enumerate(draft["videos"])}},
        "assessment_workflow": {
            "vision_model": vision_model,
            "vision_model_override": setup.visionModel,
            "camera_mode": setup.cameraMode,
            **({"recordings": draft["recordings"]} if "recordings" in draft else {}),
            "draft_id": draft["id"],
            "source": draft["source"],
            "selected_take": draft.get("selectedTake"),
            "confirmed_at": now(),
        },
    }


def publish(draft: dict[str, Any], setup: Setup) -> dict[str, str]:
    """Add a unique trial while preserving all existing recordings and metadata."""
    if draft_hidden(draft):
        raise HTTPException(409, "Restore this assessment before publishing its task.")
    if draft.get("published"):
        return draft["published"]
    root = Path(draft["studiesRoot"]).resolve()
    if not root.is_dir():
        raise HTTPException(422, "The studies folder is unavailable.")
    source_inputs_path = None
    if source_trial := draft.get("existing"):
        context = find_context(str(root), source_trial["participantKey"])
        trial = find_trial(context, source_trial["taskFolder"])
        source_inputs_path = Path(trial["path"]) / "assessment_inputs.yaml"
        # Keep the original library entry for its first, complete assessment.
        # A subset, reordered experiment or subsequent assessment gets a distinct trial.
        selected = [v["id"] for v in draft["videos"]]
        available = [v["id"] for v in draft.get("availableVideos", draft["videos"])]
        if selected == available and not jobs.analysis_jobs._existing_session(
            context, trial, setup.samplingFps
        ):
            entries = [
                {
                    "file": Path(v["path"]).name,
                    "role": v["role"],
                    "sha256": v.get("sha256") or recording_hash(Path(v["path"])),
                }
                for v in draft["videos"]
            ]
            try:
                for entry in entries:
                    recorded_path(Path(trial["path"]), entry)
            except (ValueError, OSError) as exc:
                raise HTTPException(409, str(exc)) from exc
            draft["recordings"] = entries
            draft["published"] = source_trial
            save(draft)
            return source_trial
    study_folder = f"study_assessment_{draft['id']}"
    study_config: dict[str, Any] = {
        "study_name": setup.studyName.strip(),
        "created_at": now(),
        "tasks": [],
    }
    if setup.studyFolder:
        found = next(
            (
                study
                for study in discover_studies(str(root))
                if study["folder_name"] == setup.studyFolder
            ),
            None,
        )
        if not found:
            raise HTTPException(422, "The selected study is no longer available.")
        study_folder, study_config = found["folder_name"], dict(found["config"])
    study_path = (root / study_folder).resolve()
    if not study_path.is_relative_to(root):
        raise HTTPException(422, "The study folder must be inside the selected library.")
    if task_hidden(trash_items(str(root)), f"{study_folder}::{setup.participantId}", f"assessment_{draft['id']}"):
        raise HTTPException(409, "Restore this assessment before publishing its task.")
    participant_path = study_path / f"participant_{setup.participantId}"
    # Reuse the actual folder for an existing participant, even when its folder name differs.
    if setup.studyFolder:
        participant_path = next(
            (
                Path(item["path"])
                for item in discover_participants(str(study_path))
                if str(item["id"]) == setup.participantId
            ),
            participant_path,
        )
    task_folder = f"assessment_{draft['id']}"
    if not participant_path.resolve().is_relative_to(study_path):
        raise HTTPException(422, "The participant folder must be inside the selected study.")
    task_path = participant_path / task_folder
    staging_path = participant_path / f".{task_folder}"
    working_path = task_path if task_path.exists() else staging_path
    raw = working_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    draft["recordings"] = []
    for i, video in enumerate(draft["videos"]):
        source = Path(video["path"])
        target = raw / f"cam_{i}{source.suffix.lower()}"
        digest = video.get("sha256") or recording_hash(source)
        if not target.exists() or recording_hash(target) != digest:
            partial = target.with_suffix(".partial")
            try:
                shutil.copy2(source, partial)
                if recording_hash(partial) != digest:
                    raise HTTPException(
                        409, f"The recording {video['name']} has changed. Review the source again."
                    )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        draft["recordings"].append({"file": target.name, "sha256": digest, "role": video["role"]})
    info_path = participant_path / "participant_info.json"
    if not info_path.exists():
        atomic_json(info_path, {"participant_id": setup.participantId, "height_cm": setup.heightCm})
    elif draft.get("participantKey") and setup.heightCm is not None:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("assessment_group") is True and info.get("height_cm") is None:
            atomic_json(info_path, {**info, "height_cm": setup.heightCm})
    tasks = list(study_config.get("tasks", []))
    if not any(task.get("folder_name") == task_folder for task in tasks):
        tasks.append(
            {
                "folder_name": task_folder,
                "name": setup.taskName.strip(),
                "primary_tool": setup.primaryMethod,
            }
        )
    study_config["tasks"] = tasks
    atomic_json(
        working_path / "source_manifest.json",
        {
            "assessment_id": draft["id"],
            "source": draft["source"],
            "original_trial": draft.get("existing"),
            "camera_mode": setup.cameraMode,
            "recordings": draft["recordings"],
            "selected_take": draft.get("selectedTake"),
            "videos": draft["videos"],
            "created_at": now(),
        },
    )
    # JSON is also valid YAML. Keep entered inputs with new footage for later batch runs.
    inputs: dict[str, Any] = {}
    if source_inputs_path and source_inputs_path.is_file():
        inputs = jobs._load_override(source_inputs_path)
        inputs.pop("assessment_workflow", None)
    jobs._merge_mapping(inputs, overrides(draft, setup))
    atomic_json(working_path / "assessment_inputs.yaml", inputs)
    if working_path != task_path:
        working_path.rename(task_path)
    atomic_json(study_path / "study_config.json", study_config)
    draft["published"] = {
        "participantKey": f"{study_folder}::{setup.participantId}",
        "taskFolder": task_folder,
    }
    save(draft)
    clear_all_caches()
    return draft["published"]


def analyze(identifier: str, current_root: str) -> dict[str, Any]:
    with LOCK:
        draft = load(identifier)
        if Path(current_root) != Path(draft["studiesRoot"]):
            raise HTTPException(
                409, "Switch back to this assessment's studies folder before analysis."
            )
        if identifier in UPLOADING:
            raise HTTPException(409, "Wait for the upload to finish.")
        if draft.get("analysis") and describe(draft)["status"] in {"ready", "processing"}:
            return describe(draft)
        setup = validate_ready(draft)
        if jobs.analysis_jobs.active():
            raise HTTPException(
                409, "Another analysis is running. This setup is saved; try again when it finishes."
            )
        identity = publish(draft, setup)
        job = jobs.start_analysis_job(
            scope="trial",
            participant_key=identity["participantKey"],
            task_folder=identity["taskFolder"],
            sampling=setup.samplingFps,
            assessment_overrides=overrides(draft, setup),
        )
        study, participant = identity["participantKey"].split("::", 1)
        draft["analysis"] = {
            "jobId": job["jobId"],
            "sessionId": SessionRef(
                study, participant, identity["taskFolder"], setup.samplingFps
            ).session_id,
        }
        draft["step"] = 4
        save(draft)
        return describe(draft)


class Review(BaseModel):
    reviewer: str = Field(default="", max_length=160)
    findings: str = Field(default="", max_length=12000)
    actions: str = Field(default="", max_length=12000)
    reviewed: bool = False

    @model_validator(mode="after")
    def require_review_content(self):
        if self.reviewed and (not self.reviewer.strip() or not self.findings.strip()):
            raise ValueError(
                "Enter the reviewer name and findings before marking the assessment reviewed."
            )
        return self


def review_path(session_id: str) -> Path:
    return WORKFLOW_ROOT / "reviews" / f"{hashlib.sha256(session_id.encode()).hexdigest()}.json"


def result_fingerprint(session_id: str) -> str:
    result = Path(resolve_result_dir(session_id))
    digest = hashlib.sha256()
    for name in ("video_summary.json", "run_config.json"):
        digest.update((result / name).read_bytes())
    observation = result / "frame_results.jsonl"
    if observation.exists():
        stat = observation.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def get_review(session_id: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        **Review().model_dump(),
        "reviewedAt": None,
        "updatedAt": None,
        "stale": False,
    }
    try:
        saved = json.loads(review_path(session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return value
    value.update(saved)
    try:
        stale = saved.get("fingerprint") != result_fingerprint(session_id)
    except (SessionError, OSError):
        stale = True
    value["stale"] = stale
    if stale:
        value["reviewed"] = False
        value["reviewedAt"] = None
    return value


def save_review(session_id: str, review: Review) -> dict[str, Any]:
    value = {
        **review.model_dump(),
        "updatedAt": now(),
        "reviewedAt": now() if review.reviewed else None,
        "fingerprint": result_fingerprint(session_id),
        "stale": False,
    }
    with LOCK:
        atomic_json(review_path(session_id), value)
    return value
