"""HTTP workflow for recorded-video setup, local capture, and human review."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api import assessment_trash as trash
from api import assessments as store
from api.jobs import AnalysisJobError
from api.media import ensure_proxy
from api.study_folders import require_local_request
from dashboard.camera_placement import PlacementError
from dashboard.data_loader import discover_studies
from dashboard.recording_service import CameraSpec, RecordingError, get_recording_service
from ergonomics.profiles import PROFILES
from pipeline.recording_sources import VIDEO_EXTENSIONS, recording_hash

router = APIRouter(prefix="/api", dependencies=[Depends(require_local_request)])
CAPTURE_OWNER: str | None = None
CAPTURE_TAKE: str | None = None


def studies_root() -> str:
    from api.main import STUDIES_ROOT

    return STUDIES_ROOT


def ensure_capture_idle() -> None:
    service = get_recording_service()
    if CAPTURE_OWNER or service.preview_active or service.is_recording:
        raise HTTPException(
            409, "Stop the camera preview and recording before changing the studies folder."
        )


def ensure_not_capturing(identifier: str) -> None:
    if CAPTURE_OWNER == identifier:
        raise HTTPException(409, "Stop the cameras before changing this assessment.")


@router.get("/assessment-options")
def assessment_options():
    return {
        "studies": [
            {"folder": item["folder_name"], "name": item["name"]}
            for item in discover_studies(studies_root())
        ],
        "profiles": {name: profile.values for name, profile in PROFILES.items()},
        "preferences": store.preferences().model_dump(),
    }


@router.post("/assessment-preferences")
def save_preferences(payload: store.Preferences):
    from assistant.models import validate_choice

    with store.LOCK:
        previous_model = store.preferences().visionModel
        if "visionModel" not in payload.model_fields_set:
            payload = payload.model_copy(update={"visionModel": previous_model})
        if payload.visionModel != previous_model:
            validate_choice(payload.visionModel)
        store.atomic_json(store.WORKFLOW_ROOT / "preferences.json", payload.model_dump())
    return payload.model_dump()


@router.get("/assessments")
def list_assessments():
    return {"assessments": store.list_drafts(studies_root())}


@router.get("/assessment-trash")
def list_trash():
    return {"items": trash.items(studies_root())}


@router.post("/assessment-trash")
def move_to_trash(selection: trash.TrashSelection):
    with store.LOCK:
        return trash.move(studies_root(), selection, CAPTURE_OWNER)


@router.post("/assessment-trash/{identifier}/restore")
def restore_from_trash(identifier: str):
    with store.LOCK:
        return trash.restore(studies_root(), identifier)


@router.post("/assessments")
def create_assessment(payload: store.CreateDraft):
    with store.LOCK:
        return store.create(payload, studies_root())


@router.get("/assessments/{identifier}")
def get_assessment(identifier: str):
    return store.describe(store.load(identifier))


@router.post("/assessments/{identifier}")
def update_assessment(identifier: str, payload: store.DraftUpdate):
    with store.LOCK:
        ensure_not_capturing(identifier)
        return store.update(identifier, payload)


@router.post("/assessments/{identifier}/videos")
async def upload_video(identifier: str, request: Request, name: str = Query(max_length=255)):
    extension = Path(name).suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise HTTPException(422, "Choose an MP4, MOV, AVI, MKV, or WebM video.")
    with store.LOCK:
        draft = store.load(identifier)
        store.ensure_editable(draft)
        if draft["source"] != "video":
            raise HTTPException(409, "Video upload is available for a recorded-video assessment.")
        if draft["setup"]["cameraMode"] == "single" and draft["videos"]:
            raise HTTPException(422, "Single-camera analysis accepts one recording.")
        if len(draft["videos"]) >= 3:
            raise HTTPException(422, "An assessment supports up to three camera views.")
        store.UPLOADING.add(identifier)
    target = store.draft_dir(identifier) / "uploads" / f"{uuid.uuid4().hex}{extension}"
    partial = target.with_suffix(".partial")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with partial.open("xb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > store.MAX_VIDEO_BYTES:
                    raise HTTPException(413, "The video exceeds the 10 GB per-file limit.")
                handle.write(chunk)
        partial.replace(target)
        video = store.describe_video(target, name=name)
        with store.LOCK:
            draft = store.load(identifier)
            if draft["setup"]["cameraMode"] == "multi":
                occupied = {item["role"] for item in draft["videos"]}
                video["role"] = next(role for role in store.CAMERA_ROLES if role not in occupied)
            draft["videos"].append(video)
            draft["availableVideos"].append(dict(video))
            store.reset_footage_confirmation(draft)
            store.save(draft)
        return store.describe(draft)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)
        with store.LOCK:
            store.UPLOADING.discard(identifier)


@router.post("/assessments/{identifier}/videos/{video_id}/remove")
def remove_video(identifier: str, video_id: str):
    with store.LOCK:
        draft = store.load(identifier)
        store.ensure_editable(draft)
        if draft["source"] != "video":
            raise HTTPException(409, "Camera views are selected together for this recording.")
        draft["videos"] = [video for video in draft["videos"] if video["id"] != video_id]
        draft["availableVideos"] = [v for v in draft["availableVideos"] if v["id"] != video_id]
        store.reset_footage_confirmation(draft)
        store.save(draft)
        return store.describe(draft)


def selected_video(identifier: str, video_id: str):
    draft = store.load(identifier)
    video = next((item for item in draft["availableVideos"] if item["id"] == video_id), None)
    if video is None:
        raise HTTPException(404, "This video is not selected in the assessment.")
    try:
        if video.get("sha256") and recording_hash(Path(video["path"])) != video["sha256"]:
            raise HTTPException(409, "This recording changed after selection. Review the source again.")
    except OSError as exc:
        raise HTTPException(409, "The selected recording is unavailable.") from exc
    return video


@router.get("/assessments/{identifier}/videos/{video_id}/status")
def video_status(identifier: str, video_id: str):
    video = selected_video(identifier, video_id)
    proxy = ensure_proxy(video["path"], store.draft_dir(identifier) / "media" / video_id, 0)
    return {"ready": proxy.state == "ready", "state": proxy.state, "detail": proxy.detail}


@router.get("/assessments/{identifier}/videos/{video_id}/play")
def video_play(identifier: str, video_id: str):
    video = selected_video(identifier, video_id)
    proxy = ensure_proxy(video["path"], store.draft_dir(identifier) / "media" / video_id, 0)
    if proxy.state != "ready" or proxy.path is None:
        raise HTTPException(409, proxy.detail)
    return FileResponse(proxy.path, media_type="video/mp4")


@router.post("/assessments/{identifier}/analyze")
def analyze_assessment(identifier: str):
    with store.LOCK:
        ensure_not_capturing(identifier)
        try:
            return store.analyze(identifier, studies_root())
        except AnalysisJobError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc


@router.get("/capture/status")
def capture_status():
    return {**get_recording_service().snapshot(), "assessmentId": CAPTURE_OWNER}


@router.post("/capture/devices")
def detect_devices():
    with store.LOCK:
        ensure_capture_idle()
        return {"devices": get_recording_service().detect_cameras(8)}


class CaptureCamera(BaseModel):
    deviceId: int = Field(ge=0, le=64)
    role: store.CameraRole = "camera"


class CaptureSetup(BaseModel):
    cameras: list[CaptureCamera] = Field(min_length=1, max_length=3)


@router.post("/assessments/{identifier}/capture/preview")
def start_preview(identifier: str, payload: CaptureSetup):
    global CAPTURE_OWNER
    with store.LOCK:
        draft = store.load(identifier)
        store.ensure_editable(draft)
        if draft["source"] != "live":
            raise HTTPException(409, "Camera capture requires a live-recording assessment.")
        if Path(draft["studiesRoot"]) != Path(studies_root()):
            raise HTTPException(409, "Switch back to this assessment's studies folder first.")
        ensure_capture_idle()
        mode = draft["setup"]["cameraMode"]
        store.validate_camera_count(mode, len(payload.cameras))
        if len({c.deviceId for c in payload.cameras}) != len(payload.cameras):
            raise HTTPException(422, "Choose different camera devices.")
        if mode == "multi" and (
            any(c.role not in store.CAMERA_ROLES for c in payload.cameras)
            or len({camera.role for camera in payload.cameras}) != len(payload.cameras)
        ):
            raise HTTPException(422, "Choose a different role for each camera.")
        service = get_recording_service()
        try:
            service.start_preview(
                [
                    CameraSpec(
                        camera.deviceId,
                        camera.role if mode == "multi" else "camera",
                        fps=store.preferences().cameraFps,
                    )
                    for camera in payload.cameras
                ]
            )
        except (RecordingError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        CAPTURE_OWNER = identifier
        return capture_status()


def require_capture_owner(identifier: str):
    if CAPTURE_OWNER != identifier:
        raise HTTPException(409, "This assessment does not own the active cameras.")
    return get_recording_service()


class InspectPreview(BaseModel):
    reviewId: str = Field(min_length=1, max_length=128)


class PlacementDecision(InspectPreview):
    choice: Literal["ready", "adjust", "unclear"]


def _placement_monitor(identifier: str):
    with store.LOCK:
        return require_capture_owner(identifier).placement()


@router.get("/assessments/{identifier}/capture/placement")
def get_placement(identifier: str):
    try:
        return _placement_monitor(identifier).status()
    except PlacementError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/assessments/{identifier}/capture/placement/{device}/inspect")
def inspect_preview(identifier: str, device: int, payload: InspectPreview):
    try:
        monitor = _placement_monitor(identifier)
        model = store.load(identifier)["setup"].get("visionModel")
        monitor.inspect(device, payload.reviewId, model=model)
        return monitor.status()
    except PlacementError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/assessments/{identifier}/capture/placement/{device}/decision")
def confirm_placement(identifier: str, device: int, payload: PlacementDecision):
    try:
        monitor = _placement_monitor(identifier)
        monitor.decide(device, payload.reviewId, payload.choice)
        return monitor.status()
    except PlacementError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/assessments/{identifier}/capture/placement/{device}/checks/{check_id}/{frame}.jpg")
def placement_evidence(identifier: str, device: int, check_id: str, frame: int):
    try:
        image = _placement_monitor(identifier).evidence(device, check_id, frame)
    except PlacementError as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/assessments/{identifier}/capture/{device_id}.jpg")
def preview_image(identifier: str, device_id: int):
    service = require_capture_owner(identifier)
    frame = service.latest_previews().get(device_id)
    if frame is None:
        raise HTTPException(404, "Waiting for the camera's first frame.")
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise HTTPException(503, "Camera preview could not be encoded.")
    return Response(
        encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"}
    )


@router.post("/assessments/{identifier}/capture/start")
def start_capture(identifier: str):
    global CAPTURE_TAKE
    with store.LOCK:
        service = require_capture_owner(identifier)
        take_id = uuid.uuid4().hex
        try:
            service.start_recording(store.draft_dir(identifier) / "takes" / take_id)
        except RecordingError as exc:
            raise HTTPException(409, str(exc)) from exc
        CAPTURE_TAKE = take_id
        return capture_status()


@router.post("/assessments/{identifier}/capture/stop")
def stop_capture(identifier: str):
    global CAPTURE_TAKE
    with store.LOCK:
        service = require_capture_owner(identifier)
        try:
            manifest = service.stop_recording()
        except RecordingError as exc:
            raise HTTPException(409, str(exc)) from exc
        roles = {str(spec["device_id"]): spec["label"] for spec in manifest["camera_specs"]}
        videos = [
            store.describe_video(Path(path), role=roles[str(device)])
            for device, path in manifest["raw_videos"].items()
        ]
        draft = store.load(identifier)
        videos = store.assign_roles(videos, draft["setup"]["cameraMode"])
        take = {
            "id": CAPTURE_TAKE,
            "createdAt": store.now(),
            "durationSec": manifest["duration_sec"],
            "videos": videos,
            "manifest": manifest,
        }
        draft["takes"].append(take)
        draft["selectedTake"] = CAPTURE_TAKE
        draft["videos"] = videos
        draft["availableVideos"] = [dict(v) for v in videos]
        store.reset_footage_confirmation(draft)
        store.save(draft)
        CAPTURE_TAKE = None
        return store.describe(draft)


@router.post("/assessments/{identifier}/capture/close")
def close_preview(identifier: str):
    global CAPTURE_OWNER
    with store.LOCK:
        service = require_capture_owner(identifier)
        if service.is_recording:
            raise HTTPException(409, "Stop and save the recording before closing the cameras.")
        service.stop_preview()
        CAPTURE_OWNER = None
        return capture_status()


@router.post("/assessments/{identifier}/takes/{take_id}")
def select_take(identifier: str, take_id: str):
    with store.LOCK:
        ensure_not_capturing(identifier)
        draft = store.load(identifier)
        store.ensure_editable(draft)
        take = next((item for item in draft["takes"] if item["id"] == take_id), None)
        if take is None:
            raise HTTPException(404, "Recording take not found.")
        draft["selectedTake"] = take_id
        mode = draft["setup"]["cameraMode"]
        draft["videos"] = (
            []
            if mode == "single" and len(take["videos"]) > 1
            else store.assign_roles(take["videos"], mode)
        )
        draft["availableVideos"] = [dict(v) for v in take["videos"]]
        store.reset_footage_confirmation(draft)
        store.save(draft)
        return store.describe(draft)


@router.get("/sessions/{session_id}/review")
def get_review(session_id: str):
    return store.get_review(session_id)


@router.post("/sessions/{session_id}/review")
def save_review(session_id: str, payload: store.Review):
    return store.save_review(session_id, payload)
