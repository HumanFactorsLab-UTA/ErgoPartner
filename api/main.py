"""HTTP surface for the posture analysis application.

The scoring, aggregation, and attribution all live in the modules the command
line and the previous dashboard already used. This layer does three things:
resolves a session identifier to a result directory, shapes existing analysis
into JSON, and serves the two binary things a browser cannot compute -- a decoded
camera frame with the recorded pose drawn on it, and the source video.

Run it with::

    python -m uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import mimetypes
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import Scope

from api import analysis as analysis_api
from api import frame_data as frame_data_api
from api import replay as replay_api
from api.assessment_api import ensure_capture_idle
from api.assessment_api import router as assessment_router
from api.assessment_trash import items as trash_items
from api.assessment_trash import task_hidden
from api.assessments import LOCK as ASSESSMENT_LOCK
from api.assessments import get_review
from api.jobs import (
    AnalysisJobError,
    analysis_jobs,
    get_analysis_job,
    list_active_analysis_jobs,
    start_analysis_job,
)
from api.media import ensure_proxy
from api.sessions import (
    SessionError,
    SessionRef,
    catalog,
    find_context,
    find_trial,
    load_summary,
    parse_session_id,
    resolve_result_dir,
    session_videos,
)
from api.study_folders import folder_listing, require_local_request, validate_studies_folder
from dashboard.cache import ttl_cache
from dashboard.data_loader import detect_current_results, get_processed_fps
from dashboard.result_store import ResultStoreError, open_result_store
from paths import STUDIES_DIR_STR

LOGGER = logging.getLogger(__name__)

#: The catalog, video routes and analysis jobs share the study location.
STUDIES_ROOT = STUDIES_DIR_STR
STUDIES_FOLDER_LOCK = threading.Lock()

#: The Vite dev server. Production serves the built assets from the same origin,
#: where CORS does not apply.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


async def require_active_session(request: Request) -> None:
    # Review notes and cached downloads can be served without opening the result bundle.
    if (session_id := request.path_params.get("session_id")) and (
        entries := trash_items(STUDIES_ROOT)
    ):
        ref = parse_session_id(session_id)
        if task_hidden(entries, f"{ref.study_folder}::{ref.participant_id}", ref.task_folder):
            raise HTTPException(404, "This assessment is in Trash. Restore it from Assessments.")


app = FastAPI(title="ErgoPartner Analysis API", version="1.0.0", dependencies=[Depends(require_active_session)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(assessment_router)


def mount_assistant() -> None:
    """Attach the conversational assistant, if its package imports cleanly.

    The assistant is an addition to this application, never a prerequisite for
    it. A missing dependency or a broken import there must leave the dashboard
    fully working rather than refusing to start, so the failure is logged and
    the routes simply do not exist.
    """

    try:
        from api.assistant_api import router as assistant_router
        from api.interpretation_api import router as interpretation_router
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Assistant routes unavailable: %s", exc)
        return
    app.include_router(assistant_router)
    app.include_router(interpretation_router)


mount_assistant()


@app.exception_handler(SessionError)
async def _session_error(_: Request, exc: SessionError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(ResultStoreError)
async def _store_error(_: Request, exc: ResultStoreError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/api/health")
def health() -> dict[str, Any]:
    root = Path(STUDIES_ROOT)
    return {
        "ok": True,
        "studiesRoot": str(root),
        "studiesRootExists": root.is_dir(),
    }


class StudiesFolderRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


@app.get("/api/studies-folder")
def get_studies_folder() -> dict[str, Any]:
    return {"path": STUDIES_ROOT, "exists": Path(STUDIES_ROOT).is_dir()}


@app.get("/api/studies-folder/browse")
def browse_studies_folder(request: Request, path: str = Query("", max_length=4096)) -> dict[str, Any]:
    require_local_request(request)
    return folder_listing(path or STUDIES_ROOT)


@app.post("/api/studies-folder")
def select_studies_folder(request: Request, selection: StudiesFolderRequest) -> dict[str, Any]:
    global STUDIES_ROOT
    require_local_request(request)
    root = validate_studies_folder(selection.path)
    try:
        with STUDIES_FOLDER_LOCK, ASSESSMENT_LOCK:
            ensure_capture_idle()
            analysis_jobs.set_studies_root(root)
            STUDIES_ROOT = str(root)
    except AnalysisJobError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    except OSError as exc:
        raise HTTPException(500, "The folder preference could not be saved. The current folder is unchanged.") from exc
    return get_studies_folder()


class AnalysisJobRequest(BaseModel):
    """A server-resolved analysis request; browser paths are never accepted."""

    scope: Literal["trial", "participant", "study", "all"]
    participantKey: str | None = None
    studyFolder: str | None = None
    taskFolder: str | None = None
    samplingFps: int | None = Field(default=10, ge=5, le=60)
    rerun: bool = False


def _job_error(exc: AnalysisJobError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload)


@app.post("/api/analysis/jobs", status_code=202, response_model=None)
def start_analysis(request: AnalysisJobRequest):
    """Queue one trial, one participant, one study, or every missing trial."""

    try:
        with ASSESSMENT_LOCK:
            return start_analysis_job(
                scope=request.scope,
                participant_key=request.participantKey,
                study_folder=request.studyFolder,
                task_folder=request.taskFolder,
                sampling=request.samplingFps,
                rerun=request.rerun,
            )
    except AnalysisJobError as exc:
        return _job_error(exc)


@app.get("/api/analysis/jobs", response_model=None)
def active_analysis_jobs(active: bool = Query(True)):
    """Reconnect the library to an analysis that survived page navigation."""

    if not active:
        return {"jobs": []}
    return {"jobs": list_active_analysis_jobs()}


@app.get("/api/analysis/jobs/{job_id}", response_model=None)
def analysis_job_status(job_id: str):
    try:
        return get_analysis_job(job_id)
    except AnalysisJobError as exc:
        return _job_error(exc)


# ----------------------------------------------------------------------
# Participants and sessions
# ----------------------------------------------------------------------


def _session_ref(context: dict[str, Any], trial: dict[str, Any], sampling: int | None) -> SessionRef:
    return SessionRef(
        study_folder=str(context["study"]["folder_name"]),
        participant_id=str(context["participant_id"]),
        task_folder=str(trial["folder_name"]),
        sampling=sampling,
    )


@ttl_cache(ttl=20)
def _trial_card(context: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    """A trial as the library shows it: what it is, and whether it is analysed."""

    study = str(context["study"]["folder_name"])
    participant = str(context["participant_id"])
    folder = str(trial["folder_name"])
    samplings = list(get_processed_fps(study, participant, folder))
    card: dict[str, Any] = {
        "taskId": trial.get("task_id"),
        "name": trial.get("name"),
        "folder": folder,
        "description": trial.get("description"),
        "primaryTool": trial.get("primary_tool"),
        "cameras": trial.get("cameras"),
        "samplings": [None if item is None else int(item) for item in samplings],
        "status": "not_analysed",
        "sessionId": None,
        "durationSec": None,
        "recordedAt": context.get("date_label"),
        "reba": None,
        "rula": None,
    }
    if not samplings:
        return card

    preferred = samplings[0]
    result = None
    for candidate in samplings:
        candidate_result = detect_current_results(study, participant, folder, candidate)
        if candidate_result and candidate_result.get("dir"):
            preferred = candidate
            result = candidate_result
            break
    if result is None:
        card["status"] = "incomplete"
        return card
    ref = _session_ref(context, trial, preferred)
    card["sessionId"] = ref.session_id
    card["status"] = "analysed"
    review = get_review(ref.session_id)
    try:
        updated = (Path(result["dir"]) / "video_summary.json").stat().st_mtime
        card["updatedAt"] = datetime.fromtimestamp(updated, UTC).isoformat()
    except OSError:
        pass
    if review.get("updatedAt") and str(review["updatedAt"]) > str(card.get("updatedAt", "")):
        card["updatedAt"] = review["updatedAt"]
    card["reviewStatus"] = "reviewed" if review["reviewed"] else "needs_review"
    try:
        summary = load_summary(ref.session_id)
    except SessionError:
        card["status"] = "incomplete"
        return card
    card["durationSec"] = summary.get("duration_sec")
    for method in ("reba", "rula"):
        block = summary.get(f"{method}_summary") or {}
        maximum = block.get("maximum_score")
        card[method] = {
            "maximum": maximum,
            "band": analysis_api._band_payload(maximum, method),
            "observations": block.get("observation_count"),
        }
    return card


@app.get("/api/participants")
def list_participants(search: str = Query("", max_length=120)) -> dict[str, Any]:
    """The participant library: one entry per participant with their trials."""

    needle = search.strip().lower()
    entries = []
    for context in catalog(STUDIES_ROOT):
        if needle and needle not in str(context.get("search_text", "")).lower():
            continue
        info = context.get("participant", {}).get("info") or {}
        trials = [_trial_card(context, trial) for trial in context.get("tasks", [])]
        analysed = [item for item in trials if item["status"] == "analysed"]
        entries.append(
            {
                "key": context.get("key"),
                "participantId": context.get("participant_id"),
                "label": context.get("label"),
                "recordedAt": context.get("date_label"),
                "study": context.get("study", {}).get("folder_name"),
                "studyName": context.get("study", {}).get("name"),
                "age": info.get("age"),
                "sex": info.get("sex"),
                "heightCm": info.get("height_cm"),
                "weightKg": info.get("weight_kg"),
                "dominantHand": info.get("dominant_hand"),
                "trialCount": len(trials),
                "analysedCount": len(analysed),
                "trials": trials,
            }
        )
    return {"count": len(entries), "participants": entries}


@app.get("/api/participants/{participant_key}")
def get_participant(participant_key: str) -> dict[str, Any]:
    context = find_context(STUDIES_ROOT, participant_key)
    info = context.get("participant", {}).get("info") or {}
    return {
        "key": context.get("key"),
        "participantId": context.get("participant_id"),
        "label": context.get("label"),
        "recordedAt": context.get("date_label"),
        "study": context.get("study", {}).get("folder_name"),
        "age": info.get("age"),
        "sex": info.get("sex"),
        "heightCm": info.get("height_cm"),
        "weightKg": info.get("weight_kg"),
        "dominantHand": info.get("dominant_hand"),
        "notes": info.get("notes"),
        "trials": [_trial_card(context, trial) for trial in context.get("tasks", [])],
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    ref = parse_session_id(session_id)
    resolve_result_dir(session_id)
    context = find_context(STUDIES_ROOT, f"{ref.study_folder}::{ref.participant_id}")
    trial = find_trial(context, ref.task_folder)
    info = context.get("participant", {}).get("info") or {}
    overview = analysis_api.session_overview(session_id)
    overview["participant"] = {
        "key": context.get("key"),
        "participantId": context.get("participant_id"),
        "label": context.get("label"),
        "recordedAt": context.get("date_label"),
        "age": info.get("age"),
        "sex": info.get("sex"),
        "dominantHand": info.get("dominant_hand"),
        "heightCm": info.get("height_cm"),
    }
    overview["trial"] = {
        "taskId": trial.get("task_id"),
        "name": trial.get("name"),
        "folder": trial.get("folder_name"),
        "description": trial.get("description"),
        "primaryTool": trial.get("primary_tool"),
    }
    overview["sampling"] = ref.sampling
    overview["samplingLabel"] = ref.sampling_label
    overview["cameras"] = session_videos(STUDIES_ROOT, ref)
    return overview


@app.get("/api/sessions/{session_id}/timeline")
def get_timeline(
    session_id: str,
    method: str = Query("reba"),
    person: str | None = Query(None),
) -> dict[str, Any]:
    return analysis_api.timeline(session_id, method, person)


@app.get("/api/sessions/{session_id}/frames/{frame}")
def get_frame(
    session_id: str,
    frame: int,
    method: str = Query("reba"),
    person: str | None = Query(None),
) -> dict[str, Any]:
    return analysis_api.frame_detail(session_id, frame, method, person)


@app.get("/api/sessions/{session_id}/replay")
def get_replay(
    session_id: str,
    method: str = Query("reba"),
    person: str | None = Query(None),
    camera: int = Query(0, ge=0),
    max_frames: int = Query(360, alias="maxFrames", ge=2, le=900),
    anchor_frame: int | None = Query(None, alias="anchorFrame", ge=0),
) -> dict[str, Any]:
    """A compact, peak-preserving stick-figure trace for one recorded view."""

    return replay_api.replay_payload(
        session_id,
        camera_index=camera,
        person=person,
        method=method,
        max_frames=max_frames,
        anchor_frame=anchor_frame,
    )


@app.get("/api/sessions/{session_id}/observations")
def get_observations(
    session_id: str,
    method: str = Query("reba"),
    band: str = Query("all"),
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(60, ge=1, le=frame_data_api.MAX_PAGE_SIZE),
    person: str | None = Query(None),
) -> dict[str, Any]:
    """Non-decimated frame calculations for the research table."""

    return frame_data_api.observations_payload(
        session_id,
        method,
        band=band,
        status=status,
        page=page,
        limit=limit,
        person=person,
    )


@app.get("/api/sessions/{session_id}/risk-periods")
def get_risk_periods(
    session_id: str,
    method: str = Query("reba"),
    band: str = Query("all"),
    status: str | None = Query(None),
    person: str | None = Query(None),
) -> dict[str, Any]:
    """Every contiguous official action-band period, including evidence gaps."""

    return frame_data_api.risk_periods_payload(
        session_id,
        method,
        band=band,
        status=status,
        person=person,
    )


@app.get("/api/sessions/{session_id}/exports/frame-calculations.csv")
def export_frame_calculations(
    session_id: str,
    method: str = Query("reba"),
    band: str = Query("all"),
    status: str | None = Query(None),
    person: str | None = Query(None),
) -> StreamingResponse:
    """Generate a flattened, research-friendly CSV from the full JSONL trace."""

    return frame_data_api.csv_export_response(
        session_id,
        method=method,
        band=band,
        status=status,
        person=person,
        filename=f"{session_id}_{method}_frame_calculations",
    )


# ----------------------------------------------------------------------
# Media
# ----------------------------------------------------------------------


def _frame_jpeg(
    session_id: str,
    frame: int,
    camera_index: int,
    region: str,
    overlay: bool,
    width: int,
    person: str | None,
) -> bytes:
    """Validate the source and current run before reusing decoded media."""
    ref = parse_session_id(session_id)
    videos = session_videos(STUDIES_ROOT, ref)
    match = next((item for item in videos if item["camera_index"] == camera_index), None)
    if match is None:
        raise SessionError(f"No camera {camera_index} in this recording")

    source = Path(match["path"])
    observations = Path(resolve_result_dir(session_id)) / "frame_results.jsonl"
    try:
        source_stat = source.stat()
        run_stat = observations.stat()
    except OSError as exc:
        raise SessionError("The recording or frame records are unavailable") from exc
    version = tuple((s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_ino)
                    for s in (source_stat, run_stat))
    return _decoded_jpeg(str(source), str(observations), version, frame, camera_index,
                         region, overlay, width, person)


@ttl_cache(ttl=300, copy_result=False)
def _decoded_jpeg(source: str, observations: str, version: tuple, frame: int, camera_index: int,
                  region: str, overlay: bool, width: int, person: str | None) -> bytes:
    import cv2

    from dashboard.analysis_service import read_video_frame

    store = open_result_store(Path(observations))
    entry = next(
        (
            item
            for item in store.entries
            if item.frame_number == frame
            and (person is None or item.person_identifier == person)
        ),
        None,
    )
    record = store.load_record_at_offset(entry.offset) if entry is not None else None

    source_frame = frame
    if record:
        for reference in record.get("camera_frame_references") or []:
            if isinstance(reference, dict) and int(reference.get("camera_index", -1)) == camera_index:
                candidate = reference.get("source_frame_number")
                if isinstance(candidate, int):
                    source_frame = candidate
                break

    image = read_video_frame(
        source,
        source_frame,
        record if overlay else None,
        region or None,
        camera_index,
    )
    height, original_width = image.shape[:2]
    if width and original_width > width:
        scale = width / original_width
        image = cv2.resize(
            image, (width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(".jpg", image[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise SessionError("Could not encode the requested frame")
    return bytes(buffer)


@app.get("/api/sessions/{session_id}/frames/{frame}/cameras/{camera_index}.jpg")
def get_frame_image(
    session_id: str,
    frame: int,
    camera_index: int,
    region: str = Query(""),
    overlay: bool = Query(True),
    width: int = Query(720, ge=160, le=1920),
    person: str | None = Query(None),
) -> Response:
    payload = _frame_jpeg(
        session_id,
        frame,
        camera_index,
        region,
        overlay,
        width,
        person,
    )
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-cache"},
    )


_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 1024 * 512


def _source_for(session_id: str, camera_index: int) -> Path:
    ref = parse_session_id(session_id)
    videos = session_videos(STUDIES_ROOT, ref)
    match = next((item for item in videos if item["camera_index"] == camera_index), None)
    if match is None:
        raise SessionError(f"No camera {camera_index} in this recording")
    return Path(match["path"])


@app.get("/api/sessions/{session_id}/video/{camera_index}/status")
def get_video_status(session_id: str, camera_index: int) -> dict[str, Any]:
    """Whether a browser-playable copy of this recording exists yet."""

    source = _source_for(session_id, camera_index)
    status = ensure_proxy(source, resolve_result_dir(session_id), camera_index)
    return {
        "state": status.state,
        "detail": status.detail,
        "ready": status.state == "ready",
    }


@app.get("/api/sessions/{session_id}/video/{camera_index}")
def get_video(session_id: str, camera_index: int, request: Request) -> Response:
    """Serve a browser-playable copy of a recording, with byte-range support.

    The captured files are MPEG-4 Part 2, which no browser decodes, so what is
    served is a transcoded H.264 proxy. Range support is what makes the player
    scrubbable at all; without the 206 path it can only stream from the start,
    which defeats a video panel synchronized to a timeline.
    """

    source = _source_for(session_id, camera_index)
    status = ensure_proxy(source, resolve_result_dir(session_id), camera_index)
    if status.state != "ready" or status.path is None:
        raise HTTPException(
            status_code=503 if status.state == "building" else 409,
            detail=status.detail or "No playable copy of this recording is available",
            headers={"Retry-After": "3"} if status.state == "building" else None,
        )
    path = status.path
    size = path.stat().st_size
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    range_header = request.headers.get("range")
    start, end = 0, size - 1
    status = 200
    if range_header:
        found = _RANGE.match(range_header)
        if found:
            first, last = found.group(1), found.group(2)
            if first:
                start = int(first)
            if last:
                end = int(last)
            elif first:
                end = min(size - 1, start + _CHUNK * 8 - 1)
            if start >= size:
                raise HTTPException(
                    status_code=416, detail="Requested range is past the end of the recording"
                )
            end = min(end, size - 1)
            status = 206

    length = end - start + 1

    def stream():
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(_CHUNK, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Cache-Control": "private, no-cache",
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(stream(), status_code=status, media_type=media_type, headers=headers)


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------


def _report_payload(session_id: str, path: Path) -> dict[str, Any]:
    from reporting.document_builder import REPORT_FILENAME

    return {
        "exists": True,
        "filename": REPORT_FILENAME,
        "sizeKb": round(path.stat().st_size / 1024, 1),
        "sessionId": session_id,
    }


@app.post("/api/sessions/{session_id}/report")
def generate_report(session_id: str) -> dict[str, Any]:
    """Render the PDF assessment report for an analysed session."""

    from reporting.document_builder import ReportError, build_report

    try:
        return _report_payload(session_id, build_report(session_id))
    except ReportError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str) -> dict[str, Any]:
    """Report metadata for a session, when one has already been rendered."""

    from reporting.document_builder import report_exists

    cached = report_exists(session_id)
    if cached is None:
        return {"exists": False, "sessionId": session_id}
    return _report_payload(session_id, cached)


@app.get("/api/sessions/{session_id}/report/download")
def download_report(session_id: str) -> Response:
    """Download a rendered PDF report."""

    from reporting.document_builder import REPORT_FILENAME, report_exists

    cached = report_exists(session_id)
    if cached is None:
        raise SessionError("No report has been generated for this session yet")

    return FileResponse(
        str(cached),
        media_type="application/pdf",
        filename=REPORT_FILENAME,
    )


# ----------------------------------------------------------------------
# Static front end (production build)
# ----------------------------------------------------------------------


class WebClientFiles(StaticFiles):
    """Serve React's entry point for recognized application routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        parts = Path(path).parts
        if parts in (("library",), ("record",), ("assessments",), ("overview",), ("settings",), ("new",)):
            path = "index.html"
        elif len(parts) == 2 and parts[0] == "assessments" and re.fullmatch(r"[a-f0-9]{32}", parts[1]):
            path = "index.html"
        elif len(parts) in (2, 3) and parts[0] == "s" and (len(parts) == 2 or parts[2] in ("reba", "rula")):
            # Both direct links and in-app navigation use the same session ID format.
            try:
                parse_session_id(parts[1])
            except SessionError:
                pass
            else:
                path = "index.html"
        return await super().get_response(path, scope)


def mount_web_client() -> None:
    """Serve the built front end from this app when it has been built.

    In development the Vite server owns the client and proxies /api here, so a
    missing build directory is normal and must not stop the API from starting.
    """

    dist = Path(__file__).resolve().parent.parent / "web" / "dist"
    if not dist.is_dir():
        LOGGER.info("No web/dist build found; serving the API only")
        return
    app.mount("/", WebClientFiles(directory=str(dist), html=True), name="web")


mount_web_client()

__all__ = ["app", "mount_web_client"]
