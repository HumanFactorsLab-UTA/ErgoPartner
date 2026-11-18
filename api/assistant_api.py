"""HTTP surface for the assistant.

Answers stream as server-sent events because a turn that calls three tools and
looks at two frames takes tens of seconds, and a reviewer watching it read the
timeline and open a frame is watching the audit trail rather than a spinner.

Every route degrades rather than fails: with no model server running, health
reports why and the workspace shows the reason in place of the composer. The
dashboard itself never depends on any of this.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api.focused_review_api import router as focused_review_router
from api.sessions import SessionError
from api.study_folders import require_local_request
from assistant.backend import BackendError
from assistant.backend import health as backend_health
from assistant.brief import condition_sources, suggested_questions
from assistant.config import load_config
from assistant.conversation import load_transcript, run_turn
from assistant.models import (
    effective_config,
    inventory,
    is_local,
    save_selection,
    session_selection,
)
from assistant.proposals import accept_proposal, load_proposals, reject_proposal
from assistant.scene import build_scene, load_scene

LOGGER = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["assistant"])
router.include_router(focused_review_router)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class WorkspaceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    method: Literal["rula", "reba"]
    frame: int | None = Field(default=None, ge=0)
    timestampSec: float | None = Field(default=None, ge=0)
    person: str | None = Field(default=None, max_length=128)
    camera: int | None = Field(default=None, ge=0, le=2)
    region: str | None = Field(default=None, max_length=80, pattern=r"^[\w -]*$")


class TurnRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list)
    selection: WorkspaceSelection | None = None


class ProposalDecision(BaseModel):
    condition: str = Field(min_length=1, max_length=80)
    proposalId: str | None = Field(default=None, max_length=128)
    value: Any = None
    note: str | None = Field(default=None, max_length=500)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.get("/assistant/health")
def assistant_health(sessionId: str | None = None) -> dict[str, Any]:
    config = effective_config(sessionId)
    state: dict[str, Any]
    if config.model_issue or config.model_capabilities is not None:
        issue = config.model_issue
        if not config.enabled:
            issue = "The assessment assistant is disabled."
        elif not issue and config.model_capabilities is not None and "tools" not in config.model_capabilities:
            issue = f"{config.scene_model} supports image review. Choose a model with tool support to use chat."
        state = {"ok": not issue, "state": "unavailable" if issue else "ready",
                 "model": config.chat_model, "detail": issue or f"{config.chat_model} is ready."}
    else:
        state = backend_health(config)
    state["chatModel"] = config.chat_model
    state["sceneModel"] = config.scene_model
    state["promptVersion"] = config.prompt_version
    state["local"] = is_local(config.endpoint)
    if sessionId:
        state["selection"] = session_selection(sessionId).get("model")
    return state


class ModelSelection(BaseModel):
    model: str | None = Field(default=None, min_length=1, max_length=200)


@router.get("/assistant/models")
def available_models(refresh: bool = False, sessionId: str | None = None):
    from api.assessments import preferences
    from api.sessions import load_run_config

    config = load_config()
    catalog = inventory(config, refresh=refresh)
    default = preferences().visionModel or config.scene_model
    selection = session_selection(sessionId).get("model") if sessionId else None
    inherited = (load_run_config(sessionId).get("assessment_workflow", {}).get("vision_model")
                 if sessionId else None)
    return {**catalog, "configured": config.scene_model, "default": default,
            "inherited": inherited or default, "selection": selection}


@router.post("/sessions/{session_id}/assistant/model", dependencies=[Depends(require_local_request)])
def select_model(session_id: str, request: ModelSelection):
    try:
        save_selection(session_id, request.model)
    except BackendError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"model": request.model}


@router.get("/sessions/{session_id}/assistant/context", response_model=None)
def assistant_context(
    session_id: str, method: Literal["reba", "rula"] = "reba"
) -> dict[str, Any] | JSONResponse:
    """What the panel needs before the first question: history, prompts, scene."""

    try:
        suggestions = suggested_questions(session_id, method)
    except SessionError as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    scene = load_scene(session_id)
    return {
        "sessionId": session_id,
        "suggestions": suggestions,
        "history": load_transcript(session_id, limit=30),
        "scene": scene,
        "sceneBuilt": scene is not None,
        "proposals": load_proposals(session_id),
    }


@router.post("/sessions/{session_id}/assistant/messages")
def assistant_message(session_id: str, request: TurnRequest) -> StreamingResponse:
    """Stream one answer. Events: tool, delta, done, error."""

    history = [item.model_dump() for item in request.history]

    def stream():
        try:
            selection = (
                request.selection.model_dump(exclude_none=True) if request.selection else None
            )
            for event in run_turn(session_id, request.question, history, selection=selection):
                yield _sse(event)
        except SessionError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 - a broken turn must still close the stream
            LOGGER.exception("Assistant stream failed")
            yield _sse({"type": "error", "detail": f"The assistant failed: {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/sessions/{session_id}/scene")
def get_scene(session_id: str) -> dict[str, Any]:
    card = load_scene(session_id)
    return {"built": card is not None, "scene": card}


@router.post("/sessions/{session_id}/scene")
def post_scene(session_id: str) -> StreamingResponse:
    """Build the scene card, streaming progress because it takes a minute."""

    def stream():
        try:
            messages: list[str] = []

            def progress(text: str) -> None:
                messages.append(text)

            yield _sse({"type": "status", "detail": "Sampling frames"})
            card = build_scene(session_id, progress=progress)
            for note in messages[-1:]:
                yield _sse({"type": "status", "detail": note})
            yield _sse({"type": "done", "scene": card})
        except SessionError as exc:
            yield _sse({"type": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Scene build failed")
            yield _sse({"type": "error", "detail": f"Could not read the recording: {exc}"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/sessions/{session_id}/conditions")
def get_conditions(session_id: str) -> dict[str, Any]:
    return condition_sources(session_id)


@router.get("/sessions/{session_id}/proposals")
def get_proposals(session_id: str) -> dict[str, Any]:
    return load_proposals(session_id)


@router.post("/sessions/{session_id}/proposals/accept")
def post_accept(session_id: str, decision: ProposalDecision) -> dict[str, Any]:
    """Record an ergonomist's acceptance. This does not rescore anything.

    Accepting writes the value as an entered condition for the next run. The
    existing results are left exactly as they were computed, because a stored
    score must always match the inputs it was produced from.
    """

    return accept_proposal(
        session_id,
        decision.condition,
        decision.value,
        decision.note,
        proposal_id=decision.proposalId,
    )


@router.post("/sessions/{session_id}/proposals/reject")
def post_reject(session_id: str, decision: ProposalDecision) -> dict[str, Any]:
    return reject_proposal(
        session_id, decision.condition, decision.note, proposal_id=decision.proposalId
    )


@router.get("/sessions/{session_id}/assistant/transcript")
def get_transcript(session_id: str, limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
    return {"sessionId": session_id, "entries": load_transcript(session_id, limit=limit)}


__all__ = ["router"]
