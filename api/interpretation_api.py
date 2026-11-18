from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api import interpretations
from api.study_folders import require_local_request
from assistant.backend import BackendError
from assistant.interpretation import generate

router = APIRouter(
    prefix="/api/sessions/{session_id}/interpretations",
    dependencies=[Depends(require_local_request)],
)


class Generation(BaseModel):
    requestedModel: str | None = Field(default=None, max_length=200)


class Selection(BaseModel):
    observationId: str = Field(min_length=1, max_length=32)
    include: bool


@router.get("")
def get_interpretations(session_id: str):
    return {"interpretations": interpretations.saved(session_id)}


@router.post("")
def generate_interpretation(session_id: str, request: Generation):
    try:
        return {"interpretations": generate(session_id, request.requestedModel)}
    except BackendError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/{record_id}/selection")
def include_observation(session_id: str, record_id: str, request: Selection):
    return {
        "interpretations": interpretations.select(
            session_id, record_id, request.observationId, request.include
        )
    }
