"""Read identical visual evidence, collect blind labels, or confirm AI suggestions."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from assistant import focused_benchmark as benchmark
from assistant.backend import BackendError
from assistant.focused_review import Answers, StaleVisualReview, get_case, sample_moments
from assistant.models import effective_config

router = APIRouter(tags=["focused visual review"])
BASE = "/sessions/{session_id}/focused-review"
Method = Literal["reba", "rula"]


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame: int = Field(ge=0)
    person: str = Field(min_length=1, max_length=128)
    method: Method
    camera: int = Field(ge=0, le=2)
    caseId: str = Field(pattern=r"^[0-9a-f]{64}$")

    def case(self, session_id):
        case = get_case(session_id, self.frame, self.person, self.method, self.camera)
        if case.id != self.caseId:
            raise HTTPException(409, "This set of images has changed. Refresh the review.")
        return case


class Labels(Selection):
    answers: Answers


class Decision(Labels):
    predictionId: str = Field(pattern=r"^[0-9a-f]{64}$")


@router.get(BASE)
def read_case(
    session_id: str,
    frame: int = Query(ge=0),
    person: str = Query(min_length=1, max_length=128),
    method: Method = "reba",
    camera: int = Query(0, ge=0, le=2),
):
    return benchmark.describe(get_case(session_id, frame, person, method, camera))


@router.get(BASE + "/images/{image_index}")
def evidence_image(
    session_id: str,
    image_index: int,
    caseId: str = Query(pattern=r"^[0-9a-f]{64}$"),
    frame: int = Query(ge=0),
    person: str = Query(min_length=1, max_length=128),
    method: Method = "reba",
    camera: int = Query(0, ge=0, le=2),
):
    case = Selection(frame=frame, person=person, method=method, camera=camera, caseId=caseId).case(
        session_id
    )
    if not 1 <= image_index <= len(case.images):
        raise HTTPException(404, "This image was not part of the focused review.")
    return Response(
        case.images[image_index - 1],
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-cache"},
    )


@router.get(BASE + "/samples")
def suggested_moments(session_id: str, person: str = Query(min_length=1, max_length=128)):
    return {"frames": sample_moments(session_id, person)}


@router.post(BASE + "/predict")
def predict(session_id: str, request: Selection):
    try:
        return benchmark.predict(request.case(session_id))
    except BackendError as exc:
        raise HTTPException(503, str(exc)) from exc
    except StaleVisualReview as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post(BASE + "/reference")
def lock_reference(session_id: str, request: Labels):
    try:
        return benchmark.save_reference(request.case(session_id), request.answers)
    except StaleVisualReview as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post(BASE + "/decision")
def decide(session_id: str, request: Decision):
    try:
        return benchmark.save_decision(
            request.case(session_id), request.predictionId, request.answers
        )
    except StaleVisualReview as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get(BASE + "/evaluation")
def assessment_evaluation(session_id: str):
    config = effective_config(session_id)
    return benchmark.evaluation(benchmark.records(session_id, config), config)


@router.get("/focused-review/evaluation")
def library_evaluation():
    from api.main import STUDIES_ROOT

    return benchmark.library_evaluation(STUDIES_ROOT)
