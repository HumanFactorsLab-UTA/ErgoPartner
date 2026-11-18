"""Evidence and structured visual questions for the selected worker and camera."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from api import analysis
from api.assessments import result_fingerprint
from api.sessions import SessionError, resolve_result_dir
from assistant.backend import BackendError, complete, image_message
from assistant.config import AssistantConfig
from assistant.visuals import cameras, render
from dashboard.analysis_service import DashboardDataError
from pipeline.recording_sources import recording_hash

PROMPT_VERSION = "visual-contact/1"
EVIDENCE_VERSION = "contact-window/1"
IMAGE_WIDTH = 960
QUESTIONS = {
    "grip": (
        "At the selected moment, how is the worker grasping the object?",
        {
            "handle_grasp": "Grasping a handle or handhold",
            "object_grasp": "Grasping the object directly",
            "no_grasp": "Hands visibly free of a grasp",
            "unclear": "Cannot tell",
        },
    ),
    "armSupport": (
        "Is an elbow or forearm visibly resting on a separate support?",
        {
            "contact": "Resting contact is visible",
            "clear_gap": "Visibly clear of support",
            "unclear": "Cannot tell",
        },
    ),
    "visibility": (
        "How clearly does this view show the grip and the arm-support area?",
        {
            "clear": "Both areas are clear",
            "partial": "Only part of the evidence is clear",
            "unclear": "Neither area can be judged",
        },
    ),
}

INSTRUCTIONS = """You are inspecting a short sequence from ONE fixed viewpoint. Answer three separate visual questions
about the selected worker at the TARGET frame. Neighbouring images establish continuity; a full image and crop
at the same time are one observation. When an overlay is supplied it identifies the worker. Use raw pixels for
contact. Give one brief factual reason per answer and cite the numbered images that actually show it.

Grip: Look at fingers, the object boundary and the actual contact point. Choose handle_grasp when fingers wrap
around a distinct handle or enter a handhold; object_grasp when they grasp the object's body or edge directly.
Choose no_grasp only when both hands are visible and clearly free. If different grips are present, no handhold
can be distinguished, the object blocks the fingers, or another worker could account for the cue, choose unclear.
Do not infer a good or poor ergonomic coupling rating from the presence of a handle.

Arm support: Inspect the underside of each elbow/forearm and the edge of a separate chair, rest or work surface
before, during and after the target. Contact requires visible resting contact at target and on a nearby frame.
A hand touching an object, projected overlap, or a forearm merely aligned with a bench edge does not establish
resting contact. Choose clear_gap only if both relevant arms can be judged and are visibly free of support.
Choose unclear when contact cannot be distinguished from overlap or separation. Never estimate load transfer.

Visibility: Inspect occlusion, blur, cropping, contrast, perspective and worker identity. Clear means both targets
are visible clearly. Partial means relevant pixels are only partly usable. Unclear means neither target can be
judged from these images: in that case both contact answers must be unclear. State the specific obstruction.
Never fill a missing cue from posture, the task name, camera placement or ergonomics scores.
"""


class Answers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grip: Literal["handle_grasp", "object_grasp", "no_grasp", "unclear"]
    armSupport: Literal["contact", "clear_gap", "unclear"]
    visibility: Literal["clear", "partial", "unclear"]

    @model_validator(mode="after")
    def consistent_visibility(self):
        if self.visibility == "unclear" and (
            self.grip != "unclear" or self.armSupport != "unclear"
        ):
            raise ValueError(
                "When neither area can be judged, choose Cannot tell for both observations."
            )
        return self


class _Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    reason: str = Field(min_length=1, max_length=480)
    images: list[int] = Field(min_length=1, max_length=6)


class _Grip(_Finding):
    value: Literal["handle_grasp", "object_grasp", "no_grasp", "unclear"]


class _Support(_Finding):
    value: Literal["contact", "clear_gap", "unclear"]


class _Visibility(_Finding):
    value: Literal["clear", "partial", "unclear"]


class _Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    grip: _Grip
    armSupport: _Support
    visibility: _Visibility


class StaleVisualReview(ValueError):
    pass


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def predictor(config: AssistantConfig) -> str:
    return digest(
        [
            PROMPT_VERSION,
            INSTRUCTIONS,
            QUESTIONS,
            _Reading.model_json_schema(),
            config.scene_model,
            config.endpoint,
            config.dialect,
            config.scene_temperature,
            config.seed,
            config.context_tokens,
            *([config.model_digest] if config.model_digest else []),
        ]
    )


def run_key(session_id: str) -> str:
    sources = [
        (item["camera_index"], recording_hash(Path(item["path"]))) for item in cameras(session_id)
    ]
    return digest([EVIDENCE_VERSION, IMAGE_WIDTH, result_fingerprint(session_id), sources])


def _moments(session_id: str, person: str) -> list[tuple[int, float]]:
    return sorted(
        {
            (e.frame_number, e.timestamp_sec)
            for e in analysis._store(session_id).entries
            if e.person_identifier == person
            and e.frame_number is not None
            and e.timestamp_sec is not None
        }
    )


def sample_moments(session_id: str, person: str, limit: int = 8) -> list[dict]:
    """Time-stratified, reproducible moments independent of scores and AI output."""
    moments = _moments(session_id, person)
    count = min(limit, len(moments))
    return [
        {"frame": f, "sec": sec}
        for f, sec in (moments[((2 * i + 1) * len(moments)) // (2 * count)] for i in range(count))
    ]


@dataclass(frozen=True)
class Case:
    id: str
    run: str
    session_id: str
    frame: int
    person: str
    camera: int
    camera_mode: str
    method: str
    evidence: tuple[dict, ...]
    images: tuple[bytes, ...]

    def metadata(self):
        return {
            "caseId": self.id,
            "sessionId": self.session_id,
            "frame": self.frame,
            "person": self.person,
            "camera": self.camera,
            "cameraMode": self.camera_mode,
            "method": self.method,
        }


def get_case(session_id: str, frame: int, person: str, method: str, camera: int) -> Case:
    return _build_case(session_id, run_key(session_id), frame, person, method, camera)


@lru_cache(maxsize=16)
def _build_case(
    session_id: str, run: str, frame: int, person: str, method: str, camera: int
) -> Case:
    if camera not in {item["camera_index"] for item in cameras(session_id)}:
        raise SessionError("The requested camera recording is unavailable.")
    moments = _moments(session_id, person)
    sec = dict(moments).get(frame)
    if sec is None:
        raise SessionError("Choose a recorded frame for the selected worker.")
    # A gap in tracking must not be presented as a short continuous movement.
    close = [(f, t) for f, t in moments if abs(t - sec) <= 0.75]
    frames = {
        frame,
        *(min(close, key=lambda pair: abs(pair[1] - t))[0] for t in (sec - 0.45, sec + 0.45)),
    }
    evidence, images = [], []
    for f in sorted(frames):
        detail = analysis.frame_detail(session_id, f, method, person)
        if detail["frame"] != f or detail.get("person") != person:
            raise SessionError("The target worker could not be resolved at this moment.")
        refs = detail.get("cameras") or []
        if refs and not any(
            r["cameraIndex"] == camera and r.get("sourceFrame") is not None for r in refs
        ):
            if f == frame:
                raise SessionError("This moment has no recorded frame for the selected camera.")
            continue
        try:
            rendered, receipts = render(
                session_id, detail, method, camera, "hand_and_forearm", False, IMAGE_WIDTH, 2
            )
        except (DashboardDataError, OSError, SessionError) as exc:
            if f == frame:
                raise SessionError(f"Cannot open the selected evidence. {exc}") from exc
            continue
        for (_, jpeg), receipt in zip(rendered, receipts, strict=True):
            images.append(jpeg)
            evidence.append({**receipt, "imageIndex": len(images)})
    if not any(r["frame"] == frame for r in evidence):
        raise SessionError("No image is available at the selected moment.")
    mode = "multi" if analysis.session_overview(session_id).get("cameraCount", 0) > 1 else "single"
    case_id = digest(
        [run, session_id, frame, person, camera, [hashlib.sha256(j).hexdigest() for j in images]]
    )
    query = urlencode(
        {"caseId": case_id, "frame": frame, "person": person, "camera": camera, "method": method}
    )
    for item in evidence:
        item["imageUrl"] = (
            f"/api/sessions/{quote(session_id, safe='')}/focused-review/images/{item['imageIndex']}?{query}"
        )
    return Case(
        case_id,
        run,
        session_id,
        frame,
        person,
        camera,
        mode,
        method,
        tuple(evidence),
        tuple(images),
    )


def read_images(case: Case, config: AssistantConfig) -> dict:
    if not config.enabled:
        raise BackendError("The visual assistant is disabled in the local configuration.")
    messages = [{"role": "system", "content": INSTRUCTIONS}]
    for ref, jpeg in zip(case.evidence, case.images, strict=True):
        when = "TARGET" if ref["frame"] == case.frame else "Neighbour"
        messages.append(
            image_message(
                f"Image {ref['imageIndex']}. {when}. Frame {ref['frame']}, "
                f"{ref['sec']:.3f} s. {ref['view']} of the same worker and camera.",
                jpeg,
            )
        )
    messages.append(
        {
            "role": "user",
            "content": "Answer each question at the target. Cite target images and, "
            "for resting contact, a separate supporting time. Cite obstructed images when abstaining.",
        }
    )
    for attempt in range(2):
        response = complete(
            messages,
            config=config,
            model=config.scene_model,
            temperature=config.scene_temperature,
            schema=_Reading.model_json_schema(),
            max_tokens=900,
        )
        try:
            reading = _Reading.model_validate_json(response)
            values = Answers.model_validate({key: getattr(reading, key).value for key in QUESTIONS})
            answers = {}
            for key in QUESTIONS:
                finding = getattr(reading, key)
                if any(i < 1 or i > len(case.evidence) for i in finding.images):
                    raise ValueError("An image ID was not supplied.")
                evidence = [case.evidence[i - 1] for i in dict.fromkeys(finding.images)]
                frames = {r["frame"] for r in evidence}
                if case.frame not in frames:
                    raise ValueError("Cite a target image for each answer.")
                if (
                    key == "armSupport"
                    and values.armSupport == "contact"
                    and (
                        len(frames) < 2
                        or len({r["sourceFrame"] for r in evidence}) < 2
                        or len({r["sec"] for r in evidence}) < 2
                    )
                ):
                    raise ValueError(
                        "Resting contact requires a target and a neighbouring time; otherwise select unclear."
                    )
                answers[key] = {
                    "value": finding.value,
                    "reason": finding.reason,
                    "evidence": evidence,
                }
            return {
                "id": digest([case.id, predictor(config), answers]),
                "model": config.scene_model,
                "modelDigest": config.model_digest,
                "promptVersion": PROMPT_VERSION,
                "answers": answers,
            }
        except (ValidationError, ValueError) as exc:
            if attempt:
                raise BackendError(
                    "The model could not return a verifiable visual review. Retry this moment."
                ) from exc
            messages.append(
                {
                    "role": "user",
                    "content": f"Correct the structured reading: {str(exc)[:420]}. "
                    "Use only listed choices and the image IDs shown.",
                }
            )
    raise AssertionError("unreachable")


def data_path(session_id: str) -> Path:
    return Path(resolve_result_dir(session_id)) / "focused_review.sqlite3"
