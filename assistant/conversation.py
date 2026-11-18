"""One turn of conversation: the tool loop, the citations, the transcript.

The loop is deliberately visible. Every tool call is emitted as an event before
it runs, so the front end can show "reading the REBA timeline…" and "looking at
frame 412…". That is not a loading spinner dressed up -- it is the audit trail,
live, and it is the reason a reviewer can believe the numbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api import analysis as analysis_api
from api.assessments import result_fingerprint
from api.sessions import SessionError, resolve_result_dir
from assistant.audit import AuditReport, Evidence, audit_answer
from assistant.backend import BackendError, image_message, stream_completion
from assistant.brief import analysis_brief, session_brief
from assistant.grounding import Citation, TurnEvidence
from assistant.models import effective_config
from assistant.prompts import PROMPT_VERSION, system_prompt, visual_intent, with_reminder
from assistant.tools import MAX_IMAGES_PER_TURN, run_tool, tool_schemas
from assistant.visuals import cameras
from dashboard.cache import ttl_cache
from dashboard.result_store import open_result_store

LOGGER = logging.getLogger(__name__)

#: ``<<CITE frame=412 t=44.2 method=reba region=trunk>>`` -- stripped before
#: display and turned into the chips that drive the workspace.
_CITE = re.compile(r"<<CITE\s+([^>]*)>>")
_FIELD = re.compile(r"(\w+)=([^\s]+)")

#: How many turns of history to replay. Frames already in the transcript keep
#: their text receipts but their images are dropped after the turn that fetched
#: them, so a long conversation does not grow a hundred images in context.
HISTORY_TURNS = 8


@dataclass(slots=True)
class TurnResult:
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    frames_seen: list[int] = field(default_factory=list)
    audit: AuditReport = field(default_factory=AuditReport)
    selection: dict = field(default_factory=dict)
    images: list[dict] = field(default_factory=list)
    fingerprint: str | None = None
    error: str | None = None


@ttl_cache(ttl=120, copy_result=False)
def valid_frames(session_id: str) -> frozenset[int]:
    """Every frame number this session actually scored.

    A citation is a control: clicking it moves the timeline, the replay and the
    frame panel together. So a citation naming a frame that does not exist is not
    a cosmetic slip, it is a button that breaks the workspace -- and models do
    invent them, including by copying the illustrative frame number out of their
    own instructions. Citations are checked against this set and dropped if they
    fail.
    """

    try:
        store = open_result_store(Path(resolve_result_dir(session_id)) / "frame_results.jsonl")
        return frozenset(
            int(entry.frame_number) for entry in store.entries if entry.frame_number is not None
        )
    except (SessionError, OSError, ValueError):
        return frozenset()


def _seconds(value: Any) -> float | None:
    """Read ``44.2`` or ``0:44`` or ``1:02:03`` as seconds; None when unusable."""

    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        total = 0.0
        for part in raw.split(":"):
            total = total * 60 + float(part)
        return round(total, 3) if math.isfinite(total) and total >= 0 else None
    except (TypeError, ValueError):
        return None


def parse_citations(text: str, session_id: str | None = None) -> tuple[str, list[Citation]]:
    """Strip the citation footer and return it as structured data.

    Passing ``session_id`` validates each frame against the recording; without it
    the citations are returned as written, which is what the parser tests want.
    """

    citations: list[Citation] = []
    for match in _CITE.finditer(text):
        fields = dict(_FIELD.findall(match.group(1)))
        try:
            frame = int(fields["frame"])
            camera = int(fields["camera"]) if "camera" in fields else None
        except (KeyError, TypeError, ValueError):
            continue
        method = fields.get("method", "").lower()
        kind = fields.get("kind", "").lower()
        region = fields.get("region")
        citations.append(
            Citation(
                frame=frame,
                sec=_seconds(fields.get("t")),
                camera=camera,
                method=method if method in {"reba", "rula"} else None,
                region=region if region and region.lower() not in {"none", "null"} else None,
                person=fields.get("person"),
                view=fields.get("view"),
                kind=kind if kind in {"computed", "observed"} else None,
            )
        )

    cleaned = _CITE.sub("", text).strip()

    known = valid_frames(session_id) if session_id else frozenset()
    # Duplicate citations are common when a frame is discussed twice; keep the
    # first of each so the chip row does not repeat itself.
    unique: list[Citation] = []
    seen: set[tuple] = set()
    for citation in citations:
        if session_id and citation.frame not in known:
            LOGGER.warning(
                "Dropped a citation to frame %s, which %s did not score",
                citation.frame,
                session_id,
            )
            continue
        key = (
            citation.frame,
            citation.method,
            citation.region,
            citation.camera,
            citation.person,
            citation.kind,
            citation.view,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(citation)
    return cleaned, unique


def _tool_arguments(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _current_fingerprint(session_id: str) -> str | None:
    try:
        digest = hashlib.sha256(result_fingerprint(session_id).encode())
        for source in cameras(session_id):
            path = Path(source["path"])
            stat = path.stat()
            digest.update(
                json.dumps(
                    [
                        source["camera_index"],
                        str(path),
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    ]
                ).encode()
            )
        return digest.hexdigest()
    except (SessionError, OSError, ValueError):
        return None


def _selection_context(session_id: str, selection: dict | None) -> dict:
    if not selection:
        return {}
    scope = {k: v for k, v in selection.items() if v is not None}
    overview = analysis_api.session_overview(session_id)
    people = overview.get("people") or []
    if scope.get("person") is not None and scope["person"] not in people:
        raise SessionError("The selected worker is unavailable in this analysis.")
    if len(people) == 1:
        scope.setdefault("person", people[0])
    count = overview.get("cameraCount") or 0
    if scope.get("camera") is not None and not 0 <= scope["camera"] < count:
        raise SessionError("The selected camera is unavailable in this analysis.")
    scope["cameraCount"] = count
    if count == 1:
        scope["camera"] = 0
    # Times displayed by a scrubber can fall between analyzed samples.
    scope.pop("timestampSec", None)
    if scope.get("frame") is not None:
        detail = analysis_api.frame_detail(
            session_id, scope["frame"], scope["method"], scope.get("person")
        )
        if detail["frame"] == scope["frame"]:
            scope["timestampSec"] = detail.get("timestampSec")
        else:
            scope["frameStatus"] = "No exact scored record at this selection"
    return scope


def build_messages(
    session_id: str, history: list[dict[str, Any]], question: str, selection: dict | None = None
) -> list[dict[str, Any]]:
    """System prompt, brief, prior turns, then the new question."""

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt()}]
    try:
        brief = (
            session_brief(session_id, selection["method"], selection.get("person"))
            if selection
            else session_brief(session_id)
        )
    except SessionError as exc:
        brief = f"=== ASSESSMENT BRIEF ===\nUnavailable: {exc}"
    messages.append(
        {
            "role": "user",
            "content": (
                f"{brief}\n\n"
                "That is the assessment under review. Answer questions about it using the "
                "tools; do not restate this brief back to me."
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "Understood. I have the assessment. What would you like to know?",
        }
    )
    for entry in history[-HISTORY_TURNS * 2 :]:
        role = entry.get("role")
        content = entry.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    # The reminder rides on the question rather than the history, so replayed
    # turns stay clean and only the live one carries it.
    if selection:
        messages.append(
            {
                "role": "user",
                "content": "Current dashboard selection from the application: "
                + json.dumps(selection)
                + ". Resolve ‘this’ or ‘here’ against this selection. An explicitly requested different "
                "moment or method takes precedence. Read the exact frame for frame measurements and "
                "inspect actual images for visual findings. Only cite tools used in this turn.",
            }
        )
    messages.append({"role": "user", "content": with_reminder(question)})
    return messages


def _requested_measurement(question: str, selection: dict) -> dict | None:
    """Resolve a request for a specific moment, including one part of a mixed question."""
    if not re.search(r"\b(?:scores?|angles?|degrees?)\b", question, re.I):
        return None
    frames = set(re.findall(r"\bframe\s*[#:=]?\s*(\d+)", question, re.I))
    if len(frames) == 1:
        frame = int(frames.pop())
    elif not frames and re.search(
        r"\b(?:here|(?:selected|current|same|this) (?:frame|moment|time))\b", question, re.I
    ):
        frame = selection.get("frame")
    else:
        return None
    methods = set(re.findall(r"\b(?:rula|reba)\b", question.lower()))
    if frame is None or len(methods) > 1:
        return None
    return {
        "frame": frame,
        "method": next(iter(methods), selection.get("method", "reba")),
        "person": selection.get("person"),
    }


def _evidence_fallback(
    grounding: TurnEvidence, selection: dict, visual: bool, target: dict | None = None
):
    visual_gap = "I could not verify the requested visual detail from the available footage."
    if visual and target is None:
        return visual_gap, []
    scope = target or selection
    for ref, row in reversed(grounding.measurements.items()):
        if (
            ref.frame == scope.get("frame", ref.frame)
            and ref.method == scope.get("method", ref.method)
            and scope.get("person") in (None, ref.person)
            and "score" in row
        ):
            side = row.get("selectedSide")
            side_text = f" ({side} side)" if side in {"left", "right"} else ""
            score = f"is {row['score']}{side_text}" if row["score"] is not None else "was withheld"
            text = f"The recorded {ref.method.upper()} score at frame {ref.frame} {score}. [computed · frame {ref.frame}]"
            return text + " " + (
                visual_gap if visual else "I could not verify the additional explanation."
            ), [ref]
    return (
        visual_gap if visual else "The available records did not verify the requested finding."
    ), []


def run_turn(
    session_id: str,
    question: str,
    history: list[dict[str, Any]] | None = None,
    *,
    selection: dict | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one question to completion, yielding events as they happen.

    Events: ``{"type": "tool", ...}``, ``{"type": "delta", "text": ...}``,
    ``{"type": "done", ...}``, ``{"type": "error", "detail": ...}``.
    """

    config = effective_config(session_id)
    try:
        selection = _selection_context(session_id, selection)
    except SessionError as exc:
        yield {"type": "error", "detail": str(exc)}
        return
    messages = build_messages(session_id, history or [], question, selection)
    schemas = tool_schemas()
    result = TurnResult(selection=selection, fingerprint=_current_fingerprint(session_id))
    grounding = TurnEvidence()
    repair_requested = False
    images_used = 0

    # A user's proposed angle, a prior answer and an example in the system
    # prompt can all contain numbers. None can authenticate itself.
    evidence = Evidence()
    try:
        brief = (
            analysis_brief(session_id, selection["method"], selection.get("person"))
            if selection
            else analysis_brief(session_id)
        )
        evidence.absorb_text(brief)
    except SessionError:
        pass

    target = _requested_measurement(question, selection)
    if target:
        yield {"type": "tool", "name": "frame", "arguments": target, "status": "running"}
        snapshot = run_tool("frame", session_id, target, selection=selection)
        # Include only receipt fields actually delivered to the model.
        compact = {
            k: v
            for k, v in snapshot.content.items()
            if k
            in {
                "error",
                "method",
                "frame",
                "timestampSec",
                "person",
                "score",
                "selectedSide",
                "angles",
                "sides",
                "drivers",
                "quality",
            }
        }
        if "error" not in compact:
            grounding.record(compact, selection)
            evidence.absorb(compact)
        messages.insert(
            -1,
            {
                "role": "user",
                "content": "The application retrieved the exact requested frame: "
                + json.dumps(compact, default=str)
                + ". Use this as computed evidence. "
                "Inspect the image separately for any visual part of the question.",
            },
        )
        result.tool_calls.append({"name": "frame", "arguments": target})
        yield {
            "type": "tool",
            "name": "frame",
            "arguments": target,
            "status": "done",
            "frames": snapshot.frames,
            "images": 0,
        }

    try:
        for _ in range(config.max_tool_iterations):
            answer: list[str] = []
            calls: list[dict[str, Any]] = []

            for chunk in stream_completion(messages, schemas, config=config):
                if chunk.text:
                    answer.append(chunk.text)
                    # The complete claim and its evidence are checked before display.
                if chunk.finished:
                    calls = chunk.tool_calls

            drafted = "".join(answer)
            if not calls:
                clean, footer = parse_citations(drafted)
                result.citations, problems = grounding.citations(clean, footer, selection)
                problems.extend(grounding.check_measurements(clean, result.citations))
                untraced = audit_answer(clean, evidence).untraced
                if untraced:
                    problems.append(
                        "Retrieve or omit unsupported figures: "
                        + ", ".join(c.written for c in untraced[:5])
                    )
                if (
                    target
                    and "error" not in compact
                    and not any(
                        c.frame == target["frame"]
                        and c.method == target["method"]
                        and c.kind == "computed"
                        for c in result.citations
                    )
                ):
                    problems.append(
                        "Also answer the measurement question using the exact requested frame receipt and computed citation."
                    )
                needs_image = visual_intent(question)
                if needs_image and not result.images:
                    problems.append("Inspect the relevant source images with look_at or watch.")
                if (
                    needs_image
                    and result.images
                    and not any(c.kind == "observed" for c in result.citations)
                ):
                    problems.append("Cite the images you inspected, including frame and camera.")
                if problems and not repair_requested:
                    repair_requested = True
                    messages.extend(
                        [
                            {"role": "assistant", "content": drafted},
                            {
                                "role": "user",
                                "content": "Verify the reply before display. "
                                + " ".join(problems[:4])
                                + " Use the available tools.",
                            },
                        ]
                    )
                    continue
                if problems:
                    clean, result.citations = _evidence_fallback(
                        grounding, selection, needs_image, target
                    )
                result.text = clean
                break

            # Replay the tool request so the model can associate each receipt with its call.
            messages.append(
                {
                    "role": "assistant",
                    "content": drafted,
                    "tool_calls": [
                        {
                            "id": call["id"] or f"call_{index}",
                            "type": "function",
                            "function": call["function"],
                        }
                        for index, call in enumerate(calls)
                    ],
                }
            )

            pending_images: list[dict[str, Any]] = []
            for index, call in enumerate(calls):
                name = str(call["function"]["name"])
                arguments = _tool_arguments(call["function"]["arguments"])
                yield {"type": "tool", "name": name, "arguments": arguments, "status": "running"}

                outcome = run_tool(
                    name,
                    session_id,
                    arguments,
                    selection=selection,
                    remaining_images=MAX_IMAGES_PER_TURN - images_used,
                )
                result.tool_calls.append({"name": name, "arguments": arguments})
                receipt = json.dumps(outcome.content, default=str)[:12000]
                if (
                    name not in {"scene_card", "method_reference"}
                    and "error" not in outcome.content
                ):
                    evidence.absorb_text(receipt)
                    if (
                        name not in {"look_at", "watch"}
                        and len(json.dumps(outcome.content, default=str)) <= 12000
                    ):
                        grounding.record(outcome.content, selection)
                messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "tool_call_id": call["id"] or f"call_{index}",
                        "content": receipt,
                    }
                )
                for (caption, jpeg), reference in zip(
                    outcome.images, outcome.image_refs, strict=True
                ):
                    if images_used >= MAX_IMAGES_PER_TURN:
                        break
                    images_used += 1
                    result.frames_seen.append(reference["frame"])
                    result.images.append(reference)
                    grounding.image(reference)
                    pending_images.append(image_message(caption, jpeg))

                yield {
                    "type": "tool",
                    "name": name,
                    "arguments": arguments,
                    "status": "done",
                    "frames": outcome.frames,
                    "images": len(outcome.images),
                }

            messages.extend(pending_images)
            if pending_images:
                messages.append(
                    {
                        "role": "user",
                        "content": "Those are the frames you asked for. Continue your answer, "
                        "tagging anything you take from them as [observed].",
                    }
                )
        else:
            # A final unverified generation would bypass the same evidence checks.
            result.text, result.citations = _evidence_fallback(
                grounding, selection, visual_intent(question), target
            )

    except BackendError as exc:
        LOGGER.warning("Assistant turn failed: %s", exc)
        yield {"type": "error", "detail": str(exc)}
        return
    except Exception as exc:  # noqa: BLE001 - the stream must always close cleanly
        LOGGER.exception("Assistant turn crashed")
        yield {"type": "error", "detail": f"The assistant failed: {exc}"}
        return

    if result.fingerprint and result.fingerprint != _current_fingerprint(session_id):
        yield {
            "type": "error",
            "detail": "This assessment was reanalyzed during the reply. Open its latest results and ask again.",
        }
        return
    result.audit = audit_answer(result.text, evidence)
    if config.log_conversations and result.fingerprint:
        _log_turn(session_id, question, result, config.describe())
    yield {
        "type": "done",
        "text": result.text,
        "citations": [item.payload() for item in result.citations],
        "imagesReviewed": result.images,
        "toolCalls": result.tool_calls,
        "framesSeen": sorted(set(result.frames_seen)),
        "audit": result.audit.payload(),
    }


def _log_turn(
    session_id: str, question: str, result: TurnResult, provenance: dict[str, Any]
) -> None:
    """Append the turn to the session's transcript.

    Reproducibility is the product here as much as the answer is: model, prompt
    version, temperature, and the full tool trace go down with every exchange.
    """

    try:
        path = Path(resolve_result_dir(session_id)) / "assistant_transcript.jsonl"
        entry = {
            "id": uuid.uuid4().hex[:12],
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "promptVersion": PROMPT_VERSION,
            "resultFingerprint": result.fingerprint,
            "provenance": provenance,
            "selection": result.selection,
            "imagesReviewed": result.images,
            "question": question,
            "answer": result.text,
            "citations": [item.payload() for item in result.citations],
            "toolCalls": result.tool_calls,
            "audit": result.audit.payload(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except (OSError, UnicodeError, SessionError) as exc:
        LOGGER.warning("Could not write the assistant transcript: %s", exc)


def load_transcript(session_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Prior exchanges for this session, oldest first."""

    try:
        path = Path(resolve_result_dir(session_id)) / "assistant_transcript.jsonl"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8") as handle:
            lines = handle.readlines()
    except (OSError, UnicodeError, SessionError):
        return []
    entries: list[dict[str, Any]] = []
    current = _current_fingerprint(session_id)
    for line in lines[-int(limit) :]:
        try:
            turn = json.loads(line)
            if not isinstance(turn, dict) or not all(
                isinstance(turn.get(k), str) for k in ("question", "answer")
            ):
                continue
            if not current or turn.get("resultFingerprint") != current:
                # Preserve older transcripts on disk. Only replay turns bound to this analysis.
                continue
            entries.append(turn)
        except json.JSONDecodeError:
            continue
    return entries


__all__ = [
    "Citation",
    "TurnResult",
    "build_messages",
    "load_transcript",
    "parse_citations",
    "run_turn",
]
