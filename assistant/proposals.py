"""Condition proposals, and what happens when a person accepts one.

This is the only place the assistant touches anything that could reach a score,
and it is deliberately the dullest module in the package. A proposal is written
to its own file. Accepting one records a decision, with who-what-when and the
frames the model was looking at. Nothing is rescored here, and no existing result
is edited.

That last point is not caution for its own sake. A stored score has to match the
inputs it was computed from, or the result directory stops being evidence. So an
accepted condition becomes an *entered* value for the next run, and the operator
reruns the trial when they are ready.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from api.assessments import atomic_json
from api.sessions import SessionError, resolve_result_dir
from assistant.scene import PROPOSABLE, load_scene, valid_condition_value

LOGGER = logging.getLogger(__name__)

PROPOSALS_FILE = "vlm_proposals.json"
SCHEMA = "vlm-proposals/1.0"
_DECISION_LOCK = RLock()


def proposals_path(session_id: str) -> Path:
    return Path(resolve_result_dir(session_id)) / PROPOSALS_FILE


def _read(session_id: str) -> dict[str, Any]:
    try:
        path = proposals_path(session_id)
        if not path.is_file():
            return {"schema": SCHEMA, "sessionId": session_id, "decisions": {}}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"schema": SCHEMA, "sessionId": session_id, "decisions": {}}
    if not isinstance(value, dict):
        return {"schema": SCHEMA, "sessionId": session_id, "decisions": {}}
    value.setdefault("decisions", {})
    return value


def _write(session_id: str, document: dict[str, Any]) -> None:
    atomic_json(proposals_path(session_id), document)


def _proposal_id(card: dict, proposal: dict) -> str:
    refs = sorted(json.dumps(r, sort_keys=True) for r in proposal.get("evidence", []))
    identity = [card.get("fingerprint"), proposal.get("condition"), proposal.get("value"),
                refs, sorted(proposal.get("evidence_frames", []))]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _entered(document: dict) -> dict[str, Any]:
    if "enteredValues" in document:
        return dict(document["enteredValues"])
    # Upgrade older operator decisions without associating them with new footage.
    return {name: d.get("value") for name, d in document.get("decisions", {}).items()
            if d.get("status") == "accepted" and valid_condition_value(name, d.get("value"))}


def load_proposals(session_id: str) -> dict[str, Any]:
    """The scene card's proposals joined with any decision already recorded."""

    try:
        stored = _read(session_id)
    except SessionError:
        return {"available": False, "proposals": [], "decisions": {}}

    card = load_scene(session_id)
    decisions = stored.get("decisions") or {}
    proposals: list[dict[str, Any]] = []
    for item in (card or {}).get("condition_proposals") or []:
        condition = str(item.get("condition"))
        proposal_id = _proposal_id(card, item)
        decision = decisions.get(condition) or {}
        if decision.get("proposalId") != proposal_id:
            decision = {}
        proposals.append(
            {
                **item,
                "proposalId": proposal_id,
                "status": decision.get("status", "pending"),
                "decidedValue": decision.get("value"),
                "decidedAt": decision.get("at"),
                "note": decision.get("note"),
            }
        )
    return {
        "available": card is not None,
        "sessionId": session_id,
        "proposals": proposals,
        "decisions": decisions,
        "note": (
            "Accepting records the value as an operator-entered condition for the next "
            "analysis run. It does not alter the scores already computed."
        ),
    }


def _decide(
    session_id: str,
    condition: str,
    status: str,
    value: Any = None,
    note: str | None = None,
    proposal_id: str | None = None,
) -> dict[str, Any]:
    name = str(condition).strip()
    if name not in PROPOSABLE:
        raise SessionError(
            f"{name!r} is not a condition the assistant may propose. "
            f"Allowed: {', '.join(sorted(PROPOSABLE))}."
        )
    document = _read(session_id)
    card = load_scene(session_id)
    source = next((item for item in (card or {}).get("condition_proposals") or []
                   if item.get("condition") == name), None)
    current_id = _proposal_id(card, source) if card and source else None
    if proposal_id is not None and proposal_id != current_id:
        raise SessionError("This proposal changed. Refresh the scene and review its current evidence.")
    entered = _entered(document)
    if status == "accepted":
        if value is None:
            value = (source or {}).get("value")
        if not valid_condition_value(name, value):
            raise SessionError(f"A valid operator-confirmed value is required for {name!r}.")
        entered[name] = value
    else:
        previous = document.get("decisions", {}).get(name) or {}
        if previous.get("status") == "accepted" and previous.get("proposalId") == current_id:
            entered.pop(name, None)
    document["enteredValues"] = entered
    agreement = (source or {}).get("agreement") or {}
    document["decisions"][name] = {
        "status": status,
        "value": value,
        "note": note,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "proposedValue": (source or {}).get("value", (source or {}).get("value_range")),
        "confidence": (source or {}).get("confidence"),
        "rationale": (source or {}).get("rationale"),
        "evidenceFrames": (source or {}).get("evidence_frames") or [],
        "evidence": (source or {}).get("evidence") or [],
        "proposalId": current_id,
        "model": (card or {}).get("model"),
        # The decision record carries how well the readings agreed, so a value
        # accepted over a split reading stays visibly different later from one
        # every pass produced.
        "agreement": agreement,
        "wasContested": bool((source or {}).get("contested")),
    }
    document["schema"] = SCHEMA
    document["sessionId"] = session_id
    document["updatedAt"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write(session_id, document)
    LOGGER.info("Condition %s %s for %s", name, status, session_id)
    return load_proposals(session_id)


def accept_proposal(
    session_id: str, condition: str, value: Any = None, note: str | None = None,
    proposal_id: str | None = None,
) -> dict[str, Any]:
    """Accept a proposal as an operator-entered value for the next run."""

    with _DECISION_LOCK:
        return _decide(session_id, condition, "accepted", value, note, proposal_id)


def reject_proposal(session_id: str, condition: str, note: str | None = None,
                    proposal_id: str | None = None) -> dict[str, Any]:
    with _DECISION_LOCK:
        return _decide(session_id, condition, "rejected", None, note, proposal_id)


def entered_conditions(session_id: str) -> dict[str, Any]:
    """Accepted values, in the shape ``condition_resolution`` reads entered ones.

    The analysis runner can merge this into its observed-conditions input so a
    rerun picks up what the ergonomist confirmed. Only accepted decisions appear.
    """

    return {name: value for name, value in _entered(_read(session_id)).items()
            if valid_condition_value(name, value)}


__all__ = [
    "PROPOSALS_FILE",
    "accept_proposal",
    "entered_conditions",
    "load_proposals",
    "proposals_path",
    "reject_proposal",
]
