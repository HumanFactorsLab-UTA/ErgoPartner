"""Generate a task interpretation from frame-linked visual evidence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from api.assessments import result_fingerprint
from api.interpretations import LOCK, save, saved, storage_path

_GENERATING: set[Path] = set()


def _plain(text: str) -> str:
    return re.sub(r"\s*\u2014\s*", ", ", text).strip(" ,")


def _observations(scene: dict[str, Any]) -> list[dict[str, Any]]:
    inference = scene.get("task_inference") or {}
    items = list(scene.get("findings") or [])
    items.sort(
        key=lambda item: (
            item.get("category") not in {"phase", "contact", "other"},
            -item.get("passes", 0),
        )
    )
    if inference.get("activity_class") not in (None, "unknown") and inference.get("summary"):
        items.append(
            {
                "fact": inference["summary"],
                "evidence": inference.get("evidence"),
                "category": "activity",
                "visibility": "sampled views",
            }
        )
    observations = []
    for item in items:
        evidence = [ref for ref in item.get("evidence") or [] if type(ref.get("frame")) is int]
        fact = _plain(str(item.get("fact") or ""))
        if evidence and fact:
            observations.append(
                {
                    "id": str(len(observations) + 1),
                    "text": fact,
                    "evidence": evidence,
                    "category": item.get("category", "observation"),
                    "visibility": item.get("visibility", "unknown"),
                }
            )
    return observations


def generate(session_id: str, requested_model: str | None) -> list[dict[str, Any]]:
    from assistant.models import effective_config
    from assistant.scene import build_scene, load_scene

    path = storage_path(session_id)
    with LOCK:
        if path in _GENERATING:
            raise HTTPException(409, "An interpretation of this task is already being generated.")
        _GENERATING.add(path)
    try:
        fingerprint = result_fingerprint(session_id)
        config = effective_config(session_id, requested_model=requested_model)
        scene = load_scene(session_id, config=config) or build_scene(session_id, config=config)
        identity = hashlib.sha256(f"{fingerprint}:{scene['fingerprint']}".encode()).hexdigest()
        observations = _observations(scene)
        unresolved = any(item["visibility"] in {"partial", "occluded"} for item in observations)
        record = {
            "id": identity,
            "createdAt": scene["builtAt"],
            "model": scene["model"],
            "modelDigest": scene.get("modelDigest") or config.model_digest,
            "resultFingerprint": fingerprint,
            "observations": observations,
            "selected": [],
            "limitations": (
                f"The model reviewed {len(scene['frames_sampled'])} sampled frames. "
                "These views cannot establish what happened between samples."
                + (" Some details were partly visible or obscured." if unresolved else "")
            ),
            "followUp": (
                "Review the cited footage and confirm relevant observations before including them in the report."
                if observations
                else "Inspect the source footage. The model returned no observations linked to a visible frame."
            ),
        }
        with LOCK:
            if fingerprint != result_fingerprint(session_id) or storage_path(session_id) != path:
                raise HTTPException(
                    409,
                    "The assessment changed. Generate an interpretation of the updated results.",
                )
            records = saved(session_id)
            if not any(item["id"] == identity for item in records):
                save(path, [record, *records])
            return saved(session_id)
    finally:
        with LOCK:
            _GENERATING.discard(path)
