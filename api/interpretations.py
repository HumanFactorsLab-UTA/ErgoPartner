"""Saved, reviewer-selectable observations from a task's visual analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any

from fastapi import HTTPException

from api.assessments import atomic_json, result_fingerprint
from api.sessions import resolve_result_dir

LOCK = RLock()


def storage_path(session_id: str) -> Path:
    from api.assessments import WORKFLOW_ROOT

    result = Path(resolve_result_dir(session_id)).resolve()
    key = hashlib.sha256(str(result).encode()).hexdigest()
    return WORKFLOW_ROOT / "interpretations" / f"{key}.json"


def saved(session_id: str) -> list[dict[str, Any]]:
    path = storage_path(session_id)
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))["interpretations"]
    fingerprint = result_fingerprint(session_id)
    return [{**record, "stale": record["resultFingerprint"] != fingerprint} for record in records]


def save(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_json(
        path,
        {
            "interpretations": [
                {key: value for key, value in item.items() if key != "stale"} for item in records
            ]
        },
    )


def select(
    session_id: str, record_id: str, observation_id: str, include: bool
) -> list[dict[str, Any]]:
    with LOCK:
        path = storage_path(session_id)
        records = saved(session_id)
        record = next((item for item in records if item["id"] == record_id), None)
        if record is None or not any(
            item["id"] == observation_id for item in record["observations"]
        ):
            raise HTTPException(404, "Saved observation not found.")
        if record["stale"]:
            raise HTTPException(409, "This observation belongs to an earlier analysis.")
        selected = [key for key in record["selected"] if key != observation_id]
        if include:
            selected.append(observation_id)
        record["selected"] = selected
        save(path, records)
        return records


def for_report(session_id: str) -> list[dict[str, Any]]:
    result = []
    for record in saved(session_id):
        included = [item for item in record["observations"] if item["id"] in record["selected"]]
        if included and not record["stale"]:
            result.append({**record, "observations": included})
    return result
