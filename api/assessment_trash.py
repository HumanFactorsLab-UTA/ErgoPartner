"""Reversible, library-scoped removal of participant groups, tasks, and loose drafts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from dashboard.cache import clear_all_caches


class TrashSelection(BaseModel):
    kind: Literal["draft", "task", "participant"]
    draftId: str | None = None
    participantKey: str | None = None
    taskFolder: str | None = None


def trash_path(root: str) -> Path:
    from api.assessments import WORKFLOW_ROOT

    canonical = os.path.normcase(str(Path(root).resolve()))
    return WORKFLOW_ROOT / "trash" / f"{hashlib.sha256(canonical.encode()).hexdigest()}.json"


def items(root: str) -> list[dict[str, Any]]:
    path = trash_path(root)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return []
    return _read_items(path, stat.st_mtime_ns, stat.st_ino, stat.st_size)


@lru_cache(maxsize=32)
def _read_items(path: Path, modified: int, inode: int, size: int) -> list[dict[str, Any]]:
    # Atomic replacement changes the signature, including after a restore or another process's write.
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def task_hidden(entries: list[dict], key: str | None, folder: str | None = None) -> bool:
    if not key:
        return False
    # Catalog identities refer to filesystem entries. Windows resolves case variants identically.
    key = os.path.normcase(key)
    folder = os.path.normcase(folder) if folder else None
    return any(
        os.path.normcase(item.get("participantKey") or "") == key
        and (
            item["kind"] == "participant"
            or (item["kind"] == "task" and os.path.normcase(item["taskFolder"]) == folder)
        )
        for item in entries
    )


def draft_location(draft: dict) -> tuple[str | None, str | None]:
    reference = draft.get("published") or draft.get("existing") or {}
    key = reference.get("participantKey") or draft.get("participantKey")
    setup = draft["setup"]
    if not key and setup.get("studyFolder") and setup.get("participantId"):
        key = f"{setup['studyFolder']}::{setup['participantId']}"
    return key, reference.get("taskFolder")


def draft_hidden(draft: dict) -> bool:
    entries = items(draft["studiesRoot"])
    return any(item.get("draftId") == draft["id"] for item in entries) or task_hidden(
        entries, *draft_location(draft)
    )


def filter_catalog(root: str, contexts: list[dict]) -> list[dict]:
    entries = items(root)
    if not entries:
        return contexts
    return [
        {
            **ctx,
            "tasks": [
                t for t in ctx["tasks"] if not task_hidden(entries, ctx["key"], t["folder_name"])
            ],
        }
        for ctx in contexts
        if not task_hidden(entries, ctx["key"])
    ]


def move(root: str, choice: TrashSelection, capture_owner: str | None) -> dict:
    from api import assessments as store
    from api import jobs
    from api.sessions import find_context, find_trial

    entries = items(root)
    key, folder = choice.participantKey, choice.taskFolder
    draft_id = choice.draftId
    if choice.kind == "draft":
        draft = store.load(draft_id or "")
        if Path(draft["studiesRoot"]).resolve() != Path(root).resolve():
            raise HTTPException(404, "This draft belongs to another studies folder.")
        if published := draft.get("published"):
            return move(root, TrashSelection(kind="task", **published), capture_owner)
        key, folder = draft_location(draft)
        label = draft["setup"]["taskName"] or "Untitled assessment"
        affected = {draft["id"]}
        count = 1
    else:
        if not key or (choice.kind == "task" and not folder):
            raise HTTPException(422, "Select a participant and its task to move to Trash.")
        context = find_context(root, key)
        draft_id = None
        drafts = store.list_drafts(root)
        if choice.kind == "participant":
            folder = None
            label = f"Participant {context['participant_id']}"
            selected = [d for d in drafts if draft_location(d)[0] == key]
            count = len(context["tasks"]) + sum(not d.get("published") for d in selected)
        else:
            label = find_trial(context, folder)["name"]
            selected = [d for d in drafts if draft_location(d) == (key, folder)]
            count = 1
        affected = {d["id"] for d in selected}
        if jobs.analysis_jobs.working_on(key, folder):
            raise HTTPException(
                409, "Wait for analysis of this task to finish before moving it to Trash."
            )

    if affected & store.UPLOADING or capture_owner in affected:
        raise HTTPException(
            409, "Finish the upload or stop the active cameras before moving this item to Trash."
        )
    entry = {
        "id": uuid.uuid4().hex,
        "kind": choice.kind,
        "label": label,
        "deletedAt": store.now(),
        "participantKey": key,
        "taskFolder": folder,
        "draftId": draft_id,
        "taskCount": count,
    }
    store.atomic_json(trash_path(root), {"items": [entry, *entries]})
    clear_all_caches()
    return entry


def restore(root: str, identifier: str) -> dict:
    from api.assessments import atomic_json

    entries = items(root)
    selected = next((item for item in entries if item["id"] == identifier), None)
    if selected is None:
        raise HTTPException(404, "This item is no longer in Trash.")
    remaining = [item for item in entries if item["id"] != identifier]
    if task_hidden(remaining, selected.get("participantKey"), selected.get("taskFolder")):
        raise HTTPException(409, "Restore the containing assessment first.")
    atomic_json(trash_path(root), {"items": remaining})
    clear_all_caches()
    return {"restored": True}
