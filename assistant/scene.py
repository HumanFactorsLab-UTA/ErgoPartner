"""The scene analyst: a measured visual reading of a recording, cached to disk.

This is the half of the feature that is worth something scientifically. The pose
model sees 33 landmarks; it cannot see the carton, whether the carton has cut-out
handles, or the bench a forearm is resting on. Those are precisely the fields
``ergonomics.condition_resolution.NOT_OBSERVABLE_FROM_VIDEO`` has to fill from a
profile, and a profile guess is the weakest input in the whole assessment.

Two design decisions make the reading defensible rather than merely plausible:

**The frames are chosen by posture, not by clock.** :mod:`assistant.sampling`
picks the set that best represents the angle cloud the pipeline already measured,
so a repetitive task does not spend its whole budget on the same posture. The
achieved coverage is recorded in the card.

**The reading is repeated and the agreement is measured.** Each pass sees an
overlapping but different frame set. A finding that survives every pass is
reported as unanimous; one that moves is marked contested and the interface will
not let it be accepted quietly. Self-reported confidence is kept, but it is no
longer the only thing a reviewer has to go on.

What this produces are *proposals*. They are never applied to a score. An
ergonomist accepts one, and only then does the value become an entered condition.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api.sessions import SessionError, resolve_result_dir
from assistant.backend import BackendError, complete, image_message
from assistant.config import AssistantConfig
from assistant.consistency import (
    merge_proposals,
    merge_task_inference,
    merge_workstation,
)
from assistant.models import effective_config, require_vision
from assistant.prompts import PROMPT_VERSION, scene_prompt
from assistant.sampling import Sample, coverage, representative_frames
from assistant.visuals import cameras

LOGGER = logging.getLogger(__name__)

SCENE_SCHEMA = "vlm-scene/3"
SCENE_FILE = "vlm_scene.json"

#: Conditions a visual reading may propose. Anything else the model invents is
#: dropped: the point of the authority ladder is that only these were missing.
PROPOSABLE = frozenset(
    {
        "coupling",
        "load_kg",
        "shock_or_rapid_force",
        "left_arm_supported_or_leaning",
        "right_arm_supported_or_leaning",
        "left_arm_supported_or_gravity_assisted",
        "right_arm_supported_or_gravity_assisted",
    }
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_PHYSICAL_UNIT = re.compile(
    r"\d\s*(?:mm|cm|m|meters?|metres?|ft|feet|foot|inches|inch|kg|kilograms?|lbs?|pounds?|degrees?)\b"
    r"|\d\s*°|\b(?:RULA|REBA)\D{0,12}\d",
    re.IGNORECASE,
)


def scene_path(session_id: str) -> Path:
    return Path(resolve_result_dir(session_id)) / SCENE_FILE


def _fingerprint(session_id: str, config: Any) -> str:
    root = Path(resolve_result_dir(session_id))
    paths = [root / name for name in ("frame_results.jsonl", "video_summary.json", "run_config.json")]
    paths += [Path(cam["path"]) for cam in cameras(session_id)]
    stamps = []
    for path in paths:
        stat = path.stat()
        stamps.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    signature = [stamps, SCENE_SCHEMA, PROMPT_VERSION, config.prompt_version, config.scene_model,
                 config.endpoint, config.seed, config.scene_temperature, config.budget("scene"),
                 config.scene_passes, config.frame_width]
    if config.model_digest:
        signature.append(config.model_digest)
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def load_scene(session_id: str, *, config: AssistantConfig | None = None) -> dict[str, Any] | None:
    """The cached scene card, or None when it has not been built."""

    try:
        path = scene_path(session_id)
        if not path.is_file():
            return None
        config = config or effective_config(session_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        if (config.model_issue and not config.model_digest and value.get("model") == config.scene_model
                and value.get("modelDigest")):
            from dataclasses import replace
            config = replace(config, model_digest=value["modelDigest"])
        fingerprint = _fingerprint(session_id, config)
        if value.get("fingerprint") != fingerprint:
            path = path.parent / "assistant_scenes" / f"{fingerprint}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SessionError):
        return None
    if isinstance(value, dict) and value.get("schema") == SCENE_SCHEMA and value.get("fingerprint") == fingerprint:
        return value
    return None


def valid_condition_value(condition: str, value: Any) -> bool:
    if condition == "load_kg":
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    if condition == "coupling":
        return isinstance(value, str) and value in {"good", "fair", "poor", "unacceptable"}
    return condition in PROPOSABLE and isinstance(value, bool)


def _clean_proposals(raw: Any, allowed_frames: set[int]) -> list[dict[str, Any]]:
    """Keep only well-formed proposals about conditions that were ever missing."""

    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        condition = str(item.get("condition") or "").strip()
        value = item.get("value")
        if condition in {"load_kg", "shock_or_rapid_force"} or not valid_condition_value(condition, value):
            LOGGER.info("Dropping scene proposal for unproposable condition %r", condition)
            continue
        cited = item.get("evidence_frames")
        if not isinstance(cited, list):
            continue
        frames = list(dict.fromkeys(frame for frame in cited
                                    if type(frame) is int and frame in allowed_frames))
        if not frames:
            continue
        confidence = str(item.get("confidence") or "low").lower()
        cleaned.append({
            "condition": condition,
            "value": value,
            "confidence": confidence if confidence in {"low", "medium", "high"} else "low",
            "rationale": str(item.get("rationale") or "").strip(),
            "evidence_frames": frames[:6],
        })
    return cleaned


class _Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Activity(_Schema):
    activity_class: Literal["manual_lifting", "overhead_reaching", "bench_assembly", "carrying",
                            "pushing_pulling", "seated_work", "other", "unknown"]
    evidence_images: list[int]


class _Finding(_Schema):
    fact: str = Field(min_length=1)
    visibility: Literal["clear", "partial", "occluded"]
    evidence_images: list[int]


class _Workstation(_Schema):
    objects: list[_Finding]
    tools: list[_Finding]
    surfaces: list[_Finding]


class _Observation(_Finding):
    kind: Literal["phase", "contact", "other"]


class _Condition(_Schema):
    condition: Literal["coupling", "left_arm_supported_or_leaning", "right_arm_supported_or_leaning"]
    value: bool | Literal["good", "fair", "poor", "unacceptable"]
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1)
    visibility: Literal["clear", "partial", "occluded"]
    evidence_images: list[int]


class _Reading(_Schema):
    task_inference: _Activity
    workstation: _Workstation
    observations: list[_Observation]
    condition_proposals: list[_Condition]


def _parse(text: str) -> dict[str, Any]:
    try:
        data = json.loads(_FENCE.sub("", text).strip())
    except json.JSONDecodeError as exc:
        raise BackendError("The scene reading did not contain valid JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("task_inference"), dict) or not isinstance(data.get("workstation"), dict):
        raise BackendError("The scene reading omitted its task inference or workstation.")
    # Drop claims of physical measurements from uncalibrated images. Both
    # historical string entries and current typed observations are recognized.
    for category in ("objects", "tools", "surfaces"):
        entries = data["workstation"].get(category)
        if isinstance(entries, list):
            data["workstation"][category] = [
                item for item in entries
                if not _PHYSICAL_UNIT.search(str(item.get("fact", "") if isinstance(item, dict) else item))
            ]
    return data


def _references(ids: list[int], shown: list[dict]) -> list[dict]:
    if not ids or any(index < 1 or index > len(shown) for index in ids):
        return []
    return [shown[index - 1] for index in dict.fromkeys(ids)]


def _ground_reading(text: str, shown: list[dict]) -> dict[str, Any]:
    try:
        reading = _Reading.model_validate(_parse(text))
    except ValidationError as exc:
        raise BackendError("The visual response did not match its evidence schema.") from exc
    activity = reading.task_inference
    activity_refs = _references(activity.evidence_images, shown)
    workstation: dict[str, list[str]] = {"objects": [], "tools": [], "surfaces": []}
    findings = []
    for category, items in (("objects", reading.workstation.objects), ("tools", reading.workstation.tools),
                            ("surfaces", reading.workstation.surfaces), ("other", reading.observations)):
        for item in items:
            refs = _references(item.evidence_images, shown)
            if refs and item.fact.strip() and not _PHYSICAL_UNIT.search(item.fact):
                findings.append({"fact": item.fact.strip(), "visibility": item.visibility,
                                 "category": item.kind if isinstance(item, _Observation) else category.removesuffix("s"),
                                 "evidence": refs})
                if category in workstation and item.visibility == "clear":
                    workstation[category].append(item.fact.strip())
    proposals = []
    for proposal in reading.condition_proposals:
        refs = _references(proposal.evidence_images, shown)
        if refs and proposal.visibility == "clear" and not _PHYSICAL_UNIT.search(proposal.rationale):
            item = proposal.model_dump()
            item["evidence_frames"] = [ref["frame"] for ref in refs]
            cleaned = _clean_proposals([item], set(item["evidence_frames"]))
            if cleaned:
                proposals.append({**cleaned[0], "evidence": refs})
    return {"task_inference": {"activity_class": activity.activity_class if activity_refs else "unknown",
                               "evidence": activity_refs}, "workstation": workstation,
            "findings": findings, "condition_proposals": proposals, "_images": shown}


def _read_once(session_id: str, samples: list[Sample], config: Any, camera_ids: list[int],
               pass_index: int, person: str | None) -> dict[str, Any]:
    from api.main import _frame_jpeg

    schema = _Reading.model_json_schema()
    messages = [{"role": "system", "content": scene_prompt()},
                {"role": "user", "content": "Return the following JSON schema. Use the image IDs "
                 "in each evidence_images array.\n" + json.dumps(schema)}]
    shown: list[dict] = []
    if person is None:
        messages.append({"role": "user", "content": "Identify only scene-level facts. A person has "
                         "not been uniquely identified across these frames. Omit condition proposals."})
    # Reserve enough image slots to inspect the same posture across views.
    # Rotate the primary view across passes while keeping the image budget fixed.
    budget = max(1, config.budget("scene"))
    primary = pass_index % len(camera_ids)
    peak = (next((s for s in samples if "worst" in s.reason), None)
            or next((s for s in samples if "peak" in s.reason), samples[0]))
    for sample in samples:
        views = [camera_ids[primary]]
        if sample == peak:
            views.extend(c for c in camera_ids if c != views[0])
        for camera in views:
            if len(shown) >= budget:
                break
            try:
                jpeg = _frame_jpeg(session_id, sample.frame, camera, "", False, config.frame_width, person)
            except (SessionError, OSError, ValueError) as exc:
                LOGGER.info("Scene skipped camera %s at frame %s: %s", camera, sample.frame, exc)
                continue
            shown.append({"frame": sample.frame, "sec": sample.sec, "camera": camera,
                          "person": person, "view": "raw"})
            caption = f"Image ID {len(shown)}. Frame {sample.frame}, {sample.sec:.2f} s, camera {camera}."
            # Use one labeled message per image on both provider dialects.
            messages.append(image_message(caption, jpeg))

    if not shown:
        raise BackendError("No sampled images could be decoded from the recorded camera sources.")
    text = complete(messages, config=config, model=config.scene_model,
                    temperature=config.scene_temperature, schema=schema)
    return _ground_reading(text, shown)


def _merge_findings(readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: dict[tuple, dict[str, Any]] = {}
    for reading in readings:
        counted: set[tuple] = set()
        for item in reading["findings"]:
            key = (item["category"], item["visibility"], item["fact"].casefold().strip(" ."))
            finding = facts.setdefault(key, {**item, "evidence": [], "passes": 0, "readings": len(readings)})
            if key not in counted:
                counted.add(key)
                finding["passes"] += 1
            for ref in item["evidence"]:
                if ref not in finding["evidence"]:
                    finding["evidence"].append(ref)
    return sorted(facts.values(), key=lambda f: -f["passes"])


def build_scene(
    session_id: str, progress: Any = None, passes: int | None = None,
    *, config: AssistantConfig | None = None,
) -> dict[str, Any]:
    """Read the recording and write ``vlm_scene.json``. Overwrites any prior card."""

    from api.analysis import people
    from api.assessments import atomic_json
    from assistant.brief import session_brief

    config = config or effective_config(session_id)
    require_vision(config)
    rounds = max(1, int(passes if passes is not None else config.scene_passes))
    camera_ids = [cam["camera_index"] for cam in cameras(session_id)]
    if not camera_ids:
        raise SessionError("The recordings used for this assessment are unavailable.")
    tracked = people(session_id)
    person = tracked[0] if len(tracked) == 1 else None
    budget = max(1, config.budget("scene") - len(camera_ids) + 1)
    fingerprint = _fingerprint(session_id, config)

    def say(message: str) -> None:
        if callable(progress):
            progress(message)

    readings: list[dict[str, Any]] = []
    every_frame: list[int] = []
    per_pass: list[dict[str, Any]] = []

    for index in range(rounds):
        # Use overlapping but different samples from the same recording.
        holdout = (index, rounds) if rounds > 1 else None
        samples = representative_frames(session_id, budget, holdout=holdout)
        if not samples:
            raise SessionError("This analysis has no observed frames to inspect.")
        say(f"Reading {len(samples)} frames (pass {index + 1} of {rounds})")
        try:
            reading = _read_once(session_id, samples, config, camera_ids, index, person)
        except BackendError as exc:
            LOGGER.warning("Scene pass %d failed: %s", index + 1, exc)
            if index == 0 and rounds == 1:
                raise
            continue
        readings.append(reading)
        frames = list(dict.fromkeys(ref["frame"] for ref in reading["_images"]))
        if person is None:
            # A scene-level assumption must not be attributed to the wrong worker.
            reading["condition_proposals"] = []
        every_frame.extend(frames)
        per_pass.append(
            {
                "pass": index + 1,
                "frames": frames,
                "images": reading["_images"],
                "coverage": coverage(session_id, frames),
                "activity_class": (reading.get("task_inference") or {}).get("activity_class"),
                "proposals": {
                    str(item.get("condition")): item.get("value", item.get("value_range"))
                    for item in (reading.get("condition_proposals") or [])
                    if isinstance(item, dict)
                },
            }
        )

    if not readings:
        raise BackendError("Every reading of this recording failed.")

    say("Comparing the readings")
    proposal_sets = [reading["condition_proposals"] for reading in readings]
    inference = merge_task_inference(
        [reading.get("task_inference") or {} for reading in readings]
    )
    findings = _merge_findings(readings)
    segments = [
        {"start_sec": min(ref["sec"] for ref in item["evidence"]),
         "end_sec": max(ref["sec"] for ref in item["evidence"]),
         "label": item["fact"], "evidence": item["evidence"], "sampledOnly": True}
        for item in findings if item["category"] == "phase" and item["visibility"] == "clear"
    ]

    unique_frames = sorted(set(every_frame))
    card: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "fingerprint": fingerprint,
        "sessionId": session_id,
        "builtAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": config.scene_model,
        "modelDigest": config.model_digest,
        "endpoint": config.endpoint,
        "promptVersion": PROMPT_VERSION,
        "temperature": config.scene_temperature,
        "seed": config.seed,
        "passes": len(readings),
        "passesAttempted": rounds,
        "frames_sampled": unique_frames,
        "frameSelection": {
            "method": "representative measured joint angles and scored episode peaks",
            "coverageVsUniform": coverage(session_id, unique_frames),
            "note": "Above 1 means these frames represent the recording's postures better "
            "than the same number of evenly time-spaced frames would.",
        },
        "perPass": per_pass,
        "task_inference": inference,
        "workstation": merge_workstation(
            [reading.get("workstation") or {} for reading in readings]
        ),
        "condition_proposals": merge_proposals(proposal_sets),
        "segments": segments,
        "findings": findings,
        "observations": [item["fact"] for item in findings if item["category"] in {"phase", "contact", "other"}][:10],
        "disclaimer": (
            "Agreement counts repeated readings of the same recording. Inspect each cited view "
            "before accepting a proposed condition for a future analysis run."
        ),
    }

    if _fingerprint(session_id, config) != fingerprint:
        raise SessionError("The assessment changed during scene inspection. Rebuild for the current evidence.")
    path = scene_path(session_id)
    archive = path.parent / "assistant_scenes"
    # Preserve pre-upgrade scene cards when the first new reading supersedes them.
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            old_key = previous.get("fingerprint") if isinstance(previous, dict) else None
            if isinstance(old_key, str) and re.fullmatch(r"[0-9a-f]{64}", old_key):
                atomic_json(archive / f"{old_key}.json", previous)
        except (OSError, ValueError):
            pass
    atomic_json(archive / f"{fingerprint}.json", card)
    atomic_json(path, card)
    if hasattr(session_brief, "cache_clear"):
        session_brief.cache_clear()
    LOGGER.info(
        "Wrote scene card for %s: %d passes, %d frames, %d proposals",
        session_id,
        len(readings),
        len(unique_frames),
        len(card["condition_proposals"]),
    )
    return card


__all__ = ["PROPOSABLE", "SCENE_FILE", "SCENE_SCHEMA", "build_scene", "load_scene", "scene_path"]
