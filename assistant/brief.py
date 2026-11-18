"""The context the assistant is given before anyone asks it anything.

An ergonomist opening a trial should not have to explain the trial. This module
assembles what a colleague reading over their shoulder would already know: who
was recorded, doing what, what both methods scored, where the risk concentrated,
which body regions carried it, and -- the part that decides whether the result
can be trusted -- which task conditions were measured and which were assumed.

Assumed and unknown conditions are in the brief on purpose. When someone asks
"can I rely on this REBA?", the honest answer depends on the coupling assumption,
and the model should not have to go looking for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from api import analysis as analysis_api
from api.sessions import (
    SessionError,
    load_summary,
    parse_session_id,
    resolve_result_dir,
)
from dashboard.cache import ttl_cache

#: Conditions no pose model can supply. Mirrors
#: ``ergonomics.condition_resolution.NOT_OBSERVABLE_FROM_VIDEO`` -- imported
#: rather than copied so the two cannot drift.
try:  # pragma: no cover - the scoring package is always present in practice
    from ergonomics.condition_resolution import NOT_OBSERVABLE_FROM_VIDEO
except Exception:  # noqa: BLE001
    NOT_OBSERVABLE_FROM_VIDEO = frozenset()


def _trial_context(session_id: str) -> dict[str, Any]:
    """Participant and task metadata for a session, or empty when unavailable."""

    from api.main import STUDIES_ROOT
    from api.sessions import find_context, find_trial

    ref = parse_session_id(session_id)
    try:
        context = find_context(STUDIES_ROOT, f"{ref.study_folder}::{ref.participant_id}")
        trial = find_trial(context, ref.task_folder)
    except SessionError:
        return {}
    info = context.get("participant", {}).get("info") or {}
    return {
        "participantId": context.get("participant_id"),
        "age": info.get("age"),
        "sex": info.get("sex"),
        "heightCm": info.get("height_cm"),
        "weightKg": info.get("weight_kg"),
        "dominantHand": info.get("dominant_hand"),
        "taskName": trial.get("name"),
        "taskFolder": trial.get("folder_name"),
        "taskDescription": trial.get("description"),
        "primaryTool": trial.get("primary_tool"),
        "repetitions": trial.get("num_repetitions"),
    }


@ttl_cache(ttl=60)
def condition_sources(session_id: str) -> dict[str, Any]:
    """Every task condition with its authority, and the values that were assumed.

    ``confirm_first`` is the pipeline's own list of assumed fields that no
    footage could have supplied. Those are exactly the fields a person looking at
    the video could answer, which is why the assistant is told about them.
    """

    summary = load_summary(session_id)
    provenance = summary.get("task_input_provenance") or {}
    values: dict[str, Any] = {}
    try:
        path = Path(resolve_result_dir(session_id)) / "frame_results.jsonl"
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
        record = json.loads(first) if first.strip() else {}
        values = (record.get("task_conditions") or {}).get("values") or {}
    except (OSError, UnicodeError, json.JSONDecodeError, SessionError):
        values = {}

    assumed = list(provenance.get("assumed_fields") or [])
    return {
        "observed": list(provenance.get("observed_fields") or []),
        "derived": list(provenance.get("derived_fields") or []),
        "assumed": assumed,
        "unknown": list(provenance.get("unknown_fields") or []),
        "conflicts": list(provenance.get("conflicts") or []),
        "confirmFirst": list(provenance.get("confirm_first") or []),
        "assumedValues": {name: values.get(name) for name in assumed},
        "notObservableFromPose": sorted(NOT_OBSERVABLE_FROM_VIDEO),
        "note": (
            "Authority order is observed (entered by the operator) > derived (measured from "
            "this recording) > assumed (a named profile) > unknown. An assumed value is not a "
            "measurement and must be described as an assumption."
        ),
    }


def _method_block(session_id: str, method: str, person: str | None) -> tuple[str, dict[str, Any]]:
    data = analysis_api.timeline(session_id, method, person)
    verdict = data["verdict"]
    lines = [
        f"{data['method']:<9} {verdict['headline']}",
        f"          Typical {verdict['typicalScore']}. "
        f"{verdict['percentAtAction']:.0f}% of scored time "
        f"({verdict['timeAtActionSec']:.0f} s of {verdict['scoredDurationSec']:.0f} s) "
        f"at {data['actionThreshold']}+, across {verdict['episodeCount']} sustained "
        f"{'episode' if verdict['episodeCount'] == 1 else 'episodes'}.",
    ]
    if verdict["peakFrame"] is not None:
        lines.append(
            f"          Peak at frame {verdict['peakFrame']} ({verdict['peakClock']})."
        )
    if verdict["leadingLabel"]:
        lines.append(
            f"          Leading postural contributor: {verdict['leadingLabel'].lower()}, "
            f"+{verdict['leadingPoints']:.1f} mean."
        )
    if verdict["leadingTaskLabel"]:
        lines.append(
            f"          Leading task factor: {verdict['leadingTaskLabel'].lower()}, "
            f"+{verdict['leadingTaskPoints']:.1f} mean."
        )
    return "\n".join(lines), data


@ttl_cache(ttl=60)
def analysis_brief(session_id: str, method_focus: str = "reba", tracked_person: str | None = None) -> str:
    """The recorded assessment, excluding all model-generated scene content."""

    overview = analysis_api.session_overview(session_id)
    trial = _trial_context(session_id)
    ref = parse_session_id(session_id)

    out: list[str] = []
    out.append("=== ASSESSMENT BRIEF ===")
    out.append(f"SESSION   {session_id}")

    task_line = trial.get("taskName") or ref.task_folder
    if trial.get("taskDescription"):
        task_line += f" — \"{trial['taskDescription']}\""
    if trial.get("repetitions"):
        task_line += f", {trial['repetitions']} repetitions planned"
    if trial.get("primaryTool"):
        task_line += f", primary tool {str(trial['primaryTool']).upper()}"
    out.append(f"TRIAL     {task_line}.")

    person = [
        f"P{trial.get('participantId') or ref.participant_id}",
        f"{trial['age']}" if trial.get("age") else None,
        trial.get("sex"),
        f"{trial['heightCm']} cm" if trial.get("heightCm") else None,
        f"{trial['weightKg']} kg" if trial.get("weightKg") else None,
        f"{trial['dominantHand']}-handed" if trial.get("dominantHand") else None,
    ]
    out.append("PERSON    " + ", ".join(item for item in person if item) + ".")

    out.append(
        f"RECORDING {overview['durationSec']:.0f} s, {overview['cameraCount']} "
        f"{'camera' if overview['cameraCount'] == 1 else 'cameras'}, "
        f"{ref.sampling_label.lower()} sampling, {overview['trackingMode']}, "
        f"{str(overview['calibration']).replace('_', ' ')}, software {overview['softwareVersion']}."
    )
    out.append("")

    timelines: dict[str, dict[str, Any]] = {}
    for method in ("reba", "rula"):
        try:
            block, data = _method_block(session_id, method, tracked_person)
        except SessionError:
            continue
        timelines[method] = data
        out.append(block)
    out.append("")

    primary = timelines.get(method_focus) or next(iter(timelines.values()), None)
    if primary:
        episodes = primary["episodes"][:5]
        if episodes:
            out.append(f"TOP EPISODES ({primary['method']})")
            for item in episodes:
                drivers = ", ".join(
                    f"{d['label'].lower()} +{d['pointsAdded']}" for d in item["drivers"][:3]
                )
                out.append(
                    f"  {item['index']}) {item['timeRange']}  peak {item['peakScore']} "
                    f"@ frame {item['peakFrame']}  — {drivers or 'no component evidence'}"
                )
            out.append("")

        out.append(f"BODY LOAD ({primary['method']}: mean points / % above neutral / peak)")
        for load in primary["posture"][:6]:
            measurement = (
                f"{load['measurement'].replace('_', ' ')} {load['peakValueDeg']:.0f}°"
                if load["measurement"] and load["peakValueDeg"] is not None
                else "—"
            )
            peak = f"frame {load['peakFrame']}" if load["peakFrame"] is not None else "—"
            out.append(
                f"  {load['label']:<16} +{load['meanPoints']:<5.2f} {load['percentRaised']:>5.1f}%  "
                f"{peak:<12} {measurement}"
            )
        if primary["taskFactors"]:
            out.append("TASK FACTORS")
            for load in primary["taskFactors"]:
                out.append(f"  {load['label']:<40} +{load['meanPoints']:.2f}")
        out.append("")

    try:
        sources = condition_sources(session_id)
    except SessionError:
        sources = {}
    if sources:
        out.append("CONDITION SOURCES  (observed > derived > assumed > unknown)")
        out.append(f"  derived ({len(sources['derived'])}): measured from this recording")
        assumed_values = sources.get("assumedValues") or {}
        if assumed_values:
            rendered = ", ".join(
                f"{name}={value}" for name, value in sorted(assumed_values.items())
            )
            out.append(f"  assumed ({len(sources['assumed'])}): {rendered}")
            out.append(
                "  ^ NOT measurements. These came from a profile because nothing in the "
                "pose measurements could supply them. Check the task record for measured load or force."
            )
        if sources.get("unknown"):
            out.append(f"  unknown ({len(sources['unknown'])}): {', '.join(sources['unknown'])}")
        if sources.get("conflicts"):
            out.append(f"  CONFLICTS: {len(sources['conflicts'])} entered values disagree with the footage")
        out.append("")

    rnle = overview.get("rnle") or {}
    if rnle.get("liftingIndex") is not None:
        out.append(
            f"RNLE      Lifting index {rnle['liftingIndex']:.2f}, "
            f"recommended weight {rnle.get('recommendedWeightKg')} kg "
            f"({str(rnle.get('validity')).replace('_', ' ')})."
        )
    elif rnle.get("validity"):
        out.append(
            f"RNLE      Withheld — {str(rnle['validity']).replace('_', ' ')}"
            + (f"; missing {', '.join(rnle['missing'][:4])}." if rnle.get("missing") else ".")
        )

    warnings = overview.get("warnings") or []
    if warnings:
        out.append("WARNINGS")
        for warning in warnings[:6]:
            out.append(f"  - {warning}")

    return "\n".join(out)


def session_brief(session_id: str, method_focus: str = "reba", tracked_person: str | None = None) -> str:
    """The recorded assessment plus the optional visual reading for the model."""

    from assistant.scene import load_scene

    out = [analysis_brief(session_id, method_focus, tracked_person)]
    card = load_scene(session_id)
    out.append("")
    if card:
        inference = card.get("task_inference") or {}
        out.append("SCENE CARD  (VLM visual reading — observation, NOT measurement)")
        out.append(f"  Activity classification: {inference.get('activity_class') or 'unknown'}")
        out.append(f"  One reading's description: {inference.get('summary', '—')}")
        if inference.get("alternative_summaries"):
            out.append("  Other passes described the scene differently. Check the images for details.")
        objects = (card.get("workstation") or {}).get("objects") or []
        if objects:
            out.append(f"  Objects: {', '.join(str(item) for item in objects[:6])}")
        for proposal in (card.get("condition_proposals") or [])[:6]:
            value = proposal.get("value", proposal.get("value_range"))
            out.append(
                f"  Proposes {proposal.get('condition')}={value} "
                f"({proposal.get('confidence')} confidence) — {proposal.get('rationale', '')}"
            )
    else:
        out.append(
            "SCENE CARD  Not built for this session. Call look_at or watch to inspect the "
            "footage directly before making any claim about the workplace."
        )

    return "\n".join(out)


def suggested_questions(session_id: str, method: str = "reba") -> list[str]:
    """Opening prompts generated from this trial, not from a static list."""

    questions: list[str] = []
    try:
        data = analysis_api.timeline(session_id, method)
    except SessionError:
        return ["What does this assessment show?"]

    verdict = data["verdict"]
    if verdict["peakFrame"] is not None:
        questions.append(f"Why did {method.upper()} peak at {verdict['peakClock']}?")
    if verdict["leadingLabel"]:
        questions.append(f"What is driving the {verdict['leadingLabel'].lower()} load?")

    try:
        sources = condition_sources(session_id)
        assumed = sources.get("assumedValues") or {}
        if "coupling" in assumed:
            questions.append(f"Is the coupling = {assumed['coupling']} assumption safe here?")
        elif assumed:
            questions.append("Which assumed conditions could change this result?")
    except SessionError:
        pass

    questions.append("What should we change first?")
    questions.append("Can this result support a decision?")
    return questions[:5]


__all__ = ["condition_sources", "session_brief", "suggested_questions"]
