"""The functions the model may call, and nothing else.

Every tool here is a thin wrapper over code the React workspace already renders
from. That is the whole point: a number the assistant states came back from
``api.analysis.timeline`` or ``api.analysis.frame_detail``, so if the chat and
the chart ever disagree it is a bug with a stack trace rather than a model
quirk. No tool computes an ergonomic score and no tool writes anything.

Two of them return images. The OpenAI tool protocol has no image channel in a
tool result, so those return a text receipt and hand the JPEG bytes back through
``ToolResult.images``; the caller appends them as a following user message.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from api import analysis as analysis_api
from api import frame_data as frame_data_api
from api.sessions import SessionError
from assistant.config import load_config
from assistant.reference import search_reference

LOGGER = logging.getLogger(__name__)

#: Guards a runaway model from pulling a whole trial into context one call at a
#: time. Well past what any real question needs.
MAX_IMAGES_PER_TURN = 12


@dataclass(slots=True)
class ToolResult:
    """What a tool hands back: text for the model, images for the next message."""

    content: Any
    images: list[tuple[str, bytes]] = field(default_factory=list)
    #: Frames this call put on screen, so the transcript can cite them.
    frames: list[int] = field(default_factory=list)
    # One receipt per image, in delivery order. Numeric frame lookups carry none.
    image_refs: list[dict[str, Any]] = field(default_factory=list)


def _timeline(session_id: str, method: str, person: str | None = None) -> dict[str, Any]:
    return analysis_api.timeline(session_id, _method(method), person)


def _method(value: Any) -> str:
    key = str(value or "reba").strip().lower()
    if key not in {"reba", "rula"}:
        raise SessionError(f"Unknown method {value!r}. Choose REBA or RULA.")
    return key


def _round(value: Any, digits: int = 2) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) else value


# ----------------------------------------------------------------------
# Analysis tools
# ----------------------------------------------------------------------


def tool_session_overview(session_id: str, **_: Any) -> ToolResult:
    data = analysis_api.session_overview(session_id)
    return ToolResult(
        {
            "durationSec": _round(data.get("durationSec")),
            "processedFrames": data.get("processedFrames"),
            "poseFrames": data.get("poseFrames"),
            "cameraCount": data.get("cameraCount"),
            "trackingMode": data.get("trackingMode"),
            "calibration": data.get("calibration"),
            "softwareVersion": data.get("softwareVersion"),
            "methods": data.get("methods"),
            "rnle": data.get("rnle"),
            "warnings": data.get("warnings"),
            "people": data.get("people"),
        }
    )


def tool_method_summary(session_id: str, method: str = "reba", person: str | None = None, **_: Any) -> ToolResult:
    data = _timeline(session_id, method, person)
    verdict = data["verdict"]
    return ToolResult(
        {
            "method": data["method"],
            "person": data.get("person"),
            "headline": verdict["headline"],
            "detail": verdict["detail"],
            "peakScore": verdict["peakScore"],
            "peakFrame": verdict["peakFrame"],
            "peakClock": verdict["peakClock"],
            "peakBand": verdict["peakBand"],
            "typicalScore": verdict["typicalScore"],
            "actionThreshold": data["actionThreshold"],
            "percentAtAction": verdict["percentAtAction"],
            "timeAtActionSec": verdict["timeAtActionSec"],
            "scoredDurationSec": verdict["scoredDurationSec"],
            "episodeCount": verdict["episodeCount"],
            "leadingLabel": verdict["leadingLabel"],
            "leadingPoints": verdict["leadingPoints"],
            "leadingTaskLabel": verdict["leadingTaskLabel"],
            "bandExposure": data["shares"],
        }
    )


def tool_episodes(session_id: str, method: str = "reba", limit: int = 5, person: str | None = None, **_: Any) -> ToolResult:
    data = _timeline(session_id, method, person)
    try:
        count = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        count = 5
    episodes = data["episodes"][:count]
    if not episodes:
        return ToolResult(
            {
                "episodes": [],
                "note": (
                    f"No stretch stayed at {data['method']} {data['actionThreshold']} or above "
                    "for long enough to count as a sustained episode."
                ),
            }
        )
    return ToolResult(
        {
            "method": data["method"],
            "totalEpisodes": len(data["episodes"]),
            "person": data.get("person"),
            "episodes": [
                {
                    "index": item["index"],
                    "timeRange": item["timeRange"],
                    "durationSec": item["durationSec"],
                    "peakScore": item["peakScore"],
                    "peakFrame": item["peakFrame"],
                    "peakSec": item["peakSec"],
                    "band": (item["band"] or {}).get("label"),
                    "drivers": [
                        {
                            "label": d["label"],
                            "pointsAdded": d["pointsAdded"],
                            "score": d["score"],
                            "maximum": d["maximum"],
                            "measurement": d["measurement"],
                            "valueDeg": d["valueDeg"],
                        }
                        for d in item["drivers"][:4]
                    ],
                }
                for item in episodes
            ],
        }
    )


def _exact_frame(session_id: str, frame: int, method: str, person: str | None = None) -> dict[str, Any]:
    if isinstance(frame, bool) or (isinstance(frame, float) and not frame.is_integer()):
        raise SessionError("An integer frame number is required.")
    requested = int(frame)
    data = analysis_api.frame_detail(session_id, requested, method, person)
    if data["frame"] != requested:
        raise SessionError(
            f"Frame {requested} has no scored record. The nearest record is frame "
            f"{data['frame']}; request that frame separately and identify the substitution."
        )
    return data


def tool_frame(session_id: str, frame: int, method: str = "reba", person: str | None = None, **_: Any) -> ToolResult:
    key = _method(method)
    data = _exact_frame(session_id, frame, key, person)
    return ToolResult(
        {
            "method": key,
            "frame": data["frame"],
            "person": data.get("person"),
            "clock": data["clock"],
            "timestampSec": _round(data["timestampSec"], 3),
            "score": data["score"],
            "band": data["band"],
            "selectedSide": data["selectedSide"],
            "sides": data.get("sides", {}),
            "quality": data["quality"],
            "components": data["components"],
            "drivers": [
                {
                    "label": d["label"],
                    "pointsAdded": d["pointsAdded"],
                    "score": d["score"],
                    "maximum": d["maximum"],
                    "measurement": d["measurement"],
                    "valueDeg": d["valueDeg"],
                }
                for d in data["drivers"]
            ],
            "angles": data["angles"],
            "angleSources": data.get("angleSources") or {},
            "anglesByCamera": data.get("anglesByCamera") or {},
        },
        frames=[int(data["frame"])],
    )


def tool_body_regions(session_id: str, method: str = "reba", person: str | None = None, **_: Any) -> ToolResult:
    data = _timeline(session_id, method, person)
    return ToolResult(
        {
            "method": data["method"],
            "note": "meanPoints is points added to the average score; percentRaised is the "
            "share of scored time this region spent above its own neutral.",
            "regions": [
                {
                    "label": item["label"],
                    "region": item["region"],
                    "meanPoints": item["meanPoints"],
                    "worstPoints": item["worstPoints"],
                    "percentRaised": item["percentRaised"],
                    "peakFrame": item["peakFrame"],
                    "peakSec": item["peakSec"],
                    "measurement": item["measurement"],
                    "peakValueDeg": item["peakValueDeg"],
                }
                for item in data["posture"]
            ],
        }
    )


def tool_task_factors(session_id: str, method: str = "reba", person: str | None = None, **_: Any) -> ToolResult:
    data = _timeline(session_id, method, person)
    return ToolResult(
        {
            "method": data["method"],
            "note": "Non-postural contributors. These change the score without a body part "
            "moving, so they lead to different interventions than posture does.",
            "factors": [
                {
                    "label": item["label"],
                    "meanPoints": item["meanPoints"],
                    "worstPoints": item["worstPoints"],
                    "percentRaised": item["percentRaised"],
                }
                for item in data["taskFactors"]
            ],
        }
    )


def tool_risk_periods(session_id: str, method: str = "reba", person: str | None = None, **_: Any) -> ToolResult:
    data = frame_data_api.risk_periods_payload(session_id, _method(method), band="all", person=person)
    periods = data.get("periods", [])
    return ToolResult(
        {
            "method": data.get("method"),
            "durationSec": _round(data.get("durationSec")),
            "count": len(periods),
            "periods": [
                {
                    "index": p["index"],
                    "startSec": p["startSec"],
                    "endSec": p["endSec"],
                    "durationSec": p["durationSec"],
                    "peakScore": p["peakScore"],
                    "peakFrame": p["peakFrame"],
                    "band": (p["band"] or {}).get("label"),
                }
                for p in periods[:40]
            ],
        }
    )


def tool_observations(
    session_id: str,
    method: str = "reba",
    from_sec: float | None = None,
    to_sec: float | None = None,
    band: str = "all",
    person: str | None = None,
    **_: Any,
) -> ToolResult:
    """Non-decimated frame rows, optionally restricted to a time window."""

    key = _method(method)
    rows: list[dict[str, Any]] = []
    page = 1
    while page <= 40:
        payload = frame_data_api.observations_payload(
            session_id, key, band=band, page=page, limit=frame_data_api.MAX_PAGE_SIZE, person=person
        )
        rows.extend(payload.get("rows", []))
        if page >= int(payload.get("pages", 1)):
            break
        page += 1

    def within(row: dict[str, Any]) -> bool:
        t = row.get("timestampSec")
        if not isinstance(t, (int, float)):
            return False
        if from_sec is not None and t < float(from_sec):
            return False
        if to_sec is not None and t > float(to_sec):
            return False
        return True

    if from_sec is not None or to_sec is not None:
        rows = [row for row in rows if within(row)]

    # A window can still hold hundreds of rows; the model needs the shape and the
    # extremes, not a transcript. Keep every row when it is already small.
    trimmed = rows if len(rows) <= 40 else rows[:: max(1, len(rows) // 40)][:40]
    scored = [r["score"] for r in rows if isinstance(r.get("score"), int)]
    return ToolResult(
        {
            "method": key.upper(),
            "windowFromSec": from_sec,
            "windowToSec": to_sec,
            "matchedRows": len(rows),
            "returnedRows": len(trimmed),
            "scoreMin": min(scored) if scored else None,
            "scoreMax": max(scored) if scored else None,
            "rows": [
                {
                    "frame": r["frame"],
                    "timestampSec": r["timestampSec"],
                    "person": r.get("person"),
                    "clock": r["clock"],
                    "score": r["score"],
                    "band": (r.get("band") or {}).get("label"),
                    "validity": r.get("validity"),
                    "meanVisibility": r.get("meanVisibility"),
                }
                for r in trimmed
            ],
        }
    )


def tool_condition_sources(session_id: str, frame: int | None = None, method: str = "reba",
                           person: str | None = None, **_: Any) -> ToolResult:
    """Where every task condition came from: entered, measured, assumed, unknown."""

    from assistant.brief import condition_sources

    if frame is None:
        if person is None:
            return ToolResult(condition_sources(session_id))
        frame = next((entry.frame_number for entry in analysis_api._store(session_id).entries
                      if entry.person_identifier == person and entry.frame_number is not None), None)
        if frame is None:
            return ToolResult({"error": "There are no recorded conditions for this worker."})
    detail = _exact_frame(session_id, frame, _method(method), person)
    record = analysis_api._store(session_id).load_record_at_offset(detail["offset"])
    conditions = record.get("task_conditions") or {}
    values = conditions.get("values") or {}
    return ToolResult({"frame": frame, "timestampSec": detail["timestampSec"],
                       "person": detail["person"], "method": detail["method"],
                       "values": values, "sources": conditions.get("sources", {}),
                       "conflicts": conditions.get("conflicts", []),
                       "note": "Recorded conditions for this specific worker and moment."},
                      frames=[frame])


def tool_scene_card(session_id: str, **_: Any) -> ToolResult:
    from assistant.scene import load_scene

    card = load_scene(session_id)
    if card is None:
        return ToolResult(
            {
                "available": False,
                "note": "No scene card has been built for this session yet. Use look_at to "
                "inspect frames directly instead.",
            }
        )
    return ToolResult({"available": True, **card})


def tool_method_reference(query: str, **_: Any) -> ToolResult:
    return ToolResult({"query": query, "sections": search_reference(str(query))})


# ----------------------------------------------------------------------
# Visual tools
# ----------------------------------------------------------------------


def tool_look_at(
    session_id: str, frame: int, camera: int | None = None, overlay: bool = False,
    region: str = "", method: str = "reba", person: str | None = None,
    image_budget: int = 2, **_: Any,
) -> ToolResult:
    """Return a full view and, when visible, a native detail of the requested joints."""
    from assistant.visuals import render

    key = _method(method)
    detail = _exact_frame(session_id, frame, key, person)
    images, refs = render(session_id, detail, key, camera, region, overlay,
                          load_config().frame_width, image_budget)
    return ToolResult(
        {"frame": detail["frame"], "method": key, "person": detail.get("person"),
         "camera": refs[0]["camera"], "images": refs,
         "note": "Report what is visible. A tracked landmark cannot establish contact or force."},
        images=images, frames=[detail["frame"]], image_refs=refs,
    )


def tool_watch(
    session_id: str, from_sec: float, to_sec: float, camera: int | None = None,
    max_frames: int = 6, overlay: bool = False, method: str = "reba",
    person: str | None = None, region: str = "", **_: Any,
) -> ToolResult:
    """Sample the entire requested window, including its last observed moment."""
    from assistant.visuals import render

    key = _method(method)
    tracked = person or next(iter(analysis_api.people(session_id)), None)
    if not tracked:
        return ToolResult({"error": "No tracked worker is available."})
    # Sample the original observations. A chart's decimated points can omit a
    # brief contact or exclude a whole interval when this method withheld a score.
    entries = [entry for entry in analysis_api._store(session_id).entries
               if entry.person_identifier == tracked and entry.frame_number is not None
               and entry.timestamp_sec is not None and from_sec <= entry.timestamp_sec <= to_sec]
    if not entries:
        return ToolResult({"error": "No observed frames in the requested interval for this worker."})
    config = load_config()
    count = max(1, min(max_frames, config.budget("episode"), MAX_IMAGES_PER_TURN, len(entries)))
    indices = [len(entries) // 2] if count == 1 else [
        round(i * (len(entries) - 1) / (count - 1)) for i in range(count)
    ]
    images, refs = [], []
    for index in indices:
        entry = entries[index]
        try:
            detail = _exact_frame(session_id, entry.frame_number, key, tracked)
            shown, receipts = render(session_id, detail, key, camera, region, overlay,
                                     config.frame_width, 1)
        except (ValueError, OSError) as exc:
            LOGGER.info("Unavailable series image at frame %s: %s", entry.frame_number, exc)
            continue
        # Keep a movement strip on one camera; a change in projection can
        # resemble movement. Other views can be requested as separate strips.
        camera = receipts[0]["camera"]
        images.extend(shown)
        refs.extend(receipts)
    if not refs:
        return ToolResult({"error": "None of the requested images could be rendered."})
    return ToolResult(
        {"method": key, "person": tracked, "camera": camera, "images": refs,
         "note": "The image receipts follow sampling order. Gaps do not establish continuous motion."},
        images=images, frames=[r["frame"] for r in refs], image_refs=refs,
    )


def tool_cameras(session_id: str, **_: Any) -> ToolResult:
    from assistant.visuals import cameras

    views = [{"index": v["camera_index"], "name": v["name"], "role": v.get("role"),
              "available": v.get("available", True)}
             for v in cameras(session_id)]
    return ToolResult({"cameras": views,
                       "note": "Only cameras listed as available can supply another view."})


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

_METHOD_PARAM = {
    "type": "string",
    "enum": ["reba", "rula"],
    "description": "Defaults to the method selected in the workspace.",
}

TOOLS: dict[str, tuple[Callable[..., ToolResult], dict[str, Any]]] = {
    "session_overview": (
        tool_session_overview,
        {
            "description": "Recording facts: duration, cameras, tracking mode, both method "
            "headlines, the RNLE block, and any analysis warnings.",
            "parameters": {"type": "object", "properties": {}},
        },
    ),
    "method_summary": (
        tool_method_summary,
        {
            "description": "The headline result for one method: peak, typical, percent of time "
            "at or above the action threshold, band exposure, and the leading contributor.",
            "parameters": {"type": "object", "properties": {"method": _METHOD_PARAM}},
        },
    ),
    "episodes": (
        tool_episodes,
        {
            "description": "Sustained high-risk episodes, ranked, each with its peak frame and "
            "the components that drove that peak.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": _METHOD_PARAM,
                    "limit": {"type": "integer", "description": "How many, 1-20. Default 5."},
                },
            },
        },
    ),
    "frame": (
        tool_frame,
        {
            "description": "Everything computed for one frame: score, band, every component, "
            "joint angles used for scoring, raw per-camera angles when stored, "
            "the selected sources, drivers and pose quality.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer", "description": "Frame number."},
                    "method": _METHOD_PARAM,
                },
                "required": ["frame"],
            },
        },
    ),
    "body_regions": (
        tool_body_regions,
        {
            "description": "Per-body-region load across the whole trial: mean and worst points "
            "added, percent of time above neutral, and where each peaked.",
            "parameters": {"type": "object", "properties": {"method": _METHOD_PARAM}},
        },
    ),
    "task_factors": (
        tool_task_factors,
        {
            "description": "Non-postural contributors (force/load, coupling, activity) and how "
            "many points each adds.",
            "parameters": {"type": "object", "properties": {"method": _METHOD_PARAM}},
        },
    ),
    "risk_periods": (
        tool_risk_periods,
        {
            "description": "Every contiguous action-band period across the trial, including "
            "evidence gaps.",
            "parameters": {"type": "object", "properties": {"method": _METHOD_PARAM}},
        },
    ),
    "observations": (
        tool_observations,
        {
            "description": "Frame-by-frame scored rows, optionally within a time window. Use "
            "for questions about a specific stretch of the recording.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": _METHOD_PARAM,
                    "from_sec": {"type": "number", "description": "Window start in seconds."},
                    "to_sec": {"type": "number", "description": "Window end in seconds."},
                },
            },
        },
    ),
    "condition_sources": (
        tool_condition_sources,
        {
            "description": "Where every task condition came from: entered by the operator, "
            "measured from the footage, assumed from a profile, or unknown. Use this whenever "
            "the question is whether a result can be trusted.",
            "parameters": {"type": "object", "properties": {
                "frame": {"type": "integer", "description": "Requested frame or current selection."},
                "method": _METHOD_PARAM,
            }},
        },
    ),
    "scene_card": (
        tool_scene_card,
        {
            "description": "The cached visual reading of this recording: inferred task, "
            "workstation objects, and proposed values for conditions the pose model cannot see.",
            "parameters": {"type": "object", "properties": {}},
        },
    ),
    "method_reference": (
        tool_method_reference,
        {
            "description": "Search this project's own method documentation for how RULA, REBA "
            "or the RNLE is implemented here, and what its stated limits are.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look up."}},
                "required": ["query"],
            },
        },
    ),
    "look_at": (
        tool_look_at,
        {
            "description": "Look at one frame of the recording. Use this before making any "
            "claim about the workplace, the object handled, or the surroundings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer", "description": "Frame number."},
                    "method": _METHOD_PARAM,
                    "camera": {"type": "integer", "description": "Recorded camera index. Omit to use the selected view or choose by joint visibility."},
                    "region": {"type": "string", "description": "Inspection target such as right_hand, left_forearm, wrist, trunk or neck. A visible joint group can supply a detail crop."},
                    "overlay": {
                        "type": "boolean",
                        "description": "Draw the tracked skeleton. Defaults to raw video for contact and objects.",
                    },
                },
                "required": ["frame"],
            },
        },
    ),
    "watch": (
        tool_watch,
        {
            "description": "Look at several frames spread across a time window, to see how a "
            "movement develops.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_sec": {"type": "number"},
                    "to_sec": {"type": "number"},
                    "method": _METHOD_PARAM,
                    "camera": {"type": "integer", "description": "Recorded camera index. Omit to use the selection."},
                    "region": {"type": "string", "description": "Body region being inspected."},
                    "max_frames": {"type": "integer", "description": "Up to 8. Default 6."},
                },
                "required": ["from_sec", "to_sec"],
            },
        },
    ),
    "cameras": (
        tool_cameras,
        {
            "description": "Which camera views exist for this recording.",
            "parameters": {"type": "object", "properties": {}},
        },
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """The ``tools`` array sent to the model."""

    return [
        {"type": "function", "function": {"name": name, **spec}}
        for name, (_, spec) in TOOLS.items()
    ]


def run_tool(
    name: str, session_id: str, arguments: dict[str, Any], *, remaining_images: int = MAX_IMAGES_PER_TURN,
    selection: dict[str, Any] | None = None,
) -> ToolResult:
    """Execute one tool call. Failures come back as content, never as exceptions.

    A tool that raises would abort the turn and lose the answer already drafted;
    an error the model can read lets it recover or say what it could not get.
    """

    entry = TOOLS.get(name)
    if entry is None:
        return ToolResult({"error": f"No tool named {name!r}."})
    function, _ = entry
    try:
        arguments = dict(arguments)
        selection = selection or {}
        if selection.get("method"):
            arguments.setdefault("method", selection["method"])
        if selection.get("person"):
            arguments["person"] = selection["person"]
        if name in {"frame", "look_at", "condition_sources"} and selection.get("frame") is not None:
            arguments.setdefault("frame", selection["frame"])
        if name in {"look_at", "watch"}:
            for field_name in ("region", "camera"):
                if selection.get(field_name) is not None:
                    arguments.setdefault(field_name, selection[field_name])
        if name in {"look_at", "watch"} and remaining_images <= 0:
            return ToolResult({"error": "The image budget is exhausted. No image was shown for this request."})
        if name == "look_at":
            arguments["image_budget"] = min(2, remaining_images)
        if name in {"watch", "observations"}:
            for argument, field_name in (("from_sec", "fromSec"), ("to_sec", "toSec")):
                if selection.get(field_name) is not None:
                    arguments.setdefault(argument, selection[field_name])
        if name == "watch":
            arguments = {**arguments, "max_frames": min(int(arguments.get("max_frames", 6)), remaining_images)}
        return function(session_id=session_id, **arguments)
    except SessionError as exc:
        return ToolResult({"error": str(exc)})
    except TypeError as exc:
        return ToolResult({"error": f"Bad arguments for {name}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - surfaced to the model, logged for us
        LOGGER.exception("Tool %s failed", name)
        return ToolResult({"error": f"{name} failed: {exc}"})


__all__ = ["MAX_IMAGES_PER_TURN", "TOOLS", "ToolResult", "run_tool", "tool_schemas"]
