"""Professional, validity-aware Results page for the unified dashboard.

The renderer is dependency-injected so the application owns filesystem discovery,
caching, and session state.  This module never calculates an ergonomic composite:
RULA, REBA, RNLE, posture-band exposure, and candidate movement cycles remain
separate evidence streams.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard.professional_visuals import (
    CSV_COLUMNS,
    angle_summary_figure,
    camera_evidence_figure,
    data_quality_coverage_figure,
    method_readiness_figure,
    posture_band_exposure_figure,
    professional_csv_rows,
    professional_markdown_report,
)
from dashboard.task_timeline import (
    BodyPartLoad,
    RiskEpisode,
    ScorePoint,
    Verdict,
    band_shares,
    body_part_load,
    clock,
    risk_episodes,
    score_series,
)
from dashboard.task_timeline import verdict as build_verdict
from dashboard.timeline_visuals import (
    body_load_map_figure,
    compare_methods_figure,
    contribution_figure,
    risk_ribbon_figure,
    selected_frame_from_event,
    side_comparison_figure,
    task_risk_timeline_figure,
    time_at_band_figure,
)

type SamplingValue = int | None
type ProcessedFpsLoader = Callable[[str, str, str], Sequence[SamplingValue]]
type ResultDetector = Callable[[str, str, str, SamplingValue], Mapping[str, Any] | None]
type BundleLoader = Callable[[str | Path], Mapping[str, Any]]
type SummaryLoader = Callable[[str | Path], Mapping[str, Any]]
type AnalysisCache = Callable[..., Mapping[str, Any]]

#: The section the evidence buttons hand a frame to. Kept as a constant so the
#: page and the application agree on the name without importing each other.
FRAME_REVIEW_SECTION = "Frame Review"

ALL_REGIONS = "all"
# Kept equal to ``dashboard.analysis_service.REGION_LABELS`` without importing
# OpenCV at module import time; pure page helpers remain testable without the
# video runtime. The timeline helper is imported lazily only while rendering.
REGION_LABELS: Mapping[str, str] = {
    "neck": "Neck",
    "trunk": "Trunk",
    "left_shoulder": "Left shoulder",
    "right_shoulder": "Right shoulder",
    "left_elbow": "Left elbow",
    "right_elbow": "Right elbow",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
    "legs": "Legs and balance",
}
REGION_MEASUREMENTS: Mapping[str, tuple[str, ...]] = {
    "neck": ("neck_",),
    "trunk": ("trunk_",),
    "left_shoulder": ("left_upper_arm_", "left_shoulder_"),
    "right_shoulder": ("right_upper_arm_", "right_shoulder_"),
    "left_elbow": ("left_forearm_", "left_lower_arm_", "left_elbow_"),
    "right_elbow": ("right_forearm_", "right_lower_arm_", "right_elbow_"),
    "left_wrist": ("left_wrist_",),
    "right_wrist": ("right_wrist_",),
    "legs": ("left_knee_", "right_knee_", "leg_", "legs_"),
}
# Ordered as an assessor works: what happened, when during the task, where in
# the body, why that score, can it be trusted, and what does the report say. The
# labels are the questions rather than the module names, because a reviewer
# opening this for the first time is looking for an answer and not for a section.
#
# "The verdict" comes first and is the only view the default workflow shows on
# its own, because the previous page opened on six equally weighted tabs of
# whole-trial aggregates and left the reader to work out which one answered their
# question.
RESULT_VIEWS = (
    "The verdict",
    "When in the task",
    "Where in the body",
    "Why this score",
    "Can I trust it",
    "Report & compare",
)

#: Older sessions may hold a stored view name from a previous labelling.
_LEGACY_VIEW_NAMES = {
    "Executive Summary": "The verdict",
    "What is the risk": "The verdict",
    "Posture Exposure": "Where in the body",
    "Movement & Repetition": "When in the task",
    "Data Quality & Cameras": "Can I trust it",
    "Methods & Report": "Report & compare",
    "Compare": "Report & compare",
}

#: RULA and REBA cover different body regions, use different scales, and answer
#: different questions. Showing both at once through every chart was a large part
#: of why the page could not be read; the page follows one at a time and says so.
METHOD_SCOPE = {
    "REBA": "whole body, including legs and load handling",
    "RULA": "upper limb, neck and trunk",
}


class ProfessionalPageError(ValueError):
    """Raised when injected dashboard data violates the Results-page contract."""


@dataclass(frozen=True, slots=True)
class MethodSnapshot:
    method: str
    maximum_score: float | None
    valid_result_count: int
    assessed_record_count: int
    frame_number: int | None
    timestamp_sec: float | None
    left_score: float | None
    right_score: float | None
    selected_side: str | None
    selection_reason: str | None
    interpretation: str | None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _human(value: Any) -> str:
    return str(value or "unavailable").replace("_", " ").strip().title()


def _format_number(value: Any, suffix: str = "", digits: int = 1) -> str:
    number = _finite(value)
    return "Unavailable" if number is None else f"{number:.{digits}f}{suffix}"


def _measurement_region(measurement: Any) -> str | None:
    if not isinstance(measurement, str):
        return None
    return next(
        (
            region
            for region, prefixes in REGION_MEASUREMENTS.items()
            if any(measurement.startswith(prefix) for prefix in prefixes)
        ),
        None,
    )


def scope_records(
    records: Sequence[Mapping[str, Any]],
    *,
    person_identifier: str,
    frame_range: tuple[int, int],
) -> list[Mapping[str, Any]]:
    """Return only records in the requested person and inclusive frame scope."""

    first, last = frame_range
    if first > last:
        raise ProfessionalPageError("Frame range start must not exceed its end")
    scoped: list[Mapping[str, Any]] = []
    for record in records:
        frame = _integer(record.get("frame_number"))
        person = record.get("person_identifier")
        if frame is not None and str(person) == person_identifier and first <= frame <= last:
            scoped.append(record)
    return scoped


def filter_analysis_region(analysis: Mapping[str, Any], region: str) -> dict[str, Any]:
    """Filter independent per-measurement evidence while retaining scope-wide validity."""

    if region == ALL_REGIONS:
        return dict(analysis)
    if region not in REGION_MEASUREMENTS:
        raise ProfessionalPageError(f"Unknown body-region filter: {region}")

    def matches(item: Any) -> bool:
        return _measurement_region(_mapping(item).get("measurement")) == region

    result = dict(analysis)
    result["angle_analyses"] = [
        item for item in _sequence(analysis.get("angle_analyses")) if matches(item)
    ]
    result["repetition_candidates"] = [
        item for item in _sequence(analysis.get("repetition_candidates")) if matches(item)
    ]
    camera = dict(_mapping(analysis.get("camera_sources")))
    for key in ("contributions", "agreement"):
        values = camera.get(key)
        if values is not None:
            camera[key] = [item for item in _sequence(values) if matches(item)] or None
    result["camera_sources"] = camera

    recommendations: list[Any] = []
    for item in _sequence(analysis.get("recommendations")):
        evidence = [_mapping(value) for value in _sequence(_mapping(item).get("evidence"))]
        measured = [value.get("measurement") for value in evidence if value.get("measurement")]
        if not measured or any(_measurement_region(value) == region for value in measured):
            recommendations.append(item)
    result["recommendations"] = recommendations
    return result


def method_snapshot(records: Sequence[Mapping[str, Any]], method: str) -> MethodSnapshot:
    """Return the worst valid method result in the current person/frame selection."""

    key = method.lower()
    assessed = 0
    candidates: list[tuple[float, int, float | None, Mapping[str, Any], Mapping[str, Any]]] = []
    for record in records:
        method_result = _mapping(record.get(key))
        if not method_result:
            continue
        assessed += 1
        selected = _mapping(method_result.get("selected"))
        score = _finite(selected.get("final_score"))
        frame = _integer(record.get("frame_number"))
        if score is not None:
            candidates.append(
                (
                    score,
                    frame if frame is not None else -1,
                    _finite(record.get("timestamp_sec")),
                    selected,
                    method_result,
                )
            )
    if not candidates:
        return MethodSnapshot(
            method.upper(), None, 0, assessed, None, None, None, None, None, None, None
        )
    score, frame, timestamp, selected, method_result = max(
        candidates, key=lambda row: (row[0], -(row[1] if row[1] >= 0 else math.inf))
    )
    interpretation = next(
        (
            str(selected[name])
            for name in ("action_level", "action_category", "risk_level")
            if selected.get(name)
        ),
        None,
    )
    side = method_result.get("selected_side") or selected.get("side")
    return MethodSnapshot(
        method.upper(),
        score,
        len(candidates),
        assessed,
        frame if frame >= 0 else None,
        timestamp,
        _finite(_mapping(method_result.get("left")).get("final_score")),
        _finite(_mapping(method_result.get("right")).get("final_score")),
        str(side) if side is not None else None,
        str(method_result["selection_reason"])
        if method_result.get("selection_reason") is not None
        else None,
        interpretation,
    )


def _event_customdata(event: Any) -> Any:
    try:
        points = event.selection.points
    except AttributeError:
        points = _mapping(_mapping(event).get("selection")).get("points", ())
    first = _mapping(points[0]) if points else {}
    return first.get("customdata")


def _route_evidence(
    session: MutableMapping[str, Any],
    *,
    frame_number: Any,
    measurement: Any = None,
    region: Any = None,
    open_review: bool = True,
) -> bool:
    """Send a frame to the synchronized frame review, and go there.

    This used to write ``selected_frame`` into session state and stop. Nothing
    read it unless the reviewer happened to switch sections by hand, and the
    frame review was hidden behind Advanced mode, so every "open this evidence"
    button on the page did nothing observable. The section name is parked under
    a separate key because Streamlit refuses to let a widget's own key be written
    after that widget has been created in the same run; the application applies
    it on the next run, before the navigation control exists.
    """

    frame = _integer(frame_number)
    if frame is None:
        return False
    changed = session.get("selected_frame") != frame
    session["selected_frame"] = frame
    resolved = region if isinstance(region, str) else None
    if isinstance(measurement, str):
        session["professional_selected_measurement"] = measurement
        resolved = resolved or _measurement_region(measurement)
    if resolved in REGION_LABELS:
        changed = changed or session.get("selected_region") != resolved
        session["selected_region"] = resolved
    if open_review:
        session["pending_dashboard_section"] = FRAME_REVIEW_SECTION
    return changed


def _open_frame(
    session: MutableMapping[str, Any],
    *,
    frame_number: Any,
    measurement: Any = None,
    region: Any = None,
) -> None:
    """Route to a frame and rerun immediately so the jump is not a click behind."""

    if _route_evidence(session, frame_number=frame_number, measurement=measurement, region=region):
        st.rerun()
    elif _integer(frame_number) is not None:
        st.rerun()


def _analysis_scope_label(region: str) -> str:
    return "All body regions" if region == ALL_REGIONS else REGION_LABELS[region]


def _render_method_readiness_line(
    records: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    method: str,
) -> None:
    """How complete the method was, and which side the reported score came from."""

    readiness = {
        str(_mapping(item).get("method")): _mapping(item)
        for item in _sequence(analysis.get("method_readiness"))
    }
    snapshot = method_snapshot(records, method)
    status = _human(readiness.get(method.upper(), {}).get("status"))
    line = (
        f"Readiness: {status} · scored {snapshot.valid_result_count} of "
        f"{snapshot.assessed_record_count} assessed observations · at the worst frame the "
        f"left side scored {_format_number(snapshot.left_score, digits=0)} and the right "
        f"{_format_number(snapshot.right_score, digits=0)}, so "
        f"{_human(snapshot.selected_side)} was reported."
    )
    st.caption(line if snapshot.selection_reason is None else f"{line} {snapshot.selection_reason}")


def _render_recommendations(
    analysis: Mapping[str, Any], session: MutableMapping[str, Any], key_prefix: str
) -> None:
    recommendations = [_mapping(item) for item in _sequence(analysis.get("recommendations"))]
    st.markdown("#### Evidence-linked findings and next actions")
    if not recommendations:
        st.info("No recommendation is supported by the current evidence filter.")
        return
    for index, item in enumerate(recommendations):
        with st.container(border=True):
            st.markdown(
                f"**{_human(item.get('priority'))} · {item.get('title', 'Review evidence')}**"
            )
            st.write(str(item.get("observed_fact") or "Observed fact unavailable."))
            st.caption(str(item.get("rationale") or "No interpretation supplied."))
            evidence = [_mapping(value) for value in _sequence(item.get("evidence"))]
            point = next(
                (value for value in evidence if _integer(value.get("frame_number")) is not None),
                None,
            )
            if point is not None and st.button(
                f"Open supporting frame {_integer(point.get('frame_number'))} →",
                key=f"{key_prefix}_recommendation_{index}",
            ):
                _open_frame(
                    session,
                    frame_number=point.get("frame_number"),
                    measurement=point.get("measurement"),
                )
            urls = [str(value) for value in _sequence(item.get("source_urls")) if value]
            if urls:
                st.caption(
                    "Sources: "
                    + " · ".join(f"[reference {n + 1}]({url})" for n, url in enumerate(urls))
                )


@dataclass(frozen=True, slots=True)
class TimelineBundle:
    """Everything the when/where views need for one method, computed once."""

    method: str
    series: tuple[ScorePoint, ...]
    episodes: tuple[RiskEpisode, ...]
    loads: tuple[BodyPartLoad, ...]
    verdict: Verdict

    @property
    def shares(self):
        return band_shares(self.series, self.method)


def build_timeline(records: Sequence[Mapping[str, Any]], method: str) -> TimelineBundle:
    """Compute the score series, episodes, and contributions for one method."""

    series = score_series(records, method)
    episodes = risk_episodes(series, method)
    loads = body_part_load(series, method)
    return TimelineBundle(
        method=str(method).lower(),
        series=tuple(series),
        episodes=tuple(episodes),
        loads=tuple(loads),
        verdict=build_verdict(series, method, episodes=episodes, loads=loads),
    )


def _band_chip(label: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 10px;border-radius:999px;'
        f'font-size:0.78rem;font-weight:600;white-space:nowrap;">{label}</span>'
    )


def _render_verdict_banner(bundle: TimelineBundle) -> None:
    """The headline: what the score was, how much of the task was spent there."""

    result = bundle.verdict
    if not result.scored:
        st.error(f"**{result.headline}.** {result.detail}")
        return
    band = result.peak_band
    colour = band.colour if band else "#7b8aa3"
    st.markdown(
        f"""
<div style="border-left:6px solid {colour};background:rgba(255,255,255,0.03);
padding:14px 18px;border-radius:8px;margin-bottom:12px;">
  <div style="font-size:1.28rem;font-weight:650;color:#f6f8ff;line-height:1.35;">
    {result.headline}
  </div>
  <div style="color:#a8b6ce;margin-top:6px;font-size:0.94rem;line-height:1.5;">
    {result.detail[:1].upper() + result.detail[1:]}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_verdict_metrics(bundle: TimelineBundle, summary: Mapping[str, Any]) -> None:
    result = bundle.verdict
    columns = st.columns(4)
    peak_band = result.peak_band
    with columns[0]:
        st.metric(
            f"Worst {result.method}",
            result.peak_score if result.peak_score is not None else "None",
            help="The highest score any single observation reached in this selection.",
        )
        if peak_band is not None:
            st.markdown(_band_chip(peak_band.label, peak_band.colour), unsafe_allow_html=True)
    with columns[1]:
        st.metric(
            "Typical score",
            _format_number(result.typical_score, digits=1),
            help=(
                "Duration-weighted mean: the score the task sat at, rather than the one "
                "frame that was worst. A high peak with a low typical score is one awkward "
                "movement; both high is how the job is done."
            ),
        )
        if result.typical_band is not None:
            st.markdown(
                _band_chip(result.typical_band.label, result.typical_band.colour),
                unsafe_allow_html=True,
            )
    with columns[2]:
        st.metric(
            f"Time at {result.method} {result.action_threshold}+",
            f"{result.percent_at_action:.0f}%",
            help=(
                f"Share of scored time at or above {result.method} {result.action_threshold}, "
                "the level at which this method calls for action."
            ),
        )
        st.caption(
            f"{result.time_at_action_sec:.0f} s of {result.scored_duration_sec:.0f} s · "
            f"{result.episode_count} "
            + ("episode" if result.episode_count == 1 else "episodes")
        )
    with columns[3]:
        st.metric(
            "Worst moment at",
            clock(result.peak_timestamp_sec),
            help="Where in the recording the worst observation happened.",
        )
        st.caption(
            f"Frame {result.peak_frame_number}"
            if result.peak_frame_number is not None
            else "Frame unavailable"
        )
    st.caption(
        f"{result.method} covers the {METHOD_SCOPE.get(result.method, 'assessed regions')}. "
        "RNLE below is a task-level lifting calculation and is not scoped to this window."
    )


def _render_rnle_strip(summary: Mapping[str, Any]) -> None:
    rnle = _mapping(summary.get("rnle"))
    if not rnle:
        return
    index = _finite(rnle.get("lifting_index"))
    status = _human(rnle.get("validity_status"))
    with st.container(border=True):
        columns = st.columns([1, 1, 2.4])
        columns[0].metric("RNLE Lifting Index", _format_number(index, digits=2))
        columns[1].metric(
            "Recommended weight limit",
            _format_number(rnle.get("recommended_weight_limit_kg"), " kg", 2),
        )
        with columns[2]:
            st.markdown("**NIOSH lifting equation · whole task**")
            if index is None:
                missing = [
                    str(value).replace("_", " ") for value in _sequence(rnle.get("missing_inputs"))
                ]
                invalid = [
                    str(value).replace("_", " ") for value in _sequence(rnle.get("invalid_inputs"))
                ]
                reason = []
                if missing:
                    reason.append("still needs " + ", ".join(missing))
                if invalid:
                    reason.append("rejected " + ", ".join(invalid))
                st.warning(
                    f"No Lifting Index: {status}."
                    + (" It " + "; ".join(reason) + "." if reason else "")
                    + " Supply these in Run Analysis to complete it."
                )
            else:
                st.caption(f"Validity: {status}")
            st.caption(
                "Candidate movement cycles are not lift events and are never used as "
                "RNLE lifting frequency."
            )


def _render_worst_moments(
    bundle: TimelineBundle,
    session: MutableMapping[str, Any],
    key_prefix: str,
    *,
    limit: int = 6,
) -> None:
    """The ranked list of when the risk actually happened, each openable."""

    episodes = list(bundle.episodes)[:limit]
    result = bundle.verdict
    st.markdown("#### The worst moments, worst first")
    if not episodes:
        st.success(
            f"No stretch of this selection stayed at {result.method} "
            f"{result.action_threshold} or above for long enough to count as an episode."
        )
        return
    st.caption(
        f"A stretch of the task held at {result.method} {result.action_threshold} or above. "
        "Each one names the body part that, set back to neutral, would drop the score "
        "furthest at that moment. Open it to see the recorded frames from every camera."
    )
    for episode in episodes:
        band = episode.peak_band
        with st.container(border=True):
            head, action = st.columns([4, 1])
            with head:
                chip = _band_chip(band.label, band.colour) if band else ""
                st.markdown(
                    f"**#{episode.index} · {episode.time_range_text}** "
                    f"({episode.duration_sec:.1f} s) &nbsp; {chip} &nbsp; "
                    f"peaked at **{result.method} {episode.peak_score}**",
                    unsafe_allow_html=True,
                )
                if episode.drivers:
                    st.markdown(
                        "Driven by: "
                        + " · ".join(
                            f"**{driver.label}** {driver.score_text} ({driver.points_text})"
                            for driver in episode.drivers
                        )
                    )
                    angles = [
                        f"{driver.measurement.replace('_', ' ')} {driver.value_deg:.0f}°"
                        for driver in episode.drivers
                        if driver.measurement and driver.value_deg is not None
                    ]
                    if angles:
                        st.caption("Measured at the peak: " + " · ".join(angles))
                else:
                    st.caption("No component evidence was stored for this peak.")
            with action:
                st.caption(f"frame {episode.peak_frame_number}")
                if st.button(
                    "Open frames →",
                    key=f"{key_prefix}_episode_{episode.index}",
                    width="stretch",
                ):
                    _open_frame(
                        session,
                        frame_number=episode.peak_frame_number,
                        region=episode.drivers[0].region if episode.drivers else None,
                    )


def _render_when(
    bundle: TimelineBundle,
    session: MutableMapping[str, Any],
    key_prefix: str,
    *,
    other: TimelineBundle | None = None,
) -> None:
    """The time axis the Results page never had."""

    result = bundle.verdict
    if not bundle.series:
        st.info(result.detail)
        return
    highlight = _integer(session.get("selected_frame"))
    st.markdown("#### When in the task the risk happened")
    st.caption(
        "Read the strip first: the task runs left to right and each block is coloured by "
        "the worst action level reached in that slice. Click anywhere on the strip or the "
        "chart to open those frames."
    )
    ribbon = st.plotly_chart(
        risk_ribbon_figure(list(bundle.series), bundle.method, highlight_frame=highlight),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"{key_prefix}_ribbon_{bundle.method}",
    )
    frame, _ = selected_frame_from_event(ribbon)
    if frame is not None and frame != highlight:
        _open_frame(session, frame_number=frame)

    timeline = st.plotly_chart(
        task_risk_timeline_figure(
            list(bundle.series),
            bundle.method,
            episodes=list(bundle.episodes),
            highlight_frame=highlight,
        ),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"{key_prefix}_timeline_{bundle.method}",
    )
    frame, _ = selected_frame_from_event(timeline)
    if frame is not None and frame != highlight:
        _open_frame(session, frame_number=frame)

    st.plotly_chart(
        time_at_band_figure(bundle.shares, bundle.method),
        width="stretch",
        key=f"{key_prefix}_shares_{bundle.method}",
    )
    _render_worst_moments(bundle, session, key_prefix)

    with st.expander("Left against right, and the other method on the same axis"):
        st.caption(
            f"{result.method} reports whichever side scored worse, so the headline number "
            "belongs to one arm. These show both."
        )
        st.plotly_chart(
            side_comparison_figure(list(bundle.series), bundle.method),
            width="stretch",
            key=f"{key_prefix}_sides_{bundle.method}",
        )
        if other is not None and other.series:
            rula = bundle if bundle.method == "rula" else other
            reba = bundle if bundle.method == "reba" else other
            st.plotly_chart(
                compare_methods_figure(list(rula.series), list(reba.series)),
                width="stretch",
                key=f"{key_prefix}_methods_{bundle.method}",
            )


def _render_where(
    bundle: TimelineBundle,
    session: MutableMapping[str, Any],
    key_prefix: str,
) -> None:
    """Which body part the points came from, ranked by what it cost the score."""

    result = bundle.verdict
    if not bundle.series:
        st.info(result.detail)
        return
    selected_region = session.get("selected_region")
    st.markdown("#### Where in the body the score came from")
    st.caption(
        f"Bar length is what each part costs: set that component back to neutral, run "
        f"{result.method}'s own tables again, and see how far the score falls. A joint that "
        "sits at its maximum but changes nothing scores zero here, which is why this ranking "
        "and a list of non-neutral joints do not agree. Click any part to open its worst frame."
    )
    left, right = st.columns([1, 1.25])
    with left:
        event = st.plotly_chart(
            body_load_map_figure(
                list(bundle.loads), bundle.method, selected_region=selected_region
            ),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=f"{key_prefix}_bodymap_{bundle.method}",
        )
        frame, region = selected_frame_from_event(event)
        if frame is not None:
            _open_frame(session, frame_number=frame, region=region)
    with right:
        event = st.plotly_chart(
            contribution_figure(
                list(bundle.loads),
                bundle.method,
                kind="posture",
                selected_region=selected_region,
            ),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key=f"{key_prefix}_contrib_{bundle.method}",
        )
        frame, region = selected_frame_from_event(event)
        if frame is not None:
            _open_frame(session, frame_number=frame, region=region)

    st.plotly_chart(
        contribution_figure(list(bundle.loads), bundle.method, kind="task"),
        width="stretch",
        key=f"{key_prefix}_taskfactors_{bundle.method}",
    )
    st.caption(
        "Task conditions are not postures. They are properties of the job — the load, the "
        "grip, how often the movement repeats — and they change the score without any body "
        "part moving. Where one of these is assumed rather than measured, Why this score says so."
    )

    rows = [
        {
            "Body part": item.label,
            f"Adds to {result.method}": round(item.mean_points, 2),
            "Worst component": item.worst_score_text,
            "Worst adds": f"+{item.worst_points}",
            "Share of task costing points": f"{item.percent_of_scored_time:.0f}%",
            "Worst at": clock(item.peak_timestamp_sec),
            "Frame": item.peak_frame_number,
            "Measured": (
                f"{item.measurement.replace('_', ' ')} {item.peak_value_deg:.0f}°"
                if item.measurement and item.peak_value_deg is not None
                else "—"
            ),
        }
        for item in bundle.loads
        if item.kind == "posture"
    ]
    if rows:
        with st.expander("The same ranking as a table"):
            st.dataframe(rows, hide_index=True, width="stretch")


def _render_other_method(bundle: TimelineBundle, session: MutableMapping[str, Any], key: str) -> None:
    """A one-line reminder of what the method you are not reading said."""

    result = bundle.verdict
    if not result.scored:
        st.info(f"{result.method} produced no valid score in this selection.")
        return
    band = result.peak_band
    columns = st.columns([1, 1, 1, 1.1])
    columns[0].metric(f"Worst {result.method}", result.peak_score)
    columns[1].metric("Typical", _format_number(result.typical_score, digits=1))
    columns[2].metric(
        f"Time at {result.action_threshold}+", f"{result.percent_at_action:.0f}%"
    )
    with columns[3]:
        if band is not None:
            st.markdown(_band_chip(band.label, band.colour), unsafe_allow_html=True)
        if result.peak_frame_number is not None and st.button(
            f"Open worst {result.method} frame →", key=key, width="stretch"
        ):
            _open_frame(session, frame_number=result.peak_frame_number)


def _render_verdict_view(
    bundle: TimelineBundle,
    other: TimelineBundle,
    analysis: Mapping[str, Any],
    summary: Mapping[str, Any],
    session: MutableMapping[str, Any],
    key_prefix: str,
    *,
    include_recommendations: bool = True,
) -> None:
    """What this trial showed, in one screen, with a way into the evidence.

    ``include_recommendations`` is off in the default workflow. The findings list
    runs to seven long entries on a real trial, and putting it directly under the
    verdict pushed the timeline -- the one thing the page exists to show -- below
    the fold. In Summary mode it is rendered after the where-in-the-body section,
    where "what should I do about it" actually belongs.
    """

    _render_verdict_banner(bundle)
    _render_verdict_metrics(bundle, summary)
    if bundle.verdict.scored and bundle.verdict.peak_frame_number is not None:
        if st.button(
            f"Open the worst {bundle.verdict.method} moment "
            f"({clock(bundle.verdict.peak_timestamp_sec)}) →",
            key=f"{key_prefix}_open_peak",
            type="primary",
        ):
            _open_frame(session, frame_number=bundle.verdict.peak_frame_number)

    st.markdown(f"###### What {other.verdict.method} said about the same frames")
    _render_other_method(other, session, f"{key_prefix}_open_other_peak")
    _render_rnle_strip(summary)

    quality = _mapping(analysis.get("data_quality"))
    status = str(quality.get("status") or "insufficient_data")
    coverage = _mapping(quality.get("time_coverage"))
    message = (
        f"Measurement quality: {_human(status)}. "
        f"{quality.get('selected_record_count') or 0} observations · "
        f"{_format_number(coverage.get('covered_duration_sec'), ' s')} of timed coverage · "
        f"mean pose confidence {_format_number(quality.get('mean_pose_visibility'), digits=2)} · "
        f"{quality.get('camera_count') or 'unknown'} camera views. "
        "Pose confidence and ergonomic-method validity are separate; no composite score is made."
    )
    if status == "valid":
        st.success(message)
    elif status == "valid_with_limitations":
        st.warning(message)
    else:
        st.error(message)

    if include_recommendations:
        _render_recommendations(analysis, session, key_prefix)


def _render_exposure(
    records: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    session: MutableMapping[str, Any],
    key_prefix: str,
) -> None:
    st.caption(
        "Official RULA and REBA posture bands are displayed independently as observed exposure. "
        "They are not combined into a new score."
    )
    st.plotly_chart(
        posture_band_exposure_figure(analysis), width="stretch", key=f"{key_prefix}_bands"
    )
    st.plotly_chart(angle_summary_figure(analysis), width="stretch", key=f"{key_prefix}_angles")
    bouts: list[Mapping[str, Any]] = []
    for angle in _sequence(analysis.get("angle_analyses")):
        bouts.extend(_mapping(value) for value in _sequence(_mapping(angle).get("sustained_bouts")))
    st.markdown("#### Sustained non-neutral posture evidence")
    if not bouts:
        st.info("No sustained bout met the configured minimum duration in this selection.")
    else:
        rows = [
            {
                "Method": item.get("method"),
                "Measurement": _human(item.get("measurement")),
                "Criterion": item.get("criterion"),
                "Duration (s)": round(float(item.get("duration_sec", 0.0)), 2),
                "Start frame": item.get("start_frame_number"),
                "End frame": item.get("end_frame_number"),
            }
            for item in bouts
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        selected = st.selectbox(
            "Supporting sustained-posture event",
            range(len(bouts)),
            format_func=lambda index: (
                f"{_human(bouts[index].get('measurement'))} · "
                f"{float(bouts[index].get('duration_sec', 0.0)):.2f} s · "
                f"frame {bouts[index].get('start_frame_number')}"
            ),
            key=f"{key_prefix}_bout_select",
        )
        if st.button("Open sustained-posture evidence →", key=f"{key_prefix}_bout_open"):
            _open_frame(
                session,
                frame_number=bouts[selected].get("start_frame_number"),
                measurement=bouts[selected].get("measurement"),
            )


def _render_movement(
    records: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    session: MutableMapping[str, Any],
    key_prefix: str,
) -> None:
    # Avoid requiring the OpenCV video runtime merely to import pure helpers.
    from dashboard.analysis_service import angle_timeline_figure

    angles = [
        str(_mapping(item).get("measurement"))
        for item in _sequence(analysis.get("angle_analyses"))
        if _mapping(item).get("status") != "unavailable"
    ]
    selected_angles = st.multiselect(
        "Model-estimated angle trends",
        angles,
        default=angles[: min(4, len(angles))],
        format_func=_human,
        key=f"{key_prefix}_movement_angles",
        max_selections=6,
    )
    event = st.plotly_chart(
        angle_timeline_figure(list(records), selected_angles),
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"{key_prefix}_movement_timeline",
    )
    frame = _event_customdata(event)
    if frame is not None and _integer(frame) != _integer(session.get("selected_frame")):
        _open_frame(session, frame_number=frame)
    st.caption("Select a chart point to open its source frame in the frame review.")

    candidates = [_mapping(item) for item in _sequence(analysis.get("repetition_candidates"))]
    st.markdown("#### Candidate repetition events requiring task confirmation")
    st.warning(
        "These are signal peaks, not confirmed work cycles or lifts. Candidate event rate is "
        "not RNLE lifting frequency and is not an ergonomic risk score."
    )
    if not candidates:
        st.info("No configured movement signal is available under this region filter.")
        return
    st.dataframe(
        [
            {
                "Measurement": _human(item.get("measurement")),
                "Status": _human(item.get("status")),
                "Candidate events": item.get("candidate_count")
                if item.get("candidate_count") is not None
                else "Unavailable",
                "Candidate events/min (review only)": _format_number(item.get("rate_per_minute")),
                "Analyzed duration (s)": _format_number(item.get("analysis_duration_sec")),
                "Prominence (deg)": _format_number(item.get("prominence_deg")),
                "Minimum interval (s)": _format_number(item.get("minimum_interval_sec")),
            }
            for item in candidates
        ],
        hide_index=True,
        width="stretch",
    )
    evidence = [
        (_mapping(item).get("measurement"), _mapping(point))
        for item in candidates
        for point in _sequence(_mapping(item).get("event_evidence"))
    ]
    if evidence:
        chosen = st.selectbox(
            "Candidate-event evidence",
            range(len(evidence)),
            format_func=lambda index: (
                f"{_human(evidence[index][0])} · frame "
                f"{evidence[index][1].get('frame_number')} · "
                f"{_format_number(evidence[index][1].get('timestamp_sec'), ' s', 2)}"
            ),
            key=f"{key_prefix}_candidate_select",
        )
        if st.button("Open candidate-event frame →", key=f"{key_prefix}_candidate_open"):
            _open_frame(
                session,
                frame_number=evidence[chosen][1].get("frame_number"),
                measurement=evidence[chosen][0],
            )


CONDITION_LABELS = {
    "trunk_twisted": "Trunk twisted",
    "trunk_side_bent": "Trunk side-bent",
    "trunk_twisted_or_side_flexed": "Trunk twisted or side-flexed",
    "neck_twisted": "Neck twisted",
    "neck_side_bent": "Neck side-bent",
    "neck_twisted_or_side_flexed": "Neck twisted or side-flexed",
    "seated": "Seated",
    "legs_supported_and_balanced": "Legs supported and balanced",
    "legs_weight_bearing": "Leg weight bearing",
    "static_posture_over_one_minute": "Posture held over a minute",
    "repeated_more_than_four_per_minute": "Repeated more than 4/min",
    "rapid_large_posture_change_or_unstable_base": "Rapid change or unstable base",
    "rula_load_type": "RULA load exposure",
    "load_kg": "Handled load",
    "coupling": "Hand coupling",
    "shock_or_rapid_force": "Shock or rapid force",
}


def _condition_value(value: Any) -> str:
    """Render a condition value for display, keeping unknown distinct from no."""

    if value is None:
        return "Unknown"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value).replace("_", " ")


def _condition_label(name: str) -> str:
    if name in CONDITION_LABELS:
        return CONDITION_LABELS[name]
    for side in ("left", "right"):
        if name.startswith(f"{side}_"):
            stem = name[len(side) + 1 :].replace("_", " ")
            return f"{side.title()} {stem}"
    return name.replace("_", " ").capitalize()


def _render_why(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    key_prefix: str,
) -> None:
    """Show what turned the measurements into penalties, and where each came from.

    This is the view the results were missing. Every other view says what the
    score was; none said why. RULA and REBA are built from yes/no conditions on
    top of the angles, and until a reviewer can see which conditions fired, how
    often, and whether each was measured or assumed, a score is a number with no
    argument behind it and nothing to act on.
    """

    st.markdown("#### What drove the score")
    if not records:
        st.info("No frames are in scope. Widen the selection.")
        return

    fired: Counter[str] = Counter()
    sources: dict[str, str] = {}
    departures: dict[str, list[float]] = {}
    scoped = 0
    for record in records:
        conditions = _mapping(record.get("task_conditions"))
        if not conditions:
            continue
        scoped += 1
        values = _mapping(conditions.get("values"))
        for name, source in _mapping(conditions.get("sources")).items():
            sources.setdefault(str(name), str(source))
        for name, value in values.items():
            if value is True:
                fired[str(name)] += 1
        for name, evidence in _mapping(conditions.get("evidence")).items():
            departure = _finite(_mapping(evidence).get("departure_deg"))
            if departure is not None:
                departures.setdefault(str(name), []).append(departure)

    if not scoped:
        st.info("These results predate per-condition evidence. Re-run the analysis to record it.")
        return

    rows = []
    for name, count in fired.most_common():
        margins = departures.get(name)
        rows.append(
            {
                "Condition": _condition_label(name),
                "Frames": count,
                "Share": f"{100 * count / scoped:.0f}%",
                "Source": sources.get(name, "unknown"),
                "Past neutral": (f"{max(margins):.0f}°" if margins else "—"),
            }
        )
    if rows:
        st.caption(
            "A condition adds a penalty to its body-part score. 'Source' is where the "
            "answer came from: measured from the footage, entered by you, or filled from "
            "an assumption profile. 'Past neutral' is how far the worst frame went beyond "
            "this subject's own resting posture."
        )
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.success("No posture condition was triggered anywhere in the selected frames.")

    assumed = sorted({name for name, source in sources.items() if source == "assumed"})
    if assumed:
        st.warning(
            "Filled from an assumption profile rather than observed: "
            + ", ".join(_condition_label(name) for name in assumed)
            + ". Confirm any of these that would change the outcome."
        )

    conflicts = [
        item
        for record in records
        for item in _sequence(_mapping(record.get("task_conditions")).get("conflicts"))
    ]
    if conflicts:
        with st.expander(f"Entered values the footage disagreed with ({len(conflicts)})"):
            st.caption("Your value was used. The measurement is shown for reference.")
            st.dataframe(
                [
                    {
                        "Condition": _condition_label(str(_mapping(item).get("condition"))),
                        "You entered": _condition_value(_mapping(item).get("observed_value")),
                        "Footage measured": _condition_value(_mapping(item).get("derived_value")),
                    }
                    for item in conflicts[:50]
                ],
                width="stretch",
                hide_index=True,
            )

    baseline = _mapping(_mapping(summary.get("task_input_provenance")).get("posture_baseline"))
    if baseline:
        with st.expander("The resting posture each condition was measured against"):
            st.caption(str(baseline.get("note", "")))
            neutral = _mapping(baseline.get("neutral_deg"))
            if neutral:
                st.dataframe(
                    [
                        {"Measurement": _condition_label(str(k)), "Resting level": f"{v}°"}
                        for k, v in sorted(neutral.items())
                    ],
                    width="stretch",
                    hide_index=True,
                )


def _render_trust(
    analysis: Mapping[str, Any],
    summary: Mapping[str, Any],
    key_prefix: str,
) -> None:
    """Camera evidence, plus what the score rested on that was weaker than measured."""

    _render_quality(analysis, summary, key_prefix)

    st.divider()
    st.markdown("#### What the score rested on")

    provenance = _mapping(summary.get("task_input_provenance"))
    derived = _sequence(provenance.get("derived_fields"))
    observed = _sequence(provenance.get("observed_fields"))
    assumed = _sequence(provenance.get("assumed_fields"))
    unknown = _sequence(provenance.get("unknown_fields"))
    columns = st.columns(4)
    for column, label, values, help_text in (
        (columns[0], "Measured", derived, "Derived from the footage."),
        (columns[1], "Entered", observed, "Supplied by the reviewer."),
        (columns[2], "Assumed", assumed, "Filled from a documented profile."),
        (columns[3], "Unknown", unknown, "Nothing supplied these."),
    ):
        with column:
            st.metric(label, len(values), help=help_text)

    if assumed:
        st.caption("Assumed: " + ", ".join(_condition_label(str(n)) for n in assumed))

    estimate = _mapping(summary.get("rnle_input_estimate"))
    if estimate:
        with st.expander("Lifting geometry measured from the video"):
            out_of_range = [
                name
                for name, source in _mapping(estimate.get("sources")).items()
                if str(source).endswith("out_of_range")
            ]
            if out_of_range:
                st.warning(
                    "Outside the RNLE valid range: "
                    + ", ".join(str(n).replace("_", " ") for n in out_of_range)
                )
            st.caption(str(estimate.get("scale_basis", "")))
            for note in _sequence(estimate.get("notes")):
                st.caption(f"• {note}")


def _render_quality(
    analysis: Mapping[str, Any], summary: Mapping[str, Any], key_prefix: str
) -> None:
    st.caption(
        "Acquisition, pose-model confidence, calibration, camera evidence, and ergonomic-method "
        "validity are separate quality dimensions."
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            data_quality_coverage_figure(analysis), width="stretch", key=f"{key_prefix}_quality"
        )
    with right:
        st.plotly_chart(
            camera_evidence_figure(analysis), width="stretch", key=f"{key_prefix}_cameras"
        )
    summary_quality = _mapping(summary.get("data_quality"))
    metrics = st.columns(3)
    metrics[0].metric(
        "Ambiguous multi-person frames withheld",
        summary_quality.get("ambiguous_multi_person_frames", "Unavailable"),
    )
    metrics[1].metric(
        "Timestamp fallback frames", summary_quality.get("timestamp_fallback_frames", "Unavailable")
    )
    metrics[2].metric("Tracking mode", summary_quality.get("tracking_mode", "Unavailable"))
    quality = _mapping(analysis.get("data_quality"))
    warnings = [str(value) for value in _sequence(quality.get("warnings"))]
    if warnings:
        with st.expander("Quality limitations", expanded=True):
            for warning in warnings:
                st.warning(warning)


def _report_scope(
    context: Mapping[str, Any], task: Mapping[str, Any], frame_range: tuple[int, int], region: str
) -> dict[str, Any]:
    return {
        "participant_id": str(context.get("participant_id", "Unavailable")),
        "trial": str(task.get("name") or "Unavailable"),
        "first_frame": frame_range[0],
        "last_frame": frame_range[1],
        "body_region": _analysis_scope_label(region),
        "rnle_scope": "whole task",
    }


def _csv_bytes(analysis: Mapping[str, Any]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(professional_csv_rows(analysis))
    return stream.getvalue().encode("utf-8-sig")


def _safe_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return text or "analysis"


def _render_methods_report(
    context: Mapping[str, Any],
    task: Mapping[str, Any],
    analysis: Mapping[str, Any],
    frame_range: tuple[int, int],
    region: str,
    key_prefix: str,
) -> None:
    st.plotly_chart(
        method_readiness_figure(analysis), width="stretch", key=f"{key_prefix}_readiness"
    )
    st.caption(
        "Method readiness and data quality apply to the selected person/frame window; RNLE applies to the whole task."
    )
    warnings = [str(value) for value in _sequence(analysis.get("warnings"))]
    for warning in warnings:
        st.info(warning)
    st.markdown("#### Authoritative method and measurement sources")
    for source in _sequence(analysis.get("provenance")):
        item = _mapping(source)
        if item.get("url"):
            st.markdown(
                f"- **{item.get('identifier', 'Source')}** — [{item.get('title', 'Reference')}]({item['url']})"
            )

    scope = _report_scope(context, task, frame_range, region)
    scope_markdown = "\n".join(
        f"- {key.replace('_', ' ').title()}: {value}" for key, value in scope.items()
    )
    markdown = (
        "## Applied dashboard scope\n\n"
        + scope_markdown
        + "\n\n"
        + professional_markdown_report(analysis)
    )
    export = {"scope": scope, "analysis": analysis}
    base = _safe_filename(
        f"participant_{scope['participant_id']}_{scope['trial']}_{frame_range[0]}_{frame_range[1]}"
    )
    columns = st.columns(3)
    columns[0].download_button(
        "Download evidence report (.md)",
        markdown.encode("utf-8"),
        f"{base}.md",
        "text/markdown",
        key=f"{key_prefix}_download_md",
    )
    columns[1].download_button(
        "Download evidence table (.csv)",
        _csv_bytes(analysis),
        f"{base}.csv",
        "text/csv",
        key=f"{key_prefix}_download_csv",
    )
    columns[2].download_button(
        "Download machine-readable result (.json)",
        json.dumps(export, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        f"{base}.json",
        "application/json",
        key=f"{key_prefix}_download_json",
    )


def comparison_rows(
    *,
    catalog: Sequence[Mapping[str, Any]],
    selected_context_keys: Sequence[str],
    task: Mapping[str, Any],
    sampling: SamplingValue,
    detect_current_results: ResultDetector,
    load_summary: SummaryLoader,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load comparison summaries only; never index another participant's frame JSONL."""

    selected = set(selected_context_keys)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for item in catalog:
        context_key = str(item.get("key"))
        if context_key not in selected:
            continue
        participant_id = str(item.get("participant_id"))
        study_folder = str(_mapping(item.get("study")).get("folder_name"))
        matching_task = next(
            (
                _mapping(candidate)
                for candidate in _sequence(item.get("tasks"))
                if _mapping(candidate).get("folder_name") == task.get("folder_name")
            ),
            None,
        )
        label = f"Participant {participant_id}"
        if matching_task is None:
            errors.append(f"{label}: matching trial is unavailable")
            continue
        result = detect_current_results(
            study_folder, participant_id, str(matching_task.get("folder_name")), sampling
        )
        if not result or not result.get("dir"):
            errors.append(f"{label}: no result at the selected sampling")
            continue
        try:
            summary = load_summary(str(result["dir"]))
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{label}: could not read summary ({exc})")
            continue
        rula = _mapping(summary.get("rula_summary"))
        reba = _mapping(summary.get("reba_summary"))
        rnle = _mapping(summary.get("rnle"))
        processed = _integer(summary.get("processed_frames"))
        pose = _integer(summary.get("pose_detected_frames"))
        coverage = None if not processed or pose is None else pose / processed * 100.0
        rows.append(
            {
                "Participant": participant_id,
                "RULA maximum": _format_number(rula.get("maximum_score"), digits=0),
                "RULA valid observations": rula.get("observation_count", "Unavailable"),
                "REBA maximum": _format_number(reba.get("maximum_score"), digits=0),
                "REBA valid observations": reba.get("observation_count", "Unavailable"),
                "RNLE LI (task-level)": _format_number(rnle.get("lifting_index"), digits=2),
                "RNLE validity": _human(rnle.get("validity_status")),
                "Pose-detected share": _format_number(coverage, "%"),
            }
        )
    return rows, errors


def _render_compare(
    catalog: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    task: Mapping[str, Any],
    sampling: SamplingValue,
    detect_current_results: ResultDetector,
    load_summary: SummaryLoader,
    key_prefix: str,
) -> None:
    comparable = [
        item
        for item in catalog
        if any(
            _mapping(candidate).get("folder_name") == task.get("folder_name")
            for candidate in _sequence(item.get("tasks"))
        )
    ]
    keys = [str(item.get("key")) for item in comparable]
    current_key = str(context.get("key"))
    selected = st.multiselect(
        "Participants",
        keys,
        default=[current_key] if current_key in keys else keys[:1],
        format_func=lambda key: next(
            (
                f"Participant {item.get('participant_id')} · {item.get('date_label', 'date unavailable')}"
                for item in comparable
                if str(item.get("key")) == key
            ),
            "Participant unavailable",
        ),
        key=f"{key_prefix}_compare_participants",
    )
    st.caption(
        "Comparison uses whole-trial summaries at the same sampling. Missing method results remain "
        "Unavailable; RULA, REBA, and RNLE are not combined."
    )
    rows, errors = comparison_rows(
        catalog=catalog,
        selected_context_keys=selected,
        task=task,
        sampling=sampling,
        detect_current_results=detect_current_results,
        load_summary=load_summary,
    )
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No comparable summaries are available for this selection.")
    for error in errors:
        st.warning(error)


def render_professional_results(
    *,
    context: Mapping[str, Any],
    task: Mapping[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    get_processed_fps: ProcessedFpsLoader,
    detect_current_results: ResultDetector,
    load_bundle: BundleLoader,
    load_summary: SummaryLoader,
    analyze_cached: AnalysisCache,
    session: MutableMapping[str, Any],
) -> None:
    """Render the complete Results workflow using injected project services."""

    participant_id = str(context.get("participant_id"))
    study_folder = str(_mapping(context.get("study")).get("folder_name"))
    task_folder = str(task.get("folder_name"))
    key_seed = f"{context.get('key')}|{task.get('task_id')}|professional-results"
    key_prefix = "pro_" + hashlib.sha1(key_seed.encode("utf-8")).hexdigest()[:10]

    st.subheader("Results")
    samplings = list(get_processed_fps(study_folder, participant_id, task_folder))
    if not samplings:
        st.info("No traceable analysis is available. Configure and run it in Run Analysis.")
        return

    # Simple mode answers "what did this trial show?" with sensible defaults:
    # whole recording, every body region, one method at a time, read top to
    # bottom. Advanced mode exposes every scoping and threshold control exactly
    # as before. Both read the same computed evidence; nothing is calculated
    # differently.
    detail_mode = st.segmented_control(
        "Result detail",
        ["Summary", "Advanced analysis"],
        default="Summary",
        key=f"{key_prefix}_detail_mode",
    )
    advanced = detail_mode == "Advanced analysis"

    if advanced:
        sampling = st.selectbox(
            "Result sampling",
            samplings,
            format_func=lambda value: "Native frame rate"
            if value is None
            else f"{value} frames/second",
            key=f"{key_prefix}_sampling",
        )
    else:
        sampling = samplings[0]
    result = detect_current_results(study_folder, participant_id, task_folder, sampling)
    if not result or not result.get("dir"):
        st.error("The selected result is incomplete or not scientifically traceable.")
        return
    try:
        bundle = load_bundle(str(result["dir"]))
    except (OSError, TypeError, ValueError) as exc:
        st.error(f"Could not load the selected result: {exc}")
        return
    records = [item for item in _sequence(bundle.get("records")) if isinstance(item, Mapping)]
    if not records:
        st.error("The selected result contains no indexed pose observations.")
        return
    participants = sorted(
        {
            str(item.get("person_identifier"))
            for item in records
            if item.get("person_identifier") is not None
        }
    )
    if not participants:
        st.error("No tracked person identifier is present in the selected result.")
        return
    if advanced and len(participants) > 1:
        selected_person = st.selectbox("Tracked person", participants, key=f"{key_prefix}_person")
    else:
        selected_person = participants[0]
    person_frames = sorted(
        frame
        for item in records
        if str(item.get("person_identifier")) == selected_person
        and (frame := _integer(item.get("frame_number"))) is not None
    )
    if not person_frames:
        st.error("The selected person has no valid frame numbers.")
        return
    config = _mapping(bundle.get("config"))
    quality_config = _mapping(config.get("quality"))
    temporal_config = _mapping(config.get("temporal_aggregation"))
    if advanced:
        filter_columns = st.columns([1.6, 1, 1])
        with filter_columns[0]:
            frame_range = st.slider(
                "Inclusive frame range",
                person_frames[0],
                person_frames[-1],
                (person_frames[0], person_frames[-1]),
                key=f"{key_prefix}_frames",
            )
        with filter_columns[1]:
            region = st.selectbox(
                "Body-region evidence",
                [ALL_REGIONS, *REGION_LABELS],
                format_func=lambda value: "All body regions"
                if value == ALL_REGIONS
                else REGION_LABELS[value],
                key=f"{key_prefix}_region",
            )
        with filter_columns[2]:
            view = st.selectbox("Results view", RESULT_VIEWS, key=f"{key_prefix}_view")
    else:
        frame_range = (person_frames[0], person_frames[-1])
        region = ALL_REGIONS
        # The default workflow reads one page top to bottom instead of choosing
        # between six tabs of whole-trial aggregates. Advanced mode still gates
        # scoping and thresholds and still exposes every view individually.
        view = None
    settings_expander = (
        st.expander("Evidence-analysis settings", expanded=False) if advanced else None
    )
    if settings_expander is not None:
        with settings_expander:
            settings = st.columns(5)
            minimum_visibility = settings[0].number_input(
                "Minimum visibility",
                0.0,
                1.0,
                float(quality_config.get("minimum_landmark_visibility", 0.5)),
                0.05,
                key=f"{key_prefix}_visibility",
            )
            maximum_gap = settings[1].number_input(
                "Maximum gap (s)",
                0.01,
                5.0,
                float(temporal_config.get("maximum_gap_sec", 0.25)),
                0.01,
                key=f"{key_prefix}_gap",
            )
            sustained_min = settings[2].number_input(
                "Sustained minimum (s)",
                0.05,
                120.0,
                float(temporal_config.get("sustained_min_duration_sec", 1.0)),
                0.05,
                key=f"{key_prefix}_sustained",
            )
            prominence = settings[3].number_input(
                "Cycle prominence (deg)", 0.1, 180.0, 10.0, 0.5, key=f"{key_prefix}_prominence"
            )
            minimum_interval = settings[4].number_input(
                "Cycle minimum interval (s)", 0.05, 120.0, 0.5, 0.05, key=f"{key_prefix}_interval"
            )
    else:
        minimum_visibility = float(quality_config.get("minimum_landmark_visibility", 0.5))
        maximum_gap = float(temporal_config.get("maximum_gap_sec", 0.25))
        sustained_min = float(temporal_config.get("sustained_min_duration_sec", 1.0))
        prominence = 10.0
        minimum_interval = 0.5

    scoped_records = scope_records(
        records, person_identifier=selected_person, frame_range=frame_range
    )
    if not scoped_records:
        st.warning("No pose observations match the selected person and frame range.")
        return
    summary = _mapping(bundle.get("summary"))
    rnle = _mapping(summary.get("rnle"))
    store = bundle.get("store")
    fingerprint = getattr(store, "fingerprint", None)
    if fingerprint is None:
        raise ProfessionalPageError("Result bundle does not expose a fingerprinted result store")
    try:
        analysis = analyze_cached(
            str(fingerprint.path),
            int(fingerprint.size_bytes),
            int(fingerprint.mtime_ns),
            selected_person,
            int(frame_range[0]),
            int(frame_range[1]),
            float(minimum_visibility),
            float(maximum_gap),
            float(sustained_min),
            float(prominence),
            float(minimum_interval),
            json.dumps(rnle, sort_keys=True) if rnle else "",
        )
    except (OSError, TypeError, ValueError) as exc:
        st.error(f"Could not compute professional evidence: {exc}")
        return
    selection_analysis = dict(analysis)
    region_analysis = filter_analysis_region(selection_analysis, region)

    # One method drives the page. RULA and REBA cover different regions and
    # different scales, and interleaving them through every chart was a large
    # part of why nothing on this page could be read at a glance.
    method = (
        st.segmented_control(
            "Assessment method",
            ["REBA", "RULA"],
            default=session.get("professional_method", "REBA"),
            key=f"{key_prefix}_method",
            help=(
                "REBA covers the whole body including legs and load handling. "
                "RULA covers the upper limb, neck and trunk. Both are calculated "
                "on every frame; this chooses which one the page follows."
            ),
        )
        or "REBA"
    )
    session["professional_method"] = method
    other_method = "RULA" if method == "REBA" else "REBA"
    timeline = build_timeline(scoped_records, method)
    other_timeline = build_timeline(scoped_records, other_method)

    st.caption(
        f"Scope: {selected_person} · frames {frame_range[0]}–{frame_range[1]} · "
        f"{_analysis_scope_label(region)} · "
        f"{clock(timeline.verdict.scored_duration_sec)} of scored time. "
        "Method readiness and data quality are selection-wide; RNLE is whole-task."
    )
    _render_method_readiness_line(scoped_records, region_analysis, method)

    if not advanced:
        _render_verdict_view(
            timeline,
            other_timeline,
            region_analysis,
            summary,
            session,
            key_prefix,
            include_recommendations=False,
        )
        st.divider()
        _render_when(timeline, session, key_prefix, other=other_timeline)
        st.divider()
        _render_where(timeline, session, key_prefix)
        st.divider()
        _render_recommendations(region_analysis, session, key_prefix)
        st.divider()
        with st.expander("Why this score — the conditions behind the numbers"):
            _render_why(scoped_records, summary, key_prefix)
        with st.expander("Posture-band exposure and observed range of motion"):
            _render_exposure(scoped_records, region_analysis, session, key_prefix)
        with st.expander("Angle traces and candidate repetition events"):
            _render_movement(scoped_records, region_analysis, session, key_prefix)
        with st.expander("Can I trust it — quality, cameras, and what was assumed"):
            _render_trust(region_analysis, summary, key_prefix)
        with st.expander("Download the report, and compare participants"):
            _render_methods_report(
                context, task, region_analysis, frame_range, region, key_prefix
            )
            st.divider()
            _render_compare(
                catalog, context, task, sampling, detect_current_results, load_summary, key_prefix
            )
        return

    resolved = _LEGACY_VIEW_NAMES.get(view or "", view) or RESULT_VIEWS[0]
    if resolved == "The verdict":
        _render_verdict_view(
            timeline, other_timeline, region_analysis, summary, session, key_prefix
        )
    elif resolved == "When in the task":
        _render_when(timeline, session, key_prefix, other=other_timeline)
        st.divider()
        _render_movement(scoped_records, region_analysis, session, key_prefix)
    elif resolved == "Where in the body":
        _render_where(timeline, session, key_prefix)
        st.divider()
        _render_exposure(scoped_records, region_analysis, session, key_prefix)
    elif resolved == "Why this score":
        _render_why(scoped_records, summary, key_prefix)
    elif resolved == "Can I trust it":
        _render_trust(region_analysis, summary, key_prefix)
    else:
        _render_methods_report(context, task, region_analysis, frame_range, region, key_prefix)
        st.divider()
        _render_compare(
            catalog, context, task, sampling, detect_current_results, load_summary, key_prefix
        )


__all__ = [
    "FRAME_REVIEW_SECTION",
    "RESULT_VIEWS",
    "ProfessionalPageError",
    "TimelineBundle",
    "build_timeline",
    "comparison_rows",
    "filter_analysis_region",
    "method_snapshot",
    "render_professional_results",
    "scope_records",
]
