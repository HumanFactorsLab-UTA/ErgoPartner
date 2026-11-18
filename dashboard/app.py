"""Participant-centered ergonomic study analysis dashboard.

Launch: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

from dashboard.analysis_service import (
    REGION_LABELS,
    DashboardDataError,
    angle_timeline_figure,
    interpret_frame,
    read_video_frame,
    score_timeline_figure,
    stick_figure_3d,
    video_metadata,
)
from dashboard.camera_placement import PlacementError
from dashboard.data_loader import (
    detect_current_results,
    find_videos,
    get_processed_fps,
    get_task_results_path,
    participant_catalog,
)
from dashboard.processor_wrapper import (
    build_processing_config,
    launch_batch_thread,
    launch_processing_thread,
    reconcile_processing_thread,
)
from dashboard.professional_analysis import analyze_professional_records
from dashboard.professional_page import FRAME_REVIEW_SECTION, render_professional_results
from dashboard.recording_service import (
    CameraSpec,
    RecordingError,
    get_recording_service,
)
from dashboard.result_store import (
    JsonlResultStore,
    ResultStoreError,
    build_legacy_camera_refs,
    open_result_store,
)
from ergonomics.condition_resolution import NOT_OBSERVABLE_FROM_VIDEO
from paths import BRAND_DIR, CONFIG_PATH, STUDIES_DIR_STR, brand_logo_uri
from pipeline.recording_sources import recorded_path, recording_entry, trial_manifest

st.set_page_config(
    page_title="ErgoPartner | Study Analysis",
    page_icon=str(BRAND_DIR / "ergopartner-256.png"),
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f6f8ff;
        --muted: #a8b6ce;
        --line: #263858;
        --surface: #080d18;
        --soft: #0c1425;
        --accent: #2f81f7;
        --risk: #ff334f;
    }
    .stApp { background: #05070c; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(5,7,12,.96); }
    [data-testid="stSidebar"] { background: #070b13; border-right: 1px solid #263858; }
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line);
        border-radius: 10px; padding: 14px 16px;
        box-shadow: inset 0 3px 0 #2f81f7;
    }
    [data-testid="stMetricValue"] { color: #f6f8ff; }
    [data-testid="stMetricLabel"] { color: #a8b6ce; }
    .block-container { padding-top: 3.75rem; max-width: 1500px; }
    h1 { font-size: 2.5rem !important; line-height: 1.12 !important; }
    .eyebrow {
        color: #5aa2ff; font-size: .77rem; font-weight: 700;
        letter-spacing: .08em; text-transform: uppercase; margin-bottom: .25rem;
    }
    .brand {
        display: flex; align-items: center; gap: .7rem;
        margin: 0 0 1.15rem 0; padding-bottom: .9rem;
        border-bottom: 1px solid var(--line);
    }
    .brand img { width: min(202px, 100%); height: auto; }
    .brand-tag {
        color: var(--muted); font-size: .78rem; letter-spacing: .04em;
        border-left: 1px solid var(--line); padding-left: .7rem; margin-left: .15rem;
    }
    @media (max-width: 640px) { .brand-tag { display: none; } }
    .page-intro { color: var(--muted); max-width: 860px; margin-top: -.5rem; }
    .result-card {
        background: #080d18; border: 1px solid var(--line); border-radius: 10px;
        padding: 15px 16px; margin-bottom: 10px;
    }
    .result-card h4 { margin: 0 0 5px 0; font-size: 1rem; }
    .result-card p { margin: 0; color: var(--muted); font-size: .9rem; }
    .badge {
        display: inline-block; padding: 2px 8px; border-radius: 999px;
        font-size: .75rem; font-weight: 650; background: #10264a; color: #8fc0ff;
        margin-right: 7px;
    }
    .badge-high { background: #42131c; color: #ff8091; }
    [data-testid="stSegmentedControl"] [role="radiogroup"] {
        display: flex; flex-wrap: wrap; gap: 4px; width: 100%;
        padding: 4px; border: 1px solid var(--line); border-radius: 10px;
        background: #080d18;
    }
    [data-testid="stSegmentedControl"] button {
        flex: 1 1 145px; min-height: 42px; justify-content: center;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {
        color: #ffffff !important; background: #173d72 !important;
        box-shadow: inset 0 -2px 0 #ff334f;
    }
    input, textarea, [data-baseweb="select"] > div {
        background-color: #0c1425 !important; color: #f6f8ff !important;
        border-color: #263858 !important;
    }
    .stButton button {
        border-color: #31558c; color: #f6f8ff; background: #101c31;
    }
    .stButton button[kind="primary"] {
        background: #1f63c5; border-color: #3c8cff; color: #ffffff;
    }
    .stButton button:focus-visible, input:focus-visible, [data-testid="stSegmentedControl"] button:focus-visible {
        outline: 3px solid #ff334f !important; outline-offset: 2px;
    }
    [data-testid="stAlert"] { background: #0c1425; border-color: #31558c; }
    hr { border-color: #263858 !important; }
    .risk-strip {
        background: #080d18; border: 1px solid #263858; border-left: 5px solid #ff334f;
        border-radius: 8px; padding: 12px 14px; margin: 8px 0;
    }
    .status-step {
        background: #080d18; border: 1px solid #263858; border-radius: 8px;
        padding: 10px 12px; min-height: 74px;
    }
    .status-step.active { border-color: #2f81f7; box-shadow: inset 0 0 0 1px #2f81f7; }
    .status-step.done { border-color: #2a9d78; }
    .status-step.error { border-color: #ff334f; }
    .small-note { color: var(--muted); font-size: .82rem; }
    @media (max-width: 1400px) {
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
        div[data-testid="stColumn"] {
            flex: 1 1 210px !important;
            min-width: 210px !important;
        }
    }
    @media (max-width: 900px) {
        [data-testid="stSegmentedControl"] button {
            flex: 1 1 160px;
            padding: 8px 6px;
        }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        h1 { font-size: 2rem !important; }
    }
    @media (max-width: 600px) {
        [data-testid="stSegmentedControl"] button { flex-basis: 100%; width: 100%; }
        h1 { font-size: 1.8rem !important; }
        h3 { font-size: 1.45rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)
def _video_metadata(path: str) -> dict[str, Any]:
    return video_metadata(path)


def _result_file_signature(path: str | Path) -> tuple[str, int, int]:
    candidate = Path(path).resolve(strict=True)
    stat = candidate.stat()
    return str(candidate), int(stat.st_size), int(stat.st_mtime_ns)


@st.cache_data(max_entries=64, show_spinner=False)
def _cached_json_object(path: str, size_bytes: int, mtime_ns: int) -> dict[str, Any]:
    del size_bytes, mtime_ns
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read current analysis metadata: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"Expected a JSON object in {Path(path).name}")
    return value


def _load_result_summary(result_dir: str | Path) -> dict[str, Any]:
    path, size, mtime = _result_file_signature(Path(result_dir) / "video_summary.json")
    return _cached_json_object(path, size, mtime)


@st.cache_resource(max_entries=8, show_spinner="Indexing traceable result records…")
def _cached_dashboard_bundle(
    result_dir: str,
    jsonl_size: int,
    jsonl_mtime_ns: int,
    summary_size: int,
    summary_mtime_ns: int,
    config_size: int,
    config_mtime_ns: int,
) -> dict[str, Any]:
    del jsonl_size, jsonl_mtime_ns, summary_size, summary_mtime_ns
    del config_size, config_mtime_ns
    root = Path(result_dir)
    try:
        store = open_result_store(root / "frame_results.jsonl")
        summary = json.loads((root / "video_summary.json").read_text(encoding="utf-8"))
        config = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ResultStoreError) as exc:
        raise DashboardDataError(f"Could not read current analysis: {exc}") from exc
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise DashboardDataError("Summary and run configuration must be JSON objects")
    if not store.entries:
        raise DashboardDataError("Current analysis has no frame-level pose observations")
    return {
        "dir": str(root),
        "summary": summary,
        "config": config,
        "store": store,
        "records": [entry.lightweight for entry in store.entries],
    }


def _load_dashboard_bundle(result_dir: str | Path) -> dict[str, Any]:
    root = Path(result_dir).resolve(strict=True)
    required = [
        root / "frame_results.jsonl",
        root / "video_summary.json",
        root / "run_config.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise DashboardDataError("Current analysis is incomplete; missing: " + ", ".join(missing))
    signatures = [_result_file_signature(path) for path in required]
    return _cached_dashboard_bundle(
        str(root),
        signatures[0][1],
        signatures[0][2],
        signatures[1][1],
        signatures[1][2],
        signatures[2][1],
        signatures[2][2],
    )


@st.cache_data(max_entries=144, show_spinner=False)
def _cached_review_image(
    video_path: str,
    video_mtime_ns: int,
    source_frame_number: int,
    jsonl_path: str,
    jsonl_size: int,
    jsonl_mtime_ns: int,
    record_offset: int,
    selected_region: str,
    camera_index: int,
    overlay_enabled: bool,
):
    del video_mtime_ns, jsonl_size, jsonl_mtime_ns
    record = (
        open_result_store(jsonl_path).load_record_at_offset(record_offset)
        if overlay_enabled
        else None
    )
    return read_video_frame(
        video_path,
        source_frame_number,
        record,
        selected_region,
        camera_index,
    )


@st.cache_resource(max_entries=24, show_spinner="Loading movement landmarks…")
def _cached_landmark_subset(
    jsonl_path: str,
    jsonl_size: int,
    jsonl_mtime_ns: int,
    person_identifier: str,
    stride: int,
) -> tuple[dict[str, Any], ...]:
    del jsonl_size, jsonl_mtime_ns
    records = open_result_store(jsonl_path).load_landmark_subset(
        stride=stride,
        person_identifier=person_identifier,
    )
    return tuple(records)


@st.cache_resource(max_entries=32, show_spinner="Computing exposure and movement evidence…")
def _cached_professional_analysis(
    jsonl_path: str,
    jsonl_size: int,
    jsonl_mtime_ns: int,
    person_identifier: str,
    first_frame: int,
    last_frame: int,
    minimum_visibility: float,
    maximum_gap_sec: float,
    sustained_min_duration_sec: float,
    repetition_prominence_deg: float,
    repetition_min_interval_sec: float,
    rnle_json: str,
) -> dict[str, Any]:
    del jsonl_size, jsonl_mtime_ns
    store = open_result_store(jsonl_path)
    records = [
        entry.lightweight
        for entry in store.entries
        if entry.person_identifier == person_identifier
        and entry.frame_number is not None
        and first_frame <= entry.frame_number <= last_frame
    ]
    rnle_result = json.loads(rnle_json) if rnle_json else None
    return analyze_professional_records(
        records,
        rnle_result=rnle_result,
        person_identifier=person_identifier,
        minimum_visibility=minimum_visibility,
        maximum_gap_sec=maximum_gap_sec,
        sustained_min_duration_sec=sustained_min_duration_sec,
        repetition_prominence_deg=repetition_prominence_deg,
        repetition_min_interval_sec=repetition_min_interval_sec,
    ).to_dict()


def _trial_videos(task_path: str) -> list[str]:
    task_dir = Path(task_path)
    try:
        manifest = trial_manifest(task_dir)
        if manifest is not None:
            return [str(recorded_path(task_dir, entry)) for entry in manifest]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Cannot open the selected recordings for {task_dir.name}: {exc}")
        return []
    return find_videos(task_path)["raw"]


def _camera_references(
    record: dict[str, Any], summary: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    canonical = record.get("camera_frame_references")
    if isinstance(canonical, list) and canonical:
        references: list[dict[str, Any]] = []
        for item in canonical:
            if not isinstance(item, dict):
                continue
            delta_sec = item.get("sync_delta_from_master_sec")
            references.append(
                {
                    "camera_index": int(item.get("camera_index", len(references))),
                    "role": item.get("camera_role") or item.get("role"),
                    "path": item.get("source_path") or item.get("path"),
                    "source_frame_number": item.get("source_frame_number"),
                    "timestamp_sec": item.get("timestamp_sec"),
                    "timestamp_source": item.get("timestamp_source"),
                    "sync_delta_ms": (
                        float(delta_sec) * 1000
                        if isinstance(delta_sec, (int, float))
                        else item.get("sync_delta_ms")
                    ),
                    "synchronization_status": record.get("synchronization", {}).get(
                        "status", item.get("synchronization_status", "unknown")
                    ),
                }
            )
        if references:
            return references
    camera_roles = config.get("fusion", {}).get("camera_roles", {})
    return build_legacy_camera_refs(
        record,
        summary.get("video_metadata", []),
        camera_roles if isinstance(camera_roles, dict) else {},
    )


def _record_entry(store: JsonlResultStore, person_identifier: str, frame_number: int):
    candidates = [
        entry
        for entry in store.entries
        if entry.person_identifier == person_identifier and entry.frame_number is not None
    ]
    if not candidates:
        raise DashboardDataError("No indexed pose record exists for the selected person")
    return min(candidates, key=lambda entry: abs(int(entry.frame_number) - frame_number))


def _status_text(value: Any) -> str:
    return str(value or "unavailable").replace("_", " ").title()


def _display_cell(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _tri_state(label: str, key: str, help_text: str | None = None) -> bool | None:
    options = {"Unknown": None, "No": False, "Yes": True}
    choice = st.selectbox(label, list(options), key=key, help=help_text)
    return options[choice]


def _categorical(
    label: str, key: str, options: dict[str, Any], help_text: str | None = None
) -> Any:
    choice = st.selectbox(label, list(options), key=key, help=help_text)
    return options[choice]


def _format_count(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _result_event_value(event: Any, field: str = "customdata") -> Any:
    try:
        points = event.selection.points
    except AttributeError:
        points = event.get("selection", {}).get("points", []) if event else []
    return points[0].get(field) if points else None


def _select_record(
    records: list[dict[str, Any]], frame_number: int, person_identifier: str | None
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if int(record.get("frame_number", -1)) == int(frame_number)
        and (person_identifier is None or record.get("person_identifier") == person_identifier)
    ]
    if candidates:
        return candidates[0]
    return min(
        records,
        key=lambda record: abs(int(record.get("frame_number", 0)) - int(frame_number)),
    )


def _summary_value(summary: dict[str, Any], method: str, key: str) -> Any:
    return summary.get(f"{method.lower()}_summary", {}).get(key)


def _job_status_body() -> None:
    job = st.session_state.get("analysis_job")
    if job is None:
        return
    state = job.get_state()
    st.markdown("#### Processing workflow")
    stage_order = [
        ("validating", "1", "Validate", "Inputs and output"),
        ("loading_model", "2", "Initialize", "Video and pose model"),
        ("processing", "3", "Analyze", "Frames and scores"),
        ("finalizing", "4", "Finalize", "Summaries and reports"),
        ("completed", "5", "Complete", "Results available"),
    ]
    current_stage = state.get("stage", "queued")
    current_index = next(
        (index for index, item in enumerate(stage_order) if item[0] == current_stage),
        -1,
    )
    step_columns = st.columns(len(stage_order))
    for index, ((_, number, label, detail), column) in enumerate(
        zip(stage_order, step_columns, strict=True)
    ):
        css_class = ""
        if state["status"] == "error" and index == max(current_index, 0):
            css_class = " error"
        elif index < current_index or state["status"] == "completed":
            css_class = " done"
        elif index == current_index:
            css_class = " active"
        with column:
            st.markdown(
                f'<div class="status-step{css_class}"><b>{number}. {label}</b>'
                f'<div class="small-note">{detail}</div></div>',
                unsafe_allow_html=True,
            )

    total = int(state.get("total_frames") or 0)
    current = int(state.get("progress_frame") or 0)
    percent = min(current / total, 1.0) if total > 0 else 0.0
    st.progress(
        percent,
        text=(
            f"{state.get('message', 'Working…')} · {percent * 100:.1f}%"
            if total > 0
            else state.get("message", "Preparing analysis…")
        ),
    )
    status_columns = st.columns(4)
    status_columns[0].metric(
        "Source progress",
        f"{current:,} / {total:,}" if total else f"{current:,}",
    )
    status_columns[1].metric("Elapsed", f"{state['elapsed_sec']:.1f} s")
    status_columns[2].metric("Throughput", f"{state['fps']:.2f} source frames/s")
    status_columns[3].metric(
        "Estimated remaining",
        f"{state['eta_sec']:.0f} s" if state["eta_sec"] > 0 else "Calculating",
    )
    if state["status"] == "running" and state.get("last_update_age_sec", 0) > 30:
        st.warning(
            "No progress update has arrived for 30 seconds. Model initialization or a "
            "slow frame may still be running; if this persists, inspect the displayed stage."
        )
    if state["status"] == "completed":
        result_key = state.get("output_dir")
        if st.session_state.get("refreshed_result_dir") != result_key:
            st.cache_data.clear()
            st.session_state["refreshed_result_dir"] = result_key
        st.success(
            f"Analysis completed in {state['elapsed_sec']:.1f} seconds. "
            "Results and 3-D Activity are ready."
        )
    elif state["status"] == "error":
        st.error("Analysis failed. The complete error is shown below.")
        st.code(state["error"], language="text")


@st.fragment(run_every=2.0)
def _live_batch_status() -> None:
    batch = st.session_state.get("batch_job")
    if batch is None:
        return
    state = batch.describe()
    finished = state["completed"]
    total = state["total"]
    if state["status"] == "running":
        st.progress(
            finished / total if total else 0.0,
            text=f"Batch running: {finished} of {total} trials finished · "
            f"current: {state['current_label'] or 'starting'}",
        )
        current = state.get("current")
        if current and current.get("total_frames"):
            st.caption(
                f"Current trial: {current['progress_frame']:,} / "
                f"{current['total_frames']:,} frames · {current['message']}"
            )
    elif state["status"] == "completed":
        failed = state["failed"]
        if failed:
            st.warning(f"Batch finished: {total - failed} of {total} trials succeeded.")
        else:
            st.success(f"Batch finished: all {total} trials analyzed.")
        for outcome in state["outcomes"]:
            if not outcome["ok"]:
                st.error(f"{outcome['label']}: {outcome['error']}")
    # run_every already refreshes this fragment; an explicit rerun here raises
    # on the first render because the call is outside a fragment rerun.


@st.fragment(run_every=1.0)
def _live_job_status() -> None:
    _job_status_body()


def _recording_status_body(task_path: str) -> None:
    service = get_recording_service()
    snapshot = service.snapshot()
    if not snapshot["preview_active"]:
        return
    multiple = len(snapshot["cameras"]) > 1
    status_columns = st.columns(5 if multiple else 4)
    status_columns[0].metric("Capture state", "Recording" if snapshot["recording"] else "Preview")
    status_columns[1].metric("Elapsed", f"{snapshot['elapsed_sec']:.1f} s")
    status_columns[2].metric("Synchronized frames" if multiple else "Recorded frames", f"{snapshot['synchronized_frame_sets']:,}")
    if multiple:
        status_columns[3].metric("Maximum sync delta", f"{snapshot['maximum_sync_delta_sec'] * 1000:.1f} ms")
    status_columns[-1].metric("Dropped sets" if multiple else "Dropped frames", snapshot["dropped_frame_sets"])

    controls = st.columns([1, 1, 3])
    if snapshot["recording"]:
        if controls[0].button("Stop recording", type="primary", key="record_stop", width="stretch"):
            try:
                manifest = service.stop_recording()
                st.session_state["last_recording_manifest"] = manifest
                st.cache_data.clear()
                st.rerun(scope="fragment")
            except RecordingError as exc:
                st.error(str(exc))
    elif controls[0].button("Start recording", type="primary", key="record_start", width="stretch"):
        try:
            service.start_recording(task_path)
            st.rerun(scope="fragment")
        except RecordingError as exc:
            st.error(str(exc))
    if controls[1].button("Stop cameras", key="camera_stop", width="stretch"):
        service.stop_preview()
        st.cache_data.clear()
        st.rerun(scope="fragment")
    controls[2].caption(
        "Raw and capture-annotated videos are written together. Audited ergonomic "
        "overlays are produced during post-processing."
    )

    camera_stats = snapshot["cameras"]
    previews = service.latest_previews()
    monitor = None if snapshot["recording"] else service.placement()
    checks = monitor.status() if monitor else {}
    placement = checks.get("cameras", {})
    if checks.get("noClearView"):
        st.info("No sampled view has consistent landmark coverage for "
                + ", ".join(checks["noClearView"]).lower() + ". Adjust one camera and rehearse again.")
    if previews:
        preview_columns = st.columns(len(previews))
        for column, device_id in zip(preview_columns, sorted(previews), strict=True):
            stats = camera_stats.get(device_id, {})
            with column:
                st.image(
                    previews[device_id][:, :, ::-1],
                    caption=(
                        f"{stats.get('label', f'Camera {device_id}')} · "
                        f"{stats.get('actual_fps', 0):.1f} FPS · "
                        f"{stats.get('captured_frames', 0):,} frames"
                    ),
                    width="stretch",
                )
                if stats.get("error"):
                    st.error(stats["error"])
                if monitor and device_id in placement:
                    state = placement[device_id]
                    st.markdown("**Camera placement**")
                    if state["state"] == "warming":
                        st.caption("Rehearse the task while the camera samples the working area.")
                    elif state["state"] == "clear":
                        st.success("Body regions in view")
                    for issue in state["issues"]:
                        st.warning(issue)
                    if state["regions"]:
                        st.caption(" · ".join(f"{r['name']}: {r['status']}" for r in state["regions"]))
                    visual = state["visual"] or {}
                    checking = visual.get("state") == "checking"
                    try:
                        if st.button("Inspect visibility with AI", key=f"placement_ai_{device_id}",
                                     disabled=state["samples"] < 3 or checking):
                            monitor.inspect(device_id, state["id"])
                        if checking:
                            st.caption("Inspecting sampled frames…")
                        if visual.get("detail"):
                            st.info(visual["detail"])
                        for finding in visual.get("findings", []):
                            st.write(f"{finding['label']} · {finding['visibility'].replace('_', ' ')}. "
                                     f"{finding['cue']} {finding['suggestion']}")
                        if visual.get("state") == "ready":
                            with st.expander("Inspected preview images"):
                                for frame in visual["frames"]:
                                    st.image(monitor.evidence(device_id, visual["checkId"], frame), caption=f"Frame {frame}")
                        if state["state"] != "warming":
                            decision = (state["decision"] or {}).get("value")
                            for choice, label in [("ready", "View is ready"), ("adjust", "Needs adjustment"), ("unclear", "Cannot tell")]:
                                if st.button(label, key=f"placement_{device_id}_{choice}",
                                             type="primary" if decision == choice else "secondary", disabled=checking):
                                    monitor.decide(device_id, state["id"], choice)
                    except PlacementError as exc:
                        st.info(str(exc))


@st.fragment(run_every=0.5)
def _live_recording_status(task_path: str) -> None:
    _recording_status_body(task_path)


def _task_fact_inputs(prefix: str, advanced: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Collect only the task conditions the footage cannot supply.

    Every condition that is a threshold on a measured angle -- trunk twist, arm
    abduction, wrist deviation, and the rest -- is now derived from the video by
    core/posture_conditions.py, and the conditions that are properties of the
    sequence -- hold time, repetition rate, seated, leg weight bearing -- by
    temporal/task_dynamics.py. Asking for them again would be asking the
    operator to re-enter a measurement.

    What is left is what no camera can see: how heavy the load is, how well it
    can be gripped, whether force arrives as a shock, and whether an arm is
    resting on something. Those five are the default form.

    Advanced mode adds an override for the derived conditions. An entered value
    always wins, and any disagreement with the measurement is recorded in the
    result rather than quietly resolved.
    """

    values: dict[str, Any] = {}

    st.markdown("**Confirm what the video cannot measure**")
    st.caption(
        "Everything else -- trunk, neck, arm, wrist, legs, repetition and hold time -- "
        "is measured from the footage. Leave a field Unknown and a documented "
        "assumption fills it, recorded separately from the measurements."
    )
    first, second, third = st.columns(3)
    with first:
        values["load_kg"] = st.number_input(
            "Handled load (kg)",
            min_value=0.0,
            value=None,
            step=0.1,
            key=f"{prefix}_load",
            help="The one input that moves the score most. Not recoverable from geometry.",
            placeholder="e.g. 8.0",
        )
    with second:
        values["coupling"] = _categorical(
            "Hand coupling",
            f"{prefix}_coupling",
            {"Unknown": None, "Good": "good", "Fair": "fair", "Poor": "poor"},
        )
    with third:
        values["shock_or_rapid_force"] = _tri_state(
            "Shock or rapid force buildup", f"{prefix}_shock"
        )

    left, right = st.columns(2)
    for side, column in (("left", left), ("right", right)):
        with column:
            supported = _tri_state(
                f"{side.title()} arm rests on a surface", f"{prefix}_{side}_supported"
            )
            # RULA and REBA word the same fact differently; one answer serves both.
            values[f"{side}_arm_supported_or_leaning"] = supported
            values[f"{side}_arm_supported_or_gravity_assisted"] = supported

    if advanced:
        values.update(_derived_condition_overrides(prefix))

    missing = sorted(name for name, value in values.items() if value is None)
    return values, missing


#: Conditions the pipeline derives, offered as overrides in Advanced mode only.
#: Grouped the way an ergonomist reads a body map rather than alphabetically.
_OVERRIDE_GROUPS: dict[str, dict[str, str]] = {
    "Trunk and neck": {
        "trunk_twisted": "Trunk twisted",
        "trunk_side_bent": "Trunk side-bent",
        "trunk_twisted_or_side_flexed": "Trunk twisted or side-flexed (REBA)",
        "neck_twisted": "Neck twisted",
        "neck_side_bent": "Neck side-bent",
        "neck_twisted_or_side_flexed": "Neck twisted or side-flexed (REBA)",
    },
    "Legs and activity": {
        "seated": "Seated",
        "legs_supported_and_balanced": "Legs supported and balanced",
        "static_posture_over_one_minute": "Static posture over one minute",
        "repeated_more_than_four_per_minute": "Repeated more than 4/min",
        "rapid_large_posture_change_or_unstable_base": "Rapid change or unstable base",
    },
}

_SIDE_OVERRIDES: dict[str, str] = {
    "upper_arm_abducted": "Upper arm abducted (RULA)",
    "upper_arm_abducted_or_rotated": "Upper arm abducted or rotated (REBA)",
    "shoulder_raised": "Shoulder raised",
    "lower_arm_across_midline_or_out_to_side": "Lower arm across midline/out to side",
    "wrist_deviated": "Wrist deviated (RULA)",
    "wrist_twist_at_end_range": "Wrist twist at end range (RULA)",
    "wrist_deviated_or_twisted": "Wrist deviated or twisted (REBA)",
}


def _derived_condition_overrides(prefix: str) -> dict[str, Any]:
    """Manual overrides for conditions the pipeline measures itself."""

    values: dict[str, Any] = {}
    with st.expander("Override a measured condition", expanded=False):
        st.caption(
            "These are measured from the footage. Set one only when you saw something "
            "the camera could not. Your value is used and the disagreement is recorded."
        )
        columns = st.columns(len(_OVERRIDE_GROUPS))
        for column, (group, labels) in zip(columns, _OVERRIDE_GROUPS.items(), strict=True):
            with column:
                st.markdown(f"**{group}**")
                for name, label in labels.items():
                    values[name] = _tri_state(label, f"{prefix}_{name}")

        values["legs_weight_bearing"] = _categorical(
            "Leg weight bearing",
            f"{prefix}_bearing",
            {
                "Unknown": None,
                "Bilateral": "bilateral",
                "Walking": "walking",
                "Unilateral": "unilateral",
                "Unstable": "unstable",
            },
        )
        values["rula_load_type"] = _categorical(
            "RULA load exposure",
            f"{prefix}_load_type",
            {
                "Unknown": None,
                "Intermittent": "intermittent",
                "Static or repeated": "static_or_repeated",
            },
        )

        left, right = st.columns(2)
        for side, column in (("left", left), ("right", right)):
            with column:
                st.markdown(f"**{side.title()} arm and wrist**")
                for name, label in _SIDE_OVERRIDES.items():
                    values[f"{side}_{name}"] = _tri_state(label, f"{prefix}_{side}_{name}")
    return values


#: Plain names for the conditions the video cannot supply. RULA and REBA word
#: arm support differently, so both fields collapse to the one question asked.
UNCONFIRMED_LABELS = {
    "load_kg": "handled load",
    "coupling": "hand coupling",
    "shock_or_rapid_force": "shock or rapid force",
    "left_arm_supported_or_leaning": "left arm support",
    "left_arm_supported_or_gravity_assisted": "left arm support",
    "right_arm_supported_or_leaning": "right arm support",
    "right_arm_supported_or_gravity_assisted": "right arm support",
    "left_wrist_twist_at_end_range": "left wrist twist",
    "right_wrist_twist_at_end_range": "right wrist twist",
    "left_wrist_deviated_or_twisted": "left wrist deviation or twist",
    "right_wrist_deviated_or_twisted": "right wrist deviation or twist",
}

RNLE_APPLICABILITY_LABELS = {
    "one_handed": "One-handed lifting",
    "lifting_over_eight_hours": "Lifting over 8 hours",
    "seated_or_kneeling": "Seated or kneeling lifting",
    "restricted_workspace": "Restricted workspace",
    "unstable_load": "Unstable load",
    "carrying_pushing_or_pulling": "Carrying, pushing, or pulling",
    "wheelbarrow_or_shoveling": "Wheelbarrow or shoveling",
    "high_speed_motion": "High-speed motion",
    "insufficient_foot_floor_coupling": "Insufficient foot-floor coupling",
    "unfavorable_temperature_or_humidity": "Unfavorable temperature or humidity",
    "non_lifting_work_exceeds_ten_percent": "Non-lifting work exceeds 10%",
    "multiple_task_job": "Multiple-task job",
}


def _rnle_inputs(
    prefix: str, advanced: bool = False
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Collect the RNLE inputs the footage cannot supply.

    The lifting geometry -- horizontal distance, hand height, vertical travel,
    asymmetry -- and the lift frequency are estimated from the sequence by
    ergonomics/rnle_estimation.py, scaled to real centimetres using the
    participant's recorded stature. Asking for them here would be asking the
    operator to re-measure what the video already shows.

    Two inputs stay. Load weight is not recoverable from geometry, and without
    it the lifting index has no numerator, so the recommended weight limit is
    reported alone. Work duration describes the shift, not the recording.

    Advanced mode restores the geometry fields as overrides, for a task measured
    on site or one where the estimate was reported out of range.
    """

    st.markdown("**Lifting task**")
    st.caption(
        "Distances and lift frequency are measured from the footage and the participant's "
        "recorded height. The handled load entered above is reused here; without it the "
        "recommended weight limit is still reported, but the lifting index is not."
    )
    horizontal = vertical = distance = asymmetry = frequency = None
    load = None
    first, second = st.columns(2)
    with first:
        duration = _categorical(
            "Work duration",
            f"{prefix}_rnle_duration",
            {
                "Unknown": None,
                "Short (<=1 h)": "short",
                "Moderate (1-2 h)": "moderate",
                "Long (2-8 h)": "long",
            },
        )
    with second:
        coupling = _categorical(
            "Object coupling",
            f"{prefix}_rnle_coupling",
            {"Unknown": None, "Good": "good", "Fair": "fair", "Poor": "poor"},
        )

    applicability: dict[str, bool | None] = dict.fromkeys(RNLE_APPLICABILITY_LABELS)
    if advanced:
        with st.expander("Override the measured lifting geometry", expanded=False):
            st.caption(
                "Estimated from the video unless entered here. Enter a measured value when "
                "the estimate was reported out of range, or the task was measured on site."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                horizontal = st.number_input(
                    "Horizontal distance H (cm)",
                    min_value=0.0,
                    value=None,
                    key=f"{prefix}_rnle_h",
                )
                vertical = st.number_input(
                    "Hand height V (cm)", min_value=0.0, value=None, key=f"{prefix}_rnle_v"
                )
            with c2:
                distance = st.number_input(
                    "Vertical travel D (cm)", min_value=0.0, value=None, key=f"{prefix}_rnle_d"
                )
                asymmetry = st.number_input(
                    "Asymmetry A (degrees)", min_value=0.0, value=None, key=f"{prefix}_rnle_a"
                )
            with c3:
                frequency = st.number_input(
                    "Frequency (lifts/min)", min_value=0.0, value=None, key=f"{prefix}_rnle_f"
                )

        with st.expander("Applicability exclusions", expanded=False):
            st.caption(
                "RNLE does not cover these situations. Unknown remains unreviewed and the "
                "task is reported as such."
            )
            columns = st.columns(3)
            for index, (name, label) in enumerate(RNLE_APPLICABILITY_LABELS.items()):
                with columns[index % 3]:
                    applicability[name] = _tri_state(label, f"{prefix}_rnle_app_{name}")
    else:
        # Default path: the ordinary two-handed lift the study protocol
        # describes, recorded as reviewed-applicable. Advanced mode is where an
        # unusual task gets its exclusions marked.
        applicability = dict.fromkeys(RNLE_APPLICABILITY_LABELS, False)

    values = {
        "H": horizontal,
        "V": vertical,
        "D": distance,
        "A": asymmetry,
        "F": frequency,
        "duration": duration,
        "coupling": coupling,
        "load_weight_kg": load,
        "applicability": applicability,
    }
    # Only what the video cannot supply counts as missing here; the geometry is
    # filled by the estimator once the frames have been read.
    missing = [label for label, value in {"load weight": load}.items() if value is None]
    unreviewed = [
        RNLE_APPLICABILITY_LABELS[name] for name, value in applicability.items() if value is None
    ]
    return values, missing, unreviewed


# Participant-first global selection
recording_service = get_recording_service()
recording_snapshot = recording_service.snapshot()
navigation_locked = bool(recording_snapshot["preview_active"])

st.sidebar.markdown("## Study analysis")
with st.sidebar.expander("Data source settings", expanded=False):
    studies_root = st.text_input(
        "Data folder",
        value=st.session_state.get("studies_root", STUDIES_DIR_STR),
        help="Folder containing captured studies",
    )
st.session_state["studies_root"] = studies_root

if not os.path.isdir(studies_root):
    st.error("The selected data folder does not exist.")
    st.code(studies_root)
    st.stop()

catalog = participant_catalog(studies_root)
search = st.sidebar.text_input(
    "Find a participant",
    placeholder="ID, date, trial, sex, or dominant hand",
    disabled=navigation_locked,
)
filtered_catalog = [item for item in catalog if not search or search.lower() in item["search_text"]]

if not filtered_catalog:
    st.warning("No participants match the current search.")
    st.stop()

selected_key = st.sidebar.selectbox(
    "Participant",
    [item["key"] for item in filtered_catalog],
    format_func=lambda key: next(item["label"] for item in filtered_catalog if item["key"] == key),
    disabled=navigation_locked,
)
context = next(item for item in filtered_catalog if item["key"] == selected_key)
tasks = context["tasks"]
if not tasks:
    st.warning("The selected participant has no available trials.")
    st.stop()

task_names = [task["name"] for task in tasks]
selected_task_name = st.sidebar.selectbox(
    "Trial",
    task_names,
    key="selected_trial",
    disabled=navigation_locked,
)
task = next(task for task in tasks if task["name"] == selected_task_name)

if navigation_locked:
    st.sidebar.info(
        "Participant and trial selection are locked while cameras are active. "
        "Stop the cameras from Record Activity before changing context."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Workflow\n\n1. Select a participant and trial\n2. Confirm the load\n"
    "3. Run analysis\n4. Review the findings"
)

st.markdown(
    '<div class="brand">'
    f'<img src="{brand_logo_uri(dark_surface=True)}" alt="ErgoPartner" />'
    '<div class="brand-tag">RULA · REBA · NIOSH from video</div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Ergonomic study analysis</div>', unsafe_allow_html=True)
st.title(f"Participant {context['participant_id']}")
st.markdown(
    f'<p class="page-intro">Study date {context["date_label"]} · {selected_task_name} · '
    "Scores are measured from the footage. Only the few conditions a camera cannot "
    "see are left to confirm.</p>",
    unsafe_allow_html=True,
)

# Advanced mode gates everything that is a choice rather than a result: live
# capture, running one camera or one method instead of all of them, overriding a
# measured condition, and restricting the frame range. The default path is
# select a trial, run, read the findings.
ADVANCED_MODE = st.sidebar.toggle(
    "Advanced mode",
    value=False,
    key="advanced_mode",
    help=(
        "Show live capture, per-camera and per-method selection, manual condition "
        "overrides, and frame-range limits. Off, every camera and every method runs."
    ),
)

# Frame Review is a core section. It is the payoff of every "open this evidence"
# button on the Results page, and while it sat behind the Advanced toggle those
# buttons led nowhere the default workflow could follow.
CORE_SECTIONS = ["Study Overview", "Run Analysis", "Results", FRAME_REVIEW_SECTION, "3-D Activity"]
ADVANCED_SECTIONS = ["Record Activity"]
DASHBOARD_SECTIONS = CORE_SECTIONS + (ADVANCED_SECTIONS if ADVANCED_MODE else [])

# A section requested from a Results button is parked under its own key and
# applied here, before the navigation widget exists. Streamlit refuses to let a
# widget's own session-state key be written once that widget has been created in
# the same run, which is the same reason the frame slider uses a pending key.
_requested_section = st.session_state.pop("pending_dashboard_section", None)
if _requested_section in DASHBOARD_SECTIONS:
    st.session_state["dashboard_section"] = _requested_section

active_view = st.segmented_control(
    "Dashboard section",
    DASHBOARD_SECTIONS,
    default="Study Overview",
    selection_mode="single",
    key="dashboard_section",
    label_visibility="collapsed",
    width="stretch",
)
active_view = active_view or "Study Overview"
# A section can vanish when Advanced mode is switched off while it is open.
if active_view not in DASHBOARD_SECTIONS:
    active_view = "Study Overview"


if active_view == "Study Overview":
    st.subheader("Participant and study overview")
    info = context["participant"]["info"]
    selected_status = "Ready for analysis"
    selected_options = get_processed_fps(
        context["study"]["folder_name"],
        context["participant_id"],
        task["folder_name"],
    )
    if selected_options:
        selected_result = detect_current_results(
            context["study"]["folder_name"],
            context["participant_id"],
            task["folder_name"],
            selected_options[0],
        )
        if selected_result:
            try:
                selected_summary = _load_result_summary(selected_result["dir"])
                formal_count = int(selected_summary.get("valid_rula_observations", 0)) + int(
                    selected_summary.get("valid_reba_observations", 0)
                )
                rnle_status = selected_summary.get("rnle", {}).get("validity_status")
                selected_status = (
                    "Ready for review"
                    if formal_count > 0 or rnle_status in {"valid", "valid_with_limitations"}
                    else "Formal assessment incomplete"
                )
            except DashboardDataError:
                selected_status = "Result incomplete"
    status_col, age_col, sex_col, hand_col, task_col = st.columns(5)
    status_col.metric("Study status", selected_status)
    age_col.metric("Age", info.get("age", "—"))
    sex_col.metric("Sex", info.get("sex", "—"))
    hand_col.metric("Dominant hand", str(info.get("dominant_hand", "—")).title())
    task_col.metric("Available trials", len(tasks))

    raw_recordings = []
    analyzed_trials = 0
    processed_frames = 0
    for task_item in tasks:
        task_videos = _trial_videos(task_item["path"])
        raw_recordings.extend(task_videos)
        options = get_processed_fps(
            context["study"]["folder_name"],
            context["participant_id"],
            task_item["folder_name"],
        )
        if options:
            analyzed_trials += 1
            for sampling in options:
                result = detect_current_results(
                    context["study"]["folder_name"],
                    context["participant_id"],
                    task_item["folder_name"],
                    sampling,
                )
                if result:
                    try:
                        result_summary = _load_result_summary(result["dir"])
                        processed_frames += int(result_summary.get("pose_observations", 0))
                    except DashboardDataError:
                        pass
                    break
    available_frames = sum(_video_metadata(path).get("frame_count") or 0 for path in raw_recordings)

    st.markdown("#### Available evidence")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw recordings", len(raw_recordings))
    c2.metric("Recorded frames", f"{available_frames:,}")
    c3.metric("Trials analyzed", f"{analyzed_trials} / {len(tasks)}")
    c4.metric("Pose observations", f"{processed_frames:,}")

    rows = []
    for task_item in tasks:
        videos = _trial_videos(task_item["path"])
        frames = sum(_video_metadata(path).get("frame_count") or 0 for path in videos)
        samplings = get_processed_fps(
            context["study"]["folder_name"],
            context["participant_id"],
            task_item["folder_name"],
        )
        rows.append(
            {
                "Trial": task_item["name"],
                "Description": task_item.get("description") or "—",
                "Recordings": len(videos),
                "Cameras": task_item["cameras"],
                "Recorded frames": frames,
                "Analysis status": "Current results available" if samplings else "Not run",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Archived pre-audit result files are not treated as current analysis and are not "
        "shown here."
    )

    st.markdown("#### Batch analysis")
    st.caption(
        "Runs every trial with the standard configuration, including the configured "
        "assumption profile for unobserved task conditions. Each trial uses the same "
        "validated staging-and-promote transaction as a single run; existing results "
        "are archived, not overwritten in place."
    )

    def _batch_trials(participants: list[dict]) -> list[dict]:
        trials: list[dict] = []
        for item in participants:
            for task_item in item["tasks"]:
                videos = _trial_videos(task_item["path"])
                if not videos:
                    continue
                trials.append(
                    {
                        "label": f"{item['participant_id']} · {task_item['name']}",
                        "videos": videos,
                        "output_dir": get_task_results_path(
                            item["study"]["folder_name"],
                            item["participant_id"],
                            task_item["folder_name"],
                            None,
                        ),
                        "config": build_processing_config(str(CONFIG_PATH)),
                    }
                )
        return trials

    existing_batch = st.session_state.get("batch_job")
    batch_state = existing_batch.describe() if existing_batch is not None else None
    batch_active = bool(batch_state and batch_state["status"] in {"pending", "running"})
    single_active = bool(
        st.session_state.get("analysis_job")
        and st.session_state["analysis_job"].get_state()["status"] in {"pending", "running"}
    )
    study_participants = [
        item for item in catalog if item["study"]["folder_name"] == context["study"]["folder_name"]
    ]
    participant_trials = _batch_trials([context])
    study_trials = _batch_trials(study_participants)

    batch_col1, batch_col2 = st.columns(2)
    if batch_col1.button(
        f"Analyze all trials for participant {context['participant_id']} "
        f"({len(participant_trials)})",
        disabled=batch_active or single_active or not participant_trials,
        width="stretch",
    ):
        job, thread = launch_batch_thread(participant_trials)
        st.session_state["batch_job"] = job
        st.session_state["batch_thread"] = thread
        st.rerun()
    if batch_col2.button(
        f"Analyze every participant in this study ({len(study_trials)} trials)",
        disabled=batch_active or single_active or not study_trials,
        width="stretch",
    ):
        job, thread = launch_batch_thread(study_trials)
        st.session_state["batch_job"] = job
        st.session_state["batch_thread"] = thread
        st.rerun()
    if single_active:
        st.info("A single-trial analysis is running; batch runs are disabled until it finishes.")

    if existing_batch is not None:
        _live_batch_status()


if active_view == "Record Activity":
    st.subheader("Record ergonomic activity")
    st.markdown(
        "Capture one to three local camera views for the selected participant and trial. "
        "The recorder saves footage and timestamps for analysis after capture."
    )
    current_recording = recording_service.snapshot()
    if not current_recording["preview_active"]:
        setup_left, setup_right = st.columns([1.2, 1])
        with setup_left:
            detect_col, detected_col = st.columns([1, 2])
            if detect_col.button(
                "Detect cameras",
                help="Checks local camera device IDs 0 through 7.",
                width="stretch",
            ):
                with st.spinner("Checking local cameras…"):
                    detected = recording_service.detect_cameras(8)
                st.session_state["detected_camera_ids"] = detected
                if detected:
                    st.success(f"Detected {len(detected)} available camera(s).")
                else:
                    st.warning("No camera returned a usable frame.")
            detected_ids = st.session_state.get("detected_camera_ids", [])
            detected_col.caption(
                "Detected: "
                + (
                    ", ".join(f"Camera {device_id}" for device_id in detected_ids)
                    if detected_ids
                    else "not checked"
                )
            )
            multi_record = st.radio("Camera setup", ["Single camera", "Multiple cameras"],
                                    horizontal=True, key="record_camera_mode") == "Multiple cameras"
            if multi_record:
                selected_devices = st.multiselect(
                    "Camera devices", list(range(8)), default=detected_ids[:3], max_selections=3,
                    format_func=lambda device_id: f"Camera {device_id}",
                    help="Choose two or three views of the same activity.",
                )
            else:
                selected_devices = [st.selectbox(
                    "Camera device", list(range(8)), index=detected_ids[0] if detected_ids else 0,
                    format_func=lambda device_id: f"Camera {device_id}",
                )]
        with setup_right:
            resolution = st.selectbox(
                "Capture resolution",
                [(1280, 720), (1920, 1080), (640, 480)],
                format_func=lambda size: f"{size[0]} × {size[1]}",
            )
            capture_fps = st.selectbox("Requested capture rate", [30.0, 60.0, 24.0])
            st.caption(
                ("Use front and side views where possible. " if multi_record else "")
                + "Avoid mirrored camera software, "
                "strong camera tilt, and subjects leaving the frame."
            )

        initialize = st.button(
            "Initialize camera preview",
            type="primary",
            disabled=len(selected_devices) < (2 if multi_record else 1),
        )
        if initialize:
            labels = ["Primary view", "Secondary view", "Tertiary view"]
            specs = [
                CameraSpec(
                    device_id=device_id,
                    label=labels[index] if multi_record else "Camera",
                    width=int(resolution[0]),
                    height=int(resolution[1]),
                    fps=float(capture_fps),
                )
                for index, device_id in enumerate(selected_devices)
            ]
            try:
                recording_service.start_preview(specs)
                st.rerun()
            except (RecordingError, ValueError) as exc:
                st.error(f"Camera initialization failed: {exc}")
    else:
        st.info(
            "Camera capture is active for this participant and trial. Stop the cameras "
            "before changing the participant context."
        )
        _live_recording_status(task["path"])

    completed_manifest = st.session_state.get("last_recording_manifest")
    if completed_manifest:
        st.markdown("#### Last completed recording")
        recorded_views = len(completed_manifest.get("raw_videos", {}))
        manifest_columns = st.columns(4 if recorded_views > 1 else 2)
        manifest_columns[0].metric(
            "Synchronized frame sets" if recorded_views > 1 else "Recorded frames",
            f"{completed_manifest.get('synchronized_frame_sets', 0):,}",
        )
        if recorded_views > 1:
            manifest_columns[1].metric("Recorded views", recorded_views)
            manifest_columns[2].metric(
                "Maximum sync delta", f"{completed_manifest.get('maximum_sync_delta_sec', 0) * 1000:.1f} ms"
            )
        manifest_columns[-1].metric(
            "Analysis status",
            str(completed_manifest.get("analysis_status", "not_run")).replace("_", " ").title(),
        )
        st.success("Capture completed. The raw recordings are now available in Run Analysis.")


if active_view == "Run Analysis":
    st.subheader("Configure and run analysis")
    videos = _trial_videos(task["path"])
    if not videos:
        st.error("No raw recording is available for this trial.")
    else:
        multi_analysis = st.radio(
            "Camera setup", ["Single camera", "Multiple cameras"], horizontal=True, key="run_camera_mode",
        ) == "Multiple cameras"
        if multi_analysis:
            selected_videos = st.multiselect(
                "Recordings from the same take", videos, default=videos[:3], max_selections=3,
                format_func=lambda path: Path(path).name,
            )
            if len(selected_videos) < 2:
                st.info("Select two or three synchronized recordings for a multi-camera assessment.")
        else:
            selected_videos = [st.selectbox("Recording", videos, format_func=lambda path: Path(path).name)]
        if ADVANCED_MODE:
            left, right = st.columns([1.3, 1])
            with left:
                method_focus = st.multiselect(
                    "Analysis methods",
                    ["RULA", "REBA", "RNLE"],
                    default=["RULA", "REBA", "RNLE"],
                    help=(
                        "RULA and REBA share one pose pass and are both calculated "
                        "independently. This selection controls the report focus."
                    ),
                )
            with right:
                sampling_label = st.selectbox(
                    "Frame sampling",
                    [
                        "10 frames/second",
                        "5 frames/second",
                        "15 frames/second",
                        "30 frames/second",
                        "Native (all decoded frames)",
                    ],
                )
                sampling_map = {
                    "Native (all decoded frames)": None,
                    "5 frames/second": 5,
                    "10 frames/second": 10,
                    "15 frames/second": 15,
                    "30 frames/second": 30,
                }
                target_fps = sampling_map[sampling_label]
                limit_range = st.checkbox("Analyze a frame range", value=False)
        else:
            method_focus = ["RULA", "REBA", "RNLE"]
            target_fps = 10
            limit_range = False
            st.caption(
                f"Analysing {len(selected_videos)} camera(s) at 10 frames per second, "
                "whole recording, with RULA, REBA and RNLE. Switch on Advanced mode in the "
                "sidebar to change any of that."
            )

        selected_metadata = [_video_metadata(path) for path in selected_videos]
        frame_counts = [
            int(item["frame_count"])
            for item in selected_metadata
            if item.get("frame_count") is not None
        ]
        maximum_frame = max(0, min(frame_counts) - 1) if frame_counts else 0
        start_frame, end_frame = 0, maximum_frame
        if limit_range:
            range_cols = st.columns(2)
            start_frame = range_cols[0].number_input("Start frame", 0, maximum_frame, 0)
            end_frame = range_cols[1].number_input("End frame", 0, maximum_frame, maximum_frame)

        prefix = f"run_{context['key']}_{task['task_id']}"
        task_inputs, missing_task_facts = _task_fact_inputs(prefix, advanced=ADVANCED_MODE)
        camera_roles: dict[str, int] = {} if multi_analysis else {"camera": 0}
        camera_roles_valid = len(selected_videos) >= (2 if multi_analysis else 1)

        # Roles default to front/side/iso in file order, which is how the study
        # protocol records them. Opening this by default made a correct default
        # look like an unanswered question.
        with st.expander(
            "Camera roles, synchronization, calibration, and model" if multi_analysis else "Camera calibration and model", expanded=ADVANCED_MODE
        ):
            if multi_analysis:
                st.markdown("**Camera role and timing preflight**")
            role_options = ["front", "side", "iso"]
            role_labels = {
                "front": "Front / coronal",
                "side": "Side / sagittal",
                "iso": "Oblique / ISO review",
            }
            selected_roles: list[str] = []
            if multi_analysis and selected_videos:
                role_columns = st.columns(len(selected_videos))
                for camera_index, (_video_path, column) in enumerate(
                    zip(selected_videos, role_columns, strict=True)
                ):
                    with column:
                        selected_roles.append(
                            st.selectbox(
                                f"Camera {camera_index + 1} role",
                                role_options,
                                index=min(camera_index, len(role_options) - 1),
                                format_func=lambda role: role_labels[role],
                                key=f"{prefix}_role_{camera_index}",
                            )
                        )
                camera_roles_valid = len(set(selected_roles)) == len(selected_roles) and len(selected_roles) >= 2
                camera_roles = {role: index for index, role in enumerate(selected_roles)}
                preflight_rows = []
                for camera_index, (video_path, metadata) in enumerate(
                    zip(selected_videos, selected_metadata, strict=True)
                ):
                    preflight_rows.append(
                        {
                            "View": f"Camera {camera_index + 1}",
                            "Role": role_labels[selected_roles[camera_index]],
                            "File": Path(video_path).name,
                            "Frames": metadata.get("frame_count"),
                            "FPS": metadata.get("fps"),
                            "Duration (s)": metadata.get("duration_sec"),
                        }
                    )
                st.dataframe(pd.DataFrame(preflight_rows), hide_index=True, width="stretch")
                if len(set(selected_roles)) != len(selected_roles):
                    st.error("Each selected recording must have a unique camera role.")
                durations = [
                    float(item["duration_sec"])
                    for item in selected_metadata
                    if item.get("duration_sec") is not None
                ]
                if len(durations) > 1 and max(durations) - min(durations) > 0.05:
                    st.warning(
                        "Recording durations differ by more than 50 ms. Processing will retain "
                        "only timestamp sets inside the configured synchronization tolerance."
                    )
                st.caption(
                    "Metadata checks do not prove hardware synchronization. New results record "
                    "each view's decoded timestamp and synchronization spread; older results are "
                    "labelled index-synchronized with exact offsets unknown."
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                camera_view_status = _categorical(
                    "Camera view suitability",
                    f"{prefix}_view",
                    {
                        "Unknown": "unknown",
                        "Suitable after review": "suitable",
                        "Unsupported": "unsupported",
                    },
                )
            with c2:
                mirrored = _categorical(
                    "Mirrored footage",
                    f"{prefix}_mirror",
                    {"Unknown": None, "No": False, "Yes": True},
                )
            with c3:
                calibration_status = _categorical(
                    "Physical calibration",
                    f"{prefix}_calibration",
                    {"Not calibrated": "not_calibrated", "Calibrated": "calibrated"},
                )
            calibration_method = st.text_input(
                "Calibration method",
                key=f"{prefix}_cal_method",
                disabled=calibration_status != "calibrated",
                placeholder="Describe scale and camera calibration",
            )
            allow_pose_only_wrist = st.checkbox(
                "Use pose-only wrist angles for formal scores",
                value=True,
                key=f"{prefix}_pose_wrist",
                help=(
                    "Wrist angles estimated from pose landmarks alone, without the "
                    "21-point hand mesh. This is the least reliable measurement the pose "
                    "model produces, but refusing it blocks both RULA and REBA on every "
                    "frame. Results record wrist_data_source: pose_only."
                ),
            )
            if not allow_pose_only_wrist:
                st.caption(
                    "With pose-only wrist angles disabled, RULA and REBA will withhold "
                    "numeric scores unless validated hand landmarks are supplied."
                )

        rnle_params = None
        rnle_missing: list[str] = []
        rnle_unreviewed: list[str] = []
        if "RNLE" in method_focus:
            rnle_params, rnle_missing, rnle_unreviewed = _rnle_inputs(
                prefix, advanced=ADVANCED_MODE
            )
            # RULA/REBA and RNLE both need the handled load. It is one fact
            # about one task, so asking for it twice is asking the same question
            # twice; whichever field was filled supplies both.
            if rnle_params.get("load_weight_kg") is None and task_inputs.get("load_kg") is not None:
                rnle_params["load_weight_kg"] = task_inputs["load_kg"]
                rnle_missing = [item for item in rnle_missing if item != "load weight"]
            elif (
                task_inputs.get("load_kg") is None and rnle_params.get("load_weight_kg") is not None
            ):
                task_inputs["load_kg"] = rnle_params["load_weight_kg"]
                missing_task_facts = [item for item in missing_task_facts if item != "load_kg"]

        # Only the conditions no camera can see are worth mentioning here. The
        # rest are measured during the run, and listing them as "not entered"
        # was the message that made this screen look like a data-entry form.
        unconfirmed = sorted(set(missing_task_facts) & NOT_OBSERVABLE_FROM_VIDEO)
        if unconfirmed:
            profile_name = (
                yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
                .get("task_inputs", {})
                .get("assumption_profile")
            )
            readable = ", ".join(
                sorted(
                    {UNCONFIRMED_LABELS.get(name, name.replace("_", " ")) for name in unconfirmed}
                )
            )
            if profile_name:
                st.info(
                    f"Not confirmed: {readable}. Everything else is measured from the "
                    f"footage. The '{profile_name.replace('_', ' ')}' profile supplies these "
                    "so scoring completes, and each assumed value is recorded separately "
                    "from the measured ones. Handled load is the one that moves the score "
                    "most."
                )
            else:
                st.warning(
                    f"Not confirmed and no assumption profile is set: {readable}. "
                    "RULA or REBA will withhold any score that depends on them."
                )
        # The geometry is estimated during the run, so a missing measurement is
        # no longer a reason to warn. Only load weight and an unreviewed
        # exclusion actually hold the result back.
        if rnle_missing:
            st.info(
                "Load weight was not entered. The recommended weight limit will still be "
                "reported from the measured lifting geometry, but the lifting index needs "
                "the load."
            )
        if rnle_unreviewed:
            st.warning(
                "RNLE applicability was not fully reviewed, so the task will be reported as "
                "unreviewed rather than scored: " + ", ".join(rnle_unreviewed)
            )
        if mirrored is True:
            st.error(
                "Mirrored footage is unsupported until anatomical left/right remapping is "
                "validated. Pose-derived scores will be withheld."
            )
        if limit_range and end_frame < start_frame:
            st.error("End frame must be greater than or equal to start frame.")

        existing_job = st.session_state.get("analysis_job")
        existing_state = (
            reconcile_processing_thread(
                existing_job,
                st.session_state.get("analysis_thread"),
            )
            if existing_job is not None
            else None
        )
        analysis_active = bool(
            existing_state and existing_state["status"] in {"pending", "running"}
        )
        run_disabled = (
            not selected_videos
            or not method_focus
            or not camera_roles_valid
            or (limit_range and end_frame < start_frame)
            or analysis_active
        )
        if analysis_active:
            st.info(
                "An analysis is already running. Its live status is shown below; duplicate "
                "processing jobs are disabled to protect result integrity."
            )
        run_clicked = st.button(
            "Run analysis",
            type="primary",
            disabled=run_disabled,
            width="content",
        )
        if run_clicked:
            config = build_processing_config(
                str(CONFIG_PATH),
                niosh_params=rnle_params,
                use_gpu=False,
                target_fps=target_fps,
                start_frame=int(start_frame) if limit_range else None,
                end_frame=int(end_frame) if limit_range else None,
                task_inputs=task_inputs,
                camera_view_status=camera_view_status,
                calibration_status=calibration_status,
                mirrored=mirrored,
                camera_roles=camera_roles,
            )
            roles_by_index = {index: role for role, index in camera_roles.items()}
            config["assessment_workflow"] = {
                "camera_mode": "multi" if multi_analysis else "single",
                "recordings": [recording_entry(Path(video), roles_by_index[i])
                               for i, video in enumerate(selected_videos)],
            }
            config["quality"]["allow_pose_only_wrist"] = allow_pose_only_wrist
            config["calibration"]["method"] = calibration_method or None
            config.setdefault("reporting", {})["selected_methods"] = method_focus
            # The recorded stature is what converts model units into real
            # centimetres for the RNLE geometry. Without it the estimator falls
            # back to the pose model's generic scale and says so.
            participant_height = (context["participant"]["info"] or {}).get("height_cm")
            if participant_height:
                config.setdefault("rnle", {})["participant_height_cm"] = float(participant_height)
            output_dir = get_task_results_path(
                context["study"]["folder_name"],
                context["participant_id"],
                task["folder_name"],
                target_fps,
            )
            job, thread = launch_processing_thread(selected_videos, output_dir, config)
            st.session_state["analysis_job"] = job
            st.session_state["analysis_thread"] = thread
            st.session_state["analysis_job_context"] = (context["key"], task["task_id"], target_fps)
            st.rerun()

        job = st.session_state.get("analysis_job")
        if job is not None:
            state = job.get_state()
            if state["status"] in {"pending", "running"}:
                _live_job_status()
            else:
                _job_status_body()


if active_view == "Results":
    render_professional_results(
        context=context,
        task=task,
        catalog=catalog,
        get_processed_fps=get_processed_fps,
        detect_current_results=detect_current_results,
        load_bundle=_load_dashboard_bundle,
        load_summary=_load_result_summary,
        analyze_cached=_cached_professional_analysis,
        session=st.session_state,
    )

if active_view == "3-D Activity":
    st.subheader("3-D model-estimated activity review")
    st.markdown(
        "Animate a bounded time window of stored model-world landmarks. Playback follows "
        "recorded timestamp spacing and preserves peak frames during sampling."
    )
    available_sampling = get_processed_fps(
        context["study"]["folder_name"], context["participant_id"], task["folder_name"]
    )
    if not available_sampling:
        st.info("Run a current analysis before opening the 3-D activity view.")
    else:
        activity_sampling = st.selectbox(
            "Result sampling",
            available_sampling,
            format_func=lambda value: "Native frame rate"
            if value is None
            else f"{value} frames/second",
            key="activity_3d_sampling",
        )
        result_info = detect_current_results(
            context["study"]["folder_name"],
            context["participant_id"],
            task["folder_name"],
            activity_sampling,
        )
        if not result_info:
            st.error("The selected analysis is incomplete or not scientifically traceable.")
        else:
            try:
                bundle = _load_dashboard_bundle(result_info["dir"])
                store: JsonlResultStore = bundle["store"]
            except DashboardDataError as exc:
                st.error(str(exc))
            else:
                records = bundle["records"]
                people = sorted(
                    {
                        str(record.get("person_identifier"))
                        for record in records
                        if record.get("person_identifier") is not None
                    }
                )
                if not people:
                    st.warning("No tracked person is available in this result set.")
                else:
                    selected_person = st.selectbox(
                        "Tracked person", people, key="activity_3d_person"
                    )
                    person_records = [
                        record
                        for record in records
                        if str(record.get("person_identifier")) == selected_person
                    ]
                    camera_roles = bundle["config"].get("fusion", {}).get("camera_roles", {})
                    camera_count = max(len(bundle["summary"].get("video_metadata", [])), 1)
                    roles_by_index = (
                        {int(index): str(role) for role, index in camera_roles.items()}
                        if isinstance(camera_roles, dict)
                        else {}
                    )
                    controls = st.columns(4)
                    camera_index = 0 if camera_count == 1 else controls[0].selectbox(
                        "Model-world camera source",
                        list(range(camera_count)),
                        format_func=lambda index: (
                            f"Camera {index + 1} · "
                            f"{roles_by_index.get(index, 'role unrecorded').replace('_', ' ').title()}"
                        ),
                        key="activity_3d_camera",
                    )
                    selected_region = controls[1].selectbox(
                        "Highlighted region",
                        list(REGION_LABELS),
                        format_func=lambda key: REGION_LABELS[key],
                        key="activity_3d_region",
                    )
                    timestamps = sorted(
                        float(record.get("timestamp_sec", 0)) for record in person_records
                    )
                    minimum_time, maximum_time = timestamps[0], timestamps[-1]
                    window_duration = controls[2].selectbox(
                        "Playback window",
                        [5.0, 10.0, 20.0, 30.0],
                        index=1,
                        format_func=lambda value: f"{value:g} seconds",
                    )
                    latest_start = max(minimum_time, maximum_time - window_duration)
                    window_start = (
                        controls[3].slider(
                            "Window start (s)",
                            float(minimum_time),
                            float(latest_start),
                            float(minimum_time),
                            step=0.1,
                        )
                        if latest_start > minimum_time
                        else minimum_time
                    )
                    window_end = min(float(window_start) + window_duration, maximum_time)
                    full_subset = _cached_landmark_subset(
                        store.fingerprint.path,
                        store.fingerprint.size_bytes,
                        store.fingerprint.mtime_ns,
                        selected_person,
                        10,
                    )
                    animation_records = [
                        record
                        for record in full_subset
                        if float(window_start)
                        <= float(record.get("timestamp_sec", 0))
                        <= window_end
                    ]
                    if not animation_records:
                        st.warning("No stored model-world landmarks fall inside this window.")
                    else:
                        st.plotly_chart(
                            stick_figure_3d(
                                list(animation_records),
                                selected_region=selected_region,
                                camera_index=int(camera_index),
                                maximum_animation_frames=180,
                            ),
                            width="stretch",
                            key="activity_3d_figure",
                        )
                    st.warning(
                        "This view uses one camera's hip-centered monocular model estimate. "
                        "It is not calibrated, triangulated 3-D motion capture, does not recover "
                        "room displacement, and must not be used for joint moments or forces."
                    )
                    window_records = [
                        record
                        for record in person_records
                        if float(window_start)
                        <= float(record.get("timestamp_sec", 0))
                        <= window_end
                    ]
                    chart_left, chart_right = st.columns([1.5, 1])
                    with chart_left:
                        selected_angles = st.multiselect(
                            "Displayed angle evidence",
                            [
                                "trunk_flexion",
                                "neck_flexion",
                                "left_upper_arm_flexion",
                                "right_upper_arm_flexion",
                                "left_knee_flexion",
                                "right_knee_flexion",
                            ],
                            default=["trunk_flexion", "neck_flexion"],
                            format_func=lambda name: name.replace("_", " ").title(),
                            key="activity_3d_angles",
                        )
                        activity_event = st.plotly_chart(
                            angle_timeline_figure(window_records, selected_angles),
                            width="stretch",
                            on_select="rerun",
                            selection_mode="points",
                            key="activity_3d_angle_timeline",
                        )
                        activity_frame = _result_event_value(activity_event)
                        if activity_frame is not None:
                            st.session_state["selected_frame"] = int(activity_frame)
                    with chart_right:
                        st.markdown("#### Window provenance")
                        st.metric("Start", f"{float(window_start):.2f} s")
                        st.metric("End", f"{window_end:.2f} s")
                        st.metric("Dense pose observations", len(window_records))
                        st.metric("Animation evidence frames", len(animation_records))
                        st.caption(
                            "The dense trace remains separate from the peak-preserving landmark "
                            "sample used for interactive 3-D rendering."
                        )

if active_view == FRAME_REVIEW_SECTION:
    st.subheader("Frame review and interpretation")
    st.markdown(
        "Inspect the same observed moment across every recorded camera. The shared frame "
        "controller drives all views, overlays, measurements, and method explanations."
    )
    if st.session_state.get("selected_frame") is not None and st.button(
        "← Back to Results", key="frame_review_back"
    ):
        st.session_state["pending_dashboard_section"] = "Results"
        st.rerun()
    available_sampling = get_processed_fps(
        context["study"]["folder_name"],
        context["participant_id"],
        task["folder_name"],
    )
    if not available_sampling:
        st.info("Run a current analysis before reviewing frame-level results.")
    else:
        detail_sampling = st.selectbox(
            "Result sampling",
            available_sampling,
            format_func=lambda value: "Native frame rate"
            if value is None
            else f"{value} frames/second",
            key="detail_sampling",
        )
        result_info = detect_current_results(
            context["study"]["folder_name"],
            context["participant_id"],
            task["folder_name"],
            detail_sampling,
        )
        if not result_info:
            st.error("The selected current analysis is incomplete or not traceable.")
        else:
            try:
                bundle = _load_dashboard_bundle(result_info["dir"])
                store: JsonlResultStore = bundle["store"]
            except (DashboardDataError, ResultStoreError) as exc:
                st.error(str(exc))
            else:
                records = bundle["records"]
                people = sorted(
                    {
                        str(record.get("person_identifier"))
                        for record in records
                        if record.get("person_identifier") is not None
                    }
                )
                if not people:
                    st.warning("No tracked person is available in this result set.")
                else:
                    selected_person = st.selectbox("Tracked person", people, key="detail_person")
                    person_records = [
                        record
                        for record in records
                        if str(record.get("person_identifier")) == selected_person
                    ]
                    frame_options = sorted(
                        {int(record.get("frame_number", 0)) for record in person_records}
                    )
                    scope_key = (
                        f"{context['key']}_{task['task_id']}_{detail_sampling}_{selected_person}"
                    )
                    slider_key = f"detail_frame_{scope_key}"
                    pending_key = f"{slider_key}__pending"
                    # A frame requested from the timeline chart is parked here
                    # and applied now, before the slider widget exists. Writing
                    # the widget's own key after it is created is what Streamlit
                    # rejects, so requests are never applied at the call site.
                    if pending_key in st.session_state:
                        pending_frame = int(st.session_state.pop(pending_key))
                        if pending_frame in frame_options:
                            st.session_state[slider_key] = pending_frame
                    requested = int(st.session_state.get("selected_frame", frame_options[0]))
                    nearest_frame = min(frame_options, key=lambda value: abs(value - requested))
                    if st.session_state.get("detail_scope") != scope_key:
                        st.session_state["detail_scope"] = scope_key
                        st.session_state[slider_key] = nearest_frame
                    current_frame = int(st.session_state.get(slider_key, nearest_frame))
                    if current_frame not in frame_options:
                        st.session_state[slider_key] = nearest_frame
                        current_frame = nearest_frame
                    current_index = frame_options.index(current_frame)
                    frame_timestamps = {
                        int(item.get("frame_number", 0)): float(item["timestamp_sec"])
                        for item in person_records
                        if isinstance(item.get("timestamp_sec"), (int, float))
                    }

                    nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 5, 1, 1])

                    # Every navigation button is handled before the slider is
                    # created. Streamlit refuses to let a widget's session-state
                    # key be written once that widget exists in the same run,
                    # which is what made "Next" and "+1 s" raise. The columns
                    # were created above, so on-screen order is unchanged.
                    def _jump_to(frame_value: int) -> None:
                        st.session_state[slider_key] = int(frame_value)
                        st.rerun()

                    def _frame_after(offset_sec: float, fallback_index: int) -> int:
                        base = frame_timestamps.get(current_frame)
                        if base is None:
                            return frame_options[fallback_index]
                        target = base + offset_sec
                        return min(
                            frame_timestamps,
                            key=lambda value: abs(frame_timestamps[value] - target),
                        )

                    at_start = current_index == 0
                    at_end = current_index == len(frame_options) - 1
                    if nav1.button(
                        "−1 s",
                        key=f"detail_minus_second_{scope_key}",
                        disabled=at_start,
                        width="stretch",
                    ):
                        _jump_to(_frame_after(-1.0, max(0, current_index - 1)))
                    if nav2.button(
                        "← Previous",
                        key=f"detail_previous_{scope_key}",
                        disabled=at_start,
                        width="stretch",
                    ):
                        _jump_to(frame_options[current_index - 1])
                    if nav4.button(
                        "Next →",
                        key=f"detail_next_{scope_key}",
                        disabled=at_end,
                        width="stretch",
                    ):
                        _jump_to(frame_options[current_index + 1])
                    if nav5.button(
                        "+1 s",
                        key=f"detail_plus_second_{scope_key}",
                        disabled=at_end,
                        width="stretch",
                    ):
                        _jump_to(_frame_after(1.0, min(len(frame_options) - 1, current_index + 1)))

                    with nav3:
                        selected_frame = st.select_slider(
                            "Observed source frame",
                            frame_options,
                            key=slider_key,
                        )
                    st.session_state["selected_frame"] = int(selected_frame)

                    entry = _record_entry(store, selected_person, int(selected_frame))
                    try:
                        record = store.load_record_at_offset(entry.offset)
                    except ResultStoreError as exc:
                        st.error(str(exc))
                        st.stop()

                    control_left, control_middle, control_right = st.columns([1.2, 1, 1])
                    selected_region = control_left.selectbox(
                        "Highlighted body region",
                        list(REGION_LABELS),
                        index=list(REGION_LABELS).index(
                            st.session_state.get("selected_region", "trunk")
                        )
                        if st.session_state.get("selected_region", "trunk") in REGION_LABELS
                        else 1,
                        format_func=lambda key: REGION_LABELS[key],
                        key=f"detail_region_{scope_key}",
                    )
                    method = control_middle.radio(
                        "Interpretation method",
                        ["RULA", "REBA"],
                        horizontal=True,
                        key=f"detail_method_{scope_key}",
                    )
                    overlay_enabled = control_right.checkbox(
                        "Show recorded pose overlay",
                        value=True,
                        key=f"detail_overlay_{scope_key}",
                    )

                    references = _camera_references(record, bundle["summary"], bundle["config"])
                    synchronization = record.get("synchronization", {})
                    legacy_sync = any(
                        ref.get("synchronization_status") == "exact_offsets_unknown"
                        for ref in references
                    )
                    with st.container(border=True):
                        multiple_views = len(references) > 1
                        header_columns = st.columns(4 if multiple_views else 2)
                        header_columns[0].metric("Common frame" if multiple_views else "Frame", f"{int(entry.frame_number):,}")
                        header_columns[1].metric(
                            "Common time" if multiple_views else "Time", f"{float(entry.timestamp_sec or 0):.3f} s"
                        )
                        if multiple_views:
                            header_columns[2].metric("Recorded views", len(references))
                            header_columns[3].metric(
                                "Synchronization",
                                "Offsets unknown"
                                if legacy_sync
                                else _status_text(synchronization.get("status", "recorded")),
                            )
                        if multiple_views and legacy_sync:
                            st.warning(
                                "This retained run predates per-view timestamp references. The "
                                "frames below use the common source index; exact inter-camera "
                                "offsets were not stored and are not reported as zero."
                            )
                        elif multiple_views and synchronization:
                            spread = synchronization.get("timestamp_spread_sec")
                            st.caption(
                                f"Mode: {_status_text(synchronization.get('mode'))} · "
                                f"observed spread: "
                                + (
                                    f"{float(spread) * 1000:.2f} ms"
                                    if isinstance(spread, (int, float))
                                    else "not recorded"
                                )
                                + " · no temporal alignment was invented"
                            )

                        if references:
                            view_columns = st.columns(len(references))
                            for view_column, reference in zip(
                                view_columns, references, strict=True
                            ):
                                camera_index = int(reference.get("camera_index", 0))
                                role = (
                                    str(reference.get("role") or f"camera {camera_index + 1}")
                                    .replace("_", " ")
                                    .title()
                                )
                                source = reference.get("path")
                                source_frame = reference.get("source_frame_number")
                                with view_column:
                                    st.markdown(f"**Camera {camera_index + 1} · {role}**" if multiple_views else "**Camera**")
                                    if (
                                        source
                                        and source_frame is not None
                                        and Path(str(source)).is_file()
                                    ):
                                        try:
                                            video_stat = Path(str(source)).stat()
                                            image = _cached_review_image(
                                                str(source),
                                                int(video_stat.st_mtime_ns),
                                                int(source_frame),
                                                store.fingerprint.path,
                                                store.fingerprint.size_bytes,
                                                store.fingerprint.mtime_ns,
                                                int(entry.offset),
                                                selected_region,
                                                camera_index,
                                                overlay_enabled,
                                            )
                                            st.image(image, width="stretch")
                                        except (
                                            OSError,
                                            DashboardDataError,
                                            ResultStoreError,
                                        ) as exc:
                                            st.error(f"Frame unavailable: {exc}")
                                    else:
                                        st.error("The recorded source frame is unavailable.")
                                    delta = reference.get("sync_delta_ms")
                                    view_time = reference.get("timestamp_sec")
                                    st.caption(
                                        f"Source frame {source_frame if source_frame is not None else '—'} · "
                                        + (
                                            f"{float(view_time):.3f} s"
                                            if isinstance(view_time, (int, float))
                                            else "timestamp unavailable"
                                        )
                                        + ((" · " + (f"Δt {float(delta):+.2f} ms" if isinstance(delta, (int, float))
                                                      else "exact Δt not stored")) if multiple_views else "")
                                    )
                                    landmarks_by_camera = record.get(
                                        "pose_landmarks_normalized_by_camera", {}
                                    )
                                    camera_landmarks = (
                                        landmarks_by_camera.get(str(camera_index), {})
                                        if isinstance(landmarks_by_camera, dict)
                                        else {}
                                    )
                                    if not camera_landmarks and not multiple_views:
                                        camera_landmarks = record.get("pose_landmarks_normalized") or {}
                                    visibility = [
                                        float(point[3])
                                        for point in camera_landmarks.values()
                                        if isinstance(point, list)
                                        and len(point) > 3
                                        and isinstance(point[3], (int, float))
                                    ]
                                    st.caption(
                                        "Mean landmark visibility: "
                                        + (
                                            f"{sum(visibility) / len(visibility):.2f}"
                                            if visibility
                                            else "unavailable"
                                        )
                                    )
                        else:
                            st.info("No camera evidence references are stored for this frame.")
                    st.caption(
                        "Gold lines show the stored pose skeleton; red markers identify the "
                        "selected review region. Each view shows a monocular pose estimate."
                    )

                    interpretation = interpret_frame(record, method, selected_region)
                    explanation_col, bilateral_col = st.columns([1.45, 1])
                    with explanation_col:
                        st.markdown(f"### {interpretation['headline']}")
                        st.caption(
                            f"Validity: {_status_text(interpretation.get('status'))} · "
                            f"Selected anatomical side: "
                            f"{record.get(method.lower(), {}).get('selected_side') or 'not selected'}"
                        )
                        st.markdown("**What the result means**")
                        st.write(interpretation["meaning"])
                        st.markdown("**Why it occurred**")
                        st.write(interpretation["cause"])
                        if interpretation.get("selection_reason"):
                            st.info(interpretation["selection_reason"])
                        if interpretation.get("missing"):
                            st.warning(
                                "Missing or invalid evidence: "
                                + ", ".join(interpretation["missing"])
                            )
                        for warning in interpretation.get("warnings", []):
                            st.warning(str(warning))
                    with bilateral_col:
                        st.markdown("**Left/right method status**")
                        method_result = record.get(method.lower(), {})
                        side_rows = []
                        for side in ("left", "right"):
                            side_result = method_result.get(side, {})
                            side_rows.append(
                                {
                                    "Side": side.title(),
                                    "Validity": _status_text(side_result.get("validity_status")),
                                    "Final score": side_result.get("final_score"),
                                    "Missing": ", ".join(side_result.get("missing_inputs", []))
                                    or "—",
                                    "Invalid": ", ".join(side_result.get("invalid_inputs", []))
                                    or "—",
                                }
                            )
                        st.dataframe(pd.DataFrame(side_rows), hide_index=True, width="stretch")

                    st.markdown("#### Measurements and method trace")
                    measure_col, score_col, quality_col = st.columns(3)
                    with measure_col:
                        st.markdown("**Observed angle evidence**")
                        raw_angle_units = record.get("angle_units", "degrees")
                        angle_sources = record.get("multi_view_selection", {})
                        angle_rows = [
                            {
                                "Measurement": name.replace("_", " ").title(),
                                "Value": round(float(value), 2),
                                "Unit": (
                                    raw_angle_units.get(name, "degrees")
                                    if isinstance(raw_angle_units, dict)
                                    else str(raw_angle_units or "degrees")
                                ),
                                "Selected view": angle_sources.get(name, "unavailable"),
                            }
                            for name, value in sorted(record.get("angles", {}).items())
                            if isinstance(value, (int, float))
                        ]
                        st.dataframe(pd.DataFrame(angle_rows), hide_index=True, width="stretch")
                    with score_col:
                        st.markdown("**Intermediate scores**")
                        intermediate = interpretation.get("intermediate_scores", {})
                        if intermediate:
                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        "Component": [
                                            key.replace("_", " ").title() for key in intermediate
                                        ],
                                        "Score": list(intermediate.values()),
                                    }
                                ),
                                hide_index=True,
                                width="stretch",
                            )
                        else:
                            st.info(
                                "Intermediate scores are unavailable because the formal "
                                "assessment is incomplete for this frame."
                            )
                    with quality_col:
                        st.markdown("**Pose and acquisition quality**")
                        quality = record.get("pose_quality", {})
                        quality_rows = [
                            {
                                "Field": "Mean landmark visibility",
                                "Value": quality.get("mean_keypoint_visibility"),
                            },
                            {
                                "Field": "Camera suitability",
                                "Value": quality.get("camera_view_status"),
                            },
                            {
                                "Field": "Wrist evidence source",
                                "Value": quality.get("wrist_data_source") or "unavailable",
                            },
                            {
                                "Field": "Calibration",
                                "Value": record.get("calibration_status"),
                            },
                            {
                                "Field": "Timestamp source",
                                "Value": record.get("timestamp_source"),
                            },
                            {"Field": "Tracked person", "Value": selected_person},
                        ]
                        quality_table = pd.DataFrame(quality_rows)
                        quality_table["Value"] = quality_table["Value"].map(_display_cell)
                        st.dataframe(quality_table, hide_index=True, width="stretch")

                    st.markdown("#### Nearby evidence timeline")
                    nearby = [
                        item
                        for item in person_records
                        if abs(
                            float(item.get("timestamp_sec", 0))
                            - float(record.get("timestamp_sec", 0))
                        )
                        <= 3.0
                    ]
                    has_scores = any(
                        isinstance(item.get(method.lower(), {}).get("selected"), dict)
                        for item in nearby
                    )
                    nearby_figure = (
                        score_timeline_figure(nearby, method)
                        if has_scores
                        else angle_timeline_figure(nearby, ["trunk_flexion", "neck_flexion"])
                    )
                    nearby_event = st.plotly_chart(
                        nearby_figure,
                        width="stretch",
                        on_select="rerun",
                        selection_mode="points",
                        key=f"detail_timeline_{scope_key}",
                    )
                    nearby_frame = _result_event_value(nearby_event)
                    if (
                        nearby_frame is not None
                        and int(nearby_frame) in frame_options
                        and int(nearby_frame) != int(selected_frame)
                    ):
                        st.session_state[pending_key] = int(nearby_frame)
                        st.session_state["selected_frame"] = int(nearby_frame)
                        st.rerun()

st.markdown("---")
st.caption(
    "RULA and REBA are observational screening methods. RNLE applies only to supported "
    "two-handed lifting tasks. Pose confidence and ergonomic validity are reported separately."
)
