"""The HTTP surface the browser client is written against.

These are contract tests. The React client reads specific field names off these
payloads, so a rename that a Python caller would survive silently breaks a screen
that nothing else in this repository type-checks.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from api.sessions import SessionError, parse_session_id

# ----------------------------------------------------------------------
# Session identifiers
# ----------------------------------------------------------------------


def test_a_session_id_round_trips_through_its_four_parts():
    ref = parse_session_id("study_20260212_125912__001__task1_niosh_lifting__native")

    assert ref.study_folder == "study_20260212_125912"
    assert ref.participant_id == "001"
    assert ref.task_folder == "task1_niosh_lifting"
    assert ref.sampling is None
    assert ref.session_id == "study_20260212_125912__001__task1_niosh_lifting__native"


def test_a_sampled_session_keeps_its_frame_rate():
    ref = parse_session_id("s__001__task2__10")

    assert ref.sampling == 10
    assert ref.sampling_label == "10 fps"
    assert ref.session_id == "s__001__task2__10"


def test_task_folders_containing_single_underscores_survive_parsing():
    """A single underscore is ordinary inside a folder name; the separator is two."""

    ref = parse_session_id("study_a_b__001__task1_niosh_lifting__native")

    assert ref.task_folder == "task1_niosh_lifting"


@pytest.mark.parametrize(
    "value",
    ["", "too__few__parts", "a__b__c__d__e", "a__b__c__", "a__b__c__sixty-ish"],
)
def test_a_malformed_session_id_is_rejected_rather_than_guessed(value: str):
    with pytest.raises(SessionError):
        parse_session_id(value)


# ----------------------------------------------------------------------
# Payload shaping
# ----------------------------------------------------------------------


def test_band_payload_carries_the_label_and_the_action_not_only_a_colour():
    """The client colours a band, and a colour alone is not readable."""

    from api.analysis import _band_payload

    reba = _band_payload(13, "reba")
    assert reba is not None
    assert reba["label"] == "Very high"
    assert reba["action"] == "Action is necessary NOW"
    assert reba["status"] == "critical"
    assert reba["describesRisk"] is True

    rula = _band_payload(7, "rula")
    assert rula is not None
    assert rula["label"] == "Change now"
    assert rula["describesRisk"] is False

    assert _band_payload(None, "reba") is None


def test_an_unknown_method_is_refused():
    from api.analysis import _check_method

    assert _check_method("REBA") == "reba"
    with pytest.raises(SessionError):
        _check_method("owas")


def test_trial_card_uses_the_first_complete_sampling_not_the_first_folder(
    monkeypatch: pytest.MonkeyPatch,
):
    from api import main

    context = {
        "study": {"folder_name": "study"},
        "participant_id": "001",
        "date_label": "2026-01-01",
    }
    trial = {
        "folder_name": "task",
        "name": "Task",
        "cameras": 3,
    }
    monkeypatch.setattr(main, "get_processed_fps", lambda *_: [None, 10])
    monkeypatch.setattr(
        main,
        "detect_current_results",
        lambda _study, _participant, _task, sampling: (
            None if sampling is None else {"dir": "complete"}
        ),
    )
    monkeypatch.setattr(
        main,
        "load_summary",
        lambda _: {
            "duration_sec": 12.0,
            "reba_summary": {"maximum_score": 8, "observation_count": 100},
            "rula_summary": {"maximum_score": 5, "observation_count": 100},
        },
    )

    card = main._trial_card(context, trial)

    assert card["status"] == "analysed"
    assert card["sessionId"] == "study__001__task__10"


def test_series_decimation_keeps_both_extremes_of_every_bucket():
    """The browser draws the envelope, so a dropped peak is a lie on screen."""

    from api.analysis import _decimate
    from dashboard.task_timeline import ScorePoint

    def point(index: int, score: int) -> ScorePoint:
        return ScorePoint(
            method="reba",
            frame_number=index,
            timestamp_sec=index * 0.1,
            score=score,
            side="left",
            dwell_sec=0.1,
            components={},
            left_components={},
            right_components={},
            angles={},
        )

    series = [point(i, 13 if i == 500 else (2 if i % 7 else 8)) for i in range(1200)]
    thinned = _decimate(series, 200)

    assert len(thinned) <= 202
    assert max(p.score for p in thinned) == 13
    assert min(p.score for p in thinned) == 2
    assert [p.timestamp_sec for p in thinned] == sorted(p.timestamp_sec for p in thinned)


def test_a_short_series_is_returned_untouched():
    from api.analysis import _decimate

    assert _decimate([], 100) == []


# ----------------------------------------------------------------------
# Video proxies
# ----------------------------------------------------------------------


def test_a_missing_source_reports_failure_rather_than_starting_a_build(tmp_path):
    from api.media import ensure_proxy

    status = ensure_proxy(tmp_path / "absent.mp4", tmp_path, 0)

    assert status.state in {"failed", "unavailable"}
    assert status.path is None
    assert status.detail


def test_a_built_proxy_is_reported_ready_without_rebuilding(tmp_path):
    """An existing proxy must short-circuit before ffmpeg is consulted."""

    from api.media import ensure_proxy, proxy_path
    from pipeline.recording_sources import recording_hash

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source content")
    target = proxy_path(tmp_path, 2, recording_hash(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not really an mp4, but non-empty")

    status = ensure_proxy(tmp_path / "source.mp4", tmp_path, 2)

    assert status.state == "ready"
    assert status.path == target
    source.write_bytes(b"updated source")
    assert ensure_proxy(source, tmp_path, 2, start=False).path is None


def test_camera_evidence_keeps_the_selected_person_and_source_frame_without_overlay(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Turning the drawing off must not lose multi-camera synchronization."""

    from api import main
    from dashboard import analysis_service

    entries = [
        SimpleNamespace(frame_number=12, person_identifier="person-a", offset=1),
        SimpleNamespace(frame_number=12, person_identifier="person-b", offset=2),
    ]
    records = {
        1: {
            "person_identifier": "person-a",
            "camera_frame_references": [
                {"camera_index": 0, "source_frame_number": 101},
            ],
        },
        2: {
            "person_identifier": "person-b",
            "camera_frame_references": [
                {"camera_index": 0, "source_frame_number": 202},
            ],
        },
    }
    store = SimpleNamespace(
        entries=entries,
        load_record_at_offset=lambda offset: records[offset],
    )
    received: list[tuple[int, dict | None]] = []

    def fake_read_video_frame(
        _path,
        source_frame,
        record=None,
        _region=None,
        _camera_index=0,
    ):
        received.append((source_frame, record))
        return np.zeros((8, 8, 3), dtype=np.uint8)

    source_path = tmp_path / "x.mp4"
    source_path.touch()
    (tmp_path / "frame_results.jsonl").touch()
    monkeypatch.setattr(main, "session_videos", lambda *_: [{"camera_index": 0, "path": str(source_path)}])
    monkeypatch.setattr(main, "resolve_result_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(main, "open_result_store", lambda _: store)
    monkeypatch.setattr(analysis_service, "read_video_frame", fake_read_video_frame)
    main._decoded_jpeg.cache_clear()

    main._frame_jpeg(
        "study__001__task__10",
        12,
        0,
        "",
        False,
        720,
        "person-b",
    )

    assert received == [(202, None)]
    main._decoded_jpeg.cache_clear()


def test_proxies_live_beside_the_analysis_not_in_the_raw_capture(tmp_path):
    from api.media import PROXY_DIRNAME, proxy_path

    path = proxy_path(tmp_path / "results", 1, "abc123")

    assert path.parent.name == PROXY_DIRNAME
    assert path.parent.parent.name == "results"
    assert path.name == "cam1-abc123.mp4"


# ----------------------------------------------------------------------
# The framework-free cache the data layer now uses
# ----------------------------------------------------------------------


def test_the_cache_exposes_both_clear_names_so_the_swap_was_not_a_rename():
    from dashboard.cache import ttl_cache

    @ttl_cache(ttl=60)
    def scan() -> list[int]:
        return [1, 2, 3]

    assert scan() == [1, 2, 3]
    assert callable(scan.clear)
    assert callable(scan.cache_clear)


def test_a_cached_value_cannot_be_corrupted_by_its_caller():
    from dashboard.cache import ttl_cache

    calls = {"n": 0}

    @ttl_cache(ttl=60)
    def scan() -> dict[str, list[int]]:
        calls["n"] += 1
        return {"items": [1, 2]}

    first = scan()
    first["items"].append(99)
    second = scan()

    assert calls["n"] == 1, "the second call should have been served from cache"
    assert second["items"] == [1, 2], "one caller's edit must not rewrite the next read"


def test_an_unhashable_argument_bypasses_the_cache_instead_of_raising():
    from dashboard.cache import ttl_cache

    @ttl_cache(ttl=60)
    def scan(value: object) -> str:
        return str(value)

    assert scan(["a", "b"]) == "['a', 'b']"


def test_expiry_recomputes():
    from dashboard.cache import ttl_cache

    calls = {"n": 0}

    @ttl_cache(ttl=0.0)
    def scan() -> int:
        calls["n"] += 1
        return calls["n"]

    scan()
    scan()

    assert calls["n"] == 2
