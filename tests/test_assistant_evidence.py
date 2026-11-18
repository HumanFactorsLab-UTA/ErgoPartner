"""The three things that make the visual reading a measurement rather than a guess.

Frame selection, numeric provenance, and inter-pass agreement are each testable
without a model server, and each is the sort of thing that fails silently: a
selection that quietly reverts to time spacing, an auditor that passes
everything, an agreement statistic computed over identical evidence. These are
the tests that would have caught all three.
"""

from __future__ import annotations

import pytest

# ----------------------------------------------------------------------
# Numeric provenance
# ----------------------------------------------------------------------


def _evidence(*values: float):
    from assistant.audit import Evidence

    found = Evidence()
    for value in values:
        found.add(value)
    return found


def test_a_figure_from_a_tool_result_is_traced() -> None:
    from assistant.audit import audit_answer

    report = audit_answer("The trunk reached 32 degrees.", _evidence(31.96, 9, 4.03))
    assert report.total == 1
    assert report.traced == 1
    assert not report.untraced


def test_a_figure_that_was_never_shown_is_flagged() -> None:
    """The auditor exists for exactly this: a number with no source behind it."""

    from assistant.audit import audit_answer

    report = audit_answer("The trunk reached 71 degrees.", _evidence(31.96, 9, 4.03))
    assert report.traced == 0
    assert [claim.written for claim in report.untraced] == ["71"]
    assert "71" in report.untraced[0].context


def test_rounding_is_not_treated_as_fabrication() -> None:
    """The pipeline stores 31.96 and any sane answer says 32."""

    from assistant.audit import audit_answer

    for written, stored in (("32", 31.96), ("4.0", 4.031), ("122.3", 122.34), ("38", 37.8)):
        report = audit_answer(f"Value {written} here.", _evidence(stored))
        assert report.traced == 1, f"{written} should trace to {stored}"


def test_citations_and_register_tags_are_not_audited() -> None:
    """Their frame numbers are validated separately; auditing them double-counts."""

    from assistant.audit import audit_answer

    report = audit_answer(
        "The trunk drove it. [computed · frame 825 · 0:28]\n<<CITE frame=825 t=28.0>>",
        _evidence(),
    )
    assert report.total == 0


def test_evidence_is_collected_from_nested_tool_results() -> None:
    from assistant.audit import Evidence, audit_answer

    found = Evidence()
    found.absorb(
        {
            "method": "REBA",
            "verdict": {"peakScore": 9, "typicalScore": 4.03},
            "episodes": [{"drivers": [{"pointsAdded": 1.8}]}],
        }
    )
    report = audit_answer("Peak 9, typical 4.03, shoulder added 1.8.", found)
    assert report.untraced == []


def test_evidence_is_collected_from_the_brief_prose() -> None:
    from assistant.audit import Evidence, audit_answer

    found = Evidence()
    found.absorb_text("REBA peaked at 9 at 0:28. Trunk +1.14 mean, 67.7% above neutral.")
    report = audit_answer("It peaked at 9 with the trunk at 67.7% above neutral.", found)
    assert report.untraced == []


def test_model_names_and_versions_are_not_read_as_measurements() -> None:
    """`qwen3-vl:30b` and `H.264` are names; flagging them would bury a real miss."""

    from assistant.audit import audit_answer

    report = audit_answer("Run with qwen3-vl:30b on H.264 proxies, software 4.0.0-audit.",
                          _evidence())
    assert report.untraced == []


def test_nearby_values_are_not_evidence_at_the_written_precision() -> None:
    from assistant.audit import audit_answer

    report = audit_answer("The angle was 99.98 degrees, or about 100 degrees.", _evidence(100.02))
    assert [claim.written for claim in report.untraced] == ["99.98"]
    assert report.traced == 1


def test_chat_and_visual_claims_cannot_certify_their_own_numbers(monkeypatch) -> None:
    from assistant import conversation
    from assistant.backend import StreamChunk
    from assistant.config import AssistantConfig

    monkeypatch.setattr(conversation, "effective_config", lambda _: AssistantConfig(log_conversations=False))
    monkeypatch.setattr(conversation, "analysis_brief", lambda _: "Trunk 27.15 degrees.")
    monkeypatch.setattr(conversation, "session_brief", lambda _: "Trunk 27.15 degrees. Scene guessed 63.")
    monkeypatch.setattr(conversation, "stream_completion", lambda *a, **kw: iter([
        StreamChunk(text="27.15, 51.23, 63, 412 and 71.32 degrees.", finished=True)
    ]))
    events = list(conversation.run_turn("s", "Is it 71.32 degrees?",
                                        [{"role": "assistant", "content": "It was 51.23."}]))
    assert events[-1]["type"] == "done"
    assert events[-1]["audit"]["untraced"] == []
    assert "did not verify" in events[-1]["text"]
    assert "71.32" not in events[-1]["text"]


def test_proposal_requires_valid_value_and_evidence_from_its_own_pass() -> None:
    from assistant.scene import _clean_proposals

    values = [
        {"condition": "coupling", "value": "fair", "evidence_frames": [1, 1, 2]},
        {"condition": "coupling", "value": "good", "evidence_frames": [2]},
        {"condition": "coupling", "value": "easy", "evidence_frames": [1]},
        {"condition": "load_kg", "value_range": [3, 9], "evidence_frames": [1]},
        {"condition": "left_arm_supported_or_leaning", "value": "false", "evidence_frames": [1]},
        {"condition": "right_arm_supported_or_leaning", "value": True, "evidence_frames": [True, 1.5]},
    ]
    assert _clean_proposals(values, {1}) == [{
        "condition": "coupling", "value": "fair", "evidence_frames": [1],
        "rationale": "", "confidence": "low",
    }]


def test_split_proposal_cites_only_frames_supporting_selected_value() -> None:
    from assistant.consistency import merge_proposals

    result = merge_proposals([
        [{"condition": "coupling", "value": "poor", "evidence_frames": [42]}],
        [{"condition": "coupling", "value": "good", "evidence_frames": [70]}],
    ])[0]
    assert result["contested"] is True
    assert result["evidence_frames"] == [42]


def test_scene_rejects_uncalibrated_dimensions_and_requires_structured_sections():
    import json

    from assistant.backend import BackendError
    from assistant.scene import _parse

    card = _parse(json.dumps({
        "task_inference": {}, "workstation": {"surfaces": ["cart (0.7m high)", "hip-high shelf"]},
    }))
    assert card["workstation"]["surfaces"] == ["hip-high shelf"]
    with pytest.raises(BackendError):
        _parse('{"workstation": "floor"}')


def test_invalid_legacy_confirmations_never_become_scoring_inputs(monkeypatch) -> None:
    from assistant import proposals

    values = {"load_kg": -1, "coupling": "fair", "shock_or_rapid_force": "false",
              "right_arm_supported_or_leaning": False, "invented": True}
    monkeypatch.setattr(proposals, "_read", lambda _: {"decisions": {
        key: {"status": "accepted", "value": value} for key, value in values.items()
    }})
    assert proposals.entered_conditions("s") == {
        "coupling": "fair", "right_arm_supported_or_leaning": False,
    }


# ----------------------------------------------------------------------
# Inter-pass agreement
# ----------------------------------------------------------------------


def test_unanimous_readings_report_full_agreement() -> None:
    from assistant.consistency import vote

    result = vote(["fair", "fair", "fair"])
    assert result.value == "fair"
    assert result.fraction == 1.0
    assert result.payload()["band"] == "unanimous"


def test_a_majority_keeps_the_minority_visible() -> None:
    from assistant.consistency import vote

    result = vote(["fair", "poor", "fair"])
    assert result.value == "fair"
    assert result.agreeing == 2
    assert result.alternatives == ["poor"]
    assert result.payload()["band"] == "majority"


def test_a_reading_only_one_pass_made_is_contested() -> None:
    """Silence counts against a finding: two passes declining to propose it is
    disagreement, not abstention."""

    from assistant.consistency import CONTESTED_BELOW, vote

    result = vote([None, True, None])
    assert result.value is True
    assert result.fraction < CONTESTED_BELOW
    assert result.payload()["band"] == "contested"


def test_merged_proposals_carry_agreement_and_sort_by_it() -> None:
    from assistant.consistency import merge_proposals

    merged = merge_proposals(
        [
            [
                {"condition": "coupling", "value": "fair", "confidence": "high",
                 "rationale": "bare carton", "evidence_frames": [1]},
                {"condition": "shock_or_rapid_force", "value": True, "confidence": "low",
                 "rationale": "", "evidence_frames": [2]},
            ],
            [
                {"condition": "coupling", "value": "fair", "confidence": "high",
                 "rationale": "no handles", "evidence_frames": [3]},
            ],
            [
                {"condition": "coupling", "value": "fair", "confidence": "medium",
                 "rationale": "rim grip", "evidence_frames": [4]},
            ],
        ]
    )
    assert [item["condition"] for item in merged] == ["coupling", "shock_or_rapid_force"]
    assert merged[0]["agreement"]["agreeing"] == 3
    assert merged[0]["contested"] is False
    assert merged[0]["evidence_frames"] == [1, 3, 4]
    assert merged[1]["contested"] is True


def test_a_load_range_survives_merging_as_a_range() -> None:
    from assistant.consistency import merge_proposals

    merged = merge_proposals(
        [
            [{"condition": "load_kg", "value_range": [2.0, 5.0], "confidence": "low",
              "rationale": "", "evidence_frames": []}],
            [{"condition": "load_kg", "value_range": [2.0, 5.0], "confidence": "low",
              "rationale": "", "evidence_frames": []}],
        ]
    )
    assert merged[0]["value_range"] == [2.0, 5.0]
    assert "value" not in merged[0]


def test_an_object_only_one_pass_saw_is_dropped() -> None:
    """A lone sighting across three readings is the shape of a visual hallucination."""

    from assistant.consistency import merge_workstation

    merged = merge_workstation(
        [
            {"objects": ["cardboard box", "rolling cart"]},
            {"objects": ["cardboard box", "forklift"]},
            {"objects": ["cardboard box", "rolling cart"]},
        ]
    )
    assert "cardboard box" in merged["objects"]
    assert "rolling cart" in merged["objects"]
    assert "forklift" not in merged["objects"]


def test_with_two_readings_nothing_is_dropped_for_being_lonely() -> None:
    from assistant.consistency import merge_workstation

    merged = merge_workstation([{"objects": ["box"]}, {"objects": ["cart"]}])
    assert sorted(merged["objects"]) == ["box", "cart"]


# ----------------------------------------------------------------------
# Frame selection
# ----------------------------------------------------------------------


def test_holdout_folds_select_different_frames() -> None:
    """The agreement statistic is meaningless if every pass sees the same frames.

    An earlier version rotated tie-breaking, which was far too weak to change a
    greedy trajectory, and all three passes received the identical twelve frames.
    """

    pytest.importorskip("numpy")
    from assistant.sampling import representative_frames

    session = _analysed_session()
    if session is None:
        pytest.skip("No analysed trial is available in this checkout")

    folds = [
        {sample.frame for sample in representative_frames(session, 12, holdout=(k, 3))}
        for k in range(3)
    ]
    assert all(len(fold) >= 8 for fold in folds)
    assert len(folds[0] | folds[1] | folds[2]) > len(folds[0]), (
        "Holdout folds must widen the evidence, not repeat it"
    )
    assert folds[0] != folds[1] or folds[1] != folds[2]


def test_posture_selection_beats_time_spacing() -> None:
    """The whole argument for this module, measured rather than asserted."""

    pytest.importorskip("numpy")
    from api.analysis import timeline
    from assistant.sampling import coverage, representative_frames

    session = _analysed_session()
    if session is None:
        pytest.skip("No analysed trial is available in this checkout")

    picked = [sample.frame for sample in representative_frames(session, 12)]
    points = timeline(session, "reba")["series"]["points"]
    step = max(1, len(points) // 12)
    spaced = [int(point["f"]) for point in points[::step][:12]]

    assert coverage(session, picked) > coverage(session, spaced)


def test_the_worst_moment_is_always_included() -> None:
    """Coverage is never a reason to omit the frame the assessment is about."""

    pytest.importorskip("numpy")
    from api.analysis import timeline
    from assistant.sampling import representative_frames

    session = _analysed_session()
    if session is None:
        pytest.skip("No analysed trial is available in this checkout")

    peak = timeline(session, "reba")["verdict"]["peakFrame"]
    for fold in range(3):
        frames = [s.frame for s in representative_frames(session, 12, holdout=(fold, 3))]
        assert peak in frames, f"fold {fold} dropped the peak frame"


def _analysed_session() -> str | None:
    """Sampling reads scored records and can also run on an archived trial."""

    from pathlib import Path

    from api.sessions import SessionError, resolve_result_dir
    from dashboard.data_loader import get_results_root

    root = Path(get_results_root()) / "10fps"
    for record in sorted(root.glob("*/*/*/frame_results.jsonl")):
        study, participant, task = record.parent.relative_to(root).parts
        if task.startswith("."):
            continue
        session = f"{study}__{participant}__{task}__10"
        try:
            resolve_result_dir(session)
        except SessionError:
            continue
        return session
    return None
