"""A claim must refer to the same measurement or image the assistant received."""

import json

from assistant.grounding import Citation, TurnEvidence


def test_measurement_and_image_receipts_are_distinct_and_preserve_camera_identity():
    evidence = TurnEvidence()
    scope = {"person": "worker", "method": "rula"}
    evidence.record({"frame": 8, "method": "rula", "score": 5}, scope)
    tag = "[observed · saw frame 8 · camera 1]"
    assert evidence.citations(tag, [], scope)[1]
    for camera in (0, 1):
        evidence.image({**scope, "frame": 8, "sec": 0.8, "camera": camera, "view": "raw"})
    refs, issues = evidence.citations(tag, [], scope)
    assert not issues
    assert [(r.camera, r.kind, r.person, r.sec) for r in refs] == [(1, "observed", "worker", 0.8)]
    assert evidence.citations("[observed · saw frame 8]", [], scope)[1]
    assert evidence.resolve(Citation(frame=8, person="someone-else"), scope) is None
    assert evidence.resolve(Citation(frame=8, method="reba"), scope) is None


def test_check_joint_side_and_method_even_when_another_record_contains_the_number():
    evidence = TurnEvidence()
    for method, score in (("rula", 6), ("reba", 8)):
        evidence.record(
            {
                "frame": 12,
                "method": method,
                "score": score,
                "angles": {"left_shoulder_flexion": 24, "right_shoulder_flexion": 32},
                "sides": {"left": {"score": score}, "right": {"score": 3}},
            },
            {},
        )
    for claim, expected in [
        (
            "RULA score is 6 [computed · frame 12 method=rula]. REBA score is 8 [computed · frame 12 method=reba]",
            True,
        ),
        ("RULA score is 8 [computed · frame 12 method=rula]", False),
        ("Right-side RULA score is 6 [computed · frame 12 method=rula]", False),
        ("Left shoulder flexion was 32 degrees [computed · frame 12 method=rula]", False),
        ("Left shoulder flexion was 24 degrees [computed · frame 12 method=rula]", True),
    ]:
        refs, issues = evidence.citations(claim, [], {})
        assert not issues
        assert (not evidence.check_measurements(claim, refs)) == expected, claim


def test_transcripts_only_reopen_links_for_the_current_result(tmp_path, monkeypatch):
    from assistant import conversation

    monkeypatch.setattr(conversation, "resolve_result_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(conversation, "_current_fingerprint", lambda _: "current")
    base = {"question": "What happened?", "answer": "A reading.", "citations": [{"frame": 1}]}
    turns = [base, {**base, "resultFingerprint": "old"}, {**base, "resultFingerprint": "current"}]
    (tmp_path / "assistant_transcript.jsonl").write_text(
        "\n".join(json.dumps(t) for t in turns), encoding="utf-8"
    )
    (current,) = conversation.load_transcript("s")
    assert current["citations"] == [{"frame": 1}]


def test_mixed_followup_keeps_verified_score_when_visual_claim_fails(monkeypatch):
    from assistant import conversation
    from assistant.backend import StreamChunk
    from assistant.config import AssistantConfig
    from assistant.tools import ToolResult

    scope = {"frame": 8, "method": "rula", "person": "worker", "camera": 0}
    monkeypatch.setattr(conversation, "_selection_context", lambda *_: scope)
    monkeypatch.setattr(conversation, "_current_fingerprint", lambda _: None)
    monkeypatch.setattr(conversation, "analysis_brief", lambda *_: "")
    monkeypatch.setattr(conversation, "session_brief", lambda *_: "")
    monkeypatch.setattr(
        conversation, "effective_config", lambda _: AssistantConfig(log_conversations=False)
    )
    monkeypatch.setattr(
        conversation,
        "run_tool",
        lambda *a, **k: ToolResult(
            {"frame": 8, "method": "rula", "person": "worker", "score": 7}, frames=[8]
        ),
    )
    monkeypatch.setattr(
        conversation,
        "stream_completion",
        lambda *a, **k: iter(
            [
                StreamChunk(
                    text="It weighs 19 kg. [observed · saw frame 8 · camera 0]", finished=True
                )
            ]
        ),
    )
    result = list(
        conversation.run_turn(
            "s", "What weight can you see and what is the RULA score here?", selection=scope
        )
    )[-1]
    assert result["type"] == "done"
    assert "is 7" in result["text"] and "visual detail" in result["text"]
    assert "19" not in result["text"]
    assert result["framesSeen"] == []
    assert [r["kind"] for r in result["citations"]] == ["computed"]
