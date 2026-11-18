"""The assistant's guarantees, as tests.

These are not tests of answer quality -- a language model's wording is not a
property you assert. They are tests of the boundary: that the scoring packages
never depend on the assistant, that the assistant never writes a score, that a
proposal cannot reach a result file on its own, and that the register tags a
reviewer relies on actually parse.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The deterministic core. Nothing in here may depend on a language model being
#: present, reachable, or correct.
SCORING_PACKAGES = ("core", "ergonomics", "pipeline", "temporal", "reporting")


def _imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


@pytest.mark.parametrize("package", SCORING_PACKAGES)
def test_scoring_packages_never_import_the_assistant(package: str) -> None:
    """The scoring path must run identically with the model server switched off."""

    offenders: list[str] = []
    for path in (PROJECT_ROOT / package).rglob("*.py"):
        for name in _imports(path):
            if name == "assistant" or name.startswith("assistant."):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)} imports {name}")
    assert not offenders, "The deterministic scoring path gained a model dependency: " + "; ".join(
        offenders
    )


def test_assistant_never_writes_a_result_file() -> None:
    """No module in the package may open a scored artifact for writing."""

    protected = ("frame_results.jsonl", "video_summary.json", "run_config.json")
    offenders: list[str] = []
    for path in (PROJECT_ROOT / "assistant").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in protected:
            if name not in text:
                continue
            # frame_results.jsonl is read by brief.py for the condition values;
            # what must never appear is a write mode alongside it.
            for line in text.splitlines():
                if name in line and ('"w"' in line or "'w'" in line or '"a"' in line):
                    offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, "The assistant must not write scored artifacts: " + "; ".join(offenders)


def test_only_unmeasurable_conditions_can_be_proposed() -> None:
    """A proposal may only touch a field no pose model could have supplied.

    If these ever diverge, the assistant could overrule a measurement, which is
    exactly what the authority ladder exists to prevent.
    """

    from assistant.scene import PROPOSABLE
    from ergonomics.condition_resolution import NOT_OBSERVABLE_FROM_VIDEO

    assert PROPOSABLE <= NOT_OBSERVABLE_FROM_VIDEO, (
        "The assistant may only propose conditions the pipeline itself marks as "
        f"not observable from video. Offenders: {sorted(PROPOSABLE - NOT_OBSERVABLE_FROM_VIDEO)}"
    )


def test_rejects_a_proposal_for_a_measured_condition() -> None:
    from api.sessions import SessionError
    from assistant.proposals import accept_proposal

    with pytest.raises(SessionError):
        accept_proposal("any__session__id__10", "trunk_flexion", 42)


def test_citations_parse_and_strip() -> None:
    from assistant.conversation import parse_citations

    text = (
        "The trunk drove it. [computed · frame 412 · 0:44]\n\n"
        "<<CITE frame=412 t=44.2 method=reba region=trunk>>\n"
        "<<CITE frame=825 t=28.0 method=reba>>"
    )
    cleaned, citations = parse_citations(text)

    assert "<<CITE" not in cleaned
    assert cleaned.startswith("The trunk drove it.")
    assert [item.frame for item in citations] == [412, 825]
    assert citations[0].region == "trunk"
    assert citations[0].sec == pytest.approx(44.2)
    assert citations[1].region is None


def test_citation_parser_survives_malformed_footers() -> None:
    from assistant.conversation import parse_citations

    cleaned, citations = parse_citations(
        "Answer. <<CITE frame=abc t=x>> <<CITE >> <<CITE frame=7 t=1.0 method=rula>>"
    )
    assert "<<CITE" not in cleaned
    assert [item.frame for item in citations] == [7]
    assert citations[0].method == "rula"


@pytest.mark.parametrize(
    ("written", "expected"),
    [("44.2", 44.2), ("0:28", 28.0), ("1:02:03", 3723.0), ("nonsense", None), ("", None)],
)
def test_a_timestamp_in_clock_form_still_yields_a_usable_citation(
    written: str, expected: float | None
) -> None:
    """The frame is the control; the timestamp is only its label.

    Models write "0:28" where seconds were asked for. Discarding the citation
    over that would throw away a working seek control for a formatting slip.
    """

    from assistant.conversation import parse_citations

    _, citations = parse_citations(f"<<CITE frame=825 t={written} method=reba>>")
    assert [item.frame for item in citations] == [825]
    assert citations[0].sec == expected


def test_duplicate_citations_collapse() -> None:
    from assistant.conversation import parse_citations

    _, citations = parse_citations(
        "<<CITE frame=1 t=0.1 region=trunk>><<CITE frame=1 t=0.1 region=trunk>>"
    )
    assert len(citations) == 1


def test_citations_to_frames_the_session_never_scored_are_dropped(monkeypatch) -> None:
    """A citation is a control, so an invented frame must not reach the client.

    Models do invent them -- including by copying the illustrative frame number
    out of their own instructions -- and the resulting chip would seek the
    workspace to a frame that does not exist.
    """

    import assistant.conversation as conversation

    monkeypatch.setattr(conversation, "valid_frames", lambda _session: frozenset({17, 825}))
    _, citations = conversation.parse_citations(
        "<<CITE frame=825 t=28.0>> <<CITE frame=412 t=44.2>> <<CITE frame=17 t=0.5>>",
        "study__002__task__10",
    )
    assert [item.frame for item in citations] == [825, 17]


def test_citations_are_returned_unchecked_without_a_session() -> None:
    from assistant.conversation import parse_citations

    _, citations = parse_citations("<<CITE frame=99999 t=1.0>>")
    assert [item.frame for item in citations] == [99999]


def test_empty_session_drops_citations_and_invalid_timestamps_are_omitted(monkeypatch) -> None:
    import assistant.conversation as conversation

    text = "<<CITE frame=1.7 t=1>> <<CITE frame=2 t=inf method=reba>> <<CITE frame=2 t=nan method=rula>>"
    monkeypatch.setattr(conversation, "valid_frames", lambda _: frozenset())
    assert conversation.parse_citations(text, "s")[1] == []
    _, parsed = conversation.parse_citations(text)
    assert [(c.frame, c.sec, c.method) for c in parsed] == [(2, None, "reba"), (2, None, "rula")]


def test_health_requires_requested_model_tags_for_both_roles(monkeypatch) -> None:
    from types import SimpleNamespace

    from assistant import backend
    from assistant.config import AssistantConfig

    models = {"data": [{"id": "example:8b"}]}
    monkeypatch.setattr(backend.requests, "get", lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: models,
    ))
    config = AssistantConfig(chat_model="example:8b", scene_model="example:30b", dialect="ollama")
    assert backend.health(config)["missing"] == ["example:30b"]
    models["data"].append({"id": "example:30b"})
    assert backend.health(config)["ok"] is True


def test_load_kg_is_never_accepted_as_an_entered_value() -> None:
    """A visual range is not a measurement, so it must not become an entered load."""

    import assistant.proposals as proposals

    document = {
        "decisions": {
            "load_kg": {"status": "accepted", "value": [8, 14]},
            "coupling": {"status": "accepted", "value": "poor"},
            "shock_or_rapid_force": {"status": "rejected", "value": True},
        }
    }
    original = proposals._read
    proposals._read = lambda _session_id: document  # type: ignore[assignment]
    try:
        entered = proposals.entered_conditions("session")
    finally:
        proposals._read = original  # type: ignore[assignment]

    assert entered == {"coupling": "poor"}


def test_the_live_question_carries_the_format_reminder_and_history_does_not() -> None:
    """Restating the contract beside the question is what stops tagging drift.

    The rules sit thousands of tokens back in the system prompt, behind a full
    assessment brief, and the model reliably follows them once and then stops --
    most visibly on recommendation questions. Replayed history must stay clean so
    the reminder does not accumulate once per turn.
    """

    from assistant.conversation import build_messages
    from assistant.prompts import TURN_REMINDER

    history = [
        {"role": "user", "content": "Why did REBA peak?"},
        {"role": "assistant", "content": "The trunk drove it."},
    ]
    messages = build_messages("session", history, "What should we change first?")

    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("What should we change first?")
    assert TURN_REMINDER in messages[-1]["content"]
    assert sum(TURN_REMINDER in str(item.get("content")) for item in messages) == 1


def test_a_confirmed_condition_fills_a_gap_but_never_overrides_an_entered_value() -> None:
    """Where an accepted proposal sits in the authority ladder.

    Accepting one records an operator decision, so it outranks the assumption
    profile that would otherwise answer. It does not outrank a value typed into
    assessment_inputs.yaml, which was entered just as deliberately and more
    specifically.
    """

    import api.jobs as jobs

    original = jobs.confirmed_conditions
    jobs.confirmed_conditions = lambda _session: {"coupling": "fair", "load_kg": 6.0}
    try:
        unanswered = {"task_inputs": {"coupling": None, "load_kg": None}}
        jobs._apply_confirmed_conditions(unanswered, "session")
        assert unanswered["task_inputs"] == {"coupling": "fair", "load_kg": 6.0}

        entered = {"task_inputs": {"coupling": "good", "load_kg": None}}
        jobs._apply_confirmed_conditions(entered, "session")
        assert entered["task_inputs"]["coupling"] == "good", "a typed value must win"
        assert entered["task_inputs"]["load_kg"] == 6.0
    finally:
        jobs.confirmed_conditions = original


def test_a_run_is_never_blocked_by_the_assistant_being_unavailable() -> None:
    """The scoring path must not acquire a dependency on an optional layer."""

    import api.jobs as jobs

    original = jobs.confirmed_conditions

    def explode(_session: str):
        raise RuntimeError("model server on fire")

    jobs.confirmed_conditions = explode
    try:
        config = {"task_inputs": {"coupling": None}}
        with pytest.raises(RuntimeError):
            jobs.confirmed_conditions("session")
        # The wrapper is what the run calls, and it swallows the failure.
        jobs.confirmed_conditions = original
        config_after = {"task_inputs": {"coupling": None}}
        jobs._apply_confirmed_conditions(config_after, "no__such__session__10")
        assert config_after["task_inputs"]["coupling"] is None
        assert config == {"task_inputs": {"coupling": None}}
    finally:
        jobs.confirmed_conditions = original


def test_confirmed_conditions_are_empty_for_an_unknown_session() -> None:
    from api.jobs import confirmed_conditions

    assert confirmed_conditions("no__such__session__10") == {}


def test_tool_schemas_are_well_formed() -> None:
    """Every advertised tool must be callable and describe itself to the model."""

    from assistant.tools import TOOLS, tool_schemas

    schemas = tool_schemas()
    assert len(schemas) == len(TOOLS)
    for schema in schemas:
        function = schema["function"]
        assert function["name"] in TOOLS
        assert function["description"].strip()
        assert function["parameters"]["type"] == "object"
        for name in function["parameters"].get("required", []):
            assert name in function["parameters"]["properties"], (
                f"{function['name']} requires {name!r} but does not declare it"
            )


def test_unknown_tool_returns_an_error_rather_than_raising() -> None:
    """A tool failure must reach the model as text; raising would lose the turn."""

    from assistant.tools import run_tool

    result = run_tool("no_such_tool", "session", {})
    assert "error" in result.content
    assert not result.images


@pytest.mark.parametrize("tool", ["frame", "look_at"])
def test_exact_frame_tools_report_nearest_frame_without_quoting_its_score(tool, monkeypatch):
    from assistant import tools

    monkeypatch.setattr(tools.analysis_api, "frame_detail", lambda *a: {"frame": 1008, "score": 7})
    result = tools.run_tool(tool, "s", {"frame": 1007})
    assert "1007" in result.content["error"] and "1008" in result.content["error"]
    assert "score" not in result.content and not result.frames and not result.images


def test_frame_tool_exposes_the_measurement_sources(monkeypatch):
    from assistant import tools

    data = {
        "frame": 3, "clock": "0:01", "timestampSec": 0.7, "score": 4, "band": {},
        "selectedSide": "right", "quality": {}, "components": {}, "drivers": [],
        "angles": {"trunk_flexion": 11}, "angleSources": {"trunk_flexion": "camera_1"},
        "anglesByCamera": {"0": {"trunk_flexion": 21}, "1": {"trunk_flexion": 11}},
    }
    monkeypatch.setattr(tools.analysis_api, "frame_detail", lambda *a: data)
    result = tools.run_tool("frame", "s", {"frame": 3, "method": "rula"}).content
    assert result["method"] == "rula"
    assert result["anglesByCamera"] == data["anglesByCamera"]
    assert result["angleSources"] == data["angleSources"]


def test_visual_tool_cannot_claim_an_image_after_budget_exhaustion():
    from assistant.tools import run_tool

    result = run_tool("look_at", "s", {"frame": 333}, remaining_images=0)
    assert "No image" in result.content["error"]
    assert not result.images and not result.frames


def test_reference_search_cites_a_real_document() -> None:
    from assistant.reference import CORPUS, search_reference

    results = search_reference("REBA coupling score")
    assert results, "The method documentation should answer a question about REBA coupling"
    for entry in results:
        assert entry["document"] in CORPUS
        assert entry["citation"].startswith(entry["document"])
        assert (PROJECT_ROOT / entry["document"]).is_file()


def test_config_defaults_are_usable_without_a_config_block() -> None:
    from assistant.config import _coerce

    config = _coerce(None)
    assert config.endpoint.startswith("http")
    assert config.budget("scene") > config.budget("moment")
    assert config.api_key, "A missing key env var must still yield a sendable header"
