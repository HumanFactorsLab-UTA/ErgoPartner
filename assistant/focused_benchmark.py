"""Freeze independent visual labels and model predictions, then score paired cases."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from assistant.focused_review import (
    PROMPT_VERSION,
    QUESTIONS,
    Answers,
    Case,
    StaleVisualReview,
    data_path,
    get_case,
    predictor,
    read_images,
    run_key,
)
from assistant.models import effective_config


@contextmanager
def _connect(session_id: str, *, write: bool = False) -> Iterator[sqlite3.Connection | None]:
    path = data_path(session_id)
    if not write and not path.is_file():
        yield None
        return
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        if write:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""CREATE TABLE IF NOT EXISTS cases
                         (id TEXT PRIMARY KEY, run TEXT NOT NULL, info TEXT NOT NULL, person TEXT NOT NULL,
                          start REAL, end REAL, revealed INTEGER NOT NULL DEFAULT 0, reference TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS predictions
                         (case_id TEXT NOT NULL, evaluator TEXT NOT NULL, value TEXT NOT NULL, decision TEXT,
                          PRIMARY KEY (case_id, evaluator))""")
        elif not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cases'"
        ).fetchone():
            yield None
            return
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _stamp() -> str:
    return datetime.now(UTC).isoformat()


def _interval(case: Case) -> tuple[float, float]:
    times = [ref["sec"] for ref in case.evidence]
    return min(times), max(times)


def _register(conn, case: Case) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO cases(id,run,info,person,start,end) VALUES(?,?,?,?,?,?)",
        (case.id, case.run, json.dumps(case.metadata()), case.person, *_interval(case)),
    )


def _is_blind(conn, case: Case) -> bool:
    start, end = _interval(case)
    return not conn.execute(
        """SELECT 1 FROM cases WHERE run=? AND person=? AND revealed=1
                             AND start<=? AND end>=? LIMIT 1""",
        (case.run, case.person, end, start),
    ).fetchone()


def _decode(row, field):
    return json.loads(row[field]) if row is not None and row[field] else None


def describe(case: Case, config=None) -> dict:
    config = config or effective_config(case.session_id)
    with _connect(case.session_id) as conn:
        if conn:
            record = conn.execute("SELECT reference FROM cases WHERE id=?", (case.id,)).fetchone()
            model = conn.execute(
                "SELECT value,decision FROM predictions WHERE case_id=? AND evaluator=?",
                (case.id, predictor(config)),
            ).fetchone()
        else:
            record = model = None
        return {
            **case.metadata(),
            "evidence": case.evidence,
            "blindEligible": (record is None or not record["reference"])
            and (conn is None or _is_blind(conn, case)),
            "questions": [
                {
                    "key": key,
                    "prompt": prompt,
                    "options": [
                        {"value": value, "label": label} for value, label in options.items()
                    ],
                }
                for key, (prompt, options) in QUESTIONS.items()
            ],
            "reference": _decode(record, "reference"),
            "prediction": _decode(model, "value"),
            "decision": _decode(model, "decision"),
        }


def _current(case: Case) -> None:
    if run_key(case.session_id) != case.run:
        raise StaleVisualReview(
            "The recording or analysis changed. Open the current moment and review its images."
        )


def save_reference(case: Case, answers: Answers) -> dict:
    _current(case)
    with _connect(case.session_id, write=True) as conn:
        _register(conn, case)
        prior = _decode(
            conn.execute("SELECT reference FROM cases WHERE id=?", (case.id,)).fetchone(),
            "reference",
        )
        if prior:
            if prior["answers"] != answers.model_dump():
                raise StaleVisualReview("This independent reference was already locked.")
        elif not _is_blind(conn, case):
            raise StaleVisualReview(
                "AI suggestions were revealed for this moment or overlapping footage. Choose a fresh moment for a blind label."
            )
        else:
            row = {"answers": answers.model_dump(), "at": _stamp()}
            conn.execute("UPDATE cases SET reference=? WHERE id=?", (json.dumps(row), case.id))
    return describe(case)


def predict(case: Case) -> dict:
    config = effective_config(case.session_id)
    existing = describe(case, config)
    if existing["prediction"]:
        return existing
    reading = read_images(case, config)
    reading["at"] = _stamp()
    _current(case)
    with _connect(case.session_id, write=True) as conn:
        _register(conn, case)
        conn.execute(
            "INSERT OR IGNORE INTO predictions(case_id,evaluator,value) VALUES(?,?,?)",
            (case.id, predictor(config), json.dumps(reading)),
        )
        conn.execute("UPDATE cases SET revealed=1 WHERE id=?", (case.id,))
    return describe(case, config)


def save_decision(case: Case, prediction_id: str, answers: Answers) -> dict:
    config = effective_config(case.session_id)
    _current(case)
    with _connect(case.session_id, write=True) as conn:
        model = conn.execute(
            "SELECT value FROM predictions WHERE case_id=? AND evaluator=?",
            (case.id, predictor(config)),
        ).fetchone()
        reading = _decode(model, "value")
        if not reading or reading["id"] != prediction_id:
            raise StaleVisualReview(
                "The displayed AI reading changed. Review the current suggestions."
            )
        decision = {"answers": answers.model_dump(), "at": _stamp(), "predictionId": prediction_id}
        conn.execute(
            "UPDATE predictions SET decision=? WHERE case_id=? AND evaluator=?",
            (json.dumps(decision), case.id, predictor(config)),
        )
    return describe(case, config)


def records(session_id: str, config=None) -> list[dict]:
    key = run_key(session_id)
    with _connect(session_id) as conn:
        if not conn:
            return []
        rows = conn.execute(
            """SELECT c.info,c.reference,c.revealed,p.value,p.decision FROM cases c
                             LEFT JOIN predictions p ON c.id=p.case_id AND p.evaluator=? WHERE c.run=?
                             ORDER BY c.id""",
            (predictor(config or effective_config(session_id)), key),
        )
        return [
            {
                **json.loads(row["info"]),
                "reference": _decode(row, "reference"),
                "prediction": _decode(row, "value"),
                "decision": _decode(row, "decision"),
                "revealed": bool(row["revealed"]),
            }
            for row in rows
        ]


def _metrics(pairs: list[dict]) -> dict:
    measured = {}
    for key, (_, options) in QUESTIONS.items():
        table = {v: {p: 0 for p in options} for v in options}
        for row in pairs:
            expected = row["reference"]["answers"][key]
            predicted = row["prediction"]["answers"][key]["value"]
            table[expected][predicted] += 1
        by_class = {}
        for value, counts in table.items():
            n = sum(counts.values())
            by_class[value] = {
                "n": n,
                "correct": counts[value],
                "accuracy": counts[value] / n if n else None,
            }
        correct = sum(c["correct"] for c in by_class.values())
        unclear = by_class["unclear"]
        measured[key] = {
            "n": len(pairs),
            "correct": correct,
            "accuracy": correct / len(pairs) if pairs else None,
            "unclearCases": unclear["n"],
            "correctAbstentions": unclear["correct"],
            "falseDefinites": unclear["n"] - unclear["correct"],
            "unnecessaryAbstentions": sum(t["unclear"] for v, t in table.items() if v != "unclear"),
            "byClass": by_class,
            "confusion": table,
        }
    return measured


def evaluation(rows: list[dict], config=None) -> dict:
    pairs = [r for r in rows if r["reference"] and r["prediction"]]

    def groups(options, get):
        result = {}
        for option in options:
            subset = [r for r in pairs if get(r) == option]
            result[option] = {"caseCount": len(subset), "metrics": _metrics(subset)}
        return result

    return {
        "caseCount": len(rows),
        "labelledCases": sum(bool(r["reference"]) for r in rows),
        "pairedCases": len(pairs),
        "assistedCases": sum(bool(r["revealed"] and not r["reference"]) for r in rows),
        "model": (config or effective_config()).scene_model,
        "promptVersion": PROMPT_VERSION,
        "metrics": _metrics(pairs),
        "byVisibility": groups(
            QUESTIONS["visibility"][1], lambda r: r["reference"]["answers"]["visibility"]
        ),
        "byCameraMode": groups(("single", "multi"), lambda r: r["cameraMode"]),
    }


def library_evaluation(studies_root: str) -> dict:
    from api.sessions import SessionError, SessionRef, catalog
    from dashboard.data_loader import get_results_root

    active = {
        (p["study"]["folder_name"], str(p["participant_id"]), t["folder_name"])
        for p in catalog(studies_root)
        for t in p["tasks"]
    }
    root = Path(get_results_root())
    rows, skipped = [], 0
    config = effective_config()
    for path in root.glob("*/*/*/*/focused_review.sqlite3"):
        rate, study, participant, trial = path.relative_to(root).parts[:-1]
        if (study, participant, trial) not in active:
            continue
        if rate != "native" and not (rate.endswith("fps") and rate[:-3].isdigit()):
            continue
        session = SessionRef(
            study, participant, trial, None if rate == "native" else int(rate[:-3])
        ).session_id
        try:
            if data_path(session).resolve() != path.resolve():
                continue
            current = records(session, config)
            if current:
                rows.extend(current)
            else:
                skipped += 1
        except (SessionError, OSError, sqlite3.Error):
            skipped += 1
    return {
        **evaluation(rows, config),
        "sessionCount": len({r["sessionId"] for r in rows}),
        "pairedSessions": len({r["sessionId"] for r in rows if r["reference"] and r["prediction"]}),
        "skippedSessions": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed grip, arm-support and visibility labels across reserved study sessions."
    )
    parser.add_argument("sessions", nargs="+", help="Assessment session IDs from the dashboard.")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Read independent cases missing a prediction for the configured model and prompt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON metrics, frozen references and individual predictions.",
    )
    args = parser.parse_args()
    rows = []
    for session in dict.fromkeys(args.sessions):
        items = records(session)
        if args.predict:
            for item in items:
                if item["reference"] and not item["prediction"]:
                    case = get_case(
                        session, item["frame"], item["person"], item["method"], item["camera"]
                    )
                    if case.id != item["caseId"]:
                        raise StaleVisualReview("The evidence for a labelled case has changed.")
                    predict(case)
            items = records(session)
        rows.extend(items)
    report = evaluation(rows)
    if args.output:
        args.output.write_text(json.dumps({**report, "cases": rows}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
