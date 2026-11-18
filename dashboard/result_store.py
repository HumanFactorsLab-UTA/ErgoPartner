"""Lazy, fingerprinted access to large frame-result JSON Lines files.

The dashboard's traceable result records contain large per-camera landmark
maps.  Charts and filters do not need those maps, so this module indexes the
file once, retains a lightweight copy of each record, and stores the byte
offset needed to retrieve an exact full record later.

All file offsets are measured in binary mode.  This is important because a
UTF-8 character can occupy more than one byte and text-mode ``tell`` values are
not portable byte offsets.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


class ResultStoreError(ValueError):
    """Raised when a result file cannot be indexed or retrieved safely."""


HEAVY_FIELD_PREFIXES = ("pose_landmarks_",)
LIGHTWEIGHT_TOP_LEVEL_FIELDS = (
    "software_version",
    "frame_number",
    "timestamp_sec",
    "person_identifier",
    "calibration_status",
    "timestamp_source",
    "pose_quality",
    "angles",
    "angle_units",
    "multi_view_selection",
    "camera_frame_refs",
    "camera_frame_references",
    "angles_by_camera",
    "per_camera_angles",
    "synchronization",
    "rula",
    "reba",
    "task_conditions",
)
METHOD_TOP_LEVEL_FIELDS = ("assessment_type", "selected_side", "selection_reason")
METHOD_RESULT_FIELDS = (
    "final_score",
    "validity_status",
    "risk_level",
    "action_level",
    "action_category",
    "intermediate_scores",
    "missing_inputs",
    "invalid_inputs",
    "warnings",
    "side",
)
DEFAULT_PEAK_ANGLE_NAMES = (
    "trunk_flexion",
    "neck_flexion",
    "left_upper_arm_flexion",
    "right_upper_arm_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
)


@dataclass(frozen=True)
class ResultFileFingerprint:
    """Fields that define one immutable view of a result file."""

    path: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class ResultIndexEntry:
    """One JSONL record's lookup metadata and lightweight representation."""

    offset: int
    line_number: int
    frame_number: int | None
    timestamp_sec: float | None
    person_identifier: str | None
    has_landmarks: bool
    lightweight: dict[str, Any]


def _fingerprint(path: str | Path) -> ResultFileFingerprint:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ResultStoreError(
            f"Result file does not exist or cannot be resolved: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise ResultStoreError(f"Result path is not a file: {resolved}")
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ResultStoreError(f"Could not inspect result file: {resolved}") from exc
    return ResultFileFingerprint(str(resolved), int(stat.st_size), int(stat.st_mtime_ns))


def _decode_record(raw: bytes, *, path: str, line_number: int) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResultStoreError(f"{path}: line {line_number} is not valid UTF-8: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultStoreError(
            f"{path}: malformed JSON on line {line_number}: {exc.msg} (column {exc.colno})"
        ) from exc
    if not isinstance(value, dict):
        raise ResultStoreError(f"{path}: line {line_number} must contain a JSON object")
    return value


def _is_heavy_field(name: str) -> bool:
    return name.startswith(HEAVY_FIELD_PREFIXES)


def _select_mapping_fields(value: Any, fields: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return value
    return {field: value[field] for field in fields if field in value}


def _compact_method(value: Any) -> Any:
    """Keep only RULA/REBA fields used by filtering and result charts."""

    if not isinstance(value, dict):
        return value
    compact = _select_mapping_fields(value, METHOD_TOP_LEVEL_FIELDS)
    for side in ("left", "right"):
        if side in value:
            compact[side] = _select_mapping_fields(value[side], METHOD_RESULT_FIELDS)
    if "selected" in value:
        selected = value["selected"]
        compact["selected"] = (
            _select_mapping_fields(selected, METHOD_RESULT_FIELDS) if selected is not None else None
        )
    return compact


def _compact_conditions(value: Any) -> Any:
    """Reduce per-frame task conditions to what the Results page reads.

    The full record carries an evidence entry per condition per frame -- the
    measurement, the subject's neutral, the threshold, the margin. Held for
    every frame of a long trial that is a large multiple of everything else in
    the lightweight index, which exists precisely to stay small.

    Only three things are read from it: which conditions were true, where each
    answer came from, and how far past neutral the worst frame went. Keep those
    and drop the rest; the full evidence is still one seek away in the JSONL for
    any frame a reviewer opens.
    """

    if not isinstance(value, dict):
        return None
    values = value.get("values")
    compact: dict[str, Any] = {
        "values": {
            name: item for name, item in values.items() if item is not None and item is not False
        }
        if isinstance(values, dict)
        else {},
        "sources": value.get("sources") if isinstance(value.get("sources"), dict) else {},
        "conflicts": value.get("conflicts") or [],
    }
    evidence = value.get("evidence")
    if isinstance(evidence, dict):
        departures = {
            name: item.get("departure_deg")
            for name, item in evidence.items()
            if isinstance(item, dict) and item.get("departure_deg") is not None
        }
        if departures:
            compact["evidence"] = {
                name: {"departure_deg": departure} for name, departure in departures.items()
            }
    return compact


def _lightweight_record(record: dict[str, Any]) -> dict[str, Any]:
    """Copy only fields needed for charts, filters, quality, and media lookup."""

    lightweight: dict[str, Any] = {}
    for name in LIGHTWEIGHT_TOP_LEVEL_FIELDS:
        if name not in record:
            continue
        value = record[name]
        if name in {"rula", "reba"}:
            lightweight[name] = _compact_method(value)
        elif name == "task_conditions":
            lightweight[name] = _compact_conditions(value)
        else:
            lightweight[name] = value
    return lightweight


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or int(value) != value:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _optional_person(value: Any) -> str | None:
    return str(value) if value is not None else None


class JsonlResultStore:
    """Immutable index plus lazy full-record retrieval for one JSONL file."""

    def __init__(
        self,
        fingerprint: ResultFileFingerprint,
        entries: tuple[ResultIndexEntry, ...],
    ) -> None:
        self.fingerprint = fingerprint
        self.entries = entries
        self._entries_by_offset = {entry.offset: entry for entry in entries}

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def build(cls, fingerprint: ResultFileFingerprint) -> JsonlResultStore:
        entries: list[ResultIndexEntry] = []
        path = Path(fingerprint.path)
        try:
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    line_number += 1
                    if not raw.strip():
                        continue
                    record = _decode_record(
                        raw,
                        path=fingerprint.path,
                        line_number=line_number,
                    )
                    has_landmarks = any(
                        _is_heavy_field(name) and isinstance(value, dict) and bool(value)
                        for name, value in record.items()
                    )
                    entries.append(
                        ResultIndexEntry(
                            offset=offset,
                            line_number=line_number,
                            frame_number=_optional_int(record.get("frame_number")),
                            timestamp_sec=_optional_float(record.get("timestamp_sec")),
                            person_identifier=_optional_person(record.get("person_identifier")),
                            has_landmarks=has_landmarks,
                            lightweight=_lightweight_record(record),
                        )
                    )
        except ResultStoreError:
            raise
        except OSError as exc:
            raise ResultStoreError(f"Could not read result file: {fingerprint.path}") from exc

        if _fingerprint(fingerprint.path) != fingerprint:
            raise ResultStoreError(
                "Result file changed while it was being indexed; open it again to retry"
            )
        return cls(fingerprint, tuple(entries))

    def _assert_current(self) -> None:
        if _fingerprint(self.fingerprint.path) != self.fingerprint:
            raise ResultStoreError(
                "Result store is stale because the source file changed; reopen the result store"
            )

    def lightweight_records(self) -> list[dict[str, Any]]:
        """Return chart/filter records without any ``pose_landmarks_*`` maps.

        A defensive deep copy prevents callers from mutating the process-wide
        cached store.
        """

        return [copy.deepcopy(entry.lightweight) for entry in self.entries]

    def load_record_at_offset(self, offset: int) -> dict[str, Any]:
        """Load one exact full record from a previously indexed byte offset."""

        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ResultStoreError("Record offset must be an integer")
        entry = self._entries_by_offset.get(offset)
        if entry is None:
            raise ResultStoreError(f"Offset {offset} is not a record boundary in this result file")
        self._assert_current()
        try:
            with Path(self.fingerprint.path).open("rb") as handle:
                handle.seek(offset)
                raw = handle.readline()
        except OSError as exc:
            raise ResultStoreError(f"Could not retrieve result record at offset {offset}") from exc
        if not raw:
            raise ResultStoreError(f"No result record exists at offset {offset}")
        return _decode_record(
            raw,
            path=self.fingerprint.path,
            line_number=entry.line_number,
        )

    def load_record(self, index: int) -> dict[str, Any]:
        """Load one full record by its zero-based index in the nonblank records."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise ResultStoreError("Record index must be an integer")
        try:
            entry = self.entries[index]
        except IndexError as exc:
            raise ResultStoreError(f"Record index {index} is out of range") from exc
        return self.load_record_at_offset(entry.offset)

    @staticmethod
    def _numeric_angle(entry: ResultIndexEntry, name: str) -> float | None:
        angles = entry.lightweight.get("angles")
        if not isinstance(angles, dict):
            return None
        return _optional_float(angles.get(name))

    @staticmethod
    def _numeric_score(entry: ResultIndexEntry, method: str) -> float | None:
        result = entry.lightweight.get(method)
        if not isinstance(result, dict):
            return None
        selected = result.get("selected")
        if not isinstance(selected, dict):
            return None
        return _optional_float(selected.get("final_score"))

    def load_landmark_subset(
        self,
        *,
        stride: int = 10,
        person_identifier: str | None = None,
        peak_angle_names: tuple[str, ...] = DEFAULT_PEAK_ANGLE_NAMES,
        preserve_score_peaks: bool = True,
    ) -> list[dict[str, Any]]:
        """Load a deterministic stride sample plus important peak records.

        The first and last eligible records are always retained.  The earliest
        maximum-absolute observation for each requested angle is retained, as
        is the earliest maximum RULA/REBA score when requested.  Peak anchors
        are added to (rather than replacing) the regular stride, so the return
        count can be slightly larger than ``ceil(n / stride)``.
        """

        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise ResultStoreError("Landmark subset stride must be a positive integer")
        candidates = [
            entry
            for entry in self.entries
            if entry.has_landmarks
            and (person_identifier is None or entry.person_identifier == person_identifier)
        ]
        if not candidates:
            return []

        selected: set[int] = set(range(0, len(candidates), stride))
        selected.update((0, len(candidates) - 1))

        for angle_name in peak_angle_names:
            values = [
                (abs(value), index)
                for index, entry in enumerate(candidates)
                if (value := self._numeric_angle(entry, angle_name)) is not None
            ]
            if values:
                # max() keeps the later tuple on equal values, so negate the
                # index to preserve the earliest supporting record.
                _, negative_index = max((value, -index) for value, index in values)
                selected.add(-negative_index)

        if preserve_score_peaks:
            for method in ("rula", "reba"):
                values = [
                    (score, index)
                    for index, entry in enumerate(candidates)
                    if (score := self._numeric_score(entry, method)) is not None
                ]
                if values:
                    _, negative_index = max((score, -index) for score, index in values)
                    selected.add(-negative_index)

        return [self.load_record_at_offset(candidates[index].offset) for index in sorted(selected)]


@lru_cache(maxsize=8)
def _cached_store(path: str, size_bytes: int, mtime_ns: int) -> JsonlResultStore:
    return JsonlResultStore.build(ResultFileFingerprint(path, size_bytes, mtime_ns))


def open_result_store(path: str | Path) -> JsonlResultStore:
    """Open or reuse a store keyed by resolved path, file size, and mtime."""

    fingerprint = _fingerprint(path)
    return _cached_store(
        fingerprint.path,
        fingerprint.size_bytes,
        fingerprint.mtime_ns,
    )


def clear_result_store_cache() -> None:
    """Discard every process-local result index."""

    _cached_store.cache_clear()


def build_legacy_camera_refs(
    record: dict[str, Any],
    video_metadata: list[dict[str, Any]],
    camera_roles: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build honest camera references for records predating per-view offsets.

    These references support synchronized-by-common-index review only.  Their
    exact per-camera timestamp offsets are unknown and are deliberately stored
    as ``None`` rather than fabricated as zero.
    """

    roles_by_index = {int(index): str(role) for role, index in (camera_roles or {}).items()}
    frame_number = _optional_int(record.get("frame_number"))
    timestamp_sec = _optional_float(record.get("timestamp_sec"))
    references: list[dict[str, Any]] = []
    for camera_index, metadata in enumerate(video_metadata):
        if not isinstance(metadata, dict):
            raise ResultStoreError("Each video metadata entry must be an object")
        references.append(
            {
                "camera_index": camera_index,
                "role": roles_by_index.get(camera_index, f"camera_{camera_index + 1}"),
                "path": metadata.get("path"),
                "source_frame_number": frame_number,
                "timestamp_sec": timestamp_sec,
                "timestamp_source": "legacy_common_record",
                "sync_delta_ms": None,
                "synchronization_status": "exact_offsets_unknown",
            }
        )
    return references
