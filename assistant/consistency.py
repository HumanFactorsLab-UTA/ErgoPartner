"""Summarize agreement across repeated readings of samples from one recording."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

#: Agreement at or above this is treated as settled; below the lower bound the
#: reading is contested and the interface says so rather than offering a value.
SETTLED = 0.999
CONTESTED_BELOW = 0.5


def _hashable(value: Any) -> Any:
    """A comparable key for a proposed value, including a numeric range."""

    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return str(value).strip().lower()


def _band(agreement: float) -> str:
    if agreement >= SETTLED:
        return "unanimous"
    if agreement >= 0.66:
        return "majority"
    if agreement >= CONTESTED_BELOW:
        return "split"
    return "contested"


@dataclass(slots=True)
class Agreement:
    """How many passes produced the same reading."""

    passes: int = 0
    agreeing: int = 0
    value: Any = None
    alternatives: list[Any] = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return (self.agreeing / self.passes) if self.passes else 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "agreeing": self.agreeing,
            "fraction": round(self.fraction, 3),
            "band": _band(self.fraction),
            "alternatives": self.alternatives,
        }


def vote(readings: list[Any]) -> Agreement:
    """The modal reading across passes, with how many passes produced it.

    Ties break toward the reading that appeared earliest, so the result does not
    depend on dictionary ordering. A pass that offered nothing still counts in
    the denominator: silence is disagreement about whether there was anything to
    see.
    """

    total = len(readings)
    present = [item for item in readings if item is not None]
    if not present:
        return Agreement(passes=total, agreeing=0, value=None)

    order: dict[Any, int] = {}
    counts: Counter[Any] = Counter()
    for position, item in enumerate(present):
        key = _hashable(item)
        counts[key] += 1
        order.setdefault(key, position)

    winner = min(counts, key=lambda key: (-counts[key], order[key]))
    value = next(item for item in present if _hashable(item) == winner)
    alternatives = [
        next(item for item in present if _hashable(item) == key)
        for key in counts
        if key != winner
    ]
    return Agreement(
        passes=total,
        agreeing=counts[winner],
        value=value,
        alternatives=alternatives,
    )


def merge_proposals(passes: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Combine condition proposals from repeated readings of this recording.

    A condition absent from a pass counts as that pass declining to propose it,
    which is exactly the disagreement worth recording -- a coupling judgement two
    readings out of three did not think worth making is not a finding.
    """

    total = len(passes)
    if total == 0:
        return []

    names: list[str] = []
    for reading in passes:
        for item in reading:
            name = str(item.get("condition", "")).strip()
            if name and name not in names:
                names.append(name)

    merged: list[dict[str, Any]] = []
    for name in names:
        per_pass: list[dict[str, Any] | None] = []
        for reading in passes:
            per_pass.append(
                next((item for item in reading if str(item.get("condition")) == name), None)
            )

        values = [
            (item.get("value", item.get("value_range")) if item else None) for item in per_pass
        ]
        agreement = vote(values)
        if agreement.value is None:
            continue

        # The pass whose reading won, so the rationale quoted with the merged
        # value is the rationale that actually argued for it.
        source: dict[str, Any] = {}
        for reading_item, value in zip(per_pass, values, strict=True):
            if reading_item is None:
                continue
            if _hashable(value) == _hashable(agreement.value):
                source = reading_item
                break

        frames: list[int] = []
        evidence: list[dict[str, Any]] = []
        for proposal, value in zip(per_pass, values, strict=True):
            if proposal is None or _hashable(value) != _hashable(agreement.value):
                continue
            for frame in proposal.get("evidence_frames") or []:
                if isinstance(frame, int) and frame not in frames:
                    frames.append(frame)
            for image in proposal.get("evidence", []):
                if image not in evidence:
                    evidence.append(image)

        entry: dict[str, Any] = {
            "condition": name,
            "confidence": str(source.get("confidence") or "low"),
            "rationale": str(source.get("rationale") or ""),
            "evidence_frames": sorted(frames)[:8],
            "evidence": evidence,
            "agreement": agreement.payload(),
            "contested": agreement.fraction <= 0.5,
        }
        if isinstance(agreement.value, list):
            entry["value_range"] = list(agreement.value)
        else:
            entry["value"] = agreement.value
        merged.append(entry)

    # Most-agreed first: that is the order a reviewer should work down.
    merged.sort(key=lambda item: -float(item["agreement"]["fraction"]))
    return merged


def merge_task_inference(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """The task reading several passes agreed on, with its agreement measured."""

    if not passes:
        return {}
    classes = vote([item.get("activity_class") for item in passes])
    summaries = [str(item.get("summary") or "") for item in passes if item.get("summary")]

    supported = bool(classes.value and classes.value != "unknown" and classes.fraction > 0.5)
    label = str(classes.value).replace("_", " ")
    summary = (f"The sampled activity was classified as {label}."
               if supported else "The sampled views do not establish a single activity.")
    evidence = []
    for reading in passes:
        if reading.get("activity_class") == classes.value:
            for item in reading.get("evidence", []):
                if item not in evidence:
                    evidence.append(item)

    return {
        "summary": summary,
        "activity_class": classes.value if supported else "unknown",
        "evidence": evidence,
        "agreement": classes.payload(),
        "alternative_summaries": [item for item in summaries if item != summary][:3],
    }


#: Stripped when deciding whether two passes named the same thing. A pass that
#: writes "rolling cart" and another that writes "mobile cart" saw one cart, and
#: counting them separately drops a real object for lack of a second sighting.
_MODIFIERS = frozenset(
    """rolling mobile wheeled small large tall short metal wooden plastic cardboard
    corrugated steel upper lower top bottom approx approximately standard""".split()
)


def _head_noun(phrase: str) -> str:
    """A crude but stable key for 'is this the same object'.

    The last substantive word, de-pluralised. "cardboard boxes" and "a box"
    collapse together; "rolling cart" and "mobile cart" collapse together. It
    over-merges occasionally -- a shelf and a bookshelf become one -- which is
    the safer direction here, because the alternative is discarding an object
    that every pass actually saw.
    """

    # A parenthetical is a qualifier, not the thing: without dropping it,
    # "mobile cart (approx 0.8m height)" keys on "height" and never matches the
    # pass that just wrote "cart".
    text = re.sub(r"\([^)]*\)", " ", str(phrase).lower())
    text = text.split(",")[0]
    words = [word for word in re.findall(r"[a-z]+", text) if word]
    words = [word for word in words if word not in _MODIFIERS] or words
    if not words:
        return ""
    head = words[-1]
    if len(head) > 3 and head.endswith("es") and head[-3] in "shxz":
        head = head[:-2]
    elif len(head) > 3 and head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return head


def merge_workstation(passes: list[dict[str, Any]]) -> dict[str, Any]:
    """Objects, tools and surfaces that more than one pass actually saw.

    A single pass naming something no other pass mentions is the classic shape of
    a visual hallucination, so with three or more readings a lone sighting is
    dropped. With fewer, everything is kept and the agreement figures carry the
    caveat instead.

    Sightings are matched on head noun rather than exact string, because passes
    describe the same object differently and exact matching threw away objects
    all three had seen.
    """

    total = len(passes)
    threshold = 2 if total >= 3 else 1
    out: dict[str, Any] = {}
    for field_name in ("objects", "tools", "surfaces"):
        counts: Counter[str] = Counter()
        phrases: dict[str, set[str]] = {}
        for reading in passes:
            seen: set[str] = set()
            for item in reading.get(field_name) or []:
                text = str(item).strip()
                key = _head_noun(text)
                if not key or key in seen:
                    continue
                seen.add(key)
                counts[key] += 1
                phrases.setdefault(key, set()).add(text)
        out[field_name] = [
            (next(iter(phrases[key])) if len(phrases[key]) == 1 else key)
            for key, count in counts.most_common() if count >= threshold
        ]
    return out


__all__ = [
    "CONTESTED_BELOW",
    "SETTLED",
    "Agreement",
    "merge_proposals",
    "merge_task_inference",
    "merge_workstation",
    "vote",
]
