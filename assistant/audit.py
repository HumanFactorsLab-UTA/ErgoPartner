"""Checking every number in an answer against the evidence it was given.

The prompt tells the model never to state a figure that did not come from a tool
result. Prompts are not guarantees, and this project's whole claim is that a
number can be traced to its source -- so the claim is checked rather than
trusted.

The method is deliberately dull and entirely deterministic. Every number that
enters the conversation is collected: the assessment brief, and every tool result
as it returns. Every number in the finished answer is then extracted and looked
for in that set. What comes back is a count and a list, not a verdict -- an
untraced figure is usually the model doing arithmetic out loud ("9 of 15
repetitions" from two traced numbers), which is reasonable, and occasionally it
is a fabrication, which is not. Distinguishing those is the reviewer's job; the
system's job is to stop the two looking identical on screen.

No second model pass. A verifier that can hallucinate cannot certify anything,
and it would double the latency of every turn to say so.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

#: A number as written in prose: 9, 32.5, 1,400, 48°, 38%, -12.
#:
#: Deliberately excludes anything glued to a letter (``qwen3-vl``, ``P002``) and
#: anything inside a dotted version (``4.0.0-audit``, ``H.264``). Those are names,
#: and flagging them as untraced measurements would bury the one time a real
#: figure has no source.
_NUMBER = re.compile(
    r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)(?!\.?\d)(?![\w])"
)

#: Stripped before extraction: the register tags and citation footers carry frame
#: numbers and timestamps that are validated separately, and auditing them here
#: would double-count every citation as an untraced claim.
_TAG = re.compile(r"`?\[(?:computed|observed|judgment)[^\]]*\]`?", re.IGNORECASE)
_CITE = re.compile(r"<<CITE[^>]*>>")
_CLOCK = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

#: Small counting words that are almost never a measured quantity. Auditing them
#: produces noise ("one point", "the first of two") that buries a real miss.
_ORDINAL_CONTEXT = re.compile(
    r"\b(first|second|third|one|two|three|step|point|item|number)\s*$", re.IGNORECASE
)

#: How many characters either side of a number are kept so a reviewer can see
#: what the untraced figure was claiming.
CONTEXT_CHARS = 46


@dataclass(frozen=True, slots=True)
class Claim:
    """One number as it appears in an answer, and whether the evidence held it."""

    value: float
    written: str
    context: str
    traced: bool
    matched: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "written": self.written,
            "context": self.context,
            "traced": self.traced,
            "matched": self.matched,
        }


@dataclass(slots=True)
class Evidence:
    """Every number the model was shown this turn."""

    values: set[float] = field(default_factory=set)

    def add(self, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return
        number = float(value)
        if math.isfinite(number):
            self.values.add(number)

    def absorb(self, payload: Any, depth: int = 0) -> None:
        """Collect every number anywhere in a tool result."""

        if depth > 8:
            return
        if isinstance(payload, dict):
            for value in payload.values():
                self.absorb(value, depth + 1)
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                self.absorb(value, depth + 1)
        elif isinstance(payload, str):
            self.absorb_text(payload)
        else:
            self.add(payload)

    def absorb_text(self, text: str) -> None:
        """Collect numbers out of prose -- the brief, and any text a tool returns."""

        for match in _NUMBER.finditer(str(text)):
            try:
                self.add(float(match.group(1).replace(",", "")))
            except ValueError:
                continue
        # A brief renders times as 0:28; the model may quote either form.
        for match in _CLOCK.finditer(str(text)):
            parts = match.group(0).split(":")
            try:
                total = 0.0
                for part in parts:
                    total = total * 60 + float(part)
            except ValueError:
                continue
            self.add(total)

    def holds(self, value: float, written: str) -> float | None:
        """The evidence value backing ``value``, or None.

        Rounding is the normal case, not an edge case: the pipeline stores
        31.96 degrees and any sane answer says 32. A figure is accepted when some
        evidence value rounds to it at the precision the model chose to write.
        """

        if value in self.values:
            return value
        decimals = len(written.partition(".")[2])
        candidates = (n for n in self.values if round(n, decimals) == value)
        return min(candidates, key=lambda n: (abs(n - value), n), default=None)


@dataclass(slots=True)
class AuditReport:
    claims: list[Claim] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def traced(self) -> int:
        return sum(1 for claim in self.claims if claim.traced)

    @property
    def untraced(self) -> list[Claim]:
        return [claim for claim in self.claims if not claim.traced]

    def payload(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "traced": self.traced,
            "untraced": [claim.payload() for claim in self.untraced],
            "note": (
                "Numeric matches use the recorded assessment and analysis tool results "
                "shown in this turn. Check the cited frame, method, body region and units "
                "to verify what a matching number represents."
            ),
        }


def audit_answer(text: str, evidence: Evidence) -> AuditReport:
    """Check every number in ``text`` against ``evidence``."""

    report = AuditReport()
    if not text:
        return report

    # Blank the spans whose numbers are validated elsewhere, keeping the offsets
    # intact so the surrounding context still reads correctly.
    stripped = _CITE.sub(lambda m: " " * len(m.group(0)), text)
    stripped = _TAG.sub(lambda m: " " * len(m.group(0)), stripped)
    stripped = _CLOCK.sub(lambda m: " " * len(m.group(0)), stripped)

    seen: set[str] = set()
    for match in _NUMBER.finditer(stripped):
        written = match.group(1)
        try:
            value = float(written.replace(",", ""))
        except ValueError:
            continue

        start = max(0, match.start() - CONTEXT_CHARS)
        before = stripped[start : match.start()]
        if _ORDINAL_CONTEXT.search(before):
            continue

        if written in seen:
            continue
        seen.add(written)

        matched = evidence.holds(value, written)
        context = text[start : min(len(text), match.end() + CONTEXT_CHARS)]
        report.claims.append(
            Claim(
                value=value,
                written=written,
                context=" ".join(context.split()),
                traced=matched is not None,
                matched=matched,
            )
        )
    return report


__all__ = ["AuditReport", "Claim", "Evidence", "audit_answer"]
