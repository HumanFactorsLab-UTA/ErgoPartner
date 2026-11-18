"""Per-method completion diagnostics.

Every frame record already states which inputs a method was missing, but that
evidence is only readable one frame at a time. When a method never reaches a
score, the run summary previously reported nothing beyond
``valid_rula_observations: 0``, which does not distinguish "no person was
detected" from "the knee landmark sat just under the visibility gate on every
frame".

This module aggregates the blocking reasons the calculators already produce, so
the summary and report can say which specific input to fix first. It performs no
scoring and applies no thresholds of its own.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from ergonomics.common import BilateralAssessmentResult

# Inputs RULA and REBA derive from the pose model. Anything else on those
# methods is an observed task condition the ergonomist supplies. The two
# categories need different responses: a missing measurement is a pipeline or
# recording defect, whereas a missing task condition is simply not entered yet.
POSE_DERIVED_INPUTS = frozenset(
    {
        "trunk_flexion_deg",
        "neck_flexion_deg",
        "upper_arm_flexion_deg",
        "lower_arm_flexion_deg",
        "wrist_flexion_deg",
        "knee_flexion_deg",
    }
)

POSE_MEASUREMENT = "pose_measurement"
TASK_INPUT = "task_input"


def classify_input(name: str) -> str:
    """Return whether an input comes from the pose model or from observation."""

    return POSE_MEASUREMENT if name in POSE_DERIVED_INPUTS else TASK_INPUT


@dataclass(frozen=True)
class BlockingInput:
    """One named input, and how often it prevented a score."""

    name: str
    category: str
    blocked_observations: int
    share_of_blocked: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodCompletionSummary:
    """Why a method did or did not produce a score across a run."""

    method: str
    total_observations: int
    scored_observations: int
    blocked_observations: int
    completion_rate: float | None
    primary_blocker: str | None
    blocking_inputs: list[BlockingInput] = field(default_factory=list)
    blocking_statuses: dict[str, int] = field(default_factory=dict)

    @property
    def blocking_measurements(self) -> list[BlockingInput]:
        """Blocking inputs the pose pipeline failed to produce."""

        return [item for item in self.blocking_inputs if item.category == POSE_MEASUREMENT]

    @property
    def blocking_task_inputs(self) -> list[BlockingInput]:
        """Blocking inputs still awaiting an ergonomist observation."""

        return [item for item in self.blocking_inputs if item.category == TASK_INPUT]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "total_observations": self.total_observations,
            "scored_observations": self.scored_observations,
            "blocked_observations": self.blocked_observations,
            "completion_rate": self.completion_rate,
            "primary_blocker": self.primary_blocker,
            "blocking_measurement_count": len(self.blocking_measurements),
            "blocking_task_input_count": len(self.blocking_task_inputs),
            "blocking_inputs": [item.to_dict() for item in self.blocking_inputs],
            "blocking_statuses": dict(self.blocking_statuses),
        }

    def describe(self) -> str:
        """One-line explanation suitable for a report or a progress message."""

        if self.total_observations == 0:
            return f"{self.method}: no pose observations were available to assess."
        if self.blocked_observations == 0:
            return (
                f"{self.method}: scored every one of {self.total_observations:,} pose observations."
            )
        if self.scored_observations == 0:
            lead = (
                f"{self.method}: produced no score on any of "
                f"{self.total_observations:,} pose observations."
            )
        else:
            lead = (
                f"{self.method}: scored {self.scored_observations:,} of "
                f"{self.total_observations:,} pose observations "
                f"({self.completion_rate:.0%})."
            )
        # Measurements the pipeline failed to produce are named explicitly,
        # because those are defects rather than pending data entry.
        measurements = self.blocking_measurements
        parts = []
        if measurements:
            names = ", ".join(item.name for item in measurements)
            parts.append(f"unavailable pose measurements: {names}")
        pending = len(self.blocking_task_inputs)
        if pending:
            parts.append(f"{pending} task condition(s) not yet supplied")
        return f"{lead} Blocked by {'; '.join(parts)}." if parts else lead


class MethodCompletionTracker:
    """Accumulate blocking reasons over a run for one assessment method.

    A bilateral result only yields a score when both anatomical sides are
    complete, so an observation counts as blocked whenever ``selected`` is None.
    The blocking inputs for that observation are the union of missing and
    invalid inputs across both sides: any of them would have to be resolved.
    """

    def __init__(self, method: str):
        self.method = method
        self._total = 0
        self._scored = 0
        self._inputs: Counter[str] = Counter()
        self._statuses: Counter[str] = Counter()

    def record(self, result: BilateralAssessmentResult[Any]) -> None:
        """Record one bilateral observation."""

        self._total += 1
        if result.selected is not None:
            self._scored += 1
            return

        blocking: set[str] = set()
        for side in (result.left, result.right):
            blocking.update(getattr(side, "missing_inputs", ()) or ())
            blocking.update(getattr(side, "invalid_inputs", ()) or ())
        self._inputs.update(blocking)

        # Attribute one status per observation: the first side that lacks a
        # score explains the failure, so the counts stay per-observation.
        for side in (result.left, result.right):
            if getattr(side, "final_score", None) is None:
                status = getattr(side, "validity_status", None)
                if status:
                    self._statuses[str(status)] += 1
                break

    def summary(self) -> MethodCompletionSummary:
        """Return the accumulated explanation, ordered by impact."""

        blocked = self._total - self._scored
        # Pose measurements sort first: when many inputs are tied at 100% the
        # actionable one is the measurement that failed, not the alphabetically
        # first task condition awaiting entry.
        ordered = sorted(
            self._inputs.items(),
            key=lambda item: (classify_input(item[0]) != POSE_MEASUREMENT, -item[1], item[0]),
        )
        blocking_inputs = [
            BlockingInput(
                name=name,
                category=classify_input(name),
                blocked_observations=count,
                share_of_blocked=(count / blocked) if blocked else 0.0,
            )
            for name, count in ordered
        ]
        return MethodCompletionSummary(
            method=self.method,
            total_observations=self._total,
            scored_observations=self._scored,
            blocked_observations=blocked,
            completion_rate=(self._scored / self._total) if self._total else None,
            primary_blocker=blocking_inputs[0].name if blocking_inputs else None,
            blocking_inputs=blocking_inputs,
            blocking_statuses=dict(
                sorted(self._statuses.items(), key=lambda item: (-item[1], item[0]))
            ),
        )
