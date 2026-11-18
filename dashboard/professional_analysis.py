"""Validity-aware, deterministic summaries for the professional dashboard.

This module deliberately does not calculate RULA, REBA, or RNLE scores.  It
summarizes validated angle observations, official-method posture-band exposure,
time coverage, sustained bouts, and candidate repetitions so the dashboard can
explain the evidence behind the independently calculated assessment results.

Temporal duration uses a left-continuous sample-and-hold convention.  An
observation contributes duration only until the next timestamp and only when
that interval is positive and no longer than ``maximum_gap_sec``.  The last
sample never receives an invented duration.  Repetition detection is segmented
at data gaps and camera-source changes before interpolation within each segment.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, ClassVar

import numpy as np
from scipy.signal import find_peaks

SCHEMA_VERSION = "professional-analysis/1.0"


@dataclass(frozen=True, slots=True)
class MethodProvenance:
    """Authoritative source attached to a calculation or interpretation."""

    identifier: str
    title: str
    version: str | None
    url: str
    scope: str


RULA_PROVENANCE = MethodProvenance(
    identifier="RULA",
    title="RULA: a survey method for the investigation of work-related upper limb disorders",
    version="McAtamney and Corlett, 1993",
    url="https://pubmed.ncbi.nlm.nih.gov/15676903/",
    scope="RULA posture thresholds and method interpretation",
)
REBA_PROVENANCE = MethodProvenance(
    identifier="REBA",
    title="Rapid entire body assessment (REBA)",
    version="Hignett and McAtamney, 2000",
    url="https://pubmed.ncbi.nlm.nih.gov/10711982/",
    scope="REBA posture thresholds and method interpretation",
)
RNLE_PROVENANCE = MethodProvenance(
    identifier="RNLE",
    title="Applications Manual for the Revised NIOSH Lifting Equation",
    version="NIOSH Publication No. 94-110",
    url="https://www.cdc.gov/niosh/publications/numbered/94-110.html",
    scope="RNLE applicability, required inputs, multipliers, RWL, and LI",
)
OBSERVATION_PROVENANCE = MethodProvenance(
    identifier="NIOSH-POSTURE-OBSERVATION",
    title="Observation-based posture assessment: review of current practice and recommendations",
    version="NIOSH 2014-131",
    url="https://stacks.cdc.gov/view/cdc/24085",
    scope="Use and limitations of observational posture measurements",
)
POSE_PROVENANCE = MethodProvenance(
    identifier="MEDIAPIPE-POSE-LANDMARKER",
    title="MediaPipe Pose Landmarker",
    version=None,
    url="https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker",
    scope="Pose landmark confidence and model output semantics",
)
POSE_MODEL_CARD_PROVENANCE = MethodProvenance(
    identifier="BLAZEPOSE-GHUM-3D-MODEL-CARD",
    title="BlazePose GHUM 3D Model Card",
    version=None,
    url="https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf",
    scope="Monocular landmark model limitations and non-metric depth semantics",
)
UNCERTAINTY_PROVENANCE = MethodProvenance(
    identifier="JCGM-100",
    title="Evaluation of measurement data — Guide to the expression of uncertainty in measurement",
    version="JCGM 100:2008",
    url="https://www.bipm.org/en/doi/10.59161/jcgm100-2008e",
    scope="Separation of measurement quality from result validity",
)
CONTROL_PROVENANCE = MethodProvenance(
    identifier="NIOSH-HIERARCHY-OF-CONTROLS",
    title="Hierarchy of Controls",
    version=None,
    url="https://www.cdc.gov/niosh/hierarchy-of-controls/about/index.html",
    scope="Prioritization of engineering and administrative control review",
)

PROVENANCE: tuple[MethodProvenance, ...] = (
    RULA_PROVENANCE,
    REBA_PROVENANCE,
    RNLE_PROVENANCE,
    OBSERVATION_PROVENANCE,
    POSE_PROVENANCE,
    POSE_MODEL_CARD_PROVENANCE,
    UNCERTAINTY_PROVENANCE,
    CONTROL_PROVENANCE,
)

DEFAULT_REPETITION_MEASUREMENTS: tuple[str, ...] = (
    "trunk_flexion",
    "left_upper_arm_flexion",
    "right_upper_arm_flexion",
    "left_forearm_flexion",
    "right_forearm_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
)

# Wrist observations are withheld unless the producer explicitly names a source
# whose hand landmarks can be audited separately from the pose-only hand points.
VALIDATED_WRIST_SOURCES: frozenset[str] = frozenset(
    {
        "hand_landmarker",
        "hand_landmarks",
        "validated_hand_landmarks",
        "manual_measurement",
        "instrumented",
    }
)


@dataclass(frozen=True, slots=True)
class EvidencePoint:
    frame_number: int | None
    timestamp_sec: float | None
    person_identifier: str | None
    measurement: str | None
    value_deg: float | None
    camera_source: str | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class TimeCoverage:
    timestamp_count: int
    span_sec: float | None
    covered_duration_sec: float | None
    gap_count: int | None
    gap_duration_sec: float | None
    longest_gap_sec: float | None
    coverage_percent: float | None
    maximum_gap_sec: float


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    valid_observation_count: int
    excluded_observation_count: int
    minimum_deg: float | None
    maximum_deg: float | None
    mean_deg: float | None
    median_deg: float | None
    standard_deviation_deg: float | None
    percentile_05_deg: float | None
    percentile_95_deg: float | None
    absolute_peak: EvidencePoint | None


@dataclass(frozen=True, slots=True)
class BandExposure:
    method: str
    measurement: str
    band_id: str
    label: str
    is_non_neutral: bool
    observation_count: int
    duration_sec: float | None
    percent_of_covered_duration: float | None
    evidence: EvidencePoint | None
    provenance: MethodProvenance


@dataclass(frozen=True, slots=True)
class SustainedBout:
    method: str
    measurement: str
    criterion: str
    band_ids_observed: tuple[str, ...]
    start_timestamp_sec: float
    end_timestamp_sec: float
    duration_sec: float
    start_frame_number: int | None
    end_frame_number: int | None
    camera_source: str | None
    peak_evidence: EvidencePoint
    provenance: MethodProvenance


@dataclass(frozen=True, slots=True)
class AngleAnalysis:
    measurement: str
    units: str
    status: str
    statistics: DescriptiveStatistics
    time_coverage: TimeCoverage
    exclusion_reasons: tuple[str, ...]
    band_exposure: tuple[BandExposure, ...]
    sustained_bouts: tuple[SustainedBout, ...]


@dataclass(frozen=True, slots=True)
class RepetitionCandidateSummary:
    measurement: str
    status: str
    candidate_only: bool
    algorithm: str
    candidate_count: int | None
    rate_per_minute: float | None
    analysis_duration_sec: float | None
    segments_analyzed: int | None
    prominence_deg: float
    minimum_interval_sec: float
    event_evidence: tuple[EvidencePoint, ...]
    limitations: tuple[str, ...]
    provenance: MethodProvenance


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    status: str
    input_record_count: int
    selected_record_count: int
    usable_angle_observation_count: int | None
    invalid_timestamp_count: int | None
    duplicate_timestamp_count: int | None
    out_of_order_timestamp_count: int | None
    time_coverage: TimeCoverage
    mean_pose_visibility: float | None
    minimum_pose_visibility: float | None
    pose_visibility_below_threshold_count: int | None
    calibration_statuses: tuple[str, ...] | None
    camera_view_statuses: tuple[str, ...] | None
    model_coordinate_types: tuple[str, ...] | None
    timestamp_sources: tuple[str, ...] | None
    camera_count: int | None
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MethodReadiness:
    method: str
    status: str
    assessed_record_count: int | None
    valid_result_count: int | None
    missing_inputs: tuple[str, ...]
    invalid_inputs: tuple[str, ...]
    unsupported_conditions: tuple[str, ...]
    observed_validity_statuses: tuple[str, ...]
    detail: str
    provenance: MethodProvenance


@dataclass(frozen=True, slots=True)
class SourceContribution:
    measurement: str
    camera_source: str
    observation_count: int
    percent_of_valid_observations: float


@dataclass(frozen=True, slots=True)
class CameraAgreement:
    measurement: str
    paired_observation_count: int
    mean_range_deg: float
    median_range_deg: float
    percentile_95_range_deg: float
    maximum_range_deg: float
    maximum_range_evidence: EvidencePoint
    interpretation: str


@dataclass(frozen=True, slots=True)
class CameraSourceSummary:
    contribution_status: str
    contributions: tuple[SourceContribution, ...] | None
    agreement_status: str
    agreement: tuple[CameraAgreement, ...] | None


@dataclass(frozen=True, slots=True)
class Recommendation:
    recommendation_id: str
    priority: str
    title: str
    rationale: str
    observed_fact: str
    evidence: tuple[EvidencePoint, ...]
    related_methods: tuple[str, ...]
    source_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProfessionalAnalysis:
    schema_version: str
    data_quality: DataQualitySummary
    angle_analyses: tuple[AngleAnalysis, ...]
    repetition_candidates: tuple[RepetitionCandidateSummary, ...]
    method_readiness: tuple[MethodReadiness, ...]
    camera_sources: CameraSourceSummary
    recommendations: tuple[Recommendation, ...]
    warnings: tuple[str, ...]
    provenance: tuple[MethodProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping without replacing missing data."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Observation:
    measurement: str
    value: float
    timestamp: float | None
    frame: int | None
    person: str | None
    source: str | None
    record_index: int


@dataclass(frozen=True, slots=True)
class _NormalizedRecord:
    index: int
    raw: Mapping[str, Any]
    person: str | None
    frame: int | None
    timestamp: float | None
    angles: Mapping[str, Any]
    measurements: frozenset[str]
    unavailable: frozenset[str]
    visibility: Mapping[str, float]
    selection: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _BandRule:
    method: str
    measurement_kind: str
    catalog: tuple[tuple[str, str, bool], ...]
    provenance: MethodProvenance


class _BandRules:
    """Official posture partitions only; these functions never combine scores."""

    RULA_UPPER_ARM: ClassVar[_BandRule] = _BandRule(
        "RULA",
        "upper_arm_flexion",
        (
            ("extension_lt_minus_20", "Extension below -20°", True),
            ("minus_20_to_20", "-20° to 20°", False),
            ("flexion_gt_20_to_45", "Flexion >20° to 45°", True),
            ("flexion_gt_45_to_90", "Flexion >45° to 90°", True),
            ("flexion_gt_90", "Flexion >90°", True),
        ),
        RULA_PROVENANCE,
    )
    RULA_LOWER_ARM: ClassVar[_BandRule] = _BandRule(
        "RULA",
        "forearm_flexion",
        (
            ("below_60", "Below 60°", True),
            ("from_60_to_100", "60° to 100°", False),
            ("above_100", "Above 100°", True),
        ),
        RULA_PROVENANCE,
    )
    RULA_WRIST: ClassVar[_BandRule] = _BandRule(
        "RULA",
        "wrist_flexion",
        (
            ("neutral_0", "Neutral (0°)", False),
            ("abs_gt_0_to_15", "Flexion/extension >0° to 15°", True),
            ("abs_gt_15", "Flexion/extension >15°", True),
        ),
        RULA_PROVENANCE,
    )
    RULA_NECK: ClassVar[_BandRule] = _BandRule(
        "RULA",
        "neck_flexion",
        (
            ("extension_lt_0", "Extension below 0°", True),
            ("flexion_0_to_10", "Flexion 0° to 10°", False),
            ("flexion_gt_10_to_20", "Flexion >10° to 20°", True),
            ("flexion_gt_20", "Flexion >20°", True),
        ),
        RULA_PROVENANCE,
    )
    RULA_TRUNK: ClassVar[_BandRule] = _BandRule(
        "RULA",
        "trunk_flexion",
        (
            (
                "extension_unsupported",
                "Extension below 0° (not defined by the original RULA trunk partition)",
                True,
            ),
            ("neutral_0", "Neutral (0°)", False),
            ("flexion_gt_0_to_20", "Flexion >0° to 20°", True),
            ("flexion_gt_20_to_60", "Flexion >20° to 60°", True),
            ("flexion_gt_60", "Flexion >60°", True),
        ),
        RULA_PROVENANCE,
    )
    REBA_UPPER_ARM: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "upper_arm_flexion",
        RULA_UPPER_ARM.catalog,
        REBA_PROVENANCE,
    )
    REBA_LOWER_ARM: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "forearm_flexion",
        RULA_LOWER_ARM.catalog,
        REBA_PROVENANCE,
    )
    REBA_WRIST: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "wrist_flexion",
        (
            ("abs_le_15", "Flexion/extension up to 15°", False),
            ("abs_gt_15", "Flexion/extension >15°", True),
        ),
        REBA_PROVENANCE,
    )
    REBA_NECK: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "neck_flexion",
        (
            ("flexion_0_to_20", "Flexion 0° to 20°", False),
            ("extension_lt_0", "Extension below 0°", True),
            ("flexion_gt_20", "Flexion >20°", True),
        ),
        REBA_PROVENANCE,
    )
    REBA_TRUNK: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "trunk_flexion",
        (
            ("neutral_0", "Neutral (0°)", False),
            ("extension_minus_20_to_lt_0", "Extension -20° to below 0°", True),
            ("flexion_gt_0_to_20", "Flexion >0° to 20°", True),
            ("extension_lt_minus_20", "Extension below -20°", True),
            ("flexion_gt_20_to_60", "Flexion >20° to 60°", True),
            ("flexion_gt_60", "Flexion >60°", True),
        ),
        REBA_PROVENANCE,
    )
    REBA_KNEE: ClassVar[_BandRule] = _BandRule(
        "REBA",
        "knee_flexion",
        (
            ("flexion_0_to_30", "Flexion 0° to 30°", False),
            ("flexion_gt_30_to_60", "Flexion >30° to 60°", True),
            ("flexion_gt_60", "Flexion >60°", True),
        ),
        REBA_PROVENANCE,
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value)


def _canonical_measurement(name: Any) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    return name[:-4] if name.endswith("_deg") else name


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        converted = asdict(value)
        return converted if isinstance(converted, Mapping) else None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return converted if isinstance(converted, Mapping) else None
    return None


def _normalize_record(index: int, raw: Mapping[str, Any]) -> _NormalizedRecord:
    angles = _as_mapping(raw.get("angles")) or {}
    unavailable_raw = angles.get("unavailable_measurements", ())
    if not isinstance(unavailable_raw, Sequence) or isinstance(unavailable_raw, (str, bytes)):
        unavailable_raw = ()
    unavailable = frozenset(
        name for item in unavailable_raw if (name := _canonical_measurement(item)) is not None
    )

    visibility_raw = _as_mapping(angles.get("measurement_visibility")) or {}
    visibility: dict[str, float] = {}
    for key, value in visibility_raw.items():
        name = _canonical_measurement(key)
        number = _finite_number(value)
        if name is not None and number is not None:
            visibility[name] = number

    measurements: set[str] = set(unavailable)
    for key, value in angles.items():
        name = _canonical_measurement(key)
        if (
            name is None
            or name in {"measurement_visibility", "unavailable_measurements"}
            or name.endswith("_data_source")
        ):
            continue
        if value is None or (
            isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        ):
            measurements.add(name)

    selection_raw = _as_mapping(raw.get("multi_view_selection")) or {}
    selection: dict[str, str] = {}
    for key, value in selection_raw.items():
        name = _canonical_measurement(key)
        if name is not None and isinstance(value, str) and value and value != "unavailable":
            selection[name] = value

    person_raw = raw.get("person_identifier")
    person = str(person_raw) if person_raw is not None else None
    return _NormalizedRecord(
        index=index,
        raw=raw,
        person=person,
        frame=_integer(raw.get("frame_number")),
        timestamp=_finite_number(raw.get("timestamp_sec")),
        angles=angles,
        measurements=frozenset(measurements),
        unavailable=unavailable,
        visibility=visibility,
        selection=selection,
    )


def _measurement_value(record: _NormalizedRecord, measurement: str) -> float | None:
    if measurement in record.angles:
        return _finite_number(record.angles.get(measurement))
    return _finite_number(record.angles.get(f"{measurement}_deg"))


def _wrist_exclusion_reason(record: _NormalizedRecord, measurement: str) -> str | None:
    if "wrist" not in measurement:
        return None
    side = (
        "left"
        if measurement.startswith("left_")
        else "right"
        if measurement.startswith("right_")
        else None
    )
    if side is None:
        return "wrist_side_unavailable"
    source_raw = record.angles.get(f"{side}_wrist_data_source")
    if source_raw is None:
        return "wrist_source_unavailable"
    source = str(source_raw).strip().lower()
    if source == "pose_only":
        return "unsupported_wrist_source:pose_only"
    if source not in VALIDATED_WRIST_SOURCES:
        return f"wrist_source_not_validated:{source}"
    return None


def _collect_observations(
    records: Sequence[_NormalizedRecord], measurement: str, minimum_visibility: float
) -> tuple[list[_Observation], Counter[str]]:
    observations: list[_Observation] = []
    exclusions: Counter[str] = Counter()
    for record in records:
        if measurement not in record.measurements:
            continue
        if measurement in record.unavailable:
            exclusions["explicitly_unavailable"] += 1
            continue
        value = _measurement_value(record, measurement)
        if value is None:
            exclusions["missing_or_non_finite_value"] += 1
            continue
        wrist_reason = _wrist_exclusion_reason(record, measurement)
        if wrist_reason is not None:
            exclusions[wrist_reason] += 1
            continue
        visibility = record.visibility.get(measurement)
        if visibility is not None and visibility < minimum_visibility:
            exclusions["below_minimum_measurement_visibility"] += 1
            continue
        observations.append(
            _Observation(
                measurement=measurement,
                value=value,
                timestamp=record.timestamp,
                frame=record.frame,
                person=record.person,
                source=record.selection.get(measurement),
                record_index=record.index,
            )
        )
    observations.sort(
        key=lambda item: (
            item.timestamp is None,
            item.timestamp if item.timestamp is not None else math.inf,
            item.frame if item.frame is not None else math.inf,
            item.record_index,
        )
    )
    return observations, exclusions


def _evidence(observation: _Observation, detail: str | None = None) -> EvidencePoint:
    return EvidencePoint(
        frame_number=observation.frame,
        timestamp_sec=observation.timestamp,
        person_identifier=observation.person,
        measurement=observation.measurement,
        value_deg=observation.value,
        camera_source=observation.source,
        detail=detail,
    )


def _time_coverage(timestamps: Sequence[float], maximum_gap_sec: float) -> TimeCoverage:
    ordered = sorted(timestamps)
    count = len(ordered)
    if count < 2:
        return TimeCoverage(count, None, None, None, None, None, None, maximum_gap_sec)
    span = ordered[-1] - ordered[0]
    if span <= 0:
        return TimeCoverage(count, None, None, None, None, None, None, maximum_gap_sec)
    deltas = [right - left for left, right in zip(ordered, ordered[1:], strict=False)]
    positive = [delta for delta in deltas if delta > 0]
    gaps = [delta for delta in positive if delta > maximum_gap_sec]
    covered = sum(delta for delta in positive if delta <= maximum_gap_sec)
    return TimeCoverage(
        timestamp_count=count,
        span_sec=float(span),
        covered_duration_sec=float(covered),
        gap_count=len(gaps),
        gap_duration_sec=float(sum(gaps)),
        longest_gap_sec=float(max(gaps)) if gaps else None,
        coverage_percent=float(covered / span * 100.0),
        maximum_gap_sec=maximum_gap_sec,
    )


def _statistics(observations: Sequence[_Observation], excluded_count: int) -> DescriptiveStatistics:
    if not observations:
        return DescriptiveStatistics(
            valid_observation_count=0,
            excluded_observation_count=excluded_count,
            minimum_deg=None,
            maximum_deg=None,
            mean_deg=None,
            median_deg=None,
            standard_deviation_deg=None,
            percentile_05_deg=None,
            percentile_95_deg=None,
            absolute_peak=None,
        )
    values = np.asarray([item.value for item in observations], dtype=float)
    peak = max(observations, key=lambda item: (abs(item.value), -item.record_index))
    return DescriptiveStatistics(
        valid_observation_count=len(observations),
        excluded_observation_count=excluded_count,
        minimum_deg=float(np.min(values)),
        maximum_deg=float(np.max(values)),
        mean_deg=float(np.mean(values)),
        median_deg=float(np.median(values)),
        standard_deviation_deg=float(np.std(values, ddof=1)) if len(values) > 1 else None,
        percentile_05_deg=float(np.percentile(values, 5)),
        percentile_95_deg=float(np.percentile(values, 95)),
        absolute_peak=_evidence(peak, "Largest absolute observed angle"),
    )


def _rules_for_measurement(measurement: str) -> tuple[_BandRule, ...]:
    if measurement == "trunk_flexion":
        return (_BandRules.RULA_TRUNK, _BandRules.REBA_TRUNK)
    if measurement == "neck_flexion":
        return (_BandRules.RULA_NECK, _BandRules.REBA_NECK)
    if measurement.endswith("_upper_arm_flexion"):
        return (_BandRules.RULA_UPPER_ARM, _BandRules.REBA_UPPER_ARM)
    if measurement.endswith(("_forearm_flexion", "_lower_arm_flexion")):
        return (_BandRules.RULA_LOWER_ARM, _BandRules.REBA_LOWER_ARM)
    if measurement.endswith("_wrist_flexion"):
        return (_BandRules.RULA_WRIST, _BandRules.REBA_WRIST)
    if measurement.endswith("_knee_flexion"):
        return (_BandRules.REBA_KNEE,)
    return ()


def _classify(rule: _BandRule, value: float) -> str:
    kind = rule.measurement_kind
    method = rule.method
    if kind == "upper_arm_flexion":
        if value < -20:
            return "extension_lt_minus_20"
        if value <= 20:
            return "minus_20_to_20"
        if value <= 45:
            return "flexion_gt_20_to_45"
        if value <= 90:
            return "flexion_gt_45_to_90"
        return "flexion_gt_90"
    if kind == "forearm_flexion":
        if value < 60:
            return "below_60"
        if value <= 100:
            return "from_60_to_100"
        return "above_100"
    if kind == "wrist_flexion":
        absolute = abs(value)
        if method == "RULA":
            if absolute == 0:
                return "neutral_0"
            return "abs_gt_0_to_15" if absolute <= 15 else "abs_gt_15"
        return "abs_le_15" if absolute <= 15 else "abs_gt_15"
    if kind == "neck_flexion":
        if value < 0:
            return "extension_lt_0"
        if method == "RULA":
            if value <= 10:
                return "flexion_0_to_10"
            return "flexion_gt_10_to_20" if value <= 20 else "flexion_gt_20"
        return "flexion_0_to_20" if value <= 20 else "flexion_gt_20"
    if kind == "trunk_flexion":
        if method == "RULA":
            if value < 0:
                return "extension_unsupported"
            if value == 0:
                return "neutral_0"
            if value <= 20:
                return "flexion_gt_0_to_20"
            return "flexion_gt_20_to_60" if value <= 60 else "flexion_gt_60"
        if value == 0:
            return "neutral_0"
        if -20 <= value < 0:
            return "extension_minus_20_to_lt_0"
        if 0 < value <= 20:
            return "flexion_gt_0_to_20"
        if value < -20:
            return "extension_lt_minus_20"
        return "flexion_gt_20_to_60" if value <= 60 else "flexion_gt_60"
    if kind == "knee_flexion":
        if value <= 30:
            return "flexion_0_to_30"
        return "flexion_gt_30_to_60" if value <= 60 else "flexion_gt_60"
    raise ValueError(f"Unsupported band rule: {method} {kind}")


def _band_exposure(
    observations: Sequence[_Observation], rule: _BandRule, maximum_gap_sec: float
) -> tuple[BandExposure, ...]:
    catalog = {band_id: (label, non_neutral) for band_id, label, non_neutral in rule.catalog}
    counts: Counter[str] = Counter()
    durations: defaultdict[str, float] = defaultdict(float)
    evidence: dict[str, _Observation] = {}
    timed = [item for item in observations if item.timestamp is not None]
    for item in observations:
        band_id = _classify(rule, item.value)
        counts[band_id] += 1
        current = evidence.get(band_id)
        if current is None or abs(item.value) > abs(current.value):
            evidence[band_id] = item
    duration_support = False
    for left, right in zip(timed, timed[1:], strict=False):
        delta = float(right.timestamp - left.timestamp)  # type: ignore[operator]
        if 0 < delta <= maximum_gap_sec:
            duration_support = True
            durations[_classify(rule, left.value)] += delta
    covered = sum(durations.values())
    result: list[BandExposure] = []
    for band_id, label, non_neutral in rule.catalog:
        duration = float(durations[band_id]) if duration_support else None
        percent = (
            float(duration / covered * 100.0) if duration is not None and covered > 0 else None
        )
        item = evidence.get(band_id)
        result.append(
            BandExposure(
                method=rule.method,
                measurement=observations[0].measurement,
                band_id=band_id,
                label=label,
                is_non_neutral=non_neutral,
                observation_count=counts[band_id],
                duration_sec=duration,
                percent_of_covered_duration=percent,
                evidence=_evidence(item, f"Observed in {rule.method} posture band")
                if item
                else None,
                provenance=rule.provenance,
            )
        )
    unknown = set(counts) - set(catalog)
    if unknown:
        raise AssertionError(f"Classifier emitted uncatalogued bands: {sorted(unknown)}")
    return tuple(result)


def _same_source(left: _Observation, right: _Observation) -> bool:
    return left.source is None or right.source is None or left.source == right.source


def _sustained_bouts(
    observations: Sequence[_Observation],
    rule: _BandRule,
    maximum_gap_sec: float,
    minimum_duration_sec: float,
) -> tuple[SustainedBout, ...]:
    timed = [item for item in observations if item.timestamp is not None]
    non_neutral = {band_id for band_id, _label, is_non_neutral in rule.catalog if is_non_neutral}
    bouts: list[SustainedBout] = []
    active: list[_Observation] = []
    duration = 0.0
    bands: set[str] = set()

    def flush() -> None:
        nonlocal active, duration, bands
        if active and duration >= minimum_duration_sec:
            peak = max(active, key=lambda item: (abs(item.value), -item.record_index))
            bouts.append(
                SustainedBout(
                    method=rule.method,
                    measurement=peak.measurement,
                    criterion="outside_method_neutral_posture_band",
                    band_ids_observed=tuple(sorted(bands)),
                    start_timestamp_sec=float(active[0].timestamp),  # type: ignore[arg-type]
                    end_timestamp_sec=float(active[-1].timestamp),  # type: ignore[arg-type]
                    duration_sec=float(duration),
                    start_frame_number=active[0].frame,
                    end_frame_number=active[-1].frame,
                    camera_source=active[0].source,
                    peak_evidence=_evidence(peak, "Peak within sustained posture bout"),
                    provenance=rule.provenance,
                )
            )
        active = []
        duration = 0.0
        bands = set()

    for left, right in zip(timed, timed[1:], strict=False):
        delta = float(right.timestamp - left.timestamp)  # type: ignore[operator]
        band_id = _classify(rule, left.value)
        eligible = (
            0 < delta <= maximum_gap_sec and _same_source(left, right) and band_id in non_neutral
        )
        if not eligible:
            flush()
            continue
        if not active:
            active = [left, right]
        elif active[-1] is left:
            active.append(right)
        else:
            flush()
            active = [left, right]
        duration += delta
        bands.add(band_id)
    flush()
    return tuple(bouts)


def _angle_analysis(
    records: Sequence[_NormalizedRecord],
    measurement: str,
    minimum_visibility: float,
    maximum_gap_sec: float,
    sustained_min_duration_sec: float,
) -> tuple[AngleAnalysis, list[_Observation]]:
    observations, exclusions = _collect_observations(records, measurement, minimum_visibility)
    excluded_count = sum(exclusions.values())
    timed = [item.timestamp for item in observations if item.timestamp is not None]
    coverage = _time_coverage(timed, maximum_gap_sec)
    reasons = tuple(f"{reason} ({count})" for reason, count in sorted(exclusions.items()))
    if not observations:
        status = "unavailable"
    elif (
        excluded_count
        or len(timed) < len(observations)
        or coverage.covered_duration_sec in (None, 0)
    ):
        status = "valid_with_limitations"
    else:
        status = "valid"
    exposure: list[BandExposure] = []
    bouts: list[SustainedBout] = []
    if observations:
        for rule in _rules_for_measurement(measurement):
            exposure.extend(_band_exposure(observations, rule, maximum_gap_sec))
            bouts.extend(
                _sustained_bouts(
                    observations,
                    rule,
                    maximum_gap_sec,
                    sustained_min_duration_sec,
                )
            )
    return (
        AngleAnalysis(
            measurement=measurement,
            units="degrees",
            status=status,
            statistics=_statistics(observations, excluded_count),
            time_coverage=coverage,
            exclusion_reasons=reasons,
            band_exposure=tuple(exposure),
            sustained_bouts=tuple(bouts),
        ),
        observations,
    )


def _segment_observations(
    observations: Sequence[_Observation], maximum_gap_sec: float
) -> list[list[_Observation]]:
    timed = [item for item in observations if item.timestamp is not None]
    segments: list[list[_Observation]] = []
    current: list[_Observation] = []
    for item in timed:
        if not current:
            current = [item]
            continue
        previous = current[-1]
        delta = float(item.timestamp - previous.timestamp)  # type: ignore[operator]
        if delta <= 0 or delta > maximum_gap_sec or not _same_source(previous, item):
            segments.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        segments.append(current)
    return segments


def _repetition_summary(
    measurement: str,
    observations: Sequence[_Observation],
    maximum_gap_sec: float,
    prominence_deg: float,
    minimum_interval_sec: float,
) -> RepetitionCandidateSummary:
    base_limitations = (
        "Events are kinematic candidates, not task-validated repetitions.",
        "Signal segments are split at timestamp gaps and camera-source changes.",
        "Only interpolation within a continuous observed segment is used.",
    )
    raw_segments = _segment_observations(observations, maximum_gap_sec)
    analyzable = [segment for segment in raw_segments if len(segment) >= 3]
    events: list[EvidencePoint] = []
    duration = 0.0
    used_segments = 0
    for segment in analyzable:
        times = np.asarray([float(item.timestamp) for item in segment], dtype=float)
        values = np.asarray([item.value for item in segment], dtype=float)
        deltas = np.diff(times)
        if len(deltas) == 0 or np.any(deltas <= 0):
            continue
        step = float(np.median(deltas))
        if not math.isfinite(step) or step <= 0:
            continue
        uniform_times = np.arange(times[0], times[-1] + step * 0.5, step)
        if len(uniform_times) < 3:
            continue
        uniform_values = np.interp(uniform_times, times, values)
        distance = max(1, math.ceil(minimum_interval_sec / step))
        peak_indices, _properties = find_peaks(
            uniform_values,
            prominence=prominence_deg,
            distance=distance,
        )
        duration += float(times[-1] - times[0])
        used_segments += 1
        for peak_index in peak_indices:
            peak_time = float(uniform_times[peak_index])
            nearest = min(
                segment,
                key=lambda item: (abs(float(item.timestamp) - peak_time), item.record_index),
            )
            events.append(_evidence(nearest, "Candidate repetition peak"))
    if used_segments == 0 or duration <= 0:
        return RepetitionCandidateSummary(
            measurement=measurement,
            status="unavailable",
            candidate_only=True,
            algorithm="scipy.signal.find_peaks on gap/source-segmented, uniformly resampled angles",
            candidate_count=None,
            rate_per_minute=None,
            analysis_duration_sec=None,
            segments_analyzed=None,
            prominence_deg=prominence_deg,
            minimum_interval_sec=minimum_interval_sec,
            event_evidence=(),
            limitations=base_limitations
            + ("At least three timed observations are required per segment.",),
            provenance=OBSERVATION_PROVENANCE,
        )
    events.sort(
        key=lambda item: (
            item.timestamp_sec if item.timestamp_sec is not None else math.inf,
            item.frame_number if item.frame_number is not None else math.inf,
        )
    )
    count = len(events)
    return RepetitionCandidateSummary(
        measurement=measurement,
        status="candidate_events_available",
        candidate_only=True,
        algorithm="scipy.signal.find_peaks on gap/source-segmented, uniformly resampled angles",
        candidate_count=count,
        rate_per_minute=float(count / duration * 60.0),
        analysis_duration_sec=float(duration),
        segments_analyzed=used_segments,
        prominence_deg=prominence_deg,
        minimum_interval_sec=minimum_interval_sec,
        event_evidence=tuple(events),
        limitations=base_limitations,
        provenance=OBSERVATION_PROVENANCE,
    )


def _strings(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value else set()
    if not isinstance(value, Sequence):
        return set()
    return {str(item) for item in value if item is not None and str(item)}


def _method_readiness_from_records(
    records: Sequence[_NormalizedRecord], method_key: str, provenance: MethodProvenance
) -> MethodReadiness:
    seen = 0
    valid = 0
    missing: set[str] = set()
    invalid: set[str] = set()
    unsupported: set[str] = set()
    statuses: set[str] = set()
    for record in records:
        result = _as_mapping(record.raw.get(method_key))
        if result is None:
            continue
        seen += 1
        selected = _as_mapping(result.get("selected"))
        if selected is not None and _finite_number(selected.get("final_score")) is not None:
            valid += 1
        for side_name in ("left", "right"):
            side = _as_mapping(result.get(side_name))
            if side is None:
                continue
            missing.update(_strings(side.get("missing_inputs")))
            invalid.update(_strings(side.get("invalid_inputs")))
            unsupported.update(_strings(side.get("unsupported_conditions")))
            status = side.get("validity_status")
            if status is not None:
                statuses.add(str(status))
    if seen == 0:
        return MethodReadiness(
            method=method_key.upper(),
            status="not_evaluated",
            assessed_record_count=None,
            valid_result_count=None,
            missing_inputs=(),
            invalid_inputs=(),
            unsupported_conditions=(),
            observed_validity_statuses=(),
            detail="No independently calculated method result was supplied.",
            provenance=provenance,
        )
    if valid == seen:
        status = "ready"
        detail = "Every supplied frame has an independently calculated valid result."
    elif valid > 0:
        status = "partially_ready"
        detail = "Only a subset of supplied frames has an independently calculated valid result."
    elif invalid:
        status = "invalid_inputs"
        detail = "No result is valid because one or more method inputs are invalid."
    elif unsupported:
        status = "unsupported_condition"
        detail = "No result is valid because the observed task includes unsupported conditions."
    elif missing:
        status = "missing_required_inputs"
        detail = "No result is valid until the listed method-specific inputs are provided."
    elif "insufficient_pose_quality" in statuses:
        status = "insufficient_pose_quality"
        detail = "No result is valid because the existing method calculation withheld poor-quality pose data."
    else:
        status = "unavailable"
        detail = "The supplied method results do not contain a valid final score."
    return MethodReadiness(
        method=method_key.upper(),
        status=status,
        assessed_record_count=seen,
        valid_result_count=valid,
        missing_inputs=tuple(sorted(missing)),
        invalid_inputs=tuple(sorted(invalid)),
        unsupported_conditions=tuple(sorted(unsupported)),
        observed_validity_statuses=tuple(sorted(statuses)),
        detail=detail,
        provenance=provenance,
    )


def _rnle_readiness(result_value: Mapping[str, Any] | None) -> MethodReadiness:
    result = _as_mapping(result_value)
    if result is None or not result:
        return MethodReadiness(
            method="RNLE",
            status="not_evaluated",
            assessed_record_count=None,
            valid_result_count=None,
            missing_inputs=(),
            invalid_inputs=(),
            unsupported_conditions=(),
            observed_validity_statuses=(),
            detail="No independently calculated RNLE result was supplied.",
            provenance=RNLE_PROVENANCE,
        )
    missing = _strings(result.get("missing_inputs"))
    invalid = _strings(result.get("invalid_inputs"))
    unsupported = _strings(result.get("unsupported_conditions"))
    statuses = _strings(result.get("validity_status") or result.get("status"))
    lifting_index = _finite_number(result.get("lifting_index"))
    if lifting_index is None:
        lifting_index = _finite_number(result.get("li"))
    valid = lifting_index is not None
    if valid:
        status = "ready"
        detail = "An independently calculated numeric Lifting Index is available."
    elif invalid:
        status = "invalid_inputs"
        detail = "RNLE inputs are invalid."
    elif unsupported:
        status = "unsupported_condition"
        detail = "The task is outside the stated RNLE applicability conditions."
    elif missing:
        status = "missing_required_inputs"
        detail = "RNLE requires the listed task inputs; they were not estimated silently."
    else:
        status = "unavailable"
        detail = "The supplied RNLE result does not contain a valid Lifting Index."
    return MethodReadiness(
        method="RNLE",
        status=status,
        assessed_record_count=1,
        valid_result_count=1 if valid else 0,
        missing_inputs=tuple(sorted(missing)),
        invalid_inputs=tuple(sorted(invalid)),
        unsupported_conditions=tuple(sorted(unsupported)),
        observed_validity_statuses=tuple(sorted(statuses)),
        detail=detail,
        provenance=RNLE_PROVENANCE,
    )


def _camera_count(record: _NormalizedRecord) -> int | None:
    candidates: list[int] = []
    for key in ("pose_landmarks_normalized_by_camera", "pose_landmarks_world_by_camera"):
        value = _as_mapping(record.raw.get(key))
        if value is not None:
            candidates.append(len(value))
    timestamp_source = record.raw.get("timestamp_source")
    if isinstance(timestamp_source, Sequence) and not isinstance(timestamp_source, (str, bytes)):
        candidates.append(len(timestamp_source))
    input_source = record.raw.get("input_source")
    if isinstance(input_source, Sequence) and not isinstance(input_source, (str, bytes)):
        candidates.append(len(input_source))
    return max(candidates) if candidates else None


def _data_quality(
    input_count: int,
    records: Sequence[_NormalizedRecord],
    analyses: Sequence[AngleAnalysis],
    minimum_visibility: float,
    maximum_gap_sec: float,
) -> DataQualitySummary:
    raw_timestamps = [record.timestamp for record in records]
    valid_timestamps = [value for value in raw_timestamps if value is not None]
    invalid_timestamp_count = len(records) - len(valid_timestamps)
    duplicate_count = len(valid_timestamps) - len(set(valid_timestamps))
    ordered_input = [value for value in raw_timestamps if value is not None]
    out_of_order = sum(
        right < left for left, right in zip(ordered_input, ordered_input[1:], strict=False)
    )
    pose_visibilities: list[float] = []
    calibration: set[str] = set()
    views: set[str] = set()
    coordinates: set[str] = set()
    timestamp_sources: set[str] = set()
    camera_counts: list[int] = []
    for record in records:
        quality = _as_mapping(record.raw.get("pose_quality")) or {}
        visibility = _finite_number(quality.get("mean_keypoint_visibility"))
        if visibility is not None:
            pose_visibilities.append(visibility)
        calibration_raw = record.raw.get("calibration_status")
        if calibration_raw is not None:
            calibration.add(str(calibration_raw))
        view = quality.get("camera_view_status")
        if view is not None:
            views.add(str(view))
        coordinate = quality.get("model_coordinate_type")
        if coordinate is not None:
            coordinates.add(str(coordinate))
        source = record.raw.get("timestamp_source")
        if isinstance(source, str):
            timestamp_sources.add(source)
        elif isinstance(source, Sequence):
            timestamp_sources.update(str(item) for item in source if item is not None)
        count = _camera_count(record)
        if count is not None:
            camera_counts.append(count)
    usable = sum(item.statistics.valid_observation_count for item in analyses)
    warnings: list[str] = []
    if not records:
        warnings.append("No records matched the requested participant.")
    if records and invalid_timestamp_count:
        warnings.append(f"{invalid_timestamp_count} record(s) have no finite timestamp.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate timestamp(s) were found.")
    if out_of_order:
        warnings.append(f"{out_of_order} timestamp transition(s) are out of input order.")
    coverage = _time_coverage(valid_timestamps, maximum_gap_sec)
    if coverage.gap_count:
        warnings.append(
            f"{coverage.gap_count} timestamp gap(s) exceed {maximum_gap_sec:g} seconds; "
            "durations and repetition candidates do not bridge them."
        )
    if records and not pose_visibilities:
        warnings.append(
            "Pose-model visibility is unavailable; ergonomic validity cannot be inferred from it."
        )
    below = sum(value < minimum_visibility for value in pose_visibilities)
    if below:
        warnings.append(
            f"{below} record(s) are below the configured mean pose-visibility threshold."
        )
    if calibration and any(value not in {"calibrated", "available"} for value in calibration):
        warnings.append(
            "At least one record is not calibrated; metric spatial claims must be withheld."
        )
    if any("monocular" in value.lower() for value in coordinates):
        warnings.append(
            "Monocular model-estimated 3-D landmarks are not calibrated true 3-D measurements."
        )
    if usable == 0:
        status = "insufficient_data"
    elif warnings:
        status = "valid_with_limitations"
    else:
        status = "valid"
    return DataQualitySummary(
        status=status,
        input_record_count=input_count,
        selected_record_count=len(records),
        usable_angle_observation_count=usable if records else None,
        invalid_timestamp_count=invalid_timestamp_count if records else None,
        duplicate_timestamp_count=duplicate_count if records else None,
        out_of_order_timestamp_count=out_of_order if records else None,
        time_coverage=coverage,
        mean_pose_visibility=float(np.mean(pose_visibilities)) if pose_visibilities else None,
        minimum_pose_visibility=min(pose_visibilities) if pose_visibilities else None,
        pose_visibility_below_threshold_count=below if pose_visibilities else None,
        calibration_statuses=tuple(sorted(calibration)) if calibration else None,
        camera_view_statuses=tuple(sorted(views)) if views else None,
        model_coordinate_types=tuple(sorted(coordinates)) if coordinates else None,
        timestamp_sources=tuple(sorted(timestamp_sources)) if timestamp_sources else None,
        camera_count=max(camera_counts) if camera_counts else None,
        warnings=tuple(warnings),
    )


def _per_camera_angles(record: _NormalizedRecord) -> Mapping[str, Any] | None:
    for key in ("angles_by_camera", "per_camera_angles"):
        value = _as_mapping(record.raw.get(key))
        if value is not None:
            return value
    return None


def _camera_summary(
    records: Sequence[_NormalizedRecord], observations: Mapping[str, Sequence[_Observation]]
) -> CameraSourceSummary:
    contributions: list[SourceContribution] = []
    for measurement in sorted(observations):
        sources = Counter(
            item.source for item in observations[measurement] if item.source is not None
        )
        total = sum(sources.values())
        for source, count in sorted(sources.items()):
            contributions.append(
                SourceContribution(
                    measurement=measurement,
                    camera_source=str(source),
                    observation_count=count,
                    percent_of_valid_observations=float(count / total * 100.0),
                )
            )

    ranges: defaultdict[str, list[tuple[float, _NormalizedRecord]]] = defaultdict(list)
    for record in records:
        by_camera = _per_camera_angles(record)
        if by_camera is None:
            continue
        values_by_measurement: defaultdict[str, list[float]] = defaultdict(list)
        for camera_values_raw in by_camera.values():
            camera_values = _as_mapping(camera_values_raw)
            if camera_values is None:
                continue
            for raw_name, raw_value in camera_values.items():
                name = _canonical_measurement(raw_name)
                number = _finite_number(raw_value)
                if name is not None and number is not None:
                    values_by_measurement[name].append(number)
        for measurement, values in values_by_measurement.items():
            if len(values) >= 2:
                ranges[measurement].append((max(values) - min(values), record))
    agreement: list[CameraAgreement] = []
    for measurement in sorted(ranges):
        items = ranges[measurement]
        values = np.asarray([item[0] for item in items], dtype=float)
        maximum, record = max(items, key=lambda item: (item[0], -item[1].index))
        agreement.append(
            CameraAgreement(
                measurement=measurement,
                paired_observation_count=len(items),
                mean_range_deg=float(np.mean(values)),
                median_range_deg=float(np.median(values)),
                percentile_95_range_deg=float(np.percentile(values, 95)),
                maximum_range_deg=float(maximum),
                maximum_range_evidence=EvidencePoint(
                    frame_number=record.frame,
                    timestamp_sec=record.timestamp,
                    person_identifier=record.person,
                    measurement=measurement,
                    value_deg=float(maximum),
                    camera_source=None,
                    detail="Maximum simultaneous per-camera angle range",
                ),
                interpretation=(
                    "Descriptive inter-camera range only; no universal agreement threshold is applied."
                ),
            )
        )
    return CameraSourceSummary(
        contribution_status="available" if contributions else "unavailable",
        contributions=tuple(contributions) if contributions else None,
        agreement_status="available" if agreement else "unavailable",
        agreement=tuple(agreement) if agreement else None,
    )


_RECOMMENDATION_TEXT: Mapping[str, tuple[str, str]] = {
    "trunk_flexion": (
        "Review trunk posture and task reach",
        "Evaluate whether engineering changes can bring the work or load closer and at a more suitable height.",
    ),
    "neck_flexion": (
        "Review neck posture and visual target placement",
        "Evaluate work and display height, line of sight, and the need to view the task while flexed.",
    ),
    "upper_arm_flexion": (
        "Review shoulder reach and arm elevation",
        "Evaluate work height, horizontal reach, object placement, and arm support.",
    ),
    "forearm_flexion": (
        "Review elbow working posture",
        "Evaluate reach distance, handle location, and work height to support a more neutral elbow posture.",
    ),
    "wrist_flexion": (
        "Review validated wrist posture observations",
        "Evaluate handle orientation, tool design, object geometry, and hand access.",
    ),
    "knee_flexion": (
        "Review lower-limb posture and working height",
        "Evaluate whether work height, clearance, or material position requires sustained knee flexion.",
    ),
}


def _recommendations(
    analyses: Sequence[AngleAnalysis],
    readiness: Sequence[MethodReadiness],
    quality: DataQualitySummary,
) -> tuple[Recommendation, ...]:
    result: list[Recommendation] = []
    if quality.status != "valid":
        result.append(
            Recommendation(
                recommendation_id="data-quality-review",
                priority="data_required",
                title="Resolve recording and measurement-quality limitations",
                rationale="Interpretation should not exceed the quality, calibration, timing, and camera evidence available.",
                observed_fact=" ".join(quality.warnings)
                or "The supplied records are insufficient.",
                evidence=(),
                related_methods=("RULA", "REBA", "RNLE"),
                source_urls=(
                    UNCERTAINTY_PROVENANCE.url,
                    POSE_MODEL_CARD_PROVENANCE.url,
                ),
            )
        )
    for item in readiness:
        if item.status in {
            "missing_required_inputs",
            "invalid_inputs",
            "unsupported_condition",
            "insufficient_pose_quality",
        }:
            facts = [f"missing: {', '.join(item.missing_inputs)}"] if item.missing_inputs else []
            if item.invalid_inputs:
                facts.append(f"invalid: {', '.join(item.invalid_inputs)}")
            if item.unsupported_conditions:
                facts.append(f"unsupported: {', '.join(item.unsupported_conditions)}")
            result.append(
                Recommendation(
                    recommendation_id=f"complete-{item.method.lower()}-inputs",
                    priority="data_required",
                    title=f"Complete or correct {item.method} inputs before interpretation",
                    rationale=item.detail,
                    observed_fact="; ".join(facts) or item.status,
                    evidence=(EvidencePoint(None, None, None, None, None, None, item.detail),),
                    related_methods=(item.method,),
                    source_urls=(item.provenance.url,),
                )
            )
    emitted_kinds: set[str] = set()
    for analysis in analyses:
        non_neutral = [
            band
            for band in analysis.band_exposure
            if band.is_non_neutral and band.observation_count > 0 and band.evidence is not None
        ]
        if not non_neutral:
            continue
        kind = next((key for key in _RECOMMENDATION_TEXT if key in analysis.measurement), None)
        if kind is None or kind in emitted_kinds:
            continue
        emitted_kinds.add(kind)
        title, action = _RECOMMENDATION_TEXT[kind]
        methods = tuple(sorted({band.method for band in non_neutral}))
        evidence = max(
            non_neutral,
            key=lambda band: (
                band.duration_sec if band.duration_sec is not None else -1.0,
                band.observation_count,
                band.band_id,
            ),
        )
        duration_fact = (
            f" for {evidence.duration_sec:.2f} s" if evidence.duration_sec is not None else ""
        )
        result.append(
            Recommendation(
                recommendation_id=f"review-{kind.replace('_flexion', '')}",
                priority="review",
                title=title,
                rationale=(
                    f"{action} Confirm the task context with the worker before selecting controls."
                ),
                observed_fact=(
                    f"{analysis.measurement} entered the {evidence.method} band "
                    f"'{evidence.label}'{duration_fact}."
                ),
                evidence=(evidence.evidence,) if evidence.evidence is not None else (),
                related_methods=methods,
                source_urls=tuple(
                    sorted({band.provenance.url for band in non_neutral} | {CONTROL_PROVENANCE.url})
                ),
            )
        )
    return tuple(result)


def analyze_professional_records(
    records: Iterable[Mapping[str, Any]],
    *,
    rnle_result: Mapping[str, Any] | None = None,
    person_identifier: str | None = None,
    minimum_visibility: float = 0.5,
    maximum_gap_sec: float = 0.25,
    sustained_min_duration_sec: float = 1.0,
    repetition_measurements: Sequence[str] = DEFAULT_REPETITION_MEASUREMENTS,
    repetition_prominence_deg: float = 10.0,
    repetition_min_interval_sec: float = 0.5,
) -> ProfessionalAnalysis:
    """Create validity-aware dashboard summaries from frame-result records.

    The function is pure with respect to its inputs: it performs no file, model,
    UI, or process I/O.  Missing information remains ``None`` or an explicit
    unavailable/readiness status.  Official-method posture bands are reported
    independently and are never combined into a replacement risk score.
    """

    if not 0 <= minimum_visibility <= 1:
        raise ValueError("minimum_visibility must be between 0 and 1 inclusive")
    if not math.isfinite(maximum_gap_sec) or maximum_gap_sec <= 0:
        raise ValueError("maximum_gap_sec must be finite and greater than zero")
    if not math.isfinite(sustained_min_duration_sec) or sustained_min_duration_sec <= 0:
        raise ValueError("sustained_min_duration_sec must be finite and greater than zero")
    if not math.isfinite(repetition_prominence_deg) or repetition_prominence_deg <= 0:
        raise ValueError("repetition_prominence_deg must be finite and greater than zero")
    if not math.isfinite(repetition_min_interval_sec) or repetition_min_interval_sec <= 0:
        raise ValueError("repetition_min_interval_sec must be finite and greater than zero")

    raw_records = list(records)
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise TypeError("Every record must be a mapping")
    normalized_all = [_normalize_record(index, item) for index, item in enumerate(raw_records)]
    selected = [
        record
        for record in normalized_all
        if person_identifier is None or record.person == person_identifier
    ]
    measurements = sorted(
        {measurement for record in selected for measurement in record.measurements}
    )
    analyses: list[AngleAnalysis] = []
    observation_map: dict[str, list[_Observation]] = {}
    for measurement in measurements:
        analysis, observations = _angle_analysis(
            selected,
            measurement,
            minimum_visibility,
            maximum_gap_sec,
            sustained_min_duration_sec,
        )
        analyses.append(analysis)
        observation_map[measurement] = observations
    repetitions = tuple(
        _repetition_summary(
            measurement,
            observation_map.get(measurement, ()),
            maximum_gap_sec,
            repetition_prominence_deg,
            repetition_min_interval_sec,
        )
        for measurement in dict.fromkeys(repetition_measurements)
    )
    readiness = (
        _method_readiness_from_records(selected, "rula", RULA_PROVENANCE),
        _method_readiness_from_records(selected, "reba", REBA_PROVENANCE),
        _rnle_readiness(rnle_result),
    )
    quality = _data_quality(
        len(raw_records),
        selected,
        analyses,
        minimum_visibility,
        maximum_gap_sec,
    )
    camera_sources = _camera_summary(selected, observation_map)
    warnings = (
        "Official RULA and REBA posture bands are descriptive exposure summaries; no new risk score is synthesized.",
        "RULA, REBA, and RNLE readiness and results remain method-specific and are never combined.",
        "Candidate repetitions require confirmation against task intent and do not establish force, recovery, or repetition risk by themselves.",
        "Pose-model confidence is reported separately from ergonomic-measurement validity.",
    )
    return ProfessionalAnalysis(
        schema_version=SCHEMA_VERSION,
        data_quality=quality,
        angle_analyses=tuple(analyses),
        repetition_candidates=repetitions,
        method_readiness=readiness,
        camera_sources=camera_sources,
        recommendations=_recommendations(analyses, readiness, quality),
        warnings=warnings,
        provenance=PROVENANCE,
    )


__all__ = [
    "DEFAULT_REPETITION_MEASUREMENTS",
    "PROVENANCE",
    "REBA_PROVENANCE",
    "RNLE_PROVENANCE",
    "RULA_PROVENANCE",
    "AngleAnalysis",
    "BandExposure",
    "CameraAgreement",
    "CameraSourceSummary",
    "DataQualitySummary",
    "DescriptiveStatistics",
    "EvidencePoint",
    "MethodReadiness",
    "ProfessionalAnalysis",
    "Recommendation",
    "RepetitionCandidateSummary",
    "SourceContribution",
    "SustainedBout",
    "TimeCoverage",
    "analyze_professional_records",
]
