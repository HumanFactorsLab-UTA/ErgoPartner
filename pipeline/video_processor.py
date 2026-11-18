"""Validated video-to-ergonomics orchestration.

The pipeline streams decoded frames, preserves decoder timestamps, tracks every
person in single-camera video, refuses ambiguous multi-camera frames, calculates
both anatomical sides independently, and writes auditable nested results.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from core.angle_calculator import AngleCalculator, BodyAngles
from core.coordinates import CANONICAL_COORDINATE_SYSTEM
from core.hand_landmarker import HandLandmarkerWrapper, HandLandmarks
from core.landmark_reliability import (
    LandmarkReliability,
    assess_landmarks,
)
from core.landmark_reliability import (
    policy_from_config as reliability_policy_from_config,
)
from core.multi_view_fusion import MultiViewFusion, select_landmark_source
from core.pose_estimator import PoseEstimator, PoseLandmarks
from core.pose_temporal import PoseFrame, RecoveryPolicy, recover_landmarks
from core.posture_conditions import (
    RECTIFIED_MEASUREMENTS,
    PostureBaseline,
    apply_condition_smoothing,
    baseline_policy_from_config,
    thresholds_from_config,
)
from core.tracking import CentroidPoseTracker
from ergonomics.adapters import (
    DEFAULT_NEUTRAL_TOLERANCE_DEG,
    build_reba_input,
    build_rula_input,
)
from ergonomics.common import AssessmentContext
from ergonomics.condition_resolution import ConditionResolver, ResolvedConditions
from ergonomics.niosh import NIOSHCalculator, NIOSHInput, NIOSHTaskApplicability
from ergonomics.reba import REBACalculator
from ergonomics.rnle_estimation import LiftSample, RNLEEstimate, RNLEEstimator
from ergonomics.rnle_estimation import (
    policy_from_config as rnle_policy_from_config,
)
from ergonomics.rula import RULACalculator
from paths import PROJECT_ROOT
from pipeline.video_input import DecodedFrame, VideoInputError, VideoReader
from reporting.diagnostics import MethodCompletionTracker
from reporting.writers import ResultWriter, serializable, write_summary
from temporal.aggregation import ScoreObservation, summarize_score_series
from temporal.task_dynamics import (
    PostureSample,
    TaskDynamicsAnalyzer,
    policy_from_config,
)

SOFTWARE_VERSION = "5.1.1"

# Angles a posture condition is thresholded against. These are the ones that get
# a per-subject baseline and a smoothing pass before any condition is tested.
BASELINE_MEASUREMENTS: tuple[str, ...] = (
    *sorted(RECTIFIED_MEASUREMENTS),
    "left_wrist_deviation",
    "right_wrist_deviation",
)

# Every measurement a posture condition can be derived from. Used to describe a
# run in which nothing was measurable, so provenance is still reportable.
_ALL_ANGLE_MEASUREMENTS: tuple[str, ...] = (
    "trunk_twist_deg",
    "trunk_side_bend_deg",
    "neck_rotation_deg",
    "neck_side_bend_deg",
    *(
        f"{side}_{name}"
        for side in ("left", "right")
        for name in (
            "upper_arm_abduction_deg",
            "shoulder_elevation_deg",
            "forearm_midline_cross",
            "wrist_deviation_deg",
            "wrist_pronation_deg",
        )
    ),
)


class ConfigurationError(ValueError):
    pass


@dataclass
class _PendingObservation:
    """One tracked person on one frame, held until the sequence is complete.

    Temporal landmark recovery reads the frames on both sides of a gap, so
    single-camera scoring cannot run inside the decode loop.
    """

    frame_number: int
    timestamp_sec: float
    person_id: str
    pose: PoseLandmarks
    hand: HandLandmarks | None
    timestamp_source: str
    camera_frame_references: list[dict[str, Any]]
    synchronization: dict[str, Any]


@dataclass
class _PendingMultiView:
    """One fused multi-camera observation, held until the sequence is complete.

    Multi-camera frames need no landmark recovery -- an occluded joint in one
    view is taken from another -- but scoring is still deferred, because the
    interval-level task conditions are not known until the last frame is read.
    """

    frame_number: int
    timestamp_sec: float
    angles: BodyAngles
    mean_visibility: float
    hand_confidences: list[float | None]
    world_landmarks: dict[int, np.ndarray]
    #: Which camera supplied world_landmarks, kept as provenance because it can
    #: change from frame to frame as joints move in and out of each view.
    landmark_camera_index: int
    record: dict[str, Any]


class VideoProcessor:
    """Streaming assessment pipeline configured by a validated dictionary."""

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise ConfigurationError("configuration must be a mapping")
        if config.get("schema_version") != 4:
            raise ConfigurationError(
                "configuration schema_version must be 4; pre-v4 configuration is unsafe"
            )
        self.config = config
        self._task_inputs: dict[str, Any] | None = None
        self._task_provenance: dict[str, Any] | None = None
        self._resolver: ConditionResolver | None = None
        self._resolution_sample: ResolvedConditions | None = None
        self._person_provenance: dict[str, dict[str, Any]] = {}
        self._rnle_estimate: RNLEEstimate | None = None
        self._reliability = LandmarkReliability()
        self.angle_calculator = AngleCalculator(
            neck_flexion_method=config.get("quality", {}).get("neck_flexion_method", "legacy_cone"),
            arm_abduction_method=config.get("quality", {}).get(
                "arm_abduction_method", "frontal_projection"
            ),
        )
        self.rula_calculator = RULACalculator()
        self.reba_calculator = REBACalculator()
        self.niosh_calculator = NIOSHCalculator()

    @staticmethod
    def _resolve_project_path(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    @staticmethod
    def _camera_frame_evidence(
        frames: list[DecodedFrame],
        paths: list[str],
        camera_roles: dict[int, str],
        sync_tolerance_sec: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Describe the exact lockstep frames observed by the current decoder.

        This metadata does not imply that frames were resampled, shifted, or
        geometrically aligned. Camera zero remains the master because the
        current pipeline reads one frame from each decoder in lockstep and only
        accepts the set when its observed timestamps satisfy the configured
        gate.
        """

        if not frames or len(frames) != len(paths):
            raise ConfigurationError(
                "camera evidence requires one source path for every decoded frame"
            )

        master_timestamp = float(frames[0].timestamp_sec)
        timestamps = [float(frame.timestamp_sec) for frame in frames]
        timestamp_spread_sec = max(timestamps) - min(timestamps)
        is_multicamera = len(frames) > 1
        status = (
            "not_applicable"
            if not is_multicamera
            else (
                "within_tolerance"
                if timestamp_spread_sec <= sync_tolerance_sec
                else "out_of_tolerance"
            )
        )
        references = [
            {
                "camera_index": camera_index,
                "camera_role": camera_roles.get(camera_index),
                "source_path": paths[camera_index],
                "source_frame_number": int(frame.frame_number),
                "timestamp_sec": float(frame.timestamp_sec),
                "timestamp_source": frame.timestamp_source,
                "sync_delta_from_master_sec": float(frame.timestamp_sec) - master_timestamp,
            }
            for camera_index, frame in enumerate(frames)
        ]
        synchronization = {
            "mode": "lockstep_pts_gated" if is_multicamera else "single_camera",
            "master_camera_index": 0,
            "master_camera_role": camera_roles.get(0),
            "tolerance_sec": float(sync_tolerance_sec),
            "timestamp_spread_sec": float(timestamp_spread_sec),
            "status": status,
            "alignment_performed": False,
        }
        return references, synchronization

    def _create_pose_estimators(self, camera_count: int) -> list[PoseEstimator]:
        pose = self.config.get("pose", {})
        model_path = self._resolve_project_path(
            pose.get("model_path", "models/pose_landmarker_heavy.task")
        )
        return [
            PoseEstimator(
                model_path=model_path,
                model_variant=pose.get("model_variant", "heavy"),
                min_detection_confidence=float(pose.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(pose.get("min_tracking_confidence", 0.5)),
                use_gpu=bool(pose.get("use_gpu", False)),
                max_poses=int(pose.get("max_poses", 4)),
                expected_model_sha256=pose.get("model_sha256"),
                allow_cpu_fallback=bool(pose.get("allow_cpu_fallback", False)),
            )
            for _ in range(camera_count)
        ]

    @staticmethod
    def _detect_all(
        estimators: list[PoseEstimator],
        frames: list[DecodedFrame],
        pool: ThreadPoolExecutor | None,
    ) -> list[list[PoseLandmarks]]:
        """Run each camera's pose inference, concurrently when worthwhile.

        MediaPipe releases the GIL inside detect_for_video, so a thread per
        camera gives real parallelism. Results are unchanged: every camera keeps
        its own landmarker and still sees its own frames in order, and all
        cameras are joined before the next frame set is read, so no landmarker
        is ever entered twice at once.
        """

        if pool is None or len(estimators) < 2:
            return [
                estimator.process_frame_all(
                    frame.image_bgr, frame.timestamp_sec, frame.frame_number
                )
                for estimator, frame in zip(estimators, frames, strict=True)
            ]
        futures = [
            pool.submit(
                estimator.process_frame_all,
                frame.image_bgr,
                frame.timestamp_sec,
                frame.frame_number,
            )
            for estimator, frame in zip(estimators, frames, strict=True)
        ]
        return [future.result() for future in futures]

    def _create_hand_landmarkers(self, camera_count: int) -> list[HandLandmarkerWrapper]:
        hand = self.config.get("hand", {})
        if not hand.get("enabled", False):
            return []
        model_path = self._resolve_project_path(
            hand.get("model_path", "models/hand_landmarker.task")
        )
        return [
            HandLandmarkerWrapper(
                model_path=model_path,
                model_variant=hand.get("model_variant", "full"),
                min_detection_confidence=float(hand.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(hand.get("min_tracking_confidence", 0.5)),
                use_gpu=bool(hand.get("use_gpu", False)),
                expected_model_sha256=hand.get("model_sha256"),
            )
            for _ in range(camera_count)
        ]

    @staticmethod
    def _hand_format(hand: HandLandmarks | None) -> dict[str, dict[int, np.ndarray]] | None:
        if hand is None:
            return None
        result = {}
        for side, values in (("left", hand.left_hand), ("right", hand.right_hand)):
            if values:
                result[side] = {
                    index: np.asarray(point[:3], dtype=float) for index, point in values.items()
                }
        return result or None

    @staticmethod
    def _joints(pose: PoseLandmarks) -> dict[int, np.ndarray]:
        return {index: np.asarray(point, dtype=float) for index, point in pose.world.items()}

    @staticmethod
    def _normalized_landmarks(pose: PoseLandmarks) -> dict[str, list[float]]:
        """Return JSON-safe image-normalized landmarks for traceable overlays."""
        return {
            str(index): [float(value) for value in np.asarray(point, dtype=float)[:4]]
            for index, point in pose.normalized.items()
        }

    @staticmethod
    def _world_landmarks(pose: PoseLandmarks) -> dict[str, list[float]]:
        """Return canonical model-world landmarks for 3-D posture playback."""
        return {
            str(index): [float(value) for value in np.asarray(point, dtype=float)[:4]]
            for index, point in pose.world.items()
        }

    def _estimate_rnle_inputs(self, samples: list[LiftSample]) -> RNLEEstimate | None:
        """Derive the RNLE task geometry from the lifting sequence.

        Returns None when estimation is switched off. The estimate never
        overwrites a measured value: :meth:`_compute_rnle` uses it only where the
        configuration left a field empty.
        """

        config = self.config.get("rnle", {})
        if not config.get("estimate_from_video", True):
            return None
        try:
            policy = rnle_policy_from_config(self.config.get("rnle_estimation"))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        height = config.get("participant_height_cm")
        return RNLEEstimator(policy).estimate(
            samples,
            participant_height_cm=float(height) if height else None,
            duration=config.get("duration"),
            coupling=config.get("coupling"),
        )

    def _compute_rnle(self, input_source: str, calibration_status: str):
        config = self.config.get("rnle", {})
        estimate = self._rnle_estimate
        # An entered measurement always wins; the estimate only fills a gap.
        supplied: dict[str, Any] = {
            name: config.get(name)
            for name in (
                "horizontal_cm",
                "vertical_cm",
                "travel_distance_cm",
                "asymmetry_deg",
                "frequency_lifts_per_min",
                "duration",
                "coupling",
            )
        }
        distance_source = config.get("distance_measurement_source", "manual_measurement")
        if estimate is not None:
            estimated = estimate.as_input_fields()
            filled = [
                name
                for name, value in supplied.items()
                if value is None and estimated.get(name) is not None
            ]
            for name in filled:
                supplied[name] = estimated[name]
            # Only claim an anthropometric source if a distance actually came
            # from one; an entered set of measurements stays labelled as entered.
            if {"horizontal_cm", "vertical_cm", "travel_distance_cm"}.intersection(filled):
                distance_source = "anthropometric_estimate"

        applicability_values = config.get("applicability")
        applicability = None
        if applicability_values is not None:
            if not isinstance(applicability_values, dict):
                raise ConfigurationError("rnle.applicability must be a mapping or null")
            required = {field.name for field in fields(NIOSHTaskApplicability)}
            omitted = sorted(required - set(applicability_values))
            if omitted:
                raise ConfigurationError(
                    "rnle.applicability review omitted fields: " + ", ".join(omitted)
                )
            try:
                applicability = NIOSHTaskApplicability(**applicability_values)
            except TypeError as exc:
                raise ConfigurationError(f"invalid rnle.applicability field: {exc}") from exc
        context = AssessmentContext(
            input_source=input_source,
            calibration_status=calibration_status,
        )
        return self.niosh_calculator.calculate(
            NIOSHInput(
                horizontal_cm=supplied["horizontal_cm"],
                vertical_cm=supplied["vertical_cm"],
                travel_distance_cm=supplied["travel_distance_cm"],
                asymmetry_deg=supplied["asymmetry_deg"],
                frequency_lifts_per_min=supplied["frequency_lifts_per_min"],
                duration=supplied["duration"],
                coupling=supplied["coupling"],
                load_weight_kg=config.get("load_weight_kg"),
                applicability=applicability,
                location="origin",
                distance_measurement_source=distance_source,
                calibration_status=calibration_status,
                context=context,
            )
        )

    def _score_angles(
        self,
        angles: BodyAngles,
        *,
        frame_number: int,
        timestamp_sec: float,
        person_identifier: str,
        input_source: str,
        calibration_status: str,
        mean_visibility: float,
        hand_detection_confidence: float | list[float | None] | None = None,
    ):
        resolution = self._condition_resolver().resolve(angles)
        self._resolution_sample = resolution
        task = resolution.values
        quality = self.config.get("quality", {})
        threshold = float(quality.get("minimum_landmark_visibility", 0.5))
        allow_pose_only_wrist = bool(quality.get("allow_pose_only_wrist", False))
        neutral_tolerance = float(
            quality.get("neutral_tolerance_deg", DEFAULT_NEUTRAL_TOLERANCE_DEG)
        )
        if not np.isfinite(neutral_tolerance) or neutral_tolerance < 0:
            raise ConfigurationError("quality.neutral_tolerance_deg must be a non-negative number")
        video_config = self.config.get("video", {})
        camera_view_status = quality.get("camera_view_status", "unknown")
        camera_tilt = self.config.get("calibration", {}).get("camera_tilt_deg")
        if camera_tilt is not None:
            try:
                camera_tilt = float(camera_tilt)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "calibration.camera_tilt_deg must be numeric or null"
                ) from exc
            if not np.isfinite(camera_tilt):
                raise ConfigurationError("calibration.camera_tilt_deg must be finite")
            if camera_tilt != 0:
                camera_view_status = "unsupported"
        context = AssessmentContext(
            frame_number=frame_number,
            timestamp_sec=timestamp_sec,
            person_identifier=person_identifier,
            input_source=input_source,
            calibration_status=calibration_status,
            pose_quality={
                "mean_keypoint_visibility": mean_visibility,
                "hand_detection_confidence": hand_detection_confidence,
                "model_coordinate_type": "monocular_model_estimated_3d",
                "neck_flexion_method": self.angle_calculator.neck_flexion_method,
                "arm_abduction_method": self.angle_calculator.arm_abduction_method,
                "camera_view_status": camera_view_status,
                "camera_tilt_deg": camera_tilt,
                "mirrored": video_config.get("mirrored"),
            },
        )
        left_rula = build_rula_input(
            angles,
            "left",
            task,
            context,
            visibility_threshold=threshold,
            allow_pose_only_wrist=allow_pose_only_wrist,
            neutral_tolerance_deg=neutral_tolerance,
            admitted_landmarks=self._reliability.admitted,
        )
        right_rula = build_rula_input(
            angles,
            "right",
            task,
            context,
            visibility_threshold=threshold,
            allow_pose_only_wrist=allow_pose_only_wrist,
            neutral_tolerance_deg=neutral_tolerance,
            admitted_landmarks=self._reliability.admitted,
        )
        left_reba = build_reba_input(
            angles,
            "left",
            task,
            context,
            visibility_threshold=threshold,
            allow_pose_only_wrist=allow_pose_only_wrist,
            neutral_tolerance_deg=neutral_tolerance,
            admitted_landmarks=self._reliability.admitted,
        )
        right_reba = build_reba_input(
            angles,
            "right",
            task,
            context,
            visibility_threshold=threshold,
            allow_pose_only_wrist=allow_pose_only_wrist,
            neutral_tolerance_deg=neutral_tolerance,
            admitted_landmarks=self._reliability.admitted,
        )
        return (
            self.rula_calculator.calculate_bilateral(left_rula, right_rula),
            self.reba_calculator.calculate_bilateral(left_reba, right_reba),
            context,
            resolution,
        )

    def _condition_resolver(self) -> ConditionResolver:
        """The resolver that merges entered, measured, and assumed conditions.

        Cached because it is consulted for every scored observation and because
        the entered and assumed layers must be identical across the whole run.
        """

        if self._resolver is None:
            task = self.config.get("task_inputs", {})
            if not isinstance(task, dict):
                raise ConfigurationError("task_inputs must be a mapping")
            profile_name = task.get("assumption_profile")
            if profile_name is not None and not isinstance(profile_name, str):
                raise ConfigurationError("task_inputs.assumption_profile must be a name or null")
            try:
                thresholds = thresholds_from_config(self.config.get("posture_conditions"))
                self._resolver = ConditionResolver(task, profile_name, thresholds)
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        return self._resolver

    def _smooth_condition_angles(self, sequence: list[BodyAngles]) -> None:
        """Stabilise the angles a yes/no condition is thresholded against.

        Landmark noise arrives as frame-to-frame spikes of 10-30 degrees, which
        is enough to flip a condition on and off across a trial and to make
        frame-to-frame angular speed read as rapid movement. A short median
        filter removes the spikes without shifting a real posture change.
        """

        try:
            policy = baseline_policy_from_config(self.config.get("posture_conditions"))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        apply_condition_smoothing(sequence, BASELINE_MEASUREMENTS, policy.smoothing_frames)

    def _analyze_dynamics(self, samples: list[PostureSample]) -> None:
        """Derive everything that needs the whole sequence rather than one frame.

        That is the subject's posture baseline, the interval-level RULA/REBA
        conditions, and the RNLE lifting geometry. All three read the same
        decoded sequence, so they are derived in one place once the last frame
        is known.
        """

        try:
            policy = policy_from_config(self.config.get("task_dynamics"))
            baseline_policy = baseline_policy_from_config(self.config.get("posture_conditions"))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc

        # The subject's own resting level for each condition-driving angle.
        # Without it the rectified measurements are tested against a threshold
        # that sits below their own noise floor, and the conditions fire almost
        # every frame. See core/posture_conditions.PostureBaseline.
        series: dict[str, list[float]] = {}
        for sample in samples:
            for name in BASELINE_MEASUREMENTS:
                value = sample.angles.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    series.setdefault(name, []).append(float(value))
        try:
            baseline = PostureBaseline.for_recording(series, baseline_policy, self.config.get("posture_conditions"))
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        self._condition_resolver().set_baseline(baseline)

        # Which landmarks the sequence shows were genuinely tracked. The
        # per-frame confidence gate cannot see presence or smoothness, and on a
        # single-camera trial it withheld every REBA observation over a knee that
        # was visible throughout. See core/landmark_reliability.py.
        try:
            reliability_policy = reliability_policy_from_config(
                (self.config.get("quality") or {}).get("sustained_tracking")
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        self._reliability = assess_landmarks(
            [sample.landmarks for sample in samples], reliability_policy
        )

        dynamics = TaskDynamicsAnalyzer(policy).analyze(samples)
        self._condition_resolver().set_dynamics(dynamics)

        self._rnle_estimate = self._estimate_rnle_inputs(
            [
                LiftSample(
                    timestamp_sec=item.timestamp_sec,
                    frame_number=item.frame_number,
                    landmarks=item.landmarks,
                )
                for item in samples
            ]
        )

    def _task_input_provenance(self) -> dict[str, Any]:
        """Run-level provenance for every task condition.

        Built from the last resolution performed, because which conditions the
        footage could supply is a property of the run rather than of the config.
        """

        if self._person_provenance:
            summaries = list(self._person_provenance.values())
            return {
                "scope": "Union of condition fields on the last observation of each tracked person",
                "by_person": self._person_provenance,
                "profile": summaries[0]["profile"],
                **{
                    field: sorted({name for person in summaries for name in person[field]})
                    for field in ("derived_fields", "observed_fields", "assumed_fields", "unknown_fields",
                                  "confirm_first", "not_observable_from_video")
                },
            }
        resolver = self._condition_resolver()
        sample = self._resolution_sample
        if sample is None:
            # No frame was scored, so nothing was measured; report the entered
            # and assumed layers against a neutral, fully-unavailable frame.
            sample = resolver.resolve(
                BodyAngles(unavailable_measurements=list(_ALL_ANGLE_MEASUREMENTS))
            )
        return resolver.summary(sample)

    def _recovery_policy(self) -> RecoveryPolicy | None:
        """Build the temporal recovery policy, or None when it is disabled."""

        config = self.config.get("pose_recovery", {})
        if not isinstance(config, dict):
            raise ConfigurationError("pose_recovery must be a mapping")
        if not config.get("enabled", True):
            return None
        try:
            return RecoveryPolicy(
                anchor_visibility=float(config.get("anchor_visibility", 0.60)),
                missing_visibility=float(config.get("missing_visibility", 0.50)),
                max_gap_sec=float(config.get("max_gap_sec", 0.40)),
                max_segment_strain=float(config.get("max_segment_strain", 0.20)),
                max_speed_m_per_s=float(config.get("max_speed_m_per_s", 6.0)),
                confidence_decay_per_sec=float(config.get("confidence_decay_per_sec", 0.5)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid pose_recovery configuration: {exc}") from exc

    def _emit_multi_view(
        self,
        pending: list[_PendingMultiView],
        *,
        writer: ResultWriter,
        input_source: str,
        calibration_status: str,
        rula_observations: list[ScoreObservation],
        reba_observations: list[ScoreObservation],
        rula_completion: MethodCompletionTracker,
        reba_completion: MethodCompletionTracker,
    ) -> None:
        """Derive the interval conditions, then score and write every frame."""

        if not pending:
            return

        self._smooth_condition_angles([item.angles for item in pending])
        self._analyze_dynamics(
            [
                PostureSample(
                    timestamp_sec=item.timestamp_sec,
                    frame_number=item.frame_number,
                    angles=item.angles.to_dict(),
                    landmarks=item.world_landmarks,
                )
                for item in pending
            ]
        )

        for item in pending:
            rula, reba, context, resolution = self._score_angles(
                item.angles,
                frame_number=item.frame_number,
                timestamp_sec=item.timestamp_sec,
                person_identifier="person-0001",
                input_source=input_source,
                calibration_status=calibration_status,
                mean_visibility=item.mean_visibility,
                hand_detection_confidence=item.hand_confidences,
            )
            record = dict(item.record)
            # The fused angles were captured before condition smoothing.
            # Preserve those readings and export the values used for scoring.
            record["angles_before_condition_smoothing"] = record["angles"]
            record["angles"] = item.angles.to_dict()
            record["pose_quality"] = context.pose_quality
            record["task_conditions"] = resolution.to_dict()
            record["sequence_landmark_camera_index"] = item.landmark_camera_index
            record["rula"] = rula
            record["reba"] = reba
            writer.write_frame(record)
            rula_completion.record(rula)
            reba_completion.record(reba)
            if rula.selected is not None:
                rula_observations.append(
                    ScoreObservation(
                        item.timestamp_sec,
                        rula.selected.final_score,
                        item.frame_number,
                        "person-0001",
                    )
                )
            if reba.selected is not None:
                reba_observations.append(
                    ScoreObservation(
                        item.timestamp_sec,
                        reba.selected.final_score,
                        item.frame_number,
                        "person-0001",
                    )
                )

    def _emit_pending(
        self,
        pending: list[_PendingObservation],
        *,
        writer: ResultWriter,
        input_source: str,
        calibration_status: str,
        rula_observations: list[ScoreObservation],
        reba_observations: list[ScoreObservation],
        rula_completion: MethodCompletionTracker,
        reba_completion: MethodCompletionTracker,
    ) -> dict[str, Any]:
        """Recover short landmark gaps, then score and write every observation.

        Recovery, smoothing, dynamics, and landmark reliability run independently
        for each tracked person. The writer retains the original frame order.
        """

        policy = self._recovery_policy()
        if not pending:
            return {
                "status": "no_observations",
                "policy_enabled": policy is not None,
            }

        recovered_by_frame: dict[tuple[str, int], list[int]] = {}
        report_payload: dict[str, Any] = {
            "status": "disabled" if policy is None else "applied",
            "policy_enabled": policy is not None,
        }
        tracks: dict[str, list[_PendingObservation]] = {}
        for item in pending:
            tracks.setdefault(item.person_id, []).append(item)
        condition_config = self.config.get("posture_conditions")
        reference = condition_config.get("neutral_reference") if isinstance(condition_config, dict) else None
        if len(tracks) > 1 and isinstance(reference, dict) and reference.get("source") == "reviewed_neutral":
            raise ConfigurationError("A reviewed neutral calibration requires a single tracked person")

        if policy is not None:
            totals = {
                "recovered_landmark_count": 0,
                "recovered_frame_count": 0,
                "rejected_segment_strain": 0,
                "rejected_speed": 0,
                "rejected_gap_too_long": 0,
                "rejected_unbracketed": 0,
            }
            for person_id, track in tracks.items():
                sequence = [
                    PoseFrame(
                        frame_number=item.frame_number,
                        timestamp_sec=item.timestamp_sec,
                        landmarks={
                            index: np.asarray(point, dtype=float)
                            for index, point in item.pose.world.items()
                        },
                    )
                    for item in track
                ]
                repaired, report = recover_landmarks(sequence, policy)
                by_frame = {frame.frame_number: frame.landmarks for frame in repaired}
                for item in track:
                    if item.frame_number in by_frame:
                        item.pose.world.update(by_frame[item.frame_number])
                for frame_number, landmark_ids in report.by_frame().items():
                    recovered_by_frame[(person_id, frame_number)] = landmark_ids
                payload = report.to_dict()
                for key in totals:
                    totals[key] += int(payload.get(key, 0))
            report_payload.update(totals)
            report_payload["method"] = "temporal_linear_interpolation"
            report_payload["policy"] = {
                "anchor_visibility": policy.anchor_visibility,
                "missing_visibility": policy.missing_visibility,
                "max_gap_sec": policy.max_gap_sec,
                "max_segment_strain": policy.max_segment_strain,
                "max_speed_m_per_s": policy.max_speed_m_per_s,
            }

        # Angles first, for every frame. The interval-level task conditions --
        # how long a posture was held, how often the action repeated, whether
        # the subject was seated -- are properties of the whole sequence, so
        # they have to be derived before any frame can be scored against them.
        # A separate scoring context holds the resolver, baseline and reliability
        # for each person. These processors perform no video decoding or inference.
        scorers = {
            person: self if len(tracks) == 1 else VideoProcessor(self.config) for person in tracks
        }
        measurements: dict[tuple[str, int], BodyAngles] = {}
        for person, track in tracks.items():
            scorer = scorers[person]
            angles = [
                scorer.angle_calculator.calculate_all_angles(
                    self._joints(item.pose),
                    hand_landmarks=self._hand_format(item.hand),
                )
                for item in track
            ]
            scorer._smooth_condition_angles(angles)
            samples = [
                PostureSample(
                    timestamp_sec=item.timestamp_sec,
                    frame_number=item.frame_number,
                    angles=measurement.to_dict(),
                    landmarks=item.pose.world,
                )
                for item, measurement in zip(track, angles, strict=True)
            ]
            scorer._analyze_dynamics(samples)
            measurements.update(((person, item.frame_number), measurement)
                                for item, measurement in zip(track, angles, strict=True))

        for item in pending:
            angles_at_frame = measurements[(item.person_id, item.frame_number)]
            recovered_ids = recovered_by_frame.get((item.person_id, item.frame_number), [])
            rula, reba, context, resolution = scorers[item.person_id]._score_angles(
                angles_at_frame,
                frame_number=item.frame_number,
                timestamp_sec=item.timestamp_sec,
                person_identifier=item.person_id,
                input_source=input_source,
                calibration_status=calibration_status,
                mean_visibility=item.pose.mean_keypoint_visibility,
                hand_detection_confidence=(
                    item.hand.detection_confidence if item.hand is not None else None
                ),
            )
            record = {
                "software_version": SOFTWARE_VERSION,
                "frame_number": item.frame_number,
                "timestamp_sec": item.timestamp_sec,
                "person_identifier": item.person_id,
                "input_source": input_source,
                "calibration_status": calibration_status,
                "timestamp_source": item.timestamp_source,
                "camera_frame_references": item.camera_frame_references,
                "synchronization": item.synchronization,
                "pose_quality": context.pose_quality,
                "pose_landmarks_normalized": self._normalized_landmarks(item.pose),
                "pose_landmarks_world": self._world_landmarks(item.pose),
                "pose_recovery": {
                    "recovered_landmark_ids": recovered_ids,
                    "contains_recovered_landmarks": bool(recovered_ids),
                },
                "angles": angles_at_frame.to_dict(),
                "angles_by_camera": {"0": angles_at_frame.to_dict()},
                "angle_units": "degrees",
                "coordinate_convention": CANONICAL_COORDINATE_SYSTEM,
                "task_conditions": resolution.to_dict(),
                "rula": rula,
                "reba": reba,
            }
            writer.write_frame(record)
            rula_completion.record(rula)
            reba_completion.record(reba)
            if rula.selected is not None:
                rula_observations.append(
                    ScoreObservation(
                        item.timestamp_sec,
                        rula.selected.final_score,
                        item.frame_number,
                        item.person_id,
                    )
                )
            if reba.selected is not None:
                reba_observations.append(
                    ScoreObservation(
                        item.timestamp_sec,
                        reba.selected.final_score,
                        item.frame_number,
                        item.person_id,
                    )
                )
        if len(scorers) > 1:
            self._person_provenance = {person: scorer._task_input_provenance() for person, scorer in scorers.items()}
            self._rnle_estimate = None
        return report_payload

    def process(
        self,
        video_paths: list[str],
        output_dir: str,
        progress_callback=None,
    ) -> str:
        """Process video(s) and return the absolute result directory."""

        if not video_paths:
            raise VideoInputError("at least one video path is required")
        # The same processor may be used for a second recording, even one with
        # no detections. Clear estimates and provenance belonging to the first.
        self._person_provenance = {}
        self._resolver = None
        self._resolution_sample = None
        self._rnle_estimate = None
        self._reliability = LandmarkReliability()
        paths = [str(Path(path).resolve()) for path in video_paths]
        input_source = " | ".join(paths)
        output = Path(output_dir).resolve()
        current_output_names = {
            "ASSESSMENT_REPORT.md",
            "frame_results.csv",
            "frame_results.jsonl",
            "run_config.json",
            "video_summary.json",
            "run_metadata.json",
        }
        if output.is_dir():
            incompatible = sorted(
                entry.name for entry in output.iterdir() if entry.name not in current_output_names
            )
            if incompatible:
                raise ConfigurationError(
                    "output directory contains non-v4 or unrelated entries; choose a clean directory: "
                    + ", ".join(incompatible)
                )
        output.mkdir(parents=True, exist_ok=True)
        video_config = self.config.get("video", {})
        fps_override = video_config.get("fps_override")
        target_fps = video_config.get("target_fps")
        if target_fps is not None and float(target_fps) <= 0:
            raise ConfigurationError("video.target_fps must be positive or null")
        sample_interval = 1.0 / float(target_fps) if target_fps is not None else None
        start_frame = video_config.get("start_frame")
        end_frame = video_config.get("end_frame")
        for name, value in (("start_frame", start_frame), ("end_frame", end_frame)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ConfigurationError(f"video.{name} must be a non-negative integer or null")
            if value is not None and value < 0:
                raise ConfigurationError(f"video.{name} must be a non-negative integer or null")
        if start_frame is not None and end_frame is not None and end_frame < start_frame:
            raise ConfigurationError("video.end_frame must be greater than or equal to start_frame")
        sync_tolerance = float(video_config.get("multi_camera_sync_tolerance_sec", 0.050))
        calibration_status = self.config.get("calibration", {}).get("status", "not_calibrated")

        inference_pool: ThreadPoolExecutor | None = None
        readers: list[VideoReader] = []
        estimators: list[PoseEstimator] = []
        hand_landmarkers: list[HandLandmarkerWrapper] = []
        rula_observations: list[ScoreObservation] = []
        reba_observations: list[ScoreObservation] = []
        pending: list[_PendingObservation] = []
        multi_view_pending: list[_PendingMultiView] = []
        rula_completion = MethodCompletionTracker("RULA")
        reba_completion = MethodCompletionTracker("REBA")
        processed_frames = 0
        pose_detected_frames = 0
        pose_observations = 0
        ambiguous_multi_person_frames = 0
        synchronized_frame_sets = 0
        synchronization_spread_total_sec = 0.0
        synchronization_max_spread_sec = 0.0
        synchronization_fallback_frame_sets = 0
        recovery_report: dict[str, Any] = {
            "status": "not_applicable",
            "reason": "temporal recovery runs on single-camera tracks only",
        }
        next_sample_timestamp = -math.inf
        last_video_timestamp = 0.0
        warnings = [
            "MediaPipe world landmarks are monocular model-estimated 3-D, not calibrated true 3-D.",
            "RULA and REBA are observational screening methods, not diagnostic instruments.",
            "No validated lift-event detector is available; detected_lifts is therefore not inferred.",
        ]
        if target_fps is not None:
            warnings.append(
                f"Frames were sampled at up to {float(target_fps):g} Hz; shorter posture peaks may be missed."
            )

        start_time = time.perf_counter()
        try:
            readers = [VideoReader(path, fps_override=fps_override) for path in paths]
            reported_counts = [reader.frame_count for reader in readers if reader.frame_count > 0]
            available_source_frames = min(reported_counts) if reported_counts else 0
            work_start = start_frame or 0
            work_stop = (
                min(end_frame, available_source_frames - 1)
                if end_frame is not None and available_source_frames > 0
                else available_source_frames - 1
            )
            total_source_frames = (
                max(0, work_stop - work_start + 1) if available_source_frames > 0 else 0
            )
            if progress_callback:
                progress_callback(
                    0,
                    total_source_frames,
                    "loading_model",
                    "Video opened. Loading pose model and inference resources.",
                )
            estimators = self._create_pose_estimators(len(readers))
            hand_landmarkers = self._create_hand_landmarkers(len(readers))
            if len(readers) > 1 and bool(self.config.get("pose", {}).get("parallel_cameras", True)):
                inference_pool = ThreadPoolExecutor(
                    max_workers=len(readers), thread_name_prefix="pose"
                )
            if progress_callback:
                progress_callback(
                    0,
                    total_source_frames,
                    "processing",
                    "Pose model ready. Processing decoded source frames.",
                )
            tracker_config = self.config.get("tracking", {})
            tracker = CentroidPoseTracker(
                max_distance=float(tracker_config.get("max_normalized_distance", 0.20)),
                max_missed_frames=int(tracker_config.get("max_missed_frames", 15)),
            )
            fusion_config = self.config.get("fusion", {})
            fusion = MultiViewFusion(
                camera_roles=fusion_config.get("camera_roles", {}),
                visibility_threshold=float(fusion_config.get("visibility_threshold", 0.5)),
                prefer_hand_mesh=fusion_config.get("prefer_hand_mesh", False),
            )
            # RNLE is computed after the loop: its geometry is estimated from
            # the lifting sequence, which is not known until every frame is read.
            with ResultWriter(output) as writer:
                while True:
                    decoded = [reader.read() for reader in readers]
                    if any(frame is None for frame in decoded):
                        break
                    frames: list[DecodedFrame] = decoded  # type: ignore[assignment]
                    timestamps = [frame.timestamp_sec for frame in frames]
                    timestamp_spread_sec = max(timestamps) - min(timestamps)
                    if timestamp_spread_sec > sync_tolerance:
                        raise VideoInputError(
                            "multi-camera decoder timestamps differ by "
                            f"{timestamp_spread_sec:.6f}s, exceeding {sync_tolerance:.6f}s"
                        )
                    timestamp = timestamps[0]
                    last_video_timestamp = timestamp
                    frame_number = frames[0].frame_number
                    if start_frame is not None and frame_number < start_frame:
                        continue
                    if end_frame is not None and frame_number > end_frame:
                        break
                    source_progress = max(0, frame_number - work_start + 1)
                    if timestamp < next_sample_timestamp:
                        if progress_callback and (
                            source_progress % 5 == 0 or source_progress == total_source_frames
                        ):
                            progress_callback(
                                source_progress,
                                total_source_frames,
                                "processing",
                                f"Sampling source frame {frame_number:,}.",
                            )
                        continue
                    if sample_interval is not None:
                        # Decoder time-base conversions and addition can round the same
                        # boundary in opposite directions by a few floating-point steps.
                        next_sample_timestamp = math.nextafter(
                            timestamp + sample_interval, -math.inf, steps=4
                        )
                    processed_frames += 1
                    camera_frame_references, synchronization = self._camera_frame_evidence(
                        frames,
                        paths,
                        fusion.index_to_role,
                        sync_tolerance,
                    )
                    synchronized_frame_sets += 1
                    synchronization_spread_total_sec += timestamp_spread_sec
                    synchronization_max_spread_sec = max(
                        synchronization_max_spread_sec, timestamp_spread_sec
                    )
                    if any(frame.timestamp_source != "decoder_pts" for frame in frames):
                        synchronization_fallback_frame_sets += 1

                    detections = self._detect_all(estimators, frames, inference_pool)
                    if len(readers) == 1:
                        tracked = tracker.update(detections[0])
                        if tracked:
                            pose_detected_frames += 1
                        frame_hand = None
                        if hand_landmarkers and len(tracked) == 1:
                            frame_hand = hand_landmarkers[0].process_frame(
                                frames[0].image_bgr,
                                timestamp,
                                frame_number,
                                pose_norm_landmarks=tracked[0].pose.normalized,
                            )
                        # Scoring is deferred: temporal recovery needs the frames
                        # on both sides of a gap, so observations are collected
                        # here and emitted once the whole sequence is known.
                        for tracked_pose in tracked:
                            pose_observations += 1
                            pending.append(
                                _PendingObservation(
                                    frame_number=frame_number,
                                    timestamp_sec=timestamp,
                                    person_id=tracked_pose.person_id,
                                    pose=tracked_pose.pose,
                                    hand=frame_hand,
                                    timestamp_source=frames[0].timestamp_source,
                                    camera_frame_references=camera_frame_references,
                                    synchronization=synchronization,
                                )
                            )
                    else:
                        if any(len(camera_detections) > 1 for camera_detections in detections):
                            ambiguous_multi_person_frames += 1
                            if progress_callback:
                                progress_callback(
                                    source_progress,
                                    total_source_frames,
                                    "processing",
                                    (
                                        f"Source frame {frame_number:,} withheld: "
                                        "multi-camera identity was ambiguous."
                                    ),
                                )
                            continue
                        if any(len(camera_detections) == 0 for camera_detections in detections):
                            if progress_callback:
                                progress_callback(
                                    source_progress,
                                    total_source_frames,
                                    "processing",
                                    (
                                        f"Source frame {frame_number:,} withheld: a camera "
                                        "view had no usable pose."
                                    ),
                                )
                            continue
                        pose_detected_frames += 1
                        pose_observations += 1
                        per_camera_angles = []
                        mean_visibilities = []
                        hand_confidences: list[float | None] = []
                        for camera_index, (camera_detections, frame) in enumerate(
                            zip(detections, frames, strict=True)
                        ):
                            pose = camera_detections[0]
                            hand = None
                            if hand_landmarkers:
                                hand = hand_landmarkers[camera_index].process_frame(
                                    frame.image_bgr,
                                    frame.timestamp_sec,
                                    frame.frame_number,
                                    pose_norm_landmarks=pose.normalized,
                                )
                            hand_confidences.append(
                                hand.detection_confidence if hand is not None else None
                            )
                            per_camera_angles.append(
                                self.angle_calculator.calculate_all_angles(
                                    self._joints(pose), hand_landmarks=self._hand_format(hand)
                                )
                            )
                            mean_visibilities.append(pose.mean_keypoint_visibility)
                        angles, selection_map = fusion.fuse_angles(
                            per_camera_angles, list(range(len(per_camera_angles)))
                        )
                        # The sequence analyses consume whole landmark positions
                        # rather than angles, so they get the camera that sees
                        # the relevant joints best on this frame instead of
                        # camera zero unconditionally.
                        landmark_camera, sequence_landmarks = select_landmark_source(
                            [
                                {
                                    index: np.asarray(point, dtype=float)
                                    for index, point in camera_detections[0].world.items()
                                }
                                for camera_detections in detections
                            ],
                            visibility_threshold=float(
                                fusion_config.get("visibility_threshold", 0.5)
                            ),
                        )
                        # Scoring is deferred here for the same reason as the
                        # single-camera path: the interval-level task conditions
                        # are not known until the whole sequence has been read.
                        record = {
                            "software_version": SOFTWARE_VERSION,
                            "frame_number": frame_number,
                            "timestamp_sec": timestamp,
                            "person_identifier": "person-0001",
                            "input_source": input_source,
                            "calibration_status": calibration_status,
                            "timestamp_source": [frame.timestamp_source for frame in frames],
                            "camera_frame_references": camera_frame_references,
                            "synchronization": synchronization,
                            "angles": angles.to_dict(),
                            "angles_by_camera": {
                                str(camera_index): camera_angles.to_dict()
                                for camera_index, camera_angles in enumerate(per_camera_angles)
                            },
                            "angle_units": "degrees",
                            "coordinate_convention": CANONICAL_COORDINATE_SYSTEM,
                            "multi_view_selection": selection_map,
                            "pose_landmarks_normalized_by_camera": {
                                str(camera_index): self._normalized_landmarks(camera_detections[0])
                                for camera_index, camera_detections in enumerate(detections)
                            },
                            "pose_landmarks_world_by_camera": {
                                str(camera_index): self._world_landmarks(camera_detections[0])
                                for camera_index, camera_detections in enumerate(detections)
                            },
                        }
                        multi_view_pending.append(
                            _PendingMultiView(
                                frame_number=frame_number,
                                timestamp_sec=timestamp,
                                angles=angles,
                                mean_visibility=float(np.mean(mean_visibilities)),
                                hand_confidences=hand_confidences,
                                world_landmarks=sequence_landmarks,
                                landmark_camera_index=landmark_camera,
                                record=record,
                            )
                        )

                    if progress_callback:
                        progress_callback(
                            source_progress,
                            total_source_frames,
                            "processing",
                            (
                                f"Processed source frame {frame_number:,}; "
                                f"{processed_frames:,} sampled frame(s) assessed."
                            ),
                        )

                self._emit_multi_view(
                    multi_view_pending,
                    writer=writer,
                    input_source=input_source,
                    calibration_status=calibration_status,
                    rula_observations=rula_observations,
                    reba_observations=reba_observations,
                    rula_completion=rula_completion,
                    reba_completion=reba_completion,
                )
                recovery_report = self._emit_pending(
                    pending,
                    writer=writer,
                    input_source=paths[0] if paths else input_source,
                    calibration_status=calibration_status,
                    rula_observations=rula_observations,
                    reba_observations=reba_observations,
                    rula_completion=rula_completion,
                    reba_completion=reba_completion,
                )

            rnle_result = self._compute_rnle(input_source, calibration_status)

            if progress_callback:
                progress_callback(
                    total_source_frames or processed_frames,
                    total_source_frames,
                    "finalizing",
                    "Calculating temporal summaries and writing reports.",
                )
            temporal = self.config.get("temporal_aggregation", {})
            percentile = float(temporal.get("percentile", 90.0))
            minimum_duration = float(temporal.get("sustained_min_duration_sec", 1.0))
            maximum_gap = float(temporal.get("maximum_gap_sec", 0.25))
            rula_summary = summarize_score_series(
                rula_observations,
                percentile=percentile,
                sustained_threshold=int(temporal.get("rula_high_risk_threshold", 5)),
                sustained_min_duration_sec=minimum_duration,
                maximum_gap_sec=maximum_gap,
            )
            reba_summary = summarize_score_series(
                reba_observations,
                percentile=percentile,
                sustained_threshold=int(temporal.get("reba_high_risk_threshold", 8)),
                sustained_min_duration_sec=minimum_duration,
                maximum_gap_sec=maximum_gap,
            )
            rula_completion_summary = rula_completion.summary()
            reba_completion_summary = reba_completion.summary()
            # A method that never reached a score is the most important thing the
            # run can report, so it is stated as a warning rather than left for
            # the reader to infer from a zero observation count.
            for completion in (rula_completion_summary, reba_completion_summary):
                if completion.total_observations and completion.scored_observations == 0:
                    warnings.append(completion.describe())
            fallback_count = sum(reader.timestamp_fallback_count for reader in readers)
            if fallback_count:
                warnings.append(
                    f"Decoder timestamps were unavailable/non-monotonic for {fallback_count} frames; "
                    "frame-index/FPS fallback was used and is recorded per frame."
                )
            if ambiguous_multi_person_frames:
                warnings.append(
                    f"{ambiguous_multi_person_frames} multi-camera frames contained more than one person "
                    "and were not assessed because cross-camera identity was ambiguous."
                )
            camera_count = len(readers)
            synchronization_mode = "lockstep_pts_gated" if camera_count > 1 else "single_camera"
            synchronization_status = (
                "not_applicable"
                if camera_count <= 1
                else "within_tolerance"
                if synchronized_frame_sets > 0
                else "not_checked"
            )
            summary = {
                "software_version": SOFTWARE_VERSION,
                "input_source": input_source,
                "processed_frames": processed_frames,
                "pose_detected_frames": pose_detected_frames,
                "pose_observations": pose_observations,
                "valid_rula_observations": len(rula_observations),
                "valid_reba_observations": len(reba_observations),
                "detected_lifts": (
                    None if self._rnle_estimate is None else self._rnle_estimate.lift_count
                ),
                "lift_detection_status": (
                    "withheld_multiple_tracked_people"
                    if self._person_provenance
                    else "not_performed"
                    if self._rnle_estimate is None
                    else ("detected" if self._rnle_estimate.lifts else "no_lift_detected")
                ),
                "per_lift_rnle_results": [],
                "rnle_input_estimate": (
                    None if self._rnle_estimate is None else self._rnle_estimate.to_dict()
                ),
                "duration_sec": last_video_timestamp,
                "runtime_sec": time.perf_counter() - start_time,
                "calibration_status": calibration_status,
                "rula_summary": rula_summary,
                "reba_summary": reba_summary,
                "method_completion": {
                    "rula": rula_completion_summary,
                    "reba": reba_completion_summary,
                },
                "rnle": rnle_result,
                "video_metadata": [
                    {
                        "camera_index": camera_index,
                        "camera_role": fusion.index_to_role.get(camera_index),
                        "path": str(reader.path),
                        "native_fps": reader.native_fps,
                        "effective_fps": reader.fps,
                        "reported_frame_count": reader.frame_count,
                        "decoded_width_px": reader.width,
                        "decoded_height_px": reader.height,
                        "rotation_metadata_deg": reader.rotation_deg,
                    }
                    for camera_index, reader in enumerate(readers)
                ],
                "synchronization": {
                    "mode": synchronization_mode,
                    "master_camera_index": 0,
                    "master_camera_role": fusion.index_to_role.get(0),
                    "tolerance_sec": sync_tolerance,
                    "status": synchronization_status,
                    "alignment_performed": False,
                    "checked_frame_sets": synchronized_frame_sets,
                    "frame_sets_with_timestamp_fallback": (synchronization_fallback_frame_sets),
                    "mean_timestamp_spread_sec": (
                        synchronization_spread_total_sec / synchronized_frame_sets
                        if synchronized_frame_sets
                        else None
                    ),
                    "maximum_timestamp_spread_sec": (
                        synchronization_max_spread_sec if synchronized_frame_sets else None
                    ),
                },
                "task_input_provenance": self._task_input_provenance(),
                "pose_recovery": recovery_report,
                "data_quality": {
                    "timestamp_fallback_frames": fallback_count,
                    "ambiguous_multi_person_frames": ambiguous_multi_person_frames,
                    "camera_count": len(readers),
                    "tracking_mode": (
                        "per-camera centroid tracking"
                        if len(readers) == 1
                        else "single-person-only synchronized multi-view"
                    ),
                },
                "warnings": warnings,
            }
            write_summary(output, summary)
            (output / "run_config.json").write_text(
                json.dumps(serializable(self.config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if progress_callback:
                progress_callback(
                    total_source_frames or processed_frames,
                    total_source_frames,
                    "completed",
                    "Analysis and report generation completed.",
                )
            return str(output)
        finally:
            if inference_pool is not None:
                inference_pool.shutdown(wait=True)
            for landmarker in hand_landmarkers:
                landmarker.close()
            for estimator in estimators:
                estimator.close()
            for reader in readers:
                reader.close()
