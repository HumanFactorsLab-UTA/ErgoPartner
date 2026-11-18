"""Replay the synchronized, single-worker study from cached per-camera pose evidence.

This evaluates changes after pose detection on the full recording chronology.
The v4 cache saved hand-mesh wrist angles but omitted hand coordinates, so the
three measured wrist angles from each successful hand detection are held fixed.
The key supplies frame identities only. Reviewer answers never enter scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from core.angle_calculator import AngleCalculator, BodyAngles
from core.multi_view_fusion import MultiViewFusion, select_landmark_source
from paths import PROJECT_ROOT
from pipeline.video_processor import SOFTWARE_VERSION, VideoProcessor
from temporal.task_dynamics import PostureSample

IDENTITY = ("frame_id", "study", "participant", "task", "frame_number")


def source_hashes() -> dict[str, str]:
    files = [Path(__file__).resolve()]
    for package in ("core", "ergonomics", "pipeline", "temporal"):
        files.extend(sorted((PROJECT_ROOT / package).glob("*.py")))
    return {
        p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def camera_angles(calculator: AngleCalculator, world: dict, cached: dict) -> BodyAngles:
    angles = calculator.calculate_all_angles(world)
    for side in ("left", "right"):
        source = f"{side}_wrist_data_source"
        if cached[source] != "hand_mesh":
            continue
        setattr(angles, source, "hand_mesh")
        for motion in ("flexion", "deviation", "pronation"):
            name = f"{side}_wrist_{motion}"
            unavailable = f"{name}_deg"
            value = cached[name]
            if value is None or not np.isfinite(value):
                if unavailable not in angles.unavailable_measurements:
                    angles.unavailable_measurements.append(unavailable)
                continue
            setattr(angles, name, value)
            if unavailable in angles.unavailable_measurements:
                angles.unavailable_measurements.remove(unavailable)
    return angles


def replay_recording(job: tuple[Path, list[dict], dict]) -> tuple[list[dict], dict]:
    directory, identities, overrides = job
    targets = {int(row["frame_number"]): row for row in identities}
    config_bytes = (directory / "run_config.json").read_bytes()
    config = json.loads(config_bytes)
    if overrides.get("absolute_reference"):
        config.setdefault("posture_conditions", {})["neutral_reference"] = {"source": "absolute"}
    if "neck_flexion_method" in overrides:
        config["quality"]["neck_flexion_method"] = overrides["neck_flexion_method"]
    if "arm_abduction_method" in overrides:
        config["quality"]["arm_abduction_method"] = overrides["arm_abduction_method"]
    processor = VideoProcessor(config)
    fusion = MultiViewFusion(
        config["fusion"]["camera_roles"],
        config["fusion"]["visibility_threshold"],
        **(
            {"prefer_hand_mesh": True}
            if overrides.get("prefer_hand_mesh") or config["fusion"].get("prefer_hand_mesh")
            else {}
        ),
    )
    samples: list[PostureSample] = []
    all_angles: list[BodyAngles] = []
    scored_frames = {}
    digest = hashlib.sha256()
    previous_frame, previous_time = -1, -1.0
    with (directory / "frame_results.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            number, timestamp = record["frame_number"], record["timestamp_sec"]
            if (
                record["person_identifier"] != "person-0001"
                or number <= previous_frame
                or timestamp <= previous_time
            ):
                raise ValueError(
                    f"Expected chronological single-worker observations at {directory}/{number}"
                )
            previous_frame, previous_time = number, timestamp
            worlds = record["pose_landmarks_world_by_camera"]
            cameras = sorted(map(int, worlds))
            if len(cameras) < 2 or set(worlds) != set(record["angles_by_camera"]):
                raise ValueError(f"Incomplete synchronized camera evidence at {directory}/{number}")
            coordinates = [
                {int(k): np.asarray(v, dtype=float) for k, v in worlds[str(camera)].items()}
                for camera in cameras
            ]
            per_camera = [
                camera_angles(processor.angle_calculator, world, record["angles_by_camera"][str(i)])
                for i, world in zip(cameras, coordinates, strict=True)
            ]
            angles, _ = fusion.fuse_angles(per_camera, cameras)
            _, landmarks = select_landmark_source(
                coordinates, visibility_threshold=fusion.vis_threshold
            )
            all_angles.append(angles)
            samples.append(PostureSample(timestamp, number, landmarks=landmarks))
            if number in targets:
                scored_frames[number] = (
                    len(samples) - 1,
                    {
                        "frame_number": number,
                        "timestamp_sec": timestamp,
                        "person_identifier": record["person_identifier"],
                        "input_source": record["input_source"],
                        "calibration_status": record["calibration_status"],
                        "mean_visibility": record["pose_quality"]["mean_keypoint_visibility"],
                        "hand_detection_confidence": record["pose_quality"][
                            "hand_detection_confidence"
                        ],
                    },
                )
    processor._smooth_condition_angles(all_angles)
    processor._analyze_dynamics(
        [
            PostureSample(s.timestamp_sec, s.frame_number, a.to_dict(), s.landmarks)
            for s, a in zip(samples, all_angles, strict=True)
        ]
    )
    rows = []
    for number, identity in targets.items():
        row = {**identity, "reba_final": None, "rula_final": None}
        if number in scored_frames:
            index, context = scored_frames[number]
            rula, reba, _, _ = processor._score_angles(all_angles[index], **context)
            for name, result in (("rula", rula), ("reba", reba)):
                if result.selected is not None:
                    row[f"{name}_final"] = result.selected.final_score
        rows.append(row)
    provenance = {
        "directory": str(directory),
        "observations": len(samples),
        "matched": len(scored_frames),
        "requested": len(targets),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "jsonl_sha256": digest.hexdigest(),
    }
    return rows, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True, help="Root of the saved fps variant")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--neck-flexion-method",
        choices=("legacy_cone", "sagittal_projection"),
        help="Isolate the neck measurement method in an ablation",
    )
    parser.add_argument(
        "--arm-abduction-method",
        choices=("frontal_projection", "lateral_excursion"),
        help="Compare frontal projection with outward excursion from the sagittal plane",
    )
    parser.add_argument(
        "--prefer-hand-mesh",
        action="store_true",
        help="Test preferring eligible hand-mesh cameras before the camera role",
    )
    parser.add_argument(
        "--absolute-reference",
        action="store_true",
        help="Sensitivity analysis with zero offsets for posture conditions",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs: dict[Path, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    source_keys: set[tuple] = set()
    with args.key.open(encoding="utf-8-sig", newline="") as stream:
        for full_row in csv.DictReader(stream):
            row = {name: full_row[name] for name in IDENTITY}
            directory = args.results / row["study"] / row["participant"] / row["task"]
            source_key = directory.resolve(), int(row["frame_number"])
            if row["frame_id"] in seen or source_key in source_keys:
                raise ValueError(f"Duplicate frame identity {row}")
            seen.add(row["frame_id"])
            source_keys.add(source_key)
            jobs[directory].append(row)
    if not jobs:
        parser.error("No frame identities in key")
    args.output.mkdir(parents=True, exist_ok=False)
    implementation = source_hashes()
    rows = []
    records = []
    overrides = {
        k: v
        for k, v in {
            "absolute_reference": args.absolute_reference,
            "neck_flexion_method": args.neck_flexion_method,
            "arm_abduction_method": args.arm_abduction_method,
            "prefer_hand_mesh": args.prefer_hand_mesh,
        }.items()
        if v
    }
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch, record in pool.map(
            replay_recording, [(p, identities, overrides) for p, identities in jobs.items()]
        ):
            rows.extend(batch)
            records.append(record)
    if implementation != source_hashes():
        raise RuntimeError("Replay source changed during execution. Rerun with a frozen checkout.")
    with (args.output / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*IDENTITY, "reba_final", "rula_final"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["frame_id"]))
    manifest = {
        "pipeline_version": SOFTWARE_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": implementation,
        "key_sha256": hashlib.sha256(args.key.read_bytes()).hexdigest(),
        "mode": "cached_pose_replay_with_saved_hand_angles",
        "overrides": overrides,
        "recordings": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
