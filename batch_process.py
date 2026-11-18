"""Batch entry point for schema-v4 assessments.

The runner never infers load, coupling, duration, dominant-side selection, or
other ergonomic task conditions from folder names.  A task may supply an
``assessment_inputs.yaml`` override beside its ``raw`` directory; otherwise the
base configuration's null values remain null and assessments are incomplete.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

from api.jobs import restore_reviewed_setup
from dashboard.processor_wrapper import _OutputProcessLock
from paths import CONFIG_PATH, RESULTS_DIR, STUDIES_DIR
from pipeline.recording_sources import VIDEO_EXTENSIONS, bind_recordings, summary_recordings
from pipeline.video_processor import VideoProcessor

# Trials are independent and write to separate output directories, so they
# parallelise without interacting. Each trial additionally uses one inference
# thread per camera, so the default stays modest rather than filling the machine:
# eight workers on three-camera trials already occupies 24 threads.
DEFAULT_MAX_WORKERS = 4


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def _merge(target: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if key != "camera_roles" and isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def discover_all_jobs(studies_root: Path = STUDIES_DIR) -> list[dict[str, Any]]:
    """Discover the documented study/participant/task/raw folder structure."""

    jobs: list[dict[str, Any]] = []
    if not studies_root.is_dir():
        return jobs
    for study_config_path in sorted(studies_root.glob("*/study_config.json")):
        study_path = study_config_path.parent
        with study_config_path.open(encoding="utf-8") as handle:
            study_config = json.load(handle)
        task_names = {
            task.get("folder_name"): task.get("name", task.get("folder_name"))
            for task in study_config.get("tasks", [])
            if isinstance(task, dict) and task.get("folder_name")
        }
        for participant_info_path in sorted(study_path.glob("*/participant_info.json")):
            participant_path = participant_info_path.parent
            with participant_info_path.open(encoding="utf-8") as handle:
                participant = json.load(handle)
            participant_id = str(participant.get("participant_id", participant_path.name))
            for raw_path in sorted(participant_path.glob("*/raw")):
                videos = sorted(
                    str(path.resolve())
                    for path in raw_path.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                )
                if not videos and not (raw_path.parent / "source_manifest.json").exists():
                    continue
                task_path = raw_path.parent
                override = task_path / "assessment_inputs.yaml"
                jobs.append(
                    {
                        "study_folder": study_path.name,
                        "participant_id": participant_id,
                        "task_folder": task_path.name,
                        "task_path": task_path,
                        "task_name": task_names.get(task_path.name, task_path.name),
                        "video_paths": videos,
                        "assessment_override": override if override.is_file() else None,
                    }
                )
    return jobs


def build_config(base_config: dict[str, Any], override_path: Path | None) -> dict[str, Any]:
    """Return an isolated config with only explicit task-level overrides applied."""

    config = copy.deepcopy(base_config)
    if override_path is not None:
        _merge(config, _load_yaml(override_path))
    return config


def run_job(job: dict[str, Any], base_config: dict[str, Any], output: str) -> dict[str, Any]:
    """Process one trial. Module-level so it can run in a worker process."""

    label = f"{job['participant_id']} / {job['task_name']}"
    try:
        with _OutputProcessLock(Path(output)):
            config = build_config(base_config, job["assessment_override"])
            saved = None
            prior = Path(output) / "video_summary.json"
            if prior.exists():
                prior_config = Path(output) / "run_config.json"
                if prior_config.exists():
                    previous = json.loads(prior_config.read_text(encoding="utf-8"))
                    saved = previous.get("assessment_workflow", {}).get("recordings")
                    restore_reviewed_setup(config, previous, job["task_path"])
                if saved is None:
                    saved = summary_recordings(json.loads(prior.read_text(encoding="utf-8")))
                if saved is None:
                    raise ValueError(
                        "The previous result does not identify its original recordings."
                    )
            videos = bind_recordings(job["task_path"], job["video_paths"], config, saved)
            VideoProcessor(config).process(videos, output)
        return {"label": label, "ok": True, "output": output}
    except Exception as exc:  # reported per job; one bad trial must not stop the batch
        return {"label": label, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _resolve_worker_count(requested: int | None, job_count: int) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError("--workers must be at least 1")
        return min(requested, job_count)
    return max(1, min(DEFAULT_MAX_WORKERS, job_count, os.cpu_count() or 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Process every discovered schema-v4 study task")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--studies-root", type=Path, default=STUDIES_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "trials to process concurrently. Trials are independent and write to "
            f"separate directories, so results are unaffected. Defaults to at most "
            f"{DEFAULT_MAX_WORKERS}; use 1 to process serially."
        ),
    )
    args = parser.parse_args()

    base_config = _load_yaml(args.config)
    jobs = discover_all_jobs(args.studies_root)
    if not jobs:
        print(f"No study jobs found under {args.studies_root.resolve()}")
        return 0

    pending: list[tuple[dict[str, Any], str]] = []
    for index, job in enumerate(jobs, start=1):
        output = (
            args.results_root / job["study_folder"] / job["participant_id"] / job["task_folder"]
        )
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            print(f"[{index}/{len(jobs)}] SKIP non-empty output (use --overwrite): {output}")
            continue
        pending.append((job, str(output)))

    if not pending:
        return 0

    workers = _resolve_worker_count(args.workers, len(pending))
    print(f"Processing {len(pending)} trial(s) with {workers} worker(s).")

    failures = 0
    completed = 0
    if workers == 1:
        for job, output in pending:
            result = run_job(job, base_config, output)
            completed += 1
            failures += _report(result, completed, len(pending))
        return 1 if failures else 0

    # Worker output would interleave unreadably, so each trial reports once it
    # finishes rather than streaming progress.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job, base_config, output): job for job, output in pending}
        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # worker died rather than returning a failure
                job = futures[future]
                result = {
                    "label": f"{job['participant_id']} / {job['task_name']}",
                    "ok": False,
                    "error": f"worker failed: {type(exc).__name__}: {exc}",
                }
            failures += _report(result, completed, len(pending))
    return 1 if failures else 0


def _report(result: dict[str, Any], completed: int, total: int) -> int:
    """Print one trial's outcome and return 1 when it failed."""

    prefix = f"[{completed}/{total}]"
    if result["ok"]:
        print(f"{prefix} OK   {result['label']}")
        return 0
    print(f"{prefix} FAIL {result['label']}: {result['error']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
