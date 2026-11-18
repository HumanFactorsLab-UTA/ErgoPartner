"""
Filesystem scanner for participant-centered dashboard navigation.

The primary dashboard exposes only scientifically traceable current results.
Legacy loaders remain below solely for compatibility with archived page code.

Public functions are memoized with dashboard.cache.ttl_cache, so filesystem
discovery stays usable from any interface rather than only from Streamlit.
"""

import json
import logging
import os

import pandas as pd

from dashboard.cache import ttl_cache

# Import paths module for consistent path handling
from paths import RESULTS_DIR_STR

LOGGER = logging.getLogger(__name__)

# V2 -> V3 column name mapping
V2_TO_V3 = {
    "frame": "frame_number",
    "time_sec": "timestamp_sec",
    "reba_score": "reba_final_score",
    "risk_level": "reba_risk_level",
    "neck": "reba_neck_score",
    "trunk": "reba_trunk_score",
    "legs": "reba_legs_score",
    "upper_arm": "reba_upper_arm_score",
    "lower_arm": "reba_lower_arm_score",
    "wrist": "reba_wrist_score",
    "score_a": "reba_score_a",
    "score_b": "reba_score_b",
    "activity": "reba_activity_score",
}


def _load_json(path: str) -> dict | None:
    """Load a JSON mapping and log expected file/format failures."""
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
        if not isinstance(value, dict):
            LOGGER.warning("Expected a JSON object in %s", path)
            return None
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not load JSON %s: %s", path, exc)
        return None


def get_results_root() -> str:
    """Get the results root directory (project_root/results/)."""
    return RESULTS_DIR_STR


def get_task_results_path(
    study_folder: str, participant_id: str, task_folder: str, fps_value: int | None = None
) -> str:
    """Compute the results path for a specific task.

    Always returns FPS-separated path with appropriate suffix:
      - For native/no-FPS: results/native/{study}/{participant}/{task}/
      - For specific FPS (e.g., 15): results/15fps/{study}/{participant}/{task}/
    """
    # Determine folder name based on fps_value
    if fps_value is None or fps_value == 0:
        fps_folder = "native"  # Use 'native' for non-FPS-separated results
    else:
        fps_folder = f"{fps_value}fps"  # Add "fps" suffix (e.g., "15fps")

    return os.path.join(
        get_results_root(), fps_folder, study_folder, str(participant_id), task_folder
    )


@ttl_cache(ttl=30)
def get_processed_fps(study_folder: str, participant_id: str, task_folder: str) -> list[int | None]:
    """
    Detect which FPS values have been processed for a specific task.

    Scans results/{fps}fps/<study>/<participant>/<task>/ directories and checks
    run_config.json files to extract target_fps values.

    Returns list of integers (e.g., [10, 15, 30]) representing processed FPS values.
    Also includes 'native' if native directory exists.
    """
    fps_values = set()
    results_root = get_results_root()

    if not os.path.isdir(results_root):
        return []

    # Valid FPS folder patterns: "10fps", "15fps", "30fps", or "native"
    for entry in sorted(os.listdir(results_root)):
        fps_path = os.path.join(results_root, entry)

        # Only check FPS-marked directories (with 'fps' suffix or 'native')
        if not os.path.isdir(fps_path):
            continue

        task_result_path = os.path.join(fps_path, study_folder, str(participant_id), task_folder)

        if os.path.isdir(task_result_path):
            # Check for run_config.json to confirm valid processing
            config_file = os.path.join(task_result_path, "run_config.json")
            if os.path.isfile(config_file):
                config_data = _load_json(config_file)

                # Handle 'native' directory (no FPS value needed)
                if entry == "native":
                    fps_values.add(None)  # Use None to represent native/no-FPS results
                    continue

                # Extract numeric value from folder name or run_config.json
                try:
                    if config_data and "video" in config_data:
                        fps_str = config_data["video"].get("target_fps", "")

                        if isinstance(fps_str, int):
                            fps_num = fps_str
                        elif isinstance(fps_str, str) and fps_str:
                            fps_num = int(fps_str.replace("fps", ""))
                        else:
                            # Try to parse from directory name (e.g., "15fps" -> 15)
                            if entry.endswith("fps"):
                                try:
                                    fps_num = int(entry[:-3])  # Remove "fps" suffix
                                except ValueError:
                                    continue
                            else:
                                continue

                        if fps_num and 5 <= fps_num <= 60:  # Reasonable FPS range
                            fps_values.add(fps_num)
                    elif entry.endswith("fps"):
                        # Fallback to parsing from directory name
                        try:
                            fps_num = int(entry[:-3])
                            if 5 <= fps_num <= 60:
                                fps_values.add(fps_num)
                        except ValueError:
                            pass
                except (ValueError, AttributeError):
                    pass

    return sorted(fps_values, key=lambda value: -1 if value is None else value)


@ttl_cache(ttl=30)
def participant_catalog(root: str) -> list[dict]:
    """Build the participant-first catalog used by the unified dashboard.

    Internal study and task folder names remain available as path keys for the
    application but are deliberately excluded from user-facing labels.
    """
    catalog: list[dict] = []
    for study in discover_studies(root):
        created = str(study.get("created_at") or "")
        date_label = created[:10] if len(created) >= 10 else "Date unavailable"
        for participant in discover_participants(study["path"]):
            participant_id = str(participant["id"])
            tasks = discover_tasks(
                participant["path"],
                study["tasks"],
                study_folder=study["folder_name"],
                participant_id=participant_id,
            )
            catalog.append(
                {
                    "key": f"{study['folder_name']}::{participant_id}",
                    "label": f"Participant {participant_id} · {date_label}",
                    "participant_id": participant_id,
                    "participant": participant,
                    "study": study,
                    "tasks": tasks,
                    "date_label": date_label,
                    "search_text": " ".join(
                        [
                            participant_id,
                            date_label,
                            str(participant["info"].get("sex", "")),
                            str(participant["info"].get("dominant_hand", "")),
                            *(str(task["name"]) for task in tasks),
                        ]
                    ).lower(),
                }
            )
    return catalog


def detect_current_results(
    study_folder: str,
    participant_id: str,
    task_folder: str,
    sampling_fps: int | None,
) -> dict | None:
    """Return a complete current-method result bundle, never a legacy result."""
    result_dir = get_task_results_path(study_folder, participant_id, task_folder, sampling_fps)
    result = _detect_v3_new_structure(result_dir)
    return result if result and result.get("structure") == "v4" else None


@ttl_cache(ttl=30)
def discover_studies(root: str) -> list[dict]:
    """
    Scan root folder for study directories.
    Each study must contain a study_config.json.
    Returns list of dicts with keys: name, path, config, created_at, tasks, folder_name.
    """
    studies = []
    if not os.path.isdir(root):
        return studies

    for entry in sorted(os.listdir(root)):
        study_path = os.path.join(root, entry)
        config_path = os.path.join(study_path, "study_config.json")
        if not os.path.isdir(study_path) or not os.path.isfile(config_path):
            continue
        config = _load_json(config_path)
        if config is None:
            continue
        studies.append(
            {
                "name": config.get("study_name", entry),
                "folder_name": entry,
                "path": study_path,
                "config": config,
                "created_at": config.get("created_at", ""),
                "tasks": config.get("tasks", []),
            }
        )

    return studies


@ttl_cache(ttl=30)
def discover_participants(study_path: str) -> list[dict]:
    """
    Scan study folder for participant directories.
    Each participant must contain participant_info.json.
    Returns list of dicts with keys: id, path, info (full JSON).
    """
    participants = []
    if not os.path.isdir(study_path):
        return participants

    for entry in sorted(os.listdir(study_path)):
        p_path = os.path.join(study_path, entry)
        info_path = os.path.join(p_path, "participant_info.json")
        if not os.path.isdir(p_path) or not os.path.isfile(info_path):
            continue
        info = _load_json(info_path)
        if info is None:
            continue
        participants.append(
            {
                "id": info.get("participant_id", entry),
                "path": p_path,
                "info": info,
            }
        )

    return participants


@ttl_cache(ttl=30)
def discover_tasks(
    participant_path: str, study_tasks: list[dict], study_folder: str = "", participant_id: str = ""
) -> list[dict]:
    """
    Discover tasks for a participant, matching against study_config task definitions.
    Returns list of dicts with keys: task_id, name, folder_name, path, primary_tool,
    secondary_tools, description, has_v2, cameras, study_folder, participant_id.
    Only checks FPS-separated directories (10fps, 15fps, 30fps, native). No legacy support.
    """
    tasks = []
    task_lookup = {t["folder_name"]: t for t in study_tasks}

    for entry in sorted(os.listdir(participant_path)):
        task_path = os.path.join(participant_path, entry)
        if not os.path.isdir(task_path) or entry.startswith("."):
            continue
        if entry == "participant_info.json":
            continue

        task_def = task_lookup.get(entry, {})
        videos = find_videos(task_path)
        num_cameras = len(videos.get("raw", []))

        # Check for v3 results in FPS-separated structure only (no legacy checks)
        processed_fps_list = get_processed_fps(study_folder, participant_id, entry)

        tasks.append(
            {
                "task_id": task_def.get("task_id", 0),
                "name": task_def.get("name", entry),
                "folder_name": entry,
                "path": task_path,
                "primary_tool": task_def.get("primary_tool", "reba"),
                "secondary_tools": task_def.get("secondary_tools", []),
                "description": task_def.get("description", ""),
                "num_repetitions": task_def.get("num_repetitions", 0),
                # Computed properties for backward compatibility with existing code
                "has_v3_with_fps": len(processed_fps_list) > 0 or None in processed_fps_list,
                "processed_fps_list": [f for f in processed_fps_list if f is not None],
                "has_native_results": None in processed_fps_list,
                # Backward compatible keys: has_v3 = True if any FPS-separated results exist
                "has_v3": len(processed_fps_list) > 0 or None in processed_fps_list,
                "has_v2": os.path.isfile(os.path.join(task_path, "scores.json")),
                "cameras": num_cameras,
                "study_folder": study_folder,
                "participant_id": participant_id,
            }
        )

    return tasks


@ttl_cache(ttl=30)
def find_videos(task_path: str) -> dict[str, list[str]]:
    """
    Find video files in raw/ and overlay/ subdirectories.
    Returns dict with keys 'raw' and 'overlay', each a sorted list of paths.
    """
    result = {"raw": [], "overlay": []}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for subdir in ["raw", "overlay"]:
        folder = os.path.join(task_path, subdir)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in video_exts:
                result[subdir].append(os.path.join(folder, f))

    return result


@ttl_cache(ttl=30)
def detect_v3_results(
    task_path: str,
    study_folder: str = "",
    participant_id: str = "",
    task_folder: str = "",
    fps_value: int | None = None,
) -> dict | None:
    """
    Check for v3 results in FPS-separated directories only.

    Checks path based on fps_value parameter:
      - If fps_value is provided (e.g., 15): checks results/15fps/{study}/{participant}/{task}/
      - If fps_value is None or not set: checks results/native/{study}/{participant}/{task}/

    Returns dict with paths to CSVs, peak frames (organized by camera), reports, etc.

    Note: Legacy support removed — only FPS-separated structure supported.
    """
    if study_folder and participant_id and task_folder:
        results_path = get_task_results_path(study_folder, participant_id, task_folder, fps_value)
        if os.path.isdir(results_path):
            result = _detect_v3_new_structure(results_path)
            if result:
                return result

    # Legacy support removed — only FPS-separated directories supported

    return None


def _detect_v3_new_structure(results_dir: str) -> dict | None:
    """Detect v3 results in new organized structure."""
    v4_csv = os.path.join(results_dir, "frame_results.csv")
    v4_summary = os.path.join(results_dir, "video_summary.json")
    if os.path.isfile(v4_csv) and os.path.isfile(v4_summary):
        return {
            "dir": results_dir,
            "structure": "v4",
            "scores_csv": v4_csv,
            "frame_jsonl": os.path.join(results_dir, "frame_results.jsonl"),
            "summary_csv": None,
            "summary_json": v4_summary,
            "per_camera_csv": None,
            "peak_frames": [],
            "peak_frames_by_camera": {},
            "llm_analysis": None,
            "vlm_analysis": None,
            "run_config": os.path.join(results_dir, "run_config.json"),
            "run_metadata": None,
        }

    scores_dir = os.path.join(results_dir, "scores")
    scores_csv = os.path.join(scores_dir, "frame_scores.csv")
    if not os.path.isfile(scores_csv):
        return None

    def _check(path):
        return path if os.path.exists(path) else None

    # Collect peak frames organized by camera role
    peak_dir = os.path.join(results_dir, "peak_frames")
    peak_images_by_camera: dict[str, list[str]] = {}
    all_peak_images: list[str] = []
    if os.path.isdir(peak_dir):
        for cam_role in sorted(os.listdir(peak_dir)):
            cam_dir = os.path.join(peak_dir, cam_role)
            if not os.path.isdir(cam_dir):
                continue
            images = sorted(
                [
                    os.path.join(cam_dir, f)
                    for f in os.listdir(cam_dir)
                    if f.lower().endswith((".jpg", ".png"))
                ]
            )
            peak_images_by_camera[cam_role] = images
            all_peak_images.extend(images)

    reports_dir = os.path.join(results_dir, "reports")

    return {
        "dir": results_dir,
        "structure": "new",
        "scores_csv": scores_csv,
        "summary_csv": _check(os.path.join(scores_dir, "summary.csv")),
        "per_camera_csv": _check(os.path.join(scores_dir, "per_camera_angles.csv")),
        "peak_frames": all_peak_images,
        "peak_frames_by_camera": peak_images_by_camera,
        "llm_analysis": _check(os.path.join(reports_dir, "llm_analysis.txt")),
        "vlm_analysis": _check(os.path.join(reports_dir, "vlm_analysis.txt")),
        "run_config": _check(os.path.join(results_dir, "run_config.json")),
        "run_metadata": _check(os.path.join(results_dir, "run_metadata.json")),
    }


@ttl_cache(ttl=30)
def detect_v2_results(task_path: str) -> dict | None:
    """
    Check for v2 results (scores.json, reba_scores.csv).
    Returns dict with parsed scores.json data and path to CSV, or None.
    """
    scores_json_path = os.path.join(task_path, "scores.json")
    if not os.path.isfile(scores_json_path):
        return None

    data = _load_json(scores_json_path)
    if data is None:
        return None

    csv_path = os.path.join(task_path, "reba_scores.csv")
    return {
        "scores_json": data,
        "scores_json_path": scores_json_path,
        "csv_path": csv_path if os.path.isfile(csv_path) else None,
    }


@ttl_cache(ttl=60)
def load_scores_dataframe(csv_path: str, version: str = "v3") -> pd.DataFrame | None:
    """
    Load a scores CSV into a DataFrame.
    If version='v2', renames columns to v3 conventions.
    Returns None on failure.
    """
    try:
        df = pd.read_csv(csv_path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        LOGGER.warning("Could not load CSV %s: %s", csv_path, exc)
        return None

    if version == "v2":
        df = df.rename(columns=V2_TO_V3)
    elif {"reba_score", "rula_score", "rula_status", "reba_status"}.issubset(df.columns):
        df = df.rename(
            columns={
                "reba_score": "reba_final_score",
                "rula_score": "rula_final_score",
            }
        )

    return df


@ttl_cache(ttl=60)
def get_all_participant_summaries(root: str) -> pd.DataFrame:
    """
    Build a comparison DataFrame across all studies/participants/tasks.
    """
    rows = []
    studies = discover_studies(root)

    for study in studies:
        participants = discover_participants(study["path"])
        for p in participants:
            info = p["info"]
            tasks = discover_tasks(
                p["path"],
                study["tasks"],
                study_folder=study["folder_name"],
                participant_id=str(p["id"]),
            )
            for task in tasks:
                row = {
                    "study": study["name"],
                    "study_folder": study["folder_name"],
                    "participant_id": p["id"],
                    "age": info.get("age"),
                    "sex": info.get("sex"),
                    "height_cm": info.get("height_cm"),
                    "weight_kg": info.get("weight_kg"),
                    "dominant_hand": info.get("dominant_hand"),
                    "task_name": task["name"],
                    "task_folder": task["folder_name"],
                    "primary_tool": task["primary_tool"],
                    "reba_mean": None,
                    "reba_max": None,
                    "rula_mean": None,
                    "rula_max": None,
                    "niosh_rwl": None,
                    "niosh_li": None,
                    "result_version": None,
                }

                # Try v3 first - check all available FPS values
                processed_fps = get_processed_fps(
                    study["folder_name"], str(p["id"]), task["folder_name"]
                )

                v3 = None
                # Try each FPS value (including None for native)
                for fps_val in processed_fps:
                    v3 = detect_v3_results(
                        task["path"],
                        study_folder=study["folder_name"],
                        participant_id=str(p["id"]),
                        task_folder=task["folder_name"],
                        fps_value=fps_val,
                    )
                    if v3 and v3["scores_csv"]:
                        break  # Use first valid result found

                if v3 and v3["scores_csv"]:
                    df = load_scores_dataframe(v3["scores_csv"], "v3")
                    if df is not None:
                        row["result_version"] = "v3"
                        if "reba_final_score" in df.columns:
                            row["reba_mean"] = df["reba_final_score"].mean()
                            row["reba_max"] = df["reba_final_score"].max()
                        if "rula_final_score" in df.columns:
                            row["rula_mean"] = df["rula_final_score"].mean()
                            row["rula_max"] = df["rula_final_score"].max()
                        if "niosh_rwl" in df.columns:
                            row["niosh_rwl"] = df["niosh_rwl"].iloc[0]
                            row["niosh_li"] = df["niosh_lifting_index"].iloc[0]

                # Fallback to v2
                if row["result_version"] is None:
                    v2 = detect_v2_results(task["path"])
                    if v2:
                        row["result_version"] = "v2"
                        scores = v2["scores_json"]
                        if "reba" in scores:
                            row["reba_mean"] = scores["reba"].get("mean_score")
                            row["reba_max"] = scores["reba"].get("max_score")
                        if "niosh" in scores:
                            row["niosh_rwl"] = scores["niosh"].get("rwl")
                            row["niosh_li"] = scores["niosh"].get("lifting_index")

                rows.append(row)

    return pd.DataFrame(rows)
