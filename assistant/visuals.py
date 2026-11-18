"""Select a recorded viewpoint and crop a native-resolution frame around visible joints."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from api import analysis as analysis_api
from api.replay import _landmarks_for_camera
from api.sessions import SessionError, parse_session_id, session_videos


def cameras(session_id: str) -> list[dict[str, Any]]:
    from api.main import STUDIES_ROOT

    return session_videos(STUDIES_ROOT, parse_session_id(session_id))


_JOINTS = {
    "hand_and_forearm": (13, 15, 17, 19, 21),
    "hand": (15, 17, 19, 21),
    "grip": (15, 17, 19, 21),
    "wrist": (15, 17, 19, 21),
    "forearm": (13, 15),
    "lower_arm": (13, 15),
    "upper_arm": (11, 13),
    "shoulder": (11, 13),
    "trunk": (11, 23),
    "neck": (7, 11),
    "leg": (23, 25, 27),
}


def focus_joints(region: str) -> list[int]:
    key = region.lower().replace("-", "_").replace(" ", "_")
    indices = next((indices for part, indices in _JOINTS.items() if part in key), ())
    sides = [1] if "right" in key else [0] if "left" in key else [0, 1]
    return [joint + side for joint in indices for side in sides]


def _points(record: dict, camera: int, joints: list[int]) -> list[tuple[float, float, float]]:
    raw = _landmarks_for_camera(record, camera) or {}
    result = []
    for joint in joints:
        point = raw.get(str(joint))
        if not isinstance(point, (tuple, list)) or len(point) < 4:
            continue
        x, y, visibility = point[0], point[1], point[3]
        if all(type(v) in (float, int) and math.isfinite(v) for v in (x, y, visibility)):
            if 0 <= x <= 1 and 0 <= y <= 1 and visibility >= 0.5:
                result.append((x, y, visibility))
    return result


def _crop(jpeg: bytes, points: list[tuple[float, float, float]], width: int) -> bytes | None:
    if len(points) < 2:
        return None
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    xs, ys = [p[0] * w for p in points], [p[1] * h for p in points]
    # Include adjacent surfaces and the grasped object, not just the joints.
    margin = max(0.09 * min(w, h), 0.4 * max(max(xs) - min(xs), max(ys) - min(ys)))
    roi = image[
        max(0, int(min(ys) - margin)) : min(h, math.ceil(max(ys) + margin)),
        max(0, int(min(xs) - margin)) : min(w, math.ceil(max(xs) + margin)),
    ]
    if not roi.size:
        return None
    if roi.shape[1] > width:
        roi = cv2.resize(
            roi,
            (width, max(1, round(width * roi.shape[0] / roi.shape[1]))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", roi, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return encoded.tobytes() if ok else None


def render(
    session_id: str,
    detail: dict[str, Any],
    method: str,
    camera: int | None,
    region: str,
    overlay: bool,
    width: int,
    budget: int,
) -> tuple[list[tuple[str, bytes]], list[dict[str, Any]]]:
    from api.main import _frame_jpeg

    sources = cameras(session_id)
    indices = [source["camera_index"] for source in sources if source.get("available", True)]
    if not indices:
        raise SessionError("No recorded camera source is available for this assessment.")
    if camera is not None and (type(camera) is not int or camera not in indices):
        raise SessionError(
            f"The requested camera is unavailable. Available camera indices: {indices}."
        )
    joints = focus_joints(region)
    points: dict[int, list[tuple[float, float, float]]] = {}
    multiple_workers = len(analysis_api.people(session_id)) > 1
    if joints or multiple_workers:
        record = analysis_api._store(session_id).load_record_at_offset(detail["offset"])
        points = {index: _points(record, index, joints or list(range(33))) for index in indices}
    # A selected view takes precedence; if none was selected use landmark
    # visibility as a retrieval hint. Only the pixels can establish contact.
    if camera is None:
        camera = max(indices, key=lambda i: sum(p[2] for p in points.get(i, [])))
    if multiple_workers:
        if len(_points(record, camera, list(range(33)))) < 3:
            raise SessionError("The selected worker could not be located in this camera view.")
        # Identify which worker the question refers to in the full context. A
        # separate crop below still has the raw pixels for contact inspection.
        overlay = True
    frame, person = detail["frame"], detail.get("person")
    ref = next((r for r in detail.get("cameras", []) if r["cameraIndex"] == camera), {})
    base = {
        "frame": frame,
        "sec": detail.get("timestampSec"),
        "camera": camera,
        "person": person,
        "method": method,
        "region": region or None,
        "sourceFrame": ref.get("sourceFrame", frame),
    }
    kind = "overlay" if overlay else "raw"
    jpeg = _frame_jpeg(session_id, frame, camera, region, overlay, width, person)
    refs = [{**base, "view": kind}]
    images = [(f"frame {frame}, {base['sec']} s, camera {camera}, {person}, {kind}", jpeg)]
    if joints and budget > 1 and len(points.get(camera, [])) >= 2:
        native = _frame_jpeg(session_id, frame, camera, "", False, 0, person)
        crop = _crop(native, points[camera], width)
        if crop:
            images.append(
                (f"Detail of {region}, same frame {frame}, camera {camera}, {person}", crop)
            )
            refs.append({**base, "view": "crop"})
    return images, refs
