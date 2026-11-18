"""
Visualization Utilities - Skeleton and Angle Overlays.

Draws pose skeleton, angle arcs, and score labels on video frames
using normalized (pixel-space) landmarks from MediaPipe.
Used for debug visualization and VLM context frames.
"""

import cv2
import numpy as np

from core.angle_calculator import BodyAngles
from core.skeleton import POSE_CONNECTIONS, VIRTUAL_JOINTS, augment_with_virtual_joints

__all__ = ["POSE_CONNECTIONS", "draw_skeleton", "draw_angle_labels"]

# The skeleton previously ran two lines from the nose to the shoulders and
# called them "approximate neck". They track head rotation rather than neck
# posture, and they meet at the nose rather than at the neck. core/skeleton.py
# synthesises a C7 and a head centre instead, so the drawn neck is the same
# segment the neck angle is measured from.


def draw_skeleton(
    frame: np.ndarray,
    normalized_landmarks: dict[int, np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    point_radius: int = 4,
    visibility_threshold: float = 0.5,
) -> np.ndarray:
    """
    Draw pose skeleton on frame using normalized landmarks.

    Args:
        frame: BGR image
        normalized_landmarks: {id: [x, y, z, visibility]} in [0,1] range, Y-DOWN
        color: BGR color for skeleton
        thickness: Line thickness
        point_radius: Landmark point radius
        visibility_threshold: Minimum visibility to draw

    Returns:
        Frame with skeleton overlay (in-place modification)
    """
    h, w = frame.shape[:2]
    drawable = augment_with_virtual_joints(normalized_landmarks)

    def to_pixel(lm_id):
        lm = drawable.get(lm_id)
        if lm is None:
            return None
        visibility = lm[3] if len(lm) > 3 else 1.0
        if visibility < visibility_threshold:
            return None
        return (int(lm[0] * w), int(lm[1] * h))

    # Draw connections
    for id1, id2 in POSE_CONNECTIONS:
        pt1 = to_pixel(id1)
        pt2 = to_pixel(id2)
        if pt1 is not None and pt2 is not None:
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw points for measured landmarks only. A synthesised joint is a drawing
    # aid, and marking it like an observation would overstate the evidence.
    for lm_id, lm in drawable.items():
        if lm_id in VIRTUAL_JOINTS:
            continue
        visibility = lm[3] if len(lm) > 3 else 1.0
        if visibility < visibility_threshold:
            continue
        pt = (int(lm[0] * w), int(lm[1] * h))
        cv2.circle(frame, pt, point_radius, color, -1)

    return frame


def draw_angle_labels(
    frame: np.ndarray,
    angles: BodyAngles,
    reba_score: int = 0,
    rula_score: int = 0,
    timestamp_sec: float | None = None,
    font_scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Draw angle values and scores as text overlay on frame.

    Args:
        frame: BGR image
        angles: Computed body angles
        reba_score: Current REBA score
        rula_score: Current RULA score
        timestamp_sec: Optional timestamp in seconds to display
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background panel
    panel_h = 200
    if timestamp_sec is not None:
        panel_h += 25  # extra row for timestamp
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 22
    dy = 18

    # Timestamp (top of panel)
    if timestamp_sec is not None:
        mins, secs = divmod(timestamp_sec, 60)
        ts_text = f"Time: {mins:04.0f}:{secs:05.2f}"
        cv2.putText(frame, ts_text, (10, y), font, font_scale + 0.05, (255, 255, 0), 1)
        y += dy

    # Scores
    reba_color = _score_color(reba_score, max_score=12)
    rula_color = _score_color(rula_score, max_score=7)
    cv2.putText(frame, f"REBA: {reba_score}", (10, y), font, font_scale, reba_color, 1)
    cv2.putText(frame, f"RULA: {rula_score}", (140, y), font, font_scale, rula_color, 1)
    y += dy + 5

    # Angle values
    labels = [
        (f"Trunk: {angles.trunk_flexion:+.1f} deg", "trunk_flexion"),
        (f"Neck: {angles.neck_flexion:+.1f} deg", "neck_flexion"),
        (f"UpArm: {angles.upper_arm_flexion:.1f} deg", "upper_arm_flexion"),
        (f"Elbow: {angles.forearm_flexion:.1f} deg", "forearm_flexion"),
        (f"Wrist: {angles.wrist_flexion:+.1f} deg", "wrist_flexion"),
        (f"Knee: {angles.knee_flexion:.1f} deg", "knee_flexion"),
        (f"Twist: {angles.trunk_twist:.1f} deg", "trunk_twist"),
        (f"SideBend: {angles.trunk_side_bend:.1f} deg", "trunk_side_bend"),
    ]
    for label_text, _ in labels:
        if y + dy > panel_h:
            break
        cv2.putText(frame, label_text, (10, y), font, font_scale - 0.05, color, 1)
        y += dy

    return frame


def save_annotated_frame(
    frame: np.ndarray,
    normalized_landmarks: dict[int, np.ndarray] | None,
    angles: BodyAngles | None,
    reba_score: int,
    rula_score: int,
    output_path: str,
    timestamp_sec: float | None = None,
):
    """Save a single annotated frame to disk."""
    annotated = frame.copy()
    if normalized_landmarks:
        draw_skeleton(annotated, normalized_landmarks)
    if angles:
        draw_angle_labels(annotated, angles, reba_score, rula_score, timestamp_sec)
    cv2.imwrite(output_path, annotated)


def _score_color(score: int, max_score: int) -> tuple[int, int, int]:
    """Return BGR color from green (low risk) to red (high risk)."""
    ratio = min(1.0, score / max_score)
    r = int(255 * ratio)
    g = int(255 * (1 - ratio))
    return (0, g, r)  # BGR
