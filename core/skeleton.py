"""Anatomical skeleton topology for drawing a pose.

MediaPipe's 33 landmarks have no neck and no spine. The conventional way to draw
them anyway is to run a line from each ear down to the shoulder on the same side,
which is what this project did. Those two diagonals are not anatomy: they cross
where the neck should be, they move with head rotation rather than with the neck,
and they leave the trunk drawn as an open quadrilateral with nothing along its
axis. A reviewer looking at a neck score has nothing in the picture that
corresponds to the neck.

This module supplies the two joints the drawing needs and the model does not
provide:

* **C7**, the base of the neck, taken as the shoulder midpoint.
* **Head centre**, the centroid of both ears and both eyes.

The neck is then C7 to the head centre, and the spine is C7 to the pelvis. Both
match what ``AngleCalculator`` already measures: ``_neck_relative_to_trunk``
computes neck flexion from exactly this head centroid, chosen over the nose tip
because a centroid of symmetric landmarks does not swing with head yaw. The
picture and the number now describe the same segment.

There is no landmark at the top of the head, and none can be synthesised
reliably: the crown's height above the eyes depends on hair and headwear. The
head centre is the highest anatomically defensible point available, and it is the
one the measurements use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Union

#: A landmark is [x, y, z] or [x, y, z, visibility]. Callers pass plain lists
#: from the JSON records and numpy arrays from the live pipeline; numpy arrays
#: are indexable and sized but are not typing.Sequence, so both are named.
Point = Union[Sequence[float], "Any"]

# Real MediaPipe landmarks referenced here.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24

# Synthesised joints. Numbered well above the model's range so they can never
# collide with a real landmark id, and so a reader of a serialised record can
# tell at a glance that these were constructed.
C7 = 100
HEAD_CENTRE = 101
PELVIS = 102

VIRTUAL_JOINTS = (C7, HEAD_CENTRE, PELVIS)

VIRTUAL_JOINT_NAMES = {
    C7: "c7_neck_base",
    HEAD_CENTRE: "head_centre",
    PELVIS: "pelvis",
}

#: Landmarks averaged to build each synthesised joint, in preference order. The
#: head centre falls back to the nose only when no symmetric pair is visible,
#: because the nose swings with head rotation and the centroid does not.
_VIRTUAL_SOURCES: dict[int, tuple[tuple[int, ...], ...]] = {
    C7: ((LEFT_SHOULDER, RIGHT_SHOULDER),),
    PELVIS: ((LEFT_HIP, RIGHT_HIP),),
    HEAD_CENTRE: (
        (LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE),
        (LEFT_EAR, RIGHT_EAR),
        (LEFT_EYE, RIGHT_EYE),
        (NOSE,),
    ),
}

#: Edges of the drawn skeleton. Anatomically ordered: axis first, then limbs.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Axis: neck and spine, the segments the methods actually score.
    (C7, HEAD_CENTRE),
    (C7, PELVIS),
    # Shoulder and pelvic girdles.
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_HIP, RIGHT_HIP),
    # Trunk sides, which keep the torso reading as a solid block.
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    # Left arm and hand.
    (LEFT_SHOULDER, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    # Right arm and hand.
    (RIGHT_SHOULDER, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    # Left leg and foot.
    (LEFT_HIP, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    # Right leg and foot.
    (RIGHT_HIP, 26),
    (26, 28),
    (28, 30),
    (28, 32),
)


def _average(points: Sequence[Point]) -> list[float]:
    """Component-wise mean, keeping whatever width the inputs carry.

    Landmarks arrive either as ``[x, y, z]`` or as ``[x, y, z, visibility]``
    depending on the caller. Averaging to the narrowest width preserves both,
    and averaging visibility is the right rule for a synthesised joint: it is
    only as trustworthy as the landmarks it was built from.
    """

    width = min(len(point) for point in points)
    return [sum(float(point[axis]) for point in points) / len(points) for axis in range(width)]


def augment_with_virtual_joints(
    landmarks: Mapping[int, Point],
) -> dict[int, Any]:
    """Return the landmarks plus C7, the head centre, and the pelvis.

    A synthesised joint is omitted when its source landmarks are absent, so a
    partially visible pose simply loses that segment instead of gaining an
    invented one. Existing entries are never overwritten.
    """

    augmented: dict[int, Any] = dict(landmarks)
    for joint, source_sets in _VIRTUAL_SOURCES.items():
        if joint in augmented:
            continue
        for sources in source_sets:
            points = [landmarks[index] for index in sources if index in landmarks]
            if len(points) == len(sources):
                augmented[joint] = _average(points)
                break
    return augmented
