"""Browser-playable proxies for source recordings.

The captured recordings are FMP4 -- MPEG-4 Part 2, OpenCV's default when a
``VideoWriter`` is opened without an explicit codec. No browser decodes it:
Chromium, Firefox and Safari between them cover H.264, VP8/9 and AV1, and none
of them cover this. Streaming the original therefore produces a player that
reports an error no matter how correct the byte-range handling is.

So a proxy is transcoded once per recording and cached beside the results. The
original file is never touched, and the proxy is derived data that can be deleted
at any time and will simply be rebuilt.

Transcoding a two-minute recording takes a few seconds, which is too long to
block a request. Builds therefore run on a background thread and the endpoint
reports progress, so the client can show a real state instead of a stalled video
element.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from pipeline.recording_sources import recording_hash

LOGGER = logging.getLogger(__name__)

#: Proxies live beside the analysis so they are cleaned up with it and never
#: pollute the participant's raw capture folder.
PROXY_DIRNAME = ".video_proxy"

#: 720p is enough to review posture against a timeline and keeps the transcode
#: fast. CRF 26 with a fast preset trades a little quality for a build that
#: finishes while the reviewer is still reading the verdict.
FFMPEG_ARGS = (
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "26",
    "-vf", "scale='min(1280,iw)':-2",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
)

_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
_state: dict[str, str] = {}


class ProxyError(RuntimeError):
    """Raised when a browser-playable proxy cannot be produced."""


@dataclass(frozen=True, slots=True)
class ProxyStatus:
    state: str  # "ready" | "building" | "absent" | "failed" | "unavailable"
    path: Path | None
    detail: str = ""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def proxy_path(result_dir: str | Path, camera_index: int, digest: str) -> Path:
    return Path(result_dir) / PROXY_DIRNAME / f"cam{camera_index}-{digest}.mp4"


def _lock_for(key: str) -> threading.Lock:
    with _registry_lock:
        return _locks.setdefault(key, threading.Lock())


def _build(source: Path, target: Path, key: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-nostdin", "-y", "-i", str(source), *FFMPEG_ARGS, str(staging)]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if completed.returncode != 0 or not staging.is_file():
            tail = (completed.stderr or "").strip().splitlines()[-3:]
            raise ProxyError("ffmpeg failed: " + " ".join(tail))
        # Publish atomically so a reader never sees a half-written file.
        staging.replace(target)
        with _registry_lock:
            _state[key] = "ready"
    except Exception as exc:  # noqa: BLE001 - recorded and reported to the client
        LOGGER.warning("Video proxy build failed for %s: %s", source.name, exc)
        with _registry_lock:
            _state[key] = f"failed:{exc}"
        staging.unlink(missing_ok=True)


def ensure_proxy(
    source: str | Path, result_dir: str | Path, camera_index: int, *, start: bool = True
) -> ProxyStatus:
    """Return the proxy for a recording, starting a build if one is needed."""

    source_path = Path(source)
    try:
        digest = recording_hash(source_path)
    except OSError:
        return ProxyStatus("failed", None, f"The source recording is unavailable: {source_path.name}")
    target = proxy_path(result_dir, camera_index, digest)
    key = str(target)

    if target.is_file() and target.stat().st_size > 0:
        return ProxyStatus("ready", target)
    if not ffmpeg_available():
        return ProxyStatus(
            "unavailable",
            None,
            "ffmpeg is not installed, so this recording cannot be converted for playback.",
        )
    if not source_path.is_file():
        return ProxyStatus("failed", None, f"The source recording is missing: {source_path.name}")

    with _registry_lock:
        current = _state.get(key, "")
    if current.startswith("failed:"):
        return ProxyStatus("failed", None, current.split(":", 1)[1])
    if current == "building":
        return ProxyStatus("building", None, "Converting this recording for playback.")
    if not start:
        return ProxyStatus("absent", None, "No playable copy has been built yet.")

    lock = _lock_for(key)
    if not lock.acquire(blocking=False):
        return ProxyStatus("building", None, "Converting this recording for playback.")
    try:
        with _registry_lock:
            _state[key] = "building"

        def run() -> None:
            try:
                _build(source_path, target, key)
            finally:
                lock.release()

        threading.Thread(target=run, name=f"proxy-{camera_index}", daemon=True).start()
    except Exception:
        lock.release()
        raise
    return ProxyStatus("building", None, "Converting this recording for playback.")


__all__ = [
    "PROXY_DIRNAME",
    "ProxyError",
    "ProxyStatus",
    "ensure_proxy",
    "ffmpeg_available",
    "proxy_path",
]
