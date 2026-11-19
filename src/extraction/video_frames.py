"""Shared video frame reader for the extraction pipeline.

Every detector (face, pose, hands) must see the same frames with the same indexes and timestamps,
so the video is read by this module and the frames are passed on. Nothing here changes the pixels.

Errors are raised, never hidden: a missing, unreadable, empty or truncated video is reported
with the file name, so no frames are silently dropped.

Timestamps come from the frame rate (round(i * 1000 / fps), the same rule as the V2 schema),
because decoded presentation times are not yet extracted. The source is recorded as "fps_derived".
Requires src/analysis on sys.path (for schema_v2), as the other modules do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from schema_v2 import frame_timestamps_ms


class VideoError(Exception):
    """A video cannot be read completely and correctly."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    reported_frame_count: int

    def as_dict(self) -> dict:
        return {
            "filename": self.path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 6),
            "reported_frame_count": self.reported_frame_count,
        }


def open_video_info(path) -> VideoInfo:
    """Read the video's properties without decoding its frames."""
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoError(f"video cannot be opened: {p.name}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise VideoError(f"video cannot be opened (no valid frame size): {p.name}")
    if not np.isfinite(fps) or fps <= 0:
        raise VideoError(f"video has no valid frame rate: {p.name}")
    return VideoInfo(path=p, width=width, height=height, fps=fps, reported_frame_count=max(count, 0))


def iter_frames(path, strict: bool = True) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, frame_bgr) for every decoded frame, in order.

    strict=True (default): raise VideoError if the video has no frames, or if fewer frames decode
    than the container reports. Set strict=False only when the reported count is known to be wrong.
    """
    info = open_video_info(path)
    cap = cv2.VideoCapture(str(info.path))
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield decoded, frame
            decoded += 1
    finally:
        cap.release()
    if decoded == 0:
        raise VideoError(f"video contains no frames: {info.path.name}")
    if strict and info.reported_frame_count > 0 and decoded < info.reported_frame_count:
        raise VideoError(
            f"video is truncated or damaged: decoded {decoded} of {info.reported_frame_count} "
            f"frames in {info.path.name}"
        )


def frame_timestamps(path, n_frames: int | None = None) -> tuple[np.ndarray, str]:
    """Integer millisecond timestamps for the frames of a video, and their source ("fps_derived").

    n_frames defaults to the container's reported count. Use the decoded count when it is known.
    """
    info = open_video_info(path)
    count = info.reported_frame_count if n_frames is None else n_frames
    return frame_timestamps_ms(count, fps=info.fps)
