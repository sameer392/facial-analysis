"""Pose Landmarker extraction (MediaPipe Tasks, FULL model, VIDEO mode, single subject).

Output per video, one row per decoded frame:
    image  float32 (T, 33, 5): x, y, z, visibility, presence (normalised image coordinates)
    world  float32 (T, 33, 5): the same channels in world coordinates (metres, hip centre), when provided
    detected bool (T,)         whether a pose was found on that frame

Frames without a detection are NaN rows, never dropped. More than one person in a frame is refused,
not merged. Timestamps are the shared frame timestamps from video_frames (fps_derived for now).

The detector is injected, so tests run without MediaPipe. load_pose_detector() verifies the model's
SHA-256 before MediaPipe is imported or the model is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from extraction.model_registry import get_spec, verify_model
from extraction.video_frames import frame_timestamps, iter_frames, open_video_info

N_POSE_LANDMARKS = 33
POSE_CHANNELS = ("x", "y", "z", "visibility", "presence")


class ExtractionError(Exception):
    """Pose extraction cannot produce correct data for this input."""


@dataclass(frozen=True)
class PoseConfig:
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    output_segmentation_masks: bool = False

    def __post_init__(self):
        if self.num_poses != 1:
            raise ValueError("num_poses must be 1: the schema is single-subject, multiple people are refused")
        for name in ("min_pose_detection_confidence", "min_pose_presence_confidence", "min_tracking_confidence"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")

    def as_dict(self) -> dict:
        return {
            "num_poses": self.num_poses,
            "min_pose_detection_confidence": self.min_pose_detection_confidence,
            "min_pose_presence_confidence": self.min_pose_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "output_segmentation_masks": self.output_segmentation_masks,
        }


@dataclass
class PoseResult:
    image: np.ndarray
    world: np.ndarray
    detected: np.ndarray
    timestamps_ms: np.ndarray
    timestamp_source: str = "fps_derived"
    source_filename: str = ""
    config: dict = field(default_factory=dict)

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if self.detected.size else 0.0


def _landmarks_to_array(landmarks) -> np.ndarray:
    if len(landmarks) != N_POSE_LANDMARKS:
        raise ExtractionError(f"expected {N_POSE_LANDMARKS} pose landmarks, got {len(landmarks)}")
    out = np.full((N_POSE_LANDMARKS, len(POSE_CHANNELS)), np.nan, dtype=np.float32)
    for k, lm in enumerate(landmarks):
        out[k, 0] = lm.x
        out[k, 1] = lm.y
        out[k, 2] = lm.z
        if getattr(lm, "visibility", None) is not None:
            out[k, 3] = lm.visibility
        if getattr(lm, "presence", None) is not None:
            out[k, 4] = lm.presence
    return out


def extract_pose_from_frames(frames: Iterable[tuple[int, np.ndarray]], timestamps_ms, detector,
                             source_filename: str = "", config: PoseConfig | None = None,
                             timestamp_source: str = "fps_derived") -> PoseResult:
    """Run the detector over (index, frame) pairs and assemble the arrays. Detector is any object
    with detect(frame_bgr, timestamp_ms) returning pose_landmarks and pose_world_landmarks lists."""
    ts = np.asarray(timestamps_ms, dtype=np.int64)
    n = len(ts)
    if n and np.any(np.diff(ts) <= 0):
        raise ExtractionError("timestamps must strictly increase")
    cfg = config or PoseConfig()

    image = np.full((n, N_POSE_LANDMARKS, len(POSE_CHANNELS)), np.nan, dtype=np.float32)
    world = np.full_like(image, np.nan)
    detected = np.zeros(n, dtype=bool)
    seen = 0
    for expected, (index, frame) in enumerate(frames):
        if index != expected:
            raise ExtractionError(f"frame index {index} is out of sequence (expected {expected})")
        if expected >= n:
            raise ExtractionError(f"more frames than timestamps ({n})")
        out = detector.detect(frame, int(ts[expected]))
        poses = list(out.pose_landmarks or [])
        if len(poses) > 1:
            raise ExtractionError(f"frame {index}: more than one person detected; refusing to merge identities")
        if poses:
            image[expected] = _landmarks_to_array(poses[0])
            detected[expected] = True
            worlds = list(out.pose_world_landmarks or [])
            if worlds:
                world[expected] = _landmarks_to_array(worlds[0])
        seen += 1
    if seen != n:
        raise ExtractionError(f"{seen} frames were processed but {n} timestamps were given")
    return PoseResult(image=image, world=world, detected=detected, timestamps_ms=ts,
                      timestamp_source=timestamp_source, source_filename=source_filename,
                      config=cfg.as_dict())


def extract_pose_video(video_path, detector, config: PoseConfig | None = None) -> PoseResult:
    """Read a video with the shared frame reader and extract its pose landmarks."""
    info = open_video_info(video_path)
    ts, source = frame_timestamps(video_path)
    return extract_pose_from_frames(iter_frames(video_path), ts, detector,
                                    source_filename=info.path.name, config=config,
                                    timestamp_source=source)


class PoseDetector:
    """Thin wrapper around MediaPipe's PoseLandmarker in VIDEO mode."""

    def __init__(self, landmarker, mp_module):
        self._landmarker = landmarker
        self._mp = mp_module

    def detect(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, int(timestamp_ms))

    def close(self):
        self._landmarker.close()


def load_pose_detector(models_dir=None, config: PoseConfig | None = None) -> PoseDetector:
    """Verify the Pose FULL model's SHA-256, then create the MediaPipe landmarker. Imports MediaPipe last."""
    cfg = config or PoseConfig()
    model_path = verify_model(get_spec("pose_full"), models_dir=models_dir)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(Path(model_path))),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=cfg.num_poses,
        min_pose_detection_confidence=cfg.min_pose_detection_confidence,
        min_pose_presence_confidence=cfg.min_pose_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        output_segmentation_masks=cfg.output_segmentation_masks,
    )
    return PoseDetector(vision.PoseLandmarker.create_from_options(options), mp)
