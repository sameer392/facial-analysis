"""Hand Landmarker extraction (MediaPipe Tasks, VIDEO mode, up to two hands).

Output per video, one row per decoded frame, in the same frame order and with the same timestamps as the
pose session (see video_frames.frame_timestamps):

    left, right           float32 (T, 21, 3): normalised image landmarks x, y, z for the side's hand
    left_world, right_world  float32 (T, 21, 3): world landmarks (metres) where the model gave them
    left_status, right_status  per-frame status of the side, one of pose_schema.HAND_SIDE_STATUSES
    left_score, right_score    float (T,): handedness score of the assigned hand (NaN when not detected)
    left_outside, right_outside  int (T,): landmarks outside the image frame (0..1) on that side
    left_world_available, right_world_available  bool (T,)
    unassigned  int (T,): hands found in the frame that could not be given a side

A hand is assigned to a side only when the model reports exactly one hand with that label in the frame.
If two hands share a label, or the label is unknown, every such hand is left unassigned and counted,
rather than guessed. Frames with no hand are NaN rows with status "not_detected", never dropped.

Limits of the model output: the Hand Landmarker does not report occlusion. "outside_frame" means some
landmarks lie outside the image; it is not a measure of occlusion. Handedness is the model's label.
Persistent identity across frames is not established in this module (that is phase H2).

The detector is injected, so tests run without MediaPipe. load_hand_detector() verifies the model's
SHA-256 before MediaPipe is imported or the model is opened.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from pose_schema import HAND_SIDE_STATUSES, N_HAND_LANDMARKS  # noqa: E402
from extraction.model_registry import get_spec, verify_model  # noqa: E402

LOW_CONFIDENCE = 0.7  # handedness score below this is reported as low_confidence (the data is kept)
SIDES = ("left", "right")


class HandExtractionError(Exception):
    """Hand extraction cannot produce correct data for this input."""


@dataclass(frozen=True)
class HandConfig:
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    low_confidence: float = LOW_CONFIDENCE

    def __post_init__(self):
        if self.num_hands not in (1, 2):
            raise ValueError("num_hands must be 1 or 2")
        for name in ("min_hand_detection_confidence", "min_hand_presence_confidence",
                     "min_tracking_confidence", "low_confidence"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1")

    def as_dict(self) -> dict:
        return {
            "num_hands": self.num_hands,
            "min_hand_detection_confidence": self.min_hand_detection_confidence,
            "min_hand_presence_confidence": self.min_hand_presence_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "low_confidence_threshold": self.low_confidence,
            "outside_frame_rule": "a landmark with x or y outside 0..1",
        }


@dataclass
class HandResult:
    left: np.ndarray
    right: np.ndarray
    left_world: np.ndarray
    right_world: np.ndarray
    left_status: list
    right_status: list
    left_score: np.ndarray
    right_score: np.ndarray
    left_outside: np.ndarray
    right_outside: np.ndarray
    left_world_available: np.ndarray
    right_world_available: np.ndarray
    unassigned: np.ndarray
    timestamps_ms: np.ndarray
    config: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.timestamps_ms.shape[0])

    def detected_frames(self, side: str) -> int:
        return sum(1 for s in getattr(self, f"{side}_status") if s in ("valid", "low_confidence", "outside_frame"))

    def status_counts(self) -> dict:
        counts = {}
        for side in SIDES:
            for s in HAND_SIDE_STATUSES:
                counts[f"{side}_{s}"] = sum(1 for v in getattr(self, f"{side}_status") if v == s)
        counts["unassigned_hand_frames"] = int(np.count_nonzero(self.unassigned))
        return counts


def _landmarks_to_array(landmarks) -> np.ndarray:
    if len(landmarks) != N_HAND_LANDMARKS:
        raise HandExtractionError(f"expected {N_HAND_LANDMARKS} hand landmarks, got {len(landmarks)}")
    out = np.empty((N_HAND_LANDMARKS, 3), dtype=np.float32)
    for k, lm in enumerate(landmarks):
        out[k, 0] = lm.x
        out[k, 1] = lm.y
        out[k, 2] = lm.z
    return out


def _label_and_score(categories) -> tuple[str, float]:
    if not categories:
        return "", float("nan")
    top = categories[0]
    return str(getattr(top, "category_name", "") or ""), float(getattr(top, "score", float("nan")))


def _outside(landmarks: np.ndarray) -> int:
    """Number of landmarks whose x or y lies outside the image (0..1)."""
    xy = landmarks[:, :2]
    return int(np.count_nonzero(np.any((xy < 0.0) | (xy > 1.0), axis=1)))


def extract_hands_from_frames(frames: Iterable[tuple[int, np.ndarray]], timestamps_ms, detector,
                              config: HandConfig | None = None) -> HandResult:
    """Run the detector over (index, frame) pairs. The detector returns an object with
    hand_landmarks, hand_world_landmarks and handedness lists, as MediaPipe's HandLandmarkerResult does."""
    ts = np.asarray(timestamps_ms, dtype=np.int64)
    n = len(ts)
    if n and np.any(np.diff(ts) <= 0):
        raise HandExtractionError("timestamps must strictly increase")
    cfg = config or HandConfig()

    nan_rows = lambda: np.full((n, N_HAND_LANDMARKS, 3), np.nan, dtype=np.float32)  # noqa: E731
    left, right, left_w, right_w = nan_rows(), nan_rows(), nan_rows(), nan_rows()
    left_status = ["not_processed"] * n
    right_status = ["not_processed"] * n
    left_score = np.full(n, np.nan)
    right_score = np.full(n, np.nan)
    left_out = np.zeros(n, dtype=np.int64)
    right_out = np.zeros(n, dtype=np.int64)
    left_wa = np.zeros(n, dtype=bool)
    right_wa = np.zeros(n, dtype=bool)
    unassigned = np.zeros(n, dtype=np.int64)

    seen = 0
    for expected, (index, frame) in enumerate(frames):
        if index != expected:
            raise HandExtractionError(f"frame index {index} is out of sequence (expected {expected})")
        if expected >= n:
            raise HandExtractionError(f"more frames than timestamps ({n})")
        out = detector.detect(frame, int(ts[expected]))
        hands = list(getattr(out, "hand_landmarks", None) or [])
        worlds = list(getattr(out, "hand_world_landmarks", None) or [])
        handed = list(getattr(out, "handedness", None) or [])
        if len(hands) > cfg.num_hands:
            raise HandExtractionError(f"frame {index}: {len(hands)} hands reported, configured {cfg.num_hands}")

        # group detections by label; a label that appears twice cannot be assigned safely
        by_label: dict[str, list[int]] = {}
        for j in range(len(hands)):
            label, _ = _label_and_score(handed[j] if j < len(handed) else [])
            by_label.setdefault(label, []).append(j)

        assigned: dict[str, int] = {}
        ambiguous_sides: set[str] = set()
        for label, idxs in by_label.items():
            side_of = {"Left": "left", "Right": "right"}.get(label)
            if side_of is not None and len(idxs) == 1:
                assigned[side_of] = idxs[0]
            else:
                unassigned[expected] += len(idxs)
                if side_of is not None:
                    ambiguous_sides.add(side_of)

        for side in SIDES:
            if side not in assigned:
                continue
            j = assigned[side]
            lm = _landmarks_to_array(hands[j])
            _, score = _label_and_score(handed[j] if j < len(handed) else [])
            outside = _outside(lm)
            if side == "left":
                left[expected] = lm
                left_score[expected] = score
                left_out[expected] = outside
                left_wa[expected] = j < len(worlds) and bool(worlds[j])
                if left_wa[expected]:
                    left_w[expected] = _landmarks_to_array(worlds[j])
                left_status[expected] = _side_status(score, outside, cfg)
            else:
                right[expected] = lm
                right_score[expected] = score
                right_out[expected] = outside
                right_wa[expected] = j < len(worlds) and bool(worlds[j])
                if right_wa[expected]:
                    right_w[expected] = _landmarks_to_array(worlds[j])
                right_status[expected] = _side_status(score, outside, cfg)

        for side in SIDES:
            if side not in assigned:
                status_list = left_status if side == "left" else right_status
                status_list[expected] = "unassigned" if side in ambiguous_sides else "not_detected"
        seen += 1

    if seen != n:
        raise HandExtractionError(f"{seen} frames were processed but {n} timestamps were given")
    return HandResult(
        left=left, right=right, left_world=left_w, right_world=right_w,
        left_status=left_status, right_status=right_status,
        left_score=left_score, right_score=right_score,
        left_outside=left_out, right_outside=right_out,
        left_world_available=left_wa, right_world_available=right_wa,
        unassigned=unassigned, timestamps_ms=ts, config=cfg.as_dict(),
    )


def _side_status(score: float, outside: int, cfg: HandConfig) -> str:
    if not math.isfinite(score) or score < cfg.low_confidence:
        return "low_confidence"
    if outside > 0:
        return "outside_frame"
    return "valid"


def extract_hands_video(video_path, detector, config: HandConfig | None = None) -> HandResult:
    """Read a video with the shared frame reader and extract its hands. The frames and timestamps are the
    same ones the face and pose extraction use (video_frames), so the streams stay aligned."""
    from extraction.video_frames import frame_timestamps, iter_frames

    ts, _source = frame_timestamps(video_path)
    return extract_hands_from_frames(iter_frames(video_path), ts, detector, config=config)


class HandDetector:
    """Thin wrapper around MediaPipe's HandLandmarker in VIDEO mode."""

    def __init__(self, landmarker, mp_module):
        self._landmarker = landmarker
        self._mp = mp_module

    def detect(self, frame_bgr, timestamp_ms):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, int(timestamp_ms))

    def close(self):
        self._landmarker.close()


def load_hand_detector(models_dir=None, config: HandConfig | None = None) -> HandDetector:
    """Verify the Hand Landmarker's SHA-256, then create the MediaPipe landmarker. Imports MediaPipe last."""
    cfg = config or HandConfig()
    model_path = verify_model(get_spec("hand"), models_dir=models_dir)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(Path(model_path))),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=cfg.num_hands,
        min_hand_detection_confidence=cfg.min_hand_detection_confidence,
        min_hand_presence_confidence=cfg.min_hand_presence_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
    )
    return HandDetector(vision.HandLandmarker.create_from_options(options), mp)
