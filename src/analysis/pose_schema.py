"""Pose data dataset schema: constants, array specifications and validation.

Pose data adds body and hand streams beside the face data facial outputs, in one session folder:

    datasets/<person>/<session>/
        metadata.json          completion marker, written last
        manifest.json          files created by the pose data operation (for safe rollback)
        timeline.csv           frame index, timestamp and detection flags for every frame
        facial/   landmarks.npy, blendshapes.csv, transformation_matrices.npy
        body/     pose_landmarks.npy, pose_world_landmarks.npy
        hands/    left_hand.npy, right_hand.npy, handedness.csv
        quality/  tracking_quality.json

This module only describes and checks the schema. It does not read or write datasets.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = "2.0"
SCHEMA_MAJOR = "2"
METADATA_FILE = "metadata.json"        # written last; its presence marks a complete pose data session
MANIFEST_FILE = "manifest.json"        # lists every file created by the pose data operation
TIMELINE_FILE = "timeline.csv"
STAGING_SUFFIX = ".staging"            # pose data output is built here before publication
PREVIOUS_SUFFIX = ".previous"          # the previous pose data output, kept for rollback

N_FACE_LANDMARKS = 478
N_POSE_LANDMARKS = 33
N_HAND_LANDMARKS = 21
N_BLENDSHAPES = 52

# MediaPipe Face Landmarker blendshape categories, in the order used by the face data CSV.
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)

POSE_CHANNELS = ("x", "y", "z", "visibility", "presence")
HAND_CHANNELS = ("x", "y", "z")

TIMESTAMP_SOURCES = ("decoded_pts", "fps_derived")
STREAM_STATUSES = ("complete", "partial", "absent", "failed")
STREAMS = ("facial", "body", "hands")

HANDEDNESS_COLUMNS = (
    "frame", "timestamp_ms",
    "left_detected", "left_model_label", "left_score",
    "right_detected", "right_model_label", "right_score",
)

REQUIRED_METADATA_KEYS = (
    "schema_version", "identity", "source", "timeline", "extraction", "models",
    "configuration", "coordinate_conventions", "streams", "outputs",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ArraySpec:
    """Expected dtype and trailing shape of one .npy file. The first axis is always the frame count T."""
    relative_path: str
    tail_shape: tuple
    required: bool = True
    dtype: str = "float32"


ARRAY_SPECS = (
    ArraySpec("facial/landmarks.npy", (N_FACE_LANDMARKS, 3)),
    ArraySpec("facial/transformation_matrices.npy", (4, 4)),
    ArraySpec("body/pose_landmarks.npy", (N_POSE_LANDMARKS, len(POSE_CHANNELS))),
    ArraySpec("body/pose_world_landmarks.npy", (N_POSE_LANDMARKS, len(POSE_CHANNELS))),
    ArraySpec("hands/left_hand.npy", (N_HAND_LANDMARKS, len(HAND_CHANNELS))),
    ArraySpec("hands/right_hand.npy", (N_HAND_LANDMARKS, len(HAND_CHANNELS))),
)

# Optional when the model provides world coordinates for hands.
OPTIONAL_ARRAY_SPECS = (
    ArraySpec("hands/left_hand_world.npy", (N_HAND_LANDMARKS, len(HAND_CHANNELS)), required=False),
    ArraySpec("hands/right_hand_world.npy", (N_HAND_LANDMARKS, len(HAND_CHANNELS)), required=False),
)


def frame_timestamps_ms(n_frames: int, fps: float | None = None,
                        pts_seconds: list[float] | None = None) -> tuple[np.ndarray, str]:
    """Timestamps in integer milliseconds from the start of the video, and the source used.

    Decoded presentation timestamps are preferred when they are available, match the frame count
    and strictly increase. Otherwise the timestamp is derived from the frame rate using the same rule
    as face data extraction: round(i * 1000 / fps). The source is always recorded.
    """
    if n_frames < 0:
        raise ValueError("n_frames must not be negative")
    if pts_seconds is not None:
        pts = np.asarray(pts_seconds, dtype=float)
        if len(pts) == n_frames and np.all(np.isfinite(pts)) and np.all(np.diff(pts) > 0):
            return np.round(pts * 1000.0).astype(np.int64), "decoded_pts"
    if fps is None or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps is required when decoded timestamps are unavailable")
    idx = np.arange(n_frames, dtype=np.int64)
    return np.round(idx * 1000.0 / fps).astype(np.int64), "fps_derived"


def validate_array(relative_path: str, array, n_frames: int, tail_shape: tuple, dtype: str = "float32") -> list[str]:
    issues = []
    if not isinstance(array, np.ndarray):
        return [f"{relative_path}: not a numpy array"]
    if array.dtype != np.dtype(dtype):
        issues.append(f"{relative_path}: dtype {array.dtype}, expected {dtype}")
    expected = (n_frames,) + tuple(tail_shape)
    if array.shape != expected:
        issues.append(f"{relative_path}: shape {array.shape}, expected {expected}")
    return issues


def validate_arrays(arrays: dict, n_frames: int) -> list[str]:
    """arrays maps relative paths (for example 'facial/landmarks.npy') to numpy arrays."""
    issues = []
    for spec in ARRAY_SPECS:
        if spec.relative_path not in arrays:
            issues.append(f"missing required array {spec.relative_path}")
            continue
        issues.extend(validate_array(spec.relative_path, arrays[spec.relative_path], n_frames,
                                     spec.tail_shape, spec.dtype))
    for spec in OPTIONAL_ARRAY_SPECS:
        if spec.relative_path in arrays:
            issues.extend(validate_array(spec.relative_path, arrays[spec.relative_path], n_frames,
                                         spec.tail_shape, spec.dtype))
    return issues


def validate_timeline_rows(rows: list[dict], n_frames: int) -> list[str]:
    """rows: one dict per frame with frame, timestamp_ms and the detection flags."""
    issues = []
    if len(rows) != n_frames:
        issues.append(f"timeline has {len(rows)} rows, expected {n_frames}")
        return issues
    previous = None
    for i, row in enumerate(rows):
        if row.get("frame") != i:
            issues.append(f"timeline row {i} has frame {row.get('frame')}")
            break
        ts = row.get("timestamp_ms")
        if ts is None or (isinstance(ts, float) and not math.isfinite(ts)):
            issues.append(f"timeline row {i} has no timestamp")
            break
        if previous is not None and ts <= previous:
            issues.append(f"timestamps do not increase at frame {i}")
            break
        previous = ts
    return issues


def validate_handedness_rows(rows: list[dict]) -> list[str]:
    issues = []
    for i, row in enumerate(rows):
        for side in ("left", "right"):
            score = row.get(f"{side}_score")
            if row.get(f"{side}_detected"):
                if score is None or not (0.0 <= float(score) <= 1.0):
                    issues.append(f"frame {i}: {side} hand score missing or outside 0..1")
            elif score not in (None, "") and not (isinstance(score, float) and math.isnan(score)):
                issues.append(f"frame {i}: {side} hand score present although the hand was not detected")
    return issues


def validate_metadata(meta: dict) -> list[str]:
    issues = []
    for key in REQUIRED_METADATA_KEYS:
        if key not in meta:
            issues.append(f"metadata missing '{key}'")
    version = str(meta.get("schema_version", ""))
    if not version.startswith(SCHEMA_MAJOR + "."):
        issues.append(f"schema_version '{version}' is not version {SCHEMA_MAJOR}.x")

    timeline = meta.get("timeline", {})
    if timeline.get("timestamp_unit") != "milliseconds":
        issues.append("timeline.timestamp_unit must be 'milliseconds'")
    if timeline.get("timestamp_source") not in TIMESTAMP_SOURCES:
        issues.append(f"timeline.timestamp_source must be one of {TIMESTAMP_SOURCES}")
    if timeline.get("frame_index_base") != 0:
        issues.append("timeline.frame_index_base must be 0")

    for name, model in meta.get("models", {}).items():
        digest = model.get("sha256", "")
        if not _SHA256.fullmatch(str(digest)):
            issues.append(f"model '{name}' has no valid sha256")
        if model.get("licence_status") not in ("verified", "to verify"):
            issues.append(f"model '{name}' licence_status must be 'verified' or 'to verify'")

    config = meta.get("configuration", {})
    if config.get("pose", {}).get("num_poses", 1) != 1:
        issues.append("pose num_poses must be 1 in schema 2.0 (single person per frame)")

    streams = meta.get("streams", {})
    for stream in STREAMS:
        status = streams.get(stream, {}).get("status")
        if status not in STREAM_STATUSES:
            issues.append(f"stream '{stream}' status must be one of {STREAM_STATUSES}")
    return issues


def validate_manifest(manifest: dict) -> list[str]:
    issues = []
    files = manifest.get("files")
    if not isinstance(files, list):
        return ["manifest has no 'files' list"]
    seen = set()
    for entry in files:
        path = entry.get("path", "")
        if not path or path.startswith(("/", "\\")) or ".." in path.replace("\\", "/").split("/"):
            issues.append(f"manifest path is not a safe relative path: '{path}'")
        if path in seen:
            issues.append(f"manifest lists '{path}' twice")
        seen.add(path)
        if not _SHA256.fullmatch(str(entry.get("sha256", ""))):
            issues.append(f"manifest entry '{path}' has no valid sha256")
    return issues


def session_layout(session_dir) -> str:
    """'pose' when metadata.json declares schema 2.x, 'face' when only the face data outputs exist,
    'pose_incomplete' when pose data files exist without the metadata.json completion marker,
    'none' otherwise. Readers must treat 'pose_incomplete' as absent."""
    from pathlib import Path

    folder = Path(session_dir)
    meta_path = folder / METADATA_FILE
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if str(meta.get("schema_version", "")).startswith(SCHEMA_MAJOR + "."):
                return "pose"
        except (OSError, json.JSONDecodeError):
            return "pose_incomplete"
    pose_markers = [folder / "facial", folder / "body", folder / "hands", folder / "quality", folder / MANIFEST_FILE]
    if any(p.exists() for p in pose_markers):
        return "pose_incomplete"
    if (folder / "facial_data.json").is_file():
        return "face"
    return "none"
