"""Build the hand stream of an existing pose data session.

The hand stream is written into the same session as the face and body streams (phase H1 decision). The
existing pose session is read, its face and body arrays are carried over unchanged, and the session is
published again through pose_writer, with the hand stream filled in. The writer's staging, validation,
manifest and rollback apply, so a failed hand build leaves the previous pose data in place.

Requires a complete pose data session (schema 2.x). Hand Build never runs the face or pose extractors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from extraction.model_registry import get_spec  # noqa: E402
from extraction.hand_extractor import HandResult  # noqa: E402
from pose_schema import (  # noqa: E402
    ARRAY_SPECS,
    HAND_DETECTED_STATUSES,
    METADATA_FILE,
    OPTIONAL_ARRAY_SPECS,
    TIMELINE_FILE,
    session_layout,
)
from pose_writer import BLENDSHAPE_COLUMNS, PoseDataPayload  # noqa: E402

HAND_LICENCE_NOTE = ("recorded as UNVERIFIED in model_registry.py; recorded metadata only, "
                     "not cleared for commercial use or redistribution")
HAND_COORDINATE_CONVENTIONS = {
    "hand_image_landmarks": "normalised per hand: x / frame width, y / frame height, origin top-left; "
                            "z relative to the wrist on the x scale (MediaPipe convention)",
    "hand_world_landmarks": "metres, per MediaPipe Hand Landmarker documentation (not independently verified); "
                            "NaN and world_available false when the model gave none",
}


class HandBuildError(Exception):
    """The hand stream cannot be combined with the existing pose data session."""


def read_pose_session(session_dir) -> dict:
    """Read a complete pose data session. Raises HandBuildError when it is not one."""
    session = Path(session_dir)
    if session_layout(session) != "pose":
        raise HandBuildError("no complete pose data session here. Run build_pose_session.py first.")
    meta = json.loads((session / METADATA_FILE).read_text(encoding="utf-8"))
    timeline = pd.read_csv(session / TIMELINE_FILE)
    arrays = {}
    for spec in ARRAY_SPECS:
        path = session / spec.relative_path
        if not path.is_file():
            raise HandBuildError(f"pose data is missing {spec.relative_path}")
        arrays[spec.relative_path] = np.load(path, allow_pickle=False)
    for spec in OPTIONAL_ARRAY_SPECS:  # present once the hand stream has been written
        path = session / spec.relative_path
        if path.is_file():
            arrays[spec.relative_path] = np.load(path, allow_pickle=False)
    blend_path = session / "facial" / "blendshapes.csv"
    blend = pd.read_csv(blend_path)
    if list(blend.columns) != list(BLENDSHAPE_COLUMNS):
        raise HandBuildError("facial/blendshapes.csv header does not match the schema")
    quality = json.loads((session / "quality" / "tracking_quality.json").read_text(encoding="utf-8"))
    return {"metadata": meta, "timeline": timeline, "arrays": arrays, "blendshapes": blend, "quality": quality}


def build_hand_payload(session_dir, hands: HandResult) -> PoseDataPayload:
    """Combine the existing pose data session with a HandResult into a payload for write_pose_session."""
    existing = read_pose_session(session_dir)
    n = hands.frames
    timeline = existing["timeline"]
    if len(timeline) != n:
        raise HandBuildError(f"frame counts differ: pose data {len(timeline)}, hands {n}")
    pose_ts = timeline["timestamp_ms"].to_numpy(dtype=np.int64)
    if not np.array_equal(pose_ts, hands.timestamps_ms):
        raise HandBuildError("timestamps differ between the pose data and the hand extraction")

    arrays = {rel: arr for rel, arr in existing["arrays"].items()}
    arrays["hands/left_hand.npy"] = hands.left.astype(np.float32)
    arrays["hands/right_hand.npy"] = hands.right.astype(np.float32)
    arrays["hands/left_hand_world.npy"] = hands.left_world.astype(np.float32)
    arrays["hands/right_hand_world.npy"] = hands.right_world.astype(np.float32)

    timeline_rows = []
    handedness_rows = []
    status_rows = []
    for i in range(n):
        base = timeline.iloc[i]
        ts = int(hands.timestamps_ms[i])
        left_ok = hands.left_status[i] in HAND_DETECTED_STATUSES
        right_ok = hands.right_status[i] in HAND_DETECTED_STATUSES
        timeline_rows.append({
            "frame": i,
            "timestamp_ms": ts,
            "face_detected": bool(base["face_detected"]),
            "pose_detected": bool(base["pose_detected"]),
            "left_hand_detected": bool(left_ok),
            "right_hand_detected": bool(right_ok),
        })
        handedness_rows.append({
            "frame": i, "timestamp_ms": ts,
            "left_detected": bool(left_ok), "left_model_label": "Left" if left_ok else "",
            "left_score": _score(hands.left_score[i], left_ok),
            "right_detected": bool(right_ok), "right_model_label": "Right" if right_ok else "",
            "right_score": _score(hands.right_score[i], right_ok),
        })
        status_rows.append({
            "frame": i, "timestamp_ms": ts,
            "left_status": hands.left_status[i],
            "left_score": _score(hands.left_score[i], left_ok),
            "left_outside_landmarks": int(hands.left_outside[i]) if left_ok else 0,
            "left_world_available": bool(hands.left_world_available[i]),
            "right_status": hands.right_status[i],
            "right_score": _score(hands.right_score[i], right_ok),
            "right_outside_landmarks": int(hands.right_outside[i]) if right_ok else 0,
            "right_world_available": bool(hands.right_world_available[i]),
            "unassigned_hands": int(hands.unassigned[i]),
        })

    detected_frames = sum(1 for s, t in zip(hands.left_status, hands.right_status)
                          if s in HAND_DETECTED_STATUSES or t in HAND_DETECTED_STATUSES)

    blend = existing["blendshapes"]
    quality = dict(existing["quality"])
    quality["hands"] = {
        "status": "complete",
        "frames": n,
        "left_detected_frames": hands.detected_frames("left"),
        "right_detected_frames": hands.detected_frames("right"),
        "frames_with_any_hand": detected_frames,
        "status_counts": hands.status_counts(),
        "world_available_frames": {
            "left": int(np.count_nonzero(hands.left_world_available)),
            "right": int(np.count_nonzero(hands.right_world_available)),
        },
        "scope": "tracking only; hand landmarks and handedness labels, not gestures or accuracy",
    }

    hand_spec = get_spec("hand")
    metadata = json.loads(json.dumps(existing["metadata"]))
    metadata.setdefault("models", {})["hand"] = {
        "file": hand_spec.filename,
        "sha256": hand_spec.sha256.lower(),
        "version": hand_spec.version,
        "source": hand_spec.source_url,
        "licence_status": "to verify",
        "licence_note": HAND_LICENCE_NOTE,
    }
    metadata.setdefault("configuration", {})["hands"] = dict(hands.config)
    metadata.setdefault("coordinate_conventions", {}).update(HAND_COORDINATE_CONVENTIONS)
    metadata.setdefault("streams", {})["hands"] = {"status": "complete", "detected_frames": detected_frames}

    return PoseDataPayload(
        arrays=arrays,
        timeline_rows=timeline_rows,
        handedness_rows=handedness_rows,
        blendshapes=blend,
        quality=quality,
        metadata=metadata,
        hand_status_rows=status_rows,
    )


def _score(value, detected: bool):
    if not detected or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)
