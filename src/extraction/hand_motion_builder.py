"""Hand tracking and motion for a pose data session that already has the hand stream (phase H2).

Reads the complete session (every stream, including the H1 hand arrays and hand_status.csv), runs the
tracker and the motion metrics on the H1 landmarks, and returns a payload for pose_writer. The H1 and
face/body streams are carried over unchanged. The session is republished through the same writer, so
staging, validation, the manifest and rollback apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from hand_tracking import (  # noqa: E402
    AMBIGUITY_RATIO,
    CROSSING_HAND_SIZES,
    GATE_HAND_SIZES,
    LABEL_PENALTY,
    MAX_GAP_FRAMES,
    MIN_HAND_SIZE_PX,
    NEW_TRACK_COST,
    PARTIAL_LANDMARK_LIMIT,
    make_detection,
    motion_rows,
    quality_summary,
    track_hands,
)
from pose_schema import (  # noqa: E402
    HAND_MOTION_COLUMNS,
    HAND_MOTION_FILE,
    HAND_STATUS_FILE,
    HAND_TRACK_COLUMNS,
    HAND_TRACKS_FILE,
)
from pose_writer import PoseDataPayload  # noqa: E402
from extraction.hand_builder import HandBuildError, read_pose_session  # noqa: E402

TRACKING_CONVENTIONS = {
    "hand_motion": ("image pixels: normalised x * width, y * height (not physical units). "
                    "Velocity in pixels per second, acceleration in pixels per second squared, angles in degrees. "
                    "No smoothing. See analysis/hand_metrics.py and analysis/hand_tracking.py."),
}


def tracking_config() -> dict:
    return {
        "algorithm": "per-frame exhaustive assignment on centroid distance with a label penalty; see hand_tracking.py",
        "max_gap_frames": MAX_GAP_FRAMES,
        "gate_hand_sizes": GATE_HAND_SIZES,
        "label_penalty": LABEL_PENALTY,
        "new_track_cost": NEW_TRACK_COST,
        "ambiguity_ratio": AMBIGUITY_RATIO,
        "crossing_hand_sizes": CROSSING_HAND_SIZES,
        "partial_landmark_limit": PARTIAL_LANDMARK_LIMIT,
        "min_hand_size_px": MIN_HAND_SIZE_PX,
        "usable_statuses": ["valid", "outside_frame"],
    }


def carry_payload(session_dir) -> PoseDataPayload:
    """Every stream of an existing complete pose data session, as a payload. Nothing is changed."""
    existing = read_pose_session(session_dir)
    session = Path(session_dir)
    status_path = session / HAND_STATUS_FILE
    if not status_path.is_file():
        raise HandBuildError("the hand status file is missing. Run build_hand_session.py first.")
    status = pd.read_csv(status_path)
    handedness = pd.read_csv(session / "hands" / "handedness.csv")
    return PoseDataPayload(
        arrays=dict(existing["arrays"]),
        timeline_rows=existing["timeline"].to_dict("records"),
        handedness_rows=handedness.to_dict("records"),
        blendshapes=existing["blendshapes"],
        quality=dict(existing["quality"]),
        metadata=existing["metadata"],
        hand_status_rows=status.to_dict("records"),
    )


def analyse_hand_payload(session_dir) -> PoseDataPayload:
    """The session's payload with the tracks and motion tables added. Raises TrackingError or HandBuildError."""
    payload = carry_payload(session_dir)
    meta = payload.metadata
    width = int(meta["source"]["width"])
    height = int(meta["source"]["height"])
    timeline = payload.timeline_rows
    frames = [int(r["frame"]) for r in timeline]
    timestamps = {int(r["frame"]): int(r["timestamp_ms"]) for r in timeline}
    status_rows = {int(r["frame"]): r for r in payload.hand_status_rows}

    detections: dict[int, list] = {}
    statuses: dict[tuple[str, int], str] = {}
    for f in frames:
        row = status_rows[f]
        detections[f] = []
        for side, label in (("left", "Left"), ("right", "Right")):
            st = str(row[f"{side}_status"])
            statuses[(label, f)] = st
            if st in ("not_detected", "unassigned", "not_processed"):
                continue
            norm = payload.arrays[f"hands/{side}_hand.npy"][f]
            score = row[f"{side}_score"]
            detections[f].append(make_detection(
                f, timestamps[f], label, st, float(score) if score == score else float("nan"),
                norm, width, height))

    events, observations = track_hands(detections, frames, timestamps, statuses)
    motion = motion_rows(observations)
    q = quality_summary(events, observations, motion, frames, timestamps, statuses)

    track_rows = [{c: e.get(c, "") for c in HAND_TRACK_COLUMNS} for e in events]
    motion_out = [{c: r[c] for c in HAND_MOTION_COLUMNS} for r in motion]
    payload.hand_tables = {HAND_TRACKS_FILE: track_rows, HAND_MOTION_FILE: motion_out}

    payload.quality.setdefault("hands", {})["tracking"] = q
    meta.setdefault("configuration", {}).setdefault("hands", {})["tracking"] = tracking_config()
    meta.setdefault("coordinate_conventions", {}).update(TRACKING_CONVENTIONS)
    meta.setdefault("streams", {}).setdefault("hands", {})["tracking"] = {
        "status": "complete", "tracks": q["tracks_created"], "observations": q["observations"],
    }
    payload.metadata = meta
    return payload
