"""Gesture recognition for a pose data session that has hand tracking (phase H3).

Classifies each linked observation from the H2 tracking (hands/motion_metrics.csv) using the H1 landmarks
for the same side and frame. Builds stable gesture events per track, and publishes the observations, the
events and a summary through pose_writer. Earlier streams are carried over unchanged.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import gesture_rules as gr  # noqa: E402
from hand_metrics import to_pixels  # noqa: E402
from pose_schema import (  # noqa: E402
    HAND_GESTURE_EVENTS_FILE,
    HAND_GESTURE_EVENT_COLUMNS,
    HAND_GESTURE_OBS_COLUMNS,
    HAND_GESTURE_OBS_FILE,
    HAND_MOTION_FILE,
    HAND_TRACKS_FILE,
)
from pose_writer import PoseDataPayload  # noqa: E402
from extraction.hand_motion_builder import carry_payload  # noqa: E402

GESTURE_CONVENTIONS = {
    "hand_gesture": ("rule-based, hand-relative (palm length = wrist to middle MCP); image plane; "
                     "the geometric margin is not a probability and is not calibrated. See analysis/gesture_rules.py."),
}


class GestureInputError(Exception):
    """The session does not have the tracking this stage needs."""


def gesture_config() -> dict:
    return {
        "classes": list(gr.GESTURES),
        "reach_extended": gr.REACH_EXTENDED,
        "reach_curled": gr.REACH_CURLED,
        "thumb_reach_extended": gr.THUMB_REACH_EXTENDED,
        "thumb_reach_curled": gr.THUMB_REACH_CURLED,
        "angle_extended_deg": gr.ANGLE_EXTENDED,
        "min_support": gr.MIN_SUPPORT,
        "max_bridge": gr.MAX_BRIDGE,
        "min_palm_px": gr.MIN_PALM_PX,
        "calibration": "none: thresholds set from synthetic geometry and one template hand; not validated on real footage",
    }


def analyse_gesture_payload(session_dir) -> PoseDataPayload:
    payload = carry_payload(session_dir)
    meta = payload.metadata
    width = int(meta["source"]["width"])
    height = int(meta["source"]["height"])
    tables = payload.hand_tables or {}
    motion = tables.get(HAND_MOTION_FILE)
    if not motion or HAND_TRACKS_FILE not in tables:
        raise GestureInputError("no hand tracking in this session. Run analyse_hand_session.py first.")

    observations = []
    for r in motion:
        side = "left" if str(r["label"]) == "Left" else "right"
        frame = int(r["frame"])
        norm = payload.arrays[f"hands/{side}_hand.npy"][frame]
        observations.append(gr.HandObservation(
            track_id=int(r["track_id"]), frame=frame, timestamp_ms=int(r["timestamp_ms"]),
            label=str(r["label"]), status=str(r["status"]),
            points=to_pixels(norm, width, height), partial=bool(int(r["partial_view"]))))

    rows = gr.classify_observations(observations)
    events, rows = gr.build_events(rows)
    issues = gr.validate_events(events)
    if issues:
        raise gr.GestureError("; ".join(issues))

    payload.hand_tables[HAND_GESTURE_OBS_FILE] = [{c: r[c] for c in HAND_GESTURE_OBS_COLUMNS} for r in rows]
    payload.hand_tables[HAND_GESTURE_EVENTS_FILE] = [{c: e[c] for c in HAND_GESTURE_EVENT_COLUMNS} for e in events]

    payload.quality.setdefault("hands", {})["gestures"] = _summary(rows, events, tables)
    meta.setdefault("configuration", {}).setdefault("hands", {})["gestures"] = gesture_config()
    meta.setdefault("coordinate_conventions", {}).update(GESTURE_CONVENTIONS)
    meta.setdefault("streams", {}).setdefault("hands", {})["gestures"] = {
        "status": "complete", "events": len(events), "observations": len(rows),
    }
    payload.metadata = meta
    return payload


def _summary(rows, events, tables) -> dict:
    frame_classes = Counter(r["frame_class"] for r in rows)
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    quality = Counter(r["quality"] for r in rows)
    event_counts = Counter(e["gesture"] for e in events)
    durations = [e["duration_ms"] for e in events]
    # observations whose track link was flagged as uncertain in H2
    uncertain_links = {(int(t["track_id"]), int(t["frame"])) for t in tables[HAND_TRACKS_FILE]
                       if isinstance(t.get("uncertain_reason"), str) and t.get("uncertain_reason")}
    uncertain_obs = sum(1 for r in rows if (r["track_id"], r["frame"]) in uncertain_links)
    return {
        "observations": len(rows),
        "frame_classes": {g: int(frame_classes.get(g, 0)) for g in gr.GESTURES},
        "unknown_reasons": dict(reasons),
        "observation_quality": dict(quality),
        "uncertain_tracking_observations": uncertain_obs,
        "events": len(events),
        "events_per_gesture": {g: int(event_counts.get(g, 0)) for g in gr.GESTURES if g != "unknown"},
        "duration_ms": {
            "min": min(durations) if durations else None,
            "median": float(np.median(durations)) if durations else None,
            "max": max(durations) if durations else None,
        },
        "end_reasons": dict(Counter(e["end_reason"] for e in events)),
        "bridged_observations": int(sum(e["bridged_observations"] for e in events)),
        "scope": "rule-based classification of linked hand observations; not validated on real footage",
    }
