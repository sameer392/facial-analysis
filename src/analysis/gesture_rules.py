"""Rule-based hand gesture recognition (phase H3). Pure functions. Independent of detection and tracking.

Input: one observation per frame and hand, as landmark positions in image pixels (21, 2) plus the H1 status.
Output: a class per observation, a stable gesture event per run of supporting observations, and quality data.

Classes
    open_palm      all five fingers extended
    closed_fist    four fingers curled, thumb curled
    pointing       index extended; middle, ring, pinky curled; thumb curled
    thumbs_up      thumb extended; index, middle, ring, pinky curled
    peace          index and middle extended; ring, pinky curled; thumb curled
    unknown        no rule matched, or the landmarks were not good enough to decide

Geometry (hand-relative; no fixed image coordinates)
    palm length P   distance from the wrist to the middle-finger MCP, in pixels
    reach           distance from the wrist to the fingertip, divided by P (dimensionless)
    joint angle     angle in the image plane at the PIP and DIP joints (degrees), from hand_metrics.finger_angles
    The image plane is used, so the rules see the hand's pose as it appears in the frame. Rotation of the
    hand in the image plane changes no reach and no joint angle. Left and right are not used, so a mirrored
    video gives the same classes.

Finger states (per finger)
    extended   reach >= REACH_EXTENDED (thumb: THUMB_REACH_EXTENDED) and mean joint angle >= ANGLE_EXTENDED
    curled     reach <= REACH_CURLED (thumb: THUMB_REACH_CURLED)
    unknown    anything in between. A rule that needs an unknown finger does not match.

Thresholds were set from the synthetic hand geometry used in the tests and from one template hand. They
have not been calibrated on real footage. Change them only with a measured reason, and record it.

Margin
    A geometric margin is the distance by which the decisive conditions pass their thresholds, in reach units
    or degrees. It is not a probability and it is not a calibrated confidence. Do not report it as one.

Temporal rules (per track, in frame order)
    A gesture event needs at least MIN_SUPPORT supporting observations of one class.
    Unknown observations inside a run are bridged for up to MAX_BRIDGE consecutive frames. Longer unknown
    stretches end the run.
    A different determinate class ends the run, and a new run starts. Nothing is merged across a class change.
    A missing frame (a gap in frame index) ends the run. Missing frames are never bridged.
    Two runs of the same class separated by a bridged interruption are one event. Runs separated by a longer
    unknown stretch or a gap are separate events, even when they are close in time.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from hand_metrics import finger_angles, fingertips, wrist  # noqa: E402

GESTURES = ("open_palm", "closed_fist", "pointing", "thumbs_up", "peace", "unknown")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
STATES = ("extended", "curled", "unknown")

REACH_EXTENDED = 1.6
REACH_CURLED = 1.2
THUMB_REACH_EXTENDED = 1.2
THUMB_REACH_CURLED = 0.9
ANGLE_EXTENDED = 150.0
MIN_SUPPORT = 5          # supporting observations for a gesture event
MAX_BRIDGE = 2           # consecutive unknown observations tolerated inside a run
MIN_PALM_PX = 4.0        # palm length below this is insufficient

from pose_schema import HAND_GESTURE_EVENT_COLUMNS as EVENT_COLUMNS  # noqa: E402
from pose_schema import HAND_GESTURE_OBS_COLUMNS as OBS_COLUMNS  # noqa: E402


class GestureError(Exception):
    """The observations cannot be classified correctly (for example, timestamps that do not increase)."""


@dataclass
class HandObservation:
    track_id: int
    frame: int
    timestamp_ms: int
    label: str
    status: str                 # H1 status: valid, outside_frame, low_confidence, ...
    points: np.ndarray          # (21, 2) image pixels
    partial: bool = False       # some landmarks outside the frame


@dataclass
class FrameClassification:
    gesture: str
    reason: str                 # "" for a decided class; otherwise why it is unknown
    quality: str                # ok, partial_view, low_confidence, insufficient_landmarks
    states: dict = field(default_factory=dict)
    margin: float = float("nan")


def _reach(points: np.ndarray, finger: str) -> float:
    palm = float(np.hypot(*(points[9] - points[0])))
    tip = fingertips(points)[FINGERS.index(finger) if finger != "thumb" else 0]
    return float(np.hypot(*(tip - wrist(points)))) / palm


def finger_state(points: np.ndarray, finger: str) -> tuple[str, float]:
    """(state, margin) for one finger. Margin is the distance from the nearest threshold, in reach units."""
    angles = finger_angles(points)
    if finger == "thumb":
        reach = _reach(points, "thumb")
        mean_angle = float(np.mean([angles["thumb_mcp_angle_deg"], angles["thumb_ip_angle_deg"]]))
        if reach >= THUMB_REACH_EXTENDED and mean_angle >= ANGLE_EXTENDED:
            return "extended", min(reach - THUMB_REACH_EXTENDED, (mean_angle - ANGLE_EXTENDED) / 100.0)
        if reach <= THUMB_REACH_CURLED:
            return "curled", THUMB_REACH_CURLED - reach
        return "unknown", 0.0
    reach = _reach(points, finger)
    pip, dip = {
        "index": ("index_pip_angle_deg", "index_dip_angle_deg"),
        "middle": ("middle_pip_angle_deg", "middle_dip_angle_deg"),
        "ring": ("ring_pip_angle_deg", "ring_dip_angle_deg"),
        "pinky": ("pinky_pip_angle_deg", "pinky_dip_angle_deg"),
    }[finger]
    mean_angle = float(np.mean([angles[pip], angles[dip]]))
    if reach >= REACH_EXTENDED and mean_angle >= ANGLE_EXTENDED:
        return "extended", min(reach - REACH_EXTENDED, (mean_angle - ANGLE_EXTENDED) / 100.0)
    if reach <= REACH_CURLED:
        return "curled", REACH_CURLED - reach
    return "unknown", 0.0


RULES = {
    "open_palm": {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"},
    "closed_fist": {"thumb": "curled", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"},
    "pointing": {"thumb": "curled", "index": "extended", "middle": "curled", "ring": "curled", "pinky": "curled"},
    "thumbs_up": {"thumb": "extended", "index": "curled", "middle": "curled", "ring": "curled", "pinky": "curled"},
    "peace": {"thumb": "curled", "index": "extended", "middle": "extended", "ring": "curled", "pinky": "curled"},
}


def classify_points(points: np.ndarray, status: str = "valid", partial: bool = False) -> FrameClassification:
    """Classify one hand from its landmarks. Never raises for bad landmarks; returns unknown with a reason."""
    if status not in ("valid", "outside_frame"):
        return FrameClassification("unknown", status, "low_confidence" if status == "low_confidence" else "insufficient_landmarks")
    pts = np.asarray(points, dtype=float)
    if pts.shape != (21, 2) or not np.all(np.isfinite(pts)):
        return FrameClassification("unknown", "insufficient_landmarks", "insufficient_landmarks")
    palm = float(np.hypot(*(pts[9] - pts[0])))
    if palm < MIN_PALM_PX:
        return FrameClassification("unknown", "insufficient_landmarks", "insufficient_landmarks")

    quality = "partial_view" if partial else "ok"
    states, margins = {}, {}
    for finger in FINGERS:
        states[finger], margins[finger] = finger_state(pts, finger)
    if any(s == "unknown" for s in states.values()):
        return FrameClassification("unknown", "indeterminate_finger", quality, states, float("nan"))

    matches = [g for g, rule in RULES.items() if all(states[f] == s for f, s in rule.items())]
    if len(matches) != 1:
        return FrameClassification("unknown", "no_rule_matched", quality, states, float("nan"))
    gesture = matches[0]
    margin = min(margins[f] for f in RULES[gesture])
    return FrameClassification(gesture, "", quality, states, float(margin))


def classify_observations(observations: list[HandObservation]) -> list[dict]:
    """Frame-level classification for each observation. Input is sorted by (track_id, frame) internally."""
    rows = []
    for ob in sorted(observations, key=lambda o: (o.track_id, o.frame)):
        c = classify_points(ob.points, ob.status, ob.partial)
        rows.append({
            "track_id": ob.track_id, "frame": ob.frame, "timestamp_ms": ob.timestamp_ms, "label": ob.label,
            "frame_class": c.gesture, "reason": c.reason, "quality": c.quality,
            "thumb_state": c.states.get("thumb", ""), "index_state": c.states.get("index", ""),
            "middle_state": c.states.get("middle", ""), "ring_state": c.states.get("ring", ""),
            "pinky_state": c.states.get("pinky", ""), "margin": c.margin, "event_id": "",
        })
    return rows


def build_events(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stable gesture events from frame classes, per track. Returns (events, rows annotated with event_id).

    Per track, in frame order. A run starts at the first determinate class. Observations that differ from the
    run's gesture (unknown or another class) are held as pending. When the run's gesture returns within
    MAX_BRIDGE pending observations, they are part of the same event. When more than MAX_BRIDGE observations
    are pending, the run ends. If the current observation is a determinate class, a new run starts from it.
    A missing frame ends the run at once. Events need MIN_SUPPORT supporting observations.
    """
    events: list[dict] = []
    by_track: dict[int, list[dict]] = {}
    for r in rows:
        by_track.setdefault(r["track_id"], []).append(r)

    for tid, seq in by_track.items():
        seq.sort(key=lambda r: r["frame"])
        for a, b in zip(seq, seq[1:]):
            if b["timestamp_ms"] <= a["timestamp_ms"]:
                raise GestureError(f"track {tid}: timestamps do not increase at frame {b['frame']}")
            if b["frame"] <= a["frame"]:
                raise GestureError(f"track {tid}: frame indices do not increase at frame {b['frame']}")

        state = {"run": None}

        def close(reason, _tid=tid, _state=state):
            run = _state["run"]
            _state["run"] = None
            if run is not None and len(run["support"]) >= MIN_SUPPORT:
                events.append(_event(len(events) + 1, _tid, run, reason))

        def start(r, _state=state):
            _state["run"] = {"gesture": r["frame_class"], "support": [r], "members": [r],
                             "pending": [], "bridged_total": 0}

        prev = None
        for r in seq:
            if prev is not None and r["frame"] - prev["frame"] - 1 > 0:
                close("gap")
            run = state["run"]
            cls = r["frame_class"]
            if run is None:
                if cls != "unknown":
                    start(r)
            elif cls == run["gesture"]:
                run["members"].extend(run["pending"])
                run["bridged_total"] += len(run["pending"])
                run["pending"] = []
                run["support"].append(r)
                run["members"].append(r)
            else:
                run["pending"].append(r)
                if len(run["pending"]) > MAX_BRIDGE:
                    determinate = any(p["frame_class"] != "unknown" for p in run["pending"])
                    candidates = list(run["pending"])  # already includes r
                    k = len(candidates)
                    while k > 0 and candidates[k - 1]["frame_class"] == cls and cls != "unknown":
                        k -= 1
                    close("class_changed" if determinate else "unknown_run")
                    if cls != "unknown":
                        # the new gesture starts at the first trailing frame that already has its class
                        state["run"] = {"gesture": cls, "support": candidates[k:], "members": candidates[k:],
                                        "pending": [], "bridged_total": 0}
            prev = r
        close("end_of_track")

    index = {}
    for ev in events:
        for f in ev.pop("_frames"):
            index[(ev["track_id"], f)] = ev["event_id"]
    for r in rows:
        r["event_id"] = index.get((r["track_id"], r["frame"]), "")
    return events, rows


def _event(event_id, track_id, run, reason):
    support = run["support"]
    start, end = support[0], support[-1]
    margins = [r["margin"] for r in support if math.isfinite(r["margin"])]
    ok = sum(1 for r in support if r["quality"] == "ok")
    return {
        "event_id": f"E{event_id:04d}",
        "track_id": track_id,
        "label": start["label"],
        "gesture": run["gesture"],
        "start_frame": start["frame"],
        "end_frame": end["frame"],
        "start_timestamp_ms": start["timestamp_ms"],
        "end_timestamp_ms": end["timestamp_ms"],
        "duration_ms": end["timestamp_ms"] - start["timestamp_ms"],
        "supporting_observations": len(support),
        "bridged_observations": run["bridged_total"],
        "support_quality": ok / len(support),
        "margin_median": float(np.median(margins)) if margins else float("nan"),
        "end_reason": reason,
        "_frames": [r["frame"] for r in run["members"]],
    }


def validate_events(events: list[dict]) -> list[str]:
    """Consistency checks for gesture events (used before publication and in tests)."""
    issues = []
    ids = set()
    for ev in events:
        if ev["event_id"] in ids:
            issues.append(f"duplicate event id {ev['event_id']}")
        ids.add(ev["event_id"])
        if ev["gesture"] not in GESTURES or ev["gesture"] == "unknown":
            issues.append(f"{ev['event_id']}: gesture {ev['gesture']!r} is not a stable class")
        if ev["end_frame"] < ev["start_frame"]:
            issues.append(f"{ev['event_id']}: ends before it starts")
        if ev["duration_ms"] != ev["end_timestamp_ms"] - ev["start_timestamp_ms"] or ev["duration_ms"] < 0:
            issues.append(f"{ev['event_id']}: duration does not match its timestamps")
        if ev["supporting_observations"] < MIN_SUPPORT:
            issues.append(f"{ev['event_id']}: fewer than {MIN_SUPPORT} supporting observations")
        if not (0.0 <= ev["support_quality"] <= 1.0):
            issues.append(f"{ev['event_id']}: support_quality outside 0..1")
    return issues
