"""Hand tracking across frames and per-track motion (phase H2).

Input: the H1 hand stream (per-frame status and normalised landmarks for the left and right side).
Output: track events (hands/tracks.csv), per-observation motion (hands/motion_metrics.csv) and a quality summary.

Algorithm (per frame, in frame order)
    1. Usable detections: status "valid" or "outside_frame", with every landmark finite and at most
       PARTIAL_LANDMARK_LIMIT landmarks outside the frame. "low_confidence" detections are not used
       for tracking. Detections that fail the landmark checks are treated as insufficient.
    2. Predict each active track's position for this frame: last centroid plus the last per-frame
       velocity times the number of frames since the last observation.
    3. Cost of linking a detection to a track: centroid distance from the prediction, in units of the
       track's hand size (wrist to middle-finger MCP distance), plus LABEL_PENALTY when the handedness label
       differs from the track's label. A link is allowed only within a gate of GATE_HAND_SIZES, scaled by
       the number of missing frames + 1. Starting a new track costs NEW_TRACK_COST.
    4. Choose the set of links with the lowest total cost (exhaustive; at most two detections per frame).
    5. Uncertain associations are flagged, not hidden:
         label_conflict         the linked track had a different label
         ambiguous_candidates   a second option costs at most AMBIGUITY_RATIO times the chosen one
         hands_close            two usable detections in this frame are closer than CROSSING_HAND_SIZES
    6. A track that is not linked in a frame is "interrupted" from the first missing frame. It stays
       active for up to MAX_GAP_FRAMES missing frames, and is then "terminated". A track that is linked
       again is "reacquired". Tracks still active at the end of the recording are terminated with
       reason "end_of_recording".

Limits
    Handedness labels come from the model and are not reliable under occlusion or when hands cross, so the
    label penalty is a preference, not proof of identity. The tracker has no appearance model. Two hands that
    cross closely, or a hand that leaves and returns in another place, can be linked to the wrong track.
    Those cases are flagged as uncertain, but the flag does not prove the link is correct.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from pose_schema import HAND_MOTION_COLUMNS, HAND_TRACK_COLUMNS, HAND_TRACK_EVENTS, HAND_FINGER_NAMES  # noqa: E402
from hand_metrics import (
    acceleration,
    centroid,
    direction_deg,
    displacement,
    finger_angles,
    fingertips,
    openness,
    orientation_deg,
    velocity,
    wrap_angle_deg,
    wrist,
)

USABLE_STATUSES = ("valid", "outside_frame")
PARTIAL_LANDMARK_LIMIT = 6        # more landmarks outside the frame than this: insufficient
MIN_HAND_SIZE_PX = 4.0            # wrist to middle MCP shorter than this (pixels): insufficient
MAX_GAP_FRAMES = 5
GATE_HAND_SIZES = 2.0
LABEL_PENALTY = 1.0
NEW_TRACK_COST = 3.0
AMBIGUITY_RATIO = 1.25
CROSSING_HAND_SIZES = 1.0
SIDE_LABEL = {"left": "Left", "right": "Right"}

EVENTS = HAND_TRACK_EVENTS
TRACK_COLUMNS = HAND_TRACK_COLUMNS
MOTION_COLUMNS = HAND_MOTION_COLUMNS
FINGER_NAMES = HAND_FINGER_NAMES


class TrackingError(Exception):
    """The input cannot be tracked correctly (for example, timestamps that do not increase)."""


@dataclass
class Detection:
    frame: int
    t_ms: int
    label: str                  # "Left" or "Right", as assigned in H1
    status: str                 # H1 side status
    score: float
    points: np.ndarray          # (21, 2) image pixels
    outside: int                # landmarks outside the frame
    reason: str = ""            # why it is not usable ("" when usable)

    @property
    def usable(self) -> bool:
        return self.reason == ""

    @property
    def centre(self) -> np.ndarray:
        return centroid(self.points)

    @property
    def hand_size(self) -> float:
        return float(np.hypot(*(self.points[9] - self.points[0])))


def make_detection(frame, t_ms, label, status, score, norm_points, width, height) -> Detection:
    """Build a Detection from one H1 side. norm_points is (21, 3) normalised; returns a detection with a reason
    when it cannot be used. Missing (not detected) sides are not passed here."""
    pts = np.asarray(norm_points, dtype=float)[:, :2].copy()
    pts[:, 0] *= float(width)
    pts[:, 1] *= float(height)
    outside = int(np.count_nonzero(np.any((np.asarray(norm_points)[:, :2] < 0) |
                                          (np.asarray(norm_points)[:, :2] > 1), axis=1)))
    det = Detection(frame=frame, t_ms=t_ms, label=label, status=status, score=score,
                    points=pts, outside=outside)
    if status not in USABLE_STATUSES:
        det.reason = status                       # "low_confidence", "not_detected", ...
    elif not np.all(np.isfinite(pts)):
        det.reason = "insufficient_landmarks"
    elif outside > PARTIAL_LANDMARK_LIMIT:
        det.reason = "insufficient_landmarks"
    elif det.hand_size < MIN_HAND_SIZE_PX:
        det.reason = "insufficient_landmarks"
    return det


@dataclass
class Track:
    track_id: int
    label: str
    last_frame: int
    last_t: int
    centre: np.ndarray
    hand_size: float
    prev_frame: int | None = None
    prev_centre: np.ndarray | None = None
    velocity_per_frame: np.ndarray = field(default_factory=lambda: np.zeros(2))
    interrupted: bool = False
    terminated: bool = False

    def predict(self, frame: int) -> np.ndarray:
        steps = frame - self.last_frame
        return self.centre + self.velocity_per_frame * steps


def _track_cost(det: Detection, track: Track, frame: int) -> float | None:
    gap = frame - track.last_frame - 1
    gate = GATE_HAND_SIZES * (gap + 1) * track.hand_size
    dist = float(np.hypot(*(det.centre - track.predict(frame))))
    if dist > gate:
        return None
    cost = dist / track.hand_size
    if det.label != track.label:
        cost += LABEL_PENALTY
    return cost


def _best_assignment(dets: list[Detection], tracks: list[Track], frame: int):
    """Exhaustive choice of track (or None = new track) for each detection, with each track used at most once.
    Returns (assignment list aligned with dets, total cost, per-detection cost matrix)."""
    options = []
    matrix = []
    for det in dets:
        row = {}
        for k, tr in enumerate(tracks):
            c = _track_cost(det, tr, frame)
            if c is not None:
                row[k] = c
        matrix.append(row)
        options.append(list(row.keys()) + [None])

    best = None
    for choice in itertools.product(*options):
        used = [c for c in choice if c is not None]
        if len(used) != len(set(used)):
            continue
        total = 0.0
        for i, c in enumerate(choice):
            total += NEW_TRACK_COST if c is None else matrix[i][c]
        if best is None or total < best[1] - 1e-12:
            best = (list(choice), total)
    if best is None:  # cannot happen: all-new is always feasible
        best = ([None] * len(dets), NEW_TRACK_COST * len(dets))
    return best[0], best[1], matrix


def track_hands(detections: dict[int, list[Detection]], frames: list[int], timestamps: dict[int, int],
                statuses: dict[tuple[str, int], str]) -> tuple[list[dict], list[dict]]:
    """Track hands over the given frames.

    detections: frame index -> list of Detection (any usability). frames: all frame indices in order.
    timestamps: frame index -> timestamp in ms. statuses: (label, frame) -> H1 status, for interruption reasons.
    Returns (events, observations). Each observation is one usable detection linked to a track, with the
    fields needed for TRACK_COLUMNS and the motion metrics.
    """
    last_t = None
    for f in frames:
        t = timestamps[f]
        if last_t is not None and (t <= last_t or f <= prev_f):
            raise TrackingError(f"frame {f}: timestamps or frame indices do not increase")
        last_t, prev_f = t, f

    events: list[dict] = []
    observations: list[dict] = []
    tracks: list[Track] = []
    next_id = 1
    last_frame_seen = frames[-1] if frames else None

    for f in frames:
        t = timestamps[f]
        active = [tr for tr in tracks if not tr.terminated]
        for tr in active:
            missing = f - tr.last_frame - 1
            if missing >= 1 and not tr.interrupted:
                tr.interrupted = True
                first_missing = tr.last_frame + 1
                why = statuses.get((tr.label, first_missing), "frame_missing")
                events.append(_event(tr.track_id, "interrupted", first_missing,
                                     timestamps.get(first_missing), tr.label, why, reason=why))
            if missing > MAX_GAP_FRAMES:
                tr.terminated = True
                events.append(_event(tr.track_id, "terminated", f, t, tr.label, "", reason="max_gap_exceeded",
                                     gap=missing))
        candidates = [tr for tr in tracks if not tr.terminated]

        dets_here = [d for d in detections.get(f, []) if d.usable]
        if not dets_here:
            continue

        assignment, _total, matrix = _best_assignment(dets_here, candidates, f)
        close_pair = any(
            float(np.hypot(*(a.centre - b.centre))) < CROSSING_HAND_SIZES * max(a.hand_size, b.hand_size)
            for a, b in itertools.combinations(dets_here, 2)
        )

        for i, det in enumerate(dets_here):
            choice = assignment[i]
            uncertain = []
            if close_pair:
                uncertain.append("hands_close")
            if choice is None:
                tr = Track(track_id=next_id, label=det.label, last_frame=f, last_t=t, centre=det.centre.copy(),
                           hand_size=det.hand_size)
                next_id += 1
                tracks.append(tr)
                event, association, cost, gap = "created", "new", NEW_TRACK_COST, 0
            else:
                tr = candidates[choice]
                gap = f - tr.last_frame - 1
                cost = matrix[i][choice]
                if det.label != tr.label:
                    uncertain.append("label_conflict")
                alternatives = sorted(v for k, v in matrix[i].items() if k != choice) + [NEW_TRACK_COST]
                if alternatives and alternatives[0] <= cost * AMBIGUITY_RATIO + 1e-12:
                    uncertain.append("ambiguous_candidates")
                event = "reacquired" if gap > 0 else "matched"
                association = event
                tr.prev_frame, tr.prev_centre = tr.last_frame, tr.centre.copy()
                if tr.prev_frame is not None and f > tr.prev_frame:
                    tr.velocity_per_frame = (det.centre - tr.prev_centre) / (f - tr.prev_frame)
                tr.last_frame, tr.last_t, tr.centre = f, t, det.centre.copy()
                tr.hand_size = det.hand_size
                tr.label = det.label
                tr.interrupted = False
            observations.append({
                "track_id": tr.track_id, "frame": f, "timestamp_ms": t, "label": det.label,
                "status": det.status, "association": association,
                "uncertain_reason": ";".join(uncertain), "cost": round(float(cost), 4), "gap_frames": int(gap),
                "centroid_x_px": float(det.centre[0]), "centroid_y_px": float(det.centre[1]),
                "det": det,
            })
            events.append(_event(tr.track_id, event, f, t, det.label, det.status, association=association,
                                 uncertain=";".join(uncertain), cost=cost, gap=gap,
                                 centre=det.centre, reason=""))

    for tr in tracks:
        if not tr.terminated:
            tr.terminated = True
            events.append(_event(tr.track_id, "terminated", last_frame_seen, timestamps[last_frame_seen],
                                 tr.label, "", reason="end_of_recording", centre=tr.centre))
    return events, observations


def _event(track_id, event, frame, t_ms, label, status, association="", uncertain="", cost=None, gap=0,
           centre=None, reason=""):
    return {
        "track_id": track_id, "event": event, "frame": int(frame),
        "timestamp_ms": "" if t_ms is None else int(t_ms), "label": label,
        "status": status, "association": association, "uncertain_reason": uncertain,
        "cost": "" if cost is None else round(float(cost), 4), "gap_frames": int(gap),
        "centroid_x_px": "" if centre is None else float(centre[0]),
        "centroid_y_px": "" if centre is None else float(centre[1]),
        "reason": reason,
    }


def motion_rows(observations: list[dict]) -> list[dict]:
    """Per-observation motion metrics, computed from the previous observation of the same track."""
    rows = []
    last_by_track: dict[int, dict] = {}
    for ob in observations:
        det: Detection = ob["det"]
        tid = ob["track_id"]
        prev = last_by_track.get(tid)
        pts = det.points
        c = det.centre
        row = {
            "track_id": tid, "frame": ob["frame"], "timestamp_ms": ob["timestamp_ms"], "label": det.label,
            "status": det.status, "partial_view": int(det.outside > 0),
            "centroid_x_px": float(c[0]), "centroid_y_px": float(c[1]),
            "wrist_x_px": float(pts[0, 0]), "wrist_y_px": float(pts[0, 1]),
            "displacement_px": float("nan"), "wrist_displacement_px": float("nan"),
            "velocity_px_per_s": float("nan"), "acceleration_px_per_s2": float("nan"),
            "direction_deg": float("nan"), "orientation_deg": orientation_deg(pts),
            "orientation_change_deg": float("nan"), "openness": openness(pts),
        }
        tips = fingertips(pts)
        for k, name in enumerate(FINGER_NAMES):
            row[f"{name}_tip_x_px"] = float(tips[k, 0])
            row[f"{name}_tip_y_px"] = float(tips[k, 1])
        for name, value in finger_angles(pts).items():
            row[name] = value

        if prev is None:
            row["motion_reason"] = "first_observation"
        elif ob["gap_frames"] > 0:
            row["motion_reason"] = "after_gap"
        else:
            dt = ob["timestamp_ms"] - prev["timestamp_ms"]
            disp = displacement(c, prev["centre"])
            row["displacement_px"] = disp
            row["wrist_displacement_px"] = displacement(pts[0], prev["points"][0])
            row["velocity_px_per_s"] = velocity(disp, dt)
            row["direction_deg"] = direction_deg(float(c[0] - prev["centre"][0]), float(c[1] - prev["centre"][1]))
            row["orientation_change_deg"] = wrap_angle_deg(row["orientation_deg"] - prev["orientation_deg"])
            if prev.get("velocity_px_per_s") is not None and math.isfinite(prev["velocity_px_per_s"]):
                row["acceleration_px_per_s2"] = acceleration(row["velocity_px_per_s"],
                                                             prev["velocity_px_per_s"], dt)
            row["motion_reason"] = "ok" if math.isfinite(row["velocity_px_per_s"]) else "invalid_motion"
        rows.append(row)
        last_by_track[tid] = {"centre": c, "points": pts, "timestamp_ms": ob["timestamp_ms"],
                              "orientation_deg": row["orientation_deg"],
                              "velocity_px_per_s": row["velocity_px_per_s"]}
    return rows


def quality_summary(events: list[dict], observations: list[dict], motion: list[dict],
                    frames: list[int], timestamps: dict[int, int], statuses: dict[tuple[str, int], str]) -> dict:
    """Counts that let a reader tell valid tracking, loss, uncertainty and invalid motion apart."""
    created = [e for e in events if e["event"] == "created"]
    counts_uncertain = {}
    for ob in observations:
        for reason in filter(None, ob["uncertain_reason"].split(";")):
            counts_uncertain[reason] = counts_uncertain.get(reason, 0) + 1
    missing_reasons: dict[str, int] = {}
    in_range = set(frames)
    for (label, f), status in statuses.items():
        if f in in_range and status not in USABLE_STATUSES:
            missing_reasons[status] = missing_reasons.get(status, 0) + 1
    motion_reasons: dict[str, int] = {}
    for row in motion:
        motion_reasons[row["motion_reason"]] = motion_reasons.get(row["motion_reason"], 0) + 1
    speeds = [r["velocity_px_per_s"] for r in motion if math.isfinite(r["velocity_px_per_s"])]
    intervals = np.diff([timestamps[f] for f in frames]) if len(frames) > 1 else np.array([])
    median_dt = float(np.median(intervals)) if intervals.size else float("nan")
    irregular = int(np.count_nonzero((intervals > 1.5 * median_dt) | (intervals < 0.5 * median_dt))) \
        if intervals.size else 0
    frame_gaps = int(sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1))
    return {
        "frames": len(frames),
        "tracks_created": len(created),
        "tracks_terminated": sum(1 for e in events if e["event"] == "terminated"),
        "interruptions": sum(1 for e in events if e["event"] == "interrupted"),
        "reacquisitions": sum(1 for e in events if e["event"] == "reacquired"),
        "observations": len(observations),
        "uncertain_observations": counts_uncertain,
        "missing_frame_reasons": missing_reasons,
        "motion_reasons": motion_reasons,
        "velocity_px_per_s": {
            "count": len(speeds),
            "median": float(np.median(speeds)) if speeds else None,
            "peak": float(np.max(speeds)) if speeds else None,
        },
        "timestamps": {"median_interval_ms": None if not intervals.size else median_dt,
                       "irregular_intervals": irregular, "frame_index_gaps": frame_gaps},
        "tracking_rule": "see analysis/hand_tracking.py: gate, label penalty, interruption and termination rules",
        "smoothing": "none",
        "coordinates": "image pixels (normalised x * width, y * height); not physical units",
    }
