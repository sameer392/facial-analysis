"""Read-only validation and quality summary for a hand dataset (phase H4).

Checks a pose data session that has the hand stream (H1), hand tracking (H2) and gesture output (H3).
It reports errors and warnings and computes quality metrics. Nothing is repaired or changed.

    error    the data contradict the format or each other. The session is not a valid dataset.
    warning  the data are consistent but questionable. The observations are kept.

A session with no errors is structurally consistent. That is not the same as scientifically accurate:
these checks cannot tell whether the landmarks or the gesture labels are correct.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parent
_SRC = _ANALYSIS.parent
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extraction.model_registry import get_spec  # noqa: E402
from gesture_rules import GESTURES, validate_events  # noqa: E402
from hand_tracking import USABLE_STATUSES  # noqa: E402
from pose_schema import (  # noqa: E402
    ARRAY_SPECS,
    HAND_DETECTED_STATUSES,
    HAND_GESTURE_EVENT_COLUMNS,
    HAND_GESTURE_EVENTS_FILE,
    HAND_GESTURE_OBS_COLUMNS,
    HAND_GESTURE_OBS_FILE,
    HAND_MOTION_COLUMNS,
    HAND_MOTION_FILE,
    HAND_SIDE_STATUSES,
    HAND_STATUS_COLUMNS,
    HAND_STATUS_FILE,
    HAND_TRACK_COLUMNS,
    HAND_TRACK_EVENTS,
    HAND_TRACKS_FILE,
    HANDEDNESS_COLUMNS,
    METADATA_FILE,
    MANIFEST_FILE,
    OPTIONAL_ARRAY_SPECS,
    SCHEMA_VERSION,
    TIMELINE_FILE,
    session_layout,
    validate_hand_status_rows,
    validate_metadata,
)
from pose_writer import verify_manifest  # noqa: E402

HAND_N = 21
COORD_TOLERANCE = 0.5          # normalised x, y outside [-0.5, 1.5] is implausible
WORLD_LIMIT_M = 1.0            # hand world coordinates beyond 1 m from the hip centre are implausible
CENTROID_TOLERANCE_PX = 1e-3   # H2 centroid must equal the H1 landmark mean, in pixels
MAX_JUMP_FRAME_DIAGONAL = 0.25  # centroid moves more than this fraction of the frame diagonal in one frame
MODEL_KEY = "hand"


@dataclass
class Report:
    session: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"session": self.session, "valid": self.valid, "errors": self.errors,
                "warnings": self.warnings, "metrics": self.metrics, "provenance": self.provenance}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path, columns, report: Report, name: str):
    if not path.is_file():
        report.errors.append(f"{name}: file is missing")
        return None
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        report.errors.append(f"{name}: missing columns {missing}")
        return None
    extra = [c for c in df.columns if c not in columns]
    if extra:
        report.warnings.append(f"{name}: unexpected columns {extra}")
    return df


def validate_hand_session(session_dir) -> Report:
    """Validate one pose data session. Never raises for bad data: problems are returned in the report."""
    session = Path(session_dir)
    report = Report(session=str(session.name))
    if session_layout(session) != "pose":
        report.errors.append("no complete pose data session (metadata.json missing or not schema 2.x)")
        return report

    meta = _load_json(session / METADATA_FILE, report, "metadata.json")
    manifest = _load_json(session / MANIFEST_FILE, report, "manifest.json")
    if meta is None:
        return report

    _check_metadata(meta, report)
    _check_manifest(session, manifest, report)

    timeline = _read_csv(session / TIMELINE_FILE, ["frame", "timestamp_ms", "face_detected", "pose_detected",
                                                    "left_hand_detected", "right_hand_detected"], report, TIMELINE_FILE)
    arrays = _load_arrays(session, report)
    status = _read_csv(session / HAND_STATUS_FILE, HAND_STATUS_COLUMNS, report, HAND_STATUS_FILE)
    handedness = _read_csv(session / "hands" / "handedness.csv", HANDEDNESS_COLUMNS, report, "hands/handedness.csv")
    quality = _load_json(session / "quality" / "tracking_quality.json", report, "quality/tracking_quality.json")

    if timeline is None:
        return report
    n = len(timeline)
    frames = timeline["frame"].to_numpy()
    ts = timeline["timestamp_ms"].to_numpy(dtype=np.int64)
    _check_timeline(frames, ts, meta, n, report)
    if arrays is not None:
        _check_arrays(arrays, n, report)
    if status is not None:
        _check_status(status, arrays, ts, n, report)
    if handedness is not None and status is not None:
        _check_handedness(handedness, status, report)
    if arrays is not None and "hands/left_hand_world.npy" in arrays:
        _check_world(arrays, status, report)
    if arrays is not None and status is not None:
        _check_coordinates(arrays, status, report)

    hands_streams = meta.get("streams", {}).get("hands", {})
    tracking = None
    if hands_streams.get("tracking"):
        tracking = _check_tracking(session, meta, status, ts, arrays, report)
    if hands_streams.get("gestures"):
        _check_gestures(session, status, ts, tracking, report)
    _check_quality_consistency(quality, tracking, report)

    report.metrics["detection"] = _detection_metrics(status, n)
    if tracking is not None:
        report.metrics["tracking"] = tracking.get("metrics", {})
        report.metrics["motion"] = tracking.get("motion", {})
    if hands_streams.get("gestures") and tracking is not None:
        report.metrics["gestures"] = tracking.get("gestures", {})
    report.metrics["timestamps"] = _timestamp_metrics(ts)
    report.provenance = _provenance(meta, manifest, session)
    return report


def _load_json(path: Path, report: Report, name: str):
    if not path.is_file():
        report.errors.append(f"{name}: file is missing")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"{name}: not valid JSON ({exc})")
        return None


def _check_metadata(meta: dict, report: Report) -> None:
    for issue in validate_metadata(meta):
        report.errors.append(f"metadata: {issue}")
    if not str(meta.get("schema_version", "")).startswith("2."):
        report.errors.append("metadata: schema_version is not 2.x")
    model = meta.get("models", {}).get(MODEL_KEY)
    if model is None:
        report.errors.append("metadata: no hand model recorded (models.hand)")
    else:
        recorded = str(model.get("sha256", "")).lower()
        registry = get_spec("hand").sha256.lower()
        if recorded != registry:
            report.errors.append("metadata: hand model hash does not match model_registry.py")
        if model.get("licence_status") != "to verify":
            report.warnings.append("metadata: hand model licence_status is not 'to verify'")
    extraction = meta.get("extraction", {})
    for key in ("software", "software_version", "mediapipe_version", "running_mode"):
        if not extraction.get(key):
            report.errors.append(f"metadata: extraction.{key} is missing")
    source = meta.get("source", {})
    for key in ("fps", "frame_count", "width", "height", "video_filename"):
        if source.get(key) in (None, ""):
            report.errors.append(f"metadata: source.{key} is missing")
    if "hands" not in meta.get("configuration", {}):
        report.errors.append("metadata: configuration.hands is missing")


def _check_manifest(session: Path, manifest, report: Report) -> None:
    if manifest is None:
        return
    for issue in verify_manifest(session, manifest):
        report.errors.append(f"manifest: {issue}")
    created = manifest.get("created_utc")
    try:
        datetime.strptime(str(created), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        report.errors.append("manifest: created_utc is missing or not an ISO UTC timestamp")


def _load_arrays(session: Path, report: Report):
    arrays = {}
    ok = True
    for spec in ARRAY_SPECS:
        path = session / spec.relative_path
        if not path.is_file():
            report.errors.append(f"{spec.relative_path}: file is missing")
            ok = False
            continue
        try:
            arrays[spec.relative_path] = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            report.errors.append(f"{spec.relative_path}: cannot be read ({exc})")
            ok = False
    for spec in OPTIONAL_ARRAY_SPECS:
        path = session / spec.relative_path
        if path.is_file():
            try:
                arrays[spec.relative_path] = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                report.errors.append(f"{spec.relative_path}: cannot be read ({exc})")
                ok = False
        else:
            report.warnings.append(f"{spec.relative_path}: world coordinates not written")
    return arrays if ok else None


def _check_timeline(frames, ts, meta, n, report: Report) -> None:
    expected = np.arange(n)
    if not np.array_equal(frames, expected):
        report.errors.append("timeline: frame indexes are not 0..n-1 in order (missing or repeated frames)")
    if n and np.any(np.diff(ts) <= 0):
        report.errors.append("timeline: timestamps do not strictly increase")
    frame_count = meta.get("source", {}).get("frame_count")
    if frame_count is not None and int(frame_count) != n:
        report.errors.append(f"timeline: {n} rows but source.frame_count is {frame_count}")
    fps = meta.get("source", {}).get("fps")
    if fps and n > 1:
        ideal = np.round(np.arange(n) * 1000.0 / float(fps)).astype(np.int64)
        if np.max(np.abs(ideal - ts)) > 1:
            report.warnings.append("timeline: timestamps differ from the frame rate by more than 1 ms")


def _check_arrays(arrays: dict, n: int, report: Report) -> None:
    for side in ("left", "right"):
        arr = arrays.get(f"hands/{side}_hand.npy")
        if arr is None:
            continue
        if arr.shape != (n, HAND_N, 3):
            report.errors.append(f"hands/{side}_hand.npy: shape {arr.shape}, expected {(n, HAND_N, 3)}")
        if arr.dtype != np.float32:
            report.errors.append(f"hands/{side}_hand.npy: dtype {arr.dtype}, expected float32")


def _check_status(status: pd.DataFrame, arrays, ts, n: int, report: Report) -> None:
    if len(status) != n:
        report.errors.append(f"{HAND_STATUS_FILE}: {len(status)} rows, expected {n}")
        return
    if not np.array_equal(status["frame"].to_numpy(), np.arange(n)):
        report.errors.append(f"{HAND_STATUS_FILE}: frame indexes do not match the timeline")
    if not np.array_equal(status["timestamp_ms"].to_numpy(dtype=np.int64), ts):
        report.errors.append(f"{HAND_STATUS_FILE}: timestamps differ from the timeline")
    for side in ("left", "right"):
        bad = set(status[f"{side}_status"].dropna().unique()) - set(HAND_SIDE_STATUSES)
        if bad:
            report.errors.append(f"{HAND_STATUS_FILE}: {side} statuses not allowed: {sorted(bad)}")
    rows = status.to_dict("records")
    if arrays is not None:
        issues = validate_hand_status_rows(rows, n, arrays)
        for issue in issues:
            report.errors.append(f"{HAND_STATUS_FILE}: {issue}")
    for side in ("left", "right"):
        unassigned = int((status[f"{side}_status"] == "unassigned").sum())
        if unassigned:
            report.warnings.append(f"{side} side: {unassigned} frame(s) had a hand that could not be assigned")


def _check_handedness(handedness: pd.DataFrame, status: pd.DataFrame, report: Report) -> None:
    if len(handedness) != len(status):
        report.errors.append("hands/handedness.csv: row count differs from hand_status.csv")
        return
    for side in ("left", "right"):
        flag = handedness[f"{side}_detected"].astype(bool).to_numpy()
        expected = status[f"{side}_status"].isin(HAND_DETECTED_STATUSES).to_numpy()
        if not np.array_equal(flag, expected):
            report.errors.append(f"hands/handedness.csv: {side}_detected disagrees with hand_status.csv")


def _check_world(arrays: dict, status, report: Report) -> None:
    for side in ("left", "right"):
        world = arrays.get(f"hands/{side}_hand_world.npy")
        if world is None or status is None:
            continue
        finite = np.all(np.isfinite(world), axis=(1, 2))
        avail = status[f"{side}_world_available"].astype(bool).to_numpy()
        if not np.array_equal(finite, avail):
            report.errors.append(f"hands/{side}_hand_world.npy: world availability disagrees with hand_status.csv")
        vals = world[finite]
        if vals.size and np.max(np.abs(vals)) > WORLD_LIMIT_M:
            report.warnings.append(f"{side} world coordinates: values beyond {WORLD_LIMIT_M} m (implausible)")


def _check_coordinates(arrays: dict, status: pd.DataFrame, report: Report) -> None:
    for side in ("left", "right"):
        arr = arrays.get(f"hands/{side}_hand.npy")
        if arr is None:
            continue
        detected = status[f"{side}_status"].isin(HAND_DETECTED_STATUSES).to_numpy()
        finite_rows = np.all(np.isfinite(arr), axis=(1, 2))
        if not np.array_equal(finite_rows, detected):
            report.errors.append(f"hands/{side}_hand.npy: landmarks are not finite exactly where the hand is detected")
        xy = arr[detected][:, :, :2]
        if xy.size and (np.min(xy) < -COORD_TOLERANCE or np.max(xy) > 1.0 + COORD_TOLERANCE):
            report.warnings.append(f"{side} hand: normalised coordinates beyond the implausible range")


def _detection_metrics(status, n: int) -> dict:
    if status is None or len(status) != n:
        return {}
    any_hand = (status["left_status"].isin(HAND_DETECTED_STATUSES) | status["right_status"].isin(HAND_DETECTED_STATUSES))
    out = {"frames": n, "frames_with_detected_hand": int(any_hand.sum()),
           "frames_without_detected_hand": int((~any_hand).sum())}
    for side in ("left", "right"):
        counts = Counter(status[f"{side}_status"])
        out[f"{side}"] = {s: int(counts.get(s, 0)) for s in HAND_SIDE_STATUSES}
        out[f"{side}"]["detected_frames"] = int(status[f"{side}_status"].isin(HAND_DETECTED_STATUSES).sum())
        out[f"{side}"]["outside_frame_landmarks_total"] = int(status[f"{side}_outside_landmarks"].sum())
        out[f"{side}"]["low_confidence"] = counts.get("low_confidence", 0)
    out["unassigned_hand_frames"] = int((status["unassigned_hands"] > 0).sum())
    return out


def _timestamp_metrics(ts) -> dict:
    if len(ts) < 2:
        return {}
    d = np.diff(ts).astype(float)
    med = float(np.median(d))
    return {"median_interval_ms": med,
            "irregular_intervals": int(np.count_nonzero((d > 1.5 * med) | (d < 0.5 * med)))}


def _check_centroid_consistency(motion: pd.DataFrame, arrays: dict, meta, report: Report) -> None:
    """The H2 centroid must equal the mean of the H1 landmarks for that side and frame, in pixels."""
    width = float(meta["source"]["width"])
    height = float(meta["source"]["height"])
    worst = 0.0
    for r in motion.itertuples(index=False):
        side = "left" if r.label == "Left" else "right"
        lm = arrays[f"hands/{side}_hand.npy"][int(r.frame)]
        mean = lm[:, :2].mean(axis=0) * np.array([width, height])
        err = float(np.max(np.abs(mean - np.array([r.centroid_x_px, r.centroid_y_px]))))
        worst = max(worst, err)
    if worst > CENTROID_TOLERANCE_PX:
        report.errors.append(f"{HAND_MOTION_FILE}: centroid differs from the H1 landmarks by up to {worst:.4f} px")


def _check_tracking(session: Path, meta, status, ts, arrays, report: Report):
    tracks = _read_csv(session / HAND_TRACKS_FILE, HAND_TRACK_COLUMNS, report, HAND_TRACKS_FILE)
    motion = _read_csv(session / HAND_MOTION_FILE, HAND_MOTION_COLUMNS, report, HAND_MOTION_FILE)
    if tracks is None or motion is None:
        return None
    ts_of = dict(zip(range(len(ts)), ts.tolist()))
    st_of = {}
    if status is not None:
        for side, label in (("left", "Left"), ("right", "Right")):
            for f, s in zip(status["frame"], status[f"{side}_status"]):
                st_of[(label, int(f))] = s

    bad_events = set(tracks["event"].dropna()) - set(HAND_TRACK_EVENTS)
    if bad_events:
        report.errors.append(f"{HAND_TRACKS_FILE}: events not allowed: {sorted(bad_events)}")
    created = set()
    for tid, grp in tracks.groupby("track_id"):
        grp = grp.sort_values("frame", kind="stable")
        events = grp["event"].tolist()
        frames = grp["frame"].tolist()
        if events[0] != "created":
            report.errors.append(f"track {tid}: first event is {events[0]}, expected created")
        created.add(int(tid))
        if any(b < a for a, b in zip(frames, frames[1:])):
            report.errors.append(f"track {tid}: events are not in frame order")
        if sum(1 for e in events if e == "created") != 1:
            report.errors.append(f"track {tid}: must be created exactly once")
        if events.count("terminated") > 1:
            report.errors.append(f"track {tid}: terminated more than once")
        for row in grp.itertuples(index=False):
            if row.event in ("created", "matched", "reacquired"):
                if row.status not in USABLE_STATUSES:
                    report.errors.append(f"track {tid} frame {row.frame}: linked with status {row.status}")
                if st_of and st_of.get((row.label, int(row.frame))) != row.status:
                    report.errors.append(f"track {tid} frame {row.frame}: status disagrees with hand_status.csv")
                if ts_of and int(row.timestamp_ms) != ts_of.get(int(row.frame)):
                    report.errors.append(f"track {tid} frame {row.frame}: timestamp differs from the timeline")
        ints = grp[grp["event"] == "terminated"]
        if len(ints) and events[-1] != "terminated":
            report.errors.append(f"track {tid}: has a terminated event but it is not last")

    linked = tracks[tracks["event"].isin(["created", "matched", "reacquired"])]
    linked_keys = set(zip(linked["track_id"].astype(int), linked["frame"].astype(int)))
    motion_keys = list(zip(motion["track_id"].astype(int), motion["frame"].astype(int)))
    orphans = [k for k in motion_keys if k not in linked_keys]
    if orphans:
        report.errors.append(f"{HAND_MOTION_FILE}: {len(orphans)} row(s) without a linked track event, e.g. {orphans[0]}")
    if len(set(motion_keys)) != len(motion_keys):
        report.errors.append(f"{HAND_MOTION_FILE}: duplicate (track, frame) rows")
    unknown_tracks = set(motion["track_id"].astype(int)) - created
    if unknown_tracks:
        report.errors.append(f"{HAND_MOTION_FILE}: track ids without a created event: {sorted(unknown_tracks)}")
    if ts_of:
        bad_ts = int(sum(1 for r in motion.itertuples(index=False) if int(r.timestamp_ms) != ts_of.get(int(r.frame))))
        if bad_ts:
            report.errors.append(f"{HAND_MOTION_FILE}: {bad_ts} row(s) with a timestamp that differs from the timeline")

    # Coordinate consistency between H1 landmarks and H2 centroids is checked in validate_hand_session via arrays.
    motion_reasons = Counter(motion["motion_reason"])
    vel = motion["velocity_px_per_s"].dropna()
    implausible = 0
    diag = math.hypot(float(meta["source"]["width"]), float(meta["source"]["height"]))
    for tid, grp in motion.groupby("track_id"):
        grp = grp.sort_values("frame")
        c = grp[["centroid_x_px", "centroid_y_px"]].to_numpy()
        f = grp["frame"].to_numpy()
        for i in range(1, len(grp)):
            if f[i] - f[i - 1] == 1 and float(np.hypot(*(c[i] - c[i - 1]))) > MAX_JUMP_FRAME_DIAGONAL * diag:
                implausible += 1
    if implausible:
        report.warnings.append(f"{HAND_MOTION_FILE}: {implausible} centroid jump(s) larger than "
                               f"{MAX_JUMP_FRAME_DIAGONAL} of the frame diagonal in one frame (implausible)")
    if arrays is not None and status is not None:
        _check_centroid_consistency(motion, arrays, meta, report)
    gesture_metrics = {}
    tracking = {
        "metrics": {
            "tracks": int(len(created)),
            "events": Counter(tracks["event"]),
            "interruptions": int((tracks["event"] == "interrupted").sum()),
            "reacquisitions": int((tracks["event"] == "reacquired").sum()),
            "uncertain_links": int(tracks["uncertain_reason"].fillna("").astype(bool).sum()),
            "terminations_by_reason": dict(Counter(tracks.loc[tracks["event"] == "terminated", "reason"])),
            "missing_observations": int(sum(
                1 for (lab, f), s in st_of.items() if s not in USABLE_STATUSES and f in set(ts_of))),
            "track_duration_frames": _track_durations(tracks),
        },
        "motion": {
            "rows": int(len(motion)),
            "reasons": dict(motion_reasons),
            "velocity_px_per_s": {"count": int(len(vel)),
                                  "median": float(vel.median()) if len(vel) else None,
                                  "peak": float(vel.max()) if len(vel) else None},
            "missing_metric_values": {c: int(motion[c].isna().sum()) for c in
                                      ("velocity_px_per_s", "acceleration_px_per_s2", "direction_deg")},
            "implausible_jumps": implausible,
        },
        "gestures": gesture_metrics,
        "keys": linked_keys,
        "tracks_frame": tracks,
        "motion_frame": motion,
    }
    if tracking["metrics"]["uncertain_links"]:
        report.warnings.append(f"{HAND_TRACKS_FILE}: {tracking['metrics']['uncertain_links']} uncertain association(s)")
    return tracking


def _track_durations(tracks: pd.DataFrame) -> dict:
    out = {}
    for tid, grp in tracks.groupby("track_id"):
        out[int(tid)] = int(grp["frame"].max() - grp["frame"].min() + 1)
    vals = list(out.values())
    return {"per_track": out, "median": float(np.median(vals)) if vals else None,
            "max": max(vals) if vals else None}


def _check_gestures(session: Path, status, ts, tracking, report: Report) -> None:
    obs = _read_csv(session / HAND_GESTURE_OBS_FILE, HAND_GESTURE_OBS_COLUMNS, report, HAND_GESTURE_OBS_FILE)
    events = _read_csv(session / HAND_GESTURE_EVENTS_FILE, HAND_GESTURE_EVENT_COLUMNS, report, HAND_GESTURE_EVENTS_FILE)
    if obs is None or events is None or tracking is None:
        return
    keys = tracking["keys"]
    motion_keys = set(zip(tracking["motion_frame"]["track_id"].astype(int),
                          tracking["motion_frame"]["frame"].astype(int)))
    bad_class = set(obs["frame_class"]) - set(GESTURES)
    if bad_class:
        report.errors.append(f"{HAND_GESTURE_OBS_FILE}: classes not allowed: {sorted(bad_class)}")
    obs_keys = list(zip(obs["track_id"].astype(int), obs["frame"].astype(int)))
    if len(set(obs_keys)) != len(obs_keys):
        report.errors.append(f"{HAND_GESTURE_OBS_FILE}: duplicate (track, frame) rows")
    missing_refs = [k for k in obs_keys if k not in motion_keys]
    if missing_refs:
        report.errors.append(f"{HAND_GESTURE_OBS_FILE}: {len(missing_refs)} observation(s) reference no motion row, "
                             f"e.g. {missing_refs[0]}")
    if ts is not None:
        ts_of = dict(zip(range(len(ts)), ts.tolist()))
        bad = int(sum(1 for r in obs.itertuples(index=False) if int(r.timestamp_ms) != ts_of.get(int(r.frame))))
        if bad:
            report.errors.append(f"{HAND_GESTURE_OBS_FILE}: {bad} timestamp(s) differ from the timeline")
    event_ids = set(events["event_id"])
    referenced = set(obs["event_id"].dropna())
    unknown_ids = referenced - event_ids
    if unknown_ids:
        report.errors.append(f"{HAND_GESTURE_OBS_FILE}: event ids not in {HAND_GESTURE_EVENTS_FILE}: {sorted(unknown_ids)[:5]}")

    ev_rows = events.to_dict("records")
    issues = validate_events(ev_rows)
    for issue in issues:
        report.errors.append(f"{HAND_GESTURE_EVENTS_FILE}: {issue}")
    for ev in ev_rows:
        sub = obs[obs["event_id"] == ev["event_id"]]
        if len(sub) != int(ev["supporting_observations"]) + int(ev["bridged_observations"]):
            report.errors.append(f"{ev['event_id']}: observation count {len(sub)} disagrees with the event summary")
        if len(sub) and (int(sub["frame"].min()) != int(ev["start_frame"]) or int(sub["frame"].max()) != int(ev["end_frame"])):
            report.errors.append(f"{ev['event_id']}: start/end frames disagree with its observations")
        if len(sub) and set(sub["track_id"].astype(int)) != {int(ev["track_id"])}:
            report.errors.append(f"{ev['event_id']}: observations belong to other tracks")
        if ts is not None and len(sub):
            ts_of = dict(zip(range(len(ts)), ts.tolist()))
            if int(ev["start_timestamp_ms"]) != ts_of.get(int(ev["start_frame"])) or \
                    int(ev["end_timestamp_ms"]) != ts_of.get(int(ev["end_frame"])):
                report.errors.append(f"{ev['event_id']}: timestamps disagree with the timeline")
    for t in obs.itertuples(index=False):
        if t.frame_class == "unknown" and not isinstance(t.reason, str):
            report.warnings.append("gesture observation: unknown without a reason")
            break

    counts = Counter(obs["frame_class"])
    tracking["gestures"] = {
        "observations": int(len(obs)),
        "frame_classes": {g: int(counts.get(g, 0)) for g in GESTURES},
        "unknown_rate": float(counts.get("unknown", 0) / len(obs)) if len(obs) else None,
        "unknown_reasons": dict(Counter(obs.loc[obs["frame_class"] == "unknown", "reason"].dropna())),
        "quality": dict(Counter(obs["quality"])),
        "events": int(len(events)),
        "events_per_gesture": dict(Counter(events["gesture"])),
        "duration_ms": {"median": float(events["duration_ms"].median()) if len(events) else None,
                        "min": int(events["duration_ms"].min()) if len(events) else None,
                        "max": int(events["duration_ms"].max()) if len(events) else None},
        "short_events_under_1s": int((events["duration_ms"] < 1000).sum()),
        "end_reasons": dict(Counter(events["end_reason"])),
    }
    if len(events) and (events["end_reason"] != "end_of_track").any():
        report.warnings.append("gesture events: some events end by class change, gap or unknown run")
    del keys


def _check_quality_consistency(quality, tracking, report: Report) -> None:
    if quality is None:
        return
    hands = quality.get("hands", {})
    trk = hands.get("tracking")
    if trk and tracking is not None:
        if trk.get("tracks_created") != tracking["metrics"]["tracks"]:
            report.errors.append("quality: tracks_created disagrees with hands/tracks.csv")
        if trk.get("observations") != len(tracking["motion_frame"]):
            report.errors.append("quality: observations disagrees with hands/motion_metrics.csv")
    gest = hands.get("gestures")
    if gest and tracking is not None and "gestures" in tracking:
        if gest.get("events") != tracking["gestures"].get("events"):
            report.errors.append("quality: gesture event count disagrees with hands/gesture_events.csv")
        if gest.get("observations") != tracking["gestures"].get("observations"):
            report.errors.append("quality: gesture observation count disagrees with hands/gesture_observations.csv")


def _provenance(meta: dict, manifest, session: Path) -> dict:
    return {
        "schema_version": meta.get("schema_version"),
        "extraction": meta.get("extraction", {}),
        "hand_model": meta.get("models", {}).get(MODEL_KEY, {}),
        "configuration": meta.get("configuration", {}),
        "source": meta.get("source", {}),
        "timeline": meta.get("timeline", {}),
        "processing_timestamp_utc": (manifest or {}).get("created_utc"),
        "manifest_sha256": _sha256(session / MANIFEST_FILE) if (session / MANIFEST_FILE).is_file() else None,
        "calibration": "none recorded: no calibrated thresholds exist yet",
    }
