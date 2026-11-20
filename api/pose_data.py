"""Read-only V2 Pose data and V1 head-pose data for the web pages (phase P3).

Numbers come from src/dashboard/pose_viewer.py and src/analysis/facial_3d_analyzer.py, the same code the
Streamlit pages use. NaN and infinity become null. Validation results are cached per session, keyed on
the modification time of metadata.json, so a regenerated session is checked again. Nothing is written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src" / "analysis", _ROOT / "src" / "dashboard"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import facial_3d_analyzer as f3  # noqa: E402
import facial_analyzer as fa  # noqa: E402
import project_paths as pp  # noqa: E402
import pose_viewer as vv  # noqa: E402
from pose_schema import session_layout  # noqa: E402
from api import face_data  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402

MAX_FRAMES_PER_REQUEST = 300
DEFAULT_THRESHOLD = 0.5

_cache: dict = {}


def clear_cache() -> None:
    _cache.clear()


def _scalar(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return value
    value = float(value)
    return value if math.isfinite(value) else None


def _threshold(value: float) -> float:
    if not (0.0 <= value <= 1.0):
        raise ApiError("invalid_request", "threshold must be between 0 and 1", 400)
    return float(value)


def _v2_dir(person: str, session: str) -> Path:
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    folder = pp.session_dataset_dir(person, session)
    layout = session_layout(folder)
    if layout == "incomplete_v2":
        raise ApiError("incomplete_v2", "pose data is incomplete (metadata.json missing or unreadable)", 409)
    if layout != "v2":
        raise ApiError("no_pose_data", "no pose data for this session", 404)
    return folder


def _validated(person: str, session: str):
    folder = _v2_dir(person, session)
    meta_mtime = (folder / "metadata.json").stat().st_mtime_ns
    key = ("v2", person, session, meta_mtime)
    if key not in _cache:
        try:
            info = vv.validate_v2_session(folder, person, session)
            pose = vv.load_pose(folder)
            timeline = vv.load_timeline(folder, expected_frames=pose.shape[0])
        except vv.PoseDataViewerError as exc:
            _cache[key] = ("invalid", str(exc))
        else:
            _cache.clear()
            _cache[key] = ("ok", info, pose, timeline)
    entry = _cache[key]
    if entry[0] == "invalid":
        raise ApiError("v2_invalid", entry[1], 422)
    return entry[1], entry[2], entry[3]


def info(person: str, session: str) -> dict:
    v2, _pose, _timeline = _validated(person, session)
    return {
        "session_id": session,
        "valid": True,
        "frames": v2.frames,
        "width": v2.width,
        "height": v2.height,
        "fps": _scalar(v2.fps),
        "joint_names": list(vv.joint_names()),
        "scope": "tracking only; does not measure the accuracy of the body motion",
    }


def timeline(person: str, session: str) -> list[dict]:
    _v2, _pose, tl = _validated(person, session)
    rows = []
    for record in tl.to_dict(orient="records"):
        rows.append({k: _scalar(v) if not isinstance(v, str) else v for k, v in record.items()})
    return rows


def _frame_range(start: int, end: int, n: int) -> None:
    if not (0 <= start < end <= n):
        raise ApiError("invalid_request", f"frame range must satisfy 0 <= from < to <= {n}", 400)
    if end - start > MAX_FRAMES_PER_REQUEST:
        raise ApiError("invalid_request", f"at most {MAX_FRAMES_PER_REQUEST} frames per request", 400)


def _joint_rows(frame_pose: np.ndarray, threshold: float) -> list[dict]:
    states = vv.joint_states(frame_pose, threshold)
    return [{
        "index": k,
        "x": _scalar(frame_pose[k, 0]),
        "y": _scalar(frame_pose[k, 1]),
        "z": _scalar(frame_pose[k, 2]),
        "visibility": _scalar(frame_pose[k, 3]),
        "presence": _scalar(frame_pose[k, 4]),
        "state": states[k],
    } for k in range(vv.N_POSE)]


def pose(person: str, session: str, start: int, end: int, threshold: float) -> dict:
    _v2, arr, _tl = _validated(person, session)
    threshold = _threshold(threshold)
    _frame_range(start, end, arr.shape[0])
    frames = [{"index": f, "joints": _joint_rows(arr[f], threshold)} for f in range(start, end)]
    return {"threshold": threshold, "from": start, "to": end, "frames": frames}


def quality(person: str, session: str, frame: int, threshold: float) -> dict:
    _v2, arr, tl = _validated(person, session)
    threshold = _threshold(threshold)
    if not (0 <= frame < arr.shape[0]):
        raise ApiError("not_found", "frame index out of range", 404)
    summary = vv.quality_summary(arr, tl, threshold, frame)
    table = vv.landmark_table(arr[frame], threshold).to_dict(orient="records")
    return {
        "frame": frame,
        "quality": {k: _scalar(v) if not isinstance(v, (list, str)) else v for k, v in summary.items()},
        "joints": [{k: _scalar(v) if not isinstance(v, str) else v for k, v in row.items()} for row in table],
    }


def overlay_jpeg(person: str, session: str, index: int, threshold: float,
                 show_points: bool, show_skeleton: bool) -> bytes:
    v2, arr, _tl = _validated(person, session)
    threshold = _threshold(threshold)
    if not (0 <= index < arr.shape[0]):
        raise ApiError("not_found", "frame index out of range", 404)
    video = pp.session_video(person, session)
    if not video.is_file():
        raise ApiError("no_video", "the recording for this session is missing", 404)
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ApiError("not_found", "the video frame could not be read", 404)
    drawn = vv.draw_pose_overlay(frame, arr[index], threshold, show_points=show_points, show_skeleton=show_skeleton)
    ok, buf = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ApiError("encode_failed", "the overlay image could not be encoded", 500)
    return buf.tobytes()


def head_pose(person: str, session: str) -> dict:
    """Yaw, pitch and roll, and the face depth span, per frame. Same functions as the 3D Streamlit page."""
    result = face_data._analysis(person, session)
    ds = fa.load_session(fa.make_session(person, session))
    angles = f3.head_pose_angles(ds.matrices)
    depth = f3.depth_range_per_frame(ds.landmarks)
    return {
        "time_s": [_scalar(t) for t in result.table["time_s"].to_numpy()],
        "yaw_deg": [_scalar(v) for v in angles["yaw_deg"].to_numpy()],
        "pitch_deg": [_scalar(v) for v in angles["pitch_deg"].to_numpy()],
        "roll_deg": [_scalar(v) for v in angles["roll_deg"].to_numpy()],
        "depth_range": [_scalar(v) for v in depth],
    }
