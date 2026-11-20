"""Read-only face data data for the web pages (phase P2). Numbers come from the existing Python analysis.

NaN and infinity are returned as null. Results are cached per session, keyed on the modification time
of facial_data.json, so a regenerated session is recomputed. Nothing is written.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ANALYSIS = Path(__file__).resolve().parents[1] / "src" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))
_DASHBOARD = Path(__file__).resolve().parents[1] / "src" / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import facial_analyzer as fa  # noqa: E402
import project_paths as pp  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402

SERIES_FIELDS = (
    "face_detected", "neutral_offset", "displacement", "velocity", "acceleration",
    "jaw_open", "mouth_activity", "eye_blink", "brow_inner_up", "yaw_deg", "pitch_deg", "roll_deg",
    "is_neutral",
)
COMPARISON_METRICS = ("jaw_open", "mouth_activity", "eye_blink", "brow_inner_up", "velocity", "acceleration")
HISTOGRAM_BINS = 20

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
    value = float(value)
    return value if math.isfinite(value) else None


def _column(values) -> list:
    return [_scalar(v) for v in values]


def _check_session(person: str, session: str) -> Path:
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    facial = pp.session_dataset_dir(person, session) / "facial_data.json"
    if not facial.is_file():
        raise ApiError("no_face_data", "no face data for this session", 404)
    return facial


def blendshapes_csv_path(person: str, session: str) -> Path:
    """The per-frame blendshape CSV written by the analysis (the same file the Streamlit page offers)."""
    _check_session(person, session)
    path = pp.session_dataset_dir(person, session) / "facial_blendshapes.csv"
    if not path.is_file():
        raise ApiError("not_found", "the blendshape CSV for this session is missing", 404)
    return path


def _analysis(person: str, session: str):
    facial = _check_session(person, session)
    key = (person, session, facial.stat().st_mtime_ns)
    if key not in _cache:
        _cache.clear()
        _cache[key] = fa.analyse_session(fa.make_session(person, session))
    return _cache[key]


def summary(person: str, session: str) -> dict:
    result = _analysis(person, session)
    quality = {k: _scalar(v) if not isinstance(v, str) and not isinstance(v, list) else v
               for k, v in result.quality.items()}
    quality["scope"] = result.quality["scope"]
    return {
        "session_id": session,
        "frames": int(len(result.table)),
        "neutral_frames": int(result.neutral_frames),
        "blink_events": int(len(result.blinks)),
        "quality": quality,
    }


def series(person: str, session: str, fields: list[str]) -> dict:
    result = _analysis(person, session)
    bad = [f for f in fields if f not in SERIES_FIELDS]
    if bad:
        raise ApiError("invalid_field", f"unknown field(s): {', '.join(bad)}", 400)
    out = {"time_s": _column(result.table["time_s"].to_numpy())}
    for field in fields:
        out[field] = _column(result.table[field].to_numpy())
    return out


def blinks(person: str, session: str) -> list[dict]:
    result = _analysis(person, session)
    rows = []
    for _, row in result.blinks.iterrows():
        rows.append({
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "start_s": _scalar(row["start_s"]),
            "duration_ms": _scalar(row["duration_ms"]),
            "peak_score": _scalar(row["peak_score"]),
        })
    return rows


def landmarks(person: str, session: str, index: int, compensated: bool) -> dict:
    result = _analysis(person, session)
    if not (0 <= index < len(result.table)):
        raise ApiError("not_found", "frame index out of range", 404)
    if compensated:
        frame = result.compensated[index]
    else:
        frame = fa.load_session(fa.make_session(person, session)).landmarks[index]
    points = [[_scalar(v) for v in point] for point in frame]
    return {"index": index, "compensated": compensated, "points": points}


def comparison(person: str, sessions: list[str]) -> dict:
    if not sessions:
        raise ApiError("invalid_request", "at least one session is required", 400)
    analyses = {sid: _analysis(person, sid) for sid in sessions}
    rows = []
    for sid, result in analyses.items():
        rows.append({
            "session_id": sid,
            "frames": int(len(result.table)),
            "quality_score": _scalar(result.quality["quality_score"]),
            "detection_rate": _scalar(result.quality["detection_rate"]),
            "blink_events": int(len(result.blinks)),
            "neutral_frames": int(result.neutral_frames),
        })
    distributions = {}
    for metric in COMPARISON_METRICS:
        values = {sid: result.table[metric].to_numpy(dtype=float) for sid, result in analyses.items()}
        finite = np.concatenate([v[np.isfinite(v)] for v in values.values()]) if values else np.array([])
        if finite.size == 0:
            distributions[metric] = {"edges": [], "shares": {}}
            continue
        edges = np.linspace(float(finite.min()), float(finite.max()), HISTOGRAM_BINS + 1)
        shares = {}
        for sid, v in values.items():
            v = v[np.isfinite(v)]
            counts, _ = np.histogram(v, bins=edges)
            total = int(v.size)
            shares[sid] = [_scalar(c / total) if total else None for c in counts]
        distributions[metric] = {"edges": [_scalar(e) for e in edges], "shares": shares}
    return {"sessions": rows, "distributions": distributions}


_topology_cache: dict = {}


def topology() -> dict:
    """Face mesh and region edges (pairs of landmark indices) from MediaPipe, with the overlay's colours.

    Read once. The same lists the Streamlit overlay uses, so the views match.
    """
    if not _topology_cache:
        import overlay  # the existing overlay module (no Streamlit import)

        topo = overlay.load_topology()

        def as_pairs(arr) -> list[list[int]]:
            return [[int(a), int(b)] for a, b in arr]

        def hex_colour(bgr) -> str:
            b, g, r = (int(v) for v in bgr)
            return f"#{r:02x}{g:02x}{b:02x}"

        _topology_cache["mesh"] = {"edges": as_pairs(topo["mesh"]), "colour": hex_colour(overlay.MESH_COLOUR)}
        _topology_cache["regions"] = {
            name: {"edges": as_pairs(topo[name]), "colour": hex_colour(colour)}
            for name, colour in overlay.REGION_COLOURS.items()
        }
    return {"mesh": _topology_cache["mesh"], "regions": _topology_cache["regions"]}


_frames_cache: dict = {}


def _face_frames(person: str, session: str) -> list[dict]:
    """The per-frame records from facial_data.json, read once per file change (as the Streamlit page does)."""
    import json

    path = pp.session_dataset_dir(person, session) / "facial_data.json"
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _frames_cache:
        _frames_cache.clear()
        _frames_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _frames_cache[key]


def frame_image(person: str, session: str, index: int, mode: str, opacity: float,
                thickness: int, point_radius: int) -> bytes:
    """One video frame as JPEG, with the chosen landmark overlay (mode 'off' gives the original frame)."""
    import cv2
    import overlay

    _check_session(person, session)
    frames = _face_frames(person, session)
    if not (0 <= index < len(frames)):
        raise ApiError("not_found", "frame index out of range", 404)
    if mode not in overlay.MODES:
        raise ApiError("invalid_request", f"mode must be one of {', '.join(overlay.MODES)}", 400)
    if not (0.0 <= opacity <= 1.0) or not (1 <= thickness <= 3) or not (1 <= point_radius <= 5):
        raise ApiError("invalid_request", "opacity 0 to 1, thickness 1 to 3, point size 1 to 5", 400)
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
    drawn = overlay.render_overlay(frame, frames[index], mode=mode, opacity=opacity,
                                   thickness=thickness, point_radius=point_radius)
    ok, buf = cv2.imencode(".jpg", drawn, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ApiError("encode_failed", "the image could not be encoded", 500)
    return buf.tobytes()


def video_info(person: str, session: str) -> dict:
    """Basic facts about the recording, read from the file (as the Streamlit page shows them)."""
    import cv2

    _check_session(person, session)
    video = pp.session_video(person, session)
    if not video.is_file():
        raise ApiError("no_video", "the recording for this session is missing", 404)
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return {
        "fps": fps if fps > 0 else None,
        "frames": frames,
        "width": width,
        "height": height,
        "duration_s": (frames / fps) if fps > 0 else None,
    }


def blendshape_frame(person: str, session: str, index: int) -> dict:
    """All blendshape scores for one frame, highest first (the Streamlit frame table, as JSON)."""
    _check_session(person, session)
    frames = _face_frames(person, session)
    if not (0 <= index < len(frames)):
        raise ApiError("not_found", "frame index out of range", 404)
    record = frames[index]
    values = []
    for name, score in (record.get("blendshapes") or {}).items():
        value = _scalar(score)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append([name, value])
    values.sort(key=lambda pair: pair[1], reverse=True)
    return {
        "index": index,
        "time_s": _scalar(record.get("timestamp_ms", 0) / 1000.0),
        "face_detected": bool(record.get("face_detected", False)),
        "values": values,
    }


def blendshape_series(person: str, session: str, names: list[str]) -> dict:
    """Scores over the whole recording for the chosen blendshapes. Names the recording does not have are left out."""
    _check_session(person, session)
    if not names:
        raise ApiError("invalid_request", "name at least one blendshape", 400)
    if len(names) > 60:
        raise ApiError("invalid_request", "at most 60 blendshapes at a time", 400)
    frames = _face_frames(person, session)
    present = set()
    for record in frames:
        present.update((record.get("blendshapes") or {}).keys())
    chosen = [n for n in dict.fromkeys(names) if n in present]
    values = {n: [] for n in chosen}
    time_s = []
    for record in frames:
        time_s.append(_scalar(record.get("timestamp_ms", 0) / 1000.0))
        blend = record.get("blendshapes") or {}
        for n in chosen:
            values[n].append(_scalar(blend.get(n)))
    return {"time_s": time_s, "values": values, "missing": [n for n in dict.fromkeys(names) if n not in present]}
