"""Read-only V1 data for the web pages (phase P2). Numbers come from the existing Python analysis.

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
        raise ApiError("no_v1_data", "no version 1 facial data for this session", 404)
    return facial


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
