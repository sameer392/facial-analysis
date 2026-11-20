"""Read-only hand data for the API (phase H5).

Every function here only reads. Files are opened by fixed names inside one session folder, so no request
can choose a path. CSV tables are streamed and NumPy arrays are memory-mapped, so a request reads only the
rows or frames it asks for. Errors are structured: no exception text or filesystem path is returned.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
_ANALYSIS = _SRC / "analysis"
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import project_paths as pp  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402
from pose_schema import (  # noqa: E402
    HAND_GESTURE_EVENTS_FILE,
    HAND_GESTURE_OBS_FILE,
    HAND_MOTION_FILE,
    HAND_STATUS_FILE,
    HAND_TRACKS_FILE,
    METADATA_FILE,
    session_layout,
)

PAGE_MAX = 200            # rows per table page
LANDMARK_PAGE_MAX = 100   # frames per landmark page
OFFSET_MAX = 10_000_000
HAND_PREFIX_SIDES = ("left", "right")
DETECTED = ("valid", "low_confidence", "outside_frame")


def _dataset(person: str, session: str) -> Path:
    """The session folder, validated and kept inside the datasets root."""
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    root = pp.DATASETS_ROOT.resolve()
    dataset = (pp.DATASETS_ROOT / person / session).resolve()
    if root not in dataset.parents or dataset.parent.parent != root:
        raise ApiError("invalid_name", "invalid person or session name", 400)
    if not dataset.is_dir() or session_layout(dataset) != "pose":
        raise ApiError("no_session", "no pose data for this session", 404)
    return dataset


def _meta(dataset: Path) -> dict:
    try:
        return json.loads((dataset / METADATA_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ApiError("dataset_corrupt", "the session metadata cannot be read", 422) from None


def _hands(dataset: Path) -> dict:
    """The hand stream record. Only a complete stream counts: a pose build writes the hand stream as
    "absent" or "not run", which is not hand analysis."""
    hands = _meta(dataset).get("streams", {}).get("hands")
    if not isinstance(hands, dict) or hands.get("status") != "complete":
        raise ApiError("no_hand_analysis", "no hand analysis for this session yet", 404)
    return hands


def _page_args(offset: int, limit: int, maximum: int = PAGE_MAX) -> tuple[int, int]:
    if not (0 <= offset <= OFFSET_MAX) or not (1 <= limit <= maximum):
        raise ApiError("invalid_pagination", f"offset 0 to {OFFSET_MAX}, limit 1 to {maximum}", 400)
    return offset, limit


def _convert(value: str):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer() and "." not in value and "e" not in value.lower():
        return int(number)
    return number if math.isfinite(number) else None


def _csv_page(dataset: Path, name: str, offset: int, limit: int) -> dict:
    """Rows [offset, offset + limit) of a table, streamed. Only the requested rows are kept in memory."""
    path = dataset / name
    if not path.is_file():
        raise ApiError("no_hand_analysis", "this table is not in the session yet", 404)
    rows, total = [], 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                total = i + 1
                if offset <= i < offset + limit:
                    rows.append({k: _convert(v) for k, v in row.items()})
    except (OSError, csv.Error, UnicodeDecodeError):
        raise ApiError("dataset_corrupt", "the table cannot be read", 422) from None
    return {"total": total, "offset": offset, "limit": limit, "rows": rows}


def summary(person: str, session: str) -> dict:
    dataset = _dataset(person, session)
    hands = _hands(dataset)
    meta = _meta(dataset)
    out = {
        "session_id": session,
        "streams": {"hands": hands.get("status")},
        "tracking": bool(hands.get("tracking")),
        "gestures": bool(hands.get("gestures")),
        "source": {k: meta.get("source", {}).get(k) for k in ("fps", "frame_count", "width", "height")},
        "models": {"hand": {"file": meta.get("models", {}).get("hand", {}).get("file"),
                            "version": meta.get("models", {}).get("hand", {}).get("version"),
                            "licence_status": meta.get("models", {}).get("hand", {}).get("licence_status")}},
        "detected_frames": hands.get("detected_frames"),
    }
    counts = {"left": {}, "right": {}}
    status_path = dataset / HAND_STATUS_FILE
    if status_path.is_file():
        try:
            with status_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    for side in HAND_PREFIX_SIDES:
                        counts[side][row[f"{side}_status"]] = counts[side].get(row[f"{side}_status"], 0) + 1
        except (OSError, csv.Error, KeyError, UnicodeDecodeError):
            raise ApiError("dataset_corrupt", "the hand status table cannot be read", 422) from None
    out["status_counts"] = counts
    return out


def landmarks(person: str, session: str, side: str, start: int, limit: int) -> dict:
    """Landmarks for frames [start, start + limit) on one side. Frames without that hand have null points."""
    if side not in HAND_PREFIX_SIDES:
        raise ApiError("invalid_request", "side must be left or right", 400)
    start, limit = _page_args(start, limit, LANDMARK_PAGE_MAX)
    dataset = _dataset(person, session)
    _hands(dataset)
    path = dataset / "hands" / f"{side}_hand.npy"
    if not path.is_file():
        raise ApiError("no_hand_analysis", "the hand landmarks are not in the session yet", 404)
    try:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        raise ApiError("dataset_corrupt", "the hand landmarks cannot be read", 422) from None
    n = int(arr.shape[0])
    end = min(start + limit, n)
    statuses = _status_slice(dataset, side, start, end)
    frames = []
    for f in range(start, end):
        pts = np.asarray(arr[f])
        detected = statuses.get(f) in DETECTED and bool(np.all(np.isfinite(pts)))
        frames.append({
            "frame": f,
            "status": statuses.get(f),
            "points": [[_finite(x), _finite(y), _finite(z)] for x, y, z in pts.tolist()] if detected else None,
        })
    return {"side": side, "total_frames": n, "start": start, "limit": limit, "frames": frames}


def _finite(v):
    return float(v) if math.isfinite(v) else None


def _status_slice(dataset: Path, side: str, start: int, end: int) -> dict:
    path = dataset / HAND_STATUS_FILE
    out = {}
    if not path.is_file():
        return out
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if i >= end:
                    break
                if i >= start:
                    out[i] = row[f"{side}_status"]
    except (OSError, csv.Error, KeyError, UnicodeDecodeError):
        raise ApiError("dataset_corrupt", "the hand status table cannot be read", 422) from None
    return out


def tracks(person: str, session: str, offset: int, limit: int) -> dict:
    dataset = _dataset(person, session)
    _hands(dataset)
    offset, limit = _page_args(offset, limit)
    return _csv_page(dataset, HAND_TRACKS_FILE, offset, limit)


def motion(person: str, session: str, offset: int, limit: int) -> dict:
    dataset = _dataset(person, session)
    _hands(dataset)
    offset, limit = _page_args(offset, limit)
    return _csv_page(dataset, HAND_MOTION_FILE, offset, limit)


def gesture_observations(person: str, session: str, offset: int, limit: int) -> dict:
    dataset = _dataset(person, session)
    _hands(dataset)
    offset, limit = _page_args(offset, limit)
    return _csv_page(dataset, HAND_GESTURE_OBS_FILE, offset, limit)


def gesture_events(person: str, session: str, offset: int, limit: int) -> dict:
    dataset = _dataset(person, session)
    _hands(dataset)
    offset, limit = _page_args(offset, limit)
    return _csv_page(dataset, HAND_GESTURE_EVENTS_FILE, offset, limit)


def quality(person: str, session: str, which: str) -> dict:
    """The tracking or gesture part of the quality summary, from quality/tracking_quality.json."""
    if which not in ("tracking", "gestures"):
        raise ApiError("invalid_request", "which must be tracking or gestures", 400)
    dataset = _dataset(person, session)
    _hands(dataset)
    path = dataset / "quality" / "tracking_quality.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ApiError("dataset_corrupt", "the quality summary cannot be read", 422) from None
    part = data.get("hands", {}).get(which)
    if part is None:
        raise ApiError("no_hand_analysis", f"no {which} summary in this session yet", 404)
    return part


def validation(person: str, session: str) -> dict:
    """Run the read-only H4 validator. Reads the session; never writes it."""
    from hand_dataset_validation import validate_hand_session

    dataset = _dataset(person, session)
    _hands(dataset)
    try:
        return validate_hand_session(dataset).to_dict()
    except Exception:  # never return internal details
        raise ApiError("dataset_corrupt", "the dataset could not be validated", 422) from None


def overlay_jpeg(person: str, session: str, index: int) -> bytes:
    """The video frame with the detected hands drawn on it (points only)."""
    import cv2

    dataset = _dataset(person, session)
    _hands(dataset)
    meta = _meta(dataset)
    n = int(meta.get("source", {}).get("frame_count") or 0)
    if not (0 <= index < n):
        raise ApiError("frame_out_of_range", "frame index out of range", 404)
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
        raise ApiError("frame_out_of_range", "the video frame could not be read", 404)
    h, w = frame.shape[:2]
    statuses = {}
    for side in HAND_PREFIX_SIDES:
        statuses[side] = _status_slice(dataset, side, index, index + 1).get(index)
    colours = {"left": (0, 200, 255), "right": (255, 120, 0)}
    for side in HAND_PREFIX_SIDES:
        path = dataset / "hands" / f"{side}_hand.npy"
        if statuses[side] not in DETECTED or not path.is_file():
            continue
        pts = np.load(path, mmap_mode="r")[index]
        if not np.all(np.isfinite(pts)):
            continue
        for x, y, _z in pts.tolist():
            cv2.circle(frame, (int(round(x * w)), int(round(y * h))), 3, colours[side], -1)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ApiError("encode_failed", "the image could not be encoded", 500)
    return buf.tobytes()
