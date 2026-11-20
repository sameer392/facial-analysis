"""Native video playback support (phase H8.2): local-only guard, original recording lookup, landmark windows and a
timing check between the recording and the extracted timeline.

Everything here is read-only. The original recording is never copied, re-encoded or written. Only the routes that
use the local guard (see api/main.py) accept requests from the browser; no existing face or pose route is changed.

Landmark window format, version 1 (little-endian):
    bytes 0-3    magic b"HLW1"
    bytes 4-7    uint32 length of the JSON header, N
    bytes 8..    N bytes of UTF-8 JSON header (see build_header)
    then         the stream arrays, float32, C order, in the order and at the offsets the header declares.
                 Missing values are NaN. Offsets are relative to the start of the data section.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
from fastapi import Request

_SRC = Path(__file__).resolve().parents[1] / "src"
_ANALYSIS = _SRC / "analysis"
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import project_paths as pp  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402
from pose_schema import METADATA_FILE, N_FACE_LANDMARKS, N_HAND_LANDMARKS, N_POSE_LANDMARKS, session_layout  # noqa: E402

# ---- local-only guard ------------------------------------------------------------------------------

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1:8000", "localhost:8000")
ALLOWED_HOSTS_ENV = "FACIAL_ALLOWED_HOSTS"


def allowed_hosts() -> frozenset[str]:
    """The Host header values this API accepts. Set FACIAL_ALLOWED_HOSTS (comma separated) to change them."""
    raw = os.environ.get(ALLOWED_HOSTS_ENV, "")
    values = [v.strip().lower() for v in raw.split(",") if v.strip()] or list(DEFAULT_ALLOWED_HOSTS)
    return frozenset(values)


def check_local_request(host: str | None, sec_fetch_site: str | None) -> None:
    """Refuse requests that did not come from this machine's own origin.

    Host must be one of the configured local names. X-Forwarded-Host is never consulted. Sec-Fetch-Site must be
    absent (non-browser clients such as curl) or exactly "same-origin". "same-site", "cross-site" and "none" are
    refused. The Next.js proxy sets Host to the API address, so its requests pass.
    """
    if (host or "").strip().lower() not in allowed_hosts():
        raise ApiError("host_not_allowed", "this request's host is not a configured local address", 403)
    if sec_fetch_site is not None and sec_fetch_site.strip().lower() != "same-origin":
        raise ApiError("cross_site_request", "requests from other sites are refused", 403)


def local_only(request: Request) -> None:
    """FastAPI dependency: applies check_local_request to the request's headers."""
    check_local_request(request.headers.get("host"), request.headers.get("sec-fetch-site"))


# ---- original recording ----------------------------------------------------------------------------

def recording_path(person: str, session: str) -> Path:
    """The original recording for a session, if it is a regular .mp4 inside the recordings root.

    Names are validated first. The path is then resolved, and must still lie inside the recordings root, which
    rejects symlink escapes. Errors never include the path.
    """
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    expected = pp.session_video(person, session)
    if expected.is_symlink() or not expected.is_file():
        raise ApiError("no_video", "the recording for this session is missing", 404)
    root = pp.RECORDINGS_RAW.resolve()
    resolved = expected.resolve()
    if root not in resolved.parents:
        raise ApiError("no_video", "the recording for this session is missing", 404)
    if resolved.suffix.lower() != ".mp4":
        raise ApiError("unsupported_media", "only mp4 recordings can be played", 415)
    return resolved


# ---- landmark windows ------------------------------------------------------------------------------

WINDOW_DEFAULT = 30
WINDOW_MAX = 30
MAGIC = b"HLW1"
FORMAT_VERSION = 1

# stream name -> (file in the pose session, channel names kept, channel indexes in the file)
STREAMS = {
    "face": ("facial/landmarks.npy", ("x", "y"), (0, 1), N_FACE_LANDMARKS, 3),
    "body": ("body/pose_landmarks.npy", ("x", "y", "visibility"), (0, 1, 3), N_POSE_LANDMARKS, 5),
    "left_hand": ("hands/left_hand.npy", ("x", "y"), (0, 1), N_HAND_LANDMARKS, 3),
    "right_hand": ("hands/right_hand.npy", ("x", "y"), (0, 1), N_HAND_LANDMARKS, 3),
}


def _session_dir(person: str, session: str) -> Path:
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    dataset = pp.session_dataset_dir(person, session)
    if not dataset.is_dir() or session_layout(dataset) != "pose":
        raise ApiError("no_pose_data", "no pose data for this session", 404)
    return dataset


def _timeline_rows(dataset: Path, start: int, end: int) -> list[dict]:
    rows = []
    try:
        with (dataset / "timeline.csv").open(encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if i >= end:
                    break
                if i >= start:
                    rows.append(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        raise ApiError("dataset_corrupt", "the timeline cannot be read", 422) from None
    return rows


def _status_rows(dataset: Path, start: int, end: int) -> list[dict]:
    path = dataset / "hands" / "hand_status.csv"
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(itertools.islice(csv.DictReader(handle), start, end))
    except (OSError, csv.Error, UnicodeDecodeError):
        raise ApiError("dataset_corrupt", "the hand status table cannot be read", 422) from None


def _count_frames(dataset: Path) -> int:
    meta_path = dataset / METADATA_FILE
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["source"]["frame_count"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ApiError("dataset_corrupt", "the session metadata cannot be read", 422) from None


def _window_bounds(start: int, count: int, total: int) -> tuple[int, int]:
    if count < 1 or count > WINDOW_MAX:
        raise ApiError("invalid_window", f"count must be 1 to {WINDOW_MAX} frames", 400)
    if start < 0 or start + count > total:
        raise ApiError("invalid_window", "the window runs outside the recording", 400)
    return start, start + count


def build_window(person: str, session: str, start: int, count: int) -> bytes:
    """One landmark window, in the version 1 binary format described at the top of this module."""
    dataset = _session_dir(person, session)
    total = _count_frames(dataset)
    start, end = _window_bounds(start, count, total)
    n = end - start
    timeline = _timeline_rows(dataset, start, end)
    if len(timeline) != n:
        raise ApiError("dataset_corrupt", "the timeline is shorter than the recording", 422)

    arrays: list[np.ndarray] = []
    streams_meta = {}
    offset = 0
    for name, (rel, channel_names, idx, expected_points, expected_channels) in STREAMS.items():
        path = dataset / rel
        if not path.is_file():
            streams_meta[name] = {"present": False, "reason": "not in this session"}
            continue
        try:
            full = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError):
            raise ApiError("dataset_corrupt", "landmark data cannot be read", 422) from None
        if full.ndim != 3 or full.shape[0] != total or full.shape[1] != expected_points \
                or full.shape[2] != expected_channels:
            raise ApiError("dataset_corrupt", "landmark array shape does not match the schema", 422)
        block = np.ascontiguousarray(full[start:end][:, :, list(idx)], dtype="<f4")
        arrays.append(block)
        streams_meta[name] = {
            "present": True, "channels": list(channel_names), "dtype": "float32",
            "shape": list(block.shape), "offset": offset, "bytes": int(block.nbytes),
        }
        offset += int(block.nbytes)

    status = _status_rows(dataset, start, end)
    header = {
        "format": "HLW",
        "version": FORMAT_VERSION,
        "start": start,
        "count": n,
        "total_frames": total,
        "frames": list(range(start, end)),
        "timestamps_ms": [int(float(r["timestamp_ms"])) for r in timeline],
        "face_detected": [r.get("face_detected", "") == "True" for r in timeline],
        "pose_detected": [r.get("pose_detected", "") == "True" for r in timeline],
        "hand_status": {
            "left": [r.get("left_status", "") for r in status] if status else None,
            "right": [r.get("right_status", "") for r in status] if status else None,
        },
        "streams": streams_meta,
        "coordinates": "normalised image coordinates: x / frame width, y / frame height, origin top-left",
    }
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data = b"".join(a.tobytes() for a in arrays)
    return MAGIC + struct.pack("<I", len(body)) + body + data


# ---- timing check ----------------------------------------------------------------------------------

TIMING_TOLERANCE_MS = 2.0       # largest allowed gap between a decoded frame time and the extracted timeline
TIMING_MAX_FRAMES = 12000       # longer recordings are not checked on request (reported as unavailable)
_timing_cache: dict = {}


def timing_check(person: str, session: str) -> dict:
    """Compare each decoded frame's presentation time with the extracted timeline.

    status "synchronized": every frame was checked and every gap is within TIMING_TOLERANCE_MS.
    status "deviation":    a gap is larger than the tolerance, or the decoder and the timeline disagree on the count.
    status "unavailable":  the recording cannot be read, or it is too long to check on request.
    Overlays must stay off unless the status is "synchronized".
    """
    import cv2

    dataset = _session_dir(person, session)
    recording = recording_path(person, session)
    stat = recording.stat()
    key = (str(recording), stat.st_mtime_ns, stat.st_size)
    if key in _timing_cache:
        return _timing_cache[key]

    total = _count_frames(dataset)
    timeline = _timeline_rows(dataset, 0, total)
    if len(timeline) != total:
        return {"status": "unavailable", "reason": "the timeline is shorter than the recording"}
    if total > TIMING_MAX_FRAMES:
        return {"status": "unavailable", "reason": "the recording is too long to check on request"}
    cap = cv2.VideoCapture(str(recording))
    try:
        if not cap.isOpened():
            result = {"status": "unavailable", "reason": "the recording cannot be opened"}
        else:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            decoded: list[float] = []
            while len(decoded) <= total:
                ok, _frame = cap.read()
                if not ok:
                    break
                decoded.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
            ts = np.array([int(float(r["timestamp_ms"])) for r in timeline], dtype=float)
            count_ok = len(decoded) == total
            checked = min(len(decoded), total)
            gaps = np.abs(np.array(decoded[:checked]) - ts[:checked]) if checked else np.array([np.inf])
            worst = float(np.max(gaps))
            if count_ok and worst <= TIMING_TOLERANCE_MS:
                status, reason = "synchronized", ""
            else:
                status = "deviation"
                reason = ("the decoder finds a different number of frames" if not count_ok
                          else f"a frame time differs from the timeline by up to {worst:.2f} ms")
            result = {"status": status, "reason": reason, "frames_checked": checked,
                      "frames_in_timeline": total, "max_gap_ms": round(worst, 3),
                      "tolerance_ms": TIMING_TOLERANCE_MS, "fps_reported": round(fps, 3)}
    finally:
        cap.release()
    if len(_timing_cache) > 16:
        _timing_cache.clear()
    _timing_cache[key] = result
    return result
