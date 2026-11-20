"""Support code for the Body Motion page: discovery and reading of pose data sessions, joint states,
the video overlay, and a tracking-quality summary.

Nothing here imports Streamlit, so it can be tested directly. Stored pose data arrays are only read.

Coordinates: pose x and y are normalised to the video frame (x / width, y / height, origin top-left,
y down). A pixel position is x * width and y * height. Joints whose coordinates lie outside [0, 1]
are off the frame. Visibility is the model's own estimate and is not a measure of accuracy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import project_paths as pp  # noqa: E402
from pose_schema import session_layout, validate_metadata  # noqa: E402
from pose_writer import verify_manifest  # noqa: E402

N_POSE = 33
POSE_CHANNELS = ("x", "y", "z", "visibility", "presence")
TIMELINE_REQUIRED = ("frame", "timestamp_ms", "face_detected", "pose_detected")
STATES = ("shown", "low_visibility", "off_frame", "missing")

SHOWN_COLOUR = (0, 255, 0)
LOW_VIS_COLOUR = (150, 150, 150)
EDGE_COLOUR = (0, 200, 0)


class PoseDataViewerError(Exception):
    """A pose data session or its arrays cannot be read safely."""




@dataclass(frozen=True)
class PoseSession:
    person: str
    session_id: str
    path: Path
    frames: int
    width: int
    height: int
    fps: float


@lru_cache(maxsize=1)
def joint_names() -> tuple[str, ...]:
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark

    return tuple(m.name.lower().replace("_", " ") for m in sorted(PoseLandmark, key=lambda m: m.value))


@lru_cache(maxsize=1)
def pose_edges() -> tuple[tuple[int, int], ...]:
    """MediaPipe's official pose skeleton (35 edges), read once."""
    from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

    return tuple((c.start, c.end) for c in PoseLandmarksConnections.POSE_LANDMARKS)


def load_pose(session_dir) -> np.ndarray:
    path = Path(session_dir) / "body" / "pose_landmarks.npy"
    if not path.is_file():
        raise PoseDataViewerError(f"pose array missing: {path.name}")
    try:
        arr = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise PoseDataViewerError(f"pose array unreadable: {exc}") from None
    if arr.ndim != 3 or arr.shape[1:] != (N_POSE, len(POSE_CHANNELS)):
        raise PoseDataViewerError(f"pose array has shape {arr.shape}, expected (T, {N_POSE}, {len(POSE_CHANNELS)})")
    if arr.dtype != np.float32:
        raise PoseDataViewerError(f"pose array has dtype {arr.dtype}, expected float32")
    return arr


def load_timeline(session_dir, expected_frames: int | None = None) -> pd.DataFrame:
    path = Path(session_dir) / "timeline.csv"
    if not path.is_file():
        raise PoseDataViewerError("timeline.csv missing")
    df = pd.read_csv(path)
    missing = [c for c in TIMELINE_REQUIRED if c not in df.columns]
    if missing:
        raise PoseDataViewerError(f"timeline.csv lacks columns: {missing}")
    if expected_frames is not None and len(df) != expected_frames:
        raise PoseDataViewerError(f"timeline has {len(df)} rows but the pose array has {expected_frames} frames")
    if not (df["frame"].to_numpy() == np.arange(len(df))).all():
        raise PoseDataViewerError("timeline frame numbers are not sequential from 0")
    return df


def validate_pose_session(session_dir, person: str, session_id: str) -> PoseSession:
    session = Path(session_dir)
    try:
        meta = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PoseDataViewerError("metadata.json is unreadable") from None
    issues = validate_metadata(meta)
    if issues:
        raise PoseDataViewerError("metadata invalid: " + "; ".join(issues))
    try:
        manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise PoseDataViewerError("manifest.json is unreadable") from None
    problems = verify_manifest(session, manifest, check_unexpected=True)
    if problems:
        raise PoseDataViewerError("manifest verification failed: " + "; ".join(problems[:3]))
    pose = load_pose(session)
    load_timeline(session, expected_frames=pose.shape[0])
    source = meta["source"]
    return PoseSession(person=person, session_id=session_id, path=session, frames=pose.shape[0],
                     width=int(source["width"]), height=int(source["height"]), fps=float(source["fps"]))


def discover_pose_sessions(root=None):
    """Return (valid pose data sessions, rejected as (path, reason)). Face data sessions are not listed."""
    base = Path(root) if root is not None else pp.DATASETS_ROOT
    valid: list[PoseSession] = []
    rejected: list[tuple[Path, str]] = []
    if not base.is_dir():
        return valid, rejected
    for person_dir in sorted(p for p in base.iterdir() if p.is_dir() and pp.is_valid_person(p.name)):
        for session_dir in sorted(p for p in person_dir.iterdir() if p.is_dir() and pp.is_valid_session(p.name)):
            layout = session_layout(session_dir)
            if layout in ("face", "none"):
                continue
            if layout == "pose_incomplete":
                rejected.append((session_dir, "incomplete pose data output (metadata.json missing or unreadable)"))
                continue
            try:
                valid.append(validate_pose_session(session_dir, person_dir.name, session_dir.name))
            except PoseDataViewerError as exc:
                rejected.append((session_dir, str(exc)))
    return valid, rejected


def pixel_positions(frame_pose: np.ndarray, width: int, height: int) -> np.ndarray:
    """(33, 2) pixel positions from normalised x and y. Missing joints are NaN."""
    px = np.full((N_POSE, 2), np.nan, dtype=float)
    finite = np.isfinite(frame_pose[:, :2]).all(axis=1)
    px[finite, 0] = frame_pose[finite, 0] * width
    px[finite, 1] = frame_pose[finite, 1] * height
    return px


def joint_states(frame_pose: np.ndarray, threshold: float) -> list[str]:
    """State of each joint for one frame: missing, low_visibility, off_frame or shown.

    A joint with missing coordinates is "missing". One below the visibility threshold is
    "low_visibility", whether or not it is on the frame. A joint above the threshold is "shown" if
    it lies inside the frame, and "off_frame" if not.
    """
    states = []
    for k in range(N_POSE):
        x, y, vis = frame_pose[k, 0], frame_pose[k, 1], frame_pose[k, 3]
        if not (np.isfinite(x) and np.isfinite(y)):
            states.append("missing")
        elif not (np.isfinite(vis) and vis >= threshold):
            states.append("low_visibility")
        elif 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            states.append("shown")
        else:
            states.append("off_frame")
    return states


def draw_pose_overlay(image_bgr: np.ndarray, frame_pose: np.ndarray, threshold: float,
                      show_points: bool = True, show_skeleton: bool = True,
                      thickness: int = 2, radius: int = 3) -> np.ndarray:
    """Return a copy of the frame with the pose drawn. Shown joints are green, low-visibility grey,
    and off-frame or missing joints are not drawn. The skeleton links only joints that are shown."""
    out = image_bgr.copy()
    if not np.isfinite(frame_pose[:, :2]).any():
        return out
    h, w = image_bgr.shape[:2]
    px = pixel_positions(frame_pose, w, h)
    states = joint_states(frame_pose, threshold)
    if show_skeleton:
        for a, b in pose_edges():
            if states[a] == "shown" and states[b] == "shown":
                cv2.line(out, (int(round(px[a, 0])), int(round(px[a, 1]))),
                         (int(round(px[b, 0])), int(round(px[b, 1]))), EDGE_COLOUR, thickness, cv2.LINE_AA)
    if show_points:
        for k, state in enumerate(states):
            if state in ("shown", "low_visibility") and np.isfinite(px[k]).all():
                colour = SHOWN_COLOUR if state == "shown" else LOW_VIS_COLOUR
                cv2.circle(out, (int(round(px[k, 0])), int(round(px[k, 1]))), radius, colour, -1, cv2.LINE_AA)
    return out


def landmark_table(frame_pose: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per-joint values for one frame, with the state at the selected threshold."""
    names = joint_names()
    states = joint_states(frame_pose, threshold)
    rows = []
    for k in range(N_POSE):
        rows.append({
            "index": k,
            "joint": names[k],
            "x": float(frame_pose[k, 0]),
            "y": float(frame_pose[k, 1]),
            "z": float(frame_pose[k, 2]),
            "visibility": float(frame_pose[k, 3]),
            "presence": float(frame_pose[k, 4]),
            "state": states[k],
        })
    return pd.DataFrame(rows)


def quality_summary(pose: np.ndarray, timeline: pd.DataFrame, threshold: float, selected_frame: int) -> dict:
    """Pose tracking quality from timeline.csv (the source of truth for detection flags).

    These are tracking figures only. They do not measure whether the body motion is accurate.
    """
    detected = timeline["pose_detected"].astype(bool).to_numpy()
    frames = len(detected)
    finite_detected = np.isfinite(pose[detected][:, :, :2]).all(axis=2) if detected.any() else np.zeros((0, N_POSE), bool)
    vis = pose[detected][:, :, 3][finite_detected] if detected.any() else np.array([])
    shown_total = 0
    for f in range(frames):
        if detected[f]:
            shown_total += sum(s == "shown" for s in joint_states(pose[f], threshold))
    return {
        "frames": int(frames),
        "pose_detected_frames": int(detected.sum()),
        "pose_detection_rate": float(detected.mean()) if frames else 0.0,
        "missing_pose_frames": np.nonzero(~detected)[0].tolist(),
        "median_visibility": float(np.median(vis)) if vis.size else None,
        "mean_visibility": float(np.mean(vis)) if vis.size else None,
        "valid_joints_selected_frame": sum(s == "shown" for s in joint_states(pose[selected_frame], threshold)),
        "valid_joint_share_detected_frames": (shown_total / (int(detected.sum()) * N_POSE)) if detected.any() else 0.0,
        "threshold": float(threshold),
        "scope": "tracking only; does not measure the accuracy of the body motion",
    }
