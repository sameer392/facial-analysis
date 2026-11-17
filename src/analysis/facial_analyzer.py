"""Load extracted facial data for a session and compute tracking metrics.

Every session is addressed by (person, session), for example ("sameer_gupta", "session-001").
Inputs:  recordings/raw/<person>/<session>.mp4 and datasets/<person>/<session>/facial_data.json
Outputs: datasets/<person>/<session>/ (motion_metrics.csv, blink_events.csv,
         tracking_quality.json, calibration.json) and datasets/<person>/dataset_metadata.json

Run as a script to rebuild outputs for every processed session of a person:
    python src/analysis/facial_analyzer.py <person>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from facial_3d_analyzer import head_pose_angles
from motion_analyzer import (
    ACTIVITY_PREFIXES,
    compensate_head_pose,
    landmark_kinematics,
    neutral_frame_mask,
    neutral_shape,
)
from project_paths import (
    list_processed_sessions,
    person_dataset_dir,
    session_dataset_dir,
    session_video,
)

N_LANDMARKS = 478
BLINK_LEFT = "eyeBlinkLeft"
BLINK_RIGHT = "eyeBlinkRight"
BLINK_THRESHOLD = 0.5
DEFAULT_FPS = 30.0


@dataclass(frozen=True)
class Session:
    person: str
    session_id: str
    video_path: Path
    dataset_dir: Path
    fps: float

    @property
    def json_path(self) -> Path:
        return self.dataset_dir / "facial_data.json"

    @property
    def csv_path(self) -> Path:
        return self.dataset_dir / "facial_blendshapes.csv"


@dataclass
class FacialDataset:
    session: Session
    timestamps_s: np.ndarray      # (T,)
    detected: np.ndarray          # (T,) bool
    landmarks: np.ndarray         # (T, 478, 3), NaN where no face
    matrices: np.ndarray          # (T, 4, 4), NaN where unavailable
    blendshapes: pd.DataFrame     # (T, n_blendshapes), NaN where no face


@dataclass
class SessionAnalysis:
    session_id: str
    table: pd.DataFrame           # one row per frame, session_id on every row
    blinks: pd.DataFrame          # one row per blink event, session_id on every row
    quality: dict
    neutral_frames: int
    compensated: np.ndarray       # (T, 478, 3) head-pose compensated landmarks


def video_fps(video_path: Path) -> float:
    if not video_path.is_file():
        return DEFAULT_FPS
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else DEFAULT_FPS


def make_session(person: str, session_id: str) -> Session:
    return Session(
        person=person,
        session_id=session_id,
        video_path=session_video(person, session_id),
        dataset_dir=session_dataset_dir(person, session_id),
        fps=video_fps(session_video(person, session_id)),
    )


def processed_sessions(person: str) -> list[Session]:
    return [make_session(person, s) for s in list_processed_sessions(person)]


def load_session(session: Session) -> FacialDataset:
    with session.json_path.open("r", encoding="utf-8") as f:
        frames = json.load(f)

    n = len(frames)
    timestamps = np.zeros(n)
    detected = np.zeros(n, dtype=bool)
    landmarks = np.full((n, N_LANDMARKS, 3), np.nan)
    matrices = np.full((n, 4, 4), np.nan)
    blend_rows = []

    for i, fr in enumerate(frames):
        timestamps[i] = fr["timestamp_ms"] / 1000.0
        detected[i] = bool(fr.get("face_detected", False))
        for p in fr.get("landmarks", []):
            landmarks[i, p["id"]] = (p["x"], p["y"], p["z"])
        if fr.get("transformation_matrix"):
            matrices[i] = np.asarray(fr["transformation_matrix"], dtype=float)
        blend_rows.append(fr.get("blendshapes", {}))

    blendshapes = pd.DataFrame(blend_rows).astype(float)
    return FacialDataset(session, timestamps, detected, landmarks, matrices, blendshapes)


def blink_score(blendshapes: pd.DataFrame) -> pd.Series:
    if BLINK_LEFT not in blendshapes or BLINK_RIGHT not in blendshapes:
        return pd.Series(np.nan, index=blendshapes.index)
    return (blendshapes[BLINK_LEFT] + blendshapes[BLINK_RIGHT]) / 2.0


def detect_blinks(ds: FacialDataset, threshold: float = BLINK_THRESHOLD) -> pd.DataFrame:
    """A blink is a run of consecutive frames where the mean eyeBlink score is at or above threshold."""
    session = ds.session
    score = blink_score(ds.blendshapes).to_numpy()
    closed = np.nan_to_num(score, nan=0.0) >= threshold
    events = []
    start = None
    for i, is_closed in enumerate(np.append(closed, False)):
        if is_closed and start is None:
            start = i
        elif not is_closed and start is not None:
            end = i - 1
            events.append({
                "session_id": session.session_id,
                "start_frame": start,
                "end_frame": end,
                "start_s": float(ds.timestamps_s[start]),
                "duration_ms": (end - start + 1) * 1000.0 / session.fps,
                "peak_score": float(np.nanmax(score[start:end + 1])),
            })
            start = None
    return pd.DataFrame(events, columns=[
        "session_id", "start_frame", "end_frame", "start_s", "duration_ms", "peak_score",
    ])


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def tracking_quality(ds: FacialDataset) -> dict:
    """Summarise how reliably the face was tracked in this session."""
    detected = ds.detected
    usable_matrix = np.isfinite(ds.matrices).all(axis=(1, 2))
    both = detected[1:] & detected[:-1]
    steps = np.linalg.norm(np.diff(ds.landmarks, axis=0), axis=2)  # (T-1, 478)
    jitter = float(np.nanmedian(steps[both])) if both.any() else float("nan")
    detection_rate = float(detected.mean()) if len(detected) else 0.0
    matrix_rate = float(usable_matrix.mean()) if len(detected) else 0.0
    return {
        "session_id": ds.session.session_id,
        "frames": int(len(detected)),
        "detection_rate": round(detection_rate, 4),
        "transform_matrix_rate": round(matrix_rate, 4),
        "longest_missing_run_frames": _longest_run(~detected),
        "median_landmark_jitter": jitter,
        "quality_score": round(100.0 * detection_rate * matrix_rate, 1),
    }


def analyse_session(session: Session) -> SessionAnalysis:
    ds = load_session(session)
    T = len(ds.timestamps_s)

    compensated = compensate_head_pose(ds.landmarks, ds.matrices)
    usable = np.isfinite(compensated).all(axis=(1, 2))
    mask = neutral_frame_mask(ds.blendshapes, usable)
    neutral = neutral_shape(compensated, mask)
    offset = np.linalg.norm(compensated - neutral, axis=2)
    neutral_offset = np.where(usable, np.nanmean(offset, axis=1), np.nan)
    displacement, velocity, acceleration = landmark_kinematics(compensated, ds.timestamps_s, usable)

    def column(name: str) -> np.ndarray:
        if name in ds.blendshapes:
            return ds.blendshapes[name].to_numpy()
        return np.full(T, np.nan)

    activity_cols = [c for c in ds.blendshapes.columns if c.startswith(ACTIVITY_PREFIXES)]
    angles = head_pose_angles(ds.matrices)

    table = pd.DataFrame({
        "session_id": session.session_id,
        "frame": np.arange(T),
        "time_s": ds.timestamps_s,
        "face_detected": ds.detected,
        "neutral_offset": neutral_offset,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "jaw_open": column("jawOpen"),
        "mouth_activity": ds.blendshapes[activity_cols].sum(axis=1).to_numpy() if activity_cols else np.nan,
        "eye_blink": blink_score(ds.blendshapes).to_numpy(),
        "brow_inner_up": column("browInnerUp"),
        "yaw_deg": angles["yaw_deg"].to_numpy(),
        "pitch_deg": angles["pitch_deg"].to_numpy(),
        "roll_deg": angles["roll_deg"].to_numpy(),
    })

    return SessionAnalysis(
        session_id=session.session_id,
        table=table,
        blinks=detect_blinks(ds),
        quality=tracking_quality(ds),
        neutral_frames=int(mask.sum()),
        compensated=compensated,
    )


def save_session_outputs(analysis: SessionAnalysis, session: Session) -> Path:
    """Write derived outputs into the session's dataset folder. Original recordings are never touched."""
    out = session.dataset_dir
    out.mkdir(parents=True, exist_ok=True)
    analysis.table.to_csv(out / "motion_metrics.csv", index=False)
    analysis.blinks.to_csv(out / "blink_events.csv", index=False)
    (out / "tracking_quality.json").write_text(json.dumps(analysis.quality, indent=2), encoding="utf-8")
    (out / "calibration.json").write_text(json.dumps({
        "session_id": analysis.session_id,
        "neutral_frames": analysis.neutral_frames,
        "method": "lowest 10% of mouth/jaw/eye/brow activity among tracked frames",
        "head_pose": "rotation removed per frame; translation and scale ignored",
    }, indent=2), encoding="utf-8")
    return out


def write_person_metadata(person: str) -> Path:
    """Describe the person and every processed session in datasets/<person>/dataset_metadata.json."""
    sessions = []
    for session in processed_sessions(person):
        with session.json_path.open("r", encoding="utf-8") as f:
            n_frames = len(json.load(f))
        sessions.append({
            "session_id": session.session_id,
            "recording": session.video_path.name,
            "fps": round(session.fps, 3),
            "frames": n_frames,
        })
    path = person_dataset_dir(person) / "dataset_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "person": person,
        "identity_type": "single_person",
        "sessions": sessions,
    }, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild motion metrics for every processed session of a person.")
    parser.add_argument("person")
    args = parser.parse_args()
    sessions = processed_sessions(args.person)
    if not sessions:
        raise SystemExit(f"No processed sessions for '{args.person}'. Run extract_faces.py first.")
    for session in sessions:
        result = analyse_session(session)
        path = save_session_outputs(result, session)
        print(f"{session.session_id}: {len(result.table)} frames, {len(result.blinks)} blinks -> {path}")
    print("Metadata:", write_person_metadata(args.person))
