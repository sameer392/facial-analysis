"""Generate synthetic demo recordings and datasets. No real person or recording is used.

A cartoon face is drawn into a short MP4 and its 478 landmarks, 52-style blendshapes and
transformation matrices are computed from the same geometry, so the dashboard overlay lines
up with the picture. The output has the same layout as real data:

    <home>/recordings/raw/demo_person/session-00N.mp4
    <home>/datasets/demo_person/session-00N/facial_data.json, facial_blendshapes.csv, ...
    <home>/datasets/demo_person/dataset_metadata.json

Usage (writes to a folder of your choice, never to the real project data by default):
    python src/analysis/demo_data.py <home-folder> [--force]
    set FACIAL_DATA_HOME=<home-folder>   then run the dashboard to browse the demo
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DEMO_PERSON = "demo_person"
FPS = 30
WIDTH, HEIGHT = 640, 480
N_FRAMES = 240
N_LANDMARKS = 478

# Each session has a different movement pattern, so the comparison page has something to show.
SESSIONS = {
    "session-001": {"talk_hz": 1.2, "blinks": (40, 150, 210), "no_face": (), "yaw_deg": 6.0},
    "session-002": {"talk_hz": 0.6, "blinks": (25, 90, 180), "no_face": (120, 124), "yaw_deg": 12.0},
    "session-003": {"talk_hz": 1.8, "blinks": (60, 130), "no_face": (), "yaw_deg": 3.0},
}

# Angles around the face outline (radians). Points in the lower half move when the jaw opens.
_ANGLES = np.linspace(0.0, 2.0 * math.pi, N_LANDMARKS, endpoint=False)
_LOWER = np.sin(_ANGLES) > 0.35
_EYES = (np.abs(np.cos(_ANGLES)) > 0.5) & (np.sin(_ANGLES) < -0.15) & (np.sin(_ANGLES) > -0.6)


def _geometry(t: float, spec: dict, index: int, n_frames: int):
    """Head centre, roll, jaw opening, blink strength and blendshape scores for one frame."""
    talk = 0.5 + 0.5 * math.sin(2 * math.pi * spec["talk_hz"] * t)
    jaw = 0.05 + 0.6 * talk ** 2
    blink = 0.0
    for start in spec["blinks"]:
        if start <= index < start + 6:
            blink = math.sin(math.pi * (index - start) / 6.0)
    yaw = math.radians(spec["yaw_deg"]) * math.sin(2 * math.pi * t / 4.0)
    centre = np.array([WIDTH / 2 + 25 * math.sin(2 * math.pi * t / 5.0), HEIGHT / 2 + 4 * math.sin(2 * math.pi * t / 3.0)])
    roll = math.radians(2.0) * math.sin(2 * math.pi * t / 6.0)
    return centre, roll, yaw, jaw, blink


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Same convention as tests/test_scientific.py: the image x-axis is mirrored relative to the
# matrix's rotation frame. This is the convention the compensation expects.
_AXIS_FLIP = np.diag([-1.0, 1.0, 1.0])


def _landmarks(centre, roll, yaw, jaw, blink):
    """Canonical 3D face deformed by expression, rotated by the head pose, then projected.

    Returns normalised landmarks (x / W, y / H, z / W) and the 3x3 rotation that produced them,
    so the transformation matrix and the landmarks are consistent.
    """
    rx, ry = 120.0, 165.0
    x = rx * np.cos(_ANGLES)
    y = ry * np.sin(_ANGLES)
    y = y + np.where(_LOWER, 45.0 * jaw, 0.0)  # lower lip and chin drop when the jaw opens
    squeeze = 1.0 - 0.85 * blink
    y = np.where(_EYES, -45 + (y + 45) * squeeze, y)
    z = 40.0 * np.cos(_ANGLES) * np.sin(_ANGLES)
    canonical = np.stack([x, y, z], axis=1)
    R = _rot_z(roll) @ _rot_y(yaw)
    iso = (canonical @ R.T) @ _AXIS_FLIP + np.array([centre[0], centre[1], 0.0])
    normalised = np.stack([iso[:, 0] / WIDTH, iso[:, 1] / HEIGHT, iso[:, 2] / WIDTH], axis=1)
    return normalised, R


def _draw(frame_bgr: np.ndarray, centre, roll, jaw, blink, label: str) -> None:
    cv2.rectangle(frame_bgr, (0, 0), (WIDTH, HEIGHT), (60, 45, 35), -1)
    cv2.ellipse(frame_bgr, (int(centre[0]), int(centre[1])), (120, 165), math.degrees(roll), 0, 360, (150, 190, 235), -1)
    eye_y = int(centre[1] - 45)
    for ex in (-45, 45):
        open_h = max(2, int(10 * (1 - 0.9 * blink)))
        cv2.ellipse(frame_bgr, (int(centre[0] + ex), eye_y), (18, open_h), 0, 0, 360, (40, 40, 40), -1)
    mouth_h = max(3, int(35 * jaw))
    cv2.ellipse(frame_bgr, (int(centre[0]), int(centre[1] + 75)), (45, mouth_h), 0, 0, 360, (50, 40, 120), -1)
    cv2.putText(frame_bgr, label, (16, HEIGHT - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)


def _blendshapes(jaw: float, blink: float, index: int) -> dict:
    wobble = 0.5 + 0.5 * math.sin(index / 9.0)
    return {
        "jawOpen": float(jaw),
        "mouthSmileLeft": float(0.15 + 0.1 * wobble),
        "mouthSmileRight": float(0.15 + 0.1 * (1 - wobble)),
        "mouthPucker": float(0.05 + 0.03 * wobble),
        "mouthFunnel": float(0.02 + 0.02 * (1 - wobble)),
        "eyeBlinkLeft": float(blink),
        "eyeBlinkRight": float(blink),
        "browInnerUp": float(0.1 + 0.08 * wobble),
        "browDownLeft": float(0.04 + 0.02 * (1 - wobble)),
        "browDownRight": float(0.04 + 0.02 * wobble),
    }


def _matrix(R: np.ndarray) -> list[list[float]]:
    m = np.eye(4)
    m[:3, :3] = R
    return m.round(6).tolist()


def generate_session(home: Path, session_id: str, spec: dict) -> dict:
    video_path = home / "recordings" / "raw" / DEMO_PERSON / f"{session_id}.mp4"
    dataset_dir = home / "datasets" / DEMO_PERSON / session_id
    video_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    frames, rows = [], []
    for i in range(N_FRAMES):
        t = i / FPS
        centre, roll, yaw, jaw, blink = _geometry(t, spec, i, N_FRAMES)
        image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        _draw(image, centre, roll, jaw, blink, f"SYNTHETIC DEMO  {session_id}  frame {i}")
        writer.write(image)

        detected = i not in spec["no_face"]
        timestamp_ms = round(i * 1000 / FPS)
        frame = {"frame": i, "timestamp_ms": timestamp_ms, "face_detected": detected}
        blend = {}
        if detected:
            pts, R = _landmarks(centre, roll, yaw, jaw, blink)
            frame["landmarks"] = [
                {"id": k, "x": round(float(x), 6), "y": round(float(y), 6), "z": round(float(z), 6)}
                for k, (x, y, z) in enumerate(pts)
            ]
            blend = _blendshapes(jaw, blink, i)
            frame["blendshapes"] = {k: round(v, 6) for k, v in blend.items()}
            frame["transformation_matrix"] = _matrix(R)
        frames.append(frame)
        rows.append({"frame": i, "timestamp_ms": timestamp_ms, "face_detected": detected, **blend})
    writer.release()

    import json

    (dataset_dir / "facial_data.json").write_text(json.dumps(frames), encoding="utf-8")
    pd.DataFrame(rows).to_csv(dataset_dir / "facial_blendshapes.csv", index=False)
    return {"session_id": session_id, "recording": video_path.name, "frames": N_FRAMES, "fps": FPS}


def generate_demo(home: Path, force: bool = False) -> list[str]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from file_safety import OutputExistsError, check_writable

    targets = [home / "datasets" / DEMO_PERSON, home / "recordings" / "raw" / DEMO_PERSON]
    try:
        check_writable(targets, force=force)
    except OutputExistsError as exc:
        raise SystemExit(str(exc))

    created = []
    for session_id, spec in SESSIONS.items():
        generate_session(home, session_id, spec)
        created.append(session_id)
    return created


def build_motion_outputs(home: Path) -> None:
    """Run the same motion analysis as analyze_faces.py, pointed at the demo home folder."""
    import os

    os.environ["FACIAL_DATA_HOME"] = str(home)
    import facial_analyzer as fa
    import project_paths  # noqa: F401  (reads FACIAL_DATA_HOME at import time)

    for session in fa.processed_sessions(DEMO_PERSON):
        result = fa.analyse_session(session)
        fa.save_session_outputs(result, session, force=True)
    fa.write_person_metadata(DEMO_PERSON)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic demo recordings and datasets.")
    parser.add_argument("home", help="folder that will receive recordings/ and datasets/ (not the real project)")
    parser.add_argument("--force", action="store_true", help="replace existing demo data")
    args = parser.parse_args()
    home = Path(args.home).expanduser().resolve()
    if home == Path(__file__).resolve().parents[2]:
        raise SystemExit("Refusing to write demo data into the project folder itself. Choose another folder.")
    done = generate_demo(home, force=args.force)
    build_motion_outputs(home)
    print("Demo sessions written to", home, ":", ", ".join(done))
