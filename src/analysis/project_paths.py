"""Single source of truth for the project's folder layout.

    recordings/raw/<person>/session-NNN.mp4          original videos (never modified)
    datasets/<person>/session-NNN/                   everything derived from one session
        facial_data.json, facial_blendshapes.csv, frames/,
        motion_metrics.csv, blink_events.csv, tracking_quality.json, calibration.json

<person> is a person or project folder name chosen in the dashboard or created by hand.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_RAW = PROJECT_ROOT / "recordings" / "raw"
DATASETS_ROOT = PROJECT_ROOT / "datasets"
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

PERSON_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,59}")
SESSION_PATTERN = re.compile(r"session-(\d{3})")


def is_valid_person(name: str) -> bool:
    return bool(PERSON_PATTERN.fullmatch(name or ""))


def is_valid_session(name: str) -> bool:
    return bool(SESSION_PATTERN.fullmatch(name or ""))


def suggest_person(text: str) -> str:
    """'My Project 2' -> 'my-project-2'."""
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")[:60]


def list_persons() -> list[str]:
    if not RECORDINGS_RAW.is_dir():
        return []
    return sorted(d.name for d in RECORDINGS_RAW.iterdir() if d.is_dir() and is_valid_person(d.name))


def create_person(name: str) -> Path:
    if not is_valid_person(name):
        raise ValueError("Use lowercase letters, numbers, hyphens or underscores.")
    folder = RECORDINGS_RAW / name
    folder.mkdir(parents=True, exist_ok=True)
    (DATASETS_ROOT / name).mkdir(parents=True, exist_ok=True)
    return folder


def person_recordings_dir(person: str) -> Path:
    return RECORDINGS_RAW / person


def person_dataset_dir(person: str) -> Path:
    return DATASETS_ROOT / person


def list_sessions(person: str) -> list[str]:
    """Session IDs that have a video in recordings/raw/<person>/."""
    folder = person_recordings_dir(person)
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("session-*.mp4") if is_valid_session(p.stem))


def next_session_id(person: str) -> str:
    used = [int(SESSION_PATTERN.fullmatch(s).group(1)) for s in list_sessions(person)]
    return f"session-{(max(used) + 1 if used else 1):03d}"


def session_video(person: str, session: str) -> Path:
    return person_recordings_dir(person) / f"{session}.mp4"


def session_dataset_dir(person: str, session: str) -> Path:
    return person_dataset_dir(person) / session


def session_has_dataset(person: str, session: str) -> bool:
    return (session_dataset_dir(person, session) / "facial_data.json").is_file()


def list_processed_sessions(person: str) -> list[str]:
    return [s for s in list_sessions(person) if session_has_dataset(person, s)]
