"""Read-only catalogue for the API (phase P0). Framework-free, so it can be tested directly.

Every name comes from project_paths. Nothing here writes to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS = _ROOT / "src" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import project_paths as pp  # noqa: E402
import schema_v2 as s2  # noqa: E402

API_VERSION = "0.1.0"


class ApiError(Exception):
    """An error with an HTTP status and a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# Feature names shown to users. The layout codes above are kept for compatibility.
LAYOUT_NAMES = {"v1": "face data", "v2": "pose data", "incomplete_v2": "pose data incomplete", "none": "no data"}

def _check_person(person: str) -> None:
    if not pp.is_valid_person(person):
        raise ApiError("invalid_name", "invalid person or project name", 400)


def health() -> dict:
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "schema_version": s2.SCHEMA_VERSION,
        "data_root": pp.DATA_HOME.name,  # the folder name only, never the absolute path
        "read_only": True,
    }


def list_persons() -> list[str]:
    return pp.list_persons()


def list_sessions(person: str) -> list[dict]:
    _check_person(person)
    if not (pp.person_recordings_dir(person).is_dir() or pp.person_dataset_dir(person).is_dir()):
        raise ApiError("not_found", "no such person or project", 404)
    rows = []
    for session_id in pp.list_sessions(person):
        rows.append({
            "session_id": session_id,
            "has_video": pp.session_video(person, session_id).is_file(),
            "layout": (layout := s2.session_layout(pp.session_dataset_dir(person, session_id))),
            "layout_name": LAYOUT_NAMES.get(layout, layout),
        })
    return rows
