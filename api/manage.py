"""Managing people, recordings, datasets and job history (behind the API token).

Every action here needs the API token (in the route) and a typed confirmation: the exact name of what is being
removed, or the phrase below for job history. Deletions are permanent. A deletion is refused while a job is
active on the same session. Upload writes to a temporary file, checks the MP4 signature and the size limit,
and only then moves the file into place. Nothing is overwritten.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src" / "analysis") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src" / "analysis"))

import project_paths as pp  # noqa: E402
from api import jobs  # noqa: E402
from api.catalog import ApiError  # noqa: E402

MAX_UPLOAD_BYTES = 2 * 1024 ** 3  # 2 GiB
JOB_HISTORY_PHRASE = "clear job history"
PART_SUFFIX = ".part"


def _check_name_person(person: str) -> None:
    if not pp.is_valid_person(person):
        raise ApiError("invalid_name", "invalid person or project name", 400)


def _check_session(session: str) -> None:
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name (expected session-NNN)", 400)


def _confirm(body_confirm: str | None, expected: str) -> None:
    if body_confirm != expected:
        raise ApiError("confirmation_mismatch", f"type exactly '{expected}' to confirm", 400)


def _no_active_job(person: str, session: str) -> None:
    if jobs.service().store.active_for_session(person, session):
        raise ApiError("job_active", "a job is running for this session; wait for it to finish or cancel it", 409)


def create_person(name: str) -> dict:
    _check_name_person(name)
    if pp.person_recordings_dir(name).is_dir():
        raise ApiError("exists", "a person with this name already exists", 409)
    pp.create_person(name)
    return {"person": name, "created": True}


def delete_person(person: str, confirm: str | None) -> dict:
    _check_name_person(person)
    _confirm(confirm, person)
    if any(jobs.service().store.active_for_session(person, s) for s in pp.list_sessions(person)):
        raise ApiError("job_active", "a job is running for this person; wait for it to finish", 409)
    removed = pp.delete_person(person)
    return {"person": person, "deleted": bool(removed)}


def delete_dataset(person: str, session: str, confirm: str | None) -> dict:
    _check_name_person(person)
    _check_session(session)
    _confirm(confirm, f"{person}/{session}")
    _no_active_job(person, session)
    removed = pp.delete_session_dataset(person, session)
    return {"person": person, "session": session, "deleted": bool(removed)}


def delete_video(person: str, session: str, confirm: str | None) -> dict:
    _check_name_person(person)
    _check_session(session)
    _confirm(confirm, f"{person}/{session}")
    _no_active_job(person, session)
    removed = pp.delete_session_video(person, session)
    return {"person": person, "session": session, "deleted": bool(removed)}


def clear_job_history(confirm: str | None) -> dict:
    _confirm(confirm, JOB_HISTORY_PHRASE)
    svc = jobs.service()
    removed_ids = svc.store.delete_finished()
    for job_id in removed_ids:
        log = svc.runner.log_path(job_id)
        if log.is_file():
            log.unlink()
    return {"removed": len(removed_ids)}


class UploadWriter:
    """Receives a recording in pieces. Call write() for each piece, then finish(). Call abort() on any error."""

    def __init__(self, person: str, session: str):
        _check_name_person(person)
        _check_session(session)
        self.person = person
        self.session = session
        self.final = pp.session_video(person, session)
        if not pp.person_recordings_dir(person).is_dir():
            raise ApiError("no_person", "create the person first", 404)
        if self.final.exists():
            raise ApiError("exists", "a recording with this session name already exists", 409)
        self.part = self.final.with_name(f".upload-{uuid.uuid4().hex}{PART_SUFFIX}")
        self._handle = open(self.part, "wb")
        self._size = 0
        self._head = b""

    def write(self, chunk: bytes) -> None:
        self._size += len(chunk)
        if self._size > MAX_UPLOAD_BYTES:
            self.abort()
            raise ApiError("too_large", "the file is larger than the upload limit", 413)
        if len(self._head) < 12:
            self._head += chunk[: 12 - len(self._head)]
            if len(self._head) >= 12 and self._head[4:8] != b"ftyp":
                self.abort()
                raise ApiError("not_mp4", "the file is not an MP4 video", 415)
        self._handle.write(chunk)

    def finish(self) -> dict:
        if len(self._head) < 12 or self._head[4:8] != b"ftyp":
            self.abort()
            raise ApiError("not_mp4", "the file is not an MP4 video", 415)
        if self._size == 0:
            self.abort()
            raise ApiError("empty", "the upload is empty", 400)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        if self.final.exists():
            self.part.unlink(missing_ok=True)
            raise ApiError("exists", "a recording with this session name already exists", 409)
        os.replace(self.part, self.final)
        return {"person": self.person, "session": self.session, "bytes": self._size, "uploaded": True}

    def abort(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self.part.unlink(missing_ok=True)
