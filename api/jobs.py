"""Job system for V2 builds (phase P4).

A job runs the existing command-line V2 build (src/extraction/build_v2_session.py) in a separate Python
process, so a MediaPipe crash cannot take down the API. The job record is kept in a small SQLite database
under logs/. The dataset files are written only by the existing writer, never by this module.

Lifecycle: queued -> running -> succeeded | failed | cancelled, and interrupted when the API restarts while a
job is active. Progress is reported per stage (validate, model, extract, assemble, publish, verify). It is not
per frame.

Concurrency: a unique index allows one active job per (person, session, kind). A session with the writer lock
(.v2-lock) present is refused. Mutating routes require the API token (see api/main.py).
"""

from __future__ import annotations

import hmac
import os
import secrets
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS = ROOT / "src" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import project_paths as pp  # noqa: E402
from schema_v2 import session_layout  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402

KINDS = ("v2_build",)
ACTIVE = ("queued", "running")
TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")
LOCK_FILE = ".v2-lock"
LEFTOVER_PREFIXES = (".v2-staging-", ".v2-previous-")

# Stage markers in the CLI's printed output, in order. Progress is the stage's position in this list.
STAGE_MARKERS = (
    ("Video:", "validate"),
    ("Pose Landmarker FULL loaded", "model"),
    ("Extracting pose", "extract"),
    ("Building the V2", "assemble"),
    ("Publishing", "publish"),
    ("V2 session published", "verify"),
)
STAGES = tuple(stage for _marker, stage in STAGE_MARKERS)

EXIT_MESSAGES = {
    0: None,
    1: "the build failed; see the job log",
    2: "refused by the build checks; see the job log",
    3: "the Pose model failed its SHA-256 check",
    4: "version 1 files changed during the build and were rolled back",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    person TEXT NOT NULL,
    session TEXT NOT NULL,
    regenerate INTEGER NOT NULL,
    state TEXT NOT NULL,
    stage TEXT,
    progress REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    exit_code INTEGER,
    message TEXT,
    pid INTEGER,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active
    ON jobs(person, session, kind) WHERE state IN ('queued', 'running');
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobStore:
    """SQLite job records. Each call opens its own connection, so threads do not share one."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, person: str, session: str, kind: str, regenerate: bool) -> dict:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, person, session, regenerate, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, kind, person, session, int(regenerate), now, now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row(row) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [_row(r) for r in rows]

    def update(self, job_id: str, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments}, updated_at = ? WHERE id = ?",
                         (*fields.values(), _now(), job_id))

    def active_for_session(self, person: str, session: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE person = ? AND session = ? AND state IN ('queued', 'running')",
                (person, session),
            ).fetchall()
        return [_row(r) for r in rows]


def _row(row) -> dict:
    d = dict(row)
    d["regenerate"] = bool(d["regenerate"])
    d["cancel_requested"] = bool(d["cancel_requested"])
    return d


def _stage_for(line: str) -> str | None:
    for marker, stage in STAGE_MARKERS:
        if marker in line:
            return stage
    return None


def leftovers(dataset_dir: Path) -> list[str]:
    """Names of staging, previous and lock items in a session folder. They are reported, never removed here."""
    if not dataset_dir.is_dir():
        return []
    names = []
    for p in sorted(dataset_dir.iterdir()):
        if p.name == LOCK_FILE or p.name.startswith(LEFTOVER_PREFIXES):
            names.append(p.name)
    return names


def default_command(job: dict) -> list[str]:
    cmd = [sys.executable, str(ROOT / "src" / "extraction" / "build_v2_session.py"), job["person"], job["session"]]
    if job["regenerate"]:
        cmd.append("--regenerate-v2")
    return cmd


class JobRunner:
    """Starts worker processes and watches them on a background thread. Never blocks the caller."""

    def __init__(self, store: JobStore, log_dir: Path, command_for: Callable[[dict], list[str]] = default_command):
        self.store = store
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.command_for = command_for
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def log_path(self, job_id: str) -> Path:
        return self.log_dir / f"{job_id}.log"

    def start(self, job: dict) -> None:
        cmd = self.command_for(job)
        with open(self.log_path(job["id"]), "a", encoding="utf-8") as log:
            log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1, cwd=str(ROOT),
        )
        with self._lock:
            self._procs[job["id"]] = proc
        self.store.update(job["id"], state="running", pid=proc.pid, stage="queued", progress=0.0)
        threading.Thread(target=self._watch, args=(job, proc), daemon=True).start()

    def cancel(self, job_id: str) -> bool:
        """Request cancel. Returns False if no worker is running for the job."""
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is None or proc.poll() is not None:
            return False
        self.store.update(job_id, cancel_requested=1)
        proc.kill()
        return True

    def _watch(self, job: dict, proc: subprocess.Popen) -> None:
        job_id = job["id"]
        with open(self.log_path(job_id), "a", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                stage = _stage_for(line)
                if stage:
                    self.store.update(job_id, stage=stage, progress=(STAGES.index(stage) + 1) / len(STAGES))
            code = proc.wait()
        proc.stdout.close()
        with self._lock:
            self._procs.pop(job_id, None)
        current = self.store.get(job_id)
        dataset = pp.session_dataset_dir(job["person"], job["session"])
        left = leftovers(dataset)
        if current and current["cancel_requested"]:
            state, message = "cancelled", "cancelled by user"
        elif code == 0:
            state, message = "succeeded", None
        else:
            state, message = "failed", EXIT_MESSAGES.get(code, f"exited with code {code}")
        if state != "succeeded" and left:
            message = f"{message}; leftover items in the session folder (not removed): {', '.join(left)}"
        self.store.update(job_id, state=state, exit_code=code, message=message, pid=None,
                          progress=1.0 if state == "succeeded" else current["progress"] if current else 0.0)


# ---- service object used by the API -------------------------------------------------------------

class Service:
    def __init__(self, store: JobStore, runner: JobRunner, token: str):
        self.store = store
        self.runner = runner
        self.token = token


_service: Service | None = None


def configure(store_path: Path, log_dir: Path, token: str | None = None,
              command_for: Callable[[dict], list[str]] = default_command) -> Service:
    global _service
    store = JobStore(store_path)
    runner = JobRunner(store, log_dir, command_for=command_for)
    _service = Service(store, runner, token or os.environ.get("FACIAL_API_TOKEN") or secrets.token_urlsafe(24))
    return _service


def service() -> Service:
    global _service
    if _service is None:
        configure(ROOT / "logs" / "jobs.sqlite3", ROOT / "logs" / "jobs")
    return _service


def token_ok(presented: str | None) -> bool:
    return presented is not None and hmac.compare_digest(presented, service().token)


def check_v2_build(person: str, session: str, regenerate: bool) -> Path:
    """The same refusals as the command line, checked before anything is queued. Returns the session folder."""
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    video = pp.session_video(person, session)
    dataset = pp.session_dataset_dir(person, session)
    if not video.is_file():
        raise ApiError("no_video", "the recording for this session is missing", 422)
    if not (dataset / "facial_data.json").is_file():
        raise ApiError("no_v1_data", "run the version 1 extraction for this session first", 422)
    if session_layout(dataset) == "v2" and not regenerate:
        raise ApiError("v2_exists", "a complete V2 dataset exists; set regenerate to replace it", 409)
    if (dataset / LOCK_FILE).exists():
        raise ApiError("writer_lock_present", "the writer lock is present; another run may be active or interrupted", 409)
    return dataset


def create_job(person: str, session: str, kind: str, regenerate: bool) -> dict:
    if kind not in KINDS:
        raise ApiError("invalid_request", f"unknown job kind: {kind!r}", 400)
    check_v2_build(person, session, regenerate)
    svc = service()
    try:
        job = svc.store.insert(person, session, kind, regenerate)
    except sqlite3.IntegrityError:
        raise ApiError("job_active", "a job is already running for this session", 409) from None
    svc.runner.start(job)
    return svc.store.get(job["id"])


def cancel_job(job_id: str) -> dict:
    svc = service()
    job = svc.store.get(job_id)
    if job is None:
        raise ApiError("not_found", "job not found", 404)
    if job["state"] not in ACTIVE:
        raise ApiError("not_active", f"job is {job['state']}, not active", 409)
    if not svc.runner.cancel(job_id):
        raise ApiError("not_active", "no worker is running for this job", 409)
    return svc.store.get(job_id)


def recover_interrupted() -> list[dict]:
    """On API start, jobs still marked active have no worker in this process. Mark them interrupted and report
    what was left behind. Nothing is removed, and no process is killed."""
    svc = service()
    marked = []
    for job in svc.store.list(limit=10_000):
        if job["state"] not in ACTIVE:
            continue
        dataset = pp.session_dataset_dir(job["person"], job["session"])
        left = leftovers(dataset)
        pid_note = f" The worker process {job['pid']} may still be running." if job["pid"] else ""
        note = "the API restarted while this job was active." + pid_note
        if left:
            note += f" Leftover items (not removed): {', '.join(left)}."
        svc.store.update(job["id"], state="interrupted", pid=None, message=note)
        marked.append(svc.store.get(job["id"]))
    return marked
