"""Job system for long-running work on recordings (phases P4 and P4b).

Each job runs an existing command-line script in a separate Python process, so a MediaPipe crash cannot take
down the API. Job records are kept in a small SQLite database under logs/. The dataset files are written only
by those scripts, never by this module.

Kinds:
  pose_build     src/extraction/build_v2_session.py  (pose data; --regenerate-v2 replaces it)
  face_extract   src/analysis/extract_faces.py        (face data; refuses existing data, no regenerate)
  face_analyse   src/analysis/analyze_faces.py        (blendshape and motion outputs; --force regenerates)
Older codes (v2_build, v1_extract, v1_analyse) are still accepted and mapped to these.

Lifecycle: queued -> running -> succeeded | failed | cancelled, and interrupted when the API restarts while a
job is active. Progress is per frame for face_extract (from the script's "Processed" lines) and per stage for the
other kinds.

Concurrency: one active job per session across all kinds, enforced by a unique index. A session with the
writer lock (.pose-lock, or the older .v2-lock) present is refused. Mutating routes require the API token (see api/main.py).
"""

from __future__ import annotations

import hmac
import os
import re
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
for _p in (ROOT / "src" / "analysis", ROOT / "src" / "extraction"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import project_paths as pp  # noqa: E402
from schema_v2 import session_layout  # noqa: E402
from session_writer_v2 import LOCK_FILE, LOCK_FILES, PREVIOUS_PREFIXES, STAGING_PREFIXES  # noqa: E402
from api.catalog import ApiError, _check_person  # noqa: E402
from facial_analyzer import make_session, motion_output_paths  # noqa: E402
from model_registry import ModelIntegrityError, get_spec, verify_model  # noqa: E402

KINDS = ("pose_build", "face_extract", "face_analyse")
# Older job codes, still accepted in requests and mapped when rows are read.
KIND_ALIASES = {"v2_build": "pose_build", "v1_extract": "face_extract", "v1_analyse": "face_analyse"}
SCRIPTS = {
    "pose_build": ROOT / "src" / "extraction" / "build_v2_session.py",
    "face_extract": ROOT / "src" / "analysis" / "extract_faces.py",
    "face_analyse": ROOT / "src" / "analysis" / "analyze_faces.py",
}
ACTIVE = ("queued", "running")
LEFTOVER_PREFIXES = STAGING_PREFIXES + PREVIOUS_PREFIXES

V2_STAGES = ("validate", "model", "extract", "assemble", "publish", "verify")
V2_MARKERS = (
    ("Video:", "validate"),
    ("Pose Landmarker FULL loaded", "model"),
    ("Extracting pose", "extract"),
    ("Building the pose data", "assemble"),
    ("Publishing the pose data", "publish"),
    ("Pose data published", "verify"),
)
V1_ANALYSE_MARKERS = (
    ("CSV created:", "assemble", 0.25),
    ("Motion outputs:", "publish", 0.6),
    ("Metadata:", "verify", 0.9),
)

EXIT_MESSAGES = {
    1: "failed; see the job log",
    2: "refused or invalid input; see the job log",
    3: "the Pose model failed its SHA-256 check",
    4: "face data files changed during the build and were rolled back",
}

_TOTAL_RE = re.compile(r"^Total frames:\s*(\d+)")
_PROCESSED_RE = re.compile(r"^Processed:\s*(\d+)")

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
DROP INDEX IF EXISTS jobs_one_active;
CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_per_session
    ON jobs(person, session) WHERE state IN ('queued', 'running');
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


def canonical_kind(kind: str) -> str:
    """The current name for a job kind. Older names map to the current ones."""
    return KIND_ALIASES.get(kind, kind)


def _row(row) -> dict:
    d = dict(row)
    d["kind"] = canonical_kind(d["kind"])
    d["regenerate"] = bool(d["regenerate"])
    d["cancel_requested"] = bool(d["cancel_requested"])
    return d


def parse_progress(kind: str, line: str, state: dict) -> tuple[str, float] | None:
    """Stage and progress (0 to 1) for one line of a worker's output, or None if the line says nothing new.

    state holds what earlier lines taught us (the total frame count for extraction).
    """
    kind = canonical_kind(kind)
    if kind == "pose_build":
        for marker, stage in V2_MARKERS:
            if marker in line:
                return stage, (V2_STAGES.index(stage) + 1) / len(V2_STAGES)
        return None
    if kind == "face_extract":
        m = _TOTAL_RE.match(line)
        if m:
            state["total"] = int(m.group(1))
            return "extract", 0.0
        m = _PROCESSED_RE.match(line)
        if m:
            total = state.get("total")
            fraction = min(int(m.group(1)) / total, 0.99) if total else 0.0
            return "extract", fraction
        if line.startswith("Processing completed"):
            return "publish", 0.99
        return None
    if kind == "face_analyse":
        for marker, stage, fraction in V1_ANALYSE_MARKERS:
            if marker in line:
                return stage, fraction
        return None
    return None


def leftovers(dataset_dir: Path, kind: str | None = None) -> list[str]:
    """Names of staging, previous and lock items, and a partly written frames folder from an interrupted
    extraction. They are reported, never removed here."""
    if not dataset_dir.is_dir():
        return []
    names = []
    for p in sorted(dataset_dir.iterdir()):
        if p.name in LOCK_FILES or p.name.startswith(LEFTOVER_PREFIXES):
            names.append(p.name)
    frames = dataset_dir / "frames"
    if canonical_kind(kind or "") == "face_extract" and not (dataset_dir / "facial_data.json").exists() \
            and frames.is_dir() and any(frames.iterdir()):
        names.append("frames/ (partly written)")
    return names


def default_command(job: dict) -> list[str]:
    kind = canonical_kind(job["kind"])
    cmd = [sys.executable, str(SCRIPTS[kind]), job["person"], job["session"]]
    if job["regenerate"]:
        if kind == "pose_build":
            cmd.append("--regenerate-v2")
        elif kind == "face_analyse":
            cmd.append("--force")
    return cmd


def worker_env() -> dict:
    """Output must not be buffered, or progress lines would arrive only when the job ends."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


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
            errors="replace", bufsize=1, cwd=str(ROOT), env=worker_env(),
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
        seen: dict = {}
        with open(self.log_path(job_id), "a", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                update = parse_progress(job["kind"], line.rstrip("\n"), seen)
                if update:
                    stage, progress = update
                    self.store.update(job_id, stage=stage, progress=progress)
            code = proc.wait()
        proc.stdout.close()
        with self._lock:
            self._procs.pop(job_id, None)
        current = self.store.get(job_id)
        dataset = pp.session_dataset_dir(job["person"], job["session"])
        left = leftovers(dataset, job["kind"])
        if current and current["cancel_requested"]:
            state, message = "cancelled", "cancelled by user"
        elif code == 0:
            state, message = "succeeded", None
        else:
            state, message = "failed", EXIT_MESSAGES.get(code, f"exited with code {code}")
        if state != "succeeded" and left:
            message = f"{message}; leftover items in the session folder (not removed): {', '.join(left)}"
        progress = 1.0 if state == "succeeded" else (current["progress"] if current else 0.0)
        self.store.update(job_id, state=state, exit_code=code, message=message, pid=None, progress=progress)


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


def check_job(kind: str, person: str, session: str, regenerate: bool) -> None:
    """The same refusals as the scripts, checked before anything is queued."""
    kind = canonical_kind(kind)
    if kind not in KINDS:
        raise ApiError("invalid_request", f"unknown job kind: {kind!r}", 400)
    _check_person(person)
    if not pp.is_valid_session(session):
        raise ApiError("invalid_name", "invalid session name", 400)
    video = pp.session_video(person, session)
    dataset = pp.session_dataset_dir(person, session)
    if not video.is_file():
        raise ApiError("no_video", "the recording for this session is missing", 422)
    if any((dataset / name).exists() for name in LOCK_FILES):
        raise ApiError("writer_lock_present", "the writer lock is present; another run may be active or interrupted", 409)

    if kind == "pose_build":
        if not (dataset / "facial_data.json").is_file():
            raise ApiError("no_face_data", "run the face extraction for this session first", 422)
        if session_layout(dataset) == "v2" and not regenerate:
            raise ApiError("v2_exists", "pose data already exists; set regenerate to replace it", 409)
    elif kind == "face_extract":
        frames = dataset / "frames"
        if (dataset / "facial_data.json").exists() or (frames.is_dir() and any(frames.iterdir())):
            raise ApiError("face_data_exists", "face data already exists; extraction never overwrites it", 409)
        try:
            verify_model(get_spec("face"))
        except Exception as exc:  # missing or changed model file: refuse, never load it
            raise ApiError("model_integrity", f"the face model failed its check: {exc}", 503) from None
    elif kind == "face_analyse":
        if not (dataset / "facial_data.json").is_file():
            raise ApiError("no_face_data", "run the face extraction for this session first", 422)
        outputs = [dataset / "facial_blendshapes.csv", *motion_output_paths(make_session(person, session))]
        existing = [p.name for p in outputs if p.exists()]
        if existing and not regenerate:
            raise ApiError("outputs_exist", f"analysis outputs exist ({', '.join(existing)}); set regenerate to replace them", 409)


def create_job(person: str, session: str, kind: str, regenerate: bool) -> dict:
    kind = canonical_kind(kind)
    check_job(kind, person, session, regenerate)
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
        left = leftovers(dataset, job["kind"])
        pid_note = f" The worker process {job['pid']} may still be running." if job["pid"] else ""
        note = "the API restarted while this job was active." + pid_note
        if left:
            note += f" Leftover items (not removed): {', '.join(left)}."
        svc.store.update(job["id"], state="interrupted", pid=None, message=note)
        marked.append(svc.store.get(job["id"]))
    return marked
