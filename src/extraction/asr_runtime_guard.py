"""Model integrity verification and GPU job lock for ASR runtimes (phase H9.3b-3).

Nothing here loads a model, imports an ASR runtime or touches CUDA.

Model manifest (model_manifest.json, one per model directory)
- Declares the source repository, the pinned revision, the licence status, and every file with its
  size and SHA-256.
- Verification fails closed: a missing or unreadable manifest, a schema error, a licence status other
  than "cleared", a missing, modified, wrong-size or unexpected file, or a path outside the model
  directory all raise ModelIntegrityError before anything is loaded.
- Licence status "cleared" is required for normal transcription use. The H9.3b-0 investigation
  established that the converted large-v3 weights are NOT cleared, so the real manifest must not be
  created with a cleared status until counsel has confirmed it.

GPU job lock
- One exclusive lock file for all ASR GPU inference on this machine, shared across sessions.
- Created with O_EXCL. Records the owning process ID, host and a token. Released only by the holder,
  and only when the token still matches.
- A stale lock (this host, owner process gone) is never removed automatically. An explicit recovery
  function removes it only after re-verifying staleness at that moment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction.audio_acquisition import _pid_alive  # noqa: E402

MANIFEST_NAME = "model_manifest.json"
MANIFEST_VERSION = 1
LICENCE_CLEARED = "cleared"
LICENCE_STATUSES = ("cleared", "not cleared", "under review")
ASR_MODELS_ROOT = Path(r"C:\AI\asr-models")
ASR_RUNTIME_ROOT = Path(r"C:\AI\asr-runtime")
GPU_LOCK_NAME = "gpu.lock"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FILE_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
HASH_CHUNK = 1 << 20


class ModelIntegrityError(ValueError):
    """A model directory or manifest fails verification. Nothing may be loaded."""


class GpuLockError(RuntimeError):
    """The GPU lock is held, or it cannot be acquired or recovered safely."""


class ConfigPathError(ValueError):
    """A path is outside the permitted storage root, or is not a usable directory."""


# ---------------------------------------------------------------- paths


def resolve_inside(path, root) -> Path:
    """Resolve path and require it to lie inside root. Symbolic links are refused."""
    root = Path(root).resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise ConfigPathError("the model path is a symbolic link")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ConfigPathError("the model path is outside the permitted storage root")
    if not resolved.is_dir():
        raise ConfigPathError("the model directory does not exist")
    return resolved


# ---------------------------------------------------------------- manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(model_dir: Path) -> tuple[dict, str]:
    """Read and schema-check the manifest. Returns (manifest, SHA-256 of the manifest file)."""
    path = Path(model_dir) / MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise ModelIntegrityError("the model manifest is missing")
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelIntegrityError("the model manifest cannot be read") from exc
    if not isinstance(data, dict):
        raise ModelIntegrityError("the model manifest is not an object")
    if data.get("manifest_version") != MANIFEST_VERSION:
        raise ModelIntegrityError("unsupported model manifest version")
    for key in ("model_name", "source_repository", "revision", "licence_status", "files"):
        if key not in data:
            raise ModelIntegrityError(f"the model manifest has no {key!r}")
    if not isinstance(data["model_name"], str) or not data["model_name"].strip():
        raise ModelIntegrityError("model_name must be a non-empty string")
    if not isinstance(data["source_repository"], str) or not data["source_repository"].strip():
        raise ModelIntegrityError("source_repository must be a non-empty string")
    if not isinstance(data["revision"], str) or not REVISION_PATTERN.fullmatch(data["revision"]):
        raise ModelIntegrityError("revision must be a 40-character lowercase commit hash")
    if data["licence_status"] not in LICENCE_STATUSES:
        raise ModelIntegrityError("licence_status is not a recognised value")
    files = data["files"]
    if not isinstance(files, list) or not files:
        raise ModelIntegrityError("files must be a non-empty list")
    names = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ModelIntegrityError("each file entry must be an object")
        name = entry.get("name")
        size = entry.get("size")
        sha = entry.get("sha256")
        if not isinstance(name, str) or not FILE_NAME_PATTERN.fullmatch(name) or name in (".", ".."):
            raise ModelIntegrityError("a file name is not a plain file name")
        if name == MANIFEST_NAME or name in names:
            raise ModelIntegrityError("a file name is duplicated or reserved")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ModelIntegrityError("a file size is not a non-negative integer")
        if not isinstance(sha, str) or not SHA256_PATTERN.fullmatch(sha):
            raise ModelIntegrityError("a file SHA-256 is not 64 lowercase hex characters")
        names.add(name)
    return data, hashlib.sha256(raw).hexdigest()


def verify_model_dir(model_dir, *, root=ASR_MODELS_ROOT, require_cleared: bool = True) -> dict:
    """Verify a model directory against its manifest before any load. Fails closed.

    Returns a summary with the manifest hash, so the run metadata can record exactly what was loaded.
    The file set is exact: every file in the directory must be declared, except the manifest itself.
    """
    resolved = resolve_inside(model_dir, root)
    manifest, manifest_sha = load_manifest(resolved)
    if require_cleared and manifest["licence_status"] != LICENCE_CLEARED:
        raise ModelIntegrityError(
            f"licence status is {manifest['licence_status']!r}; normal transcription requires 'cleared'")
    declared = {entry["name"]: entry for entry in manifest["files"]}
    for name, entry in declared.items():
        path = resolved / name
        if path.is_symlink() or not path.is_file():
            raise ModelIntegrityError(f"a declared file is missing or is not a regular file: {name}")
        if path.stat().st_size != entry["size"]:
            raise ModelIntegrityError(f"a file has the wrong size: {name}")
        if _sha256(path) != entry["sha256"]:
            raise ModelIntegrityError(f"a file fails its SHA-256 check: {name}")
    present = {p.name for p in resolved.iterdir()} - {MANIFEST_NAME}
    unexpected = present - set(declared)
    if unexpected:
        raise ModelIntegrityError("the model directory contains undeclared files")
    return {
        "model_name": manifest["model_name"],
        "source_repository": manifest["source_repository"],
        "revision": manifest["revision"],
        "licence_status": manifest["licence_status"],
        "manifest_sha256": manifest_sha,
        "file_count": len(declared),
        "path": str(resolved),
    }


# ---------------------------------------------------------------- GPU lock


def _record(token: str) -> str:
    return json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "token": token,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def _read_record(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _is_stale(path: Path) -> bool:
    """Stale only when this host wrote it and its owning process no longer exists."""
    record = _read_record(path)
    if record is None or record.get("host") != socket.gethostname() or not isinstance(record.get("pid"), int):
        return False
    return not _pid_alive(record["pid"])


class GpuLock:
    """Exclusive lock for ASR GPU inference. The path is injectable for tests."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else ASR_RUNTIME_ROOT / GPU_LOCK_NAME
        self._token: str | None = None

    def acquire(self) -> None:
        if self._token is not None:
            raise GpuLockError("this GPU lock object is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Never removed here. A stale lock is reported, and only recover_stale_gpu_lock may clear it.
            if _is_stale(self.path):
                raise GpuLockError(
                    "a GPU inference lock was left by a process that has stopped. Confirm no inference is "
                    "running, then clear it with recover_stale_gpu_lock.") from None
            raise GpuLockError("GPU inference is already running on this machine") from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_record(token))
        self._token = token

    def release(self) -> None:
        if self._token is None:
            return
        token, self._token = self._token, None
        record = _read_record(self.path)
        if record is not None and record.get("token") == token:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass

    @contextmanager
    def held(self):
        self.acquire()
        try:
            yield self
        finally:
            self.release()


def recover_stale_gpu_lock(path=None) -> dict:
    """Explicit operator recovery. Removes the lock only if it is stale at this moment.

    Refuses a live owner, a lock from another host, and an unreadable lock. Never called automatically.
    """
    lock_path = Path(path) if path is not None else ASR_RUNTIME_ROOT / GPU_LOCK_NAME
    if not lock_path.exists():
        raise GpuLockError("there is no GPU lock to recover")
    record = _read_record(lock_path)
    if record is None:
        raise GpuLockError("the GPU lock cannot be read, so it will not be cleared")
    if not _is_stale(lock_path):
        raise GpuLockError("the GPU lock's owner is still running, or it belongs to another machine")
    os.remove(lock_path)
    return record
