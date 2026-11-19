"""Read-only system views (phase P5): model registry and integrity, job logs, and settings.

Models are hashed, never loaded. Licence text is shown exactly as recorded in model_registry; nothing here
marks a licence as cleared. Logs are the job worker logs, with the project and home folders replaced by
placeholders. Settings show names, not full paths, and never the API token.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from api import catalog, jobs
from api.catalog import ApiError

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src" / "analysis", _ROOT / "src" / "extraction"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import project_paths as pp  # noqa: E402
from model_registry import MODELS, MODELS_DIR, ModelSpec, sha256_file  # noqa: E402
from schema_v2 import SCHEMA_VERSION  # noqa: E402

_hash_cache: dict = {}
MAX_TAIL_LINES = 2000


def _model_status(spec: ModelSpec, path: Path) -> dict:
    """Hash status for one model. The hash is cached on path, size and modification time."""
    row = {
        "key": spec.key,
        "filename": spec.filename,
        "expected_sha256": spec.sha256.upper(),
        "version": spec.version,
        "source_url": spec.source_url,
        "licence_status": spec.licence_status,
    }
    if not path.is_file():
        row.update({"present": False, "size_bytes": None, "sha256_status": "missing", "actual_sha256": None})
        return row
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_cache:
        _hash_cache.clear()
        _hash_cache[key] = sha256_file(path)
    actual = _hash_cache[key]
    row.update({
        "present": True,
        "size_bytes": stat.st_size,
        "actual_sha256": actual,
        "sha256_status": "match" if actual == spec.sha256.upper() else "mismatch",
    })
    return row


def models(specs: dict[str, ModelSpec] | None = None, models_dir: Path | None = None) -> list[dict]:
    folder = Path(models_dir) if models_dir is not None else MODELS_DIR
    rows = []
    for key, spec in (specs or MODELS).items():
        row = _model_status(spec, folder / spec.filename)
        row["location"] = f"models/mediapipe/{spec.filename}"
        rows.append(row)
    return rows


def redact(text: str) -> str:
    """Replace the project and home folders in log text, so logs show relative locations only."""
    for folder, placeholder in ((str(_ROOT), "<project>"), (str(Path.home()), "<home>")):
        for variant in {folder, folder.replace("\\", "/")}:
            text = text.replace(variant, placeholder)
    return text


def job_logs(limit: int = 50) -> list[dict]:
    rows = []
    for job in jobs.service().store.list(limit=min(max(limit, 1), 200)):
        log = jobs.service().runner.log_path(job["id"])
        rows.append({
            "job_id": job["id"],
            "kind": job["kind"],
            "person": job["person"],
            "session": job["session"],
            "state": job["state"],
            "created_at": job["created_at"],
            "log_bytes": log.stat().st_size if log.is_file() else None,
        })
    return rows


def job_log_tail(job_id: str, lines: int = 200) -> dict:
    job = jobs.service().store.get(job_id)
    if job is None:
        raise ApiError("not_found", "job not found", 404)
    lines = min(max(lines, 1), MAX_TAIL_LINES)
    log = jobs.service().runner.log_path(job_id)
    if not log.is_file():
        return {"job_id": job_id, "lines": [], "total_lines": 0}
    text = log.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "job_id": job_id,
        "total_lines": len(text),
        "lines": [redact(line) for line in text[-lines:]],
    }


def settings() -> dict:
    return {
        "api_version": catalog.API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "data_root_name": pp.DATA_HOME.name,
        "model_folder": "models/mediapipe",
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "api_token_configured": bool(jobs.service().token),
        "token_header": "X-Facial-Token",
        "writes": "only through job routes; each job runs an existing script",
        "bind_address": "127.0.0.1",
    }
