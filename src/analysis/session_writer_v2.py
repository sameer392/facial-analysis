"""Version 2 session writer: staging, validation, manifest, atomic publication, regeneration and rollback.

Publication order inside a session folder (datasets/<person>/<session>/):

    1. stage       every v2 file is written to .v2-staging-<id>/ first
    2. validate    staged files are read back and checked against schema_v2.py
    3. checksum    SHA-256 and size of every staged file are recorded in manifest.json
    4. verify      manifest.json is checked against the staged files
    5. publish     manifest.json, then each data file, are moved into the session folder
    6. complete    metadata.json is moved last. Its presence is the completion marker

Interruption at any point leaves either no metadata.json (incomplete, and readers ignore it) or a
complete session. recover_incomplete_v2() removes only files listed in the manifest of an unfinished
operation and restores any previous version 2 dataset. rollback_v2() removes a completed version 2
dataset the same way. Version 1 files and recordings are never written, moved or deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from schema_v2 import (
    ARRAY_SPECS,
    BLENDSHAPE_NAMES,
    HANDEDNESS_COLUMNS,
    MANIFEST_FILE,
    METADATA_FILE,
    OPTIONAL_ARRAY_SPECS,
    SCHEMA_VERSION,
    TIMELINE_FILE,
    session_layout,
    validate_arrays,
    validate_handedness_rows,
    validate_manifest,
    validate_metadata,
    validate_timeline_rows,
)

STAGING_PREFIX = ".v2-staging-"
PREVIOUS_PREFIX = ".v2-previous-"
MANIFEST_VERSION = 1

TIMELINE_COLUMNS = ("frame", "timestamp_ms", "face_detected", "pose_detected",
                    "left_hand_detected", "right_hand_detected")
BLENDSHAPE_COLUMNS = ("frame", "timestamp_ms", "face_detected") + tuple(BLENDSHAPE_NAMES)

# Names used by version 1 at the session root. Version 2 must never write these.
V1_PROTECTED_NAMES = frozenset({
    "facial_data.json", "facial_blendshapes.csv", "motion_metrics.csv",
    "blink_events.csv", "tracking_quality.json", "calibration.json", "frames",
})
V2_DIRECTORIES = ("facial", "body", "hands", "quality")
V2_ROOT_FILES = (TIMELINE_FILE, MANIFEST_FILE, METADATA_FILE)


class V2Error(Exception):
    """Base error for version 2 writing."""


class V2ExistsError(V2Error):
    """A complete version 2 dataset exists and regeneration was not requested."""


class V2ValidationError(V2Error):
    """Payload or staged output does not satisfy schema_v2.py."""


class SimulatedInterruption(Exception):
    """Raised by tests at a chosen stage to imitate a crash. Not used in normal operation."""


FAIL_POINTS = ("arrays", "csv", "validate", "manifest", "publish", "metadata")


@dataclass
class V2Payload:
    """Everything needed to write one version 2 session.

    arrays:          relative path -> numpy array (required and optional schema arrays)
    timeline_rows:   one dict per frame with TIMELINE_COLUMNS
    handedness_rows: one dict per frame with HANDEDNESS_COLUMNS
    blendshapes:     DataFrame with BLENDSHAPE_COLUMNS, one row per frame
    quality:         JSON-serialisable dict for quality/tracking_quality.json
    metadata:        metadata dict; the writer adds the outputs section and manifest checksum
    """
    arrays: dict
    timeline_rows: list
    handedness_rows: list
    blendshapes: pd.DataFrame
    quality: dict
    metadata: dict
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def _rel_to_path(rel: str, root: Path) -> Path:
    """Convert a POSIX-style manifest path to a platform path under root. Refuses escapes."""
    parts = rel.split("/")
    if not rel or rel.startswith("/") or any(p in ("", "..") for p in parts):
        raise V2ValidationError(f"unsafe relative path: {rel!r}")
    return root.joinpath(*parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_fail(point: str, fail_at: str | None) -> None:
    if fail_at == point:
        raise SimulatedInterruption(point)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _expected_arrays(payload: V2Payload) -> list[str]:
    names = [spec.relative_path for spec in ARRAY_SPECS]
    names += [spec.relative_path for spec in OPTIONAL_ARRAY_SPECS if spec.relative_path in payload.arrays]
    return names


def _planned_paths(payload: V2Payload) -> list[str]:
    """Every data file the operation will publish, excluding the two structural files."""
    return _expected_arrays(payload) + [
        "facial/blendshapes.csv", TIMELINE_FILE, "hands/handedness.csv",
        "quality/tracking_quality.json",
    ]


# ---------------------------------------------------------------- validation before writing

def prevalidate(payload: V2Payload) -> int:
    """Check the payload against schema_v2.py before anything is written. Returns the frame count."""
    n = len(payload.timeline_rows)
    issues = []
    for rel in payload.arrays:
        if rel.split("/")[-1] in V1_PROTECTED_NAMES or rel in V1_PROTECTED_NAMES:
            issues.append(f"{rel} collides with a version 1 name")
    for spec in ARRAY_SPECS:
        if spec.relative_path not in payload.arrays:
            issues.append(f"missing required array {spec.relative_path}")
    issues += validate_arrays(payload.arrays, n)
    issues += validate_timeline_rows(payload.timeline_rows, n)
    for row in payload.timeline_rows:
        missing = [c for c in TIMELINE_COLUMNS if c not in row]
        if missing:
            issues.append(f"timeline row missing columns {missing}")
            break
    if len(payload.handedness_rows) != n:
        issues.append(f"handedness has {len(payload.handedness_rows)} rows, expected {n}")
    else:
        issues += validate_handedness_rows(payload.handedness_rows)
    if list(payload.blendshapes.columns) != list(BLENDSHAPE_COLUMNS):
        issues.append("blendshapes columns do not match the schema")
    if len(payload.blendshapes) != n:
        issues.append(f"blendshapes has {len(payload.blendshapes)} rows, expected {n}")
    if not isinstance(payload.quality, dict):
        issues.append("quality must be a dict")
    issues += validate_metadata(payload.metadata)
    if issues:
        raise V2ValidationError("; ".join(issues))
    return n


# ---------------------------------------------------------------- staging

def _write_staging(staging: Path, payload: V2Payload, fail_at: str | None) -> list[str]:
    written = []
    for i, rel in enumerate(_expected_arrays(payload)):
        _write_npy_atomic(_rel_to_path(rel, staging), payload.arrays[rel])
        written.append(rel)
        if i == 0:
            _check_fail("arrays", fail_at)

    csv_jobs = [
        ("facial/blendshapes.csv", payload.blendshapes.to_csv(index=False)),
        (TIMELINE_FILE, pd.DataFrame(payload.timeline_rows, columns=list(TIMELINE_COLUMNS)).to_csv(index=False)),
        ("hands/handedness.csv", pd.DataFrame(payload.handedness_rows, columns=list(HANDEDNESS_COLUMNS)).to_csv(index=False)),
    ]
    for i, (rel, text) in enumerate(csv_jobs):
        _write_text_atomic(_rel_to_path(rel, staging), text)
        written.append(rel)
        if i == 0:
            _check_fail("csv", fail_at)

    quality_text = json.dumps(payload.quality, indent=2, sort_keys=True)
    _write_text_atomic(_rel_to_path("quality/tracking_quality.json", staging), quality_text)
    written.append("quality/tracking_quality.json")
    return written


def _validate_staged(staging: Path, payload: V2Payload, written: list[str], fail_at: str | None) -> None:
    n = len(payload.timeline_rows)
    arrays = {}
    issues = []
    for rel in written:
        path = _rel_to_path(rel, staging)
        if rel.endswith(".npy"):
            if not path.is_file():
                issues.append(f"{rel} was not written")
                continue
            arrays[rel] = np.load(path, allow_pickle=False)
        elif rel.endswith(".csv"):
            if not path.is_file():
                issues.append(f"{rel} was not written")
                continue
            header = pd.read_csv(path, nrows=0).columns.tolist()
            expected = {
                "facial/blendshapes.csv": list(BLENDSHAPE_COLUMNS),
                TIMELINE_FILE: list(TIMELINE_COLUMNS),
                "hands/handedness.csv": list(HANDEDNESS_COLUMNS),
            }[rel]
            if header != expected:
                issues.append(f"{rel} header does not match the schema")
            rows = len(pd.read_csv(path))
            if rows != n:
                issues.append(f"{rel} has {rows} rows, expected {n}")
        elif rel.endswith(".json"):
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{rel} is not valid JSON: {exc}")
    issues += validate_arrays(arrays, n)
    if issues:
        raise V2ValidationError("; ".join(issues))
    _check_fail("validate", fail_at)


def _build_manifest(staging: Path, written: list[str], payload: V2Payload, fail_at: str | None) -> dict:
    entries = []
    for rel in written:
        path = _rel_to_path(rel, staging)
        entry = {"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        if rel.endswith(".npy"):
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            entry["shape"] = list(arr.shape)
            entry["dtype"] = str(arr.dtype)
        entries.append(entry)
    _check_fail("manifest", fail_at)
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frame_count": len(payload.timeline_rows),
        "files": entries,
    }


def verify_manifest(root: Path, manifest: dict, check_unexpected: bool = False) -> list[str]:
    """Check every manifest entry against the files under root. Returns a list of problems."""
    root = Path(root)
    issues = validate_manifest(manifest)
    listed = set()
    for entry in manifest.get("files", []):
        rel = entry.get("path", "")
        listed.add(rel)
        try:
            path = _rel_to_path(rel, root)
        except V2ValidationError as exc:
            issues.append(str(exc))
            continue
        if not path.is_file():
            issues.append(f"missing file: {rel}")
            continue
        if path.stat().st_size != entry.get("bytes"):
            issues.append(f"size mismatch: {rel}")
        if _sha256(path) != entry.get("sha256"):
            issues.append(f"checksum mismatch: {rel}")
        if "shape" in entry:
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(arr.shape) != entry["shape"] or str(arr.dtype) != entry.get("dtype"):
                    issues.append(f"shape or dtype mismatch: {rel}")
            except (OSError, ValueError) as exc:
                issues.append(f"unreadable array {rel}: {exc}")
    if check_unexpected:
        for rel in _v2_files_on_disk(root):
            if rel not in listed:
                issues.append(f"unexpected file not in manifest: {rel}")
    return issues


def _v2_files_on_disk(root: Path) -> list[str]:
    found = []
    for directory in V2_DIRECTORIES:
        folder = root / directory
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file():
                    found.append(path.relative_to(root).as_posix())
    # manifest.json and metadata.json are structural and cannot list themselves
    if (root / TIMELINE_FILE).is_file():
        found.append(TIMELINE_FILE)
    return sorted(found)


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _staging_dirs(session: Path) -> list[Path]:
    return sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith(STAGING_PREFIX))


def _previous_dirs(session: Path) -> list[Path]:
    return sorted(p for p in session.iterdir() if p.is_dir() and p.name.startswith(PREVIOUS_PREFIX))


# ---------------------------------------------------------------- publication

def _move_previous(session: Path, manifest: dict | None) -> Path | None:
    """Move the existing version 2 files aside. Nothing is deleted."""
    if manifest is None:
        # Without a readable manifest we cannot tell which files are version 2 output, so refuse
        # rather than overwrite them. Nothing has been moved at this point.
        raise V2Error("existing version 2 manifest is unreadable; refusing to replace the dataset")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    previous = session / f"{PREVIOUS_PREFIX}{stamp}-{uuid.uuid4().hex[:8]}"
    listed = [e["path"] for e in manifest.get("files", [])] + [MANIFEST_FILE, METADATA_FILE]
    for rel in listed:
        if rel in V1_PROTECTED_NAMES:
            raise V2Error(f"refusing to move a version 1 name: {rel}")
        src = _rel_to_path(rel, session)
        if src.is_file():
            dest = _rel_to_path(rel, previous)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest)
    return previous


def _restore_previous(session: Path, previous: Path) -> None:
    """Move every file from a previous-version folder back into the session. Refuses to overwrite."""
    for path in sorted(previous.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(previous).as_posix()
        dest = _rel_to_path(rel, session)
        if dest.exists():
            raise V2Error(f"cannot restore {rel}: a file is already in place")
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dest)
    shutil.rmtree(previous, ignore_errors=True)


def _publish(session: Path, staging: Path, manifest: dict, fail_at: str | None) -> None:
    # manifest first: it names every file that will exist, so an interruption is identifiable
    os.replace(staging / MANIFEST_FILE, session / MANIFEST_FILE)
    entries = [e["path"] for e in manifest["files"]]
    for i, rel in enumerate(entries):
        dest = _rel_to_path(rel, session)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(_rel_to_path(rel, staging), dest)
        if fail_at == "publish" and i == 1:
            raise SimulatedInterruption("publish")
    _check_fail("metadata", fail_at)
    os.replace(staging / METADATA_FILE, session / METADATA_FILE)


# ---------------------------------------------------------------- public operations

def write_v2_session(session_dir, payload: V2Payload, regenerate: bool = False,
                     fail_at: str | None = None) -> dict:
    """Write one version 2 session. Returns the published manifest.

    regenerate=False raises V2ExistsError when a complete version 2 dataset already exists.
    regenerate=True moves the existing version 2 files to a .v2-previous-* folder first.
    fail_at simulates an interruption at one stage (tests only).
    """
    if fail_at is not None and fail_at not in FAIL_POINTS:
        raise ValueError(f"unknown fail_at {fail_at!r}")
    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)

    layout = session_layout(session)
    if layout == "incomplete_v2":
        recover_incomplete_v2(session)
        layout = session_layout(session)
    if layout == "v2" and not regenerate:
        raise V2ExistsError(f"complete version 2 dataset exists: {session}. Use regenerate=True (--regenerate-v2).")

    prevalidate(payload)

    # Refuse before staging if a planned destination is already occupied by something that is
    # not part of an existing version 2 dataset. Such files are never overwritten.
    owned = set()
    if layout == "v2":
        old_manifest = _load_manifest(session / MANIFEST_FILE)
        if old_manifest is None:
            raise V2Error("existing version 2 manifest is unreadable; refusing to replace the dataset")
        owned = {e["path"] for e in old_manifest.get("files", [])} | {MANIFEST_FILE, METADATA_FILE}
    collisions = [rel for rel in _planned_paths(payload)
                  if rel not in owned and _rel_to_path(rel, session).exists()]
    if collisions:
        raise V2Error("destination already exists, refusing to overwrite: " + ", ".join(collisions))

    staging = session / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        manifest = _stage_and_verify(staging, payload, fail_at)
    except SimulatedInterruption:
        raise  # a real crash cannot clean up, so the staging folder is left for recovery
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # an ordinary error: leave nothing behind
        raise

    if layout == "v2":
        _move_previous(session, _load_manifest(session / MANIFEST_FILE))
    _publish(session, staging, manifest, fail_at)
    shutil.rmtree(staging, ignore_errors=True)
    return manifest


def _stage_and_verify(staging: Path, payload: V2Payload, fail_at: str | None) -> dict:
    """Write, validate and checksum every staged file. Returns the manifest, which is also written to staging."""
    written = _write_staging(staging, payload, fail_at)
    _validate_staged(staging, payload, written, fail_at)

    manifest = _build_manifest(staging, written, payload, fail_at)
    issues = verify_manifest(staging, manifest)
    if issues:
        raise V2ValidationError("manifest verification failed: " + "; ".join(issues))

    # Write the manifest file first, then record the checksum of its exact bytes in metadata.json.
    _write_text_atomic(staging / MANIFEST_FILE, json.dumps(manifest, indent=2, sort_keys=True))
    metadata = json.loads(json.dumps(payload.metadata))
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["outputs"] = {
        "manifest_file": MANIFEST_FILE,
        "manifest_sha256": _sha256(staging / MANIFEST_FILE),
        "file_count": len(manifest["files"]),
    }
    metadata_issues = validate_metadata(metadata)
    if metadata_issues:
        raise V2ValidationError("metadata: " + "; ".join(metadata_issues))
    _write_text_atomic(staging / METADATA_FILE, json.dumps(metadata, indent=2, sort_keys=True))
    return manifest


def recover_incomplete_v2(session_dir) -> list[str]:
    """Undo an interrupted version 2 operation. Touches only version 2 files and staging folders.

    Removes staged leftovers, removes published files listed in the manifest of an unfinished
    publication, and restores a previous version 2 dataset that was moved aside.
    Returns the list of actions taken.
    """
    session = Path(session_dir)
    actions = []
    for staging in _staging_dirs(session):
        shutil.rmtree(staging, ignore_errors=True)
        actions.append(f"removed staging folder {staging.name}")

    if (session / METADATA_FILE).is_file():
        return actions  # already complete: nothing to recover

    manifest_path = session / MANIFEST_FILE
    if manifest_path.is_file():
        manifest = _load_manifest(manifest_path)
        if manifest is not None:
            for entry in manifest.get("files", []):
                rel = entry.get("path", "")
                path = _rel_to_path(rel, session)
                if path.is_file():
                    path.unlink()
                    actions.append(f"removed interrupted file {rel}")
        manifest_path.unlink()
        actions.append(f"removed {MANIFEST_FILE}")
        _remove_empty_v2_dirs(session)

    for previous in _previous_dirs(session):
        _restore_previous(session, previous)
        actions.append(f"restored previous version 2 dataset from {previous.name}")
    return actions


def rollback_v2(session_dir) -> list[str]:
    """Remove a complete version 2 dataset and restore the previous one if it was kept.

    Only files named in the manifest are removed, and only when their checksum still matches.
    A file that has changed is reported and left in place.
    """
    session = Path(session_dir)
    manifest = _load_manifest(session / MANIFEST_FILE)
    if manifest is None:
        raise V2Error("no readable manifest: nothing to roll back safely")
    issues = verify_manifest(session, manifest)
    changed = [i for i in issues if i.startswith("checksum mismatch") or i.startswith("size mismatch")]
    if changed:
        raise V2Error("refusing to roll back modified files: " + "; ".join(changed))

    actions = []
    for entry in manifest["files"]:
        path = _rel_to_path(entry["path"], session)
        if path.is_file():
            path.unlink()
            actions.append(f"removed {entry['path']}")
    for name in (METADATA_FILE, MANIFEST_FILE):
        if (session / name).is_file():
            (session / name).unlink()
            actions.append(f"removed {name}")
    _remove_empty_v2_dirs(session)
    for previous in _previous_dirs(session):
        _restore_previous(session, previous)
        actions.append(f"restored previous version 2 dataset from {previous.name}")
    return actions


def _remove_empty_v2_dirs(session: Path) -> None:
    for directory in V2_DIRECTORIES:
        folder = session / directory
        if folder.is_dir():
            for sub in sorted(folder.rglob("*"), reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
            if not any(folder.iterdir()):
                folder.rmdir()
