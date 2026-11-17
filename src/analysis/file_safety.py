"""Overwrite protection and safe file writing for derived outputs.

Rules used across the project:
- Original recordings (recordings/raw/) and extraction results (facial_data.json) are
  never written by analysis code.
- Extraction never overwrites an existing dataset. There is no --force for extraction.
- Analysis refuses to overwrite existing outputs unless --force is given.
- Writes go to a temporary file in the same folder, then replace the target in one step,
  so an interrupted run never leaves a half-written file behind.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class OutputExistsError(FileExistsError):
    """Raised when an output already exists and force is not set."""


def check_writable(paths, force: bool = False) -> None:
    """Raise OutputExistsError if any target exists, unless force is True. Writes nothing."""
    existing = [str(p) for p in map(Path, paths) if p.exists()]
    if existing and not force:
        raise OutputExistsError(
            "Refusing to overwrite existing output (rerun with --force to regenerate):\n  "
            + "\n  ".join(existing)
        )


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return path


def atomic_write_csv(df, path: Path) -> Path:
    return atomic_write_text(path, df.to_csv(index=False))
