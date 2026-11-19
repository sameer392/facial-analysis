"""Registry of MediaPipe model files and SHA-256 verification before any model is loaded.

Each entry records what is known about a model file: its filename, the expected SHA-256, the
version (or "unpinned (latest)" when none is established) and the licence status. Licence status
is evidence, not clearance: see THIRD_PARTY_NOTICES.md. Nothing here downloads a model.

A file that is missing or whose hash does not match is refused, so it cannot be loaded by accident.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "mediapipe"
CHUNK_SIZE = 1024 * 1024


class ModelIntegrityError(Exception):
    """A model file is missing or does not match its recorded SHA-256."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    sha256: str
    version: str
    source_url: str
    licence_status: str


MODELS = {
    "face": ModelSpec(
        key="face",
        filename="face_landmarker.task",
        sha256="64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF",
        version="unknown (already present before provenance recording)",
        source_url="unknown",
        licence_status="UNVERIFIED",
    ),
    "pose_full": ModelSpec(
        key="pose_full",
        filename="pose_landmarker_full.task",
        sha256="4EAA5EB7A98365221087693FCC286334CF0858E2EB6E15B506AA4A7ECDCEC4AD",
        version="unpinned (latest)",
        source_url=("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
        licence_status="partly verified (one-line maintainer statement); not cleared for commercial use or redistribution",
    ),
    "hand": ModelSpec(
        key="hand",
        filename="hand_landmarker.task",
        sha256="FBC2A30080C3C557093B5DDFC334698132EB341044CCEE322CCF8BCF3607CDE1",
        version="unpinned (latest)",
        source_url=("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/latest/hand_landmarker.task"),
        licence_status="UNVERIFIED",
    ),
}


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise KeyError(f"unknown model: {key!r}. Known models: {sorted(MODELS)}") from None


def sha256_file(path, chunk_size: int = CHUNK_SIZE) -> str:
    """Upper-case hex SHA-256, computed in chunks so large model files are not read into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_model(spec: ModelSpec, models_dir=None) -> Path:
    """Return the model's path if it exists and matches the recorded SHA-256. Otherwise raise."""
    folder = Path(models_dir) if models_dir is not None else MODELS_DIR
    path = folder / spec.filename
    if not path.is_file():
        raise ModelIntegrityError(f"model not found: {path} (download it and place it there)")
    actual = sha256_file(path)
    if actual != spec.sha256.upper():
        raise ModelIntegrityError(
            f"SHA-256 mismatch for {spec.filename}: expected {spec.sha256.upper()}, got {actual}. "
            "The file has changed or is not the recorded model; it has not been loaded."
        )
    return path
