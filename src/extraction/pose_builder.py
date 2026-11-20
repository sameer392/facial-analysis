"""Assemble a pose data session payload from face data and a Pose extraction result.

Streams in a pose data session:
    facial  taken from the existing face_data.json (read only; face data is never modified)
    body    Pose Landmarker FULL output from pose_extractor.PoseResult
    hands   not run yet: recorded as an absent stream, with NaN arrays and no detections

The builder refuses to combine streams whose frame counts or timestamps differ. The writer
(pose_writer.write_pose_session) does the validation, staging, manifest and publication.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from extraction.model_registry import get_spec  # noqa: E402
from extraction.pose_extractor import PoseResult  # noqa: E402
from facial_analyzer import Session, load_session, tracking_quality  # noqa: E402
from pose_schema import BLENDSHAPE_NAMES, N_HAND_LANDMARKS, SCHEMA_VERSION  # noqa: E402
from pose_writer import PoseDataPayload  # noqa: E402


class BuildError(Exception):
    """The streams cannot be combined into one correct session."""


FACE_LICENCE_NOTE = "UNVERIFIED: no model-specific licence found; source and download date unknown"
POSE_LICENCE_NOTE = ("partly verified: one-line maintainer statement (GitHub issue #6306); "
                     "not cleared for commercial use or redistribution")
COORDINATE_CONVENTIONS = {
    "image_landmarks": "normalised: x / frame width, y / frame height; z relative on the x scale; origin top-left, y down",
    "world_landmarks": "metres; origin at the hip centre (MediaPipe world coordinates)",
    "transformation_matrix": "4x4; top-left 3x3 is head rotation; see docs/validation.md (axis-flip convention empirically supported)",
}


def _mediapipe_version() -> str:
    try:
        return importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_payload(dataset_dir, pose: PoseResult, fps: float, person: str, session_id: str,
                  video_filename: str, video_size: tuple[int, int]) -> PoseDataPayload:
    """Combine face data (in dataset_dir) with the Pose result into a PoseDataPayload."""
    width, height = video_size
    facial_session = Session(person=person, session_id=session_id, video_path=Path(video_filename),
                             dataset_dir=Path(dataset_dir), fps=float(fps))
    face = load_session(facial_session)
    n = len(face.detected)
    n_pose = pose.image.shape[0]
    if n != n_pose:
        raise BuildError(f"frame counts differ: facial {n}, pose {n_pose}")

    facial_ts = np.round(face.timestamps_s * 1000.0).astype(np.int64)
    if not np.array_equal(facial_ts, pose.timestamps_ms):
        raise BuildError("timestamps differ between the facial and pose streams")

    blend = face.blendshapes.reindex(columns=list(BLENDSHAPE_NAMES)).astype(float)
    blend.insert(0, "face_detected", face.detected)
    blend.insert(0, "timestamp_ms", facial_ts)
    blend.insert(0, "frame", np.arange(n))

    hands_nan = np.full((n, N_HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
    arrays = {
        "facial/landmarks.npy": face.landmarks.astype(np.float32),
        "facial/transformation_matrices.npy": face.matrices.astype(np.float32),
        "body/pose_landmarks.npy": pose.image.astype(np.float32),
        "body/pose_world_landmarks.npy": pose.world.astype(np.float32),
        "hands/left_hand.npy": hands_nan,
        "hands/right_hand.npy": hands_nan.copy(),
    }

    timeline = [{
        "frame": i,
        "timestamp_ms": int(facial_ts[i]),
        "face_detected": bool(face.detected[i]),
        "pose_detected": bool(pose.detected[i]),
        "left_hand_detected": False,
        "right_hand_detected": False,
    } for i in range(n)]
    handedness = [{
        "frame": i,
        "timestamp_ms": int(facial_ts[i]),
        "left_detected": False, "left_model_label": "", "left_score": None,
        "right_detected": False, "right_model_label": "", "right_score": None,
    } for i in range(n)]

    face_q = tracking_quality(face)
    pose_det = int(pose.detected.sum())
    quality = {
        "facial": face_q,
        "body": {"frames": n, "detected_frames": pose_det, "detection_rate": round(pose.detection_rate, 4),
                 "timestamp_source": pose.timestamp_source},
        "hands": {"status": "absent", "detected_frames": 0},
        "scope": "tracking only; does not measure the accuracy of the facial or body motion",
    }

    face_spec = get_spec("face")
    pose_spec = get_spec("pose_full")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "identity": {"person_id": person, "session_id": session_id},
        "source": {"video_filename": video_filename, "fps": float(fps), "frame_count": n,
                   "width": int(width), "height": int(height)},
        "timeline": {"reference": "start of video", "timestamp_unit": "milliseconds",
                     "timestamp_source": pose.timestamp_source, "frame_index_base": 0,
                     "timestamp_rule": "round(frame_index * 1000 / fps) when fps_derived"},
        "extraction": {"software": "facial-analysis", "software_version": "2.0.0-dev",
                       "python_version": "3.11", "mediapipe_version": _mediapipe_version(),
                       "running_mode": "VIDEO"},
        "models": {
            "face": {"file": face_spec.filename, "sha256": face_spec.sha256.lower(), "version": face_spec.version,
                     "source": face_spec.source_url, "licence_status": "to verify",
                     "licence_note": FACE_LICENCE_NOTE},
            "pose_full": {"file": pose_spec.filename, "sha256": pose_spec.sha256.lower(), "version": pose_spec.version,
                          "source": pose_spec.source_url, "licence_status": "to verify",
                          "licence_note": POSE_LICENCE_NOTE},
        },
        "configuration": {
            "face": {"num_faces": 1, "output_blendshapes": True, "output_transformation_matrices": True},
            "pose": pose.config,
            "hands": {"status": "not run"},
        },
        "coordinate_conventions": dict(COORDINATE_CONVENTIONS),
        "streams": {
            "facial": {"status": "complete", "detected_frames": int(face.detected.sum())},
            "body": {"status": "complete", "detected_frames": pose_det},
            "hands": {"status": "absent", "detected_frames": 0},
        },
        "outputs": {},
    }
    return PoseDataPayload(arrays=arrays, timeline_rows=timeline, handedness_rows=handedness,
                     blendshapes=blend, quality=quality, metadata=metadata)
