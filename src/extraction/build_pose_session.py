"""Command-line entry point: process an existing face data session into a pose data session with Pose output.

Usage:
    python src/extraction/build_pose_session.py <person> <session> [--regenerate] [--models-dir DIR]

Example:
    python src/extraction/build_pose_session.py demo_person session-001

Requires: the raw video recordings/raw/<person>/<session>.mp4, the face data
datasets/<person>/<session>/facial_data.json (from extract_faces.py), and the Pose FULL model in
models/mediapipe/ (its SHA-256 is verified before it is loaded).

Exit codes:
    0  pose data published
    1  processing failed (video, extraction, build or write error); nothing published
    2  refused before processing (bad names, missing inputs, existing pose data without --regenerate)
    3  model missing or SHA-256 mismatch; the model was not loaded
    4  a face data file changed during the run. The pose data output of this run is rolled back with
       rollback_pose_session: if there was a previous pose data dataset it is restored, otherwise the new pose data files
       are removed. The CLI reports the outcome. If the rollback is refused or fails, it says so,
       and the files are left in place. Restoration is reported only after the resulting session
       layout has been checked.

Face data files and the raw video are never written. The MediaPipe detector is closed on every path.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ANALYSIS = _SRC / "analysis"
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extraction.model_registry import MODELS_DIR, ModelIntegrityError  # noqa: E402
from extraction.pose_extractor import ExtractionError, extract_pose_video, load_pose_detector  # noqa: E402
from extraction.pose_builder import BuildError, build_payload  # noqa: E402
from extraction.video_frames import VideoError, open_video_info  # noqa: E402
from project_paths import (  # noqa: E402
    is_valid_person,
    is_valid_session,
    session_dataset_dir,
    session_video,
)
from pose_schema import session_layout  # noqa: E402
from pose_writer import PoseDataError, PoseDataExistsError, rollback_pose_session, write_pose_session  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_MODEL, EXIT_FACE_CHANGED = 0, 1, 2, 3, 4
FACE_FILES = ("facial_data.json", "facial_blendshapes.csv", "motion_metrics.csv",
            "blink_events.csv", "tracking_quality.json", "calibration.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pose data (Pose Landmarker output) for a session that already has face data.")
    parser.add_argument("person", help="person or project folder, e.g. sameer_gupta")
    parser.add_argument("session", help="session ID, e.g. session-001")
    parser.add_argument("--regenerate", action="store_true",
                        help="replace an existing pose data dataset (the previous one is kept for rollback)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="folder holding pose_landmarker_full.task (default: models/mediapipe)")
    return parser


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def face_fingerprint(dataset: Path) -> dict:
    """SHA-256 of each face data file, and the names and sizes of the frames folder."""
    fingerprint = {}
    for name in FACE_FILES:
        path = dataset / name
        if path.is_file():
            fingerprint[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    frames = dataset / "frames"
    if frames.is_dir():
        fingerprint["frames/"] = sorted((p.name, p.stat().st_size) for p in frames.iterdir() if p.is_file())
    return fingerprint


def _rollback_after_face_change(dataset: Path, had_previous: bool) -> str:
    """Roll back the pose data output of this run. Returns an honest one-line outcome for the user.

    Restoration is only claimed when the resulting layout is the expected one. A failed or
    refused rollback leaves the files in place and says so.
    """
    expected = "pose" if had_previous else "face"
    try:
        rollback_pose_session(dataset)
    except PoseDataError as exc:
        return (f"rollback FAILED: {exc}. The pose data files from this run were left in place, "
                "so the session state is not confirmed.")
    layout = session_layout(dataset)
    if layout != expected:
        return (f"rollback ran but the session is '{layout}' instead of '{expected}'. "
                "Check the session folder by hand.")
    if had_previous:
        return "rolled back: the previous pose data was restored; face data files are not changed by the rollback."
    return "rolled back: no pose data remains; face data files are not changed by the rollback."


def main(argv=None, detector_factory=None) -> int:
    """Run the CLI. detector_factory is for tests; by default the verified Pose FULL model is loaded."""
    args = build_parser().parse_args(argv)

    if not is_valid_person(args.person):
        _error(f"invalid person or project name: {args.person!r}")
        return EXIT_REFUSED
    if not is_valid_session(args.session):
        _error(f"invalid session name (expected session-NNN): {args.session!r}")
        return EXIT_REFUSED

    video = session_video(args.person, args.session)
    dataset = session_dataset_dir(args.person, args.session)
    if not video.is_file():
        _error(f"raw video not found: {video}")
        return EXIT_REFUSED
    if not (dataset / "facial_data.json").is_file():
        _error(f"face data not found: {dataset / 'facial_data.json'}. "
               "Run extract_faces.py for this session first.")
        return EXIT_REFUSED

    layout = session_layout(dataset)
    if layout == "pose" and not args.regenerate:
        _error("pose data already exists. Rerun with --regenerate to replace it.")
        return EXIT_REFUSED

    try:
        info = open_video_info(video)
    except VideoError as exc:
        _error(str(exc))
        return EXIT_FAILED

    before = face_fingerprint(dataset)
    print(f"Video: {video.name} ({info.width}x{info.height}, {info.fps:.3f} fps)")

    factory = detector_factory or (lambda: load_pose_detector(models_dir=args.models_dir))
    try:
        detector = factory()
    except ModelIntegrityError as exc:
        _error(str(exc))
        return EXIT_MODEL
    print("Pose Landmarker FULL loaded (SHA-256 verified).")

    try:
        print("Extracting pose landmarks...")
        pose = extract_pose_video(video, detector=detector)
        print(f"  frames: {pose.image.shape[0]}, pose detected on {int(pose.detected.sum())} "
              f"({pose.detection_rate:.1%})")
        print("Building the pose data payload...")
        payload = build_payload(dataset_dir=dataset, pose=pose, fps=info.fps,
                                person=args.person, session_id=args.session,
                                video_filename=video.name, video_size=(info.width, info.height))
        print("Publishing the pose data...")
        write_pose_session(dataset, payload, regenerate=args.regenerate)
    except PoseDataExistsError as exc:
        _error(str(exc))
        return EXIT_REFUSED
    except (VideoError, ExtractionError, BuildError) as exc:
        _error(str(exc))
        return EXIT_FAILED
    except PoseDataError as exc:
        _error(f"{exc}. If no other run is active, run recover_incomplete_pose for this session.")
        return EXIT_FAILED
    except Exception as exc:  # report anything unexpected; the detector is still closed below
        _error(f"unexpected error: {type(exc).__name__}: {exc}")
        return EXIT_FAILED
    finally:
        detector.close()

    after = face_fingerprint(dataset)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        _error("face data files changed during processing: " + ", ".join(changed))
        _error(_rollback_after_face_change(dataset, had_previous=(layout == "pose")))
        return EXIT_FACE_CHANGED

    print(f"Pose data published: {dataset}")
    print("Face data files: unchanged.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
