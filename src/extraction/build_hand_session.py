"""Command-line entry point: add the hand stream to an existing pose data session.

Usage:
    python src/extraction/build_hand_session.py <person> <session> [--regenerate] [--models-dir DIR]

Requires a complete pose data session (run build_pose_session.py first), the raw video
recordings/raw/<person>/<session>.mp4, and the Hand Landmarker model in models/mediapipe/
(its SHA-256 is verified before it is loaded).

The hand stream is published into the same session. The face and body arrays are carried over
unchanged. The previous pose data is kept as a .pose-previous-* folder for rollback, as with pose builds.

Exit codes:
    0  hand stream published
    1  processing failed (video, extraction, build or write error); nothing published
    2  refused before processing (bad names, missing inputs, no pose data, hand stream already complete
       without --regenerate)
    3  model missing or SHA-256 mismatch; the model was not loaded
    4  a face data file changed during the run; the hand build is rolled back to the previous pose data

Face data files and the raw video are never written. The detector is closed on every path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ANALYSIS = _SRC / "analysis"
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extraction.build_pose_session import EXIT_FACE_CHANGED, EXIT_FAILED, EXIT_MODEL, EXIT_OK, EXIT_REFUSED  # noqa: E402
from extraction.build_pose_session import _rollback_after_face_change, face_fingerprint  # noqa: E402
from extraction.hand_builder import HandBuildError, build_hand_payload, read_pose_session  # noqa: E402
from extraction.hand_extractor import HandExtractionError, extract_hands_video, load_hand_detector  # noqa: E402
from extraction.model_registry import MODELS_DIR, ModelIntegrityError  # noqa: E402
from extraction.video_frames import VideoError, open_video_info  # noqa: E402
from project_paths import is_valid_person, is_valid_session, session_dataset_dir, session_video  # noqa: E402
from pose_schema import session_layout  # noqa: E402
from pose_writer import PoseDataError, write_pose_session  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add hand landmarks (Hand Landmarker output) to a session that already has pose data.")
    parser.add_argument("person", help="person or project folder, e.g. sameer_gupta")
    parser.add_argument("session", help="session ID, e.g. session-001")
    parser.add_argument("--regenerate", action="store_true",
                        help="replace an existing hand stream (the previous pose data is kept for rollback)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="folder holding hand_landmarker.task (default: models/mediapipe)")
    return parser


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv=None, detector_factory=None) -> int:
    """Run the CLI. detector_factory is for tests; by default the verified Hand Landmarker is loaded."""
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
    if session_layout(dataset) != "pose":
        _error("no complete pose data for this session. Run build_pose_session.py first.")
        return EXIT_REFUSED
    try:
        existing = read_pose_session(dataset)
    except HandBuildError as exc:
        _error(str(exc))
        return EXIT_REFUSED
    hand_status = existing["metadata"].get("streams", {}).get("hands", {}).get("status")
    if hand_status == "complete" and not args.regenerate:
        _error("the hand stream already exists. Rerun with --regenerate to replace it.")
        return EXIT_REFUSED

    try:
        info = open_video_info(video)
    except VideoError as exc:
        _error(str(exc))
        return EXIT_FAILED

    before = face_fingerprint(dataset)
    print(f"Video: {video.name} ({info.width}x{info.height}, {info.fps:.3f} fps)")

    factory = detector_factory or (lambda: load_hand_detector(models_dir=args.models_dir))
    try:
        detector = factory()
    except ModelIntegrityError as exc:
        _error(str(exc))
        return EXIT_MODEL
    print("Hand Landmarker loaded (SHA-256 verified).")

    try:
        print("Extracting hand landmarks...")
        hands = extract_hands_video(video, detector=detector)
        print(f"  frames: {hands.frames}; left detected {hands.detected_frames('left')}, "
              f"right detected {hands.detected_frames('right')}")
        print("Building the hand stream...")
        payload = build_hand_payload(dataset, hands)
        print("Publishing the pose data with the hand stream...")
        write_pose_session(dataset, payload, regenerate=True)
    except (VideoError, HandExtractionError, HandBuildError) as exc:
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
        _error(_rollback_after_face_change(dataset, had_previous=True))
        return EXIT_FACE_CHANGED

    print(f"Hand stream published: {dataset}")
    print("Face data files: unchanged.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
