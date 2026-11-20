"""Command-line entry point: track hands and compute hand motion for a session that has the hand stream.

Usage:
    python src/extraction/analyse_hand_session.py <person> <session> [--regenerate]

Requires a session whose hand stream is complete (run build_hand_session.py first). No video or model
is needed: the tracking uses the landmarks already stored in the session.

The tracks and motion tables are published into the same session, with the rest of the pose data
unchanged. The previous pose data is kept as a .pose-previous-* folder for rollback.

Exit codes:
    0  tracking and motion published
    1  processing failed (bad input, tracking or write error); nothing published
    2  refused (bad names, no hand stream, or tracking already present without --regenerate)
    4  a face data file changed during the run; the output is rolled back to the previous pose data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ANALYSIS = _SRC / "analysis"
for _p in (_SRC, _ANALYSIS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extraction.build_pose_session import EXIT_FACE_CHANGED, EXIT_FAILED, EXIT_OK, EXIT_REFUSED  # noqa: E402
from extraction.build_pose_session import _rollback_after_face_change, face_fingerprint  # noqa: E402
from extraction.hand_builder import HandBuildError  # noqa: E402
from extraction.hand_motion_builder import analyse_hand_payload  # noqa: E402
from hand_tracking import TrackingError  # noqa: E402
from project_paths import is_valid_person, is_valid_session, session_dataset_dir  # noqa: E402
from pose_schema import session_layout  # noqa: E402
from pose_writer import PoseDataError, write_pose_session  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track hands and compute hand motion from stored hand landmarks.")
    parser.add_argument("person", help="person or project folder, e.g. sameer_gupta")
    parser.add_argument("session", help="session ID, e.g. session-001")
    parser.add_argument("--regenerate", action="store_true",
                        help="replace existing tracking and motion output (the previous pose data is kept)")
    return parser


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not is_valid_person(args.person):
        _error(f"invalid person or project name: {args.person!r}")
        return EXIT_REFUSED
    if not is_valid_session(args.session):
        _error(f"invalid session name (expected session-NNN): {args.session!r}")
        return EXIT_REFUSED

    dataset = session_dataset_dir(args.person, args.session)
    if session_layout(dataset) != "pose":
        _error("no complete pose data for this session. Run build_pose_session.py first.")
        return EXIT_REFUSED
    meta = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    hands = meta.get("streams", {}).get("hands", {})
    if hands.get("status") != "complete":
        _error("the hand stream is not complete. Run build_hand_session.py first.")
        return EXIT_REFUSED
    if hands.get("tracking") and not args.regenerate:
        _error("hand tracking already exists. Rerun with --regenerate to replace it.")
        return EXIT_REFUSED

    before = face_fingerprint(dataset)
    try:
        print("Tracking hands and computing motion...")
        payload = analyse_hand_payload(dataset)
        tracking = payload.quality["hands"]["tracking"]
        print(f"  tracks: {tracking['tracks_created']}, observations: {tracking['observations']}, "
              f"interruptions: {tracking['interruptions']}, reacquisitions: {tracking['reacquisitions']}")
        print("Publishing the tracks and motion...")
        write_pose_session(dataset, payload, regenerate=True)
    except (HandBuildError, TrackingError) as exc:
        _error(str(exc))
        return EXIT_FAILED
    except PoseDataError as exc:
        _error(f"{exc}. If no other run is active, run recover_incomplete_pose for this session.")
        return EXIT_FAILED

    after = face_fingerprint(dataset)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        _error("face data files changed during processing: " + ", ".join(changed))
        _error(_rollback_after_face_change(dataset, had_previous=True))
        return EXIT_FACE_CHANGED
    print(f"Hand tracking published: {dataset}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
