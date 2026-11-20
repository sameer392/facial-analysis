"""Command-line entry point: classify hand gestures for a session that has hand tracking (H3).

Usage:
    python src/extraction/classify_hand_gestures.py <person> <session> [--regenerate]

Requires hand tracking (run analyse_hand_session.py first). No video or model is needed: the rules use the
landmarks already stored in the session.

Outputs, published into the same session: hands/gesture_observations.csv, hands/gesture_events.csv, and the
gesture summary in quality/tracking_quality.json (hands.gestures). The previous pose data is kept as a
.pose-previous-* folder for rollback.

Exit codes:
    0  gestures published
    1  processing failed (bad input or write error); nothing published
    2  refused (bad names, no hand tracking, or gestures already present without --regenerate)
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
from extraction.hand_gesture_builder import GestureInputError, analyse_gesture_payload  # noqa: E402
from gesture_rules import GestureError  # noqa: E402
from project_paths import is_valid_person, is_valid_session, session_dataset_dir  # noqa: E402
from pose_schema import session_layout  # noqa: E402
from pose_writer import PoseDataError, write_pose_session  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify hand gestures from stored hand tracking.")
    parser.add_argument("person", help="person or project folder, e.g. sameer_gupta")
    parser.add_argument("session", help="session ID, e.g. session-001")
    parser.add_argument("--regenerate", action="store_true",
                        help="replace existing gesture output (the previous pose data is kept)")
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
    if hands.get("status") != "complete" or not hands.get("tracking"):
        _error("hand tracking is not available. Run build_hand_session.py and analyse_hand_session.py first.")
        return EXIT_REFUSED
    if hands.get("gestures") and not args.regenerate:
        _error("gesture output already exists. Rerun with --regenerate to replace it.")
        return EXIT_REFUSED

    before = face_fingerprint(dataset)
    try:
        print("Classifying gestures...")
        payload = analyse_gesture_payload(dataset)
        summary = payload.quality["hands"]["gestures"]
        print(f"  observations: {summary['observations']}, events: {summary['events']}, "
              f"unknown observations: {summary['frame_classes']['unknown']}")
        print("Publishing the gestures...")
        write_pose_session(dataset, payload, regenerate=True)
    except (GestureInputError, GestureError, HandBuildError) as exc:
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
    print(f"Gestures published: {dataset}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
