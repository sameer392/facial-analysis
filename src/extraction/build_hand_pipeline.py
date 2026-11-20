"""Command-line entry point for the Hand Motion pipeline: hand landmarks, tracking, gestures, then validation.

Usage:
    python src/extraction/build_hand_pipeline.py <person> <session> [--regenerate] [--models-dir DIR]

Runs, in order, the approved CLIs of phases H1 to H3, then the read-only H4 validation. Each stage publishes
into the same pose data session through the existing writer. A stage that fails stops the run, and its exit
code is returned. Stages already published stay as they are, and each one is a complete, valid state.

Requires a complete pose data session (Face Build and Pose Build first). Face and pose data are never rebuilt.

Exit codes (as the stages): 0 done, 1 failed, 2 refused, 3 model check failed, 4 face data changed (rolled back).
Stage markers are printed as lines starting with "Stage" so the API job can report progress.
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

from extraction import analyse_hand_session as h2  # noqa: E402
from extraction import build_hand_session as h1  # noqa: E402
from extraction import classify_hand_gestures as h3  # noqa: E402
from extraction.build_pose_session import EXIT_FAILED, EXIT_OK, EXIT_REFUSED  # noqa: E402
from extraction.model_registry import MODELS_DIR  # noqa: E402
from hand_dataset_validation import validate_hand_session  # noqa: E402
from project_paths import is_valid_person, is_valid_session, session_dataset_dir  # noqa: E402
from pose_schema import session_layout  # noqa: E402
import json  # noqa: E402

STAGES = ("hand landmarks", "hand tracking", "gestures", "validation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hand Motion pipeline: landmarks, tracking, gestures, validation.")
    parser.add_argument("person")
    parser.add_argument("session")
    parser.add_argument("--regenerate", action="store_true", help="replace existing hand outputs")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    return parser


def _hand_present(dataset: Path) -> bool:
    try:
        meta = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(meta.get("streams", {}).get("hands"))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not is_valid_person(args.person) or not is_valid_session(args.session):
        print("error: invalid person or session name", file=sys.stderr)
        return EXIT_REFUSED
    dataset = session_dataset_dir(args.person, args.session)
    if session_layout(dataset) != "pose":
        print("error: no complete pose data for this session. Run Face Build and Pose Build first.", file=sys.stderr)
        return EXIT_REFUSED
    if _hand_present(dataset) and not args.regenerate:
        print("error: hand outputs already exist. Rerun with --regenerate to replace them.", file=sys.stderr)
        return EXIT_REFUSED

    flag = ["--regenerate"] if args.regenerate else []
    print(f"Stage 1 of {len(STAGES)}: {STAGES[0]}")
    code = h1.main([args.person, args.session, *flag, "--models-dir", str(args.models_dir)])
    if code != EXIT_OK:
        return code
    print(f"Stage 2 of {len(STAGES)}: {STAGES[1]}")
    code = h2.main([args.person, args.session, *flag])
    if code != EXIT_OK:
        return code
    print(f"Stage 3 of {len(STAGES)}: {STAGES[2]}")
    code = h3.main([args.person, args.session, *flag])
    if code != EXIT_OK:
        return code
    print(f"Stage 4 of {len(STAGES)}: {STAGES[3]}")
    report = validate_hand_session(dataset)
    for e in report.errors:
        print(f"validation error: {e}")
    for w in report.warnings:
        print(f"validation warning: {w}")
    if not report.valid:
        print("Hand validation found errors. The outputs are kept as published; see the messages above.")
        return EXIT_FAILED
    print("Hand Motion pipeline complete. Validation: no errors.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
