"""Command-line entry point: validate a hand dataset and print its quality summary (phase H4). Read-only.

Usage:
    python src/extraction/validate_hand_session.py <person> <session> [--json PATH]

Exit codes:
    0  no errors (warnings may be present)
    1  errors found; the dataset is not valid
    2  refused (bad names or no pose data session)

A session with no errors is structurally consistent. It is not thereby scientifically accurate.
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

from hand_dataset_validation import validate_hand_session  # noqa: E402
from project_paths import is_valid_person, is_valid_session, session_dataset_dir  # noqa: E402

EXIT_VALID, EXIT_INVALID, EXIT_REFUSED = 0, 1, 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a hand dataset (read-only).")
    parser.add_argument("person")
    parser.add_argument("session")
    parser.add_argument("--json", type=Path, help="also write the full report to this file")
    args = parser.parse_args(argv)
    if not is_valid_person(args.person) or not is_valid_session(args.session):
        print("error: invalid person or session name", file=sys.stderr)
        return EXIT_REFUSED
    dataset = session_dataset_dir(args.person, args.session)
    report = validate_hand_session(dataset)
    if report.metrics == {} and report.errors and not (dataset / "metadata.json").is_file():
        print("error: no pose data session here", file=sys.stderr)
        return EXIT_REFUSED
    print(f"Session {args.person}/{args.session}: {'VALID' if report.valid else 'INVALID'}")
    for e in report.errors:
        print(f"  error: {e}")
    for w in report.warnings:
        print(f"  warning: {w}")
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"report written: {args.json}")
    return EXIT_VALID if report.valid else EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
