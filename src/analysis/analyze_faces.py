"""Write the blendshape table and motion/quality outputs for one session.

Usage:   python src/analysis/analyze_faces.py <person> <session> [--force]
Example: python src/analysis/analyze_faces.py demo_person session-001

Requires extract_faces.py to have produced datasets/<person>/<session>/facial_data.json.
Writes into datasets/<person>/<session>/ and refreshes datasets/<person>/dataset_metadata.json.

By default the script refuses to overwrite existing outputs. Pass --force to regenerate them.
facial_data.json and the recording are never modified.
"""

import argparse

import pandas as pd

from facial_analyzer import (
    analyse_session,
    load_session,
    make_session,
    motion_output_paths,
    save_session_outputs,
    write_person_metadata,
)
from file_safety import OutputExistsError, atomic_write_csv, check_writable
from project_paths import is_valid_person, is_valid_session, session_dataset_dir


def build_blendshape_table(ds) -> pd.DataFrame:
    meta = pd.DataFrame({
        "frame": range(len(ds.timestamps_s)),
        "timestamp_ms": [round(t * 1000) for t in ds.timestamps_s],
        "face_detected": ds.detected,
    })
    return pd.concat([meta, ds.blendshapes.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write blendshape CSV and motion outputs for one session.")
    parser.add_argument("person", help="Person or project folder, e.g. demo_person")
    parser.add_argument("session", help="Session ID, e.g. session-001")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing analysis outputs (never touches facial_data.json)")
    args = parser.parse_args()

    if not is_valid_person(args.person):
        parser.error(f"Invalid person/project name: {args.person}")
    if not is_valid_session(args.session):
        parser.error(f"Invalid session ID (expected session-NNN): {args.session}")

    session = make_session(args.person, args.session)
    if not session.json_path.is_file():
        raise SystemExit(f"Missing {session.json_path}. Run extract_faces.py first.")

    csv_path = session_dataset_dir(args.person, args.session) / "facial_blendshapes.csv"
    try:
        check_writable([csv_path, *motion_output_paths(session)], force=args.force)
    except OutputExistsError as exc:
        raise SystemExit(str(exc))

    ds = load_session(session)
    atomic_write_csv(build_blendshape_table(ds), csv_path)
    print("CSV created:", csv_path)
    print("Total frames:", len(ds.timestamps_s))

    result = analyse_session(session)
    out = save_session_outputs(result, session, force=True)  # already checked above
    print("Motion outputs:", out)
    print("Metadata:", write_person_metadata(args.person))
