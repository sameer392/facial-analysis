"""Build the face data for one recording: extraction, then analysis, in one run.

Usage:   python src/analysis/build_face_data.py <person> <session>
Example: python src/analysis/build_face_data.py demo_person session-001

Runs extract_faces.py and then analyze_faces.py, with their output passed through unchanged. If extraction
fails, analysis is not run, and the exit code is the extraction's. Neither script is changed by this wrapper,
and both keep their own refusals (existing data is never overwritten).
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ("extract_faces.py", "analyze_faces.py")


def run_steps(person: str, session: str, python: str = sys.executable, steps=STEPS, here: Path = HERE) -> int:
    for script in steps:
        # output is inherited, so the caller sees each script's lines as they are printed
        code = subprocess.call([python, str(here / script), person, session])
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: build_face_data.py <person> <session>", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_steps(sys.argv[1], sys.argv[2]))
