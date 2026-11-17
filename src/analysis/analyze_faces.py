import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Recording to process. Pass an ID on the command line, e.g. "python analyze_faces.py presenter-02",
# or change the default below.
VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "presenter-01"
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "datasets" / VIDEO_ID

INPUT = DATASET_DIR / "facial_data.json"
OUTPUT = DATASET_DIR / "facial_blendshapes.csv"

if not INPUT.is_file():
    raise FileNotFoundError(f"Landmark data not found: {INPUT}")

with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for frame in data:

    row = {
        "frame": frame["frame"],
        "timestamp_ms": frame["timestamp_ms"],
        "face_detected": frame["face_detected"]
    }

    # Extract facial blendshape scores
    blendshapes = frame.get("blendshapes", {})

    for name, score in blendshapes.items():
        row[name] = score

    rows.append(row)

df = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("Video:", VIDEO_ID)
print("CSV created:", OUTPUT)
print("Total frames:", len(df))
