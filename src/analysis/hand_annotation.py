"""Manual gesture annotations for evaluation (phase H4). Kept separate from classifier output.

An annotation file is a CSV with one row per labelled interval (or one row per frame). Annotations are never
written into the dataset folders and never overwrite H3 output. They are compared with predictions only by
analysis/hand_evaluation.py.

Columns (ANNOTATION_COLUMNS)
    annotation_id       unique within the file, e.g. A0001
    person, session     the dataset identifiers, e.g. sameer_gupta, session-003
    track_id            the H2 track ID the label applies to, or blank when the track is not identifiable
    hand                Left or Right. Required when track_id is blank.
    frame_start         first labelled frame (integer, 0-based, as in the timeline)
    frame_end           last labelled frame, inclusive (equal to frame_start for a single frame)
    timestamp_start_ms  optional, checked against the timeline when given
    timestamp_end_ms    optional, checked against the timeline when given
    gesture_label       open_palm, closed_fist, pointing, thumbs_up, peace, unknown, or occluded
    visibility          full, partial or occluded
    certainty           certain, probable or uncertain: the annotator's confidence in the label
    annotator           who labelled it (a name or initials). Required.
    annotation_version  version of the annotation set, e.g. 1.0. Required.
    notes               free text, optional

"occluded" means the hand cannot be seen well enough to judge the gesture. It is a label for the
observation, separate from visibility, which records the annotator's view of the frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from gesture_rules import GESTURES  # noqa: E402

ANNOTATION_LABELS = tuple(GESTURES) + ("occluded",)
VISIBILITY = ("full", "partial", "occluded")
CERTAINTY = ("certain", "probable", "uncertain")
ANNOTATION_COLUMNS = (
    "annotation_id", "person", "session", "track_id", "hand", "frame_start", "frame_end",
    "timestamp_start_ms", "timestamp_end_ms", "gesture_label", "visibility", "certainty",
    "annotator", "annotation_version", "notes",
)


def read_annotations(path) -> pd.DataFrame:
    """Read an annotation CSV. Empty cells become NaN, track ids stay integers when given."""
    df = pd.read_csv(path, dtype={"annotator": str, "annotation_version": str, "notes": str})
    missing = [c for c in ANNOTATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"annotation file is missing columns {missing}")
    return df


def validate_annotations(df: pd.DataFrame, timelines: dict | None = None) -> list[str]:
    """Problems with an annotation table. timelines maps (person, session) to a dict frame -> timestamp_ms."""
    issues = []
    if df.empty:
        return ["annotation file has no rows"]
    if df["annotation_id"].duplicated().any():
        issues.append("annotation_id values are not unique")
    for i, r in df.iterrows():
        where = f"row {i + 2} ({r['annotation_id']})"
        if r["gesture_label"] not in ANNOTATION_LABELS:
            issues.append(f"{where}: gesture_label {r['gesture_label']!r} is not allowed")
        if r["visibility"] not in VISIBILITY:
            issues.append(f"{where}: visibility {r['visibility']!r} is not allowed")
        if r["certainty"] not in CERTAINTY:
            issues.append(f"{where}: certainty {r['certainty']!r} is not allowed")
        if not isinstance(r["annotator"], str) or not r["annotator"].strip():
            issues.append(f"{where}: annotator is required")
        if not isinstance(r["annotation_version"], str) or not r["annotation_version"].strip():
            issues.append(f"{where}: annotation_version is required")
        try:
            fs, fe = int(r["frame_start"]), int(r["frame_end"])
        except (TypeError, ValueError):
            issues.append(f"{where}: frame_start and frame_end must be integers")
            continue
        if fs < 0 or fe < fs:
            issues.append(f"{where}: frame range {fs}..{fe} is invalid")
        track = r["track_id"]
        if pd.isna(track) and r["hand"] not in ("Left", "Right"):
            issues.append(f"{where}: hand must be Left or Right when track_id is blank")
        if not pd.isna(track):
            try:
                int(track)
            except (TypeError, ValueError):
                issues.append(f"{where}: track_id must be an integer or blank")
        if timelines is not None and ("person" in df.columns):
            key = (str(r["person"]), str(r["session"]))
            tl = timelines.get(key)
            if tl is None:
                issues.append(f"{where}: no timeline for {key[0]}/{key[1]}")
            else:
                if fe not in tl:
                    issues.append(f"{where}: frame {fe} is not in the timeline")
                if not pd.isna(r["timestamp_start_ms"]) and fs in tl and int(r["timestamp_start_ms"]) != tl[fs]:
                    issues.append(f"{where}: timestamp_start_ms differs from the timeline")
    issues += _overlap_conflicts(df)
    return issues


def _overlap_conflicts(df: pd.DataFrame) -> list[str]:
    """Two intervals for the same person, session, track (or hand) that overlap in frames must agree."""
    issues = []
    work = df.copy()
    work["who"] = work.apply(lambda r: (str(r["person"]), str(r["session"]),
                                        f"t{int(r['track_id'])}" if not pd.isna(r["track_id"]) else f"h{r['hand']}"),
                             axis=1)
    for who, grp in work.groupby("who"):
        rows = grp.sort_values("frame_start").to_dict("records")
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                x, y = rows[a], rows[b]
                if int(y["frame_start"]) > int(x["frame_end"]):
                    break
                if x["gesture_label"] != y["gesture_label"]:
                    issues.append(f"{x['annotation_id']} and {y['annotation_id']} overlap with different labels "
                                  f"for {who}")
    return issues


def frame_labels(df: pd.DataFrame) -> list[dict]:
    """One record per labelled frame: key (track or hand), frame, label, visibility, certainty, annotator, version."""
    out = []
    for r in df.to_dict("records"):
        track = None if pd.isna(r["track_id"]) else int(r["track_id"])
        for f in range(int(r["frame_start"]), int(r["frame_end"]) + 1):
            out.append({
                "person": str(r["person"]), "session": str(r["session"]),
                "track_id": track, "hand": None if pd.isna(r["hand"]) else r["hand"],
                "frame": f, "label": r["gesture_label"], "visibility": r["visibility"],
                "certainty": r["certainty"], "annotator": r["annotator"],
                "annotation_version": str(r["annotation_version"]), "annotation_id": r["annotation_id"],
            })
    return out
