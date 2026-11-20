"""Evaluation of H3 gesture predictions against manual annotations (phase H4).

Matching: an annotation matches the observation with the same person, session and frame, and either the same
track_id (when the annotation has one) or the same hand side (when it does not).

Scoring
    Labels "occluded" are not scored as a gesture. They are counted separately.
    Only annotations at or above the minimum certainty are scored (default "probable").
    Predictions "unknown" are scored like any class: they are a prediction, and a wrong unknown is an error.
    Precision, recall and F1 per class are None (not 0) when the denominator is zero.
    Partially visible and fully visible observations are reported separately, using the H2/H3 quality flag.

Calibration and evaluation sets must be separate sessions. A split file lists them; sessions in both sets are
refused. Thresholds may be tuned on the calibration set only. The evaluation set is reported once per change.

Nothing here claims accuracy. The metrics describe the labelled sessions they were computed on, and only those.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from gesture_rules import GESTURES  # noqa: E402
from hand_annotation import frame_labels  # noqa: E402

CERTAINTY_RANK = {"certain": 2, "probable": 1, "uncertain": 0}
SCORED = tuple(GESTURES)  # open_palm, closed_fist, pointing, thumbs_up, peace, unknown


class SplitError(ValueError):
    """Calibration and evaluation sessions overlap, or the split file is malformed."""


def load_predictions(session_dir, person: str, session: str) -> pd.DataFrame:
    """H3 frame-level predictions for one session, with the identifiers needed for matching."""
    path = Path(session_dir) / "hands" / "gesture_observations.csv"
    df = pd.read_csv(path)
    df.insert(0, "session", session)
    df.insert(0, "person", person)
    return df


def load_split(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cal = {tuple(s) for s in data.get("calibration", [])}
    ev = {tuple(s) for s in data.get("evaluation", [])}
    if not cal or not ev:
        raise SplitError("a split needs at least one calibration and one evaluation session")
    if cal & ev:
        raise SplitError(f"sessions in both sets: {sorted(cal & ev)}")
    return {"calibration": sorted(cal), "evaluation": sorted(ev)}


def match(predictions: pd.DataFrame, labels: list[dict]) -> tuple[list[dict], int]:
    """Pairs of (label, prediction row) and the number of labels with no matching observation."""
    pairs = []
    missing = 0
    index = {}
    for r in predictions.to_dict("records"):
        index.setdefault((r["person"], r["session"], int(r["frame"])), []).append(r)
    for lab in labels:
        cands = index.get((lab["person"], lab["session"], lab["frame"]), [])
        if lab["track_id"] is not None:
            cands = [c for c in cands if int(c["track_id"]) == lab["track_id"]]
        elif lab["hand"] is not None:
            cands = [c for c in cands if c["label"] == lab["hand"]]
        if not cands:
            missing += 1
            continue
        pairs.append((lab, cands[0]))
    return pairs, missing


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None
    return p, r, f1


def evaluate(predictions: pd.DataFrame, labels: list[dict], min_certainty: str = "probable") -> dict:
    """Metrics for one set of labelled frames against the predictions."""
    if min_certainty not in CERTAINTY_RANK:
        raise ValueError(f"min_certainty must be one of {sorted(CERTAINTY_RANK)}")
    keep = [lab for lab in labels if CERTAINTY_RANK[lab["certainty"]] >= CERTAINTY_RANK[min_certainty]]
    occluded = [lab for lab in keep if lab["label"] == "occluded"]
    scored_labels = [lab for lab in keep if lab["label"] != "occluded"]
    pairs, missing = match(predictions, scored_labels)

    idx = {g: i for i, g in enumerate(SCORED)}
    conf = np.zeros((len(SCORED), len(SCORED)), dtype=int)
    for lab, pred in pairs:
        conf[idx[lab["label"]], idx[pred["frame_class"]]] += 1

    per_class = {}
    for g in SCORED:
        i = idx[g]
        tp = int(conf[i, i])
        fn = int(conf[i, :].sum() - tp)
        fp = int(conf[:, i].sum() - tp)
        p, r, f1 = _prf(tp, fp, fn)
        per_class[g] = {"support": int(conf[i, :].sum()), "precision": p, "recall": r, "f1": f1}
    scored_total = int(conf.sum())
    correct = int(np.trace(conf))
    macro = [v["f1"] for v in per_class.values() if v["support"] > 0 and v["f1"] is not None]
    predicted_unknown = int(conf[:, idx["unknown"]].sum())

    def group(flag):
        sub = [(lab, pred) for lab, pred in pairs if pred["quality"] == flag]
        n = len(sub)
        acc = (sum(1 for lab, pred in sub if lab["label"] == pred["frame_class"]) / n) if n else None
        unk = (sum(1 for _, pred in sub if pred["frame_class"] == "unknown") / n) if n else None
        return {"observations": n, "accuracy": acc, "unknown_rate": unk}

    return {
        "min_certainty": min_certainty,
        "labelled_frames": len(keep),
        "scored": scored_total,
        "unmatched_labels": missing,
        "occluded_labels": len(occluded),
        "accuracy": (correct / scored_total) if scored_total else None,
        "unknown_rate": (predicted_unknown / scored_total) if scored_total else None,
        "macro_f1": float(np.mean(macro)) if macro else None,
        "per_class": per_class,
        "confusion_matrix": {"labels": list(SCORED), "rows_truth": conf.tolist()},
        "by_visibility": {"fully_visible": group("ok"), "partially_visible": group("partial_view")},
    }


def evaluate_sessions(labels_df: pd.DataFrame, predictions: dict[tuple[str, str], pd.DataFrame],
                      sessions: list[tuple[str, str]], min_certainty: str = "probable") -> dict:
    """Evaluate the labelled frames of the given sessions only."""
    wanted = set(sessions)
    records = [r for r in frame_labels(labels_df) if (r["person"], r["session"]) in wanted]
    preds = [predictions[s] for s in sessions if s in predictions]
    combined = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["person", "session", "frame"])
    result = evaluate(combined, records, min_certainty)
    result["sessions"] = [list(s) for s in sessions]
    return result


def evaluate_split(split: dict, labels_df: pd.DataFrame, predictions: dict, min_certainty: str = "probable") -> dict:
    """Metrics for the calibration and evaluation sets, reported separately. Refuses overlapping sets."""
    cal, ev = split["calibration"], split["evaluation"]
    if set(cal) & set(ev):
        raise SplitError("calibration and evaluation sessions overlap")
    return {
        "calibration": evaluate_sessions(labels_df, predictions, cal, min_certainty),
        "evaluation": evaluate_sessions(labels_df, predictions, ev, min_certainty),
    }


REQUIRED_CHANGE_FIELDS = ("threshold", "original", "proposed", "calibration_set", "evaluation_set",
                          "calibration_before", "calibration_after", "evaluation_before", "evaluation_after",
                          "false_positive_impact", "false_negative_impact")


def threshold_change_record(change: dict) -> list[str]:
    """Problems with a proposed threshold change. A change is recorded only when every field is present."""
    missing = [f for f in REQUIRED_CHANGE_FIELDS if f not in change]
    issues = [f"missing field: {f}" for f in missing]
    if not missing and {tuple(s) for s in change["calibration_set"]} & {tuple(s) for s in change["evaluation_set"]}:
        issues.append("calibration and evaluation sets overlap")
    return issues
