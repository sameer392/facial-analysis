"""Hand dataset validation, annotations and evaluation (phase H4). Synthetic data only."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

from analysis.hand_dataset_validation import validate_hand_session  # noqa: E402
import hand_annotation as ha  # noqa: E402
import hand_evaluation as he  # noqa: E402
import test_classify_hand_cli as tcl  # noqa: E402  (full H1 to H3 synthetic session)
from extraction.hand_gesture_builder import analyse_gesture_payload  # noqa: E402


class SessionCase(tcl.GestureCase):
    """A complete synthetic session (H1, H2, H3) in a temporary folder, copied for each corruption."""

    def setUp(self):
        super().setUp()
        self.build_all()
        self.assertEqual(self.run_gestures()[0], 0)
        self.good = Path(tempfile.mkdtemp(prefix="hand_valid_"))
        shutil.copytree(self.dataset, self.good / "session", dirs_exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.good, ignore_errors=True)
        super().tearDown()

    def copy(self) -> Path:
        dst = Path(tempfile.mkdtemp(prefix="hand_bad_")) / "session"
        shutil.copytree(self.good / "session", dst)
        self.addCleanup(shutil.rmtree, dst.parent, True)
        return dst

    def errors_of(self, folder):
        return validate_hand_session(folder).errors


def _rewrite_json(path, fn):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fn(data)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class ValidSessionTests(SessionCase):
    def test_a_complete_session_has_no_errors(self):
        report = validate_hand_session(self.good / "session")
        self.assertEqual(report.errors, [])
        self.assertTrue(report.valid)
        self.assertIn("gestures", report.metrics)
        self.assertEqual(report.metrics["detection"]["frames"], tcl.N)

    def test_provenance_is_recorded(self):
        prov = validate_hand_session(self.good / "session").provenance
        self.assertIn("mediapipe_version", prov["extraction"])
        self.assertTrue(prov["hand_model"]["sha256"])
        self.assertTrue(prov["processing_timestamp_utc"].endswith("Z"))
        self.assertEqual(prov["source"]["fps"], tcl.tah.tbh.tbc.FPS)

    def test_validation_is_reproducible(self):
        a = validate_hand_session(self.good / "session").to_dict()
        b = validate_hand_session(self.good / "session").to_dict()
        self.assertEqual(a, b)

    def test_classification_is_reproducible_from_the_same_inputs(self):
        p1 = analyse_gesture_payload(self.good / "session")
        p2 = analyse_gesture_payload(self.good / "session")
        for rel in ("hands/gesture_observations.csv", "hands/gesture_events.csv"):
            a = pd.DataFrame(p1.hand_tables[rel])
            b = pd.DataFrame(p2.hand_tables[rel])
            self.assertTrue(a.equals(b), rel)

    def test_validation_does_not_change_the_dataset(self):
        before = {p.name: p.read_bytes() for p in (self.good / "session" / "hands").iterdir() if p.is_file()}
        validate_hand_session(self.good / "session")
        after = {p.name: p.read_bytes() for p in (self.good / "session" / "hands").iterdir() if p.is_file()}
        self.assertEqual(before, after)


class CorruptionTests(SessionCase):
    def test_missing_tracking_file_is_an_error(self):
        d = self.copy()
        (d / "hands" / "tracks.csv").unlink()
        self.assertTrue(any("tracks.csv" in e and "missing" in e for e in self.errors_of(d)))

    def test_incorrect_array_shape_is_an_error(self):
        d = self.copy()
        np.save(d / "hands" / "left_hand.npy", np.zeros((tcl.N, 20, 3), np.float32))
        self.assertTrue(any("left_hand.npy: shape" in e for e in self.errors_of(d)))

    def test_invalid_coordinates_are_an_error(self):
        d = self.copy()
        arr = np.load(d / "hands" / "right_hand.npy")
        arr[0, 0, 0] = np.nan   # frame 0 is a detected right hand
        np.save(d / "hands" / "right_hand.npy", arr)
        self.assertTrue(any("not finite exactly where" in e for e in self.errors_of(d)))

    def test_inconsistent_timestamps_are_an_error(self):
        d = self.copy()
        tl = pd.read_csv(d / "timeline.csv")
        tl.loc[3, "timestamp_ms"] = tl.loc[2, "timestamp_ms"]
        tl.to_csv(d / "timeline.csv", index=False)
        self.assertTrue(any("strictly increase" in e for e in self.errors_of(d)))

    def test_missing_frame_is_an_error(self):
        d = self.copy()
        tl = pd.read_csv(d / "timeline.csv")
        tl.drop(index=3).to_csv(d / "timeline.csv", index=False)
        self.assertTrue(any("frame indexes" in e or "rows but" in e for e in self.errors_of(d)))

    def test_invalid_track_reference_is_an_error(self):
        d = self.copy()
        m = pd.read_csv(d / "hands" / "motion_metrics.csv")
        m.loc[0, "track_id"] = 999
        m.to_csv(d / "hands" / "motion_metrics.csv", index=False)
        self.assertTrue(any("without a created event" in e or "without a linked track" in e
                            for e in self.errors_of(d)))

    def test_invalid_gesture_event_range_is_an_error(self):
        d = self.copy()
        ev = pd.read_csv(d / "hands" / "gesture_events.csv")
        ev.loc[0, "end_frame"] = ev.loc[0, "start_frame"] - 1
        ev.to_csv(d / "hands" / "gesture_events.csv", index=False)
        self.assertTrue(any("ends before it starts" in e or "start/end" in e for e in self.errors_of(d)))

    def test_gesture_observation_with_unknown_event_is_an_error(self):
        d = self.copy()
        obs = pd.read_csv(d / "hands" / "gesture_observations.csv")
        obs.loc[0, "event_id"] = "E9999"
        obs.to_csv(d / "hands" / "gesture_observations.csv", index=False)
        self.assertTrue(any("E9999" in e for e in self.errors_of(d)))

    def test_inconsistent_status_is_an_error(self):
        d = self.copy()
        st = pd.read_csv(d / "hands" / "hand_status.csv")
        not_det = st.index[st["left_status"] == "not_detected"][0]
        st.loc[not_det, "left_status"] = "valid"
        st.to_csv(d / "hands" / "hand_status.csv", index=False)
        self.assertTrue(any("left hand" in e or "left" in e or "status" in e for e in self.errors_of(d)))

    def test_missing_model_metadata_is_an_error(self):
        d = self.copy()
        _rewrite_json(d / "metadata.json", lambda m: m["models"].pop("hand"))
        self.assertTrue(any("no hand model recorded" in e for e in self.errors_of(d)))

    def test_model_hash_that_differs_from_the_registry_is_an_error(self):
        d = self.copy()
        _rewrite_json(d / "metadata.json", lambda m: m["models"]["hand"].update(sha256="0" * 64))
        self.assertTrue(any("does not match model_registry" in e for e in self.errors_of(d)))

    def test_quality_counts_must_agree_with_the_tables(self):
        d = self.copy()
        _rewrite_json(d / "quality" / "tracking_quality.json",
                      lambda q: q["hands"]["tracking"].update(tracks_created=42))
        self.assertTrue(any("tracks_created disagrees" in e for e in self.errors_of(d)))

    def test_centroid_that_disagrees_with_landmarks_is_an_error(self):
        d = self.copy()
        m = pd.read_csv(d / "hands" / "motion_metrics.csv")
        m.loc[0, "centroid_x_px"] = m.loc[0, "centroid_x_px"] + 5.0
        m.to_csv(d / "hands" / "motion_metrics.csv", index=False)
        self.assertTrue(any("centroid differs" in e for e in self.errors_of(d)))


class AnnotationTests(unittest.TestCase):
    def _frame(self, **kw):
        base = {"annotation_id": "A1", "person": "p", "session": "session-001", "track_id": np.nan,
                "hand": "Right", "frame_start": 0, "frame_end": 2, "timestamp_start_ms": np.nan,
                "timestamp_end_ms": np.nan, "gesture_label": "pointing", "visibility": "full",
                "certainty": "certain", "annotator": "AB", "annotation_version": "1.0", "notes": ""}
        base.update(kw)
        return pd.DataFrame([base])

    def test_valid_annotation_has_no_issues(self):
        self.assertEqual(ha.validate_annotations(self._frame()), [])

    def test_invalid_label_is_reported(self):
        self.assertTrue(any("gesture_label" in e for e in ha.validate_annotations(self._frame(gesture_label="wave"))))

    def test_reversed_range_is_reported(self):
        self.assertTrue(any("range" in e for e in ha.validate_annotations(self._frame(frame_start=5, frame_end=2))))

    def test_missing_annotator_and_version_are_reported(self):
        issues = ha.validate_annotations(self._frame(annotator="", annotation_version=np.nan))
        self.assertTrue(any("annotator" in e for e in issues))
        self.assertTrue(any("annotation_version" in e for e in issues))

    def test_blank_track_needs_a_hand(self):
        self.assertTrue(any("hand must be" in e for e in ha.validate_annotations(self._frame(hand=np.nan))))

    def test_overlapping_different_labels_conflict(self):
        a = self._frame(annotation_id="A1", frame_start=0, frame_end=4, gesture_label="pointing")
        b = self._frame(annotation_id="A2", frame_start=3, frame_end=6, gesture_label="peace")
        self.assertTrue(any("overlap" in e for e in ha.validate_annotations(pd.concat([a, b]))))

    def test_frame_labels_expand_intervals_and_keep_occluded(self):
        df = self._frame(frame_start=1, frame_end=3, gesture_label="occluded")
        recs = ha.frame_labels(df)
        self.assertEqual([r["frame"] for r in recs], [1, 2, 3])
        self.assertTrue(all(r["label"] == "occluded" for r in recs))

    def test_timeline_check_is_applied_when_given(self):
        df = self._frame(timestamp_start_ms=999)
        issues = ha.validate_annotations(df, timelines={("p", "session-001"): {0: 0, 1: 100, 2: 200}})
        self.assertTrue(any("timestamp_start_ms" in e for e in issues))


def _predictions(rows):
    recs = []
    for i, (frame_class, quality) in enumerate(rows):
        recs.append({"person": "p", "session": "s1", "track_id": 1, "frame": i, "label": "Right",
                     "frame_class": frame_class, "quality": quality, "timestamp_ms": i * 100})
    return pd.DataFrame(recs)


def _labels(pairs, certainty="certain"):
    out = []
    for i, label in enumerate(pairs):
        out.append({"person": "p", "session": "s1", "track_id": 1, "hand": None, "frame": i, "label": label,
                    "visibility": "full", "certainty": certainty, "annotator": "AB",
                    "annotation_version": "1.0", "annotation_id": f"A{i}"})
    return out


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_give_perfect_scores(self):
        preds = _predictions([("pointing", "ok")] * 3 + [("peace", "ok")] * 3)
        res = he.evaluate(preds, _labels(["pointing"] * 3 + ["peace"] * 3))
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["per_class"]["pointing"]["f1"], 1.0)
        self.assertEqual(res["macro_f1"], 1.0)

    def test_precision_recall_and_confusion_are_computed(self):
        preds = _predictions([("pointing", "ok"), ("pointing", "ok"), ("peace", "ok"), ("unknown", "ok")])
        res = he.evaluate(preds, _labels(["pointing", "peace", "peace", "pointing"]))
        pc = res["per_class"]
        self.assertAlmostEqual(pc["pointing"]["precision"], 0.5)  # frame 1 is peace, predicted pointing
        self.assertAlmostEqual(pc["pointing"]["recall"], 0.5)
        self.assertAlmostEqual(pc["peace"]["recall"], 0.5)
        self.assertAlmostEqual(pc["peace"]["precision"], 1.0)
        self.assertAlmostEqual(res["unknown_rate"], 0.25)
        conf = res["confusion_matrix"]["rows_truth"]
        self.assertEqual(conf[he.SCORED.index("pointing")][he.SCORED.index("unknown")], 1)

    def test_zero_denominators_are_none_not_zero(self):
        preds = _predictions([("pointing", "ok")] * 2)
        res = he.evaluate(preds, _labels(["pointing"] * 2))
        self.assertIsNone(res["per_class"]["peace"]["precision"])
        self.assertIsNone(res["per_class"]["peace"]["recall"])

    def test_occluded_labels_are_counted_not_scored(self):
        preds = _predictions([("pointing", "ok"), ("unknown", "partial_view")])
        res = he.evaluate(preds, _labels(["pointing", "occluded"]))
        self.assertEqual(res["occluded_labels"], 1)
        self.assertEqual(res["scored"], 1)

    def test_partial_and_full_visibility_are_reported_separately(self):
        preds = _predictions([("pointing", "ok"), ("unknown", "partial_view"), ("unknown", "partial_view")])
        res = he.evaluate(preds, _labels(["pointing", "pointing", "pointing"]))
        self.assertEqual(res["by_visibility"]["fully_visible"]["accuracy"], 1.0)
        self.assertEqual(res["by_visibility"]["partially_visible"]["unknown_rate"], 1.0)

    def test_uncertain_labels_are_left_out_at_the_default_threshold(self):
        preds = _predictions([("pointing", "ok")])
        res = he.evaluate(preds, _labels(["pointing"], certainty="uncertain"))
        self.assertEqual(res["scored"], 0)
        res_all = he.evaluate(preds, _labels(["pointing"], certainty="uncertain"), min_certainty="uncertain")
        self.assertEqual(res_all["scored"], 1)

    def test_unmatched_labels_are_counted(self):
        preds = _predictions([("pointing", "ok")])
        res = he.evaluate(preds, _labels(["pointing", "pointing"]))
        self.assertEqual(res["unmatched_labels"], 1)


class SplitTests(unittest.TestCase):
    def test_overlapping_calibration_and_evaluation_sets_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({"calibration": [["p", "s1"]], "evaluation": [["p", "s1"]]}))
            with self.assertRaises(he.SplitError):
                he.load_split(path)

    def test_a_split_needs_both_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps({"calibration": [["p", "s1"]], "evaluation": []}))
            with self.assertRaises(he.SplitError):
                he.load_split(path)

    def test_threshold_change_needs_every_field_and_disjoint_sets(self):
        issues = he.threshold_change_record({"threshold": "REACH_CURLED", "original": 1.2})
        self.assertTrue(any("missing field" in e for e in issues))
        full = {f: 0 for f in he.REQUIRED_CHANGE_FIELDS}
        full["calibration_set"] = [["p", "s1"]]
        full["evaluation_set"] = [["p", "s1"]]
        self.assertTrue(any("overlap" in e for e in he.threshold_change_record(full)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
