"""Tests for the version 2 schema (synthetic data only)."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import pose_schema as s2  # noqa: E402

T = 6
GOOD_SHA = hashlib.sha256(b"model").hexdigest()


def good_arrays(n=T):
    arrays = {}
    for spec in s2.ARRAY_SPECS:
        arrays[spec.relative_path] = np.full((n,) + spec.tail_shape, np.nan, dtype=np.float32)
    return arrays


def good_metadata():
    return {
        "schema_version": s2.SCHEMA_VERSION,
        "identity": {"person_id": "demo_person", "session_id": "session-001"},
        "source": {"video_filename": "session-001.mp4", "fps": 30.0, "frame_count": T},
        "timeline": {"reference": "start of video", "timestamp_unit": "milliseconds",
                     "timestamp_source": "fps_derived", "frame_index_base": 0},
        "extraction": {"software": "facial-analysis", "mediapipe_version": "1.2.3"},
        "models": {"face_landmarker": {"file": "face_landmarker.task", "sha256": GOOD_SHA,
                                       "licence_status": "to verify"}},
        "configuration": {"pose": {"num_poses": 1}, "hands": {"num_hands": 2}},
        "coordinate_conventions": {"image_landmarks": "normalised"},
        "streams": {"facial": {"status": "complete"}, "body": {"status": "absent"},
                    "hands": {"status": "partial"}},
        "outputs": {"files": []},
    }


class BlendshapeNamesTests(unittest.TestCase):
    def test_there_are_52_named_categories(self):
        self.assertEqual(len(s2.BLENDSHAPE_NAMES), s2.N_BLENDSHAPES)
        self.assertEqual(len(set(s2.BLENDSHAPE_NAMES)), s2.N_BLENDSHAPES)

    def test_names_include_jaw_and_blink(self):
        self.assertIn("jawOpen", s2.BLENDSHAPE_NAMES)
        self.assertIn("eyeBlinkLeft", s2.BLENDSHAPE_NAMES)


class TimestampTests(unittest.TestCase):
    def test_fps_derived_matches_face_data_rule(self):
        ts, source = s2.frame_timestamps_ms(10, fps=30.0)
        self.assertEqual(source, "fps_derived")
        np.testing.assert_array_equal(ts, [round(i * 1000 / 30) for i in range(10)])
        self.assertEqual(ts.dtype, np.int64)

    def test_decoded_pts_are_preferred_when_valid(self):
        pts = [0.0, 0.0334, 0.0668, 0.1002]
        ts, source = s2.frame_timestamps_ms(4, fps=30.0, pts_seconds=pts)
        self.assertEqual(source, "decoded_pts")
        np.testing.assert_array_equal(ts, [0, 33, 67, 100])

    def test_pts_are_ignored_when_they_do_not_increase(self):
        ts, source = s2.frame_timestamps_ms(3, fps=30.0, pts_seconds=[0.0, 0.0, 0.1])
        self.assertEqual(source, "fps_derived")

    def test_pts_are_ignored_when_count_differs(self):
        _, source = s2.frame_timestamps_ms(3, fps=30.0, pts_seconds=[0.0, 0.1])
        self.assertEqual(source, "fps_derived")

    def test_missing_fps_and_pts_is_an_error(self):
        with self.assertRaises(ValueError):
            s2.frame_timestamps_ms(3)

    def test_invalid_fps_is_an_error(self):
        with self.assertRaises(ValueError):
            s2.frame_timestamps_ms(3, fps=0)


class ArrayValidationTests(unittest.TestCase):
    def test_correct_arrays_pass(self):
        self.assertEqual(s2.validate_arrays(good_arrays(), T), [])

    def test_wrong_shape_is_reported(self):
        arrays = good_arrays()
        arrays["body/pose_landmarks.npy"] = np.zeros((T, 32, 5), dtype=np.float32)
        issues = s2.validate_arrays(arrays, T)
        self.assertTrue(any("body/pose_landmarks.npy" in m and "shape" in m for m in issues))

    def test_wrong_dtype_is_reported(self):
        arrays = good_arrays()
        arrays["facial/landmarks.npy"] = arrays["facial/landmarks.npy"].astype(np.float64)
        self.assertTrue(any("dtype" in m for m in s2.validate_arrays(arrays, T)))

    def test_missing_required_array_is_reported(self):
        arrays = good_arrays()
        del arrays["hands/left_hand.npy"]
        self.assertTrue(any("missing required array hands/left_hand.npy" in m for m in s2.validate_arrays(arrays, T)))

    def test_frame_count_must_match(self):
        arrays = good_arrays(T)
        self.assertTrue(s2.validate_arrays(arrays, T + 1))

    def test_hand_arrays_have_21_landmarks_and_xyz(self):
        for name in ("hands/left_hand.npy", "hands/right_hand.npy"):
            spec = next(sp for sp in s2.ARRAY_SPECS if sp.relative_path == name)
            self.assertEqual(spec.tail_shape, (21, 3))

    def test_pose_arrays_have_33_landmarks_and_five_channels(self):
        spec = next(sp for sp in s2.ARRAY_SPECS if sp.relative_path == "body/pose_landmarks.npy")
        self.assertEqual(spec.tail_shape, (33, 5))

    def test_optional_world_hand_arrays_are_checked_only_when_present(self):
        arrays = good_arrays()
        self.assertEqual(s2.validate_arrays(arrays, T), [])
        arrays["hands/left_hand_world.npy"] = np.zeros((T, 20, 3), dtype=np.float32)
        self.assertTrue(s2.validate_arrays(arrays, T))

    def test_missing_detection_is_nan_not_zero(self):
        arrays = good_arrays()
        arrays["body/pose_landmarks.npy"][2] = np.nan
        self.assertTrue(np.isnan(arrays["body/pose_landmarks.npy"][2]).all())


class TimelineTests(unittest.TestCase):
    def _rows(self, ts):
        return [{"frame": i, "timestamp_ms": t} for i, t in enumerate(ts)]

    def test_valid_timeline_passes(self):
        self.assertEqual(s2.validate_timeline_rows(self._rows([0, 33, 67, 100, 133, 167]), 6), [])

    def test_row_count_must_match(self):
        self.assertTrue(s2.validate_timeline_rows(self._rows([0, 33]), 6))

    def test_frame_numbers_must_be_sequential(self):
        rows = self._rows([0, 33, 67])
        rows[1]["frame"] = 5
        self.assertTrue(s2.validate_timeline_rows(rows, 3))

    def test_timestamps_must_increase(self):
        self.assertTrue(s2.validate_timeline_rows(self._rows([0, 33, 33]), 3))


class HandednessTests(unittest.TestCase):
    def test_detected_hand_needs_a_score_in_range(self):
        rows = [{"left_detected": True, "left_score": 0.9, "right_detected": False, "right_score": None}]
        self.assertEqual(s2.validate_handedness_rows(rows), [])
        rows = [{"left_detected": True, "left_score": 1.7, "right_detected": False, "right_score": None}]
        self.assertTrue(s2.validate_handedness_rows(rows))

    def test_undetected_hand_must_not_have_a_score(self):
        rows = [{"left_detected": False, "left_score": 0.4, "right_detected": False, "right_score": None}]
        self.assertTrue(s2.validate_handedness_rows(rows))

    def test_undetected_hand_with_nan_score_is_fine(self):
        rows = [{"left_detected": False, "left_score": float("nan"), "right_detected": False, "right_score": None}]
        self.assertEqual(s2.validate_handedness_rows(rows), [])


class MetadataTests(unittest.TestCase):
    def test_complete_metadata_passes(self):
        self.assertEqual(s2.validate_metadata(good_metadata()), [])

    def test_each_required_key_is_checked(self):
        for key in s2.REQUIRED_METADATA_KEYS:
            with self.subTest(key=key):
                meta = good_metadata()
                del meta[key]
                self.assertTrue(any(key in m for m in s2.validate_metadata(meta)))

    def test_timestamp_unit_must_be_milliseconds(self):
        meta = good_metadata()
        meta["timeline"]["timestamp_unit"] = "seconds"
        self.assertTrue(s2.validate_metadata(meta))

    def test_timestamp_source_must_be_known(self):
        meta = good_metadata()
        meta["timeline"]["timestamp_source"] = "guess"
        self.assertTrue(s2.validate_metadata(meta))

    def test_model_needs_a_sha256(self):
        meta = good_metadata()
        meta["models"]["face_landmarker"]["sha256"] = "abc"
        self.assertTrue(s2.validate_metadata(meta))

    def test_model_licence_status_is_required(self):
        meta = good_metadata()
        meta["models"]["face_landmarker"]["licence_status"] = "assumed"
        self.assertTrue(s2.validate_metadata(meta))

    def test_pose_must_be_single_person(self):
        meta = good_metadata()
        meta["configuration"]["pose"]["num_poses"] = 2
        self.assertTrue(s2.validate_metadata(meta))

    def test_stream_status_must_be_known(self):
        meta = good_metadata()
        meta["streams"]["body"]["status"] = "maybe"
        self.assertTrue(s2.validate_metadata(meta))

    def test_schema_major_version_must_be_2(self):
        meta = good_metadata()
        meta["schema_version"] = "1.0"
        self.assertTrue(s2.validate_metadata(meta))


class ManifestTests(unittest.TestCase):
    def test_safe_manifest_passes(self):
        manifest = {"files": [{"path": "facial/landmarks.npy", "sha256": GOOD_SHA}]}
        self.assertEqual(s2.validate_manifest(manifest), [])

    def test_absolute_and_parent_paths_are_refused(self):
        for bad in ("/etc/passwd", "..\\facial_data.json", "facial/../../x"):
            with self.subTest(path=bad):
                manifest = {"files": [{"path": bad, "sha256": GOOD_SHA}]}
                self.assertTrue(s2.validate_manifest(manifest))

    def test_duplicate_entries_are_refused(self):
        manifest = {"files": [{"path": "a.npy", "sha256": GOOD_SHA}, {"path": "a.npy", "sha256": GOOD_SHA}]}
        self.assertTrue(s2.validate_manifest(manifest))


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pose_schema_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_face_data_session(self):
        (self.tmp / "facial_data.json").write_text("[]")
        self.assertEqual(s2.session_layout(self.tmp), "v1")

    def test_pose_data_session_needs_metadata(self):
        (self.tmp / "metadata.json").write_text(json.dumps(good_metadata()))
        self.assertEqual(s2.session_layout(self.tmp), "v2")

    def test_pose_folders_without_metadata_are_incomplete(self):
        (self.tmp / "facial").mkdir()
        self.assertEqual(s2.session_layout(self.tmp), "incomplete_v2")

    def test_unreadable_metadata_is_incomplete(self):
        (self.tmp / "metadata.json").write_text("{not json")
        self.assertEqual(s2.session_layout(self.tmp), "incomplete_v2")

    def test_empty_folder(self):
        self.assertEqual(s2.session_layout(self.tmp), "none")

    def test_face_and_pose_can_coexist(self):
        (self.tmp / "facial_data.json").write_text("[]")
        (self.tmp / "metadata.json").write_text(json.dumps(good_metadata()))
        self.assertEqual(s2.session_layout(self.tmp), "v2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
