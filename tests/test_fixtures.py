"""Tests on temporary synthetic datasets. No recordings or local datasets are needed, so these run in CI.

Every test works inside a temporary project folder, so real data is never read or written.
"""

import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "src" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

import analyze_faces  # noqa: E402
import facial_analyzer as fa  # noqa: E402
import project_paths as pp  # noqa: E402
from file_safety import OutputExistsError, atomic_write_csv, check_writable  # noqa: E402
from validation import N_LANDMARKS, validate_frame, validate_frames, validate_timestamps  # noqa: E402

PERSON = "demo_person"
SESSION = "session-001"
FPS = 30
N_FRAMES = 12
NO_FACE_FRAME = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_frames(n: int = N_FRAMES, fps: int = FPS) -> list[dict]:
    """Deterministic frames: a ring of 478 landmarks, sinusoidal jaw and blink, one frame without a face."""
    frames = []
    for i in range(n):
        detected = i != NO_FACE_FRAME
        t = i / fps
        frame = {
            "frame": i,
            "timestamp_ms": round(i * 1000 / fps),
            "face_detected": detected,
        }
        if detected:
            angle = np.linspace(0, 2 * np.pi, N_LANDMARKS, endpoint=False)
            frame["landmarks"] = [
                {
                    "id": k,
                    "x": float(0.5 + 0.2 * math.cos(a)),
                    "y": float(0.5 + 0.2 * math.sin(a) + 0.01 * math.sin(2 * math.pi * t)),
                    "z": float(0.01 * math.cos(a)),
                }
                for k, a in enumerate(angle)
            ]
            blink = 0.9 if i in (6, 7) else 0.05
            frame["blendshapes"] = {
                "jawOpen": 0.1 + 0.05 * math.sin(2 * math.pi * t),
                "mouthPucker": 0.02,
                "browInnerUp": 0.03,
                "eyeBlinkLeft": blink,
                "eyeBlinkRight": blink,
            }
            frame["transformation_matrix"] = np.eye(4).tolist()
        frames.append(frame)
    return frames


class FixtureProjectTestCase(unittest.TestCase):
    """Creates a temporary recordings/ and datasets/ tree and points project_paths at it."""

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="facial_fixture_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self._tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self._tmp / "datasets"
        self.video = pp.session_video(PERSON, SESSION)
        self.video.parent.mkdir(parents=True, exist_ok=True)
        self.video.write_bytes(b"synthetic video placeholder, not a real recording")
        self.dataset_dir = pp.session_dataset_dir(PERSON, SESSION)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.dataset_dir / "facial_data.json"
        self.frames = synthetic_frames()
        self.json_path.write_text(json.dumps(self.frames), encoding="utf-8")

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)


class JsonLoadingTests(FixtureProjectTestCase):
    def test_loads_expected_shapes(self):
        ds = fa.load_session(fa.make_session(PERSON, SESSION))
        self.assertEqual(ds.landmarks.shape, (N_FRAMES, N_LANDMARKS, 3))
        self.assertEqual(ds.matrices.shape, (N_FRAMES, 4, 4))
        self.assertEqual(len(ds.timestamps_s), N_FRAMES)

    def test_detection_flags_and_missing_face(self):
        ds = fa.load_session(fa.make_session(PERSON, SESSION))
        self.assertFalse(ds.detected[NO_FACE_FRAME])
        self.assertEqual(int(ds.detected.sum()), N_FRAMES - 1)
        self.assertTrue(np.isnan(ds.landmarks[NO_FACE_FRAME]).all())
        self.assertTrue(np.isfinite(ds.landmarks[0]).all())

    def test_blendshapes_loaded_as_numeric_columns(self):
        ds = fa.load_session(fa.make_session(PERSON, SESSION))
        self.assertIn("jawOpen", ds.blendshapes.columns)
        self.assertEqual(ds.blendshapes["jawOpen"].dtype.kind, "f")


class CsvGenerationTests(FixtureProjectTestCase):
    def test_blendshape_csv_columns_and_rows(self):
        ds = fa.load_session(fa.make_session(PERSON, SESSION))
        table = analyze_faces.build_blendshape_table(ds)
        self.assertEqual(list(table.columns[:3]), ["frame", "timestamp_ms", "face_detected"])
        self.assertIn("eyeBlinkLeft", table.columns)
        self.assertEqual(len(table), N_FRAMES)

    def test_csv_round_trip_keeps_values(self):
        ds = fa.load_session(fa.make_session(PERSON, SESSION))
        table = analyze_faces.build_blendshape_table(ds)
        out = self.dataset_dir / "facial_blendshapes.csv"
        atomic_write_csv(table, out)
        back = pd.read_csv(out)
        self.assertEqual(len(back), N_FRAMES)
        np.testing.assert_allclose(back["jawOpen"], table["jawOpen"], atol=1e-12)
        self.assertEqual(back["face_detected"].tolist(), [bool(v) for v in table["face_detected"]])


class LandmarkValidationTests(unittest.TestCase):
    def test_synthetic_frames_are_valid(self):
        self.assertEqual(validate_frames(synthetic_frames()), [])

    def test_wrong_landmark_count_is_reported(self):
        frame = synthetic_frames()[0]
        frame["landmarks"] = frame["landmarks"][:-1]
        self.assertTrue(any("expected 478" in msg for msg in validate_frame(frame)))

    def test_non_finite_coordinate_is_reported(self):
        frame = synthetic_frames()[0]
        frame["landmarks"][10]["x"] = float("nan")
        self.assertTrue(any("non-finite" in msg for msg in validate_frame(frame)))

    def test_point_outside_frame_is_reported(self):
        frame = synthetic_frames()[0]
        frame["landmarks"][5]["y"] = 9.0
        self.assertTrue(any("outside the frame" in msg for msg in validate_frame(frame)))

    def test_landmarks_without_face_are_reported(self):
        frame = synthetic_frames()[NO_FACE_FRAME]
        frame["landmarks"] = [{"id": 0, "x": 0.5, "y": 0.5, "z": 0.0}]
        self.assertTrue(any("no face was detected" in msg for msg in validate_frame(frame)))

    def test_bad_matrix_is_reported(self):
        frame = synthetic_frames()[0]
        frame["transformation_matrix"] = [[1, 0], [0, 1]]
        self.assertTrue(any("4x4" in msg for msg in validate_frame(frame)))

    def test_non_sequential_frame_numbers_are_reported(self):
        frames = synthetic_frames()
        frames[3]["frame"] = 99
        self.assertTrue(validate_frames(frames))


class TimestampTests(unittest.TestCase):
    def test_extraction_style_timestamps_pass(self):
        timestamps = [round(i * 1000 / FPS) for i in range(N_FRAMES)]
        self.assertEqual(validate_timestamps(timestamps, FPS), [])

    def test_non_increasing_timestamps_fail(self):
        self.assertTrue(validate_timestamps([0, 33, 33, 100], 30))

    def test_wrong_step_fails(self):
        self.assertTrue(validate_timestamps([0, 33, 66, 200], 30))

    def test_negative_start_fails(self):
        self.assertTrue(validate_timestamps([-5, 28, 61], 30))

    def test_synthetic_frame_timestamps_are_consistent(self):
        timestamps = [f["timestamp_ms"] for f in synthetic_frames()]
        self.assertEqual(validate_timestamps(timestamps, FPS), [])


class DatasetLoadingTests(FixtureProjectTestCase):
    def test_processed_session_is_discovered(self):
        self.assertEqual(pp.list_persons(), [PERSON])
        self.assertEqual(pp.list_processed_sessions(PERSON), [SESSION])
        sessions = fa.processed_sessions(PERSON)
        self.assertEqual([s.session_id for s in sessions], [SESSION])
        self.assertEqual(sessions[0].dataset_dir, self.dataset_dir)

    def test_unprocessed_video_is_not_listed_as_processed(self):
        other = pp.session_video(PERSON, "session-002")
        other.write_bytes(b"x")
        self.assertIn("session-002", pp.list_sessions(PERSON))
        self.assertNotIn("session-002", pp.list_processed_sessions(PERSON))

    def test_person_metadata_describes_sessions(self):
        path = fa.write_person_metadata(PERSON)
        meta = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(meta["person"], PERSON)
        self.assertEqual(meta["sessions"][0]["session_id"], SESSION)
        self.assertEqual(meta["sessions"][0]["frames"], N_FRAMES)


class OverwriteProtectionTests(FixtureProjectTestCase):
    def test_check_writable_refuses_existing_files(self):
        target = self.dataset_dir / "x.csv"
        target.write_text("a")
        with self.assertRaises(OutputExistsError):
            check_writable([target])

    def test_check_writable_allows_force(self):
        target = self.dataset_dir / "x.csv"
        target.write_text("a")
        check_writable([target], force=True)  # no exception

    def test_analysis_outputs_refuse_second_run_without_force(self):
        session = fa.make_session(PERSON, SESSION)
        result = fa.analyse_session(session)
        fa.save_session_outputs(result, session)
        with self.assertRaises(OutputExistsError):
            fa.save_session_outputs(result, session)

    def test_analysis_outputs_regenerate_with_force(self):
        session = fa.make_session(PERSON, SESSION)
        result = fa.analyse_session(session)
        fa.save_session_outputs(result, session)
        fa.save_session_outputs(result, session, force=True)
        self.assertTrue((self.dataset_dir / "motion_metrics.csv").is_file())

    def test_atomic_write_leaves_no_temporary_files(self):
        fa.save_session_outputs(fa.analyse_session(fa.make_session(PERSON, SESSION)),
                                fa.make_session(PERSON, SESSION))
        leftovers = [p for p in self.dataset_dir.iterdir() if p.suffix == ".tmp"]
        self.assertEqual(leftovers, [])


class DatasetImmutabilityTests(FixtureProjectTestCase):
    def test_analysis_never_changes_extraction_json_or_recording(self):
        json_before = sha256(self.json_path)
        video_before = sha256(self.video)
        session = fa.make_session(PERSON, SESSION)
        result = fa.analyse_session(session)
        fa.save_session_outputs(result, session)
        atomic_write_csv(analyze_faces.build_blendshape_table(fa.load_session(session)),
                         self.dataset_dir / "facial_blendshapes.csv")
        self.assertEqual(sha256(self.json_path), json_before)
        self.assertEqual(sha256(self.video), video_before)


class MotionOnFixtureTests(FixtureProjectTestCase):
    def test_session_metrics_have_one_row_per_frame_with_session_id(self):
        session = fa.make_session(PERSON, SESSION)
        result = fa.analyse_session(session)
        self.assertEqual(len(result.table), N_FRAMES)
        self.assertTrue((result.table["session_id"] == SESSION).all())
        self.assertFalse(result.table["face_detected"].iloc[NO_FACE_FRAME])

    def test_blink_detected_in_synthetic_blink(self):
        result = fa.analyse_session(fa.make_session(PERSON, SESSION))
        self.assertEqual(len(result.blinks), 1)
        self.assertEqual(int(result.blinks.iloc[0]["start_frame"]), 6)
        self.assertEqual(int(result.blinks.iloc[0]["end_frame"]), 7)

    def test_tracking_quality_matches_known_detection_rate(self):
        result = fa.analyse_session(fa.make_session(PERSON, SESSION))
        self.assertAlmostEqual(result.quality["detection_rate"], (N_FRAMES - 1) / N_FRAMES, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
