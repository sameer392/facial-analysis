"""Regression tests for the facial analysis project.

Run from the project root with the project's virtual environment:
    python -m unittest tests.test_regression -v

Dataset tests run only for sessions that have been extracted (datasets/<person>/<session>/).
They are skipped, not failed, when nothing has been processed yet.
"""

import py_compile
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "src" / "analysis"
DASHBOARD_DIR = ROOT / "src" / "dashboard"
sys.path.insert(0, str(ANALYSIS_DIR))

import facial_analyzer as fa  # noqa: E402
import project_paths as pp  # noqa: E402
from facial_3d_analyzer import head_pose_angles  # noqa: E402
from motion_analyzer import (  # noqa: E402
    compensate_head_pose,
    landmark_kinematics,
    neutral_frame_mask,
)


def rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class NamingTests(unittest.TestCase):
    def test_session_names(self):
        self.assertTrue(pp.is_valid_session("session-001"))
        self.assertTrue(pp.is_valid_session("session-042"))
        self.assertFalse(pp.is_valid_session("presenter-01"))
        self.assertFalse(pp.is_valid_session("session-1"))
        self.assertFalse(pp.is_valid_session("../session-001"))

    def test_person_names(self):
        self.assertTrue(pp.is_valid_person("demo_person"))
        self.assertTrue(pp.is_valid_person("my-project-2"))
        self.assertFalse(pp.is_valid_person("Demo Person"))
        self.assertFalse(pp.is_valid_person("../etc"))
        self.assertFalse(pp.is_valid_person(""))

    def test_suggest_person(self):
        self.assertEqual(pp.suggest_person("My Project 2"), "my-project-2")
        self.assertEqual(pp.suggest_person("  demo_person!! "), "demo_person")

    def test_session_paths_follow_layout(self):
        video = pp.session_video("demo_person", "session-001")
        dataset = pp.session_dataset_dir("demo_person", "session-001")
        self.assertEqual(video.parts[-3:], ("raw", "demo_person", "session-001.mp4"))
        self.assertEqual(dataset.parts[-3:], ("datasets", "demo_person", "session-001"))


class DeletionTests(unittest.TestCase):
    """Deletion runs in a temporary folder, never in the real project data."""

    def setUp(self):
        import shutil
        import tempfile

        self._tmp = Path(tempfile.mkdtemp())
        self._shutil = shutil
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self._tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self._tmp / "datasets"
        for person in ("alice", "bob"):
            (pp.RECORDINGS_RAW / person).mkdir(parents=True)
            (pp.DATASETS_ROOT / person / "session-001").mkdir(parents=True)
            (pp.DATASETS_ROOT / person / "session-001" / "facial_data.json").write_text("[]")
            (pp.RECORDINGS_RAW / person / "session-001.mp4").write_bytes(b"x")

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        self._shutil.rmtree(self._tmp, ignore_errors=True)

    def test_delete_dataset_keeps_video(self):
        self.assertTrue(pp.delete_session_dataset("alice", "session-001"))
        self.assertFalse((pp.DATASETS_ROOT / "alice" / "session-001").exists())
        self.assertTrue((pp.RECORDINGS_RAW / "alice" / "session-001.mp4").exists())

    def test_delete_video_keeps_other_person(self):
        self.assertTrue(pp.delete_session_video("alice", "session-001"))
        self.assertFalse((pp.RECORDINGS_RAW / "alice" / "session-001.mp4").exists())
        self.assertTrue((pp.RECORDINGS_RAW / "bob" / "session-001.mp4").exists())

    def test_delete_person_removes_raw_and_datasets_only_for_that_person(self):
        self.assertTrue(pp.delete_person("alice"))
        self.assertFalse((pp.RECORDINGS_RAW / "alice").exists())
        self.assertFalse((pp.DATASETS_ROOT / "alice").exists())
        self.assertTrue((pp.RECORDINGS_RAW / "bob").exists())
        self.assertTrue((pp.DATASETS_ROOT / "bob").exists())

    def test_invalid_names_are_refused(self):
        for bad in ("..", "../bob", "alice/../bob", ""):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                pp.delete_person(bad)
        with self.assertRaises(ValueError):
            pp.delete_session_video("alice", "../session-001")

    def test_missing_targets_return_false(self):
        self.assertFalse(pp.delete_session_dataset("alice", "session-009"))
        self.assertFalse(pp.delete_person("nobody"))


class SyntheticMotionTests(unittest.TestCase):
    def test_missing_matrix_gives_nan(self):
        pts = np.random.default_rng(2).random((2, 478, 3))
        out = compensate_head_pose(pts, np.full((2, 4, 4), np.nan), (640, 480))
        self.assertTrue(np.isnan(out).all())

    def test_constant_motion_gives_constant_velocity_and_zero_acceleration(self):
        T = 6
        points = np.zeros((T, 4, 3))
        points[:, 0, 0] = np.arange(T) * 0.01  # one of four landmarks moves 0.01 per frame
        usable = np.ones(T, dtype=bool)
        disp, vel, acc = landmark_kinematics(points, np.arange(T) / 30.0, usable)
        np.testing.assert_allclose(disp[1:], 0.01 / 4, atol=1e-12)
        np.testing.assert_allclose(vel[1:], (0.01 / 4) * 30.0, atol=1e-9)
        np.testing.assert_allclose(acc[2:], 0.0, atol=1e-9)
        self.assertTrue(np.isnan(disp[0]))

    def test_motion_is_nan_across_missing_frames(self):
        points = np.zeros((3, 2, 3))
        points[:, 0, 0] = [0.0, 0.1, 0.2]
        usable = np.array([True, False, True])
        disp, _, _ = landmark_kinematics(points, np.arange(3) / 30.0, usable)
        self.assertTrue(np.isnan(disp[1]) and np.isnan(disp[2]))

    def test_neutral_mask_selects_calmest_tracked_frames(self):
        blend = pd.DataFrame({
            "jawOpen": [0.9, 0.0, 0.8, 0.05],
            "mouthPucker": [0.5, 0.0, 0.4, 0.0],
        })
        usable = np.array([True, True, True, False])
        mask = neutral_frame_mask(blend, usable, quantile=0.5)
        self.assertTrue(mask[1])
        self.assertFalse(mask[3])  # untracked frames are never used for calibration
        self.assertFalse(mask[0])

    def test_head_pose_identity_is_zero(self):
        angles = head_pose_angles(np.tile(np.eye(4), (2, 1, 1)))
        np.testing.assert_allclose(angles.to_numpy(), 0.0, atol=1e-9)

    def test_blink_detection_on_synthetic_frames(self):
        session = fa.Session("person-test", "session-001", Path("none.mp4"), Path("."), 30.0)
        blend = pd.DataFrame({
            fa.BLINK_LEFT: [0.0, 0.9, 0.9, 0.0, 0.0, 0.8],
            fa.BLINK_RIGHT: [0.0, 0.8, 0.9, 0.0, 0.0, 0.8],
        })
        ds = fa.FacialDataset(
            session=session,
            timestamps_s=np.arange(6) / 30.0,
            detected=np.ones(6, dtype=bool),
            landmarks=np.zeros((6, 478, 3)),
            matrices=np.tile(np.eye(4), (6, 1, 1)),
            blendshapes=blend,
        )
        blinks = fa.detect_blinks(ds)
        self.assertEqual(len(blinks), 2)
        self.assertEqual(int(blinks.iloc[0]["start_frame"]), 1)
        self.assertEqual(int(blinks.iloc[0]["end_frame"]), 2)
        self.assertEqual(int(blinks.iloc[1]["start_frame"]), 5)
        self.assertTrue((blinks["session_id"] == "session-001").all())


class ProcessedDatasetTests(unittest.TestCase):
    """Checks on every session that has been extracted. Skipped when none exist yet."""

    @classmethod
    def setUpClass(cls):
        cls.sessions = []
        for person in pp.list_persons():
            for sid in pp.list_processed_sessions(person):
                cls.sessions.append(fa.make_session(person, sid))
        if not cls.sessions:
            raise unittest.SkipTest("No processed sessions in datasets/ yet")

    def test_every_processed_session_has_its_video(self):
        for s in self.sessions:
            with self.subTest(session=f"{s.person}/{s.session_id}"):
                self.assertTrue(s.video_path.is_file(), s.video_path)

    def test_csv_blendshapes_match_json(self):
        for s in self.sessions:
            if not s.csv_path.is_file():
                continue
            with self.subTest(session=f"{s.person}/{s.session_id}"):
                ds = fa.load_session(s)
                csv = pd.read_csv(s.csv_path)
                self.assertEqual(len(csv), len(ds.timestamps_s))
                np.testing.assert_allclose(
                    csv["jawOpen"].to_numpy(dtype=float),
                    ds.blendshapes["jawOpen"].to_numpy(dtype=float),
                    atol=1e-9,
                    equal_nan=True,
                )

    def test_metrics_keep_session_id_and_frame_count(self):
        for s in self.sessions:
            with self.subTest(session=f"{s.person}/{s.session_id}"):
                a = fa.analyse_session(s)
                self.assertEqual(len(a.table), len(fa.load_session(s).timestamps_s))
                self.assertTrue((a.table["session_id"] == s.session_id).all())
                if len(a.blinks):
                    self.assertTrue((a.blinks["session_id"] == s.session_id).all())
                self.assertGreater(a.quality["detection_rate"], 0.0)


class ScriptCompileTests(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for path in list(ANALYSIS_DIR.glob("*.py")) + list(DASHBOARD_DIR.rglob("*.py")):
            with self.subTest(file=path.name):
                py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
