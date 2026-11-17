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
        self.assertTrue(pp.is_valid_person("sameer_gupta"))
        self.assertTrue(pp.is_valid_person("my-project-2"))
        self.assertFalse(pp.is_valid_person("Sameer Gupta"))
        self.assertFalse(pp.is_valid_person("../etc"))
        self.assertFalse(pp.is_valid_person(""))

    def test_suggest_person(self):
        self.assertEqual(pp.suggest_person("My Project 2"), "my-project-2")
        self.assertEqual(pp.suggest_person("  Sameer_Gupta!! "), "sameer_gupta")

    def test_session_paths_follow_layout(self):
        video = pp.session_video("sameer_gupta", "session-001")
        dataset = pp.session_dataset_dir("sameer_gupta", "session-001")
        self.assertEqual(video.parts[-3:], ("raw", "sameer_gupta", "session-001.mp4"))
        self.assertEqual(dataset.parts[-3:], ("datasets", "sameer_gupta", "session-001"))


class SyntheticMotionTests(unittest.TestCase):
    def test_identity_rotation_only_centres_the_face(self):
        pts = np.random.default_rng(0).random((1, 478, 3))
        out = compensate_head_pose(pts, np.tile(np.eye(4), (1, 1, 1)))
        np.testing.assert_allclose(out[0], pts[0] - pts[0].mean(axis=0), atol=1e-12)

    def test_head_rotation_is_removed(self):
        rng = np.random.default_rng(1)
        base = rng.random((478, 3))
        R = rotation_z(0.4)
        rotated = base @ R.T  # what the camera sees after the head turns
        matrix = np.eye(4)
        matrix[:3, :3] = R
        out = compensate_head_pose(rotated[None], matrix[None])
        np.testing.assert_allclose(out[0], base - base.mean(axis=0), atol=1e-10)

    def test_missing_matrix_gives_nan(self):
        pts = np.random.default_rng(2).random((2, 478, 3))
        out = compensate_head_pose(pts, np.full((2, 4, 4), np.nan))
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


class DashboardRunTests(unittest.TestCase):
    """Runs the main app and every page headlessly and fails on any uncaught exception."""

    def _run(self, path: Path):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(path), default_timeout=180)
        at.run()
        self.assertFalse(at.exception, f"{path.name} raised: {[e.value for e in at.exception]}")
        return at

    def test_main_app_runs(self):
        self._run(DASHBOARD_DIR / "app.py")

    def test_all_pages_run(self):
        pages = sorted((DASHBOARD_DIR / "pages").glob("*.py"))
        self.assertEqual(len(pages), 5)
        for page in pages:
            with self.subTest(page=page.name):
                self._run(page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
