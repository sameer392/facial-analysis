"""Tests for the V2 Body Motion viewer support code. Synthetic datasets in temporary folders only."""

import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "dashboard"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import v2_viewer as vv  # noqa: E402
import session_writer_v2 as w  # noqa: E402
import schema_v2 as s2  # noqa: E402
import project_paths as pp  # noqa: E402
import test_session_writer_v2 as tsw  # noqa: E402  (synthetic V2 payload)

N = 8
W, H = 64, 48


def write_v2(session_dir: Path, pose=None, seed=0) -> Path:
    """Write a valid synthetic V2 session through the real writer, with optional pose overrides."""
    payload = tsw.make_payload(n=N, seed=seed)
    if pose is not None:
        payload.arrays["body/pose_landmarks.npy"] = pose
    payload.metadata["source"]["width"] = W
    payload.metadata["source"]["height"] = H
    session_dir.mkdir(parents=True, exist_ok=True)
    w.write_v2_session(session_dir, payload)
    return session_dir


def pose_frame(x=0.5, y=0.5, vis=0.9):
    arr = np.zeros((33, 5), dtype=np.float32)
    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 3] = vis
    arr[:, 4] = 0.9
    return arr


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_viewer_"))
        self.root = self.tmp / "datasets"
        self._orig = pp.DATASETS_ROOT
        pp.DATASETS_ROOT = self.root

    def tearDown(self):
        pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_v2_session_is_discovered(self):
        write_v2(self.root / "demo_person" / "session-001")
        valid, rejected = vv.discover_v2_sessions(self.root)
        self.assertEqual([(v.person, v.session_id) for v in valid], [("demo_person", "session-001")])
        self.assertEqual(rejected, [])
        self.assertEqual(valid[0].frames, N)
        self.assertEqual((valid[0].width, valid[0].height), (W, H))

    def test_version_one_only_session_is_not_a_v2_session(self):
        session = self.root / "demo_person" / "session-002"
        session.mkdir(parents=True)
        (session / "facial_data.json").write_text("[]", encoding="utf-8")
        valid, rejected = vv.discover_v2_sessions(self.root)
        self.assertEqual(valid, [])
        self.assertEqual(rejected, [])

    def test_incomplete_v2_session_is_rejected_with_reason(self):
        session = self.root / "demo_person" / "session-003"
        write_v2(session)
        (session / "metadata.json").unlink()
        valid, rejected = vv.discover_v2_sessions(self.root)
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), 1)
        self.assertIn("incomplete", rejected[0][1])

    def test_tampered_file_is_rejected(self):
        session = write_v2(self.root / "demo_person" / "session-004")
        (session / "timeline.csv").write_text("changed\n", encoding="utf-8")
        valid, rejected = vv.discover_v2_sessions(self.root)
        self.assertEqual(valid, [])
        self.assertIn("manifest", rejected[0][1])

    def test_corrupt_metadata_is_rejected(self):
        session = write_v2(self.root / "demo_person" / "session-005")
        (session / "metadata.json").write_text("{not json", encoding="utf-8")
        valid, rejected = vv.discover_v2_sessions(self.root)
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), 1)

    def test_missing_root_gives_no_sessions(self):
        valid, rejected = vv.discover_v2_sessions(self.tmp / "absent")
        self.assertEqual((valid, rejected), ([], []))


class ArrayAndTimelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_viewer_arr_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pose_array_shape_is_checked(self):
        session = write_v2(self.tmp / "s")
        np.save(session / "body" / "pose_landmarks.npy", np.zeros((N, 32, 5), dtype=np.float32))
        with self.assertRaisesRegex(vv.V2ViewerError, "shape"):
            vv.load_pose(session)

    def test_pose_array_dtype_is_checked(self):
        session = write_v2(self.tmp / "s")
        np.save(session / "body" / "pose_landmarks.npy", np.zeros((N, 33, 5), dtype=np.float64))
        with self.assertRaisesRegex(vv.V2ViewerError, "dtype"):
            vv.load_pose(session)

    def test_missing_detection_frames_are_nan_rows(self):
        pose = np.stack([pose_frame() for _ in range(N)])
        pose[3] = np.nan
        session = write_v2(self.tmp / "s", pose=pose)
        loaded = vv.load_pose(session)
        self.assertTrue(np.isnan(loaded[3]).all())
        self.assertTrue(np.isfinite(loaded[2]).all())

    def test_timeline_must_align_with_the_pose_frames(self):
        session = write_v2(self.tmp / "s")
        timeline = pd.read_csv(session / "timeline.csv").iloc[:-1]
        timeline.to_csv(session / "timeline.csv", index=False)
        with self.assertRaises(vv.V2ViewerError):
            vv.load_timeline(session, expected_frames=N)


class JointTests(unittest.TestCase):
    def test_there_are_33_named_joints(self):
        names = vv.joint_names()
        self.assertEqual(len(names), 33)
        self.assertEqual(len(set(names)), 33)
        self.assertEqual(names[0], "nose")

    def test_skeleton_uses_the_official_35_edges(self):
        edges = vv.pose_edges()
        self.assertEqual(len(edges), 35)
        self.assertTrue(all(0 <= a < 33 and 0 <= b < 33 for a, b in edges))

    def test_joint_states_distinguish_missing_low_visibility_off_frame_and_shown(self):
        frame = pose_frame(x=0.5, y=0.5, vis=0.9)
        frame[1] = np.nan                      # missing
        frame[2, 3] = 0.1                      # low visibility, inside frame
        frame[3, :2] = (1.4, 0.5)              # off frame, visible
        states = vv.joint_states(frame, threshold=0.5)
        self.assertEqual(states[0], "shown")
        self.assertEqual(states[1], "missing")
        self.assertEqual(states[2], "low_visibility")
        self.assertEqual(states[3], "off_frame")

    def test_visibility_threshold_changes_states_but_not_stored_values(self):
        frame = pose_frame(vis=0.6)
        before = frame.copy()
        vv.joint_states(frame, threshold=0.5)
        vv.joint_states(frame, threshold=0.9)
        np.testing.assert_array_equal(frame, before)
        self.assertEqual(vv.joint_states(frame, 0.5)[0], "shown")
        self.assertEqual(vv.joint_states(frame, 0.9)[0], "low_visibility")


class CoordinateAndOverlayTests(unittest.TestCase):
    def test_normalised_centre_maps_to_frame_centre(self):
        px = vv.pixel_positions(pose_frame(x=0.5, y=0.5), W, H)
        np.testing.assert_allclose(px[0], (W * 0.5, H * 0.5))

    def test_nan_joint_has_nan_pixel_position(self):
        frame = pose_frame()
        frame[4] = np.nan
        px = vv.pixel_positions(frame, W, H)
        self.assertTrue(np.isnan(px[4]).all())

    def test_overlay_keeps_frame_shape_and_does_not_change_input(self):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        before = img.copy()
        out = vv.draw_pose_overlay(img, pose_frame(), threshold=0.5)
        self.assertEqual(out.shape, img.shape)
        np.testing.assert_array_equal(img, before)
        self.assertTrue(out.any())

    def test_frame_without_pose_returns_the_frame_unchanged(self):
        img = np.full((H, W, 3), 40, dtype=np.uint8)
        out = vv.draw_pose_overlay(img, np.full((33, 5), np.nan, dtype=np.float32), threshold=0.5)
        np.testing.assert_array_equal(out, img)

    def test_joint_is_drawn_at_its_pixel_position(self):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        frame = pose_frame(x=0.5, y=0.5, vis=0.9)
        out = vv.draw_pose_overlay(img, frame, threshold=0.5, show_skeleton=False)
        ys, xs = np.nonzero(out.any(axis=2))
        self.assertTrue(abs(xs.mean() - W * 0.5) < 3 and abs(ys.mean() - H * 0.5) < 3)


class QualityTests(unittest.TestCase):
    def test_summary_reports_detection_rate_missing_frames_and_visibility(self):
        pose = np.stack([pose_frame(vis=0.8) for _ in range(N)])
        pose[5] = np.nan
        timeline = pd.DataFrame({"pose_detected": [int(np.isfinite(p).all()) for p in pose],
                                 "face_detected": [1] * N})
        summary = vv.quality_summary(pose, timeline, threshold=0.5, selected_frame=0)
        self.assertEqual(summary["frames"], N)
        self.assertAlmostEqual(summary["pose_detection_rate"], (N - 1) / N)
        self.assertEqual(summary["missing_pose_frames"], [5])
        self.assertAlmostEqual(summary["median_visibility"], 0.8, places=5)
        self.assertEqual(summary["valid_joints_selected_frame"], 33)

    def test_summary_is_not_labelled_as_motion_accuracy(self):
        pose = np.stack([pose_frame() for _ in range(N)])
        timeline = pd.DataFrame({"pose_detected": [1] * N, "face_detected": [1] * N})
        summary = vv.quality_summary(pose, timeline, threshold=0.5, selected_frame=0)
        self.assertIn("does not measure", summary["scope"])


class ExistingDashboardTests(unittest.TestCase):
    def test_v1_helpers_still_import_and_the_body_page_runs_without_v2_data(self):
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "src" / "dashboard" / "pages" / "04_Body_Motion.py"), default_timeout=120)
        at.run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main(verbosity=2)
