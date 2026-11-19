"""Tests for the pose visibility gate (synthetic arrays only)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import pose_gating as pg  # noqa: E402


def pose(rows):
    """rows: list of (x, y, z, visibility, presence) for each of 33 joints, repeated per frame."""
    arr = np.zeros((1, 33, 5), dtype=np.float32)
    for k, row in enumerate(rows):
        arr[0, k] = row
    return arr


class GateRuleTests(unittest.TestCase):
    def test_in_frame_joint_is_kept_whatever_its_visibility(self):
        arr = np.full((1, 33, 5), 0.5, dtype=np.float32)
        arr[0, :, 3] = 0.0  # visibility 0 everywhere
        out, _ = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isfinite(out[0, :, :3]).all())

    def test_out_of_frame_joint_with_low_visibility_is_gated(self):
        arr = np.full((1, 33, 5), 0.5, dtype=np.float32)
        arr[0, 5, :2] = (1.4, 2.0)
        arr[0, 5, 3] = 0.1
        out, report = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isnan(out[0, 5, :3]).all())
        self.assertEqual(float(out[0, 5, 3]), np.float32(0.1))  # visibility kept as recorded
        self.assertEqual(report.gated_joint_samples, 1)

    def test_out_of_frame_joint_with_high_visibility_is_kept(self):
        arr = np.full((1, 33, 5), 0.5, dtype=np.float32)
        arr[0, 7, :2] = (-0.1, 0.5)
        arr[0, 7, 3] = 0.9
        out, report = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isfinite(out[0, 7, :3]).all())
        self.assertEqual(report.gated_joint_samples, 0)

    def test_boundary_values_0_and_1_are_inside_the_frame(self):
        arr = np.full((1, 33, 5), 0.5, dtype=np.float32)
        arr[0, 2, :2] = (0.0, 1.0)
        arr[0, 2, 3] = 0.0
        out, _ = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isfinite(out[0, 2, :3]).all())

    def test_threshold_boundary_is_inclusive(self):
        arr = np.full((1, 33, 5), 0.5, dtype=np.float32)
        arr[0, 3, :2] = (1.2, 0.5)
        arr[0, 3, 3] = 0.5
        out, _ = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isfinite(out[0, 3, :3]).all())


class ArrayHandlingTests(unittest.TestCase):
    def test_shape_and_dtype_are_preserved(self):
        arr = np.full((4, 33, 5), 0.5, dtype=np.float32)
        out, _ = pg.gate_pose(arr, threshold=0.5)
        self.assertEqual(out.shape, arr.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_input_is_not_modified(self):
        arr = np.full((2, 33, 5), 0.5, dtype=np.float32)
        arr[0, 0, :2] = (1.5, 0.5)
        arr[0, 0, 3] = 0.0
        before = arr.copy()
        pg.gate_pose(arr, threshold=0.5)
        np.testing.assert_array_equal(arr, before)

    def test_missing_detections_stay_nan(self):
        arr = np.full((2, 33, 5), np.nan, dtype=np.float32)
        out, report = pg.gate_pose(arr, threshold=0.5)
        self.assertTrue(np.isnan(out).all())
        self.assertEqual(report.gated_joint_samples, 0)

    def test_report_counts_frames_and_joints(self):
        arr = np.full((3, 33, 5), 0.5, dtype=np.float32)
        arr[:, 10, :2] = 1.5
        arr[:, 10, 3] = 0.0
        _, report = pg.gate_pose(arr, threshold=0.5)
        self.assertEqual(report.gated_joint_samples, 3)
        self.assertEqual(report.frames_with_gated_joints, 3)
        self.assertEqual(report.threshold, 0.5)


class ValidationTests(unittest.TestCase):
    def test_threshold_must_be_between_zero_and_one(self):
        arr = np.zeros((1, 33, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            pg.gate_pose(arr, threshold=1.5)
        with self.assertRaises(ValueError):
            pg.gate_pose(arr, threshold=-0.1)

    def test_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            pg.gate_pose(np.zeros((1, 32, 5), dtype=np.float32), threshold=0.5)

    def test_default_threshold_is_documented_value(self):
        self.assertEqual(pg.DEFAULT_VISIBILITY_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
