"""Scientific validation of head-pose compensation and motion metrics.

The synthetic model used here follows the convention found in the local study
(see docs/validation.md):

    iso_image = (P @ R.T) * s @ AXIS_FLIP + t          (pixel-proportional units)
    x_norm = iso_x / W,  y_norm = iso_y / H,  z_norm = iso_z / W     (MediaPipe-style normalisation)

P is a canonical face shape, R a rotation, s a scale, t a translation, AXIS_FLIP = diag(-1, 1, 1).
Compensation must recover the canonical shape (centred, scale-normalised) for any R, s and t.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from motion_analyzer import (  # noqa: E402
    compensate_head_pose,
    landmark_kinematics,
    neutral_frame_mask,
    neutral_shape,
)

AXIS_FLIP = np.diag([-1.0, 1.0, 1.0])
W, H = 576, 1024  # portrait frame, as in the local recordings
N = 478


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def canonical_shape(seed: int = 7) -> np.ndarray:
    """A fixed, non-degenerate face-like point cloud (not a real face)."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    pts[:, 1] *= 1.3
    pts[:, 2] *= 0.6
    return pts


def project(P, R, s=1.0, t=(0.0, 0.0, 0.0), flip=AXIS_FLIP):
    """Synthetic MediaPipe-style landmarks for one frame, normalised to the frame size."""
    iso = (P @ R.T) * s @ flip + np.asarray(t, dtype=float)
    iso = iso + np.array([W / 2, H / 2, 0.0])
    return np.stack([iso[:, 0] / W, iso[:, 1] / H, iso[:, 2] / W], axis=1)


def matrix(R, t=(0.0, 0.0, 0.0), scale=1.0):
    m = np.eye(4)
    m[:3, :3] = R * scale
    m[:3, 3] = t
    return m


def expected_shape(P):
    c = P - P.mean(axis=0)
    return c / np.sqrt(np.mean(np.sum(c ** 2, axis=1)))


class HeadPoseCompensationTests(unittest.TestCase):
    P = canonical_shape()

    def _compensate(self, frames):
        landmarks = np.stack([project(**f["proj"]) for f in frames])
        mats = np.stack([matrix(f["R"], f.get("t_mat", (0, 0, 0)), f.get("scale", 1.0)) for f in frames])
        return compensate_head_pose(landmarks, mats, (W, H))

    def test_rotation_about_each_axis_is_removed(self):
        exp = expected_shape(self.P)
        for name, R in [("x", rot_x(0.25)), ("y", rot_y(-0.35)), ("z", rot_z(0.5))]:
            with self.subTest(axis=name):
                out = self._compensate([{"proj": {"P": self.P, "R": R}, "R": R}])
                np.testing.assert_allclose(out[0], exp, atol=1e-9)

    def test_combined_rotation_is_removed(self):
        R = rot_z(0.3) @ rot_y(-0.2) @ rot_x(0.4)
        out = self._compensate([{"proj": {"P": self.P, "R": R}, "R": R}])
        np.testing.assert_allclose(out[0], expected_shape(self.P), atol=1e-9)

    def test_translation_along_each_axis_has_no_effect(self):
        R = rot_y(0.2)
        exp = expected_shape(self.P)
        for t in [(40, 0, 0), (0, -60, 0), (0, 0, 15), (30, 25, -8)]:
            with self.subTest(t=t):
                out = self._compensate([{"proj": {"P": self.P, "R": R, "t": t}, "R": R, "t_mat": t}])
                np.testing.assert_allclose(out[0], exp, atol=1e-9)

    def test_scale_change_has_no_effect(self):
        R = rot_x(0.1)
        exp = expected_shape(self.P)
        for s in [0.5, 1.0, 2.7]:
            with self.subTest(scale=s):
                out = self._compensate([{"proj": {"P": self.P, "R": R, "s": s}, "R": R, "scale": s}])
                np.testing.assert_allclose(out[0], exp, atol=1e-9)

    def test_combined_rotation_translation_and_scale(self):
        R = rot_x(0.2) @ rot_z(-0.3)
        out = self._compensate([{"proj": {"P": self.P, "R": R, "s": 1.8, "t": (12, -7, 3)},
                                 "R": R, "t_mat": (12, -7, 3), "scale": 1.8}])
        np.testing.assert_allclose(out[0], expected_shape(self.P), atol=1e-9)

    def test_matrix_scale_factor_is_ignored(self):
        R = rot_y(0.3)
        out = self._compensate([{"proj": {"P": self.P, "R": R}, "R": R, "scale": 3.0}])
        np.testing.assert_allclose(out[0], expected_shape(self.P), atol=1e-9)

    def test_missing_matrix_gives_nan_and_other_frames_still_work(self):
        R = rot_x(0.1)
        landmarks = np.stack([project(self.P, R), project(self.P, R)])
        mats = np.stack([matrix(R), np.full((4, 4), np.nan)])
        out = compensate_head_pose(landmarks, mats, (W, H))
        np.testing.assert_allclose(out[0], expected_shape(self.P), atol=1e-9)
        self.assertTrue(np.isnan(out[1]).all())

    def test_missing_landmarks_give_nan(self):
        R = rot_x(0.1)
        landmarks = np.stack([project(self.P, R), np.full((N, 3), np.nan)])
        mats = np.stack([matrix(R), matrix(R)])
        out = compensate_head_pose(landmarks, mats, (W, H))
        self.assertTrue(np.isnan(out[1]).all())

    def test_aspect_ratio_matters(self):
        """Compensation must use the frame size. Ignoring the aspect ratio gives the wrong shape."""
        R = rot_y(0.4)
        landmarks = np.stack([project(self.P, R)])
        mats = np.stack([matrix(R)])
        right = compensate_head_pose(landmarks, mats, (W, H))[0]
        wrong = compensate_head_pose(landmarks, mats, (H, W))[0]  # transposed size
        self.assertGreater(np.max(np.abs(right - wrong)), 1e-3)

    def test_axis_flip_is_required_for_this_convention(self):
        """Without the x-axis flip the rotation is undone in the wrong direction."""
        R = rot_z(0.4)
        landmarks = np.stack([project(self.P, R)])
        mats = np.stack([matrix(R)])
        out = compensate_head_pose(landmarks, mats, (W, H))[0]
        no_flip = compensate_head_pose(landmarks, mats, (W, H), axis_flip=np.eye(3))[0]
        self.assertGreater(np.max(np.abs(out - no_flip)), 1e-3)
        np.testing.assert_allclose(out, expected_shape(self.P), atol=1e-9)


class NeutralCalibrationTests(unittest.TestCase):
    def test_neutral_mask_selects_lowest_activity_frames(self):
        import pandas as pd

        activity = np.linspace(0.0, 1.0, 20)
        blend = pd.DataFrame({"jawOpen": activity})
        usable = np.ones(20, dtype=bool)
        mask = neutral_frame_mask(blend, usable, quantile=0.10)
        self.assertTrue(mask[0])
        self.assertFalse(mask[10])
        self.assertLessEqual(mask.sum(), 3)

    def test_neutral_shape_is_mean_of_selected_frames(self):
        shapes = np.stack([np.full((N, 3), v) for v in (1.0, 2.0, 9.0)])
        mask = np.array([True, True, False])
        np.testing.assert_allclose(neutral_shape(shapes, mask), 1.5)

    def test_untracked_frames_are_never_neutral(self):
        import pandas as pd

        blend = pd.DataFrame({"jawOpen": [0.0, 0.0, 0.0]})
        usable = np.array([False, True, True])
        mask = neutral_frame_mask(blend, usable, quantile=0.5)
        self.assertFalse(mask[0])


class MotionMetricTests(unittest.TestCase):
    def _single_point(self, x_of_t, times):
        pts = np.zeros((len(times), 2, 3))
        pts[:, 0, 0] = x_of_t(times)
        return pts

    def test_constant_speed_on_regular_timestamps(self):
        t = np.arange(8) / 30.0
        pts = self._single_point(lambda tt: 0.4 * tt, t)
        disp, vel, acc = landmark_kinematics(pts, t, np.ones(len(t), bool))
        # the single moving landmark is half of the two-point mean, so mean speed = 0.4 / 2
        np.testing.assert_allclose(vel[1:], 0.2, atol=1e-12)
        np.testing.assert_allclose(acc[2:], 0.0, atol=1e-9)

    def test_constant_acceleration_on_regular_timestamps(self):
        t = np.arange(12) / 30.0
        a = 3.0
        pts = self._single_point(lambda tt: 0.5 * a * tt ** 2, t)
        _, vel, acc = landmark_kinematics(pts, t, np.ones(len(t), bool))
        # speed of the moving landmark is a*t, mean over two landmarks halves it
        np.testing.assert_allclose(acc[2:], a / 2, rtol=1e-6)

    def test_constant_acceleration_on_irregular_timestamps(self):
        """Acceleration must use the time between velocity samples (midpoints), not one step."""
        rng = np.random.default_rng(3)
        steps = rng.uniform(0.02, 0.06, size=14)
        t = np.concatenate([[0.0], np.cumsum(steps)])
        a = 2.0
        pts = self._single_point(lambda tt: 0.5 * a * tt ** 2, t)
        _, _, acc = landmark_kinematics(pts, t, np.ones(len(t), bool))
        np.testing.assert_allclose(acc[3:], a / 2, rtol=1e-6)

    def test_velocity_on_irregular_timestamps_is_step_over_dt(self):
        t = np.array([0.0, 0.1, 0.3, 0.35, 0.5])
        pts = self._single_point(lambda tt: 0.6 * tt, t)
        disp, vel, _ = landmark_kinematics(pts, t, np.ones(len(t), bool))
        np.testing.assert_allclose(vel[1:], 0.3, atol=1e-12)

    def test_gap_in_tracking_gives_nan_around_the_gap(self):
        t = np.arange(7) / 30.0
        pts = self._single_point(lambda tt: 0.4 * tt, t)
        usable = np.array([True, True, False, True, True, True, True])
        disp, vel, acc = landmark_kinematics(pts, t, usable)
        self.assertTrue(np.isnan(disp[2]) and np.isnan(disp[3]))
        self.assertTrue(np.isnan(vel[2]) and np.isnan(vel[3]))
        self.assertTrue(np.isnan(acc[2]) or np.isnan(acc[3]) or np.isnan(acc[4]))
        self.assertFalse(np.isnan(disp[4]))

    def test_non_increasing_timestamp_gives_nan(self):
        t = np.array([0.0, 0.1, 0.1, 0.2])
        pts = self._single_point(lambda tt: tt, t)
        _, vel, _ = landmark_kinematics(pts, t, np.ones(len(t), bool))
        self.assertTrue(np.isnan(vel[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
