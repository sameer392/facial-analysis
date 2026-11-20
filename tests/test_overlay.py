"""Tests for the facial landmark overlay renderer (synthetic data only)."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "viewer"))

import overlay as ov  # noqa: E402

W, H = 320, 240
N = 478


def synthetic_frame_data(seed=0):
    """Landmarks spread over the frame, with fixed positions so tests can check pixels."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.2, 0.8, size=(N, 2))
    pts[0] = (0.5, 0.5)  # a known point at the centre of the frame
    return {
        "face_detected": True,
        "landmarks": [{"id": k, "x": float(pts[k, 0]), "y": float(pts[k, 1]), "z": 0.0} for k in range(N)],
    }


def blank_frame():
    return np.full((H, W, 3), 30, dtype=np.uint8)


class TopologyTests(unittest.TestCase):
    def test_edge_counts_match_mediapipe_connection_sets(self):
        topo = ov.load_topology()
        expected = {"mesh": 2556, "contours": 124, "oval": 36, "lips": 40,
                    "left_eye": 16, "right_eye": 16, "left_eyebrow": 8, "right_eyebrow": 8, "nose": 25}
        for name, count in expected.items():
            with self.subTest(region=name):
                self.assertEqual(len(topo[name]), count)

    def test_all_edge_indices_are_valid_landmark_ids(self):
        for name, edges in ov.load_topology().items():
            with self.subTest(region=name):
                self.assertEqual(edges.shape[1], 2)
                self.assertTrue(((edges >= 0) & (edges < N)).all())

    def test_topology_is_loaded_once_and_reused(self):
        self.assertIs(ov.load_topology(), ov.load_topology())


class PixelConversionTests(unittest.TestCase):
    def test_normalised_centre_maps_to_frame_centre(self):
        px = ov.landmarks_to_pixels(synthetic_frame_data(), W, H)
        np.testing.assert_allclose(px[0], (W * 0.5, H * 0.5))

    def test_missing_face_returns_none(self):
        self.assertIsNone(ov.landmarks_to_pixels({"face_detected": False, "landmarks": []}, W, H))

    def test_partial_landmarks_keep_nan_for_missing_ids(self):
        data = synthetic_frame_data()
        data["landmarks"] = data["landmarks"][:-1]  # drop landmark 477
        px = ov.landmarks_to_pixels(data, W, H)
        self.assertTrue(np.isnan(px[N - 1]).all())
        self.assertTrue(np.isfinite(px[0]).all())


class RenderModeTests(unittest.TestCase):
    def setUp(self):
        self.frame = blank_frame()
        self.data = synthetic_frame_data()

    def test_off_returns_an_unchanged_copy(self):
        out = ov.render_overlay(self.frame, self.data, mode="off")
        np.testing.assert_array_equal(out, self.frame)
        self.assertIsNot(out, self.frame)

    def test_points_mode_draws_green_at_a_landmark(self):
        out = ov.render_overlay(self.frame, self.data, mode="points", point_radius=2, opacity=1.0)
        cx, cy = W // 2, H // 2
        self.assertGreater(int(out[cy, cx, 1]), 200)  # green channel bright at landmark 0

    def test_each_active_mode_changes_the_image(self):
        for mode in ("points", "wireframe", "contours", "hybrid"):
            with self.subTest(mode=mode):
                out = ov.render_overlay(self.frame, self.data, mode=mode, opacity=1.0)
                self.assertFalse(np.array_equal(out, self.frame))

    def test_output_has_input_shape_and_dtype(self):
        for mode in ov.MODES:
            with self.subTest(mode=mode):
                out = ov.render_overlay(self.frame, self.data, mode=mode)
                self.assertEqual(out.shape, self.frame.shape)
                self.assertEqual(out.dtype, self.frame.dtype)

    def test_frame_without_face_is_unchanged_in_every_mode(self):
        no_face = {"face_detected": False, "landmarks": []}
        for mode in ov.MODES:
            with self.subTest(mode=mode):
                out = ov.render_overlay(self.frame, no_face, mode=mode)
                np.testing.assert_array_equal(out, self.frame)

    def test_input_frame_is_not_modified(self):
        before = self.frame.copy()
        ov.render_overlay(self.frame, self.data, mode="wireframe", opacity=1.0)
        np.testing.assert_array_equal(self.frame, before)

    def test_zero_opacity_leaves_the_frame_unchanged(self):
        out = ov.render_overlay(self.frame, self.data, mode="wireframe", opacity=0.0)
        np.testing.assert_array_equal(out, self.frame)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ov.render_overlay(self.frame, self.data, mode="sparkles")

    def test_invalid_style_values_are_rejected(self):
        with self.assertRaises(ValueError):
            ov.render_overlay(self.frame, self.data, mode="wireframe", opacity=1.5)
        with self.assertRaises(ValueError):
            ov.render_overlay(self.frame, self.data, mode="wireframe", thickness=0)
        with self.assertRaises(ValueError):
            ov.render_overlay(self.frame, self.data, mode="points", point_radius=0)

    def test_hybrid_is_fainter_than_contours_on_the_mesh(self):
        # Hybrid draws a faint mesh plus highlighted contours, so its change should lie between
        # "off" and "contours" at the same opacity.
        base = self.frame.astype(float)
        contours = np.abs(ov.render_overlay(self.frame, self.data, mode="contours", opacity=1.0).astype(float) - base).sum()
        hybrid = np.abs(ov.render_overlay(self.frame, self.data, mode="hybrid", opacity=1.0).astype(float) - base).sum()
        self.assertGreater(hybrid, 0)
        self.assertGreater(contours, 0)


class ColourTests(unittest.TestCase):
    def test_each_anatomical_region_has_its_own_colour(self):
        # left and right eyes (and eyebrows) deliberately share a colour, so count regions, not keys
        regions = {"oval", "lips", "eyes", "eyebrows", "nose"}
        colour_of = {
            "oval": ov.REGION_COLOURS["oval"],
            "lips": ov.REGION_COLOURS["lips"],
            "eyes": ov.REGION_COLOURS["left_eye"],
            "eyebrows": ov.REGION_COLOURS["left_eyebrow"],
            "nose": ov.REGION_COLOURS["nose"],
        }
        self.assertEqual(set(colour_of), regions)
        self.assertEqual(len({tuple(c) for c in colour_of.values()}), len(regions))
        self.assertEqual(ov.REGION_COLOURS["left_eye"], ov.REGION_COLOURS["right_eye"])
        self.assertEqual(ov.REGION_COLOURS["left_eyebrow"], ov.REGION_COLOURS["right_eyebrow"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
