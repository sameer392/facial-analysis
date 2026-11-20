"""Hand extraction (phase H1): status rules, coordinates, frame alignment and error handling.

Synthetic detector output only. No MediaPipe model and no personal video is needed.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import hand_extractor as he  # noqa: E402
import pose_schema as s2  # noqa: E402


def lm(x=0.5, y=0.5, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def hand(x=0.5, y=0.5, outside=0):
    """21 landmarks spread around (x, y). outside > 0 moves that many landmarks beyond the frame edge."""
    pts = [lm(x + 0.01 * k, y + 0.01 * k, 0.001 * k) for k in range(21)]
    for k in range(outside):
        pts[k] = lm(1.2, 0.5, 0.0)
    return pts


def world_hand():
    return [SimpleNamespace(x=0.01 * k, y=-0.02 * k, z=0.005 * k) for k in range(21)]


def categories(label, score):
    return [SimpleNamespace(category_name=label, score=score)]


def output(hands=(), worlds=None, labels=()):
    """hands: landmark lists; labels: (label, score) per hand; worlds: world lists or None to omit."""
    return SimpleNamespace(
        hand_landmarks=list(hands),
        hand_world_landmarks=list(worlds) if worlds is not None else [],
        handedness=[categories(lbl, sc) for lbl, sc in labels],
    )


class ScriptedDetector:
    """Returns one scripted output per frame. Records the timestamps it was called with."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.timestamps = []
        self.closed = 0

    def detect(self, frame_bgr, timestamp_ms):
        self.timestamps.append(int(timestamp_ms))
        return self.outputs[len(self.timestamps) - 1]

    def close(self):
        self.closed += 1


def run(outputs, timestamps=None, config=None):
    n = len(outputs)
    ts = np.asarray(timestamps if timestamps is not None else [i * 100 for i in range(n)], dtype=np.int64)
    frames = [(i, np.zeros((4, 4, 3), np.uint8)) for i in range(n)]
    det = ScriptedDetector(outputs)
    result = he.extract_hands_from_frames(frames, ts, det, config=config)
    return result, det


class StatusRuleTests(unittest.TestCase):
    def test_single_right_hand_is_valid_and_left_is_not_detected(self):
        result, _ = run([output([hand()], [world_hand()], [("Right", 0.98)])])
        self.assertEqual(result.right_status, ["valid"])
        self.assertEqual(result.left_status, ["not_detected"])
        self.assertAlmostEqual(float(result.right_score[0]), 0.98, places=5)
        self.assertTrue(np.all(np.isfinite(result.right[0])))
        self.assertTrue(np.all(np.isnan(result.left[0])))

    def test_two_hands_are_assigned_by_label(self):
        result, _ = run([output([hand(0.2), hand(0.8)], [world_hand(), world_hand()],
                                [("Left", 0.9), ("Right", 0.95)])])
        self.assertEqual(result.left_status, ["valid"])
        self.assertEqual(result.right_status, ["valid"])
        self.assertAlmostEqual(float(result.left[0, 0, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(result.right[0, 0, 0]), 0.8, places=5)

    def test_frame_without_hands_is_a_nan_row_and_is_kept(self):
        result, _ = run([output(), output([hand()], [world_hand()], [("Left", 0.9)])])
        self.assertEqual(result.frames, 2)
        self.assertEqual(result.left_status, ["not_detected", "valid"])
        self.assertTrue(np.all(np.isnan(result.left[0])))
        self.assertTrue(np.all(np.isnan(result.right[0])))

    def test_low_confidence_keeps_the_landmarks_and_says_so(self):
        result, _ = run([output([hand()], [world_hand()], [("Right", 0.40)])])
        self.assertEqual(result.right_status, ["low_confidence"])
        self.assertTrue(np.all(np.isfinite(result.right[0])))

    def test_landmarks_outside_the_frame_are_counted(self):
        result, _ = run([output([hand(outside=3)], [world_hand()], [("Right", 0.95)])])
        self.assertEqual(result.right_status, ["outside_frame"])
        self.assertEqual(int(result.right_outside[0]), 3)

    def test_low_confidence_takes_precedence_over_outside_frame(self):
        result, _ = run([output([hand(outside=2)], [world_hand()], [("Right", 0.3)])])
        self.assertEqual(result.right_status, ["low_confidence"])

    def test_two_hands_with_the_same_label_are_unassigned_not_guessed(self):
        result, _ = run([output([hand(0.2), hand(0.8)], [world_hand(), world_hand()],
                                [("Left", 0.9), ("Left", 0.8)])])
        self.assertEqual(result.left_status, ["unassigned"])
        self.assertEqual(result.right_status, ["not_detected"])
        self.assertTrue(np.all(np.isnan(result.left[0])))
        self.assertEqual(int(result.unassigned[0]), 2)

    def test_unknown_label_is_counted_but_not_assigned(self):
        result, _ = run([output([hand()], [world_hand()], [("Hand", 0.9)])])
        self.assertEqual(result.left_status, ["not_detected"])
        self.assertEqual(result.right_status, ["not_detected"])
        self.assertEqual(int(result.unassigned[0]), 1)


class WorldCoordinateTests(unittest.TestCase):
    def test_world_coordinates_are_stored_when_given(self):
        result, _ = run([output([hand()], [world_hand()], [("Right", 0.95)])])
        self.assertTrue(bool(result.right_world_available[0]))
        self.assertAlmostEqual(float(result.right_world[0, 1, 1]), -0.02, places=5)

    def test_missing_world_coordinates_are_explicit_not_estimated(self):
        result, _ = run([output([hand()], None, [("Right", 0.95)])])
        self.assertFalse(bool(result.right_world_available[0]))
        self.assertTrue(np.all(np.isnan(result.right_world[0])))
        self.assertTrue(np.all(np.isfinite(result.right[0])))

    def test_image_and_world_are_kept_separately(self):
        result, _ = run([output([hand(0.3, 0.6)], [world_hand()], [("Right", 0.95)])])
        self.assertAlmostEqual(float(result.right[0, 0, 0]), 0.3, places=5)
        self.assertAlmostEqual(float(result.right_world[0, 0, 0]), 0.0, places=5)


class FrameAlignmentTests(unittest.TestCase):
    def test_timestamps_are_passed_through_unchanged(self):
        result, det = run([output(), output(), output()], timestamps=[0, 33, 67])
        self.assertEqual(det.timestamps, [0, 33, 67])
        self.assertEqual(result.timestamps_ms.tolist(), [0, 33, 67])

    def test_timestamps_must_increase(self):
        with self.assertRaises(he.HandExtractionError):
            run([output(), output()], timestamps=[100, 100])

    def test_out_of_sequence_frame_index_is_refused(self):
        det = ScriptedDetector([output(), output()])
        frames = [(0, np.zeros((2, 2, 3), np.uint8)), (2, np.zeros((2, 2, 3), np.uint8))]
        with self.assertRaises(he.HandExtractionError):
            he.extract_hands_from_frames(frames, np.array([0, 100]), det)

    def test_frame_count_mismatch_is_refused(self):
        det = ScriptedDetector([output(), output()])
        frames = [(0, np.zeros((2, 2, 3), np.uint8))]
        with self.assertRaises(he.HandExtractionError):
            he.extract_hands_from_frames(frames, np.array([0, 100]), det)


class LimitTests(unittest.TestCase):
    def test_more_hands_than_configured_is_refused(self):
        one = he.HandConfig(num_hands=1)
        with self.assertRaises(he.HandExtractionError):
            run([output([hand(), hand()], [world_hand(), world_hand()], [("Left", 0.9), ("Right", 0.9)])],
                config=one)

    def test_wrong_landmark_count_is_refused(self):
        bad = [lm() for _ in range(20)]
        with self.assertRaises(he.HandExtractionError):
            run([output([bad], [world_hand()], [("Right", 0.9)])])

    def test_config_values_are_checked(self):
        with self.assertRaises(ValueError):
            he.HandConfig(num_hands=3)
        with self.assertRaises(ValueError):
            he.HandConfig(low_confidence=1.5)

    def test_result_is_deterministic(self):
        outs = [output([hand(0.2)], [world_hand()], [("Left", 0.9)]), output()]
        a, _ = run(outs)
        b, _ = run(outs)
        self.assertTrue(np.array_equal(a.left, b.left, equal_nan=True))
        self.assertEqual(a.left_status, b.left_status)

    def test_status_counts_add_up(self):
        result, _ = run([output(), output([hand()], [world_hand()], [("Right", 0.95)]),
                         output([hand(outside=1)], [world_hand()], [("Right", 0.95)])])
        counts = result.status_counts()
        self.assertEqual(counts["right_valid"], 1)
        self.assertEqual(counts["right_outside_frame"], 1)
        self.assertEqual(counts["right_not_detected"], 1)
        self.assertEqual(counts["left_not_detected"], 3)


class SchemaAgreementTests(unittest.TestCase):
    def test_extracted_rows_pass_the_schema_checks(self):
        outs = [output(), output([hand()], [world_hand()], [("Right", 0.95)]),
                output([hand(outside=1)], [world_hand()], [("Left", 0.3)])]
        result, _ = run(outs)
        rows = []
        for i in range(result.frames):
            rows.append({
                "frame": i, "timestamp_ms": int(result.timestamps_ms[i]),
                "left_status": result.left_status[i],
                "left_score": None if np.isnan(result.left_score[i]) else float(result.left_score[i]),
                "left_outside_landmarks": int(result.left_outside[i]),
                "left_world_available": bool(result.left_world_available[i]),
                "right_status": result.right_status[i],
                "right_score": None if np.isnan(result.right_score[i]) else float(result.right_score[i]),
                "right_outside_landmarks": int(result.right_outside[i]),
                "right_world_available": bool(result.right_world_available[i]),
                "unassigned_hands": int(result.unassigned[i]),
            })
        arrays = {
            "hands/left_hand.npy": result.left, "hands/right_hand.npy": result.right,
            "hands/left_hand_world.npy": result.left_world, "hands/right_hand_world.npy": result.right_world,
        }
        self.assertEqual(s2.validate_hand_status_rows(rows, result.frames, arrays), [])

    def test_schema_rejects_a_detected_hand_with_nan_landmarks(self):
        rows = [{"frame": 0, "timestamp_ms": 0, "left_status": "valid", "left_score": 0.9,
                 "left_outside_landmarks": 0, "left_world_available": False,
                 "right_status": "not_detected", "right_score": None, "right_outside_landmarks": 0,
                 "right_world_available": False, "unassigned_hands": 0}]
        arrays = {"hands/left_hand.npy": np.full((1, 21, 3), np.nan, np.float32)}
        self.assertTrue(s2.validate_hand_status_rows(rows, 1, arrays))

    def test_schema_rejects_an_unknown_status(self):
        rows = [{"frame": 0, "timestamp_ms": 0, "left_status": "maybe", "left_score": None,
                 "left_outside_landmarks": 0, "left_world_available": False,
                 "right_status": "not_detected", "right_score": None, "right_outside_landmarks": 0,
                 "right_world_available": False, "unassigned_hands": 0}]
        self.assertTrue(s2.validate_hand_status_rows(rows, 1, {}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
