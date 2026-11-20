"""Tests for the Pose Landmarker extractor. A fake detector replaces MediaPipe, so no model runs here."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import pose_extractor as pe  # noqa: E402
from extraction.model_registry import ModelIntegrityError  # noqa: E402
from extraction.video_frames import VideoError  # noqa: E402
from pose_schema import frame_timestamps_ms  # noqa: E402

W, H, FPS, N = 64, 48, 10, 8


def fake_landmark(k, t):
    return SimpleNamespace(x=0.1 * k + 0.01 * t, y=0.2 + 0.01 * k, z=-0.05 * k,
                           visibility=0.9, presence=0.8)


class FakeDetector:
    """Returns a pose on every frame except those listed in missing; records the timestamps it receives."""

    def __init__(self, missing=(), world=True, n_points=33, poses_per_frame=1):
        self.missing = set(missing)
        self.world = world
        self.n_points = n_points
        self.poses_per_frame = poses_per_frame
        self.timestamps = []
        self.frame_shapes = []

    def detect(self, frame_bgr, timestamp_ms):
        index = len(self.timestamps)
        self.timestamps.append(timestamp_ms)
        self.frame_shapes.append(frame_bgr.shape)
        if index in self.missing:
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        poses = [[fake_landmark(k, index) for k in range(self.n_points)] for _ in range(self.poses_per_frame)]
        world = [[fake_landmark(k, index) for k in range(self.n_points)]] if self.world else []
        return SimpleNamespace(pose_landmarks=poses, pose_world_landmarks=world)


def frames_of(n=N):
    return [(i, np.full((H, W, 3), i * 10 % 256, np.uint8)) for i in range(n)]


def timestamps_of(n=N):
    ts, _ = frame_timestamps_ms(n, fps=FPS)
    return ts


class ArrayTests(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        result = pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector())
        self.assertEqual(result.image.shape, (N, 33, 5))
        self.assertEqual(result.world.shape, (N, 33, 5))
        self.assertEqual(result.image.dtype, np.float32)
        self.assertEqual(result.world.dtype, np.float32)
        self.assertEqual(result.detected.shape, (N,))
        self.assertEqual(result.detected.dtype, bool)

    def test_channels_are_x_y_z_visibility_presence(self):
        result = pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector())
        self.assertAlmostEqual(float(result.image[2, 5, 0]), 0.5 + 0.02, places=5)  # x = 0.1*5 + 0.01*2
        self.assertAlmostEqual(float(result.image[2, 5, 2]), -0.25, places=5)      # z = -0.05*5
        self.assertAlmostEqual(float(result.image[2, 5, 3]), 0.9, places=5)        # visibility
        self.assertAlmostEqual(float(result.image[2, 5, 4]), 0.8, places=5)        # presence


class MissingDetectionTests(unittest.TestCase):
    def test_missing_pose_is_a_nan_row_not_a_dropped_frame(self):
        result = pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector(missing={3}))
        self.assertEqual(result.image.shape[0], N)
        self.assertFalse(result.detected[3])
        self.assertTrue(np.isnan(result.image[3]).all())
        self.assertTrue(np.isnan(result.world[3]).all())
        self.assertTrue(result.detected[2] and result.detected[4])

    def test_missing_world_landmarks_keep_image_landmarks(self):
        result = pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector(world=False))
        self.assertTrue(np.isfinite(result.image[0]).all())
        self.assertTrue(np.isnan(result.world).all())

    def test_detection_rate_is_reported(self):
        result = pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector(missing={0, 1}))
        self.assertAlmostEqual(result.detection_rate, (N - 2) / N)


class TimingAndFrameTests(unittest.TestCase):
    def test_detector_receives_the_shared_timestamps_in_order(self):
        det = FakeDetector()
        pe.extract_pose_from_frames(frames_of(), timestamps_of(), det)
        self.assertEqual(det.timestamps, [int(t) for t in timestamps_of()])
        self.assertTrue(all(b > a for a, b in zip(det.timestamps, det.timestamps[1:])))

    def test_every_frame_is_passed_to_the_detector(self):
        det = FakeDetector()
        pe.extract_pose_from_frames(frames_of(), timestamps_of(), det)
        self.assertEqual(len(det.frame_shapes), N)

    def test_timestamp_count_must_match_frame_count(self):
        with self.assertRaisesRegex(pe.ExtractionError, "timestamps"):
            pe.extract_pose_from_frames(frames_of(), timestamps_of(N - 1), FakeDetector())

    def test_frame_index_gap_is_refused(self):
        frames = frames_of()
        del frames[4]
        with self.assertRaisesRegex(pe.ExtractionError, "frame index"):
            pe.extract_pose_from_frames(frames, timestamps_of(), FakeDetector())


class SingleSubjectTests(unittest.TestCase):
    def test_more_than_one_person_is_refused_not_merged(self):
        with self.assertRaisesRegex(pe.ExtractionError, "more than one person"):
            pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector(poses_per_frame=2))

    def test_wrong_landmark_count_is_refused(self):
        with self.assertRaisesRegex(pe.ExtractionError, "33"):
            pe.extract_pose_from_frames(frames_of(), timestamps_of(), FakeDetector(n_points=32))


class ConfigTests(unittest.TestCase):
    def test_default_config_is_single_person(self):
        self.assertEqual(pe.PoseConfig().num_poses, 1)

    def test_more_than_one_pose_is_rejected_by_the_config(self):
        with self.assertRaises(ValueError):
            pe.PoseConfig(num_poses=2)

    def test_thresholds_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            pe.PoseConfig(min_pose_detection_confidence=1.5)
        with self.assertRaises(ValueError):
            pe.PoseConfig(min_tracking_confidence=-0.1)

    def test_config_is_recorded_for_metadata(self):
        record = pe.PoseConfig().as_dict()
        self.assertEqual(record["num_poses"], 1)
        self.assertIn("min_pose_detection_confidence", record)


class ModelGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pose_gate_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unverified_model_is_refused_before_mediapipe_is_imported(self):
        (self.tmp / "pose_landmarker_full.task").write_bytes(b"not the real model")
        with self.assertRaises(ModelIntegrityError):
            pe.load_pose_detector(models_dir=self.tmp)

    def test_missing_model_is_refused(self):
        with self.assertRaisesRegex(ModelIntegrityError, "not found"):
            pe.load_pose_detector(models_dir=self.tmp)


class VideoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pose_video_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, name, n):
        path = self.tmp / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i in range(n):
            writer.write(np.full((H, W, 3), i * 10 % 256, np.uint8))
        writer.release()
        return path

    def test_video_path_is_read_and_extracted_with_a_fake_detector(self):
        path = self._write("clip.mp4", N)
        result = pe.extract_pose_video(path, detector=FakeDetector())
        self.assertEqual(result.image.shape, (N, 33, 5))
        self.assertEqual(result.source_filename, "clip.mp4")
        self.assertEqual(result.timestamp_source, "fps_derived")

    def test_corrupt_video_raises_a_video_error(self):
        bad = self.tmp / "bad.mp4"
        bad.write_bytes(b"not a video" * 30)
        with self.assertRaises(VideoError):
            pe.extract_pose_video(bad, detector=FakeDetector())


if __name__ == "__main__":
    unittest.main(verbosity=2)
