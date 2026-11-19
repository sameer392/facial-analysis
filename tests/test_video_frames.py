"""Tests for the shared video frame reader (synthetic videos written to a temporary folder)."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src"))

from extraction import video_frames as vf  # noqa: E402
from schema_v2 import frame_timestamps_ms  # noqa: E402

W, H, FPS, N = 64, 48, 10, 12


def write_video(path: Path, n_frames: int = N, fps: int = FPS) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(n_frames):
        frame = np.full((H, W, 3), (i * 10) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class VideoFramesTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="video_frames_"))
        self.video = write_video(self.tmp / "clip.mp4")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class VideoInfoTests(VideoFramesTestCase):
    def test_reports_properties(self):
        info = vf.open_video_info(self.video)
        self.assertEqual((info.width, info.height), (W, H))
        self.assertAlmostEqual(info.fps, FPS, places=1)
        self.assertEqual(info.reported_frame_count, N)

    def test_missing_file_is_a_clear_error(self):
        with self.assertRaisesRegex(vf.VideoError, "not found"):
            vf.open_video_info(self.tmp / "absent.mp4")

    def test_corrupt_file_is_a_clear_error(self):
        bad = self.tmp / "corrupt.mp4"
        bad.write_bytes(b"this is not a video file" * 20)
        with self.assertRaisesRegex(vf.VideoError, "cannot be opened|no frames"):
            vf.open_video_info(bad)


class IterationTests(VideoFramesTestCase):
    def test_yields_every_frame_in_order(self):
        frames = list(vf.iter_frames(self.video))
        self.assertEqual([i for i, _ in frames], list(range(N)))

    def test_frames_have_expected_shape_and_dtype(self):
        for _, frame in vf.iter_frames(self.video):
            self.assertEqual(frame.shape, (H, W, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_timestamps_follow_the_schema_rule(self):
        ts, source = vf.frame_timestamps(self.video)
        self.assertEqual(source, "fps_derived")
        expected, _ = frame_timestamps_ms(N, fps=FPS)
        np.testing.assert_array_equal(ts, expected)
        self.assertTrue((np.diff(ts) > 0).all())

    def test_zero_frame_video_is_rejected(self):
        empty = write_video(self.tmp / "empty.mp4", n_frames=0)
        # An empty container may fail to open or open with no frames; both must be reported.
        with self.assertRaisesRegex(vf.VideoError, "no frames|cannot be opened"):
            list(vf.iter_frames(empty))

    def test_truncated_video_is_reported_not_silently_shortened(self):
        raw = self.video.read_bytes()
        cut = self.tmp / "cut.mp4"
        cut.write_bytes(raw[: len(raw) // 2])
        with self.assertRaises(vf.VideoError):
            list(vf.iter_frames(cut, strict=True))


class MetadataTests(VideoFramesTestCase):
    def test_info_as_dict_contains_the_fields_metadata_needs(self):
        info = vf.open_video_info(self.video).as_dict()
        for key in ("filename", "width", "height", "fps", "reported_frame_count"):
            self.assertIn(key, info)
        self.assertEqual(info["filename"], "clip.mp4")


if __name__ == "__main__":
    unittest.main(verbosity=2)
