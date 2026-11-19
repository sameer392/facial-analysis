"""Tests for the Pose-to-V2 command-line entry point. Synthetic data and a fake detector only."""

import contextlib
import hashlib
import io
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
sys.path.insert(0, str(ROOT / "tests"))

from extraction import build_v2_session as cli  # noqa: E402
from extraction import model_registry as mr  # noqa: E402
import project_paths as pp  # noqa: E402
import schema_v2 as s2  # noqa: E402
import test_v2_session_builder as tvb  # noqa: E402  (synthetic V1 frames and fake detector)

W, H, FPS, N = 64, 48, 10, 8
V1_NAMES = ["facial_data.json", "facial_blendshapes.csv", "motion_metrics.csv",
            "blink_events.csv", "tracking_quality.json", "calibration.json"]


class FakeDetector(tvb.FakePoseDetector):
    def __init__(self, *args, on_detect=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = 0
        self.on_detect = on_detect

    def detect(self, frame_bgr, timestamp_ms):
        if self.on_detect:
            self.on_detect(self.calls)
        return super().detect(frame_bgr, timestamp_ms)

    def close(self):
        self.closed += 1


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="build_v2_cli_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        self.person, self.session = "demo_person", "session-001"
        self.video = pp.session_video(self.person, self.session)
        self.video.parent.mkdir(parents=True)
        self._write_video(self.video)
        self.dataset = pp.session_dataset_dir(self.person, self.session)
        self.dataset.mkdir(parents=True)
        import json
        (self.dataset / "facial_data.json").write_text(json.dumps(tvb.v1_facial_frames(N)), encoding="utf-8")
        for name in V1_NAMES[1:]:
            (self.dataset / name).write_bytes(b"v1 " + name.encode())
        (self.dataset / "frames").mkdir()
        (self.dataset / "frames" / "frame_000000.jpg").write_bytes(b"jpg")
        self.models = self.tmp / "models"
        self.models.mkdir()

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_video(self, path):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i in range(N):
            writer.write(np.full((H, W, 3), i * 10 % 256, np.uint8))
        writer.release()

    def run_cli(self, *extra, detector=None, models_dir=None):
        argv = [self.person, self.session, *extra]
        if models_dir is not None:
            argv += ["--models-dir", str(models_dir)]
        out, err = io.StringIO(), io.StringIO()
        factory = (lambda: detector) if detector is not None else None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv, detector_factory=factory)
        return code, out.getvalue(), err.getvalue()

    def v1_hashes(self):
        names = V1_NAMES + ["frames/frame_000000.jpg"]
        return {n: hashlib.sha256((self.dataset / n).read_bytes()).hexdigest() for n in names}


class SuccessTests(CliTestCase):
    def test_writes_a_complete_v2_session_and_leaves_v1_untouched(self):
        before = self.v1_hashes()
        det = FakeDetector()
        code, out, err = self.run_cli(detector=det)
        self.assertEqual(code, 0, err)
        self.assertEqual(s2.session_layout(self.dataset), "v2")
        self.assertEqual(self.v1_hashes(), before)
        self.assertIn("published", out.lower())

    def test_detector_is_closed_after_success(self):
        det = FakeDetector()
        self.run_cli(detector=det)
        self.assertEqual(det.closed, 1)

    def test_second_run_is_refused_without_regenerate(self):
        self.run_cli(detector=FakeDetector())
        before = (self.dataset / "metadata.json").read_bytes()
        code, _, err = self.run_cli(detector=FakeDetector())
        self.assertEqual(code, 2)
        self.assertIn("--regenerate-v2", err)
        self.assertEqual((self.dataset / "metadata.json").read_bytes(), before)

    def test_regenerate_flag_replaces_the_version_two_dataset(self):
        self.run_cli(detector=FakeDetector())
        code, _, err = self.run_cli("--regenerate-v2", detector=FakeDetector())
        self.assertEqual(code, 0, err)
        self.assertEqual(s2.session_layout(self.dataset), "v2")
        self.assertTrue(any(p.name.startswith(".v2-previous") for p in self.dataset.iterdir()))


class InputFailureTests(CliTestCase):
    def test_invalid_person_name_is_refused(self):
        argv_code = cli.main(["Bad Name", self.session], detector_factory=lambda: FakeDetector())
        self.assertEqual(argv_code, 2)

    def test_invalid_session_name_is_refused(self):
        self.assertEqual(cli.main([self.person, "presenter-01"], detector_factory=lambda: FakeDetector()), 2)

    def test_missing_video_writes_nothing(self):
        self.video.unlink()
        det = FakeDetector()
        code, _, err = self.run_cli(detector=det)
        self.assertEqual(code, 2)
        self.assertIn("video", err.lower())
        self.assertEqual(s2.session_layout(self.dataset), "v1")
        self.assertEqual(det.closed, 0)  # detector never created

    def test_missing_version_one_facial_data_writes_nothing(self):
        (self.dataset / "facial_data.json").unlink()
        code, _, err = self.run_cli(detector=FakeDetector())
        self.assertEqual(code, 2)
        self.assertIn("extract_faces", err)
        self.assertEqual(s2.session_layout(self.dataset), "none")

    def test_corrupt_video_reports_an_error_and_closes_the_detector(self):
        self.video.write_bytes(b"not a video" * 40)
        det = FakeDetector()
        code, _, err = self.run_cli(detector=det)
        self.assertEqual(code, 1)
        self.assertIn("video", err.lower())
        self.assertFalse((self.dataset / "metadata.json").exists())


class ModelGateTests(CliTestCase):
    def test_model_with_wrong_hash_is_refused_before_loading(self):
        (self.models / "pose_landmarker_full.task").write_bytes(b"not the real model")
        code, _, err = self.run_cli(models_dir=self.models)
        self.assertEqual(code, 3)
        self.assertIn("SHA-256", err)
        self.assertFalse((self.dataset / "metadata.json").exists())

    def test_missing_model_is_refused(self):
        code, _, err = self.run_cli(models_dir=self.models)
        self.assertEqual(code, 3)
        self.assertIn("not found", err)


class FailurePathTests(CliTestCase):
    def test_exception_during_detection_closes_the_detector_and_writes_nothing(self):
        def boom(_):
            raise RuntimeError("detector failed")
        det = FakeDetector(on_detect=boom)
        code, _, err = self.run_cli(detector=det)
        self.assertEqual(code, 1)
        self.assertIn("detector failed", err)
        self.assertEqual(det.closed, 1)
        self.assertEqual(s2.session_layout(self.dataset), "v1")

    def test_version_one_file_changed_during_run_is_reported(self):
        def tamper(index):
            if index == 0:
                (self.dataset / "calibration.json").write_bytes(b"changed during run")
        code, _, err = self.run_cli(detector=FakeDetector(on_detect=tamper))
        self.assertEqual(code, 4)
        self.assertIn("calibration.json", err)

    def test_frame_count_mismatch_is_reported_without_publishing(self):
        det = FakeDetector()
        (self.dataset / "facial_data.json").write_text(
            __import__("json").dumps(tvb.v1_facial_frames(N - 2)), encoding="utf-8")
        code, _, err = self.run_cli(detector=det)
        self.assertEqual(code, 1)
        self.assertIn("frame counts", err)
        self.assertFalse((self.dataset / "metadata.json").exists())
        self.assertEqual(det.closed, 1)


class ArgumentTests(unittest.TestCase):
    def test_help_exits_cleanly(self):
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_regenerate_flag_is_optional(self):
        parser = cli.build_parser()
        self.assertFalse(parser.parse_args(["p", "session-001"]).regenerate_v2)
        self.assertTrue(parser.parse_args(["p", "session-001", "--regenerate-v2"]).regenerate_v2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
