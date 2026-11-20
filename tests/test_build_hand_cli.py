"""Hand build command-line entry point (phase H1). Synthetic video, fake pose and hand detectors only.

Each test first builds a complete pose data session with the existing pose CLI, then adds the hand
stream. The face and body arrays must come out unchanged.
"""

import contextlib
import hashlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

from extraction import build_hand_session as hcli  # noqa: E402
from extraction import build_pose_session as pcli  # noqa: E402
import pose_schema as s2  # noqa: E402
import test_build_pose_cli as tbc  # noqa: E402  (CLI test base: synthetic video, face data, folders)
import test_hand_extractor as th  # noqa: E402  (synthetic hand output)


class ScriptedHands:
    """Hand detector: right hand on even frames, a low-confidence left hand on frame 3, nothing otherwise."""

    def __init__(self, fail_on=None):
        self.calls = 0
        self.closed = 0
        self.fail_on = fail_on

    def detect(self, frame_bgr, timestamp_ms):
        index = self.calls
        self.calls += 1
        if self.fail_on is not None and index == self.fail_on:
            raise RuntimeError("boom during hand detection")
        if index == 3:
            return th.output([th.hand()], [th.world_hand()], [("Left", 0.4)])
        if index % 2 == 0:
            return th.output([th.hand(0.6, 0.4)], [th.world_hand()], [("Right", 0.95)])
        return th.output()

    def close(self):
        self.closed += 1


class HandCliTestCase(tbc.CliTestCase):
    def build_pose(self):
        code, _out, err = self.run_cli(detector=tbc.FakeDetector())
        self.assertEqual(code, 0, err)

    def run_hands(self, *extra, detector=None, models_dir=None):
        argv = [self.person, self.session, *extra]
        if models_dir is not None:
            argv += ["--models-dir", str(models_dir)]
        out, err = io.StringIO(), io.StringIO()
        factory = (lambda: detector) if detector is not None else None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hcli.main(argv, detector_factory=factory)
        return code, out.getvalue(), err.getvalue()

    def pose_arrays(self):
        names = ["facial/landmarks.npy", "facial/transformation_matrices.npy",
                 "body/pose_landmarks.npy", "body/pose_world_landmarks.npy"]
        return {n: np.load(self.dataset / n, allow_pickle=False) for n in names}

    def meta(self):
        import json
        return json.loads((self.dataset / "metadata.json").read_text(encoding="utf-8"))


class SuccessTests(HandCliTestCase):
    def test_adds_the_hand_stream_and_keeps_face_and_body_unchanged(self):
        self.build_pose()
        before_arrays = self.pose_arrays()
        before_face = self.face_hashes()
        det = ScriptedHands()
        code, out, err = self.run_hands(detector=det)
        self.assertEqual(code, 0, err)
        self.assertEqual(s2.session_layout(self.dataset), "pose")
        after = self.pose_arrays()
        for name, arr in before_arrays.items():
            self.assertTrue(np.array_equal(arr, after[name], equal_nan=True), name)
        self.assertEqual(self.face_hashes(), before_face)
        self.assertEqual(det.closed, 1)

    def test_writes_hand_arrays_status_file_and_metadata(self):
        self.build_pose()
        code, _out, err = self.run_hands(detector=ScriptedHands())
        self.assertEqual(code, 0, err)
        for rel in ("hands/left_hand.npy", "hands/right_hand.npy", "hands/left_hand_world.npy",
                    "hands/right_hand_world.npy", "hands/handedness.csv", "hands/hand_status.csv"):
            self.assertTrue((self.dataset / rel).is_file(), rel)
        status = pd.read_csv(self.dataset / "hands" / "hand_status.csv")
        self.assertEqual(len(status), tbc.N)
        self.assertEqual(status["right_status"].tolist(), ["valid", "not_detected", "valid", "not_detected",
                                                            "valid", "not_detected", "valid", "not_detected"])
        self.assertEqual(status["left_status"].iloc[3], "low_confidence")
        meta = self.meta()
        self.assertEqual(meta["streams"]["hands"]["status"], "complete")
        self.assertEqual(meta["streams"]["hands"]["detected_frames"], 5)
        self.assertEqual(meta["models"]["hand"]["licence_status"], "to verify")
        self.assertIn("not cleared", meta["models"]["hand"]["licence_note"])
        self.assertIn("hand_image_landmarks", meta["coordinate_conventions"])

    def test_timeline_flags_follow_the_hand_status(self):
        self.build_pose()
        self.run_hands(detector=ScriptedHands())
        timeline = pd.read_csv(self.dataset / "timeline.csv")
        self.assertEqual(timeline["right_hand_detected"].astype(int).tolist(), [1, 0, 1, 0, 1, 0, 1, 0])
        self.assertEqual(timeline["left_hand_detected"].astype(int).tolist(), [0, 0, 0, 1, 0, 0, 0, 0])

    def test_regenerate_replaces_the_hand_stream(self):
        self.build_pose()
        self.assertEqual(self.run_hands(detector=ScriptedHands())[0], 0)
        code, _out, err = self.run_hands("--regenerate", detector=ScriptedHands())
        self.assertEqual(code, 0, err)
        self.assertEqual(self.meta()["streams"]["hands"]["status"], "complete")


class RefusalTests(HandCliTestCase):
    def test_without_pose_data_nothing_is_written(self):
        code, _out, err = self.run_hands(detector=ScriptedHands())
        self.assertEqual(code, 2)
        self.assertIn("pose", err)
        self.assertFalse((self.dataset / "hands").exists())

    def test_second_run_is_refused_without_regenerate(self):
        self.build_pose()
        self.assertEqual(self.run_hands(detector=ScriptedHands())[0], 0)
        det = ScriptedHands()
        code, _out, err = self.run_hands(detector=det)
        self.assertEqual(code, 2)
        self.assertIn("--regenerate", err)
        self.assertEqual(det.calls, 0)

    def test_invalid_session_name_is_refused(self):
        self.build_pose()
        code = hcli.main(["demo_person", "bad-name"], detector_factory=lambda: ScriptedHands())
        self.assertEqual(code, 2)

    def test_missing_model_is_refused_with_exit_3(self):
        self.build_pose()
        code, _out, _err = self.run_hands(models_dir=self.models)
        self.assertEqual(code, 3)
        self.assertFalse((self.dataset / "hands" / "hand_status.csv").exists())


class FailureTests(HandCliTestCase):
    def test_detection_error_leaves_the_pose_data_intact(self):
        self.build_pose()
        before = self.pose_arrays()
        det = ScriptedHands(fail_on=2)
        code, _out, err = self.run_hands(detector=det)
        self.assertEqual(code, 1)
        self.assertIn("boom", err)
        self.assertEqual(det.closed, 1)
        self.assertEqual(s2.session_layout(self.dataset), "pose")
        self.assertEqual(self.meta()["streams"]["hands"]["status"], "absent")
        after = self.pose_arrays()
        for name, arr in before.items():
            self.assertTrue(np.array_equal(arr, after[name], equal_nan=True), name)

    def test_corrupt_video_reports_an_error(self):
        self.build_pose()
        self.video.write_bytes(b"not a video")
        code, _out, _err = self.run_hands(detector=ScriptedHands())
        self.assertEqual(code, 1)
        self.assertEqual(self.meta()["streams"]["hands"]["status"], "absent")

    def test_face_data_change_during_the_run_is_reported(self):
        self.build_pose()
        face = self.dataset / "tracking_quality.json"

        class Tamper(ScriptedHands):
            def detect(self, frame_bgr, timestamp_ms):
                if self.calls == 0:
                    face.write_bytes(face.read_bytes() + b" ")
                return super().detect(frame_bgr, timestamp_ms)

        code, _out, err = self.run_hands(detector=Tamper())
        self.assertEqual(code, 4)
        self.assertIn("tracking_quality.json", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
