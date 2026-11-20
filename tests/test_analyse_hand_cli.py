"""Hand tracking and motion command-line entry point (phase H2). Synthetic data and fake detectors only.

Each test builds a pose data session, adds the hand stream (H1), then runs the H2 analysis.
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

from extraction import analyse_hand_session as acli  # noqa: E402
import pose_schema as s2  # noqa: E402
import test_build_hand_cli as tbh  # noqa: E402  (H1 CLI base and scripted hand detector)
import test_hand_extractor as th  # noqa: E402


N = 8  # frames in the synthetic video (test_build_pose_cli)


class SteadyHands(tbh.ScriptedHands):
    """A right hand that moves 6 px per frame across the image, visible in every frame."""

    def detect(self, frame_bgr, timestamp_ms):
        index = self.calls
        self.calls += 1
        if self.fail_on is not None and index == self.fail_on:
            raise RuntimeError("boom")
        x = 150 + 6 * index
        return th.output([th.hand(x / 640.0, 0.5)], [th.world_hand()], [("Right", 0.95)])


class AnalyseCase(tbh.HandCliTestCase):
    def build_hands(self):
        self.build_pose()
        code, _o, err = self.run_hands(detector=SteadyHands())
        self.assertEqual(code, 0, err)

    def run_analysis(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        argv = [self.person, self.session, *extra]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = acli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def arrays(self):
        names = ["hands/left_hand.npy", "hands/right_hand.npy", "hands/right_hand_world.npy", "facial/landmarks.npy",
                 "body/pose_landmarks.npy"]
        return {n: np.load(self.dataset / n, allow_pickle=False) for n in names}


class SuccessTests(AnalyseCase):
    def test_publishes_tracks_and_motion_and_keeps_all_streams(self):
        self.build_hands()
        before_arrays = self.arrays()
        before_face = self.face_hashes()
        code, _out, err = self.run_analysis()
        self.assertEqual(code, 0, err)
        self.assertEqual(s2.session_layout(self.dataset), "pose")
        for name, arr in before_arrays.items():
            self.assertTrue(np.array_equal(arr, np.load(self.dataset / name), equal_nan=True), name)
        self.assertEqual(self.face_hashes(), before_face)

    def test_tables_have_the_declared_columns_and_one_track(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        tracks = pd.read_csv(self.dataset / "hands" / "tracks.csv")
        motion = pd.read_csv(self.dataset / "hands" / "motion_metrics.csv")
        self.assertEqual(list(tracks.columns), list(s2.HAND_TRACK_COLUMNS))
        self.assertEqual(list(motion.columns), list(s2.HAND_MOTION_COLUMNS))
        self.assertEqual(tracks["track_id"].nunique(), 1)
        self.assertEqual(tracks.iloc[0]["event"], "created")
        self.assertEqual(len(motion), N)  # one linked observation per frame

    def test_motion_is_in_pixels_per_second_for_a_steady_hand(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        motion = pd.read_csv(self.dataset / "hands" / "motion_metrics.csv")
        speeds = motion["velocity_px_per_s"].dropna()
        self.assertGreater(len(speeds), 0)
        # 6 px per frame in normalised x = 6 / 640 * width(64 in the synthetic video) px, at 10 fps
        expected = (6 / 640.0) * 64 * 10
        self.assertTrue(np.allclose(speeds, expected, rtol=0.05), speeds.tolist())
        self.assertEqual(motion["motion_reason"].iloc[0], "first_observation")

    def test_metadata_and_quality_record_the_method(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        meta = json.loads((self.dataset / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("tracking", meta["streams"]["hands"])
        self.assertIn("max_gap_frames", meta["configuration"]["hands"]["tracking"])
        self.assertIn("image pixels", meta["coordinate_conventions"]["hand_motion"])
        quality = json.loads((self.dataset / "quality" / "tracking_quality.json").read_text(encoding="utf-8"))
        self.assertEqual(quality["hands"]["tracking"]["smoothing"], "none")

    def test_every_motion_row_has_a_matching_track_event(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        tracks = pd.read_csv(self.dataset / "hands" / "tracks.csv")
        motion = pd.read_csv(self.dataset / "hands" / "motion_metrics.csv")
        linked = set(zip(tracks[tracks["event"].isin(["created", "matched", "reacquired"])]["track_id"],
                         tracks[tracks["event"].isin(["created", "matched", "reacquired"])]["frame"]))
        self.assertTrue(set(zip(motion["track_id"], motion["frame"])) <= linked)


class RefusalTests(AnalyseCase):
    def test_without_the_hand_stream_nothing_is_written(self):
        self.build_pose()
        code, _o, err = self.run_analysis()
        self.assertEqual(code, 2)
        self.assertIn("hand stream", err)
        self.assertFalse((self.dataset / "hands" / "tracks.csv").exists())

    def test_second_run_is_refused_without_regenerate(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        code, _o, err = self.run_analysis()
        self.assertEqual(code, 2)
        self.assertIn("--regenerate", err)

    def test_regenerate_replaces_the_tracking(self):
        self.build_hands()
        self.assertEqual(self.run_analysis()[0], 0)
        self.assertEqual(self.run_analysis("--regenerate")[0], 0)

    def test_invalid_names_are_refused(self):
        self.build_hands()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err):
            code = acli.main(["demo_person", "bad"])
        self.assertEqual(code, 2)


class FailureTests(AnalyseCase):
    def test_face_data_change_is_reported_and_rolled_back(self):
        self.build_hands()
        face = self.dataset / "tracking_quality.json"
        original = self.dataset / "hands" / "tracks.csv"
        self.assertFalse(original.exists())
        orig_bytes = face.read_bytes()

        import extraction.analyse_hand_session as mod
        real = mod.analyse_hand_payload

        def tampering(session):
            face.write_bytes(orig_bytes + b" ")
            return real(session)

        mod.analyse_hand_payload = tampering
        try:
            code, _o, err = self.run_analysis()
        finally:
            mod.analyse_hand_payload = real
        self.assertEqual(code, 4)
        self.assertIn("tracking_quality.json", err)
        self.assertFalse(original.exists())  # the previous pose data (without tracking) is restored


if __name__ == "__main__":
    unittest.main(verbosity=2)
