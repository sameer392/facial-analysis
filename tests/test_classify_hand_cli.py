"""Gesture command-line entry point (phase H3). Synthetic video, synthetic hands, no personal data."""

import contextlib
import io
import json
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

from extraction import classify_hand_gestures as gcli  # noqa: E402
import pose_schema as s2  # noqa: E402
import test_analyse_hand_cli as tah  # noqa: E402  (H2 CLI base)
import test_gesture_rules as tg  # noqa: E402  (synthetic hand geometry)
import test_hand_extractor as th  # noqa: E402

W, H = tah.tbh.tbc.W, tah.tbh.tbc.H   # synthetic video frame size (64 x 48)
N = tah.N


def _pixels_to_landmarks(pts):
    """Normalised MediaPipe-style landmarks from pixel points, in the test frame size."""
    return [SimpleNamespace(x=float(x / W), y=float(y / H), z=0.0) for x, y in pts]


class Pointing:
    """Right hand held in the pointing shape for every frame of the 8-frame synthetic video."""

    def __init__(self):
        self.calls = 0
        self.closed = 0

    def detect(self, frame_bgr, timestamp_ms):
        index = self.calls
        self.calls += 1
        states = tg.POINTING
        pts = tg.make_hand(states, scale=12.0, centre=(32.0, 24.0))
        return th.output([_pixels_to_landmarks(pts)], [th.world_hand()], [("Right", 0.95)])

    def close(self):
        self.closed += 1


class GestureCase(tah.AnalyseCase):
    def build_all(self):
        self.build_hands_with(Pointing())
        self.assertEqual(self.run_analysis()[0], 0)

    def build_hands_with(self, det):
        self.build_pose()
        code, _o, err = self.run_hands(detector=det)
        self.assertEqual(code, 0, err)

    def run_gestures(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gcli.main([self.person, self.session, *extra])
        return code, out.getvalue(), err.getvalue()


class SuccessTests(GestureCase):
    def test_publishes_observations_events_and_summary(self):
        self.build_all()
        code, _o, err = self.run_gestures()
        self.assertEqual(code, 0, err)
        obs = pd.read_csv(self.dataset / "hands" / "gesture_observations.csv")
        events = pd.read_csv(self.dataset / "hands" / "gesture_events.csv")
        self.assertEqual(list(obs.columns), list(s2.HAND_GESTURE_OBS_COLUMNS))
        self.assertEqual(list(events.columns), list(s2.HAND_GESTURE_EVENT_COLUMNS))
        self.assertEqual(len(obs), N)
        self.assertEqual(obs["frame_class"].tolist(), ["pointing"] * N)
        self.assertEqual(events["gesture"].tolist(), ["pointing"])
        self.assertEqual(events["supporting_observations"].tolist(), [N])
        self.assertEqual(events["end_reason"].tolist(), ["end_of_track"])
        self.assertTrue((obs["event_id"].notna()).all())

    def test_summary_is_recorded_and_metadata_describes_the_rules(self):
        self.build_all()
        self.run_gestures()
        quality = json.loads((self.dataset / "quality" / "tracking_quality.json").read_text(encoding="utf-8"))
        summary = quality["hands"]["gestures"]
        self.assertEqual(summary["events"], 1)
        self.assertEqual(summary["events_per_gesture"]["pointing"], 1)
        meta = json.loads((self.dataset / "metadata.json").read_text(encoding="utf-8"))
        self.assertIn("calibration", meta["configuration"]["hands"]["gestures"])
        self.assertIn("not a probability", meta["coordinate_conventions"]["hand_gesture"])

    def test_earlier_hand_outputs_are_unchanged(self):
        self.build_all()
        keep = ["hands/tracks.csv", "hands/motion_metrics.csv", "hands/hand_status.csv",
                "hands/left_hand.npy", "hands/right_hand.npy", "facial/landmarks.npy"]
        import hashlib
        before = {k: hashlib.sha256((self.dataset / k).read_bytes()).hexdigest() for k in keep}
        self.assertEqual(self.run_gestures()[0], 0)
        after = {k: hashlib.sha256((self.dataset / k).read_bytes()).hexdigest() for k in keep}
        self.assertEqual(before, after)
        self.assertEqual(s2.session_layout(self.dataset), "pose")


class RefusalTests(GestureCase):
    def test_without_tracking_nothing_is_written(self):
        self.build_pose()
        code, _o, err = self.run_gestures()
        self.assertEqual(code, 2)
        self.assertIn("tracking", err)
        self.assertFalse((self.dataset / "hands" / "gesture_events.csv").exists())

    def test_second_run_is_refused_without_regenerate(self):
        self.build_all()
        self.assertEqual(self.run_gestures()[0], 0)
        code, _o, err = self.run_gestures()
        self.assertEqual(code, 2)
        self.assertIn("--regenerate", err)

    def test_regenerate_replaces_the_gestures(self):
        self.build_all()
        self.assertEqual(self.run_gestures()[0], 0)
        self.assertEqual(self.run_gestures("--regenerate")[0], 0)

    def test_invalid_names_are_refused(self):
        self.build_all()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gcli.main(["demo_person", "x"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
