"""Integration tests: Pose output and V1 facial data written into a V2 session (synthetic data only)."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import pose_extractor as pe  # noqa: E402
from extraction import v2_session_builder as sb  # noqa: E402
import schema_v2 as s2  # noqa: E402
import session_writer_v2 as w  # noqa: E402

W, H, FPS, N = 64, 48, 10, 8
V1_FILES = {
    "facial_data.json": None,  # written by the fixture from synthetic frames
    "facial_blendshapes.csv": b"frame,timestamp_ms,face_detected\n0,0,True\n",
    "motion_metrics.csv": b"session_id,frame\nsession-001,0\n",
    "blink_events.csv": b"session_id\n",
    "tracking_quality.json": b"{}",
    "calibration.json": b"{}",
}


def face_facial_frames(n=N, missing_face=()):
    """Synthetic frames in the same shape as the V1 extractor writes."""
    frames = []
    for i in range(n):
        detected = i not in missing_face
        frame = {"frame": i, "timestamp_ms": round(i * 1000 / FPS), "face_detected": detected}
        if detected:
            frame["landmarks"] = [{"id": k, "x": 0.4 + 0.001 * k, "y": 0.5, "z": 0.0} for k in range(478)]
            frame["blendshapes"] = {"jawOpen": 0.1, "eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0}
            frame["transformation_matrix"] = np.eye(4).tolist()
        frames.append(frame)
    return frames


class FakePoseDetector:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = 0

    def detect(self, frame_bgr, timestamp_ms):
        index = self.calls
        self.calls += 1
        if index in self.missing:
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        lm = [SimpleNamespace(x=0.1 * k, y=0.2, z=0.0, visibility=0.9, presence=0.9) for k in range(33)]
        return SimpleNamespace(pose_landmarks=[lm], pose_world_landmarks=[lm])


def pose_result(n=N, missing=()):
    frames = [(i, np.zeros((H, W, 3), np.uint8)) for i in range(n)]
    ts, source = s2.frame_timestamps_ms(n, fps=FPS)
    return pe.extract_pose_from_frames(frames, ts, FakePoseDetector(missing=missing),
                                       source_filename="session-001.mp4", timestamp_source=source)


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_builder_"))
        self.session = self.tmp / "datasets" / "demo_person" / "session-001"
        self.session.mkdir(parents=True)
        self.write_face()
        self.face_hashes = self.hash_face()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_face(self, missing_face=()):
        for name, data in V1_FILES.items():
            if name == "facial_data.json":
                data = json.dumps(face_facial_frames(missing_face=missing_face)).encode("utf-8")
            (self.session / name).write_bytes(data)

    def hash_face(self):
        return {name: hashlib.sha256((self.session / name).read_bytes()).hexdigest() for name in V1_FILES}

    def assert_face_unchanged(self):
        for name, digest in self.face_hashes.items():
            self.assertEqual(hashlib.sha256((self.session / name).read_bytes()).hexdigest(), digest,
                             f"version 1 file changed: {name}")

    def build(self, pose=None, **kwargs):
        return sb.build_payload(
            dataset_dir=self.session,
            pose=pose if pose is not None else pose_result(),
            fps=FPS,
            person="demo_person",
            session_id="session-001",
            video_filename="session-001.mp4",
            video_size=(W, H),
            **kwargs,
        )


class PayloadTests(SessionTestCase):
    def test_payload_passes_the_schema(self):
        payload = self.build()
        self.assertEqual(w.prevalidate(payload), N)

    def test_pose_arrays_come_from_the_extractor(self):
        pose = pose_result()
        payload = self.build(pose=pose)
        np.testing.assert_array_equal(payload.arrays["body/pose_landmarks.npy"], pose.image)
        np.testing.assert_array_equal(payload.arrays["body/pose_world_landmarks.npy"], pose.world)

    def test_facial_landmarks_come_from_the_face_json(self):
        payload = self.build()
        face = payload.arrays["facial/landmarks.npy"]
        self.assertEqual(face.shape, (N, 478, 3))
        self.assertAlmostEqual(float(face[0, 3, 0]), 0.403, places=5)

    def test_hands_are_recorded_as_an_absent_stream(self):
        payload = self.build()
        self.assertTrue(np.isnan(payload.arrays["hands/left_hand.npy"]).all())
        self.assertTrue(np.isnan(payload.arrays["hands/right_hand.npy"]).all())
        self.assertEqual(payload.metadata["streams"]["hands"]["status"], "absent")
        self.assertFalse(any(row["left_hand_detected"] or row["right_hand_detected"] for row in payload.timeline_rows))

    def test_timeline_flags_follow_the_detections(self):
        pose = pose_result(missing=(2,))
        payload = self.build(pose=pose)
        flags = [row["pose_detected"] for row in payload.timeline_rows]
        self.assertFalse(flags[2])
        self.assertTrue(flags[1] and flags[3])
        self.assertEqual(len(payload.timeline_rows), N)

    def test_metadata_records_models_and_configuration(self):
        meta = self.build().metadata
        # schema_v2 stores SHA-256 in lowercase, as hashlib produces it
        self.assertEqual(meta["models"]["pose_full"]["sha256"],
                         "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad")
        self.assertIn(meta["models"]["pose_full"]["licence_status"], ("verified", "to verify"))
        self.assertIn("not cleared", meta["models"]["pose_full"]["licence_note"])
        self.assertEqual(meta["configuration"]["pose"]["num_poses"], 1)
        self.assertEqual(meta["timeline"]["timestamp_source"], "fps_derived")
        self.assertEqual(meta["streams"]["body"]["status"], "complete")


class MismatchTests(SessionTestCase):
    def test_frame_count_mismatch_is_refused(self):
        with self.assertRaisesRegex(sb.BuildError, "frame counts"):
            self.build(pose=pose_result(n=N - 1))

    def test_timestamp_mismatch_is_refused(self):
        frames = [(i, np.zeros((H, W, 3), np.uint8)) for i in range(N)]
        bad_ts = np.arange(N, dtype=np.int64) * 101  # not the video's timestamps
        pose = pe.extract_pose_from_frames(frames, bad_ts, FakePoseDetector(), source_filename="x.mp4")
        with self.assertRaisesRegex(sb.BuildError, "timestamps"):
            self.build(pose=pose)

    def test_refusal_writes_nothing(self):
        with self.assertRaises(sb.BuildError):
            self.build(pose=pose_result(n=N - 1))
        self.assertFalse((self.session / "metadata.json").exists())
        self.assertFalse(any(p.name.startswith(".v2") for p in self.session.iterdir()))
        self.assert_face_unchanged()


class WriteIntegrationTests(SessionTestCase):
    def test_full_session_writes_verifies_and_leaves_face_untouched(self):
        payload = self.build(pose=pose_result(missing=(4,)))
        manifest = w.write_v2_session(self.session, payload)
        self.assertEqual(w.verify_manifest(self.session, manifest, check_unexpected=True), [])
        self.assertEqual(s2.session_layout(self.session), "v2")
        self.assert_face_unchanged()

    def test_pose_on_disk_keeps_nan_rows_for_missing_detections(self):
        w.write_v2_session(self.session, self.build(pose=pose_result(missing=(4,))))
        arr = np.load(self.session / "body" / "pose_landmarks.npy")
        self.assertEqual(arr.shape, (N, 33, 5))
        self.assertTrue(np.isnan(arr[4]).all())
        self.assertTrue(np.isfinite(arr[3]).all())

    def test_interrupted_publication_is_incomplete_and_recoverable(self):
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, self.build(), fail_at="publish")
        self.assertEqual(s2.session_layout(self.session), "incomplete_v2")
        w.recover_incomplete_v2(self.session)
        self.assertEqual(s2.session_layout(self.session), "v1")
        self.assert_face_unchanged()
        w.write_v2_session(self.session, self.build())
        self.assertEqual(s2.session_layout(self.session), "v2")

    def test_rollback_after_regeneration_restores_the_previous_pose_data(self):
        w.write_v2_session(self.session, self.build())
        w.write_v2_session(self.session, self.build(), regenerate=True)
        w.rollback_v2(self.session)
        self.assertEqual(s2.session_layout(self.session), "v2")  # previous version 2 restored
        self.assert_face_unchanged()

    def test_rollback_of_the_only_pose_data_leaves_face_data(self):
        w.write_v2_session(self.session, self.build())
        w.rollback_v2(self.session)
        self.assertEqual(s2.session_layout(self.session), "v1")
        self.assert_face_unchanged()


if __name__ == "__main__":
    unittest.main(verbosity=2)
