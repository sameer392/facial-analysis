"""Tests for the version 2 session writer (synthetic data only, temporary folders)."""

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import pose_schema as s2  # noqa: E402
import pose_writer as w  # noqa: E402

T = 6
MODEL_SHA = hashlib.sha256(b"synthetic model").hexdigest()
V1_FILES = {
    "facial_data.json": b'[{"frame": 0}]',
    "facial_blendshapes.csv": b"frame,timestamp_ms,face_detected\n0,0,True\n",
    "motion_metrics.csv": b"session_id,frame\nsession-001,0\n",
    "blink_events.csv": b"session_id\n",
    "tracking_quality.json": b"{}",
    "calibration.json": b"{}",
}


def make_payload(n=T, seed=0) -> w.PoseDataPayload:
    rng = np.random.default_rng(seed)
    arrays = {
        "facial/landmarks.npy": rng.random((n, 478, 3)).astype(np.float32),
        "facial/transformation_matrices.npy": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        "body/pose_landmarks.npy": np.full((n, 33, 5), np.nan, dtype=np.float32),
        "body/pose_world_landmarks.npy": np.full((n, 33, 5), np.nan, dtype=np.float32),
        "hands/left_hand.npy": np.full((n, 21, 3), np.nan, dtype=np.float32),
        "hands/right_hand.npy": np.full((n, 21, 3), np.nan, dtype=np.float32),
    }
    arrays["body/pose_landmarks.npy"][0, :, 0] = 0.5
    timeline = [{
        "frame": i,
        "timestamp_ms": round(i * 1000 / 30),
        "face_detected": True,
        "pose_detected": i == 0,
        "left_hand_detected": False,
        "right_hand_detected": False,
    } for i in range(n)]
    handedness = [{
        "frame": i, "timestamp_ms": round(i * 1000 / 30),
        "left_detected": False, "left_model_label": "", "left_score": None,
        "right_detected": False, "right_model_label": "", "right_score": None,
    } for i in range(n)]
    blend = pd.DataFrame(rng.random((n, len(s2.BLENDSHAPE_NAMES))), columns=list(s2.BLENDSHAPE_NAMES))
    blend.insert(0, "face_detected", True)
    blend.insert(0, "timestamp_ms", [round(i * 1000 / 30) for i in range(n)])
    blend.insert(0, "frame", range(n))
    metadata = {
        "schema_version": s2.SCHEMA_VERSION,
        "identity": {"person_id": "demo_person", "session_id": "session-001"},
        "source": {"video_filename": "session-001.mp4", "fps": 30.0, "frame_count": n},
        "timeline": {"reference": "start of video", "timestamp_unit": "milliseconds",
                     "timestamp_source": "fps_derived", "frame_index_base": 0},
        "extraction": {"software": "facial-analysis", "mediapipe_version": "1.2.3"},
        "models": {"face_landmarker": {"file": "face_landmarker.task", "sha256": MODEL_SHA,
                                       "licence_status": "to verify"}},
        "configuration": {"pose": {"num_poses": 1}, "hands": {"num_hands": 2}},
        "coordinate_conventions": {"image_landmarks": "normalised"},
        "streams": {"facial": {"status": "complete"}, "body": {"status": "partial"},
                    "hands": {"status": "absent"}},
        "outputs": {},
    }
    return w.PoseDataPayload(arrays=arrays, timeline_rows=timeline, handedness_rows=handedness,
                       blendshapes=blend, quality={"frames": n}, metadata=metadata)


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_writer_"))
        self.session = self.tmp / "datasets" / "demo_person" / "session-001"
        self.session.mkdir(parents=True)
        for name, data in V1_FILES.items():
            (self.session / name).write_bytes(data)
        (self.session / "frames").mkdir()
        (self.session / "frames" / "frame_000000.jpg").write_bytes(b"jpeg")
        self.face_hashes = self._hash_face()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _hash_face(self):
        names = list(V1_FILES) + ["frames/frame_000000.jpg"]
        return {name: hashlib.sha256((self.session / name).read_bytes()).hexdigest() for name in names}

    def assert_face_unchanged(self):
        for name, digest in self.face_hashes.items():
            path = self.session / name
            self.assertTrue(path.is_file(), f"v1 file removed: {name}")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest, f"v1 file changed: {name}")


class CreationTests(SessionTestCase):
    def test_all_expected_files_are_created(self):
        manifest = w.write_v2_session(self.session, make_payload())
        expected = {e["path"] for e in manifest["files"]}
        self.assertTrue({"facial/landmarks.npy", "facial/transformation_matrices.npy", "facial/blendshapes.csv",
                         "body/pose_landmarks.npy", "body/pose_world_landmarks.npy",
                         "hands/left_hand.npy", "hands/right_hand.npy", "hands/handedness.csv",
                         "timeline.csv", "quality/tracking_quality.json"} <= expected)
        for name in ("metadata.json", "manifest.json"):
            self.assertTrue((self.session / name).is_file())
        self.assertEqual(s2.session_layout(self.session), "v2")

    def test_array_shapes_and_dtypes_on_disk(self):
        w.write_v2_session(self.session, make_payload())
        for spec in s2.ARRAY_SPECS:
            with self.subTest(file=spec.relative_path):
                arr = np.load(self.session / spec.relative_path, allow_pickle=False)
                self.assertEqual(arr.shape, (T,) + spec.tail_shape)
                self.assertEqual(arr.dtype, np.float32)

    def test_missing_values_are_nan_on_disk(self):
        w.write_v2_session(self.session, make_payload())
        pose = np.load(self.session / "body" / "pose_landmarks.npy")
        self.assertTrue(np.isnan(pose[1]).all())
        self.assertTrue(np.isfinite(pose[0, :, 0]).all())

    def test_metadata_manifest_checksum_matches_the_manifest_file_bytes(self):
        w.write_v2_session(self.session, make_payload())
        meta = json.loads((self.session / "metadata.json").read_text(encoding="utf-8"))
        on_disk = hashlib.sha256((self.session / "manifest.json").read_bytes()).hexdigest()
        self.assertEqual(meta["outputs"]["manifest_sha256"], on_disk)

    def test_timeline_and_metadata_are_written(self):
        w.write_v2_session(self.session, make_payload())
        timeline = pd.read_csv(self.session / "timeline.csv")
        self.assertEqual(list(timeline.columns), list(w.TIMELINE_COLUMNS))
        self.assertEqual(len(timeline), T)
        meta = json.loads((self.session / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(s2.validate_metadata(meta), [])
        self.assertIn("manifest_sha256", meta["outputs"])

    def test_invalid_payload_is_rejected_before_anything_is_written(self):
        payload = make_payload()
        payload.arrays["body/pose_landmarks.npy"] = np.zeros((T, 32, 5), dtype=np.float32)
        with self.assertRaises(w.PoseDataValidationError):
            w.write_v2_session(self.session, payload)
        self.assertEqual([p.name for p in self.session.iterdir() if p.name.startswith(".v2")], [])
        self.assertFalse((self.session / "metadata.json").exists())


class ManifestTests(SessionTestCase):
    def setUp(self):
        super().setUp()
        self.manifest = w.write_v2_session(self.session, make_payload())

    def test_manifest_verifies(self):
        self.assertEqual(w.verify_manifest(self.session, self.manifest, check_unexpected=True), [])

    def test_manifest_records_size_checksum_shape_and_dtype(self):
        entry = next(e for e in self.manifest["files"] if e["path"] == "facial/landmarks.npy")
        self.assertEqual(entry["shape"], [T, 478, 3])
        self.assertEqual(entry["dtype"], "float32")
        self.assertEqual(entry["bytes"], (self.session / "facial" / "landmarks.npy").stat().st_size)
        self.assertEqual(len(entry["sha256"]), 64)

    def test_corrupted_file_is_detected(self):
        path = self.session / "facial" / "landmarks.npy"
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
        issues = w.verify_manifest(self.session, self.manifest)
        self.assertTrue(any("checksum mismatch: facial/landmarks.npy" in m for m in issues))

    def test_corrupted_manifest_checksum_is_detected(self):
        bad = json.loads(json.dumps(self.manifest))
        bad["files"][0]["sha256"] = "0" * 64
        self.assertTrue(w.verify_manifest(self.session, bad))

    def test_missing_file_is_detected(self):
        (self.session / "timeline.csv").unlink()
        issues = w.verify_manifest(self.session, self.manifest)
        self.assertIn("missing file: timeline.csv", issues)

    def test_unexpected_file_is_detected(self):
        (self.session / "hands" / "stray.npy").write_bytes(b"x")
        issues = w.verify_manifest(self.session, self.manifest, check_unexpected=True)
        self.assertTrue(any("unexpected file" in m and "stray.npy" in m for m in issues))

    def test_unsafe_manifest_paths_are_refused(self):
        bad = json.loads(json.dumps(self.manifest))
        bad["files"].append({"path": "../facial_data.json", "bytes": 1, "sha256": "0" * 64})
        self.assertTrue(w.verify_manifest(self.session, bad))


class InterruptionTests(SessionTestCase):
    def _interrupt(self, point):
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, make_payload(), fail_at=point)

    def test_interrupted_array_writing(self):
        self._interrupt("arrays")
        self.assertTrue(any(p.name.startswith(w.STAGING_PREFIX) for p in self.session.iterdir()))
        self.assertFalse((self.session / "metadata.json").exists())
        self.assert_face_unchanged()

    def test_interrupted_csv_writing(self):
        self._interrupt("csv")
        self.assertFalse((self.session / "metadata.json").exists())
        self.assert_face_unchanged()

    def test_interrupted_validation(self):
        self._interrupt("validate")
        self.assertFalse((self.session / "manifest.json").exists())
        self.assert_face_unchanged()

    def test_interrupted_manifest_creation(self):
        self._interrupt("manifest")
        self.assertFalse((self.session / "manifest.json").exists())
        self.assert_face_unchanged()

    def test_interrupted_publication_leaves_identifiable_incomplete_state(self):
        self._interrupt("publish")
        self.assertEqual(s2.session_layout(self.session), "incomplete_v2")
        self.assertTrue((self.session / "manifest.json").is_file())
        self.assert_face_unchanged()

    def test_interruption_before_metadata_is_incomplete(self):
        self._interrupt("metadata")
        self.assertFalse((self.session / "metadata.json").exists())
        self.assertEqual(s2.session_layout(self.session), "incomplete_v2")
        self.assert_face_unchanged()

    def test_incomplete_data_is_never_read_as_complete(self):
        self._interrupt("publish")
        self.assertNotEqual(s2.session_layout(self.session), "v2")

    def test_recovery_after_each_stage_returns_to_clean_state(self):
        for point in w.FAIL_POINTS:
            with self.subTest(point=point):
                self.setUp_clean()
                self._interrupt(point)
                w.recover_incomplete_v2(self.session)
                self.assertEqual([p.name for p in self.session.iterdir() if p.name.startswith(".v2")], [])
                self.assertFalse((self.session / "metadata.json").exists())
                self.assertFalse((self.session / "manifest.json").exists())
                self.assert_face_unchanged()
                self.assertEqual(s2.session_layout(self.session), "v1")

    def setUp_clean(self):
        for p in list(self.session.iterdir()):
            if p.name.startswith(".v2") or p.name in w.V2_ROOT_FILES or p.name in w.V2_DIRECTORIES:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

    def test_writing_after_recovery_succeeds(self):
        self._interrupt("publish")
        w.recover_incomplete_v2(self.session)
        w.write_v2_session(self.session, make_payload())
        self.assertEqual(s2.session_layout(self.session), "v2")
        self.assert_face_unchanged()


class RegenerationAndRollbackTests(SessionTestCase):
    def test_existing_pose_is_protected_by_default(self):
        w.write_v2_session(self.session, make_payload())
        with self.assertRaises(w.PoseDataExistsError):
            w.write_v2_session(self.session, make_payload(seed=1))

    def test_regeneration_refuses_when_existing_manifest_is_unreadable(self):
        w.write_v2_session(self.session, make_payload(seed=0))
        old = (self.session / "facial" / "landmarks.npy").read_bytes()
        (self.session / "manifest.json").write_text("{corrupt", encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.write_v2_session(self.session, make_payload(seed=1), regenerate=True)
        self.assertEqual((self.session / "facial" / "landmarks.npy").read_bytes(), old)

    def test_regeneration_replaces_and_keeps_previous(self):
        first = w.write_v2_session(self.session, make_payload(seed=0))
        second = w.write_v2_session(self.session, make_payload(seed=1), regenerate=True)
        self.assertNotEqual(first["files"][0]["sha256"], second["files"][0]["sha256"])
        previous = [p for p in self.session.iterdir() if p.name.startswith(w.PREVIOUS_PREFIX)]
        self.assertEqual(len(previous), 1)
        self.assertTrue((previous[0] / "metadata.json").is_file())
        self.assert_face_unchanged()

    def test_failed_regeneration_is_recovered_with_previous_dataset(self):
        w.write_v2_session(self.session, make_payload(seed=0))
        old_sha = hashlib.sha256((self.session / "facial" / "landmarks.npy").read_bytes()).hexdigest()
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, make_payload(seed=1), regenerate=True, fail_at="publish")
        w.recover_incomplete_v2(self.session)
        self.assertEqual(s2.session_layout(self.session), "v2")
        restored = hashlib.sha256((self.session / "facial" / "landmarks.npy").read_bytes()).hexdigest()
        self.assertEqual(restored, old_sha)
        self.assert_face_unchanged()

    def test_rollback_removes_only_pose_files(self):
        w.write_v2_session(self.session, make_payload())
        actions = w.rollback_v2(self.session)
        self.assertTrue(actions)
        self.assertFalse((self.session / "metadata.json").exists())
        self.assertFalse((self.session / "facial").exists())
        self.assertEqual(s2.session_layout(self.session), "v1")
        self.assert_face_unchanged()

    def test_rollback_restores_previous_dataset(self):
        w.write_v2_session(self.session, make_payload(seed=0))
        w.write_v2_session(self.session, make_payload(seed=1), regenerate=True)
        w.rollback_v2(self.session)
        self.assertEqual(s2.session_layout(self.session), "v2")
        self.assert_face_unchanged()

    def test_rollback_refuses_modified_files(self):
        w.write_v2_session(self.session, make_payload())
        (self.session / "timeline.csv").write_text("changed\n")
        with self.assertRaises(w.PoseDataError):
            w.rollback_v2(self.session)
        self.assertTrue((self.session / "timeline.csv").is_file())

    def test_rollback_without_manifest_is_refused(self):
        with self.assertRaises(w.PoseDataError):
            w.rollback_v2(self.session)


class LockTests(SessionTestCase):
    def test_second_writer_is_refused_while_lock_is_held(self):
        (self.session / w.LOCK_FILE).write_text("12345", encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.write_v2_session(self.session, make_payload())
        self.assertTrue((self.session / w.LOCK_FILE).is_file())
        self.assertEqual([p.name for p in self.session.iterdir() if p.name.startswith(".pose-staging")], [])

    def test_lock_is_released_after_success(self):
        w.write_v2_session(self.session, make_payload())
        self.assertFalse((self.session / w.LOCK_FILE).exists())

    def test_lock_is_released_after_an_ordinary_error(self):
        from unittest import mock

        with mock.patch.object(w, "_build_manifest", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                w.write_v2_session(self.session, make_payload())
        self.assertFalse((self.session / w.LOCK_FILE).exists())

    def test_crash_leaves_lock_and_recovery_clears_it(self):
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, make_payload(), fail_at="publish")
        self.assertTrue((self.session / w.LOCK_FILE).is_file())
        w.recover_incomplete_v2(self.session)
        self.assertFalse((self.session / w.LOCK_FILE).exists())
        w.write_v2_session(self.session, make_payload())
        self.assertEqual(s2.session_layout(self.session), "v2")


class FaceDataNameTests(unittest.TestCase):
    def test_only_root_level_face_data_names_are_protected(self):
        self.assertTrue(w.is_v1_path("facial_data.json"))
        self.assertTrue(w.is_v1_path("frames/frame_000001.jpg"))
        self.assertFalse(w.is_v1_path("quality/tracking_quality.json"))
        self.assertFalse(w.is_v1_path("facial/landmarks.npy"))
        self.assertFalse(w.is_v1_path("hands/calibration.json"))


class PostPublicationTests(SessionTestCase):
    def test_damage_during_publication_is_caught_before_metadata_is_written(self):
        from unittest import mock

        real_replace = w.os.replace

        def damaging_replace(src, dst):
            real_replace(src, dst)
            in_session = not any(part.startswith(w.STAGING_PREFIX) for part in Path(dst).parts)
            if in_session and Path(dst).name == "landmarks.npy":
                Path(dst).write_bytes(b"damaged")

        with mock.patch.object(w.os, "replace", side_effect=damaging_replace):
            with self.assertRaises(w.PoseDataError):
                w.write_v2_session(self.session, make_payload())
        self.assertFalse((self.session / "metadata.json").exists())
        self.assertEqual(s2.session_layout(self.session), "incomplete_v2")
        self.assert_face_unchanged()


class RecoverySafetyTests(SessionTestCase):
    def _interrupted_publish(self):
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, make_payload(), fail_at="publish")

    def test_recovery_refuses_a_corrupt_manifest_and_deletes_nothing(self):
        self._interrupted_publish()
        (self.session / "manifest.json").write_text("{corrupt", encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.recover_incomplete_v2(self.session)
        self.assertTrue((self.session / "manifest.json").is_file())
        self.assertTrue((self.session / "facial" / "landmarks.npy").is_file())
        self.assert_face_unchanged()

    def test_recovery_refuses_a_manifest_that_lists_a_face_data_file(self):
        self._interrupted_publish()
        manifest = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
        manifest["files"].append({"path": "facial_data.json", "bytes": 1, "sha256": "0" * 64})
        (self.session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.recover_incomplete_v2(self.session)
        self.assert_face_unchanged()

    def test_recovery_refuses_to_delete_a_modified_file(self):
        self._interrupted_publish()
        (self.session / "timeline.csv").write_text("edited by hand\n", encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.recover_incomplete_v2(self.session)
        self.assertEqual((self.session / "timeline.csv").read_text(encoding="utf-8"), "edited by hand\n")

    def test_rollback_refuses_a_manifest_that_lists_a_face_data_file(self):
        w.write_v2_session(self.session, make_payload())
        manifest = json.loads((self.session / "manifest.json").read_text(encoding="utf-8"))
        manifest["files"].append({"path": "calibration.json", "bytes": 1, "sha256": "0" * 64})
        (self.session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.rollback_v2(self.session)
        self.assert_face_unchanged()


class HeaderReadTests(SessionTestCase):
    def test_header_reader_matches_the_array_on_disk(self):
        path = self.tmp / "x.npy"
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        np.save(path, arr)
        self.assertEqual(w._npy_header(path), ((2, 3, 4), "float32"))

    def test_writer_does_not_memory_map_outputs(self):
        # Memory-mapped views can keep a file open, which blocks renames on Windows.
        source = (ROOT / "src" / "analysis" / "pose_writer.py").read_text(encoding="utf-8")
        self.assertNotIn("mmap_mode", source)


class DurabilityTests(SessionTestCase):
    def test_every_written_file_is_flushed_to_disk_before_it_is_renamed(self):
        from unittest import mock

        with mock.patch.object(w.os, "fsync", wraps=w.os.fsync) as fsync:
            manifest = w.write_v2_session(self.session, make_payload())
        # one fsync per staged data file, plus manifest.json and metadata.json
        self.assertGreaterEqual(fsync.call_count, len(manifest["files"]) + 2)


class StagingCleanupTests(SessionTestCase):
    def test_ordinary_error_during_staging_removes_the_staging_folder(self):
        from unittest import mock

        with mock.patch.object(w, "_build_manifest", side_effect=RuntimeError("disk problem")):
            with self.assertRaises(RuntimeError):
                w.write_v2_session(self.session, make_payload())
        self.assertEqual([p.name for p in self.session.iterdir() if p.name.startswith(".pose-staging")], [])
        self.assertFalse((self.session / "metadata.json").exists())
        self.assert_face_unchanged()

    def test_simulated_crash_keeps_the_staging_folder_for_recovery(self):
        with self.assertRaises(w.SimulatedInterruption):
            w.write_v2_session(self.session, make_payload(), fail_at="validate")
        self.assertEqual(len([p for p in self.session.iterdir() if p.name.startswith(".pose-staging")]), 1)


class DestinationCollisionTests(SessionTestCase):
    def test_stray_file_at_a_destination_is_not_overwritten(self):
        (self.session / "timeline.csv").write_text("stray\n", encoding="utf-8")
        with self.assertRaises(w.PoseDataError):
            w.write_v2_session(self.session, make_payload())
        self.assertEqual((self.session / "timeline.csv").read_text(encoding="utf-8"), "stray\n")
        self.assertFalse((self.session / "metadata.json").exists())
        self.assert_face_unchanged()

    def test_collision_check_leaves_no_staging_folder(self):
        (self.session / "facial").mkdir()
        (self.session / "facial" / "landmarks.npy").write_bytes(b"stray")
        with self.assertRaises(w.PoseDataError):
            w.write_v2_session(self.session, make_payload())
        self.assertEqual([p.name for p in self.session.iterdir() if p.name.startswith(".pose-staging")], [])


class IncompleteDetectionTests(SessionTestCase):
    def test_face_only(self):
        self.assertEqual(s2.session_layout(self.session), "v1")

    def test_pose_folders_without_metadata(self):
        (self.session / "facial").mkdir()
        self.assertEqual(s2.session_layout(self.session), "incomplete_v2")

    def test_staging_folder_is_visible_to_recovery(self):
        staging = self.session / (w.STAGING_PREFIX + "abc")
        staging.mkdir()
        actions = w.recover_incomplete_v2(self.session)
        self.assertFalse(staging.exists())
        self.assertTrue(any("staging" in a for a in actions))


class PathTests(unittest.TestCase):
    def test_manifest_paths_use_forward_slashes(self):
        root = Path(tempfile.mkdtemp())
        try:
            manifest = w.write_v2_session(root, make_payload())
            for entry in manifest["files"]:
                self.assertNotIn("\\", entry["path"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_relative_paths_map_to_platform_paths(self):
        base = Path("some") / "root"
        self.assertEqual(w._rel_to_path("facial/landmarks.npy", base), base / "facial" / "landmarks.npy")

    def test_escaping_paths_are_refused(self):
        for bad in ("../x", "/abs/x", "a/../../x", "", "a//b"):
            with self.subTest(path=bad), self.assertRaises(w.PoseDataValidationError):
                w._rel_to_path(bad, Path("base"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
