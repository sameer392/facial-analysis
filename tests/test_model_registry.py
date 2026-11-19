"""Tests for the model registry and SHA-256 verification (temporary files only; no model is loaded)."""

import hashlib
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extraction import model_registry as mr  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_every_registered_model_has_a_valid_sha256(self):
        for name, spec in mr.MODELS.items():
            with self.subTest(model=name):
                self.assertRegex(spec.sha256, r"^[0-9A-F]{64}$")

    def test_registered_filenames_are_task_files(self):
        for name, spec in mr.MODELS.items():
            with self.subTest(model=name):
                self.assertTrue(spec.filename.endswith(".task"))

    def test_pose_full_entry_is_partly_verified_not_cleared(self):
        spec = mr.MODELS["pose_full"]
        self.assertEqual(spec.filename, "pose_landmarker_full.task")
        self.assertEqual(spec.sha256, "4EAA5EB7A98365221087693FCC286334CF0858E2EB6E15B506AA4A7ECDCEC4AD")
        self.assertIn("not cleared", spec.licence_status.lower())

    def test_version_is_recorded_as_unpinned_when_unknown(self):
        self.assertEqual(mr.MODELS["pose_full"].version, "unpinned (latest)")


class HashingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="model_registry_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sha256_matches_hashlib_for_a_small_file(self):
        path = self.tmp / "small.bin"
        data = b"synthetic model bytes" * 100
        path.write_bytes(data)
        self.assertEqual(mr.sha256_file(path), hashlib.sha256(data).hexdigest().upper())

    def test_sha256_is_computed_in_chunks_for_large_files(self):
        path = self.tmp / "large.bin"
        data = bytes(range(256)) * (3 * 1024 * 1024 // 256)  # 3 MiB, more than one chunk
        path.write_bytes(data)
        self.assertEqual(mr.sha256_file(path, chunk_size=64 * 1024), hashlib.sha256(data).hexdigest().upper())


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="model_registry_"))
        self.payload = b"synthetic task bundle"
        self.good_hash = hashlib.sha256(self.payload).hexdigest().upper()
        self.spec = mr.ModelSpec(
            key="synthetic",
            filename="synthetic.task",
            sha256=self.good_hash,
            version="test-1",
            source_url="https://example.invalid/synthetic.task",
            licence_status="synthetic test data",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matching_file_verifies(self):
        (self.tmp / "synthetic.task").write_bytes(self.payload)
        path = mr.verify_model(self.spec, models_dir=self.tmp)
        self.assertEqual(path, self.tmp / "synthetic.task")

    def test_altered_file_is_refused(self):
        (self.tmp / "synthetic.task").write_bytes(self.payload + b"!")
        with self.assertRaisesRegex(mr.ModelIntegrityError, "SHA-256 mismatch"):
            mr.verify_model(self.spec, models_dir=self.tmp)

    def test_missing_file_is_reported_with_its_expected_location(self):
        with self.assertRaisesRegex(mr.ModelIntegrityError, "not found"):
            mr.verify_model(self.spec, models_dir=self.tmp)

    def test_unknown_model_name_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "unknown model"):
            mr.get_spec("does_not_exist")

    def test_refused_file_is_never_opened_for_loading(self):
        # verify_model only hashes; it must not return a path for a bad file, so the caller
        # cannot load it by accident.
        (self.tmp / "synthetic.task").write_bytes(b"corrupt")
        with self.assertRaises(mr.ModelIntegrityError):
            mr.verify_model(self.spec, models_dir=self.tmp)


class RealModelTests(unittest.TestCase):
    """Checks the real files when they are present locally. Skipped in CI, where models are absent."""

    def test_present_models_match_their_registry_hash(self):
        present = [spec for spec in mr.MODELS.values() if (mr.MODELS_DIR / spec.filename).is_file()]
        if not present:
            self.skipTest("no model files present locally")
        for spec in present:
            with self.subTest(model=spec.key):
                mr.verify_model(spec)


if __name__ == "__main__":
    unittest.main(verbosity=2)
