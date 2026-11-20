"""Naming migration, end state: the new paths, codes and module names work, and the old ones are gone.

Synthetic data only: a face data session from the demo generator and one synthetic pose session in temporary folders.
"""

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "viewer"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
import demo_data  # noqa: E402
import test_pose_viewer as tv  # noqa: E402
from api import face_data, jobs, pose_data  # noqa: E402
from test_api_p4 import call  # noqa: E402

TOKEN = "t-migration"


class MigrationTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="naming_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        face_data.clear_cache()
        pose_data.clear_cache()
        demo_data.generate_session(self.tmp, "session-001", demo_data.SESSIONS["session-001"])
        tv.write_pose(pp.DATASETS_ROOT / "demo_person" / "session-002")
        for session in ("session-001", "session-002"):
            video = pp.session_video("demo_person", session)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"placeholder recording")
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token=TOKEN,
                       command_for=lambda job: [sys.executable, "-c", "print('nothing')"])

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        face_data.clear_cache()
        pose_data.clear_cache()
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)


class RoutesTests(MigrationTestCase):
    def test_feature_paths_answer(self):
        from api.main import app

        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-001/face/summary")
        self.assertEqual(status, 200)
        self.assertIn("quality", body)
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-002/pose")
        self.assertEqual(status, 200)
        self.assertTrue(body["valid"])
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-002/pose/timeline")
        self.assertEqual(status, 200)

    def test_old_paths_are_gone(self):
        from api.main import app

        for path in ("/api/persons/demo_person/sessions/session-001/v1/summary",
                     "/api/persons/demo_person/sessions/session-002/v2",
                     "/api/persons/demo_person/sessions/session-002/v2/timeline"):
            with self.subTest(path=path):
                status, _body = call(app, "GET", path)
                self.assertEqual(status, 404)

    def test_no_route_mentions_the_old_names(self):
        from api.main import app

        for route in app.routes:
            self.assertNotIn("/v1/", route.path)
            self.assertNotIn("/v2", route.path)


class JobCodeTests(MigrationTestCase):
    def test_new_codes_are_accepted(self):
        from api.main import app

        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001", "kind": "pose_build",
                             "regenerate": True}, token=TOKEN)
        self.assertEqual(status, 202)
        self.assertEqual(body["kind"], "pose_build")
        jobs.service().runner.cancel(body["id"])

    def test_old_codes_are_refused(self):
        from api.main import app

        for old in ("v2_build", "v1_extract", "v1_analyse"):
            with self.subTest(kind=old):
                status, body = call(app, "POST", "/api/jobs",
                                    {"person": "demo_person", "session": "session-001", "kind": old}, token=TOKEN)
                self.assertEqual((status, body["code"]), (400, "invalid_request"))

    def test_rows_saved_with_old_codes_are_moved_to_the_new_codes(self):
        store = jobs.service().store
        job = store.insert("demo_person", "session-002", "pose_build", False)
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE jobs SET kind = 'v2_build' WHERE id = ?", (job["id"],))
        reopened = jobs.JobStore(store.path)
        self.assertEqual(reopened.get(job["id"])["kind"], "pose_build")
        reopened = jobs.JobStore(store.path)  # running it again changes nothing
        self.assertEqual(reopened.get(job["id"])["kind"], "pose_build")

    def test_worker_command_uses_the_renamed_scripts(self):
        base = {"person": "demo_person", "session": "session-001", "regenerate": True}
        self.assertIn("build_pose_session.py", str(jobs.default_command({**base, "kind": "pose_build"})[1]))
        self.assertNotIn("--force", jobs.default_command({**base, "kind": "face_build"}))


class ModuleAndClassNameTests(unittest.TestCase):
    def test_old_module_files_are_gone(self):
        old = ["schema_v2.py", "session_writer_v2.py", "v2_viewer.py", "v2_session_builder.py", "build_v2_session.py"]
        found = []
        for path in ROOT.rglob("*.py"):
            if ".venv" in path.parts or "node_modules" in path.parts:
                continue
            if path.name in old or path.name in ("v1_data.py", "v2_data.py"):
                found.append(str(path))
        self.assertEqual(found, [])

    def test_new_names_import_and_old_class_names_are_gone(self):
        import pose_schema, pose_writer, pose_viewer  # noqa: F401
        self.assertTrue(hasattr(pose_writer, "PoseDataError"))
        # the old names are spelt in pieces so that renames do not change what is being checked
        for old_name in ("V" + "2Error", "V" + "2Payload"):
            self.assertFalse(hasattr(pose_writer, old_name), old_name)
        self.assertFalse(hasattr(pose_viewer, "V" + "2ViewerError"))

    def test_no_old_temporary_or_lock_names_are_used(self):
        import pose_writer as w
        self.assertEqual(w.STAGING_PREFIX, ".pose-staging-")
        self.assertEqual(w.PREVIOUS_PREFIX, ".pose-previous-")
        self.assertEqual(w.LOCK_FILE, ".pose-lock")
        for name in ("LEGACY_LOCK_FILE", "LEGACY_STAGING_PREFIX", "LEGACY_PREVIOUS_PREFIX", "LOCK_FILES",
                     "STAGING_PREFIXES", "PREVIOUS_PREFIXES"):
            self.assertFalse(hasattr(w, name), name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
