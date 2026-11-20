"""Naming migration, step 2: feature paths and job codes. The old paths and codes must keep working.

Synthetic data only: V1 sessions from the demo generator and one synthetic V2 session in temporary folders.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "dashboard"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
import demo_data  # noqa: E402
import test_pose_viewer as tv  # noqa: E402
from api import catalog, jobs, v1_data, v2_data  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402


def call(app, method, path, body=None, token=None):
    """Call the ASGI app directly, with a query string and an optional JSON body and token."""
    target, _, query = path.partition("?")
    raw = json.dumps(body).encode() if body is not None else b""
    headers = [(b"content-type", b"application/json")] if body is not None else []
    if token is not None:
        headers.append((b"x-facial-token", token.encode()))
    scope = {"type": "http", "method": method, "path": target, "raw_path": target.encode(),
             "query_string": query.encode(), "headers": headers, "server": ("127.0.0.1", 8000),
             "scheme": "http", "root_path": "", "http_version": "1.1"}
    result = {"body": []}

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
        elif message["type"] == "http.response.body":
            result["body"].append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return result["status"], json.loads(b"".join(result["body"]) or b"null")


class StepTwoTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="naming2_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        v1_data.clear_cache()
        v2_data.clear_cache()
        demo_data.generate_session(self.tmp, "session-001", demo_data.SESSIONS["session-001"])
        tv.write_pose(pp.DATASETS_ROOT / "demo_person" / "session-002")
        for session in ("session-001", "session-002"):
            video = pp.session_video("demo_person", session)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"placeholder recording")
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token="t-step2",
                       command_for=lambda job: [sys.executable, "-c", "print('nothing')"])

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        v1_data.clear_cache()
        v2_data.clear_cache()
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)


class FeaturePathTests(StepTwoTestCase):
    def test_face_paths_return_the_same_as_the_older_paths(self):
        from api.main import app

        pairs = [
            ("/api/persons/demo_person/sessions/session-001/face/summary",
             "/api/persons/demo_person/sessions/session-001/v1/summary"),
            ("/api/persons/demo_person/sessions/session-001/face/blinks",
             "/api/persons/demo_person/sessions/session-001/v1/blinks"),
            ("/api/persons/demo_person/sessions/session-001/face/head-pose",
             "/api/persons/demo_person/sessions/session-001/v1/head-pose"),
            ("/api/persons/demo_person/sessions/session-001/face/series?fields=velocity",
             "/api/persons/demo_person/sessions/session-001/v1/series?fields=velocity"),
            ("/api/persons/demo_person/sessions/session-001/face/landmarks/3",
             "/api/persons/demo_person/sessions/session-001/v1/landmarks/3"),
        ]
        for new, old in pairs:
            with self.subTest(new=new):
                status_new, body_new = call(app, "GET", new)
                status_old, body_old = call(app, "GET", old)
                self.assertEqual(status_new, 200)
                self.assertEqual(status_new, status_old)
                self.assertEqual(body_new, body_old)

    def test_pose_paths_return_the_same_as_the_older_paths(self):
        from api.main import app

        base = "/api/persons/demo_person/sessions/session-002"
        pairs = [
            (f"{base}/pose", f"{base}/v2"),
            (f"{base}/pose/timeline", f"{base}/v2/timeline"),
            (f"{base}/pose/frames/1/quality", f"{base}/v2/frames/1/quality"),
        ]
        for new, old in pairs:
            with self.subTest(new=new):
                status_new, body_new = call(app, "GET", new)
                status_old, body_old = call(app, "GET", old)
                self.assertEqual(status_new, 200)
                self.assertEqual(body_new, body_old)

    def test_overlay_path_is_served_under_the_new_name(self):
        from api.main import app

        routes = {r.path for r in app.routes}
        self.assertIn("/api/persons/{person}/sessions/{session}/pose/frames/{index}/overlay.jpg", routes)
        self.assertIn("/api/persons/{person}/sessions/{session}/v2/frames/{index}/overlay.jpg", routes)

    def test_new_paths_are_still_read_only(self):
        from api.main import app

        for route in app.routes:
            if "/face/" in route.path or "/pose" in route.path and not route.path.startswith("/api/jobs"):
                self.assertTrue(set(route.methods) <= {"GET", "HEAD"}, route.path)


class SessionNameTests(StepTwoTestCase):
    def test_session_rows_carry_the_display_name_and_keep_the_code(self):
        rows = {r["session_id"]: r for r in catalog.list_sessions("demo_person")}
        self.assertEqual(rows["session-001"]["layout"], "v1")
        self.assertEqual(rows["session-001"]["layout_name"], "face data")
        self.assertEqual(rows["session-002"]["layout"], "v2")
        self.assertEqual(rows["session-002"]["layout_name"], "pose data")


class JobCodeTests(StepTwoTestCase):
    def test_old_codes_map_to_the_new_codes(self):
        self.assertEqual(jobs.canonical_kind("v2_build"), "pose_build")
        self.assertEqual(jobs.canonical_kind("v1_extract"), "face_extract")
        self.assertEqual(jobs.canonical_kind("v1_analyse"), "face_analyse")
        self.assertEqual(jobs.canonical_kind("pose_build"), "pose_build")

    def test_a_job_created_with_an_old_code_is_stored_and_listed_with_the_new_code(self):
        from api.main import app

        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001", "kind": "v1_analyse", "regenerate": True},
                            token="t-step2")
        self.assertEqual(status, 202)
        self.assertEqual(body["kind"], "face_analyse")
        jobs.service().runner.cancel(body["id"])

    def test_a_row_written_with_an_old_code_reads_as_the_new_code(self):
        store = jobs.service().store
        job = store.insert("demo_person", "session-002", "v2_build", False)
        listed = {j["id"]: j for j in store.list()}
        self.assertEqual(listed[job["id"]]["kind"], "pose_build")

    def test_worker_command_is_the_same_for_old_and_new_codes(self):
        base = {"person": "demo_person", "session": "session-001", "regenerate": True}
        self.assertEqual(jobs.default_command({**base, "kind": "v1_analyse"}),
                         jobs.default_command({**base, "kind": "face_analyse"}))
        self.assertIn("--regenerate-v2", jobs.default_command({**base, "kind": "pose_build"}))

    def test_refusal_codes_use_the_feature_names(self):
        from api.main import app

        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001", "kind": "face_extract"},
                            token="t-step2")
        self.assertEqual((status, body["code"]), (409, "face_data_exists"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
