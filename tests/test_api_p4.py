"""Phase P4 tests: the job system. A fake worker stands in for the V2 build, so no model is needed.

Covers: refusals before queueing, token checks, success, failure, cancel, one active job per session,
the writer lock, restart handling (interrupted jobs, leftovers reported and not removed), and that the create
request returns before the worker finishes.
"""

import asyncio
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

import project_paths as pp  # noqa: E402
from api import jobs  # noqa: E402

TOKEN = "test-token-p4"

FAKE_WORKER = '''
import sys, time
mode = {"session-001": "ok", "session-002": "fail"}.get(sys.argv[1], "slow")
print("Video: fake.mp4 (64x48, 25.000 fps)", flush=True)
print("Pose Landmarker FULL loaded (SHA-256 verified).", flush=True)
print("Extracting pose landmarks...", flush=True)
if mode == "ok":
    print("V2 session published: fake", flush=True)
    sys.exit(0)
if mode == "fail":
    print("boom", flush=True)
    sys.exit(1)
time.sleep(60)
'''


def call(app, method, path, body=None, token=None):
    """Call the ASGI app directly. Returns (status, json body)."""
    raw = json.dumps(body).encode() if body is not None else b""
    headers = [(b"content-type", b"application/json")] if body is not None else []
    if token is not None:
        headers.append((b"x-facial-token", token.encode()))
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": headers, "server": ("127.0.0.1", 8000),
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


class P4TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p4_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        self.worker = self.tmp / "fake_worker.py"
        self.worker.write_text(FAKE_WORKER, encoding="utf-8")
        # sessions: 001 succeeds, 002 fails, 003 runs until cancelled, 004 has the writer lock
        for session in ("session-001", "session-002", "session-003", "session-004"):
            folder = pp.DATASETS_ROOT / "demo_person" / session
            folder.mkdir(parents=True)
            (folder / "facial_data.json").write_text("[]", encoding="utf-8")
            video = pp.session_video("demo_person", session)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"not a real video; the fake worker does not read it")
        (pp.DATASETS_ROOT / "demo_person" / "session-004" / jobs.LOCK_FILE).write_text("", encoding="utf-8")
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token=TOKEN,
                       command_for=lambda job: [sys.executable, str(self.worker), job["session"]])
        # the real face model is not in CI; the model check is tested by patching it
        patcher = mock.patch.object(jobs, "verify_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        # stop any worker left running by a test, so no process outlives the test
        svc = jobs.service()
        for job in svc.store.list(limit=100):
            if job["state"] in jobs.ACTIVE:
                svc.runner.cancel(job["id"])
        time.sleep(0.5)
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def wait_for(self, job_id, states, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = jobs.service().store.get(job_id)
            if job["state"] in states:
                return job
            time.sleep(0.1)
        self.fail(f"job {job_id} did not reach {states}; last state {job['state']}")

    def post_job(self, session, regenerate=False, kind="v2_build"):
        from api.main import app

        return call(app, "POST", "/api/jobs",
                    {"person": "demo_person", "session": session, "kind": kind, "regenerate": regenerate},
                    token=TOKEN)


class RefusalTests(P4TestCase):
    def test_mutating_routes_require_the_token(self):
        from api.main import app

        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001"})
        self.assertEqual((status, body["code"]), (401, "unauthorized"))
        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001"}, token="wrong")
        self.assertEqual(status, 401)
        self.assertEqual(jobs.service().store.list(), [])

    def test_missing_video_is_refused_without_queueing(self):
        pp.session_video("demo_person", "session-001").unlink()
        status, body = self.post_job("session-001")
        self.assertEqual((status, body["code"]), (422, "no_video"))
        self.assertEqual(jobs.service().store.list(), [])

    def test_missing_version_one_data_is_refused(self):
        (pp.DATASETS_ROOT / "demo_person" / "session-001" / "facial_data.json").unlink()
        status, body = self.post_job("session-001")
        self.assertEqual((status, body["code"]), (422, "no_v1_data"))

    def test_bad_names_and_kinds_are_refused(self):
        from api.main import app

        status, body = call(app, "POST", "/api/jobs",
                            {"person": "../etc", "session": "session-001"}, token=TOKEN)
        self.assertEqual(status, 400)
        status, body = call(app, "POST", "/api/jobs",
                            {"person": "demo_person", "session": "session-001", "kind": "delete_all"}, token=TOKEN)
        self.assertEqual((status, body["code"]), (400, "invalid_request"))

    def test_writer_lock_is_refused(self):
        status, body = self.post_job("session-004")
        self.assertEqual((status, body["code"]), (409, "writer_lock_present"))


class LifecycleTests(P4TestCase):
    def test_create_returns_before_the_worker_finishes(self):
        start = time.monotonic()
        status, job = self.post_job("session-003")
        elapsed = time.monotonic() - start
        self.assertEqual(status, 202)
        self.assertLess(elapsed, 10, "the create request must not wait for the worker")
        self.assertIn(job["state"], ("queued", "running"))
        jobs.cancel_job(job["id"])
        self.wait_for(job["id"], ("cancelled",))

    def test_success_reports_stages_and_log(self):
        status, job = self.post_job("session-001")
        self.assertEqual(status, 202)
        done = self.wait_for(job["id"], ("succeeded", "failed"))
        self.assertEqual(done["state"], "succeeded")
        self.assertEqual(done["exit_code"], 0)
        self.assertEqual(done["stage"], "verify")
        self.assertEqual(done["progress"], 1.0)
        log = jobs.service().runner.log_path(job["id"]).read_text(encoding="utf-8")
        self.assertIn("V2 session published", log)

    def test_failure_is_recorded_with_its_exit_code(self):
        status, job = self.post_job("session-002")
        done = self.wait_for(job["id"], ("succeeded", "failed"))
        self.assertEqual((done["state"], done["exit_code"]), ("failed", 1))
        self.assertIn("failed; see the job log", done["message"])

    def test_cancel_stops_the_worker_and_marks_cancelled(self):
        from api.main import app

        status, job = self.post_job("session-003")
        self.wait_for(job["id"], ("running",))
        status, body = call(app, "POST", f"/api/jobs/{job['id']}/cancel", token=TOKEN)
        self.assertEqual(status, 200)
        done = self.wait_for(job["id"], ("cancelled",))
        self.assertTrue(done["cancel_requested"])
        self.assertIsNone(done["pid"])

    def test_cancel_refuses_a_finished_job(self):
        from api.main import app

        status, job = self.post_job("session-001")
        self.wait_for(job["id"], ("succeeded",))
        status, body = call(app, "POST", f"/api/jobs/{job['id']}/cancel", token=TOKEN)
        self.assertEqual((status, body["code"]), (409, "not_active"))

    def test_second_job_for_the_same_session_is_refused(self):
        status, first = self.post_job("session-003")
        self.assertEqual(status, 202)
        self.wait_for(first["id"], ("running",))
        status, body = self.post_job("session-003")
        self.assertEqual((status, body["code"]), (409, "job_active"))
        jobs.cancel_job(first["id"])
        self.wait_for(first["id"], ("cancelled",))

    def test_completed_v2_session_needs_regenerate(self):
        original = jobs.session_layout
        jobs.session_layout = lambda _folder: "v2"
        try:
            status, body = self.post_job("session-001")
            self.assertEqual((status, body["code"]), (409, "v2_exists"))
            status, job = self.post_job("session-001", regenerate=True)
            self.assertEqual(status, 202)
            self.wait_for(job["id"], ("succeeded", "failed"))
        finally:
            jobs.session_layout = original


class RestartTests(P4TestCase):
    def test_active_jobs_are_marked_interrupted_and_leftovers_kept(self):
        store = jobs.service().store
        job = store.insert("demo_person", "session-001", "v2_build", False)
        store.update(job["id"], state="running", pid=4242)
        staging = pp.DATASETS_ROOT / "demo_person" / "session-001" / ".v2-staging-abc"
        staging.mkdir()

        marked = jobs.recover_interrupted()
        self.assertEqual([m["id"] for m in marked], [job["id"]])
        after = store.get(job["id"])
        self.assertEqual(after["state"], "interrupted")
        self.assertIsNone(after["pid"])
        self.assertIn("4242", after["message"])
        self.assertIn(".v2-staging-abc", after["message"])
        self.assertTrue(staging.is_dir(), "leftovers must not be removed automatically")

    def test_finished_jobs_are_not_touched_on_restart(self):
        store = jobs.service().store
        job = store.insert("demo_person", "session-002", "v2_build", False)
        store.update(job["id"], state="succeeded")
        self.assertEqual(jobs.recover_interrupted(), [])
        self.assertEqual(store.get(job["id"])["state"], "succeeded")

    def test_only_one_active_job_per_session_is_enforced_by_the_database(self):
        import sqlite3

        store = jobs.service().store
        store.insert("demo_person", "session-001", "v2_build", False)
        with self.assertRaises(sqlite3.IntegrityError):
            store.insert("demo_person", "session-001", "v2_build", False)


class V1KindTests(P4TestCase):
    def test_extraction_refuses_existing_version_one_data(self):
        status, body = self.post_job("session-001", kind="v1_extract")
        self.assertEqual((status, body["code"]), (409, "v1_exists"))

    def test_extraction_refuses_when_the_face_model_fails_its_check(self):
        (pp.DATASETS_ROOT / "demo_person" / "session-003" / "facial_data.json").unlink()
        with mock.patch.object(jobs, "verify_model", side_effect=jobs.ModelIntegrityError("hash mismatch")):
            status, body = self.post_job("session-003", kind="v1_extract")
        self.assertEqual((status, body["code"]), (503, "model_integrity"))
        self.assertEqual(jobs.service().store.list(), [])

    def test_analysis_runs_and_succeeds(self):
        status, job = self.post_job("session-001", kind="v1_analyse")
        self.assertEqual(status, 202)
        done = self.wait_for(job["id"], ("succeeded", "failed"))
        self.assertEqual(done["kind"], "v1_analyse")
        self.assertEqual(done["state"], "succeeded")

    def test_analysis_refuses_existing_outputs_unless_regenerate(self):
        (pp.DATASETS_ROOT / "demo_person" / "session-002" / "facial_blendshapes.csv").write_text("x", encoding="utf-8")
        status, body = self.post_job("session-002", kind="v1_analyse")
        self.assertEqual((status, body["code"]), (409, "outputs_exist"))

    def test_one_active_job_per_session_across_kinds(self):
        status, first = self.post_job("session-003", kind="v2_build")
        self.assertEqual(status, 202)
        self.wait_for(first["id"], ("running",))
        status, body = self.post_job("session-003", kind="v1_analyse")
        self.assertEqual((status, body["code"]), (409, "job_active"))
        jobs.cancel_job(first["id"])
        self.wait_for(first["id"], ("cancelled",))

    def test_regenerate_adds_the_matching_flag(self):
        base = {"person": "demo_person", "session": "session-001", "regenerate": True}
        self.assertIn("--force", jobs.default_command({**base, "kind": "v1_analyse"}))
        self.assertIn("--regenerate-v2", jobs.default_command({**base, "kind": "v2_build"}))
        self.assertNotIn("--force", jobs.default_command({**base, "kind": "v1_extract"}))

    def test_worker_output_is_unbuffered(self):
        self.assertEqual(jobs.worker_env()["PYTHONUNBUFFERED"], "1")


class ProgressParsingTests(unittest.TestCase):
    def test_extraction_progress_is_per_frame(self):
        state = {}
        self.assertEqual(jobs.parse_progress("v1_extract", "Total frames: 200", state), ("extract", 0.0))
        self.assertEqual(jobs.parse_progress("v1_extract", "Processed: 100", state), ("extract", 0.5))
        self.assertEqual(jobs.parse_progress("v1_extract", "Processed: 300", state), ("extract", 0.99))
        self.assertEqual(jobs.parse_progress("v1_extract", "Processing completed!", state), ("publish", 0.99))
        self.assertIsNone(jobs.parse_progress("v1_extract", "FPS: 30.0", state))

    def test_analysis_and_build_markers(self):
        self.assertEqual(jobs.parse_progress("v1_analyse", "CSV created: x", {}), ("assemble", 0.25))
        self.assertEqual(jobs.parse_progress("v1_analyse", "Metadata: x", {}), ("verify", 0.9))
        self.assertEqual(jobs.parse_progress("v2_build", "Extracting pose landmarks...", {}), ("extract", 3 / 6))
        self.assertIsNone(jobs.parse_progress("v2_build", "unrelated line", {}))

    def test_partly_written_frames_are_reported_for_extraction(self):
        tmp = Path(tempfile.mkdtemp(prefix="leftover_"))
        try:
            (tmp / "frames").mkdir()
            (tmp / "frames" / "frame_000000.jpg").write_bytes(b"x")
            self.assertEqual(jobs.leftovers(tmp, "v1_extract"), ["frames/ (partly written)"])
            (tmp / "facial_data.json").write_text("[]", encoding="utf-8")
            self.assertEqual(jobs.leftovers(tmp, "v1_extract"), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ReadTests(P4TestCase):
    def test_job_reads_do_not_need_the_token(self):
        from api.main import app

        status, job = self.post_job("session-001")
        self.wait_for(job["id"], ("succeeded",))
        status, body = call(app, "GET", f"/api/jobs/{job['id']}")
        self.assertEqual((status, body["state"]), (200, "succeeded"))
        status, body = call(app, "GET", "/api/jobs")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), 1)
        status, body = call(app, "GET", "/api/jobs/" + "0" * 32)
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
