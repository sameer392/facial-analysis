"""Hand motion API and job tests (phase H5). Synthetic data and a fake worker only; no model is needed for jobs."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
from api import hand_data, jobs  # noqa: E402
from api.main import app  # noqa: E402
from pose_schema import session_layout  # noqa: E402
import test_classify_hand_cli as tcl  # noqa: E402  (full synthetic H1 to H3 session)

TOKEN = "test-token-h5"

FAKE_HAND_WORKER = '''
import sys, time
mode = {"session-001": "ok", "session-002": "fail"}.get(sys.argv[2], "slow")
print("Stage 1 of 4: hand landmarks", flush=True)
print("Stage 2 of 4: hand tracking", flush=True)
print("Stage 3 of 4: gestures", flush=True)
print("Stage 4 of 4: validation", flush=True)
if mode == "ok":
    print("Hand Motion pipeline complete. Validation: no errors.", flush=True)
    sys.exit(0)
if mode == "fail":
    print("boom", flush=True)
    sys.exit(1)
time.sleep(60)
'''


def call(method, path, query="", body=None, token=None):
    """Call the ASGI app directly. Returns (status, body bytes, headers)."""
    raw = json.dumps(body).encode() if body is not None else b""
    headers = [(b"content-type", b"application/json")] if body is not None else []
    if token is not None:
        headers.append((b"x-facial-token", token.encode()))
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": query.encode(), "headers": headers, "server": ("127.0.0.1", 8000),
             "scheme": "http", "root_path": "", "http_version": "1.1"}
    result = {"body": b"", "headers": {}}

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = {k.decode(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            result["body"] += message.get("body", b"")

    import asyncio
    asyncio.run(app(scope, receive, send))
    return result["status"], result["body"], result["headers"]


def js(method, path, query="", body=None, token=None):
    status, raw, _ = call(method, path, query, body, token)
    try:
        return status, json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        return status, raw


class JobCase(unittest.TestCase):
    """Job tests use a fake hand worker, so they do not need MediaPipe."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_hand_jobs_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        self.worker = self.tmp / "fake_hand_worker.py"
        self.worker.write_text(FAKE_HAND_WORKER, encoding="utf-8")
        self.svc = jobs.configure(self.tmp / "jobs.sqlite3", self.tmp / "logs", token=TOKEN,
                                  command_for=lambda job: [sys.executable, str(self.worker),
                                                           job["person"], job["session"]])

    def tearDown(self):
        for job in self.svc.store.list(limit=1000):
            if job["state"] in jobs.ACTIVE:
                self.svc.runner.cancel(job["id"])
        time.sleep(0.2)
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_session(self, session="session-001", pose=True):
        video = pp.session_video("demo_person", session)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 64)
        dataset = pp.session_dataset_dir("demo_person", session)
        dataset.mkdir(parents=True, exist_ok=True)
        if pose:
            (dataset / "metadata.json").write_text(json.dumps({"schema_version": "2.0", "streams": {}}))
            (dataset / "facial").mkdir(exist_ok=True)
        return dataset


class JobTests(JobCase):
    def body(self, session="session-001", kind="hand_build", regenerate=False):
        return {"person": "demo_person", "session": session, "kind": kind, "regenerate": regenerate}

    def test_creating_a_job_needs_the_token(self):
        self.make_session()
        status, _ = js("POST", "/api/jobs", body=self.body())
        self.assertEqual(status, 401)

    def test_missing_recording_is_refused(self):
        self.make_session()
        pp.session_video("demo_person", "session-001").unlink()
        status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (422, "no_video"))

    def test_missing_pose_session_is_refused(self):
        self.make_session(pose=False)
        status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (422, "no_pose_data"))

    def test_invalid_session_name_is_refused(self):
        status, body = js("POST", "/api/jobs", body=self.body(session="../etc"), token=TOKEN)
        self.assertEqual(status, 400)

    def test_successful_job_reaches_succeeded_with_progress(self):
        self.make_session()
        status, job = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual(status, 202)
        for _ in range(100):
            _, job = js("GET", f"/api/jobs/{job['id']}")
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(job["state"], "succeeded")
        self.assertEqual(job["progress"], 1.0)

    def test_failed_worker_marks_the_job_failed(self):
        self.make_session(session="session-002")
        status, job = js("POST", "/api/jobs", body=self.body(session="session-002"), token=TOKEN)
        for _ in range(100):
            _, job = js("GET", f"/api/jobs/{job['id']}")
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        self.assertEqual(job["state"], "failed")

    def test_second_job_for_the_same_session_is_refused(self):
        self.make_session()
        js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (409, "job_active"))

    def test_cancelling_a_running_job(self):
        self.make_session(session="session-003")
        _, job = js("POST", "/api/jobs", body=self.body(session="session-003"), token=TOKEN)
        time.sleep(0.5)
        status, body = js("POST", f"/api/jobs/{job['id']}/cancel", token=TOKEN)
        self.assertEqual(status, 200)
        for _ in range(100):
            _, job = js("GET", f"/api/jobs/{job['id']}")
            if job["state"] == "cancelled":
                break
            time.sleep(0.1)
        self.assertEqual(job["state"], "cancelled")

    def test_existing_hand_outputs_need_regenerate(self):
        dataset = self.make_session()
        (dataset / "metadata.json").write_text(json.dumps(
            {"schema_version": "2.0", "streams": {"hands": {"status": "complete"}}}))
        status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (409, "hand_exists"))

    def test_model_failure_is_refused_without_loading_the_model(self):
        self.make_session()
        with mock.patch.object(jobs, "verify_model", side_effect=RuntimeError("changed")):
            status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (503, "model_integrity"))
        self.assertNotIn(str(self.tmp), json.dumps(body))

    def test_writer_lock_blocks_a_second_publication(self):
        dataset = self.make_session()
        (dataset / ".pose-lock").write_text("pid=1")
        status, body = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        self.assertEqual((status, body["code"]), (409, "writer_lock_present"))

    def test_hand_build_does_not_change_the_pose_build_command(self):
        cmd = jobs.default_command({"kind": "pose_build", "person": "p", "session": "session-001", "regenerate": True})
        self.assertIn("--regenerate", cmd)
        self.assertTrue(str(jobs.SCRIPTS["hand_build"]).endswith("build_hand_pipeline.py"))

    def test_progress_lines_map_to_stages(self):
        state = {}
        self.assertEqual(jobs.parse_progress("hand_build", "Stage 2 of 4: hand tracking", state), ("hand tracking", 0.25))
        self.assertEqual(jobs.parse_progress("hand_build", "Hand Motion pipeline complete.", state), ("validation", 1.0))
        self.assertIsNone(jobs.parse_progress("hand_build", "some other line", state))

    def test_token_is_never_written_to_the_job_log(self):
        self.make_session()
        _, job = js("POST", "/api/jobs", body=self.body(), token=TOKEN)
        for _ in range(100):
            _, job = js("GET", f"/api/jobs/{job['id']}")
            if job["state"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        log = self.svc.runner.log_path(job["id"]).read_text(encoding="utf-8")
        self.assertNotIn(TOKEN, log)


class HandReadTests(tcl.GestureCase):
    """Read API against a complete synthetic session (H1, H2, H3)."""

    def setUp(self):
        super().setUp()
        self.build_all()
        self.assertEqual(self.run_gestures()[0], 0)
        self.person, self.session = "demo_person", "session-001"
        self.path = f"/api/persons/{self.person}/sessions/{self.session}/hands"

    def get(self, sub, query=""):
        return js("GET", f"{self.path}/{sub}", query)

    def test_summary_reports_the_streams(self):
        status, body = self.get("summary")
        self.assertEqual(status, 200)
        self.assertTrue(body["tracking"])
        self.assertTrue(body["gestures"])
        self.assertIn("left", body["status_counts"])

    def test_landmarks_are_paged_and_null_where_no_hand(self):
        status, body = self.get("landmarks", "side=right&start=0&limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["frames"]), 3)
        self.assertIsNotNone(body["frames"][0]["points"])
        self.assertIsNone(body["frames"][1]["points"]) if body["frames"][1]["status"] == "not_detected" else None
        self.assertEqual(len(body["frames"][0]["points"]), 21)

    def test_landmark_page_is_bounded(self):
        status, body = self.get("landmarks", "side=left&start=0&limit=500")
        self.assertEqual((status, body["code"]), (400, "invalid_pagination"))

    def test_bad_side_is_rejected(self):
        status, body = self.get("landmarks", "side=middle&start=0&limit=3")
        self.assertEqual(status, 400)

    def test_table_pages_report_total_and_slice(self):
        status, body = self.get("gestures/observations", "offset=2&limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(body["total"], tcl.N)
        self.assertEqual(len(body["rows"]), 3)
        self.assertEqual(body["rows"][0]["frame"], 2)

    def test_invalid_pagination_is_rejected(self):
        for query in ("offset=-1&limit=5", "offset=0&limit=0", "offset=0&limit=5000"):
            status, body = self.get("tracks", query)
            self.assertEqual((status, body["code"]), (400, "invalid_pagination"), query)

    def test_events_and_motion_are_listed(self):
        status, body = self.get("gestures/events", "offset=0&limit=10")
        self.assertEqual(status, 200)
        self.assertEqual(body["rows"][0]["gesture"], "pointing")
        status, body = self.get("motion", "offset=0&limit=2")
        self.assertEqual(len(body["rows"]), 2)

    def test_quality_summaries_are_returned(self):
        status, body = self.get("quality/tracking")
        self.assertEqual(status, 200)
        self.assertIn("tracks_created", body)
        status, body = self.get("quality/gestures")
        self.assertIn("events", body)
        status, body = self.get("quality/nonsense")
        self.assertEqual(status, 400)

    def test_validation_report_has_no_errors_for_the_valid_session(self):
        status, body = self.get("validation")
        self.assertEqual(status, 200)
        self.assertTrue(body["valid"])
        self.assertEqual(body["errors"], [])

    def test_overlay_frame_is_a_jpeg(self):
        status, raw, headers = call("GET", f"{self.path}/frames/0/overlay.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(raw[:2], b"\xff\xd8")

    def test_frame_out_of_range_is_not_found(self):
        status, body = self.get("landmarks", "side=left&start=100000&limit=1")
        self.assertEqual(status, 200)  # past the end: empty page, not an error
        self.assertEqual(body["frames"], [])
        status, _raw, _h = call("GET", f"{self.path}/frames/100000/overlay.jpg")
        self.assertEqual(status, 404)

    def test_unknown_session_is_not_found(self):
        status, body = js("GET", "/api/persons/demo_person/sessions/session-999/hands/summary")
        self.assertEqual((status, body["code"]), (404, "no_session"))

    def test_path_traversal_is_rejected(self):
        status, body = js("GET", "/api/persons/demo_person/sessions/..%2F..%2Fx/hands/summary")
        self.assertEqual(status, 400)

    def test_missing_hand_analysis_is_structured(self):
        (self.dataset / "metadata.json").write_text(json.dumps({**json.loads(
            (self.dataset / "metadata.json").read_text()), "streams": {}}))
        status, body = self.get("summary")
        self.assertEqual((status, body["code"]), (404, "no_hand_analysis"))

    def test_corrupt_table_is_reported_without_internal_detail(self):
        (self.dataset / "hands" / "tracks.csv").write_bytes(b"\xff\xfe\x00bad")
        status, body = self.get("tracks", "offset=0&limit=5")
        self.assertEqual((status, body["code"]), (422, "dataset_corrupt"))
        self.assertNotIn(str(self.dataset), json.dumps(body))

    def test_read_routes_do_not_modify_the_dataset(self):
        before = {p.name: p.read_bytes() for p in (self.dataset / "hands").iterdir() if p.is_file()}
        for sub in ("summary", "tracks?offset=0&limit=5", "validation"):
            path, _, query = sub.partition("?")
            self.get(path, query)
        after = {p.name: p.read_bytes() for p in (self.dataset / "hands").iterdir() if p.is_file()}
        self.assertEqual(before, after)


class LayerTests(unittest.TestCase):
    def test_hand_layer_is_marked_implemented_and_keeps_the_licence_note_out_of_the_claims(self):
        from api import system
        rows = {r["key"]: r for r in system.layers()}
        self.assertTrue(rows["hand"]["implemented"])
        self.assertIn("not calibrated", rows["hand"]["will_show"])
        self.assertIn("not cleared", rows["hand"]["needed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
