"""Live progress stream for a job. Fake worker only; no model."""

import asyncio
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

from api import job_events, jobs  # noqa: E402
from test_api_p4 import call  # noqa: E402


def stream_call(app, path):
    """Like test_api_p4.call, but the client disconnects after the response, as a browser does."""
    scope = {"type": "http", "method": "GET", "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": [], "server": ("127.0.0.1", 8000), "scheme": "http", "root_path": "", "http_version": "1.1"}
    result = {"body": []}
    messages = [{"type": "http.request", "body": b"", "more_body": False}]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
        elif message["type"] == "http.response.body":
            result["body"].append(message.get("body", b""))

    asyncio.run(asyncio.wait_for(app(scope, receive, send), timeout=10))
    return result["status"], b"".join(result["body"])


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="job_events_"))
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token="t",
                       command_for=lambda job: [sys.executable, "-c", "pass"])
        self.store = jobs.service().store

    def tearDown(self):
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def collect(self, job_id, **kwargs):
        async def run():
            out = []
            async for chunk in job_events.event_stream(job_id, poll_seconds=0.01, **kwargs):
                out.append(chunk)
            return out
        return asyncio.run(run())

    def test_a_finished_job_sends_its_state_once_and_closes(self):
        job = self.store.insert("demo_person", "session-001", "face_build", False)
        self.store.update(job["id"], state="succeeded", stage="verify", progress=1.0)
        chunks = self.collect(job["id"])
        data = [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ")]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["state"], "succeeded")
        self.assertTrue(chunks[0].startswith("retry:"))

    def test_progress_changes_are_sent_while_the_job_runs(self):
        job = self.store.insert("demo_person", "session-001", "face_build", False)
        self.store.update(job["id"], state="running", stage="extract", progress=0.1)

        def finish_later():
            time.sleep(0.1)
            self.store.update(job["id"], stage="extract", progress=0.6)
            time.sleep(0.1)
            self.store.update(job["id"], state="succeeded", stage="verify", progress=1.0)

        threading.Thread(target=finish_later, daemon=True).start()
        chunks = self.collect(job["id"])
        states = [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ")]
        self.assertGreaterEqual(len(states), 3)
        self.assertEqual(states[-1]["state"], "succeeded")
        progress = [s["progress"] for s in states]
        self.assertEqual(progress, sorted(progress))

    def test_the_stream_route_answers_and_unknown_jobs_are_not_found(self):
        from api.main import app
        job = self.store.insert("demo_person", "session-001", "face_build", False)
        self.store.update(job["id"], state="succeeded")
        status, body = stream_call(app, f"/api/jobs/{job['id']}/events")
        self.assertEqual(status, 200)
        self.assertIn(b"data: ", body)
        status, _ = stream_call(app, "/api/jobs/" + "0" * 32 + "/events")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
