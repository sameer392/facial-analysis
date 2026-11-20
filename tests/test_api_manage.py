"""Management API: people, uploads, typed-confirmation deletes and job history. Synthetic files in temp folders only."""

import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
from api import jobs, manage  # noqa: E402
from api.catalog import ApiError  # noqa: E402
from test_api_p4 import call  # noqa: E402

TOKEN = "t-manage"
MP4_HEAD = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00"


def call_raw(app, method, path, body: bytes, token=None):
    headers = [(b"content-type", b"video/mp4"), (b"content-length", str(len(body)).encode())]
    if token is not None:
        headers.append((b"x-facial-token", token.encode()))
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(), "query_string": b"",
             "headers": headers, "server": ("127.0.0.1", 8000), "scheme": "http", "root_path": "",
             "http_version": "1.1"}
    result = {"body": []}
    sent = {"done": False}

    async def receive():
        if sent["done"]:
            return {"type": "http.disconnect"}
        sent["done"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
        elif message["type"] == "http.response.body":
            result["body"].append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return result["status"], json.loads(b"".join(result["body"]) or b"null")


class ManageTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="manage_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        pp.create_person("demo_person")
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token=TOKEN,
                       command_for=lambda job: [sys.executable, "-c", "import time; time.sleep(30)"])

    def tearDown(self):
        svc = jobs.service()
        for job in svc.store.list(limit=100):
            if job["state"] in jobs.ACTIVE:
                svc.runner.cancel(job["id"])
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def add_session(self, session, with_dataset=False):
        video = pp.session_video("demo_person", session)
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(MP4_HEAD + b"data")
        if with_dataset:
            folder = pp.session_dataset_dir("demo_person", session)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "facial_data.json").write_text("[]", encoding="utf-8")


class PersonTests(ManageTestCase):
    def test_create_person_needs_the_token(self):
        from api.main import app
        status, body = call(app, "POST", "/api/persons", {"name": "new_person"})
        self.assertEqual(status, 401)
        status, body = call(app, "POST", "/api/persons", {"name": "new_person"}, token=TOKEN)
        self.assertEqual(status, 201)
        self.assertTrue(pp.person_recordings_dir("new_person").is_dir())

    def test_duplicate_and_bad_names_are_refused(self):
        with self.assertRaises(ApiError) as ctx:
            manage.create_person("demo_person")
        self.assertEqual(ctx.exception.code, "exists")
        with self.assertRaises(ApiError) as ctx:
            manage.create_person("Bad Name")
        self.assertEqual(ctx.exception.status, 400)

    def test_delete_person_needs_the_typed_name(self):
        self.add_session("session-001", with_dataset=True)
        with self.assertRaises(ApiError) as ctx:
            manage.delete_person("demo_person", "demo")
        self.assertEqual(ctx.exception.code, "confirmation_mismatch")
        self.assertTrue(pp.person_recordings_dir("demo_person").is_dir())
        self.assertTrue(manage.delete_person("demo_person", "demo_person")["deleted"])
        self.assertFalse(pp.person_recordings_dir("demo_person").exists())


class UploadTests(ManageTestCase):
    def test_valid_mp4_is_stored_in_pieces(self):
        from api.main import app
        pieces = MP4_HEAD + b"x" * 5000
        status, body = call_raw(app, "PUT", "/api/persons/demo_person/sessions/session-001/video", pieces,
                                token=TOKEN)
        self.assertEqual(status, 201)
        self.assertEqual(body["bytes"], len(pieces))
        self.assertEqual(pp.session_video("demo_person", "session-001").read_bytes(), pieces)
        self.assertEqual(list(pp.person_recordings_dir("demo_person").glob(".upload-*")), [])

    def test_upload_needs_the_token(self):
        from api.main import app
        status, _body = call_raw(app, "PUT", "/api/persons/demo_person/sessions/session-001/video", MP4_HEAD)
        self.assertEqual(status, 401)

    def test_non_mp4_is_refused_and_nothing_is_left(self):
        from api.main import app
        status, body = call_raw(app, "PUT", "/api/persons/demo_person/sessions/session-001/video",
                                b"this is not a video at all", token=TOKEN)
        self.assertEqual((status, body["code"]), (415, "not_mp4"))
        self.assertFalse(pp.session_video("demo_person", "session-001").exists())
        self.assertEqual(list(pp.person_recordings_dir("demo_person").glob(".upload-*")), [])

    def test_existing_video_is_never_overwritten(self):
        self.add_session("session-001")
        before = pp.session_video("demo_person", "session-001").read_bytes()
        with self.assertRaises(ApiError) as ctx:
            writer = manage.UploadWriter("demo_person", "session-001")
        self.assertEqual(ctx.exception.code, "exists")
        self.assertEqual(pp.session_video("demo_person", "session-001").read_bytes(), before)

    def test_too_large_is_refused(self):
        with mock.patch.object(manage, "MAX_UPLOAD_BYTES", 100):
            writer = manage.UploadWriter("demo_person", "session-001")
            writer.write(MP4_HEAD)
            with self.assertRaises(ApiError) as ctx:
                writer.write(b"x" * 200)
        self.assertEqual(ctx.exception.code, "too_large")
        self.assertFalse(pp.session_video("demo_person", "session-001").exists())

    def test_person_must_exist(self):
        with self.assertRaises(ApiError) as ctx:
            manage.UploadWriter("nobody", "session-001")
        self.assertEqual(ctx.exception.status, 404)


class DeleteTests(ManageTestCase):
    def test_dataset_delete_needs_the_exact_session_name(self):
        self.add_session("session-001", with_dataset=True)
        with self.assertRaises(ApiError):
            manage.delete_dataset("demo_person", "session-001", "session-001")
        self.assertTrue(pp.session_dataset_dir("demo_person", "session-001").is_dir())
        result = manage.delete_dataset("demo_person", "session-001", "demo_person/session-001")
        self.assertTrue(result["deleted"])
        self.assertFalse(pp.session_dataset_dir("demo_person", "session-001").exists())
        self.assertTrue(pp.session_video("demo_person", "session-001").is_file(), "the video is kept")

    def test_video_delete_keeps_nothing_it_should_not(self):
        self.add_session("session-001", with_dataset=True)
        result = manage.delete_video("demo_person", "session-001", "demo_person/session-001")
        self.assertTrue(result["deleted"])
        self.assertFalse(pp.session_video("demo_person", "session-001").exists())

    def test_delete_is_refused_while_a_job_runs_on_the_session(self):
        self.add_session("session-001", with_dataset=True)
        store = jobs.service().store
        store.insert("demo_person", "session-001", "face_build", True)
        with self.assertRaises(ApiError) as ctx:
            manage.delete_dataset("demo_person", "session-001", "demo_person/session-001")
        self.assertEqual(ctx.exception.code, "job_active")
        self.assertTrue(pp.session_dataset_dir("demo_person", "session-001").is_dir())


class JobHistoryTests(ManageTestCase):
    def test_clear_history_needs_the_phrase_and_keeps_active_jobs(self):
        store = jobs.service().store
        done = store.insert("demo_person", "session-001", "face_build", False)
        store.update(done["id"], state="succeeded")
        active = store.insert("demo_person", "session-002", "face_build", False)
        store.update(active["id"], state="running")
        log = jobs.service().runner.log_path(done["id"])
        log.write_text("finished\n", encoding="utf-8")
        with self.assertRaises(ApiError):
            manage.clear_job_history("yes")
        result = manage.clear_job_history(manage.JOB_HISTORY_PHRASE)
        self.assertEqual(result["removed"], 1)
        self.assertIsNone(store.get(done["id"]))
        self.assertIsNotNone(store.get(active["id"]))
        self.assertFalse(log.exists())

    def test_clear_history_route_needs_the_token(self):
        from api.main import app
        status, _body = call(app, "POST", "/api/jobs/history/clear", {"confirm": manage.JOB_HISTORY_PHRASE})
        self.assertEqual(status, 401)
        status, body = call(app, "POST", "/api/jobs/history/clear", {"confirm": manage.JOB_HISTORY_PHRASE},
                            token=TOKEN)
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class RenameTests(ManageTestCase):
    def test_session_rename_moves_the_recording(self):
        self.add_session("session-001")
        result = manage.rename_session("demo_person", "session-001", "session-009")
        self.assertEqual(result["renamed_to"], "session-009")
        self.assertFalse(pp.session_video("demo_person", "session-001").exists())
        self.assertTrue(pp.session_video("demo_person", "session-009").is_file())

    def test_session_rename_is_refused_while_processed_data_exists(self):
        self.add_session("session-001", with_dataset=True)
        with self.assertRaises(ApiError) as ctx:
            manage.rename_session("demo_person", "session-001", "session-009")
        self.assertEqual(ctx.exception.code, "has_data")
        self.assertTrue(pp.session_video("demo_person", "session-001").is_file())

    def test_session_rename_refuses_an_existing_name_and_bad_names(self):
        self.add_session("session-001")
        self.add_session("session-002")
        with self.assertRaises(ApiError) as ctx:
            manage.rename_session("demo_person", "session-001", "session-002")
        self.assertEqual(ctx.exception.code, "exists")
        with self.assertRaises(ApiError) as ctx:
            manage.rename_session("demo_person", "session-001", "Not A Name")
        self.assertEqual(ctx.exception.status, 400)

    def test_session_rename_is_refused_while_a_job_runs(self):
        self.add_session("session-001")
        jobs.service().store.insert("demo_person", "session-001", "face_build", False)
        with self.assertRaises(ApiError) as ctx:
            manage.rename_session("demo_person", "session-001", "session-009")
        self.assertEqual(ctx.exception.code, "job_active")

    def test_person_rename_moves_recordings_and_empty_data_folders(self):
        self.add_session("session-001")
        manage.rename_person("demo_person", "new_person")
        self.assertTrue(pp.session_video("new_person", "session-001").is_file())
        self.assertFalse(pp.person_recordings_dir("demo_person").exists())
        self.assertTrue(pp.person_dataset_dir("new_person").is_dir())

    def test_person_rename_is_refused_while_processed_data_exists(self):
        self.add_session("session-001", with_dataset=True)
        with self.assertRaises(ApiError) as ctx:
            manage.rename_person("demo_person", "new_person")
        self.assertEqual(ctx.exception.code, "has_data")
        self.assertTrue(pp.person_recordings_dir("demo_person").is_dir())

    def test_rename_routes_need_the_token(self):
        from api.main import app
        status, _ = call(app, "POST", "/api/persons/demo_person/sessions/session-001/rename", {"to": "session-009"})
        self.assertEqual(status, 401)


class TokenFileTests(unittest.TestCase):
    def test_token_is_created_once_and_reused(self):
        tmp = Path(tempfile.mkdtemp(prefix="token_"))
        try:
            path = tmp / "config" / "api_token.txt"
            first = jobs.load_or_create_token(path)
            self.assertEqual(jobs.load_or_create_token(path), first)
            self.assertEqual(path.read_text(encoding="utf-8").strip(), first)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_empty_token_file_is_refused(self):
        tmp = Path(tempfile.mkdtemp(prefix="token_"))
        try:
            path = tmp / "api_token.txt"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                jobs.load_or_create_token(path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
