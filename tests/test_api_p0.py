"""Phase P0 tests: the read-only API must agree with project_paths and must never write (synthetic data)."""

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
import schema_v2 as s2  # noqa: E402
import session_writer_v2 as w  # noqa: E402
import test_session_writer_v2 as tsw  # noqa: E402  (synthetic V2 payload)
from api import catalog  # noqa: E402
from api.catalog import ApiError  # noqa: E402


def snapshot(root: Path) -> dict:
    """Every file path with its size and modification time, to prove that nothing was written."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            st = p.stat()
            out[str(p)] = (st.st_size, st.st_mtime_ns)
    return out


def call(app, method: str, path: str):
    """Call the ASGI app directly (no HTTP client library needed). Returns (status, json body)."""
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": [], "server": ("127.0.0.1", 8000),
             "scheme": "http", "root_path": "", "http_version": "1.1"}
    result = {"body": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
        elif message["type"] == "http.response.body":
            result["body"].append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return result["status"], json.loads(b"".join(result["body"]) or b"null")


class P0TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p0_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        # person A: one V1 session (with video), one V2 session (with video), one video only
        for name in ("session-001", "session-002", "session-003"):
            pp.session_video("person_a", name).parent.mkdir(parents=True, exist_ok=True)
            pp.session_video("person_a", name).write_bytes(b"synthetic video placeholder")
        v1 = pp.session_dataset_dir("person_a", "session-001")
        v1.mkdir(parents=True)
        (v1 / "facial_data.json").write_text("[]", encoding="utf-8")
        v2 = pp.session_dataset_dir("person_a", "session-002")
        w.write_v2_session(v2, tsw.make_payload())
        pp.person_recordings_dir("person_b").mkdir(parents=True)  # a person with no sessions

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)


class ParityTests(P0TestCase):
    def test_persons_match_project_paths(self):
        self.assertEqual(catalog.list_persons(), pp.list_persons())
        self.assertEqual(catalog.list_persons(), ["person_a", "person_b"])

    def test_sessions_match_project_paths_and_report_layouts(self):
        rows = catalog.list_sessions("person_a")
        self.assertEqual([r["session_id"] for r in rows], pp.list_sessions("person_a"))
        layouts = {r["session_id"]: r["layout"] for r in rows}
        self.assertEqual(layouts["session-001"], "v1")
        self.assertEqual(layouts["session-002"], "v2")
        self.assertEqual(layouts["session-003"], "none")

    def test_person_without_sessions_gives_an_empty_list(self):
        self.assertEqual(catalog.list_sessions("person_b"), [])

    def test_session_rows_say_whether_a_video_exists(self):
        rows = {r["session_id"]: r for r in catalog.list_sessions("person_a")}
        self.assertTrue(all(r["has_video"] for r in rows.values()))


class ValidationTests(P0TestCase):
    def test_path_traversal_names_are_refused(self):
        for bad in ("..", "../person_a", "person_a/..", "Bad Name", ""):
            with self.subTest(name=bad), self.assertRaises(ApiError) as ctx:
                catalog.list_sessions(bad)
            self.assertEqual(ctx.exception.status, 400)
            self.assertEqual(ctx.exception.code, "invalid_name")

    def test_unknown_person_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            catalog.list_sessions("nobody")
        self.assertEqual(ctx.exception.status, 404)


class ReadOnlyTests(P0TestCase):
    def test_catalog_calls_write_nothing(self):
        before = snapshot(self.tmp)
        catalog.health()
        catalog.list_persons()
        catalog.list_sessions("person_a")
        catalog.list_sessions("person_b")
        with self.assertRaises(ApiError):
            catalog.list_sessions("nobody")
        self.assertEqual(snapshot(self.tmp), before)

    def test_every_route_is_read_only(self):
        from api.main import app

        methods = set()
        for route in app.routes:
            if route.path.startswith("/api"):
                methods |= set(getattr(route, "methods", set()))
        self.assertTrue(methods <= {"GET", "HEAD"}, methods)

    def test_health_does_not_reveal_absolute_paths(self):
        body = json.dumps(catalog.health())
        self.assertNotIn(str(self.tmp), body)
        self.assertEqual(catalog.health()["read_only"], True)


class HttpTests(P0TestCase):
    def test_http_endpoints_return_the_expected_status_codes(self):
        from api.main import app

        status, body = call(app, "GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

        status, body = call(app, "GET", "/api/persons")
        self.assertEqual((status, body), (200, ["person_a", "person_b"]))

        status, body = call(app, "GET", "/api/persons/person_a/sessions")
        self.assertEqual(status, 200)
        self.assertEqual([r["session_id"] for r in body], ["session-001", "session-002", "session-003"])

        status, body = call(app, "GET", "/api/persons/..%2Fx/sessions")
        self.assertIn(status, (400, 404))

        status, body = call(app, "GET", "/api/persons/nobody/sessions")
        self.assertEqual(status, 404)
        self.assertEqual(body["code"], "not_found")

    def test_http_requests_write_nothing(self):
        from api.main import app

        before = snapshot(self.tmp)
        call(app, "GET", "/api/persons")
        call(app, "GET", "/api/persons/person_a/sessions")
        call(app, "GET", "/api/health")
        self.assertEqual(snapshot(self.tmp), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
