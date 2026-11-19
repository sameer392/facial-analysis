"""Phase P5 tests: model integrity, job logs and settings. Synthetic files in temporary folders only."""

import dataclasses
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "extraction"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
from api import jobs, system  # noqa: E402
from model_registry import MODELS, ModelSpec, sha256_file  # noqa: E402
from test_api_p4 import call  # noqa: E402

TOKEN = "test-token-p5"


class P5TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p5_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        jobs.configure(self.tmp / "logs" / "jobs.sqlite3", self.tmp / "logs" / "jobs", token=TOKEN,
                       command_for=lambda job: [sys.executable, "-c", "print('nothing')"])

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        jobs._service = None
        shutil.rmtree(self.tmp, ignore_errors=True)


class ModelIntegrityTests(P5TestCase):
    def setUp(self):
        super().setUp()
        self.models_dir = self.tmp / "models"
        self.models_dir.mkdir()
        self.content = b"synthetic model bytes"
        self.good = self.models_dir / "good.task"
        self.good.write_bytes(self.content)
        self.good_sha = sha256_file(self.good)

    def test_missing_file_is_reported_as_missing(self):
        rows = system.models(specs={"face": MODELS["face"]}, models_dir=self.models_dir)
        self.assertEqual(rows[0]["sha256_status"], "missing")
        self.assertFalse(rows[0]["present"])
        self.assertIsNone(rows[0]["actual_sha256"])

    def test_matching_file_is_reported_as_match(self):
        spec = ModelSpec(key="t", filename="good.task", sha256=self.good_sha, version="v",
                         source_url="u", licence_status="UNVERIFIED")
        rows = system.models(specs={"t": spec}, models_dir=self.models_dir)
        self.assertEqual(rows[0]["sha256_status"], "match")
        self.assertEqual(rows[0]["actual_sha256"], self.good_sha)

    def test_changed_file_is_reported_as_mismatch(self):
        spec = dataclasses.replace(MODELS["face"], filename="good.task", sha256="00" * 32)
        rows = system.models(specs={"face": spec}, models_dir=self.models_dir)
        self.assertEqual(rows[0]["sha256_status"], "mismatch")

    def test_licence_text_is_shown_exactly_as_recorded(self):
        rows = system.models(specs=MODELS, models_dir=self.models_dir)
        by_key = {r["key"]: r for r in rows}
        for key, spec in MODELS.items():
            self.assertEqual(by_key[key]["licence_status"], spec.licence_status)
        # the recorded text may say "not cleared"; the API must never report a licence as cleared
        self.assertFalse(any(r["licence_status"].strip().lower() == "cleared" for r in rows))

    def test_hash_is_cached_until_the_file_changes(self):
        spec = dataclasses.replace(MODELS["face"], filename="good.task", sha256=self.good_sha)
        system.models(specs={"face": spec}, models_dir=self.models_dir)
        self.assertEqual(system.models(specs={"face": spec}, models_dir=self.models_dir)[0]["sha256_status"], "match")
        self.good.write_bytes(b"changed")
        self.assertEqual(system.models(specs={"face": spec}, models_dir=self.models_dir)[0]["sha256_status"], "mismatch")


class LogTests(P5TestCase):
    def test_redaction_replaces_project_and_home_folders(self):
        text = f"$ {ROOT}\\.venv\\python.exe {ROOT}/src/x.py {Path.home()}\\Documents"
        redacted = system.redact(text)
        self.assertNotIn(str(ROOT), redacted)
        self.assertIn("<project>", redacted)
        self.assertNotIn(str(Path.home()), redacted)

    def test_log_list_and_tail_for_a_job(self):
        store = jobs.service().store
        job = store.insert("demo_person", "session-001", "v1_analyse", False)
        log = jobs.service().runner.log_path(job["id"])
        log.write_text("\n".join(f"line {i} at {ROOT}" for i in range(5)) + "\n", encoding="utf-8")
        listed = system.job_logs()
        self.assertEqual(listed[0]["job_id"], job["id"])
        self.assertGreater(listed[0]["log_bytes"], 0)
        tail = system.job_log_tail(job["id"], lines=2)
        self.assertEqual(tail["total_lines"], 5)
        self.assertEqual(len(tail["lines"]), 2)
        self.assertTrue(all("<project>" in line for line in tail["lines"]))

    def test_tail_of_unknown_job_is_not_found(self):
        from api.catalog import ApiError

        with self.assertRaises(ApiError) as ctx:
            system.job_log_tail("0" * 32)
        self.assertEqual(ctx.exception.status, 404)


class SettingsTests(P5TestCase):
    def test_settings_show_names_not_full_paths_and_no_token(self):
        data = system.settings()
        text = json.dumps(data)
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn(TOKEN, text)
        self.assertTrue(data["api_token_configured"])
        self.assertEqual(data["bind_address"], "127.0.0.1")


class RouteTests(P5TestCase):
    def test_p5_routes_respond_without_a_token(self):
        from api.main import app

        status, body = call(app, "GET", "/api/models")
        self.assertEqual(status, 200)
        self.assertEqual({r["key"] for r in body}, set(MODELS))
        status, body = call(app, "GET", "/api/logs/jobs")
        self.assertEqual(status, 200)
        status, body = call(app, "GET", "/api/settings")
        self.assertEqual(status, 200)
        self.assertNotIn(TOKEN, json.dumps(body))

    def test_p5_routes_are_read_only(self):
        from api.main import app

        for route in app.routes:
            if route.path in ("/api/models", "/api/logs/jobs", "/api/logs/jobs/{job_id}", "/api/settings"):
                self.assertEqual(set(route.methods) - {"GET", "HEAD"}, set(), route.path)


class LayerTests(P5TestCase):
    def test_missing_extractors_are_reported_as_not_implemented(self):
        empty = self.tmp / "project"
        rows = {r["key"]: r for r in system.layers(root=empty)}
        self.assertEqual(set(rows), {"audio", "hand"})
        self.assertFalse(rows["audio"]["implemented"])
        self.assertFalse(rows["hand"]["implemented"])
        self.assertIn("needed", rows["hand"])

    def test_present_extractor_is_reported_as_implemented(self):
        root = self.tmp / "project"
        (root / "src" / "extraction").mkdir(parents=True)
        (root / "src" / "extraction" / "audio_extractor.py").write_text("", encoding="utf-8")
        rows = {r["key"]: r for r in system.layers(root=root)}
        self.assertTrue(rows["audio"]["implemented"])
        self.assertFalse(rows["hand"]["implemented"])

    def test_layers_route_is_read_only_and_answers(self):
        from api.main import app

        status, body = call(app, "GET", "/api/layers")
        self.assertEqual(status, 200)
        self.assertEqual({r["key"] for r in body}, {"audio", "hand"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
