"""The combined Face data job: extraction then analysis. Fake scripts and synthetic folders only."""

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
from api import jobs  # noqa: E402
from api.catalog import ApiError  # noqa: E402
import build_face_data  # noqa: E402

FAKE_STEP = (
    "import sys\n"
    "open({log!r}, 'a').write(' '.join([{name!r}] + sys.argv[1:]) + '\\n')\n"
    "sys.exit({code})\n"
)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="face_build_"))
        self.log = self.tmp / "order.txt"
        for name, code in [("step_a.py", 0), ("step_b.py", 3), ("step_c.py", 0)]:
            (self.tmp / name).write_text(FAKE_STEP.format(log=str(self.log), name=name, code=code), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_steps_run_in_order_and_stop_at_the_first_failure(self):
        code = build_face_data.run_steps("demo_person", "session-001",
                                         steps=("step_a.py", "step_b.py", "step_c.py"), here=self.tmp)
        self.assertEqual(code, 3)
        lines = self.log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["step_a.py demo_person session-001", "step_b.py demo_person session-001"])

    def test_the_real_steps_are_extraction_then_analysis(self):
        self.assertEqual(build_face_data.STEPS, ("extract_faces.py", "analyze_faces.py"))


class FaceBuildJobTests(unittest.TestCase):
    def test_command_runs_the_combined_script_without_flags(self):
        job = {"kind": "face_build", "person": "demo_person", "session": "session-001", "regenerate": True}
        cmd = jobs.default_command(job)
        self.assertEqual(Path(cmd[1]).name, "build_face_data.py")
        self.assertEqual(cmd[2:], ["demo_person", "session-001"])

    def test_progress_covers_extraction_then_analysis(self):
        state = {}
        self.assertEqual(jobs.parse_progress("face_build", "Total frames: 100", state), ("extract", 0.0))
        self.assertEqual(jobs.parse_progress("face_build", "Processed: 50", state), ("extract", 0.4))
        self.assertEqual(jobs.parse_progress("face_build", "Processing completed!", state), ("publish", 0.792))
        stage, fraction = jobs.parse_progress("face_build", "CSV created: x", state)
        self.assertEqual(stage, "assemble")
        self.assertAlmostEqual(fraction, 0.85)
        stage, fraction = jobs.parse_progress("face_build", "Metadata: x", state)
        self.assertEqual(stage, "verify")
        self.assertAlmostEqual(fraction, 0.98)

    def test_the_combined_job_is_an_accepted_kind(self):
        self.assertIn("face_build", jobs.KINDS)


class FaceBuildRefusalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="face_refuse_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        video = pp.session_video("demo_person", "session-001")
        video.parent.mkdir(parents=True)
        video.write_bytes(b"x")
        self.dataset = pp.session_dataset_dir("demo_person", "session-001")
        self.dataset.mkdir(parents=True)
        self.patch = mock.patch.object(jobs, "verify_model", return_value=None)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refused_when_face_data_exists(self):
        (self.dataset / "facial_data.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(ApiError) as ctx:
            jobs.check_job("face_build", "demo_person", "session-001", False)
        self.assertEqual(ctx.exception.code, "face_data_exists")

    def test_refused_when_the_model_fails_its_check(self):
        with mock.patch.object(jobs, "verify_model", side_effect=Exception("hash mismatch")):
            with self.assertRaises(ApiError) as ctx:
                jobs.check_job("face_build", "demo_person", "session-001", False)
        self.assertEqual(ctx.exception.code, "model_integrity")

    def test_allowed_on_a_recording_with_no_face_data(self):
        jobs.check_job("face_build", "demo_person", "session-001", False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
