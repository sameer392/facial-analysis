"""Phase P2 tests: the V1 data API must give the same numbers as the Python analysis (synthetic data)."""

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
import facial_analyzer as fa  # noqa: E402
import demo_data  # noqa: E402
from api import v1_data  # noqa: E402
from api.catalog import ApiError  # noqa: E402


class P2TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p2_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        v1_data.clear_cache()
        # two synthetic V1 sessions for one person (same generator as the demo)
        demo_data.generate_session(self.tmp, "session-001", demo_data.SESSIONS["session-001"])
        demo_data.generate_session(self.tmp, "session-002", demo_data.SESSIONS["session-002"])

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        v1_data.clear_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def python_analysis(self, session_id):
        return fa.analyse_session(fa.make_session("demo_person", session_id))


class SummaryParityTests(P2TestCase):
    def test_summary_quality_matches_the_python_analysis(self):
        result = v1_data.summary("demo_person", "session-001")
        expected = self.python_analysis("session-001")
        for key, value in expected.quality.items():
            if key in ("session_id", "interruptions", "timestamp_issues", "scope", "frame_size_known"):
                continue
            with self.subTest(key=key):
                got = result["quality"][key]
                if isinstance(value, float) and math.isnan(value):
                    self.assertIsNone(got)
                else:
                    self.assertAlmostEqual(got, value, places=9)
        self.assertEqual(result["neutral_frames"], expected.neutral_frames)
        self.assertEqual(result["blink_events"], len(expected.blinks))
        self.assertEqual(result["frames"], len(expected.table))

    def test_summary_carries_the_scope_statement(self):
        self.assertIn("does not measure", v1_data.summary("demo_person", "session-001")["quality"]["scope"])

    def test_missing_v1_data_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            v1_data.summary("demo_person", "session-009")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.code, "no_face_data")


class SeriesParityTests(P2TestCase):
    def test_series_values_match_the_metrics_table(self):
        expected = self.python_analysis("session-001").table
        result = v1_data.series("demo_person", "session-001", ["velocity", "jaw_open", "yaw_deg"])
        self.assertEqual(len(result["time_s"]), len(expected))
        for field in ("velocity", "jaw_open", "yaw_deg"):
            with self.subTest(field=field):
                got = np.array([np.nan if v is None else v for v in result[field]], dtype=float)
                np.testing.assert_allclose(got, expected[field].to_numpy(dtype=float), atol=1e-9, equal_nan=True)

    def test_unknown_field_is_refused(self):
        with self.assertRaises(ApiError) as ctx:
            v1_data.series("demo_person", "session-001", ["not_a_field"])
        self.assertEqual(ctx.exception.code, "invalid_field")
        self.assertEqual(ctx.exception.status, 400)

    def test_nan_values_become_null(self):
        result = v1_data.series("demo_person", "session-001", ["velocity"])
        self.assertIsNone(result["velocity"][0])  # the first frame has no velocity


class BlinkAndLandmarkTests(P2TestCase):
    def test_blink_events_match_the_python_analysis(self):
        expected = self.python_analysis("session-001").blinks
        got = v1_data.blinks("demo_person", "session-001")
        self.assertEqual(len(got), len(expected))
        for row, (_, ref) in zip(got, expected.iterrows()):
            self.assertEqual(row["start_frame"], int(ref["start_frame"]))
            self.assertEqual(row["end_frame"], int(ref["end_frame"]))

    def test_landmark_frame_has_478_points_with_nulls_for_missing_face(self):
        out = v1_data.landmarks("demo_person", "session-001", 0, compensated=False)
        self.assertEqual(len(out["points"]), 478)
        self.assertEqual(len(out["points"][0]), 3)

    def test_landmark_index_out_of_range_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            v1_data.landmarks("demo_person", "session-001", 10_000, compensated=True)
        self.assertEqual(ctx.exception.status, 404)


class ComparisonParityTests(P2TestCase):
    def test_comparison_summary_matches_each_session(self):
        out = v1_data.comparison("demo_person", ["session-001", "session-002"])
        by_id = {row["session_id"]: row for row in out["sessions"]}
        for sid in ("session-001", "session-002"):
            with self.subTest(session=sid):
                ref = self.python_analysis(sid)
                self.assertEqual(by_id[sid]["frames"], len(ref.table))
                self.assertEqual(by_id[sid]["blink_events"], len(ref.blinks))

    def test_histogram_shares_sum_to_one_for_each_session(self):
        out = v1_data.comparison("demo_person", ["session-001", "session-002"])
        for name, hist in out["distributions"].items():
            for sid, shares in hist["shares"].items():
                with self.subTest(metric=name, session=sid):
                    finite = [s for s in shares if s is not None]
                    if finite:
                        self.assertAlmostEqual(sum(finite), 1.0, places=9)

    def test_comparison_needs_at_least_one_session(self):
        with self.assertRaises(ApiError) as ctx:
            v1_data.comparison("demo_person", [])
        self.assertEqual(ctx.exception.code, "invalid_request")


class HttpP2Tests(P2TestCase):
    def test_p2_routes_are_read_only_and_respond(self):
        from api.main import app
        from test_api_p0 import call

        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-001/v1/summary")
        self.assertEqual(status, 200)
        self.assertIn("quality", body)
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-009/v1/summary")
        self.assertEqual(status, 404)
        # only the job routes (phase P4) may write; every other API route is read-only
        writes = set()
        for route in app.routes:
            if route.path.startswith("/api") and not route.path.startswith("/api/jobs"):
                writes |= set(getattr(route, "methods", set())) - {"GET", "HEAD"}
        self.assertEqual(writes, set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
