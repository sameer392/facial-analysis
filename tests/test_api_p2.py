"""Phase P2 tests: the face data data API must give the same numbers as the Python analysis (synthetic data)."""

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
from api import face_data  # noqa: E402
from api.catalog import ApiError  # noqa: E402


class P2TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p2_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        face_data.clear_cache()
        # two synthetic face data sessions for one person (same generator as the demo)
        demo_data.generate_session(self.tmp, "session-001", demo_data.SESSIONS["session-001"])
        demo_data.generate_session(self.tmp, "session-002", demo_data.SESSIONS["session-002"])

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        face_data.clear_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def python_analysis(self, session_id):
        return fa.analyse_session(fa.make_session("demo_person", session_id))


class SummaryParityTests(P2TestCase):
    def test_summary_quality_matches_the_python_analysis(self):
        result = face_data.summary("demo_person", "session-001")
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
        self.assertIn("does not measure", face_data.summary("demo_person", "session-001")["quality"]["scope"])

    def test_missing_face_data_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            face_data.summary("demo_person", "session-009")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.code, "no_face_data")


class SeriesParityTests(P2TestCase):
    def test_series_values_match_the_metrics_table(self):
        expected = self.python_analysis("session-001").table
        result = face_data.series("demo_person", "session-001", ["velocity", "jaw_open", "yaw_deg"])
        self.assertEqual(len(result["time_s"]), len(expected))
        for field in ("velocity", "jaw_open", "yaw_deg"):
            with self.subTest(field=field):
                got = np.array([np.nan if v is None else v for v in result[field]], dtype=float)
                np.testing.assert_allclose(got, expected[field].to_numpy(dtype=float), atol=1e-9, equal_nan=True)

    def test_unknown_field_is_refused(self):
        with self.assertRaises(ApiError) as ctx:
            face_data.series("demo_person", "session-001", ["not_a_field"])
        self.assertEqual(ctx.exception.code, "invalid_field")
        self.assertEqual(ctx.exception.status, 400)

    def test_nan_values_become_null(self):
        result = face_data.series("demo_person", "session-001", ["velocity"])
        self.assertIsNone(result["velocity"][0])  # the first frame has no velocity


class BlinkAndLandmarkTests(P2TestCase):
    def test_blink_events_match_the_python_analysis(self):
        expected = self.python_analysis("session-001").blinks
        got = face_data.blinks("demo_person", "session-001")
        self.assertEqual(len(got), len(expected))
        for row, (_, ref) in zip(got, expected.iterrows()):
            self.assertEqual(row["start_frame"], int(ref["start_frame"]))
            self.assertEqual(row["end_frame"], int(ref["end_frame"]))

    def test_landmark_frame_has_478_points_with_nulls_for_missing_face(self):
        out = face_data.landmarks("demo_person", "session-001", 0, compensated=False)
        self.assertEqual(len(out["points"]), 478)
        self.assertEqual(len(out["points"][0]), 3)

    def test_landmark_index_out_of_range_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            face_data.landmarks("demo_person", "session-001", 10_000, compensated=True)
        self.assertEqual(ctx.exception.status, 404)


class ComparisonParityTests(P2TestCase):
    def test_comparison_summary_matches_each_session(self):
        out = face_data.comparison("demo_person", ["session-001", "session-002"])
        by_id = {row["session_id"]: row for row in out["sessions"]}
        for sid in ("session-001", "session-002"):
            with self.subTest(session=sid):
                ref = self.python_analysis(sid)
                self.assertEqual(by_id[sid]["frames"], len(ref.table))
                self.assertEqual(by_id[sid]["blink_events"], len(ref.blinks))

    def test_histogram_shares_sum_to_one_for_each_session(self):
        out = face_data.comparison("demo_person", ["session-001", "session-002"])
        for name, hist in out["distributions"].items():
            for sid, shares in hist["shares"].items():
                with self.subTest(metric=name, session=sid):
                    finite = [s for s in shares if s is not None]
                    if finite:
                        self.assertAlmostEqual(sum(finite), 1.0, places=9)

    def test_comparison_needs_at_least_one_session(self):
        with self.assertRaises(ApiError) as ctx:
            face_data.comparison("demo_person", [])
        self.assertEqual(ctx.exception.code, "invalid_request")


class HttpP2Tests(P2TestCase):
    def test_p2_routes_are_read_only_and_respond(self):
        from api.main import app
        from test_api_p0 import call

        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-001/face/summary")
        self.assertEqual(status, 200)
        self.assertIn("quality", body)
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-009/face/summary")
        self.assertEqual(status, 404)
        # only the job routes (phase P4) may write; every other API route is read-only
        # every route that writes must require the API token; reads need no token
        from api.main import require_token
        for route in app.routes:
            if route.path.startswith("/api") and set(getattr(route, "methods", set())) - {"GET", "HEAD"}:
                calls = [d.call for d in route.dependant.dependencies]
                self.assertIn(require_token, calls, route.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TopologyTests(unittest.TestCase):
    def test_topology_has_the_mesh_and_coloured_regions(self):
        from api import face_data
        topo = face_data.topology()
        self.assertGreater(len(topo["mesh"]["edges"]), 1000)
        self.assertTrue(topo["mesh"]["colour"].startswith("#"))
        self.assertIn("lips", topo["regions"])
        for name, region in topo["regions"].items():
            self.assertGreater(len(region["edges"]), 0, name)
            self.assertRegex(region["colour"], r"^#[0-9a-f]{6}$")
            for a, b in region["edges"]:
                self.assertTrue(0 <= a < 478 and 0 <= b < 478)


class FaceOverlayImageTests(P2TestCase):
    def test_overlay_and_original_frames_are_jpegs_of_the_video_size(self):
        import cv2
        import numpy as np
        from api import face_data
        video = pp.session_video("demo_person", "session-001")
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
        for i in range(3):
            writer.write(np.full((48, 64, 3), 60 + 20 * i, dtype=np.uint8))
        writer.release()
        original = face_data.frame_image("demo_person", "session-001", 0, "off", 0.0, 1, 1)
        overlaid = face_data.frame_image("demo_person", "session-001", 0, "hybrid", 0.8, 1, 2)
        for data in (original, overlaid):
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape[:2], (48, 64))
        with self.assertRaises(ApiError) as ctx:
            face_data.frame_image("demo_person", "session-001", 0, "sparkles", 0.8, 1, 2)
        self.assertEqual(ctx.exception.status, 400)


class VideoInfoTests(P2TestCase):
    def test_video_facts_come_from_the_file(self):
        import cv2
        import numpy as np
        from api import face_data
        video = pp.session_video("demo_person", "session-001")
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
        for i in range(5):
            writer.write(np.full((48, 64, 3), 40 + i, dtype=np.uint8))
        writer.release()
        info = face_data.video_info("demo_person", "session-001")
        self.assertEqual((info["width"], info["height"]), (64, 48))
        self.assertEqual(info["frames"], 5)
        self.assertAlmostEqual(info["fps"], 25.0, places=1)
        self.assertAlmostEqual(info["duration_s"], 0.2, places=2)


class BlendshapeFrameTests(P2TestCase):
    def test_frame_values_match_the_facial_data_record_sorted_highest_first(self):
        import json
        path = pp.session_dataset_dir("demo_person", "session-001") / "facial_data.json"
        record = json.loads(path.read_text(encoding="utf-8"))[3]
        result = face_data.blendshape_frame("demo_person", "session-001", 3)
        self.assertEqual(result["index"], 3)
        self.assertEqual(result["face_detected"], bool(record.get("face_detected", False)))
        scores = [score for _, score in result["values"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(
            {name: score for name, score in result["values"]},
            {k: v for k, v in record.get("blendshapes", {}).items() if isinstance(v, (int, float))},
        )

    def test_out_of_range_frame_is_not_found(self):
        with self.assertRaises(ApiError):
            face_data.blendshape_frame("demo_person", "session-001", 10**6)


class BlendshapeSeriesTests(P2TestCase):
    def test_series_follows_the_frames_and_reports_missing_names(self):
        import json
        path = pp.session_dataset_dir("demo_person", "session-001") / "facial_data.json"
        records = json.loads(path.read_text(encoding="utf-8"))
        name = next(iter(records[0].get("blendshapes", {})))
        result = face_data.blendshape_series("demo_person", "session-001", [name, "notAShape"])
        self.assertEqual(len(result["time_s"]), len(records))
        self.assertEqual(result["missing"], ["notAShape"])
        self.assertEqual(list(result["values"]), [name])
        self.assertEqual(len(result["values"][name]), len(records))
        with self.assertRaises(ApiError):
            face_data.blendshape_series("demo_person", "session-001", [])


class BlendshapeCsvTests(P2TestCase):
    def test_csv_path_points_at_the_dataset_file(self):
        path = face_data.blendshapes_csv_path("demo_person", "session-001")
        self.assertEqual(path.name, "facial_blendshapes.csv")
        self.assertTrue(path.is_file())


class ComparisonBlendshapeTests(P2TestCase):
    def test_comparison_gives_blendshape_means_and_landmark_stability(self):
        result = face_data.comparison("demo_person", ["session-001", "session-002"])
        self.assertIn("jawOpen", result["blendshape_means"])
        self.assertEqual(set(result["blendshape_means"]["jawOpen"]), {"session-001", "session-002"})
        for row in result["sessions"]:
            self.assertIn("landmark_stability", row)
        frames = face_data._face_frames("demo_person", "session-001")
        values = [f["blendshapes"]["jawOpen"] for f in frames if f.get("blendshapes", {}).get("jawOpen") is not None]
        self.assertAlmostEqual(result["blendshape_means"]["jawOpen"]["session-001"], sum(values) / len(values), places=6)
