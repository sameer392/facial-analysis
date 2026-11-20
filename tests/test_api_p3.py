"""Phase P3 tests: V2 Pose and head-pose API routes must match the Python viewer and 3D code (synthetic data)."""

import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "dashboard"))
sys.path.insert(0, str(ROOT / "tests"))

import project_paths as pp  # noqa: E402
import demo_data  # noqa: E402
import facial_3d_analyzer as f3  # noqa: E402
import facial_analyzer as fa  # noqa: E402
import v2_viewer as vv  # noqa: E402
import test_pose_viewer as tv  # noqa: E402  (synthetic V2 writer)
from api import v1_data, v2_data  # noqa: E402
from api.catalog import ApiError  # noqa: E402
from test_api_p0 import call  # noqa: E402

PERSON = "demo_person"


def call_raw(app, method: str, path: str):
    """Like test_api_p0.call, but returns the raw body bytes and the content type."""
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": [], "server": ("127.0.0.1", 8000),
             "scheme": "http", "root_path": "", "http_version": "1.1"}
    result = {"body": [], "headers": {}}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = {k.decode(): v.decode() for k, v in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            result["body"].append(message.get("body", b""))

    asyncio.run(app(scope, receive, send))
    return result["status"], b"".join(result["body"]), result["headers"]


class P3TestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="api_p3_"))
        self._orig = (pp.RECORDINGS_RAW, pp.DATASETS_ROOT)
        pp.RECORDINGS_RAW = self.tmp / "recordings" / "raw"
        pp.DATASETS_ROOT = self.tmp / "datasets"
        v1_data.clear_cache()
        v2_data.clear_cache()
        # V1 analysis data and a V2 session in the same folder, as the real layout has them
        demo_data.generate_session(self.tmp, "session-001", demo_data.SESSIONS["session-001"])
        demo_data.generate_session(self.tmp, "session-002", demo_data.SESSIONS["session-002"])
        self.v2_dir = pp.DATASETS_ROOT / PERSON / "session-001"
        tv.write_pose(self.v2_dir)
        self.frames = tv.N
        self.write_video(pp.session_video(PERSON, "session-001"))

    def tearDown(self):
        pp.RECORDINGS_RAW, pp.DATASETS_ROOT = self._orig
        v1_data.clear_cache()
        v2_data.clear_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_video(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (tv.W, tv.H))
        self.assertTrue(writer.isOpened(), "test video writer could not open")
        for i in range(self.frames):
            frame = np.full((tv.H, tv.W, 3), 40 + 10 * i, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def stored_pose(self):
        return np.load(self.v2_dir / "body" / "pose_landmarks.npy")


class PoseParityTests(P3TestCase):
    def test_info_reports_the_validated_session(self):
        info = v2_data.info(PERSON, "session-001")
        self.assertTrue(info["valid"])
        self.assertEqual(info["frames"], self.frames)
        self.assertEqual((info["width"], info["height"]), (tv.W, tv.H))
        self.assertEqual(len(info["joint_names"]), vv.N_POSE)

    def test_pose_slice_matches_stored_values_and_viewer_states(self):
        pose = self.stored_pose()
        result = v2_data.pose(PERSON, "session-001", 2, 5, 0.5)
        self.assertEqual([f["index"] for f in result["frames"]], [2, 3, 4])
        for frame in result["frames"]:
            expected_states = vv.joint_states(pose[frame["index"]], 0.5)
            for joint in frame["joints"]:
                k = joint["index"]
                self.assertEqual(joint["state"], expected_states[k])
                stored = float(pose[frame["index"], k, 3])
                if np.isfinite(stored):
                    self.assertAlmostEqual(joint["visibility"], stored, places=6)
                else:
                    self.assertIsNone(joint["visibility"])  # NaN is returned as null

    def test_timeline_rows_match_the_csv(self):
        rows = v2_data.timeline(PERSON, "session-001")
        self.assertEqual(len(rows), self.frames)
        self.assertEqual([r["frame"] for r in rows], list(range(self.frames)))
        self.assertIn("pose_detected", rows[0])

    def test_quality_matches_the_viewer_summary(self):
        pose = self.stored_pose()
        timeline = vv.load_timeline(self.v2_dir, expected_frames=self.frames)
        result = v2_data.quality(PERSON, "session-001", 3, 0.5)
        expected = vv.quality_summary(pose, timeline, 0.5, 3)
        self.assertEqual(result["quality"]["pose_detected_frames"], expected["pose_detected_frames"])
        self.assertAlmostEqual(result["quality"]["pose_detection_rate"], expected["pose_detection_rate"])
        self.assertEqual(result["quality"]["valid_joints_selected_frame"], expected["valid_joints_selected_frame"])
        self.assertEqual(len(result["joints"]), vv.N_POSE)

    def test_overlay_is_a_decodable_jpeg_of_the_video_size(self):
        from api.main import app

        status, body, headers = call_raw(app, "GET", "/api/persons/demo_person/sessions/session-001/v2/frames/1/overlay.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "image/jpeg")
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (tv.H, tv.W))

    def test_overlay_without_video_is_a_clean_not_found(self):
        pp.session_video(PERSON, "session-001").unlink()
        with self.assertRaises(ApiError) as ctx:
            v2_data.overlay_jpeg(PERSON, "session-001", 0, 0.5, True, True)
        self.assertEqual((ctx.exception.status, ctx.exception.code), (404, "no_video"))


class HeadPoseParityTests(P3TestCase):
    def test_head_pose_matches_the_3d_analyzer(self):
        result = v2_data.head_pose(PERSON, "session-001")
        ds = fa.load_session(fa.make_session(PERSON, "session-001"))
        angles = f3.head_pose_angles(ds.matrices)
        depth = f3.depth_range_per_frame(ds.landmarks)
        self.assertEqual(len(result["yaw_deg"]), len(angles))
        for key in ("yaw_deg", "pitch_deg", "roll_deg"):
            got = np.array([np.nan if v is None else v for v in result[key]], dtype=float)
            np.testing.assert_allclose(got, angles[key].to_numpy(), equal_nan=True)
        got_depth = np.array([np.nan if v is None else v for v in result["depth_range"]], dtype=float)
        np.testing.assert_allclose(got_depth, depth, equal_nan=True)


class ErrorTests(P3TestCase):
    def test_bad_requests_are_refused(self):
        cases = [
            (lambda: v2_data.pose(PERSON, "session-001", 0, 2, 1.5), "invalid_request"),
            (lambda: v2_data.pose(PERSON, "session-001", 4, 4, 0.5), "invalid_request"),
            (lambda: v2_data.pose(PERSON, "session-001", 0, self.frames + 1, 0.5), "invalid_request"),
            (lambda: v2_data.quality(PERSON, "session-001", self.frames, 0.5), "not_found"),
            (lambda: v2_data.quality(PERSON, "session-001", 0, -0.1), "invalid_request"),
        ]
        for fn, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(ApiError) as ctx:
                    fn()
                self.assertEqual(ctx.exception.code, code)

    def test_session_without_pose_data_is_not_found(self):
        with self.assertRaises(ApiError) as ctx:
            v2_data.info(PERSON, "session-002")
        self.assertEqual((ctx.exception.status, ctx.exception.code), (404, "no_v2_data"))

    def test_invalid_pose_session_is_refused_with_reason(self):
        (self.v2_dir / "body" / "pose_landmarks.npy").write_bytes(b"not a numpy file")
        with self.assertRaises(ApiError) as ctx:
            v2_data.info(PERSON, "session-001")
        self.assertEqual((ctx.exception.status, ctx.exception.code), (422, "v2_invalid"))

    def test_p3_routes_respond_over_http(self):
        from api.main import app

        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-001/v2")
        self.assertEqual(status, 200)
        self.assertTrue(body["valid"])
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-002/v2")
        self.assertEqual(status, 404)
        status, body = call(app, "GET", "/api/persons/demo_person/sessions/session-001/v1/head-pose")
        self.assertEqual(status, 200)
        self.assertIn("depth_range", body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
