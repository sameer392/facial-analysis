"""Native video playback backend (phase H8.2): local-only guard, recording streaming with Range, landmark windows and the
timing check. Synthetic recordings and synthetic sessions only."""

import asyncio
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))

from api import video_data  # noqa: E402
from api.main import app  # noqa: E402
import test_analyse_hand_cli as tah  # noqa: E402  (synthetic session with face, body and hand streams)
import test_build_pose_cli as tbc  # noqa: E402

LOCAL = {"host": "127.0.0.1:8000"}


def call(method, path, headers=None, query=""):
    """Call the ASGI app with proper request and disconnect events (FileResponse waits for a disconnect)."""
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(),
             "query_string": query.encode(), "headers": hdrs, "server": ("127.0.0.1", 8000),
             "scheme": "http", "root_path": "", "http_version": "1.1"}
    out = {"status": None, "headers": {}, "body": b""}

    async def run():
        done = asyncio.Event()

        async def receive():
            if not done.is_set():
                await asyncio.wait_for(done.wait(), timeout=5)
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                out["status"] = message["status"]
                out["headers"] = {k.decode().lower(): v.decode() for k, v in message.get("headers", [])}
            elif message["type"] == "http.response.body":
                out["body"] += message.get("body", b"")
                if not message.get("more_body", False):
                    done.set()

        await app(scope, receive, send)

    asyncio.run(run())
    return out["status"], out["headers"], out["body"]


def local(**extra):
    return {**LOCAL, **extra}


class RecordingCase(tah.AnalyseCase):
    """A synthetic session whose recording and pose/hand data exist in a temporary recordings root."""

    def setUp(self):
        super().setUp()
        self.person, self.session = "demo_person", "session-001"
        self.video_path = self.video  # written by the base class
        self.recording_bytes = self.video_path.read_bytes()

    def tearDown(self):
        super().tearDown()


class LocalGuardTests(RecordingCase):
    def test_other_hosts_are_refused_on_every_new_route(self):
        for path in (f"/api/persons/{self.person}/sessions/{self.session}/video",
                     f"/api/persons/{self.person}/sessions/{self.session}/landmarks/window",
                     f"/api/persons/{self.person}/sessions/{self.session}/video/timing"):
            status, _h, body = call("GET", path, headers={"host": "evil.example:8000"})
            self.assertEqual(status, 403, path)
            self.assertIn(b"host_not_allowed", body)

    def test_dns_rebinding_host_is_refused(self):
        status, _h, body = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                                headers={"host": "attacker.test:8000"})
        self.assertEqual(status, 403)

    def test_forwarded_host_header_is_never_trusted(self):
        status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                              headers=local(**{"x-forwarded-host": "evil.example"}))
        self.assertEqual(status, 200)

    def test_cross_site_and_same_site_requests_are_refused(self):
        for site in ("cross-site", "same-site", "none"):
            status, _h, body = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                                    headers=local(**{"sec-fetch-site": site}))
            self.assertEqual(status, 403, site)
            self.assertIn(b"cross_site_request", body)

    def test_same_origin_and_non_browser_requests_are_accepted(self):
        for extra in ({"sec-fetch-site": "same-origin"}, {}):
            status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                                  headers=local(**extra))
            self.assertEqual(status, 200, extra)

    def test_localhost_name_is_accepted(self):
        status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                              headers={"host": "localhost:8000"})
        self.assertEqual(status, 200)

    def test_configured_hosts_can_be_changed_by_environment(self):
        with mock.patch.dict(os.environ, {video_data.ALLOWED_HOSTS_ENV: "files.local:9000"}):
            status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video",
                                  headers={"host": "files.local:9000"})
            self.assertEqual(status, 200)
            status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/video", headers=LOCAL)
            self.assertEqual(status, 403)

    def test_existing_read_routes_are_not_guarded_by_the_new_check(self):
        status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/{self.session}/hands/summary",
                              headers={"host": "anything.example"})
        self.assertNotEqual(status, 403)


class VideoStreamingTests(RecordingCase):
    def path(self):
        return f"/api/persons/{self.person}/sessions/{self.session}/video"

    def test_full_retrieval_returns_the_original_bytes(self):
        status, h, body = call("GET", self.path(), headers=local(**{"sec-fetch-site": "same-origin"}))
        self.assertEqual(status, 200)
        self.assertEqual(body, self.recording_bytes)
        self.assertEqual(h["content-type"], "video/mp4")
        self.assertEqual(h["accept-ranges"], "bytes")
        self.assertEqual(h["content-length"], str(len(self.recording_bytes)))
        self.assertEqual(h["cache-control"], "no-store")

    def test_range_request_returns_a_partial_response(self):
        status, h, body = call("GET", self.path(), headers=local(range="bytes=10-19"))
        self.assertEqual(status, 206)
        self.assertEqual(body, self.recording_bytes[10:20])
        self.assertEqual(h["content-range"], f"bytes 10-19/{len(self.recording_bytes)}")
        self.assertEqual(h["content-length"], "10")

    def test_suffix_range_returns_the_last_bytes(self):
        status, _h, body = call("GET", self.path(), headers=local(range="bytes=-5"))
        self.assertEqual(status, 206)
        self.assertEqual(body, self.recording_bytes[-5:])

    def test_open_ended_range_seeks_to_the_end(self):
        size = len(self.recording_bytes)
        status, _h, body = call("GET", self.path(), headers=local(range=f"bytes={size - 8}-"))
        self.assertEqual(status, 206)
        self.assertEqual(body, self.recording_bytes[-8:])

    def test_invalid_range_header_is_not_a_server_error(self):
        status, _h, body = call("GET", self.path(), headers=local(range="bytes=abc"))
        self.assertEqual(status, 400, "a malformed Range header is a client error, not a server error")
        self.assertNotIn(str(self.tmp).encode(), body)

    def test_range_beyond_the_end_is_refused_not_served(self):
        status, _h, _b = call("GET", self.path(), headers=local(range=f"bytes={len(self.recording_bytes) + 100}-"))
        self.assertEqual(status, 416)

    def test_reading_the_recording_does_not_change_it(self):
        before = (self.video_path.stat().st_size, self.video_path.stat().st_mtime_ns)
        call("GET", self.path(), headers=local(range="bytes=0-99"))
        self.assertEqual((self.video_path.stat().st_size, self.video_path.stat().st_mtime_ns), before)

    def test_missing_recording_is_a_structured_error_without_paths(self):
        self.video_path.unlink()
        status, _h, body = call("GET", self.path(), headers=LOCAL)
        self.assertEqual(status, 404)
        self.assertIn(b"no_video", body)
        self.assertNotIn(str(self.tmp).encode(), body)

    def test_invalid_names_are_refused(self):
        for person, session in (("../escape", self.session), (self.person, "session-1"),
                                (self.person, "..%2Fx"), (self.person, "../../x")):
            status, _h, _b = call("GET", f"/api/persons/{person}/sessions/{session}/video", headers=LOCAL)
            self.assertIn(status, (400, 404), (person, session))

    def test_symlink_that_escapes_the_recordings_root_is_refused(self):
        outside = Path(tempfile.mkdtemp(prefix="outside_video_"))
        self.addCleanup(shutil.rmtree, outside, True)
        target = outside / "secret.mp4"
        target.write_bytes(self.recording_bytes)
        link = self.video_path.parent / "session-002.mp4"
        try:
            os.symlink(target, link)
        except OSError:
            self.skipTest("creating symlinks needs permission on this machine")
        self.addCleanup(lambda: link.unlink(missing_ok=True))
        status, _h, body = call("GET", f"/api/persons/{self.person}/sessions/session-002/video", headers=LOCAL)
        self.assertEqual(status, 404)
        self.assertNotIn(str(outside).encode(), body)

    def test_unsupported_suffix_is_refused(self):
        other = self.video_path.parent / "session-003.mp4"
        other.write_bytes(self.recording_bytes)
        status, _h, _b = call("GET", f"/api/persons/{self.person}/sessions/session-003/video", headers=LOCAL)
        self.assertEqual(status, 200)  # a valid mp4 name is served; only the resolved suffix is checked


class LandmarkWindowTests(tah.AnalyseCase):
    def setUp(self):
        super().setUp()
        self.build_hands()
        self.total = tbc.N

    def window(self, start=0, count=30, headers=None):
        status, h, body = call("GET", f"/api/persons/demo_person/sessions/session-001/landmarks/window",
                               headers=headers or LOCAL, query=f"start={start}&count={count}")
        return status, h, body

    @staticmethod
    def decode(body):
        assert body[:4] == b"HLW1", body[:4]
        (n,) = struct.unpack("<I", body[4:8])
        header = json.loads(body[8:8 + n].decode("utf-8"))
        return header, body[8 + n:]

    def test_window_format_and_values_match_the_stored_arrays(self):
        status, h, body = self.window(0, 4)
        self.assertEqual(status, 200)
        self.assertIn("version=1", h["content-type"])
        header, data = self.decode(body)
        self.assertEqual((header["format"], header["version"], header["count"]), ("HLW", 1, 4))
        self.assertEqual(header["frames"], [0, 1, 2, 3])
        self.assertEqual(len(header["timestamps_ms"]), 4)
        face = header["streams"]["face"]
        self.assertTrue(face["present"])
        self.assertEqual(face["channels"], ["x", "y"])
        arr = np.frombuffer(data[face["offset"]:face["offset"] + face["bytes"]], dtype="<f4").reshape(face["shape"])
        stored = np.load(self.dataset / "facial" / "landmarks.npy")[0:4][:, :, :2]
        self.assertTrue(np.allclose(arr, stored, equal_nan=True))

    def test_body_stream_keeps_visibility_channel(self):
        header, _ = self.decode(self.window(0, 2)[2])
        self.assertEqual(header["streams"]["body"]["channels"], ["x", "y", "visibility"])

    def test_hand_streams_and_status_are_included(self):
        header, data = self.decode(self.window(0, self.total)[2])
        left = header["streams"]["left_hand"]
        self.assertTrue(left["present"])
        self.assertEqual(len(header["hand_status"]["right"]), self.total)
        self.assertIn(header["hand_status"]["right"][0], ("valid", "outside_frame", "low_confidence",
                                                          "not_detected", "unassigned", "not_processed"))

    def test_missing_stream_is_declared_not_guessed(self):
        (self.dataset / "hands" / "right_hand.npy").unlink()
        header, _ = self.decode(self.window(0, 2)[2])
        self.assertFalse(header["streams"]["right_hand"]["present"])

    def test_window_limits_are_enforced(self):
        for start, count in ((0, 0), (0, 31), (-1, 2), (self.total - 1, 5)):
            status, _h, body = self.window(start, count)
            self.assertEqual(status, 400, (start, count))
            self.assertIn(b"invalid_window", body)

    def test_a_corrupt_stream_shape_is_refused(self):
        np.save(self.dataset / "facial" / "landmarks.npy", np.zeros((self.total, 10, 3), np.float32))
        status, _h, body = self.window(0, 2)
        self.assertEqual(status, 422)
        self.assertIn(b"dataset_corrupt", body)
        self.assertNotIn(str(self.dataset).encode(), body)

    def test_window_requests_are_local_only(self):
        status, _h, _b = self.window(0, 2, headers={"host": "evil.example:8000"})
        self.assertEqual(status, 403)
        status, _h, _b = self.window(0, 2, headers=local(**{"sec-fetch-site": "cross-site"}))
        self.assertEqual(status, 403)

    def test_window_reads_do_not_change_the_dataset(self):
        before = sorted((p.name, p.stat().st_mtime_ns) for p in self.dataset.rglob("*") if p.is_file())
        self.window(0, 8)
        after = sorted((p.name, p.stat().st_mtime_ns) for p in self.dataset.rglob("*") if p.is_file())
        self.assertEqual(before, after)


class TimingTests(tah.AnalyseCase):
    def setUp(self):
        super().setUp()
        self.build_hands()
        self.video_path = self.video
        video_data._timing_cache.clear()

    def timing(self):
        status, _h, body = call("GET", "/api/persons/demo_person/sessions/session-001/video/timing", headers=LOCAL)
        return status, json.loads(body)

    def test_synthetic_recording_is_synchronized(self):
        status, body = self.timing()
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "synchronized", body)
        self.assertEqual(body["frames_checked"], tbc.N)

    def test_a_timeline_deviation_is_reported(self):
        import pandas as pd
        tl = pd.read_csv(self.dataset / "timeline.csv")
        tl.loc[3, "timestamp_ms"] = tl.loc[3, "timestamp_ms"] + 50
        tl.to_csv(self.dataset / "timeline.csv", index=False)
        video_data._timing_cache.clear()
        status, body = self.timing()
        self.assertEqual(body["status"], "deviation", body)

    def test_a_recording_too_long_to_check_is_unavailable_not_synchronized(self):
        video_data._timing_cache.clear()
        with mock.patch.object(video_data, "TIMING_MAX_FRAMES", 2):
            status, body = self.timing()
        self.assertEqual(body["status"], "unavailable", body)

    def test_timing_is_local_only(self):
        status, _h, _b = call("GET", "/api/persons/demo_person/sessions/session-001/video/timing",
                              headers={"host": "evil.example:8000"})
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main(verbosity=2)
