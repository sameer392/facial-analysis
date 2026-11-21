"""Audio acquisition (phase H9.1): metadata, timing, retention, refusals and cleanup.

Synthetic audio only. Pure logic is tested without FFmpeg. Tests that decode real containers need
ffmpeg and ffprobe; they are skipped with a stated reason when those are not installed.
Every test uses temporary folders; nothing under recordings/ or datasets/ is read or written.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import audio_acquisition as aa  # noqa: E402
from extraction import build_audio_session as cli  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
REQUIRE_FFMPEG = os.environ.get("AUDIO_TESTS_REQUIRE_FFMPEG") == "1"


def needs_ffmpeg(cls):
    """Run the class when FFmpeg is present. Otherwise skip it with a visible reason, or, when
    AUDIO_TESTS_REQUIRE_FFMPEG=1 (as CI should set), fail: a missing FFmpeg is never a silent pass."""
    if HAVE_FFMPEG:
        return cls
    if REQUIRE_FFMPEG:
        def fail(_cls):
            raise AssertionError("ffmpeg and ffprobe are required because AUDIO_TESTS_REQUIRE_FFMPEG=1")
        cls.setUpClass = classmethod(fail)
        return cls
    return unittest.skip("ffmpeg/ffprobe not on PATH: FFmpeg tests were NOT run")(cls)


class FFmpegPresenceTests(unittest.TestCase):
    def test_ffmpeg_is_present_when_required(self):
        if REQUIRE_FFMPEG:
            self.assertTrue(HAVE_FFMPEG)
        else:
            self.skipTest("set AUDIO_TESTS_REQUIRE_FFMPEG=1 to make a missing FFmpeg a failure")


def stream(index=1, rate=44100, channels=2, start_us=0, duration=1.0, codec="pcm_s16le"):
    return {"index": index, "codec_type": "audio", "codec_name": codec, "sample_rate": str(rate),
            "channels": channels, "start_time": f"{start_us / 1e6:.6f}", "duration": f"{duration:.6f}"}


class PureLogicTests(unittest.TestCase):
    """No FFmpeg needed."""

    def test_only_audio_streams_are_parsed(self):
        data = {"streams": [
            {"index": 0, "codec_type": "video", "start_time": "0.0"},
            stream(index=1, rate=48000, channels=2, start_us=21333),
        ]}
        parsed = aa.parse_audio_streams(data)
        self.assertEqual(len(parsed), 1)
        self.assertEqual((parsed[0].index, parsed[0].sample_rate, parsed[0].channels), (1, 48000, 2))
        self.assertEqual(parsed[0].start_us, 21333)

    def test_unknown_start_time_is_recorded_as_none(self):
        s = stream()
        s["start_time"] = "N/A"
        self.assertIsNone(aa.parse_audio_streams({"streams": [s]})[0].start_us)

    def test_missing_sample_rate_is_refused(self):
        s = stream()
        del s["sample_rate"]
        with self.assertRaises(aa.AudioError):
            aa.parse_audio_streams({"streams": [s]})

    def test_zero_channels_is_refused(self):
        with self.assertRaises(aa.AudioError):
            aa.parse_audio_streams({"streams": [stream(channels=0)]})

    def test_cover_art_is_not_a_video_stream(self):
        data = {"streams": [
            {"index": 0, "codec_type": "video", "disposition": {"attached_pic": 1}, "start_time": "0.0"},
        ]}
        self.assertIsNone(aa.first_video_stream(data))

    def test_one_audio_stream_is_selected_automatically(self):
        streams = aa.parse_audio_streams({"streams": [stream(index=3)]})
        self.assertEqual(aa.select_stream(streams, None).index, 3)

    def test_several_audio_streams_need_an_explicit_choice(self):
        streams = aa.parse_audio_streams({"streams": [stream(index=1), stream(index=2, rate=48000)]})
        with self.assertRaises(aa.AudioError) as ctx:
            aa.select_stream(streams, None)
        self.assertIn("choose one", str(ctx.exception))
        self.assertEqual(aa.select_stream(streams, 2).sample_rate, 48000)

    def test_missing_stream_index_is_refused(self):
        streams = aa.parse_audio_streams({"streams": [stream(index=1)]})
        with self.assertRaises(aa.AudioError):
            aa.select_stream(streams, 9)

    def test_no_audio_stream_is_refused(self):
        with self.assertRaises(aa.AudioError) as ctx:
            aa.select_stream([], None)
        self.assertIn("no audio stream", str(ctx.exception))

    def test_timeline_places_samples_in_integer_microseconds(self):
        t = aa.session_timeline(audio_start_us=21333, sample_rate=44100, sample_count=44100,
                                offset_us=250_000, origin_us=40_000)
        self.assertEqual(t["first_sample_session_us"], 21333 + 250_000 - 40_000)
        # The last sample is at (N - 1) / rate seconds after the first, rounded to whole microseconds.
        self.assertEqual(t["last_sample_session_us"], t["first_sample_session_us"] + 999_977)
        self.assertIsInstance(t["last_sample_session_us"], int)

    def test_timeline_keeps_sub_millisecond_precision(self):
        t = aa.session_timeline(audio_start_us=0, sample_rate=16000, sample_count=3, offset_us=0, origin_us=0)
        self.assertEqual(t["last_sample_session_us"], 125)  # 2 / 16000 s = 125 us

    def test_timeline_refuses_empty_audio(self):
        with self.assertRaises(aa.AudioError):
            aa.session_timeline(audio_start_us=0, sample_rate=44100, sample_count=0, offset_us=0, origin_us=0)

    def test_names_are_validated_before_any_file_is_touched(self):
        with self.assertRaises(aa.AudioError):
            aa.acquire_session_audio("../escape", "session-001")
        with self.assertRaises(aa.AudioError):
            aa.acquire_session_audio("person", "session-1")

    def test_offset_is_refused_for_embedded_audio(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", offset_us=5, recordings_root=Path(root))
            self.assertIn("offset", str(ctx.exception))

    def test_retention_is_refused_for_embedded_audio(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", retain=True, recordings_root=Path(root))
            self.assertIn("never copied", str(ctx.exception))

    def test_missing_embedded_recording_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", recordings_root=Path(root))
            self.assertIn("not found", str(ctx.exception))

    def test_non_audio_extension_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "notes.txt"
            bad.write_text("not audio", encoding="utf-8")
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", external=bad)
            self.assertIn(".wav or .mp3", str(ctx.exception))

    def test_errors_carry_no_full_path(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "gone.wav"
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", external=missing)
            self.assertNotIn(folder, str(ctx.exception))


class FileRuleTests(unittest.TestCase):
    def test_symlinked_external_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            real = Path(folder) / "real.wav"
            real.write_bytes(b"RIFF")
            link = Path(folder) / "link.wav"
            try:
                os.symlink(real, link)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available on this system")
            with self.assertRaises(aa.AudioError) as ctx:
                aa.acquire_session_audio("person", "session-001", external=link)
            self.assertIn("symbolic link", str(ctx.exception))

    def test_source_outside_the_root_is_refused(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as elsewhere:
            with self.assertRaises(aa.AudioError):
                aa._require_inside(Path(elsewhere) / "x.mp4", Path(root))

    def test_existing_manifest_is_refused_without_force(self):
        with tempfile.TemporaryDirectory() as datasets, tempfile.TemporaryDirectory() as recs:
            manifest = Path(datasets) / "person" / "session-001" / "audio" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            (Path(recs) / "person").mkdir()
            (Path(recs) / "person" / "session-001.mp4").write_bytes(b"not a real video")
            from file_safety import OutputExistsError
            with self.assertRaises(OutputExistsError):
                aa.acquire_session_audio("person", "session-001", recordings_root=Path(recs),
                                         datasets_root=Path(datasets))
            self.assertEqual(manifest.read_text(encoding="utf-8"), "{}")


class LockTests(unittest.TestCase):
    """The per-session lock. No FFmpeg needed."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audio_lock_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.audio = self.tmp / "audio"
        self.lock = self.audio / aa.LOCK_NAME

    def dead_pid(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def test_a_second_acquisition_is_refused_while_the_lock_is_held(self):
        with aa.session_lock(self.audio):
            with self.assertRaises(aa.AudioError) as ctx:
                with aa.session_lock(self.audio):
                    pass
            self.assertIn("another audio acquisition is running", str(ctx.exception))
            self.assertTrue(self.lock.exists(), "the first holder's lock must survive the refusal")
        self.assertFalse(self.lock.exists())

    def test_the_lock_is_released_after_success(self):
        with aa.session_lock(self.audio):
            pass
        self.assertFalse(self.lock.exists())
        with aa.session_lock(self.audio):  # can be taken again
            pass

    def test_the_lock_is_released_after_a_failure(self):
        with self.assertRaises(RuntimeError):
            with aa.session_lock(self.audio):
                raise RuntimeError("processing failed")
        self.assertFalse(self.lock.exists())
        with aa.session_lock(self.audio):
            pass

    def test_concurrent_threads_admit_exactly_one_holder(self):
        import threading
        barrier = threading.Barrier(8)
        results = []
        results_lock = threading.Lock()

        def attempt():
            barrier.wait()
            try:
                with aa.session_lock(self.audio):
                    time.sleep(0.3)  # long enough that every other thread sees the lock
                outcome = "held"
            except aa.AudioError:
                outcome = "refused"
            with results_lock:
                results.append(outcome)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count("held"), 1, results)
        self.assertEqual(results.count("refused"), 7, results)
        self.assertFalse(self.lock.exists())

    def test_a_stale_lock_is_reported_and_never_removed_automatically(self):
        self.audio.mkdir(parents=True)
        pid = self.dead_pid()
        self.lock.write_text(json.dumps({"pid": pid, "host": socket.gethostname(), "token": "old",
                                         "created_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
        with self.assertRaises(aa.AudioError) as ctx:
            with aa.session_lock(self.audio):
                pass
        message = str(ctx.exception)
        self.assertIn(str(pid), message)
        self.assertIn("2026-01-01T00:00:00+00:00", message)
        self.assertIn("--clear-stale-lock", message)
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["token"], "old",
                         "a stale lock must stay in place until an operator clears it")

    def test_explicit_clear_removes_a_verified_stale_lock(self):
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": self.dead_pid(), "host": socket.gethostname(),
                                         "token": "old"}), encoding="utf-8")
        owner = aa.clear_stale_lock(self.audio)
        self.assertEqual(owner["token"], "old")
        self.assertFalse(self.lock.exists())
        with aa.session_lock(self.audio):  # the session is usable again
            pass

    def test_explicit_clear_refuses_a_live_lock(self):
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "token": "live"}),
                             encoding="utf-8")
        with self.assertRaises(aa.AudioError):
            aa.clear_stale_lock(self.audio)
        self.assertTrue(self.lock.exists())

    def test_explicit_clear_refuses_a_foreign_or_unreadable_lock(self):
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": self.dead_pid(), "host": "other-machine", "token": "x"}),
                             encoding="utf-8")
        with self.assertRaises(aa.AudioError):
            aa.clear_stale_lock(self.audio)
        self.lock.write_bytes(b"\x00 garbage")
        with self.assertRaises(aa.AudioError):
            aa.clear_stale_lock(self.audio)
        self.assertTrue(self.lock.exists())

    def test_simultaneous_attempts_on_a_stale_lock_never_admit_a_holder(self):
        """The race from review: several processes see the same stale lock at once. None may take it,
        and the lock must still be the original one afterwards."""
        import threading
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": self.dead_pid(), "host": socket.gethostname(),
                                         "token": "original"}), encoding="utf-8")
        barrier = threading.Barrier(8)
        outcomes = []
        outcomes_lock = threading.Lock()
        holders = []
        max_holders = [0]

        def attempt():
            barrier.wait()
            try:
                with aa.session_lock(self.audio):
                    with outcomes_lock:
                        holders.append(1)
                        max_holders[0] = max(max_holders[0], len(holders))
                    time.sleep(0.05)
                    with outcomes_lock:
                        holders.pop()
                result = "held"
            except aa.AudioError:
                result = "refused"
            with outcomes_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(outcomes.count("held"), 0, outcomes)
        self.assertEqual(max_holders[0], 0)
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["token"], "original")

    def test_simultaneous_clear_attempts_cannot_remove_a_fresh_lock_taken_in_between(self):
        """After a stale lock is cleared, a new holder may appear. Clearing again must refuse it."""
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": self.dead_pid(), "host": socket.gethostname(),
                                         "token": "stale"}), encoding="utf-8")
        aa.clear_stale_lock(self.audio)
        with aa.session_lock(self.audio):
            with self.assertRaises(aa.AudioError):
                aa.clear_stale_lock(self.audio)
            self.assertTrue(self.lock.exists())
        self.assertFalse(self.lock.exists())

    def test_a_lock_held_by_a_live_process_is_not_stolen(self):
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(),
                                         "token": "live"}), encoding="utf-8")
        with self.assertRaises(aa.AudioError):
            with aa.session_lock(self.audio):
                pass
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["token"], "live")

    def test_a_lock_from_another_machine_is_never_treated_as_stale(self):
        self.audio.mkdir(parents=True)
        self.lock.write_text(json.dumps({"pid": self.dead_pid(), "host": "some-other-machine",
                                         "token": "remote"}), encoding="utf-8")
        with self.assertRaises(aa.AudioError):
            with aa.session_lock(self.audio):
                pass
        self.assertTrue(self.lock.exists())

    def test_an_unreadable_lock_is_not_stolen(self):
        self.audio.mkdir(parents=True)
        self.lock.write_bytes(b"\x00 half written")
        with self.assertRaises(aa.AudioError):
            with aa.session_lock(self.audio):
                pass
        self.assertTrue(self.lock.exists())

    def test_release_never_removes_a_lock_owned_by_someone_else(self):
        with aa.session_lock(self.audio):
            self.lock.write_text(json.dumps({"pid": 1, "host": "x", "token": "someone-else"}), encoding="utf-8")
        self.assertTrue(self.lock.exists())
        self.assertEqual(json.loads(self.lock.read_text(encoding="utf-8"))["token"], "someone-else")

    def test_acquisition_checks_the_lock_before_anything_else(self):
        datasets = self.tmp / "datasets"
        held = datasets / "person" / "session-001" / "audio"
        with aa.session_lock(held):
            with mock.patch.object(aa, "probe", side_effect=AssertionError("probe must not run")):
                with self.assertRaises(aa.AudioError) as ctx:
                    aa.acquire_session_audio("person", "session-001", recordings_root=self.tmp / "none",
                                             datasets_root=datasets)
        self.assertIn("another audio acquisition", str(ctx.exception))


class PcmLimitPureTests(unittest.TestCase):
    def test_expected_size_is_duration_times_rate_times_channels_times_two(self):
        self.assertEqual(aa.expected_pcm_bytes(1.0, 44100, 2), 44100 * 2 * 2)
        self.assertEqual(aa.expected_pcm_bytes(0.5, 16000, 1), 16000)

    def test_four_hours_of_48k_stereo_is_within_the_limit(self):
        size = aa.expected_pcm_bytes(4 * 3600, 48000, 2)
        self.assertEqual(size, 2_764_800_000)
        self.assertLessEqual(size, aa.MAX_DECODED_PCM_BYTES)

    def test_four_hours_of_96k_eight_channel_is_above_the_limit(self):
        size = aa.expected_pcm_bytes(4 * 3600, 96000, 8)
        self.assertGreater(size, aa.MAX_DECODED_PCM_BYTES)

    def test_planned_size_above_the_limit_is_refused_before_decoding(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(aa, "probe", return_value={"format": {"format_name": "wav", "duration": "3600.0"},
                                                              "streams": [
                    {"index": 0, "codec_type": "audio", "sample_rate": "96000", "channels": 8,
                     "start_time": "0.0", "duration": "3600.0"}]}), \
                    mock.patch.object(aa, "_decode", side_effect=AssertionError("must not decode")):
                ext = Path(root) / "huge.wav"
                ext.write_bytes(b"RIFF")
                with self.assertRaises(aa.AudioError) as ctx:
                    aa.acquire_session_audio("person", "session-001", external=ext,
                                             recordings_root=Path(root), datasets_root=Path(root) / "d")
        self.assertIn("nothing was decoded", str(ctx.exception))


@needs_ffmpeg
class FFmpegTests(unittest.TestCase):
    """Real decoding with synthetic inputs generated in temporary folders."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audio_acq_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.recs = self.tmp / "recordings"
        self.datasets = self.tmp / "datasets"
        self.work = self.tmp / "scratch"
        self.work.mkdir()

    def run_acquire(self, **kwargs):
        return aa.acquire_session_audio("person", kwargs.pop("session", "session-001"),
                                        recordings_root=self.recs, datasets_root=self.datasets, **kwargs)

    def write_wav(self, name, rate=44100, channels=2, seconds=1.0):
        path = self.work / name
        t = np.arange(int(rate * seconds)) / rate
        tone = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
        frames = np.repeat(tone[:, None], channels, axis=1).reshape(-1)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(frames.tobytes())
        return path

    def ffmpeg(self, *args):
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
                       check=True, capture_output=True)

    def test_external_wav_is_decoded_at_its_own_rate_and_channels(self):
        wav = self.write_wav("voice.wav", rate=44100, channels=2, seconds=1.0)
        manifest = self.run_acquire(external=wav)
        self.assertEqual(manifest["source"]["type"], "external")
        self.assertEqual(manifest["decoded"]["sample_rate"], 44100)
        self.assertEqual(manifest["decoded"]["channels"], 2)
        self.assertEqual(manifest["decoded"]["sample_count"], 44100)
        self.assertEqual(manifest["duration_check"]["status"], "pass")
        self.assertEqual(manifest["retention"]["policy"], "external_not_retained")
        self.assertEqual(manifest["retention"]["reanalysis"], "requires the file to be supplied again")
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "source").exists())

    def test_decoding_is_deterministic(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.5)
        first = self.run_acquire(external=wav)["decoded"]["pcm_sha256"]
        shutil.rmtree(self.datasets)
        second = self.run_acquire(external=wav)["decoded"]["pcm_sha256"]
        self.assertEqual(first, second)

    def test_offset_places_the_first_sample_on_the_session_clock(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.5)
        manifest = self.run_acquire(external=wav, offset_us=250_000)
        self.assertEqual(manifest["timeline"]["external_offset_us"], 250_000)
        self.assertEqual(manifest["timeline"]["first_sample_session_us"], 250_000)

    def test_retained_copy_is_verified_and_matches_the_source(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.5)
        manifest = self.run_acquire(external=wav, retain=True)
        stored = self.datasets / "person" / "session-001" / "audio" / "source" / f"{manifest['source']['source_id']}.wav"
        self.assertTrue(stored.is_file())
        self.assertEqual(_sha(stored), _sha(wav))
        self.assertEqual(manifest["retention"]["policy"], "external_retained")
        self.assertEqual(manifest["retention"]["reanalysis"], "from the retained copy")
        self.assertEqual(manifest["retention"]["stored_file"],
                         f"audio/source/{manifest['source']['source_id']}.wav")

    def test_four_channel_wav_is_decoded_and_measured(self):
        """Regression: FFmpeg writes WAVE_FORMAT_EXTENSIBLE headers for >2 channels, which the standard
        wave module cannot read. H9.1 now measures raw PCM, so this must succeed."""
        wav_path = self.work / "quad.wav"
        pcm = np.zeros((16000, 4), dtype="<i2")
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(4)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        manifest = self.run_acquire(external=wav_path)
        self.assertEqual(manifest["decoded"]["channels"], 4)
        self.assertEqual(manifest["decoded"]["sample_count"], 16000)
        self.assertEqual(manifest["duration_check"]["status"], "pass")

    def test_mp3_is_accepted_and_its_contents_are_checked(self):
        wav = self.write_wav("voice.wav", rate=44100, channels=1, seconds=0.5)
        mp3 = self.work / "voice.mp3"
        self.ffmpeg("-i", str(wav), "-codec:a", "libmp3lame", str(mp3))
        manifest = self.run_acquire(external=mp3)
        self.assertEqual(manifest["source"]["container_format"], "mp3")
        self.assertEqual(manifest["decoded"]["sample_format"], "pcm_s16le")
        self.assertEqual(manifest["duration_check"]["status"], "pass")

    def test_a_wav_renamed_to_mp3_is_refused(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.2)
        fake = self.work / "fake.mp3"
        shutil.copyfile(wav, fake)
        with self.assertRaises(aa.AudioError) as ctx:
            self.run_acquire(external=fake)
        self.assertIn("do not match", str(ctx.exception))
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "manifest.json").exists())

    def test_a_corrupt_file_is_refused_and_nothing_is_written(self):
        bad = self.work / "broken.wav"
        bad.write_bytes(b"this is not audio at all")
        with self.assertRaises(aa.AudioError):
            self.run_acquire(external=bad)
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "manifest.json").exists())

    def make_video(self, name, audio_tracks=(44100,), with_audio=True):
        path = self.recs / "person" / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        args = ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1"]
        if with_audio:
            for rate in audio_tracks:
                args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={rate}:duration=1"]
        args += ["-map", "0:v"]
        if with_audio:
            for i in range(len(audio_tracks)):
                args += ["-map", f"{i + 1}:a"]
        args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest"]
        if with_audio:
            args += ["-c:a", "aac"]
        self.ffmpeg(*args, str(path))
        return path

    def test_embedded_audio_is_taken_from_the_video(self):
        self.make_video("session-001")
        manifest = self.run_acquire()
        self.assertEqual(manifest["source"]["type"], "video_embedded")
        self.assertEqual(manifest["retention"]["policy"], "video_not_copied")
        self.assertEqual(manifest["retention"]["reanalysis"], "from the original recording")
        self.assertIsNotNone(manifest["video"]["stream_start_us"])
        self.assertEqual(manifest["timeline"]["origin"], "video_first_presentation")
        self.assertEqual(manifest["duration_check"]["status"], "pass")

    def test_the_source_video_is_not_modified(self):
        video = self.make_video("session-001")
        before = _sha(video)
        self.run_acquire()
        self.assertEqual(_sha(video), before)

    def test_video_with_two_audio_tracks_needs_a_stream_choice(self):
        self.make_video("session-001", audio_tracks=(44100, 22050))
        with self.assertRaises(aa.AudioError) as ctx:
            self.run_acquire()
        self.assertIn("choose one", str(ctx.exception))
        # Stream 0 is the video; audio tracks are 1 (44100 Hz) and 2 (22050 Hz).
        manifest = self.run_acquire(stream=2)
        self.assertEqual(len(manifest["streams"]["all"]), 2)
        self.assertEqual(manifest["streams"]["selected"]["index"], 2)
        self.assertEqual(manifest["decoded"]["sample_rate"], 22050)

    def test_video_without_audio_is_refused_and_writes_nothing(self):
        self.make_video("session-001", with_audio=False)
        with self.assertRaises(aa.AudioError) as ctx:
            self.run_acquire()
        self.assertIn("no audio stream", str(ctx.exception))
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "manifest.json").exists())

    def test_temporary_files_are_removed_after_a_decode_failure(self):
        self.make_video("session-001")
        temp_root = self.tmp / "tmp"
        temp_root.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(temp_root)), \
                mock.patch.object(aa, "_measure_wav", side_effect=aa.AudioError("forced failure")):
            with self.assertRaises(aa.AudioError):
                self.run_acquire()
        self.assertEqual(list(temp_root.iterdir()), [], "temporary decoded audio was left behind")

    def test_temporary_files_are_removed_after_success(self):
        self.make_video("session-001")
        temp_root = self.tmp / "tmp"
        temp_root.mkdir()
        with mock.patch.object(tempfile, "tempdir", str(temp_root)):
            self.run_acquire()
        self.assertEqual(list(temp_root.iterdir()), [])

    def test_manifest_is_valid_json_with_the_expected_sections(self):
        self.make_video("session-001")
        self.run_acquire()
        path = self.datasets / "person" / "session-001" / "audio" / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("schema", "source", "streams", "video", "decoded", "duration_check", "timeline",
                    "retention", "tool"):
            self.assertIn(key, data)
        self.assertNotIn(str(self.tmp), path.read_text(encoding="utf-8").replace("\\", "/"),
                         "the manifest must not contain a machine path")

    def test_external_filename_is_absent_from_manifest_and_retained_copy(self):
        private = "private-person-name-2026"
        wav = self.write_wav(f"{private}.wav", rate=16000, channels=1, seconds=0.3)
        manifest = self.run_acquire(external=wav, retain=True)
        text = json.dumps(manifest)
        self.assertNotIn(private, text)
        self.assertEqual(len(manifest["source"]["source_id"]), 32)
        self.assertEqual(manifest["source"]["extension"], ".wav")
        stored = self.datasets / "person" / "session-001" / "audio" / "source"
        self.assertEqual([p.name for p in stored.iterdir()], [f"{manifest['source']['source_id']}.wav"])
        manifest_text = (self.datasets / "person" / "session-001" / "audio" / "manifest.json").read_text(
            encoding="utf-8")
        self.assertNotIn(private, manifest_text)
        self.assertNotIn(str(self.work), manifest_text)

    def test_source_ids_differ_between_runs(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.2)
        first = self.run_acquire(external=wav)["source"]["source_id"]
        shutil.rmtree(self.datasets)
        second = self.run_acquire(external=wav)["source"]["source_id"]
        self.assertNotEqual(first, second)

    def test_decoding_stops_when_the_output_exceeds_the_limit(self):
        wav = self.write_wav("voice.wav", rate=44100, channels=2, seconds=1.0)
        stream = aa.parse_audio_streams(aa.probe(wav))[0]
        work = self.tmp / "bounded"
        work.mkdir()
        with self.assertRaises(aa.AudioError) as ctx:
            aa._decode(wav, stream, work, max_bytes=1000)
        self.assertIn("size limit", str(ctx.exception))

    def test_oversized_decode_is_refused_and_leaves_no_temporary_files_or_lock(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.5)
        temp_root = self.tmp / "tmp"
        temp_root.mkdir()
        # Make the pre-check pass so the bounded decode itself is what refuses.
        with mock.patch.object(aa, "MAX_DECODED_PCM_BYTES", 1000), \
                mock.patch.object(aa, "WAV_HEADER_ALLOWANCE", 0), \
                mock.patch.object(aa, "expected_pcm_bytes", return_value=0), \
                mock.patch.object(tempfile, "tempdir", str(temp_root)):
            with self.assertRaises(aa.AudioError) as ctx:
                self.run_acquire(external=wav)
        self.assertIn("size limit", str(ctx.exception))
        self.assertEqual(list(temp_root.iterdir()), [])
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "manifest.json").exists())
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / aa.LOCK_NAME).exists())

    def test_actual_decoded_size_is_checked_after_decoding(self):
        wav = self.write_wav("voice.wav", rate=16000, channels=1, seconds=0.5)
        real = aa._measure_wav

        def inflated(path, stream):
            m = real(path, stream)
            m["sample_count"] = aa.MAX_DECODED_PCM_BYTES  # claims far more samples than the limit allows
            return m

        with mock.patch.object(aa, "_measure_wav", side_effect=inflated):
            with self.assertRaises(aa.AudioError) as ctx:
                self.run_acquire(external=wav)
        self.assertIn("above the", str(ctx.exception))
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / "manifest.json").exists())

    def test_lock_is_released_after_a_refused_acquisition(self):
        self.make_video("session-001", with_audio=False)
        with self.assertRaises(aa.AudioError):
            self.run_acquire()
        self.assertFalse((self.datasets / "person" / "session-001" / "audio" / aa.LOCK_NAME).exists())


class CommandLineTests(unittest.TestCase):
    def test_cli_returns_one_for_a_refused_request(self):
        self.assertEqual(cli.main(["../escape", "session-001"]), 1)

    def test_cli_refuses_an_offset_for_embedded_audio(self):
        with mock.patch.object(cli, "acquire_session_audio", side_effect=aa.AudioError("refused")) as call:
            self.assertEqual(cli.main(["person", "session-001", "--offset-us", "5"]), 1)
            self.assertEqual(call.call_args.kwargs["offset_us"], 5)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
