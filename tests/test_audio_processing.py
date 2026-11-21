"""Audio preprocessing, timing, features and dataset integrity (phase H9.2).

Synthetic inputs in temporary folders. Tests that decode through FFmpeg are skipped with a visible
reason when ffmpeg and ffprobe are not installed, unless AUDIO_TESTS_REQUIRE_FFMPEG=1, in which case
a missing FFmpeg is a failure. Nothing under recordings/ or datasets/ is read or written.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))
sys.path.insert(0, str(ROOT / "src" / "extraction"))

from extraction import audio_acquisition as aa  # noqa: E402
from extraction import audio_processing as ap  # noqa: E402
from extraction import build_audio_features as cli  # noqa: E402
from extraction.audio_signal import AudioConfig  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
REQUIRE_FFMPEG = os.environ.get("AUDIO_TESTS_REQUIRE_FFMPEG") == "1"


def needs_ffmpeg(cls):
    if HAVE_FFMPEG:
        return cls
    if REQUIRE_FFMPEG:
        def fail(_cls):
            raise AssertionError("ffmpeg and ffprobe are required because AUDIO_TESTS_REQUIRE_FFMPEG=1")
        cls.setUpClass = classmethod(fail)
        return cls
    return unittest.skip("ffmpeg/ffprobe not on PATH: H9.2 FFmpeg tests were NOT run")(cls)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_wav(path, samples, rate, channels):
    """samples: float array shaped (n, channels) or (n,), written as 16-bit PCM."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim == 1:
        data = data[:, None]
    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.reshape(-1).tobytes())
    return path


def tone(freq, rate, seconds, amp=0.5):
    t = np.arange(int(rate * seconds)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


@needs_ffmpeg
class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audio_h92_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.recs = self.tmp / "recordings"
        self.datasets = self.tmp / "datasets"
        self.work = self.tmp / "inputs"
        self.work.mkdir()

    # -- helpers
    def audio_dir(self, session="session-001", person="person"):
        return self.datasets / person / session / "audio"

    def acquire(self, external=None, session="session-001", **kw):
        return aa.acquire_session_audio("person", session, external=external,
                                        recordings_root=self.recs, datasets_root=self.datasets, **kw)

    def process(self, session="session-001", **kw):
        return ap.process_session_audio("person", session, recordings_root=self.recs,
                                        datasets_root=self.datasets, **kw)

    def features(self, session="session-001"):
        path = self.audio_dir(session) / "features" / "audio_features.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        return [dict(zip(lines[0].split(","), line.split(","))) for line in lines[1:]]

    def quality(self, session="session-001"):
        return json.loads((self.audio_dir(session) / "quality" / "audio_quality.json").read_text(encoding="utf-8"))

    def make_video(self, name="session-001", audio_delay=0.0, video_delay=0.0):
        path = self.recs / "person" / f"{name}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        if video_delay:
            args += ["-itsoffset", str(video_delay)]
        args += ["-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=1.5"]
        if audio_delay:
            args += ["-itsoffset", str(audio_delay)]
        args += ["-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=1.5",
                 "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-shortest", str(path)]
        subprocess.run(args, check=True, capture_output=True)
        return path

    # -- preprocessing
    def test_mono_16k_wav_keeps_rate_and_sample_count(self):
        wav = write_wav(self.work / "mono.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav, retain=True)
        m = self.process(external=None)
        self.assertEqual(m["frames"], (16000 - 640) // 160 + 1)
        q = self.quality()
        self.assertEqual(q["analysis"]["sample_rate"], 16000)
        self.assertEqual(q["analysis"]["sample_count"], 16000)
        self.assertEqual(q["source"]["source_channels"], 1)

    def test_stereo_44k_is_resampled_to_the_expected_length(self):
        wav = write_wav(self.work / "stereo.wav", np.stack([tone(200, 44100, 1.0)] * 2, axis=1), 44100, 2)
        self.acquire(external=wav, retain=True)
        self.process()
        q = self.quality()
        self.assertEqual(q["source"]["source_sample_rate"], 44100)
        self.assertEqual(q["analysis"]["sample_count"], 16000)
        self.assertLessEqual(abs(q["analysis"]["length_difference_samples"]), 1.0)

    def test_stereo_downmix_is_the_equal_weight_mean_of_the_channels(self):
        left_only = np.stack([tone(200, 16000, 1.0, amp=0.5), np.zeros(16000)], axis=1)
        wav = write_wav(self.work / "left.wav", left_only, 16000, 2)
        self.acquire(external=wav, retain=True)
        self.process()
        peak = self.quality()["amplitude"]["peak_linear"]
        self.assertAlmostEqual(peak, 0.25, delta=0.01)  # 0.5 from the left, halved by the mean

    def test_four_channel_downmix_divides_by_four(self):
        data = np.zeros((16000, 4))
        data[:, 0] = tone(200, 16000, 1.0, amp=0.8)
        wav = write_wav(self.work / "quad.wav", data, 16000, 4)
        self.acquire(external=wav, retain=True)
        self.process()
        self.assertAlmostEqual(self.quality()["amplitude"]["peak_linear"], 0.2, delta=0.01)
        self.assertEqual(self.quality()["source"]["source_channels"], 4)

    def test_mp3_input_is_processed(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 44100, 1.0), 44100, 1)
        mp3 = self.work / "tone.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                        "-codec:a", "libmp3lame", str(mp3)], check=True, capture_output=True)
        self.acquire(external=mp3)
        self.process(external=mp3)
        self.assertEqual(self.quality()["analysis"]["sample_rate"], 16000)

    def test_silence_is_flagged_silent_and_has_no_pitch(self):
        wav = write_wav(self.work / "silence.wav", np.zeros(16000), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        q = self.quality()
        self.assertEqual(q["frames"]["silent"], q["frames"]["count"])
        self.assertEqual(q["frames"]["voiced"], 0)
        self.assertEqual(q["vad"]["speech_frames"], 0)
        for row in self.features():
            self.assertEqual(row["pitch_status"], "silent")
            self.assertEqual(row["f0_hz"], "")

    def test_full_scale_square_wave_reports_clipping(self):
        square = np.where(np.sin(2 * np.pi * 100 * np.arange(16000) / 16000) >= 0, 1.0, -1.0)
        wav = write_wav(self.work / "square.wav", square, 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        self.assertGreater(self.quality()["amplitude"]["clipping_fraction"], 0.5)

    def test_quality_separates_measurements_from_judgements(self):
        wav = write_wav(self.work / "tone.wav", tone(220, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        q = self.quality()
        self.assertIn("no subjective quality score", q["kind"])
        self.assertFalse(q["vad"]["is_ground_truth"])
        self.assertFalse(any("accuracy" in k for k in q))

    # -- timing
    def test_external_offset_places_frames_on_the_session_clock(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 0.5), 16000, 1)
        self.acquire(external=wav, offset_us=250_000)
        self.process(external=wav)
        first = self.features()[0]
        centre = int(first["centre_sample"])
        self.assertEqual(centre, 320)
        self.assertEqual(int(first["t_us"]), 250_000 + 20_000)

    def test_timestamps_have_no_drift_over_a_long_run(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 44100, 2.0), 44100, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        rows = self.features()
        for row in rows[:: max(1, len(rows) // 25)]:
            centre = int(row["centre_sample"])
            expected = round(centre * 1_000_000 / 16000)
            self.assertLessEqual(abs(int(row["t_us"]) - expected), 1)
        steps = {int(b["t_us"]) - int(a["t_us"]) for a, b in zip(rows, rows[1:])}
        self.assertEqual(steps, {10_000})

    def test_non_zero_audio_and_video_start_times_are_placed_correctly(self):
        self.make_video(audio_delay=0.25, video_delay=0.5)
        self.acquire()
        self.process()
        acq = json.loads((self.audio_dir() / "manifest.json").read_text(encoding="utf-8"))
        tl = acq["timeline"]
        self.assertNotEqual(tl["audio_start_us"], 0)
        self.assertNotEqual(tl["video_start_us"], 0)
        base = tl["audio_start_us"] - tl["video_start_us"]
        first = self.features()[0]
        self.assertEqual(int(first["t_us"]), base + 20_000)
        self.assertEqual(tl["origin_us"], tl["video_start_us"])

    # -- dataset integrity
    def test_processing_leaves_h91_outputs_and_the_source_unchanged(self):
        video = self.make_video()
        self.acquire()
        before_manifest = sha(self.audio_dir() / "manifest.json")
        before_video = sha(video)
        self.process()
        self.assertEqual(sha(self.audio_dir() / "manifest.json"), before_manifest)
        self.assertEqual(sha(video), before_video)

    def test_published_outputs_match_the_processing_manifest(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        m = self.process(external=wav)
        for rel, entry in m["files"].items():
            self.assertEqual(sha(self.audio_dir() / rel), entry["sha256"], rel)
        self.assertTrue((self.audio_dir() / ap.PROCESSING_MANIFEST).is_file())

    def test_existing_outputs_are_refused_without_regenerate(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        from file_safety import OutputExistsError
        with self.assertRaises(OutputExistsError):
            self.process(external=wav)

    def test_regenerate_replaces_the_outputs_and_leaves_no_previous_folder(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        self.process(external=wav, regenerate=True)
        leftovers = [p.name for p in self.audio_dir().iterdir() if p.name.startswith(".h92")]
        self.assertEqual(leftovers, [])

    def test_failure_during_analysis_publishes_nothing_and_cleans_up(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        acq_before = sha(self.audio_dir() / "manifest.json")
        with mock.patch.object(ap, "analyse", side_effect=ap.FeatureError("forced failure")):
            with self.assertRaises(ap.FeatureError):
                self.process(external=wav)
        audio = self.audio_dir()
        self.assertFalse((audio / "features").exists())
        self.assertFalse((audio / ap.PROCESSING_MANIFEST).exists())
        self.assertEqual([p.name for p in audio.iterdir() if p.name.startswith(".h92")], [])
        self.assertFalse((audio / aa.LOCK_NAME).exists())
        self.assertEqual(sha(audio / "manifest.json"), acq_before)

    def test_failed_regeneration_restores_the_previous_outputs(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        before = sha(self.audio_dir() / "features" / "audio_features.csv")
        real_replace = os.replace
        calls = {"n": 0}

        def failing_replace(src, dst):
            calls["n"] += 1
            if calls["n"] == 5:  # old outputs are moved first (3 calls); this is the 2nd new output
                raise OSError("simulated failure during publication")
            return real_replace(src, dst)

        with mock.patch.object(ap.os, "replace", side_effect=failing_replace):
            with self.assertRaises(OSError):
                self.process(external=wav, regenerate=True)
        self.assertEqual(sha(self.audio_dir() / "features" / "audio_features.csv"), before)
        self.assertEqual([p.name for p in self.audio_dir().iterdir() if p.name.startswith(".h92")], [])

    def test_a_changed_source_is_refused(self):
        video = self.make_video()
        self.acquire()
        with open(video, "ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(ap.FeatureError) as ctx:
            self.process()
        self.assertIn("does not match", str(ctx.exception))

    def test_an_external_file_that_was_not_retained_must_be_supplied_again(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav, retain=False)
        with self.assertRaises(ap.FeatureError) as ctx:
            self.process()
        self.assertIn("--external", str(ctx.exception))
        self.process(external=wav)  # the same file is accepted

    def test_missing_manifest_is_refused(self):
        with self.assertRaises(ap.FeatureError):
            self.process()

    def test_invalid_manifest_is_refused(self):
        self.make_video()
        self.acquire()
        (self.audio_dir() / "manifest.json").write_text("{not json", encoding="utf-8")
        with self.assertRaises(ap.FeatureError):
            self.process()

    def test_a_manifest_with_an_unknown_schema_is_refused(self):
        self.make_video()
        self.acquire()
        path = self.audio_dir() / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema"] = "something-else"
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ap.FeatureError):
            self.process()

    def test_a_lock_held_by_another_run_refuses_processing(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        with aa.session_lock(self.audio_dir()):
            with self.assertRaises(aa.AudioError):
                self.process(external=wav)

    def test_traversal_in_names_is_refused(self):
        with self.assertRaises(ap.FeatureError):
            ap.process_session_audio("../escape", "session-001", datasets_root=self.datasets)

    def test_no_stage_folders_remain_after_success(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        self.assertEqual([p.name for p in self.audio_dir().iterdir() if p.name.startswith(".h92")], [])

    def test_the_effective_configuration_is_recorded(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav, config=AudioConfig(pitch_threshold=0.7, vad_db=-35))
        meta = json.loads((self.audio_dir() / "features" / "feature_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["config"]["pitch_threshold"], 0.7)
        self.assertEqual(meta["config"]["vad_db"], -35)
        self.assertEqual(meta["config"]["hop_samples"], 160)

    def test_cli_refuses_an_invalid_configuration_without_processing(self):
        with mock.patch.object(cli, "process_session_audio") as call:
            self.assertEqual(cli.main(["person", "session-001", "--hop-ms", "0"]), 1)
            call.assert_not_called()

    def test_cli_does_not_run_acquisition(self):
        with mock.patch.object(aa, "acquire_session_audio") as acquire, \
                mock.patch.object(cli, "process_session_audio", side_effect=ap.FeatureError("refused")):
            self.assertEqual(cli.main(["person", "session-001"]), 1)
            acquire.assert_not_called()

    def test_negative_external_offset_moves_frames_before_the_origin(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 0.5), 16000, 1)
        self.acquire(external=wav, offset_us=-100_000)
        self.process(external=wav)
        first = self.features()[0]
        # Sign convention: session = audio_start + offset - origin, so -100 ms moves frame 0 to -80 ms.
        self.assertEqual(int(first["t_us"]), -100_000 + 20_000)

    def test_resampling_is_stopped_while_running_when_its_output_exceeds_the_limit(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        out = self.work / "analysis.f32"
        args = [aa._tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_f32le", "-f", "f32le", str(out)]
        with self.assertRaises(ap.FeatureError) as ctx:
            ap._run_bounded(args, out, 1000, wav)
        self.assertIn("size limit", str(ctx.exception))

    def test_planned_size_over_the_limit_is_refused_before_any_decode(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav, retain=True)
        with mock.patch.object(ap, "MAX_ANALYSIS_BYTES", 1000), \
                mock.patch.object(ap, "_run_bounded", side_effect=AssertionError("must not decode")):
            with self.assertRaises(ap.FeatureError) as ctx:
                self.process()
        self.assertIn("analysis limit", str(ctx.exception))

    def test_failed_regeneration_restores_the_previous_completion_marker(self):
        wav = write_wav(self.work / "tone.wav", tone(200, 16000, 1.0), 16000, 1)
        self.acquire(external=wav)
        self.process(external=wav)
        marker = self.audio_dir() / ap.PROCESSING_MANIFEST
        before = sha(marker)
        real_write = ap._write_json

        def failing_write(path, data):
            if Path(path).name == ap.PROCESSING_MANIFEST:
                raise OSError("simulated failure writing the completion marker")
            return real_write(path, data)

        with mock.patch.object(ap, "_write_json", side_effect=failing_write):
            with self.assertRaises(OSError):
                self.process(external=wav, regenerate=True)
        self.assertEqual(sha(marker), before)
        for rel, entry in json.loads(marker.read_text(encoding="utf-8"))["files"].items():
            self.assertEqual(sha(self.audio_dir() / rel), entry["sha256"], rel)

    # -- memory and performance evidence is recorded by the report, not asserted here.


if __name__ == "__main__":
    unittest.main()
