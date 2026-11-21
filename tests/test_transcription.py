"""Transcription dataset layer (phase H9.3a): records, timing, validation, publication and integrity.

A deterministic fake engine. No ASR model, no runtime, no downloads. End-to-end tests decode through
FFmpeg and are skipped with a visible reason when FFmpeg is missing, unless AUDIO_TESTS_REQUIRE_FFMPEG=1.
"""

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import audio_acquisition as aa  # noqa: E402
from extraction import audio_processing as ap  # noqa: E402
from extraction import transcription as tr  # noqa: E402
from extraction.asr_interface import (  # noqa: E402
    EngineInfo,
    EngineResponseError,
    EngineResult,
    RawSegment,
    RawWord,
    TranscriptionRequest,
    validate_engine_result,
)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
REQUIRE_FFMPEG = os.environ.get("AUDIO_TESTS_REQUIRE_FFMPEG") == "1"
INFO = EngineInfo(engine="fake", version="1.0", model_name="fake-small", model_revision="rev1",
                  model_sha256="ab" * 32)


def needs_ffmpeg(cls):
    if HAVE_FFMPEG:
        return cls
    if REQUIRE_FFMPEG:
        def fail(_cls):
            raise AssertionError("ffmpeg and ffprobe are required because AUDIO_TESTS_REQUIRE_FFMPEG=1")
        cls.setUpClass = classmethod(fail)
        return cls
    return unittest.skip("ffmpeg/ffprobe not on PATH: H9.3a end-to-end tests were NOT run")(cls)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeEngine:
    """Returns one scripted result, deterministically. Implements the AsrEngine shape."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = []

    @property
    def info(self):
        return INFO

    def transcribe(self, samples, sample_rate, request):
        self.calls.append((len(samples), sample_rate, request.language))
        if self._error is not None:
            raise self._error
        return self._result


def hindi_result(language="hi", probability=0.91):
    segs = (RawSegment(
        text=" नमस्ते दुनिया, Hello world!",
        start_s=0.25, end_s=0.75,
        words=(RawWord(" नमस्ते", 0.25, 0.4, 0.9), RawWord(" दुनिया,", 0.4, 0.6, None),
               RawWord(" Hello", 0.6, 0.7, 0.5)),
        avg_logprob=-0.2, no_speech_prob=None),)
    return EngineResult(text=segs[0].text, language=language, language_probability=probability,
                        segments=segs, info=INFO)


def records_for(result, rate=16000, n=16000, base_us=0):
    req = TranscriptionRequest(language="hi" if result.language == "hi" else "en")
    result = validate_engine_result(result, req, n / rate)
    segs, words = tr.build_records(result, req, rate, n, base_us)
    tr.validate_records(segs, words, rate, n, base_us)
    return result, req, segs, words


class TextTests(unittest.TestCase):
    def test_normalisation_is_nfc_and_whitespace_only(self):
        decomposed = "café   test"  # e + combining acute accent
        self.assertEqual(tr.normalize_text(decomposed), "café test")

    def test_case_punctuation_and_script_are_not_changed(self):
        self.assertEqual(tr.normalize_text("Hello, World!"), "Hello, World!")
        self.assertEqual(tr.normalize_text(" नमस्ते "), "नमस्ते")  # Devanagari kept as spoken

    def test_raw_text_is_stored_unchanged_beside_the_normalised_text(self):
        _, _, segs, words = records_for(hindi_result())
        self.assertEqual(segs[0]["text_raw"], " नमस्ते दुनिया, Hello world!")
        self.assertEqual(segs[0]["text_normalized"], "नमस्ते दुनिया, Hello world!")
        self.assertEqual(words[0]["word_raw"], " नमस्ते")

    def test_raw_float_times_are_kept_exactly(self):
        _, _, segs, words = records_for(hindi_result())
        self.assertEqual(segs[0]["raw_start_s"], repr(0.25))
        self.assertEqual(words[1]["raw_end_s"], repr(0.6))


class RecordTimingTests(unittest.TestCase):
    def test_samples_and_microseconds_follow_the_session_convention(self):
        _, _, segs, words = records_for(hindi_result(), base_us=21_333)
        self.assertEqual(segs[0]["start_sample"], 4000)          # 0.25 s at 16 kHz
        self.assertEqual(segs[0]["start_us"], 21_333 + 250_000)
        self.assertEqual(words[0]["end_sample"], 6400)
        self.assertEqual(words[0]["end_us"], 21_333 + 400_000)

    def test_negative_session_timestamps_are_valid(self):
        base = 0 + (-500_000) - 0  # audio starts 0.5 s before the video origin
        _, _, segs, words = records_for(hindi_result(), base_us=base)
        self.assertEqual(segs[0]["start_us"], -250_000)
        self.assertLess(words[0]["start_us"], 0)

    def test_an_external_offset_moves_times_by_exactly_that_amount(self):
        _, _, a, _ = records_for(hindi_result(), base_us=0)
        _, _, b, _ = records_for(hindi_result(), base_us=1_000_000)
        self.assertEqual(b[0]["start_us"] - a[0]["start_us"], 1_000_000)

    def test_no_cumulative_drift_over_a_long_signal(self):
        n = 16000 * 3600
        req = TranscriptionRequest(language="en")
        seg = RawSegment(text="x", start_s=3599.0, end_s=3599.5, words=(), avg_logprob=None, no_speech_prob=None)
        result = EngineResult(text="x", language="en", language_probability=None, segments=(seg,), info=INFO)
        result = validate_engine_result(result, req, 3600.0)
        segs, _ = tr.build_records(result, req, 16000, n, 0)
        self.assertEqual(segs[0]["start_us"], 3_599_000_000)

    def test_a_segment_ending_beyond_the_signal_is_refused(self):
        seg = RawSegment(text="x", start_s=0.0, end_s=0.9, words=(), avg_logprob=None, no_speech_prob=None)
        result = EngineResult(text="x", language="en", language_probability=None, segments=(seg,), info=INFO)
        req = TranscriptionRequest(language="en")
        with self.assertRaises(tr.TranscriptionError):
            tr.build_records(result, req, 16000, 8000, 0)  # signal is 0.5 s, segment ends at 0.9 s

    def test_tampered_microseconds_are_detected(self):
        _, _, segs, words = records_for(hindi_result())
        segs[0]["start_us"] += 1
        with self.assertRaises(tr.TranscriptionError):
            tr.validate_records(segs, words, 16000, 16000, 0)

    def test_a_word_probability_outside_range_is_detected(self):
        _, _, segs, words = records_for(hindi_result())
        words[0]["probability"] = 1.5
        with self.assertRaises(tr.TranscriptionError):
            tr.validate_records(segs, words, 16000, 16000, 0)

    def test_missing_probabilities_stay_null(self):
        _, _, segs, words = records_for(hindi_result())
        self.assertIsNone(words[1]["probability"])
        self.assertIsNone(segs[0]["no_speech_prob"])
        self.assertIsNone(segs[0]["speaker"])


class CsvTests(unittest.TestCase):
    def test_missing_values_are_empty_cells_and_text_round_trips(self):
        _, _, _, words = records_for(hindi_result())
        d = Path(tempfile.mkdtemp(prefix="tr_csv_"))
        self.addCleanup(shutil.rmtree, d, True)
        tr._write_csv(d / "w.csv", tr.WORD_COLUMNS, words)
        with open(d / "w.csv", encoding="utf-8", newline="") as h:
            rows = list(csv.DictReader(h))
        self.assertEqual(rows[1]["probability"], "")          # None written as empty, never 0
        self.assertEqual(rows[0]["word_raw"], " नमस्ते")         # Devanagari survives UTF-8
        self.assertEqual(rows[1]["word_raw"], " दुनिया,")        # a comma inside a field is quoted correctly
        self.assertEqual(rows[0]["probability"], "0.9")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tr_pub_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.audio = self.tmp / "audio"
        self.audio.mkdir()

    def stage_for(self, marker):
        """A complete staged transcription with a recognisable marker in transcript.json."""
        stage = self.audio / f".h93-stage-{marker}"
        _, _, segs, words = records_for(hindi_result())
        doc = {"schema": tr.SCHEMA, "counts": {"segments": len(segs), "words": len(words)},
               "engine": INFO.as_dict(), "language": {"selected": "hi"}, "raw_text": marker,
               "created_utc": "2026-01-01T00:00:00+00:00"}
        tr.write_outputs(stage, doc, segs, words)
        tr.read_back_check(stage, doc, segs, words)
        tr.write_metadata(stage, doc, [tr.TRANSCRIPT_JSON, tr.SEGMENTS_CSV, tr.WORDS_CSV])
        return stage

    def test_a_fresh_publication_moves_the_staged_folder_into_place(self):
        stage = self.stage_for("first")
        tr.publish(stage, self.audio, regenerate=False)
        self.assertFalse(stage.exists())
        self.assertTrue((self.audio / tr.TRANSCRIPTION_DIR / tr.METADATA_JSON).is_file())

    def test_regeneration_replaces_the_previous_outputs(self):
        tr.publish(self.stage_for("old"), self.audio, regenerate=False)
        tr.publish(self.stage_for("new"), self.audio, regenerate=True)
        doc = json.loads((self.audio / tr.TRANSCRIPTION_DIR / tr.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
        self.assertEqual(doc["raw_text"], "new")
        self.assertEqual([p.name for p in self.audio.iterdir() if p.name.startswith(".h93")], [])

    def test_failed_regeneration_restores_the_previous_outputs(self):
        tr.publish(self.stage_for("old"), self.audio, regenerate=False)
        before = sha(self.audio / tr.TRANSCRIPTION_DIR / tr.TRANSCRIPT_JSON)
        stage = self.stage_for("new")
        real_replace = os.replace
        calls = {"n": 0}

        def failing(src, dst):
            calls["n"] += 1
            if calls["n"] == 2:  # 1st: old folder moved aside; 2nd: new folder into place
                raise OSError("simulated failure during publication")
            return real_replace(src, dst)

        with mock.patch.object(tr.os, "replace", side_effect=failing):
            with self.assertRaises(OSError):
                tr.publish(stage, self.audio, regenerate=True)
        self.assertEqual(sha(self.audio / tr.TRANSCRIPTION_DIR / tr.TRANSCRIPT_JSON), before)
        self.assertEqual([p.name for p in self.audio.iterdir() if p.name.startswith(".h93-previous")], [])

    def test_existing_outputs_are_refused_without_regenerate(self):
        tr.publish(self.stage_for("old"), self.audio, regenerate=False)
        from file_safety import OutputExistsError
        from file_safety import check_writable
        with self.assertRaises(OutputExistsError):
            check_writable([self.audio / tr.TRANSCRIPTION_DIR], False)

    def test_read_back_detects_a_row_count_mismatch(self):
        _, _, segs, words = records_for(hindi_result())
        stage = self.audio / ".h93-readback"
        doc = {"counts": {}, "raw_text": "x"}
        tr.write_outputs(stage, doc, segs, words)
        with self.assertRaises(tr.TranscriptionError):
            tr.read_back_check(stage, doc, segs, words + words)

    def test_h92_outputs_are_verified_before_use(self):
        feats = self.audio / "features"
        feats.mkdir()
        (feats / "audio_features.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        marker = {"files": {"features/audio_features.csv": {"sha256": sha(feats / "audio_features.csv")}}}
        (self.audio / ap.PROCESSING_MANIFEST).write_text(json.dumps(marker), encoding="utf-8")
        tr.verify_processing_outputs(self.audio)  # passes
        (feats / "audio_features.csv").write_text("a,b\n9,9\n", encoding="utf-8")
        with self.assertRaises(tr.TranscriptionError):
            tr.verify_processing_outputs(self.audio)

    def test_missing_h92_outputs_are_refused(self):
        with self.assertRaises(tr.TranscriptionError):
            tr.verify_processing_outputs(self.audio)


@needs_ffmpeg
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tr_e2e_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.recs, self.datasets = self.tmp / "recordings", self.tmp / "datasets"
        self.inputs = self.tmp / "inputs"
        self.inputs.mkdir()

    def wav(self, name="speech.wav", seconds=1.0):
        path = self.inputs / name
        n = int(16000 * seconds)
        t = np.arange(n) / 16000
        pcm = (np.sin(2 * np.pi * 200 * t) * 0.3 * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm.tobytes())
        return path

    def prepare(self, wav, offset=0):
        aa.acquire_session_audio("person", "session-001", external=wav, offset_us=offset, retain=True,
                                 recordings_root=self.recs, datasets_root=self.datasets)
        ap.process_session_audio("person", "session-001", external=wav, recordings_root=self.recs,
                                 datasets_root=self.datasets)

    def transcribe(self, engine, language="hi", **kw):
        return tr.process_transcription("person", "session-001", engine, TranscriptionRequest(language=language),
                                        recordings_root=self.recs, datasets_root=self.datasets, **kw)

    def audio_dir(self):
        return self.datasets / "person" / "session-001" / "audio"

    def test_end_to_end_publishes_the_dataset_and_keeps_raw_text(self):
        wav = self.wav()
        self.prepare(wav)
        before = {p: sha(self.audio_dir() / p) for p in ("manifest.json", ap.PROCESSING_MANIFEST)}
        engine = FakeEngine(hindi_result("hi"))
        meta = self.transcribe(engine, language="hi")
        out = self.audio_dir() / tr.TRANSCRIPTION_DIR
        doc = json.loads((out / tr.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
        self.assertEqual(doc["raw_text"], " नमस्ते दुनिया, Hello world!")
        self.assertEqual(doc["language"]["selected"], "hi")
        self.assertEqual(doc["language"]["requested"], "hi")
        self.assertEqual(doc["engine"]["engine"], "fake")
        self.assertEqual(doc["analysis"]["sample_rate"], 16000)
        self.assertEqual(engine.calls[-1], (16000, 16000, "hi"))
        for rel, entry in meta["files"].items():
            self.assertEqual(sha(out / rel), entry["sha256"], rel)
        self.assertEqual({p: sha(self.audio_dir() / p) for p in before}, before)  # H9.1 / H9.2 untouched

    def test_automatic_detection_records_the_detected_language_and_probability(self):
        self.prepare(self.wav())
        self.transcribe(FakeEngine(hindi_result("en", 0.97)), language="auto")
        doc = json.loads((self.audio_dir() / tr.TRANSCRIPTION_DIR / tr.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
        self.assertEqual(doc["language"]["requested"], "auto")
        self.assertEqual(doc["language"]["selected"], "en")
        self.assertTrue(doc["language"]["detected"])
        self.assertEqual(doc["language"]["probability"], 0.97)

    def test_an_engine_failure_publishes_nothing(self):
        self.prepare(self.wav())
        with self.assertRaises(tr.TranscriptionError) as ctx:
            self.transcribe(FakeEngine(error=RuntimeError("boom")))
        self.assertIn("engine failed", str(ctx.exception))
        self.assertFalse((self.audio_dir() / tr.TRANSCRIPTION_DIR).exists())
        self.assertEqual([p.name for p in self.audio_dir().iterdir() if p.name.startswith(".h93")], [])
        self.assertFalse((self.audio_dir() / aa.LOCK_NAME).exists())

    def test_an_invalid_engine_response_publishes_nothing(self):
        self.prepare(self.wav())
        bad = hindi_result("hi")
        with self.assertRaises(EngineResponseError):
            self.transcribe(FakeEngine(bad), language="en")  # explicit English answered with Hindi
        self.assertFalse((self.audio_dir() / tr.TRANSCRIPTION_DIR).exists())

    def test_existing_transcription_is_refused_without_regenerate_and_replaced_with_it(self):
        self.prepare(self.wav())
        self.transcribe(FakeEngine(hindi_result("hi")))
        from file_safety import OutputExistsError
        with self.assertRaises(OutputExistsError):
            self.transcribe(FakeEngine(hindi_result("hi")))
        self.transcribe(FakeEngine(hindi_result("hi")), regenerate=True)

    def test_a_changed_source_is_refused(self):
        wav = self.wav()
        self.prepare(wav)
        acq = json.loads((self.audio_dir() / "manifest.json").read_text(encoding="utf-8"))
        retained = self.audio_dir() / acq["retention"]["stored_file"].split("/", 1)[1]
        with open(retained, "ab") as h:  # the retained copy is what processing reads
            h.write(b"tamper")
        with self.assertRaises(ap.FeatureError):
            self.transcribe(FakeEngine(hindi_result("hi")))

    def test_a_tampered_h92_output_is_refused(self):
        self.prepare(self.wav())
        target = self.audio_dir() / "features" / "audio_features.csv"
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(tr.TranscriptionError):
            self.transcribe(FakeEngine(hindi_result("hi")))

    def test_the_engine_receives_the_analysis_signal_at_the_analysis_rate(self):
        self.prepare(self.wav(seconds=0.5))
        short = EngineResult(text="hi", language="hi", language_probability=None, info=INFO,
                             segments=(RawSegment(text="hi", start_s=0.1, end_s=0.4, words=(),
                                                  avg_logprob=None, no_speech_prob=None),))
        engine = FakeEngine(short)
        self.transcribe(engine)
        self.assertEqual(engine.calls[-1], (8000, 16000, "hi"))

    def test_negative_external_offset_gives_negative_session_times(self):
        wav = self.wav()
        self.prepare(wav, offset=-500_000)
        self.transcribe(FakeEngine(hindi_result("hi")))
        doc = json.loads((self.audio_dir() / tr.TRANSCRIPTION_DIR / tr.TRANSCRIPT_JSON).read_text(encoding="utf-8"))
        self.assertEqual(doc["timeline"]["external_offset_us"], -500_000)
        rows = list(csv.DictReader(open(self.audio_dir() / tr.TRANSCRIPTION_DIR / tr.SEGMENTS_CSV,
                                        encoding="utf-8", newline="")))
        self.assertEqual(int(rows[0]["start_us"]), -500_000 + 250_000)

    def test_no_stage_folders_remain_after_success(self):
        self.prepare(self.wav())
        self.transcribe(FakeEngine(hindi_result("hi")))
        self.assertEqual([p.name for p in self.audio_dir().iterdir() if p.name.startswith(".h93")], [])


if __name__ == "__main__":
    unittest.main()
