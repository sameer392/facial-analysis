"""ASR adapter interface (phase H9.3a): fake engine, language selection and response validation.

No ASR model, no runtime and no downloads. A deterministic fake engine stands in for an ASR engine.
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction.asr_interface import (  # noqa: E402
    AsrEngine,
    AUTO,
    ConfigError,
    DecodingOptions,
    EngineInfo,
    EngineResponseError,
    EngineResult,
    RawSegment,
    RawWord,
    TranscriptionRequest,
    seconds_to_sample,
    validate_engine_result,
)

INFO = EngineInfo(engine="fake", version="0", model_name="fake-model", model_revision=None, model_sha256=None)


class FakeEngine(AsrEngine):
    """Deterministic: returns the same scripted result for the same request."""

    def __init__(self, result: EngineResult):
        self._result = result
        self.calls = []

    @property
    def info(self):
        return INFO

    def transcribe(self, samples, sample_rate, request):
        self.calls.append((len(samples), sample_rate, request.language))
        return self._result


def make_result(language="en", probability=None, segments=None, text=None):
    if segments is None:
        segments = (
            RawSegment(text=" hello world", start_s=0.25, end_s=1.0,
                       words=(RawWord(" hello", 0.25, 0.5, 0.9), RawWord(" world", 0.5, 1.0, None)),
                       avg_logprob=-0.3, no_speech_prob=0.01),
        )
    if text is None:
        text = "".join(s.text for s in segments)
    return EngineResult(text=text, language=language, language_probability=probability,
                        segments=tuple(segments), info=INFO)


class RequestTests(unittest.TestCase):
    def test_explicit_english_and_hindi_are_accepted(self):
        self.assertEqual(TranscriptionRequest(language="en").language, "en")
        self.assertEqual(TranscriptionRequest(language="hi").language, "hi")
        self.assertTrue(TranscriptionRequest(language="hi").explicit)

    def test_automatic_detection_is_the_default(self):
        req = TranscriptionRequest()
        self.assertEqual(req.language, AUTO)
        self.assertFalse(req.explicit)

    def test_unsupported_language_request_is_refused(self):
        with self.assertRaises(ConfigError):
            TranscriptionRequest(language="fr")

    def test_decoding_options_are_range_checked(self):
        with self.assertRaises(ConfigError):
            DecodingOptions(beam_size=0)
        with self.assertRaises(ConfigError):
            DecodingOptions(temperature=float("nan"))

    def test_engine_info_needs_a_name_and_version(self):
        with self.assertRaises(ConfigError):
            EngineInfo(engine="", version="1", model_name="m")


class FakeEngineTests(unittest.TestCase):
    def test_fake_engine_is_deterministic(self):
        engine = FakeEngine(make_result())
        req = TranscriptionRequest(language="en")
        a = validate_engine_result(engine.transcribe(np.zeros(16000, "f4"), 16000, req), req, 1.0)
        b = validate_engine_result(engine.transcribe(np.zeros(16000, "f4"), 16000, req), req, 1.0)
        self.assertEqual(a, b)
        self.assertEqual(engine.calls[-1], (16000, 16000, "en"))

    def test_explicit_english_selection_is_honoured(self):
        req = TranscriptionRequest(language="en")
        result = validate_engine_result(FakeEngine(make_result("en")).transcribe(np.zeros(16000), 16000, req),
                                        req, 1.0)
        self.assertEqual(result.language, "en")

    def test_explicit_hindi_selection_is_honoured(self):
        req = TranscriptionRequest(language="hi")
        result = validate_engine_result(FakeEngine(make_result("hi")).transcribe(np.zeros(16000), 16000, req),
                                        req, 1.0)
        self.assertEqual(result.language, "hi")

    def test_explicit_request_answered_in_another_language_is_refused(self):
        req = TranscriptionRequest(language="hi")
        with self.assertRaises(EngineResponseError):
            validate_engine_result(make_result("en"), req, 1.0)

    def test_automatic_detection_accepts_english_and_hindi(self):
        for lang, prob in (("en", 0.97), ("hi", 0.88)):
            result = validate_engine_result(make_result(lang, prob), TranscriptionRequest(), 1.0)
            self.assertEqual((result.language, result.language_probability), (lang, prob))

    def test_automatic_detection_returning_an_unsupported_language_is_refused(self):
        with self.assertRaises(EngineResponseError):
            validate_engine_result(make_result("fr", 0.9), TranscriptionRequest(), 1.0)

    def test_missing_language_probability_is_null_not_invented(self):
        result = validate_engine_result(make_result("en", None), TranscriptionRequest(language="en"), 1.0)
        self.assertIsNone(result.language_probability)

    def test_missing_word_probability_is_null(self):
        result = validate_engine_result(make_result(), TranscriptionRequest(), 1.0)
        self.assertIsNone(result.segments[0].words[1].probability)
        self.assertEqual(result.segments[0].words[0].probability, 0.9)


class TimeConversionTests(unittest.TestCase):
    def test_seconds_round_half_up_to_samples(self):
        self.assertEqual(seconds_to_sample(0.0, 16000), 0)
        self.assertEqual(seconds_to_sample(1.0, 16000), 16000)
        self.assertEqual(seconds_to_sample(0.00003125, 16000), 1)  # half a sample rounds up
        self.assertEqual(seconds_to_sample(0.25, 16000), 4000)

    def test_negative_or_non_finite_times_are_refused(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(EngineResponseError):
                seconds_to_sample(bad, 16000)


class InvalidResponseTests(unittest.TestCase):
    """Each broken response must be refused, never repaired."""

    def check(self, result, language=AUTO, duration=2.0):
        req = TranscriptionRequest(language=language)
        with self.assertRaises(EngineResponseError):
            validate_engine_result(result, req, duration)

    def seg(self, **kw):
        base = dict(text="x", start_s=0.0, end_s=1.0, words=(), avg_logprob=None, no_speech_prob=None)
        base.update(kw)
        return RawSegment(**base)

    def test_not_an_engine_result(self):
        with self.assertRaises(EngineResponseError):
            validate_engine_result({"text": "x"}, TranscriptionRequest(), 1.0)

    def test_text_that_is_not_a_string(self):
        self.check(replace(make_result(), text=123))

    def test_unsupported_language_code(self):
        self.check(make_result("xx"))

    def test_segment_ending_before_it_starts(self):
        self.check(make_result(segments=(self.seg(start_s=0.8, end_s=0.4),)))

    def test_segment_beyond_the_audio(self):
        self.check(make_result(segments=(self.seg(start_s=0.0, end_s=5.0),)), duration=2.0)

    def test_segments_out_of_order(self):
        self.check(make_result(segments=(self.seg(start_s=1.0, end_s=1.5), self.seg(start_s=0.2, end_s=0.8))))

    def test_negative_segment_time(self):
        self.check(make_result(segments=(self.seg(start_s=-0.1, end_s=0.5),)))

    def test_word_outside_its_segment(self):
        w = RawWord("a", 0.0, 0.5)
        self.check(make_result(segments=(self.seg(start_s=0.2, end_s=1.0, words=(w,)),)))

    def test_words_out_of_order(self):
        words = (RawWord("a", 0.5, 0.6), RawWord("b", 0.2, 0.3))
        self.check(make_result(segments=(self.seg(start_s=0.0, end_s=1.0, words=words),)))

    def test_probability_outside_zero_to_one(self):
        w = RawWord("a", 0.1, 0.2, probability=1.5)
        self.check(make_result(segments=(self.seg(start_s=0.0, end_s=1.0, words=(w,)),)))

    def test_probability_that_is_not_finite(self):
        w = RawWord("a", 0.1, 0.2, probability=float("nan"))
        self.check(make_result(segments=(self.seg(start_s=0.0, end_s=1.0, words=(w,)),)))

    def test_positive_avg_logprob_is_refused(self):
        self.check(make_result(segments=(self.seg(avg_logprob=0.4),)))

    def test_word_text_that_is_not_a_string(self):
        w = RawWord(None, 0.1, 0.2)
        self.check(make_result(segments=(self.seg(start_s=0.0, end_s=1.0, words=(w,)),)))


if __name__ == "__main__":
    unittest.main()
