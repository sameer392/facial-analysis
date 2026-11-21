"""faster-whisper adapter, model integrity and GPU lock (phase H9.3b-3).

Mock-based only. Nothing here installs or imports faster-whisper or CTranslate2, touches CUDA or cuDNN,
downloads a model, or loads weights. Every model directory is a temporary synthetic folder with synthetic
bytes and a manifest written for the test. No real manifest is created, and no licence status is changed.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "analysis"))

from extraction import asr_runtime_guard as guard  # noqa: E402
from extraction import faster_whisper_engine as fwe  # noqa: E402
from extraction.asr_interface import (  # noqa: E402
    ConfigError,
    TranscriptionRequest,
    seconds_to_sample,
    validate_engine_result,
)

REVISION = "a" * 40  # a synthetic test revision; it is not the real upstream commit
PINS = {"faster-whisper": fwe.FASTER_WHISPER_PIN, "ctranslate2": fwe.CTRANSLATE2_PIN}


def good_versions(name):
    return PINS[name]


class Fixture(unittest.TestCase):
    """Temporary model root with one synthetic, correctly declared model directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fwe_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.models = self.tmp / "asr-models"
        self.runtime = self.tmp / "asr-runtime"
        self.models.mkdir()
        self.model_dir = self.make_model("synthetic-model")

    def make_model(self, name, licence="cleared", files=None, extra=None):
        folder = self.models / name
        folder.mkdir(parents=True, exist_ok=True)
        files = files or {"model.bin": b"synthetic weights, not a model", "config.json": b"{}"}
        entries = []
        for fname, data in files.items():
            (folder / fname).write_bytes(data)
            entries.append({"name": fname, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {
            "manifest_version": 1,
            "model_name": "synthetic-large-v3",
            "source_repository": "synthetic/test-only",
            "revision": REVISION,
            "licence_status": licence,
            "files": entries,
        }
        if extra is not None:
            manifest = extra(manifest)
        (folder / guard.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
        return folder

    def engine(self, factory=None, device="cpu", compute="int8", versions=good_versions, model_dir=None,
               beam=5, word_timestamps=True, vad_filter=False, lock=None):
        cfg = fwe.FasterWhisperConfig(
            device=device, compute_type=compute,
            model_dir=model_dir if model_dir is not None else self.model_dir,
            beam_size=beam, word_timestamps=word_timestamps, vad_filter=vad_filter)
        return fwe.FasterWhisperEngine(
            cfg, models_root=self.models, runtime_root=self.runtime,
            gpu_lock=lock if lock is not None else guard.GpuLock(self.runtime / "gpu.lock"),
            model_factory=factory or FakeFactory(), version_provider=versions)


class FakeModel:
    def __init__(self, segments, language="en", probability=0.9):
        self._segments = segments
        self._language = language
        self._probability = probability
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append(kwargs)
        info = SimpleNamespace(language=self._language, language_probability=self._probability)
        return iter(self._segments), info


class FakeFactory:
    def __init__(self, model=None, error=None):
        self.model = model or FakeModel(default_segments())
        self.error = error
        self.calls = []

    def __call__(self, model_path, device, compute_type):
        self.calls.append((Path(model_path), device, compute_type))
        if self.error is not None:
            raise self.error
        return self.model


def default_segments(text=" नमस्ते दुनिया", start=0.25, end=0.75):
    return [SimpleNamespace(
        text=text, start=start, end=end, avg_logprob=-0.2, no_speech_prob=None,
        words=[SimpleNamespace(word=" नमस्ते", start=0.25, end=0.4, probability=0.9),
               SimpleNamespace(word=" दुनिया", start=0.4, end=0.75, probability=None)])]


def signal(n=16000, dtype=np.float32):
    return np.zeros(n, dtype=dtype)


class ConfigurationTests(Fixture):
    def test_missing_device_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device=None, compute_type="int8", model_dir=self.model_dir).validate()

    def test_unknown_device_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="gpu", compute_type="float16", model_dir=self.model_dir).validate()

    def test_missing_compute_type_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type=None, model_dir=self.model_dir).validate()

    def test_cuda_with_int8_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cuda", compute_type="int8", model_dir=self.model_dir).validate()

    def test_cpu_with_float16_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type="float16", model_dir=self.model_dir).validate()

    def test_missing_model_dir_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type="int8", model_dir=None).validate()

    def test_beam_size_out_of_range_is_refused(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type="int8", model_dir=self.model_dir, beam_size=0).validate()

    def test_word_timestamps_cannot_be_disabled(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type="int8", model_dir=self.model_dir,
                                    word_timestamps=False).validate()

    def test_vad_filter_cannot_be_enabled(self):
        with self.assertRaises(ConfigError):
            fwe.FasterWhisperConfig(device="cpu", compute_type="int8", model_dir=self.model_dir,
                                    vad_filter=True).validate()

    def test_valid_cuda_and_cpu_pairings_are_accepted(self):
        fwe.FasterWhisperConfig(device="cuda", compute_type="float16", model_dir=self.model_dir).validate()
        fwe.FasterWhisperConfig(device="cpu", compute_type="int8", model_dir=self.model_dir).validate()

    def test_model_path_outside_the_storage_root_is_refused(self):
        outside = self.tmp / "elsewhere"
        outside.mkdir()
        with self.assertRaises(guard.ConfigPathError):
            self.engine(model_dir=outside)

    def test_nonexistent_model_directory_is_refused(self):
        with self.assertRaises(guard.ConfigPathError):
            self.engine(model_dir=self.models / "no-such-model")

    def test_path_traversal_out_of_the_root_is_refused(self):
        with self.assertRaises(guard.ConfigPathError):
            self.engine(model_dir=self.models / ".." / "elsewhere")

    def test_symbolic_link_model_directory_is_refused(self):
        link = self.models / "linked"
        try:
            os.symlink(self.model_dir, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symbolic links are not available on this system")
        with self.assertRaises(guard.ConfigPathError):
            self.engine(model_dir=link)

    def test_device_is_never_silently_changed(self):
        factory = FakeFactory(error=RuntimeError("cuda unavailable"))
        engine = self.engine(factory=factory, device="cuda", compute="float16")
        with self.assertRaises(RuntimeError):
            engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertEqual([c[1] for c in factory.calls], ["cuda"])  # never retried on cpu


class AdapterBehaviourTests(Fixture):
    def run_engine(self, language="en", model=None, **kw):
        factory = FakeFactory(model=model)
        engine = self.engine(factory=factory, **kw)
        result = engine.transcribe(signal(), 16000, TranscriptionRequest(language=language))
        return engine, factory, result

    def test_explicit_english_is_passed_through_and_probability_is_null(self):
        model = FakeModel(default_segments(), language="en", probability=0.97)
        _, _, result = self.run_engine("en", model=model)
        self.assertEqual(result.language, "en")
        self.assertIsNone(result.language_probability)
        self.assertEqual(model.calls[-1]["language"], "en")

    def test_explicit_hindi_is_passed_through_and_probability_is_null(self):
        model = FakeModel(default_segments(), language="hi", probability=0.8)
        _, _, result = self.run_engine("hi", model=model)
        self.assertEqual(result.language, "hi")
        self.assertIsNone(result.language_probability)
        self.assertEqual(model.calls[-1]["language"], "hi")

    def test_automatic_detection_maps_language_and_probability(self):
        model = FakeModel(default_segments(), language="hi", probability=0.88)
        _, _, result = self.run_engine("auto", model=model)
        self.assertEqual(result.language, "hi")
        self.assertEqual(result.language_probability, 0.88)
        self.assertIsNone(model.calls[-1]["language"])  # auto means no forced language

    def test_raw_text_is_preserved_exactly(self):
        _, _, result = self.run_engine("hi")
        self.assertEqual(result.text, " नमस्ते दुनिया")
        self.assertEqual(result.segments[0].text, " नमस्ते दुनिया")

    def test_segment_and_word_seconds_pass_through_unchanged(self):
        _, _, result = self.run_engine("hi")
        seg = result.segments[0]
        self.assertEqual((seg.start_s, seg.end_s), (0.25, 0.75))
        self.assertEqual((seg.words[0].start_s, seg.words[0].end_s), (0.25, 0.4))

    def test_engine_seconds_convert_through_the_single_h93a_conversion(self):
        _, _, result = self.run_engine("hi")
        self.assertEqual(seconds_to_sample(result.segments[0].start_s, 16000), 4000)
        validated = validate_engine_result(result, TranscriptionRequest(language="hi"), 1.0)
        self.assertIs(type(validated), type(result))

    def test_missing_confidence_values_stay_null(self):
        model = FakeModel([SimpleNamespace(text=" x", start=0.0, end=0.5, avg_logprob=None, no_speech_prob=None,
                                           words=[SimpleNamespace(word=" x", start=0.0, end=0.5, probability=None)])],
                          language="en", probability=None)
        _, _, result = self.run_engine("auto", model=model)
        self.assertIsNone(result.segments[0].avg_logprob)
        self.assertIsNone(result.segments[0].no_speech_prob)
        self.assertIsNone(result.segments[0].words[0].probability)
        self.assertIsNone(result.language_probability)

    def test_word_timestamps_are_requested_and_vad_is_off(self):
        model = FakeModel(default_segments(), language="en")
        self.run_engine("en", model=model)
        self.assertTrue(model.calls[-1]["word_timestamps"])
        self.assertFalse(model.calls[-1]["vad_filter"])

    def test_model_is_loaded_once_per_engine(self):
        factory = FakeFactory()
        engine = self.engine(factory=factory)
        for _ in range(2):
            engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertEqual(len(factory.calls), 1)

    def test_run_metadata_records_versions_device_and_manifest_hash(self):
        engine, _, _ = self.run_engine("en")
        meta = engine.last_run
        self.assertEqual(meta["faster_whisper_version"], "1.2.1")
        self.assertEqual(meta["ctranslate2_version"], "4.9.0")
        self.assertEqual(meta["device"], "cpu")
        self.assertEqual(meta["compute_type"], "int8")
        self.assertEqual(meta["model_revision"], REVISION)
        self.assertRegex(meta["model_manifest_sha256"], r"^[0-9a-f]{64}$")

    def test_unmeasured_values_are_null_not_invented(self):
        engine, _, _ = self.run_engine("en")
        meta = engine.last_run
        self.assertIsNone(meta["cuda_dll_paths_and_hashes"])
        self.assertIsNone(meta["peak_ram_mb"])
        self.assertIsNone(meta["peak_vram_mb"])

    def test_engine_info_reports_the_manifest_identity(self):
        engine = self.engine()
        info = engine.info
        self.assertEqual((info.engine, info.model_name, info.model_revision),
                         ("faster-whisper", "synthetic-large-v3", REVISION))

    def test_non_array_input_is_refused(self):
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe([0.0] * 16000, 16000, TranscriptionRequest(language="en"))

    def test_two_dimensional_input_is_refused(self):
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe(np.zeros((2, 8000), np.float32), 16000, TranscriptionRequest(language="en"))

    def test_wrong_dtype_is_refused(self):
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe(signal(dtype=np.float64), 16000, TranscriptionRequest(language="en"))

    def test_wrong_sample_rate_is_refused(self):
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe(signal(), 44100, TranscriptionRequest(language="en"))

    def test_empty_input_is_refused(self):
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe(signal(0), 16000, TranscriptionRequest(language="en"))

    def test_non_finite_input_is_refused(self):
        bad = signal()
        bad[10] = np.nan
        with self.assertRaises(fwe.InputError):
            self.engine().transcribe(bad, 16000, TranscriptionRequest(language="en"))

    def test_the_caller_signal_is_not_modified(self):
        samples = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        before = samples.copy()
        self.engine().transcribe(samples, 16000, TranscriptionRequest(language="en"))
        np.testing.assert_array_equal(samples, before)


class DecodingSettingsTests(Fixture):
    """H9.3b-1 corrections: the decoding settings recorded by H9.3a are the ones that run."""

    def run_with(self, options, factory=None):
        factory = factory or FakeFactory()
        engine = self.engine(factory=factory)
        request = TranscriptionRequest(language="en", options=options)
        return engine, factory, request

    def test_temperature_from_the_request_is_passed_to_the_model(self):
        from extraction.asr_interface import DecodingOptions
        model = FakeModel(default_segments(), language="en")
        engine, _, request = self.run_with(DecodingOptions(temperature=0.0), FakeFactory(model=model))
        engine.transcribe(signal(), 16000, request)
        self.assertEqual(model.calls[-1]["temperature"], 0.0)

    def test_beam_size_from_the_request_is_passed_to_the_model(self):
        from extraction.asr_interface import DecodingOptions
        model = FakeModel(default_segments(), language="en")
        engine, _, request = self.run_with(DecodingOptions(beam_size=5), FakeFactory(model=model))
        engine.transcribe(signal(), 16000, request)
        self.assertEqual(model.calls[-1]["beam_size"], 5)

    def test_a_beam_size_mismatch_is_refused_before_loading(self):
        from extraction.asr_interface import DecodingOptions
        engine, factory, request = self.run_with(DecodingOptions(beam_size=3))
        with self.assertRaises(ConfigError):
            engine.transcribe(signal(), 16000, request)
        self.assertEqual(factory.calls, [])

    def test_vad_filter_in_the_request_is_refused(self):
        from extraction.asr_interface import DecodingOptions
        engine, factory, request = self.run_with(DecodingOptions(vad_filter=True))
        with self.assertRaises(ConfigError):
            engine.transcribe(signal(), 16000, request)
        self.assertEqual(factory.calls, [])

    def test_word_timestamps_off_in_the_request_is_refused(self):
        from extraction.asr_interface import DecodingOptions
        engine, factory, request = self.run_with(DecodingOptions(word_timestamps=False))
        with self.assertRaises(ConfigError):
            engine.transcribe(signal(), 16000, request)
        self.assertEqual(factory.calls, [])


class ModelIntegrityTests(Fixture):
    def expect_refused(self, engine_kwargs=None, model_dir=None, exc=guard.ModelIntegrityError):
        factory = FakeFactory()
        with self.assertRaises(exc):
            engine = self.engine(factory=factory, model_dir=model_dir, **(engine_kwargs or {}))
            engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertEqual(factory.calls, [], "the model was loaded despite a failed check")

    def test_a_valid_synthetic_manifest_allows_transcription(self):
        factory = FakeFactory()
        engine = self.engine(factory=factory)
        engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertEqual(len(factory.calls), 1)

    def test_missing_manifest_is_refused_before_loading(self):
        (self.model_dir / guard.MANIFEST_NAME).unlink()
        with self.assertRaises(guard.ModelIntegrityError):
            self.engine()

    def test_unsupported_manifest_version_is_refused(self):
        self.model_dir = self.make_model("v2", extra=lambda m: {**m, "manifest_version": 2})
        self.expect_refused(model_dir=self.model_dir)

    def test_bad_revision_is_refused(self):
        self.model_dir = self.make_model("badrev", extra=lambda m: {**m, "revision": "main"})
        with self.assertRaises(guard.ModelIntegrityError):
            self.engine(model_dir=self.model_dir)

    def test_unknown_licence_status_is_refused(self):
        self.model_dir = self.make_model("badlicence", extra=lambda m: {**m, "licence_status": "probably fine"})
        with self.assertRaises(guard.ModelIntegrityError):
            self.engine(model_dir=self.model_dir)

    def test_path_traversal_in_a_manifest_file_name_is_refused(self):
        self.model_dir = self.make_model(
            "traversal", extra=lambda m: {**m, "files": [{"name": "../escape.bin", "size": 1, "sha256": "0" * 64}]})
        with self.assertRaises(guard.ModelIntegrityError):
            self.engine(model_dir=self.model_dir)

    def test_bad_hash_format_in_manifest_is_refused(self):
        self.model_dir = self.make_model(
            "badhash", extra=lambda m: {**m, "files": [{"name": "model.bin", "size": 1, "sha256": "XYZ"}]})
        with self.assertRaises(guard.ModelIntegrityError):
            self.engine(model_dir=self.model_dir)

    def test_missing_model_file_is_refused_before_loading(self):
        (self.model_dir / "model.bin").unlink()
        self.expect_refused()

    def test_modified_file_with_the_same_size_is_refused(self):
        original = (self.model_dir / "model.bin").read_bytes()
        (self.model_dir / "model.bin").write_bytes(b"X" * len(original))
        self.expect_refused()

    def test_wrong_size_is_refused(self):
        (self.model_dir / "model.bin").write_bytes(b"short")
        self.expect_refused()

    def test_wrong_sha256_in_the_manifest_is_refused(self):
        self.model_dir = self.make_model(
            "wronghash", files={"model.bin": b"abc"},
            extra=lambda m: {**m, "files": [{"name": "model.bin", "size": 3, "sha256": "0" * 64}]})
        self.expect_refused(model_dir=self.model_dir)

    def test_an_undeclared_file_in_the_model_directory_is_refused(self):
        (self.model_dir / "extra.bin").write_bytes(b"not declared")
        self.expect_refused()

    def test_licence_status_not_cleared_is_refused(self):
        self.model_dir = self.make_model("notcleared", licence="not cleared")
        self.expect_refused(model_dir=self.model_dir)

    def test_licence_status_under_review_is_refused(self):
        self.model_dir = self.make_model("review", licence="under review")
        self.expect_refused(model_dir=self.model_dir)

    def test_wrong_ctranslate2_version_is_refused_before_loading(self):
        def wrong(name):
            return "0.0.1" if name == "ctranslate2" else PINS[name]
        self.expect_refused(engine_kwargs={"versions": wrong}, exc=fwe.ModelLoadError)

    def test_missing_faster_whisper_is_refused_before_loading(self):
        import importlib.metadata as md
        factory = FakeFactory()
        with mock.patch.object(fwe.importlib.metadata, "version",
                               side_effect=md.PackageNotFoundError("faster-whisper")):
            with self.assertRaises(fwe.ModelLoadError):
                self.engine(factory=factory, versions=None).transcribe(
                    signal(), 16000, TranscriptionRequest(language="en"))
        self.assertEqual(factory.calls, [])


class GpuLockTests(Fixture):
    def setUp(self):
        super().setUp()
        self.lock_path = self.runtime / "gpu.lock"

    def dead_pid(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        return proc.pid

    def write_lock(self, record):
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(json.dumps(record), encoding="utf-8")

    def test_successful_acquisition_writes_a_record(self):
        lock = guard.GpuLock(self.lock_path)
        with lock.held():
            record = json.loads(self.lock_path.read_text(encoding="utf-8"))
            self.assertEqual(record["pid"], os.getpid())
            self.assertEqual(record["host"], socket.gethostname())
        self.assertFalse(self.lock_path.exists())

    def test_a_held_lock_refuses_a_second_acquisition(self):
        first = guard.GpuLock(self.lock_path)
        second = guard.GpuLock(self.lock_path)
        first.acquire()
        try:
            with self.assertRaises(guard.GpuLockError) as ctx:
                second.acquire()
            self.assertIn("already running", str(ctx.exception))
        finally:
            first.release()

    def test_release_happens_after_an_exception(self):
        lock = guard.GpuLock(self.lock_path)
        with self.assertRaises(RuntimeError):
            with lock.held():
                raise RuntimeError("inference failed")
        self.assertFalse(self.lock_path.exists())
        with lock.held():
            pass

    def test_release_never_removes_another_holders_lock(self):
        lock = guard.GpuLock(self.lock_path)
        lock.acquire()
        self.write_lock({"pid": 1, "host": "x", "token": "someone-else"})
        lock.release()
        self.assertEqual(json.loads(self.lock_path.read_text(encoding="utf-8"))["token"], "someone-else")

    def test_a_stale_lock_is_never_removed_automatically(self):
        self.write_lock({"pid": self.dead_pid(), "host": socket.gethostname(), "token": "old"})
        lock = guard.GpuLock(self.lock_path)
        with self.assertRaises(guard.GpuLockError) as ctx:
            lock.acquire()
        self.assertIn("recover_stale_gpu_lock", str(ctx.exception))
        self.assertTrue(self.lock_path.exists())

    def test_explicit_recovery_clears_a_verified_stale_lock(self):
        self.write_lock({"pid": self.dead_pid(), "host": socket.gethostname(), "token": "old"})
        owner = guard.recover_stale_gpu_lock(self.lock_path)
        self.assertEqual(owner["token"], "old")
        self.assertFalse(self.lock_path.exists())

    def test_recovery_refuses_a_live_owner(self):
        self.write_lock({"pid": os.getpid(), "host": socket.gethostname(), "token": "live"})
        with self.assertRaises(guard.GpuLockError):
            guard.recover_stale_gpu_lock(self.lock_path)
        self.assertTrue(self.lock_path.exists())

    def test_recovery_refuses_a_lock_from_another_machine(self):
        self.write_lock({"pid": self.dead_pid(), "host": "other-machine", "token": "x"})
        with self.assertRaises(guard.GpuLockError):
            guard.recover_stale_gpu_lock(self.lock_path)
        self.assertTrue(self.lock_path.exists())

    def test_recovery_refuses_an_unreadable_lock(self):
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_bytes(b"\x00 half written")
        with self.assertRaises(guard.GpuLockError):
            guard.recover_stale_gpu_lock(self.lock_path)
        self.assertTrue(self.lock_path.exists())

    def test_recovery_reports_when_there_is_no_lock(self):
        with self.assertRaises(guard.GpuLockError):
            guard.recover_stale_gpu_lock(self.lock_path)

    def test_concurrent_threads_admit_exactly_one_holder(self):
        barrier = threading.Barrier(8)
        outcomes = []
        guard_lock = threading.Lock()

        def attempt():
            lock = guard.GpuLock(self.lock_path)
            barrier.wait()
            try:
                with lock.held():
                    threading.Event().wait(0.2)
                result = "held"
            except guard.GpuLockError:
                result = "refused"
            with guard_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(outcomes.count("held"), 1, outcomes)
        self.assertEqual(outcomes.count("refused"), 7, outcomes)
        self.assertFalse(self.lock_path.exists())

    def test_cuda_inference_is_refused_while_the_gpu_lock_is_held(self):
        blocker = guard.GpuLock(self.lock_path)
        blocker.acquire()
        try:
            factory = FakeFactory()
            engine = self.engine(factory=factory, device="cuda", compute="float16",
                                 lock=guard.GpuLock(self.lock_path))
            with self.assertRaises(guard.GpuLockError):
                engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
            self.assertEqual(factory.calls, [])
        finally:
            blocker.release()

    def test_cpu_inference_does_not_use_the_gpu_lock(self):
        blocker = guard.GpuLock(self.lock_path)
        blocker.acquire()
        try:
            engine = self.engine(device="cpu", compute="int8", lock=guard.GpuLock(self.lock_path))
            engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
            self.assertFalse(engine.last_run["gpu_lock_used"])
        finally:
            blocker.release()

    def test_cuda_run_records_that_the_gpu_lock_was_used(self):
        engine = self.engine(device="cuda", compute="float16", lock=guard.GpuLock(self.lock_path))
        engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertTrue(engine.last_run["gpu_lock_used"])
        self.assertFalse(self.lock_path.exists())

    def test_gpu_lock_is_released_when_cuda_inference_fails(self):
        engine = self.engine(factory=FakeFactory(error=RuntimeError("cuda crash")),
                             device="cuda", compute="float16", lock=guard.GpuLock(self.lock_path))
        with self.assertRaises(RuntimeError):
            engine.transcribe(signal(), 16000, TranscriptionRequest(language="en"))
        self.assertFalse(self.lock_path.exists())


class ModuleBoundaryTests(unittest.TestCase):
    def test_importing_the_adapter_loads_no_runtime(self):
        for name in ("faster_whisper", "ctranslate2"):
            self.assertNotIn(name, sys.modules, f"{name} was imported by the adapter import")


if __name__ == "__main__":
    unittest.main()
