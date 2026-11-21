"""faster-whisper adapter for the H9.3a ASR interface (phase H9.3b-3).

Implements AsrEngine. It produces an EngineResult, and the existing transcription layer validates and
publishes it. The adapter does no timestamp arithmetic of its own: float seconds from faster-whisper go
into EngineResult unchanged, and seconds_to_sample (asr_interface) is the only conversion.

Safety rules
- device is mandatory: "cuda" or "cpu". There is no silent fallback from CUDA to CPU.
- compute_type is fixed per device: "float16" on CUDA, "int8" on CPU. Any other pairing is refused.
- The model directory must resolve inside C:\\AI\\asr-models\\ and must pass manifest verification
  (asr_runtime_guard.verify_model_dir) before the model is loaded. A licence status other than
  "cleared" is refused.
- The installed faster-whisper and CTranslate2 versions must equal the approved pins.
- CUDA inference holds the GPU lock for the whole call, so only one GPU job runs at a time.
- Nothing is imported or loaded at module import time. The ASR runtime is imported only when a
  transcription is actually requested, and tests inject a fake model factory.

Explicit language requests store language_probability as None: the engine reports a probability even
when the language was forced, which is not a detection result.
"""

from __future__ import annotations

import importlib.metadata
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction.asr_interface import (  # noqa: E402
    AUTO,
    AsrEngine,
    ConfigError,
    EngineInfo,
    EngineResult,
    RawSegment,
    RawWord,
    TranscriptionRequest,
)
from extraction.asr_runtime_guard import (  # noqa: E402
    ASR_MODELS_ROOT,
    ASR_RUNTIME_ROOT,
    GpuLock,
    load_manifest,
    resolve_inside,
    verify_model_dir,
)

ENGINE_NAME = "faster-whisper"
FASTER_WHISPER_PIN = "1.2.1"
CTRANSLATE2_PIN = "4.9.0"
DEVICES = {"cuda": "float16", "cpu": "int8"}  # the only permitted device and compute-type pairs
SAMPLE_RATE = 16000
REQUIRED_SAMPLE_DTYPE = np.float32


class InputError(ValueError):
    """The audio given to the engine is not in the required format."""


class ModelLoadError(RuntimeError):
    """The runtime or model cannot be loaded. Nothing is transcribed."""


@dataclass(frozen=True)
class FasterWhisperConfig:
    device: Optional[str]
    compute_type: Optional[str]
    model_dir: Optional[Path]
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = False

    def validate(self) -> "FasterWhisperConfig":
        if self.device is None:
            raise ConfigError("device is required: 'cuda' or 'cpu'. There is no default.")
        if self.device not in DEVICES:
            raise ConfigError(f"unknown device {self.device!r}; use 'cuda' or 'cpu'")
        if self.compute_type is None:
            raise ConfigError(f"compute_type is required; {self.device} requires {DEVICES[self.device]!r}")
        if self.compute_type != DEVICES[self.device]:
            raise ConfigError(
                f"compute_type {self.compute_type!r} is not permitted on {self.device}; "
                f"use {DEVICES[self.device]!r}")
        if self.model_dir is None:
            raise ConfigError("model_dir is required")
        if not isinstance(self.beam_size, int) or not 1 <= self.beam_size <= 20:
            raise ConfigError("beam_size must be an integer from 1 to 20")
        if self.word_timestamps is not True:
            raise ConfigError("word_timestamps must be true: the dataset requires word-level timing")
        if self.vad_filter is not False:
            raise ConfigError("vad_filter must be false in this phase")
        return self


def _installed_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelLoadError(f"{name} is not installed in this environment") from exc


def _default_model_factory(model_path: Path, device: str, compute_type: str):
    # Imported here, not at module level, so that importing this module never requires the runtime.
    from faster_whisper import WhisperModel  # noqa: WPS433 (deliberate lazy import)

    return WhisperModel(str(model_path), device=device, compute_type=compute_type, local_files_only=True)


class FasterWhisperEngine(AsrEngine):
    """faster-whisper behind the H9.3a interface. Model loading is lazy and verified."""

    def __init__(
        self,
        config: FasterWhisperConfig,
        *,
        models_root=ASR_MODELS_ROOT,
        runtime_root=ASR_RUNTIME_ROOT,
        gpu_lock: GpuLock | None = None,
        model_factory: Callable | None = None,
        version_provider: Callable[[str], str] | None = None,
    ):
        self.config = config.validate()
        self.models_root = Path(models_root)
        self.model_dir = resolve_inside(self.config.model_dir, self.models_root)
        self._gpu_lock = gpu_lock if gpu_lock is not None else GpuLock(Path(runtime_root) / "gpu.lock")
        self._model_factory = model_factory or _default_model_factory
        self._version_provider = version_provider or _installed_version
        self._model = None
        self._verified: dict | None = None
        self.last_run: dict | None = None
        manifest, _sha = load_manifest(self.model_dir)  # schema only; no hashing, no loading
        self._declared = manifest

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            engine=ENGINE_NAME,
            version=FASTER_WHISPER_PIN,
            model_name=self._declared["model_name"],
            model_revision=self._declared["revision"],
            model_sha256=self._verified["manifest_sha256"] if self._verified else None,
        )

    # ---------------------------------------------------------- input and runtime checks

    @staticmethod
    def check_signal(samples, sample_rate) -> np.ndarray:
        if not isinstance(samples, np.ndarray):
            raise InputError("the audio must be a NumPy array")
        if samples.ndim != 1:
            raise InputError("the audio must be mono (one dimension)")
        if samples.dtype != REQUIRED_SAMPLE_DTYPE:
            raise InputError("the audio must be float32")
        if sample_rate != SAMPLE_RATE:
            raise InputError(f"the audio must be {SAMPLE_RATE} Hz")
        if samples.size == 0:
            raise InputError("the audio is empty")
        if not np.all(np.isfinite(samples)):
            raise InputError("the audio contains non-finite values")
        return samples

    def _check_versions(self) -> None:
        for name, pin in ((ENGINE_NAME, FASTER_WHISPER_PIN), ("ctranslate2", CTRANSLATE2_PIN)):
            found = self._version_provider(name)
            if found != pin:
                raise ModelLoadError(f"{name} {found} is installed; the approved pin is {pin}")

    def _load_model(self):
        if self._model is None:
            self._verified = verify_model_dir(self.model_dir, root=self.models_root, require_cleared=True)
            self._check_versions()
            self._model = self._model_factory(self.model_dir, self.config.device, self.config.compute_type)
        return self._model

    # ---------------------------------------------------------- transcription

    def transcribe(self, samples: np.ndarray, sample_rate: int, request: TranscriptionRequest) -> EngineResult:
        samples = self.check_signal(samples, sample_rate)
        if self.config.device == "cuda":
            with self._gpu_lock.held():
                return self._run(samples, request)
        return self._run(samples, request)

    def _check_decoding(self, request: TranscriptionRequest) -> None:
        """The decoding settings recorded by H9.3a must be the ones that run. Mismatches are refused."""
        options = request.options
        if options.beam_size != self.config.beam_size:
            raise ConfigError(
                f"beam_size {options.beam_size} in the request does not match the adapter's {self.config.beam_size}")
        if options.vad_filter is not False:
            raise ConfigError("vad_filter must be false in this phase")
        if options.word_timestamps is not True:
            raise ConfigError("word_timestamps must be true: the dataset requires word-level timing")

    def _run(self, samples: np.ndarray, request: TranscriptionRequest) -> EngineResult:
        self._check_decoding(request)
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.monotonic()
        model = self._load_model()
        language = None if request.language == AUTO else request.language
        raw_segments, run_info = model.transcribe(
            samples,
            language=language,
            beam_size=request.options.beam_size,
            temperature=request.options.temperature,  # a single value: disables faster-whisper's fallback list
            word_timestamps=True,
            vad_filter=False,
        )
        segments = []
        texts = []
        for seg in raw_segments:
            words = tuple(
                RawWord(word=w.word, start_s=float(w.start), end_s=float(w.end),
                        probability=None if w.probability is None else float(w.probability))
                for w in (seg.words or ())
            )
            texts.append(seg.text)
            segments.append(RawSegment(
                text=seg.text,
                start_s=float(seg.start),
                end_s=float(seg.end),
                words=words,
                avg_logprob=None if seg.avg_logprob is None else float(seg.avg_logprob),
                no_speech_prob=None if seg.no_speech_prob is None else float(seg.no_speech_prob),
            ))
        if request.language == AUTO:
            detected = run_info.language
            probability = None if run_info.language_probability is None else float(run_info.language_probability)
        else:
            detected, probability = request.language, None  # forced language: no detection result
        result = EngineResult(
            text="".join(texts),  # raw engine text, unchanged
            language=detected,
            language_probability=probability,
            segments=tuple(segments),
            info=self.info,
        )
        self.last_run = {
            "engine": ENGINE_NAME,
            "faster_whisper_version": FASTER_WHISPER_PIN,
            "ctranslate2_version": CTRANSLATE2_PIN,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "model_revision": self._declared["revision"],
            "model_manifest_sha256": self._verified["manifest_sha256"] if self._verified else None,
            "cuda_dll_paths_and_hashes": None,      # not measured in H9.3b-3
            "peak_ram_mb": None,                    # not measured in H9.3b-3
            "peak_vram_mb": None,                   # not measured in H9.3b-3
            "gpu_lock_used": self.config.device == "cuda",
            "started_utc": started,
            "ended_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }
        return result
