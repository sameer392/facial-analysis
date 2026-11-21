"""ASR adapter interface and engine-response validation (phase H9.3a).

Engine-independent. Nothing here imports an ASR runtime, and no model is loaded. An engine adapter
implements AsrEngine and returns an EngineResult. Its output is checked by validate_engine_result
before any dataset file is written.

Contract
- Engines report times in seconds, as floats, relative to the start of the analysis signal they were
  given. These floats are kept unchanged as raw values. They are converted once, by
  seconds_to_sample, to integer analysis-sample indexes, which are the authoritative reference.
- Confidence and probability values are optional. A missing value is None. No value is invented.
- Language: an explicit request must be honoured exactly. An automatic request must return a supported
  language code. Supported codes are "en" and "hi" for this phase.
- Engines must not translate. The text returned is the text kept as the raw transcript.

A response that breaks any rule is refused with EngineResponseError. Times are never clamped or
silently repaired.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

SUPPORTED_LANGUAGES = ("en", "hi")
AUTO = "auto"
LANGUAGE_REQUESTS = (AUTO,) + SUPPORTED_LANGUAGES


class EngineResponseError(ValueError):
    """An engine returned a response that violates the contract."""


class ConfigError(ValueError):
    """An invalid transcription request or option."""


@dataclass(frozen=True)
class EngineInfo:
    engine: str
    version: str
    model_name: str
    model_revision: Optional[str] = None
    model_sha256: Optional[str] = None

    def __post_init__(self):
        for name in ("engine", "version", "model_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{name} must be a non-empty string")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DecodingOptions:
    beam_size: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_filter: bool = False

    def __post_init__(self):
        if not isinstance(self.beam_size, int) or not 1 <= self.beam_size <= 20:
            raise ConfigError("beam_size must be an integer from 1 to 20")
        if not math.isfinite(self.temperature) or not 0.0 <= self.temperature <= 1.0:
            raise ConfigError("temperature must be between 0 and 1")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptionRequest:
    language: str = AUTO
    options: DecodingOptions = field(default_factory=DecodingOptions)

    def __post_init__(self):
        if self.language not in LANGUAGE_REQUESTS:
            raise ConfigError(f"language must be one of {', '.join(LANGUAGE_REQUESTS)}")

    @property
    def explicit(self) -> bool:
        return self.language != AUTO


@dataclass(frozen=True)
class RawWord:
    word: str
    start_s: float
    end_s: float
    probability: Optional[float] = None


@dataclass(frozen=True)
class RawSegment:
    text: str
    start_s: float
    end_s: float
    words: tuple = ()
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass(frozen=True)
class EngineResult:
    text: str
    language: str
    language_probability: Optional[float]
    segments: tuple
    info: EngineInfo


class AsrEngine(ABC):
    """Adapter boundary. Implementations wrap one ASR runtime and nothing else."""

    @property
    @abstractmethod
    def info(self) -> EngineInfo:
        """Engine name, version and model identity. Must not load a model to answer."""

    @abstractmethod
    def transcribe(self, samples: np.ndarray, sample_rate: int, request: TranscriptionRequest) -> EngineResult:
        """Transcribe mono float32 samples. Return raw output; do not clean or translate it."""


def seconds_to_sample(seconds: float, sample_rate: int) -> int:
    """The one place where an engine's float time becomes an integer sample index (round half up)."""
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise EngineResponseError("a time is not a finite number")
    if seconds < 0:
        raise EngineResponseError("a time is negative")
    return int(math.floor(seconds * sample_rate + 0.5))


def _check_probability(value, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise EngineResponseError(f"{label} must be null or a number from 0 to 1")


def validate_engine_result(result: EngineResult, request: TranscriptionRequest, duration_s: float) -> EngineResult:
    """Refuse any response that breaks the contract. Returns the result unchanged when it is valid."""
    if not isinstance(result, EngineResult):
        raise EngineResponseError("the engine did not return an EngineResult")
    if not isinstance(result.text, str):
        raise EngineResponseError("the transcript text is not a string")
    if result.language not in SUPPORTED_LANGUAGES:
        raise EngineResponseError(f"unsupported language code {result.language!r}")
    if request.explicit and result.language != request.language:
        raise EngineResponseError(
            f"explicit language {request.language!r} was requested but the engine returned {result.language!r}")
    _check_probability(result.language_probability, "language_probability")
    if not isinstance(result.info, EngineInfo):
        raise EngineResponseError("engine information is missing")

    previous_end = None
    for s_index, seg in enumerate(result.segments):
        if not isinstance(seg.text, str):
            raise EngineResponseError(f"segment {s_index} text is not a string")
        # Finite and non-negative, or refused; the sample index itself is computed later.
        seconds_to_sample(seg.start_s, 1)
        seconds_to_sample(seg.end_s, 1)
        if seg.end_s < seg.start_s:
            raise EngineResponseError(f"segment {s_index} ends before it starts")
        if seg.end_s > duration_s + 1e-9:
            raise EngineResponseError(f"segment {s_index} ends after the audio")
        if previous_end is not None and seg.start_s < previous_end:
            raise EngineResponseError(f"segment {s_index} starts before the previous segment ends")
        previous_end = seg.end_s
        if seg.avg_logprob is not None and not (isinstance(seg.avg_logprob, (int, float))
                                                and math.isfinite(seg.avg_logprob) and seg.avg_logprob <= 0):
            raise EngineResponseError("avg_logprob must be null or a finite number not above 0")
        _check_probability(seg.no_speech_prob, "no_speech_prob")
        prev_word_start = None
        for w_index, w in enumerate(seg.words):
            if not isinstance(w.word, str):
                raise EngineResponseError(f"word {w_index} of segment {s_index} is not a string")
            if w.end_s < w.start_s:
                raise EngineResponseError(f"word {w_index} of segment {s_index} ends before it starts")
            if w.start_s < seg.start_s or w.end_s > seg.end_s:
                raise EngineResponseError(f"word {w_index} of segment {s_index} lies outside its segment")
            if prev_word_start is not None and w.start_s < prev_word_start:
                raise EngineResponseError(f"words of segment {s_index} are not in time order")
            prev_word_start = w.start_s
            _check_probability(w.probability, "word probability")
    return result
