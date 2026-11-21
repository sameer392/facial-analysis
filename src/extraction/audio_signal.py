"""Pure signal functions for audio features (H9.2). NumPy only; no file access.

Frame grid
- Frames are windows of window_ms, advanced by hop_ms. Both are converted to whole samples at the
  analysis rate (round half up), and every frame starts at sample i * hop_samples. Frame times are
  computed from sample indexes with integer arithmetic, so there is no cumulative drift.
- A frame is emitted only when its whole window lies inside the signal. The samples left over at
  the end are reported as a count; they are not padded.

Energy
- rms: root mean square of the window, linear (full scale = 1.0).
- energy_db: 10 * log10 of the mean square, in dBFS. A window with zero energy is given the floor
  value SILENCE_FLOOR_DB and flagged silent; it is not reported as a measured level.

Pitch
- Normalised cross-correlation over lags covering [pitch_min_hz, pitch_max_hz]. The best lag gives
  F0 = analysis_rate / lag, with integer-sample resolution. A frame is voiced only when the peak
  correlation reaches pitch_threshold and the frame is not silent. Other frames have NaN F0.
- Known limits: the estimator can pick a sub-octave or octave error, and it has not been validated
  on real speech. It is isolated here so that it can be replaced.

VAD (baseline, energy only)
- A frame is speech when its energy_db is at or above vad_db. Runs of speech are then merged
  across gaps up to vad_merge_gap_ms, speech runs shorter than vad_min_speech_ms are dropped, and
  silence runs of at least vad_min_silence_ms are reported. Durations are measured in hops.
- This is not a ground-truth speech annotation and has not been evaluated against one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SILENCE_FLOOR_DB = -120.0
RATE_MIN = 8000
RATE_MAX = 48000


class SignalError(ValueError):
    """Invalid configuration or input to a signal function."""


@dataclass(frozen=True)
class AudioConfig:
    analysis_rate: int = 16000
    window_ms: float = 40.0
    hop_ms: float = 10.0
    pitch_min_hz: float = 60.0
    pitch_max_hz: float = 400.0
    pitch_threshold: float = 0.6
    silence_db: float = -50.0
    vad_db: float = -40.0
    vad_min_speech_ms: float = 100.0
    vad_min_silence_ms: float = 200.0
    vad_merge_gap_ms: float = 200.0

    def validate(self) -> "AudioConfig":
        if not isinstance(self.analysis_rate, int) or not RATE_MIN <= self.analysis_rate <= RATE_MAX:
            raise SignalError(f"analysis_rate must be an integer from {RATE_MIN} to {RATE_MAX}")
        for name in ("window_ms", "hop_ms", "pitch_min_hz", "pitch_max_hz"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise SignalError(f"{name} must be a positive number")
        if self.hop_samples < 1:
            raise SignalError("hop_ms is shorter than one sample at the analysis rate")
        if self.pitch_min_hz >= self.pitch_max_hz:
            raise SignalError("pitch_min_hz must be below pitch_max_hz")
        if self.pitch_max_hz >= self.analysis_rate / 4:
            raise SignalError("pitch_max_hz must be below a quarter of the analysis rate")
        if not 0 < self.pitch_threshold <= 1:
            raise SignalError("pitch_threshold must be greater than 0 and at most 1")
        if not (SILENCE_FLOOR_DB < self.silence_db < 0 and SILENCE_FLOOR_DB < self.vad_db < 0):
            raise SignalError("dB thresholds must lie between the silence floor and 0")
        for name in ("vad_min_speech_ms", "vad_min_silence_ms", "vad_merge_gap_ms"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise SignalError(f"{name} must be zero or positive")
        if self.window_samples < 2 * self.lag_max:
            raise SignalError("window_ms is too short for pitch_min_hz (the window must hold two periods)")
        return self

    @property
    def window_samples(self) -> int:
        return max(1, round(self.window_ms * self.analysis_rate / 1000))

    @property
    def hop_samples(self) -> int:
        return max(0, round(self.hop_ms * self.analysis_rate / 1000))

    @property
    def lag_min(self) -> int:
        return max(2, int(np.ceil(self.analysis_rate / self.pitch_max_hz)))

    @property
    def lag_max(self) -> int:
        return int(np.floor(self.analysis_rate / self.pitch_min_hz))

    def as_dict(self) -> dict:
        data = asdict(self)
        data.update(window_samples=self.window_samples, hop_samples=self.hop_samples,
                    lag_min=self.lag_min, lag_max=self.lag_max)
        return data


# ---------------------------------------------------------------- sample-index timing


def sample_time_us(sample_index, rate: int, base_us: int):
    """Session time in integer microseconds of analysis sample(s) sample_index.

    base_us is the session time of sample 0. The conversion is integer arithmetic with rounding half
    up, so it does not accumulate error over long recordings.
    """
    idx = np.asarray(sample_index, dtype=np.int64)
    scaled = (2 * idx * 1_000_000 + rate) // (2 * rate)
    return base_us + scaled


def frame_grid(n_samples: int, cfg: AudioConfig) -> tuple[int, int]:
    """(number of complete frames, samples left over after the last complete frame)."""
    w, hop = cfg.window_samples, cfg.hop_samples
    if n_samples < w:
        return 0, n_samples
    frames = (n_samples - w) // hop + 1
    used = (frames - 1) * hop + w
    return frames, n_samples - used


# ---------------------------------------------------------------- energy


def energy(frames: np.ndarray, cfg: AudioConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rms, energy_db and silent flags for a (n, window) block of float samples."""
    x = np.asarray(frames, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SignalError("non-finite samples in the audio")
    mean_square = np.mean(x * x, axis=1)
    floor_power = 10.0 ** (SILENCE_FLOOR_DB / 10.0)
    power = np.maximum(mean_square, floor_power)
    energy_db = 10.0 * np.log10(power)
    rms = np.sqrt(mean_square)
    silent = energy_db < cfg.silence_db
    return rms.astype(np.float64), energy_db.astype(np.float64), silent


# ---------------------------------------------------------------- pitch


def pitch(frames: np.ndarray, cfg: AudioConfig, silent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """F0 (NaN where not voiced) and the normalised correlation peak for each frame."""
    x = np.asarray(frames, dtype=np.float64)
    n = x.shape[0]
    lag_min, lag_max = cfg.lag_min, cfg.lag_max
    span = cfg.window_samples - lag_max  # correlation length, the same for every lag
    head = x[:, :span]
    e0 = np.sum(head * head, axis=1)
    lags = np.arange(lag_min, lag_max + 1)
    r = np.empty((n, lags.size), dtype=np.float64)
    for k, lag in enumerate(lags):
        shifted = x[:, lag:lag + span]
        el = np.sum(shifted * shifted, axis=1)
        num = np.sum(head * shifted, axis=1)
        r[:, k] = num / (np.sqrt(e0 * el) + 1e-12)
    peak = np.max(r, axis=1)
    # Candidates are local maxima of the correlation curve. The first candidate within 90% of the
    # best peak is taken, so a multiple of the true period (which scores almost as high) is not chosen.
    # Points on the slope of the curve are not peaks and are not candidates.
    local = np.zeros_like(r, dtype=bool)
    if lags.size >= 3:
        local[:, 1:-1] = (r[:, 1:-1] >= r[:, :-2]) & (r[:, 1:-1] >= r[:, 2:])
    local[:, 0] = r[:, 0] >= r[:, 1] if lags.size > 1 else True
    local[:, -1] = r[:, -1] >= r[:, -2] if lags.size > 1 else True
    candidates = local & (r >= (0.9 * peak)[:, None])
    has_candidate = candidates.any(axis=1)
    chosen = np.where(has_candidate, np.argmax(candidates, axis=1), np.argmax(r, axis=1))
    # Parabolic interpolation around the chosen lag gives a sub-sample period estimate.
    k = np.clip(chosen, 1, lags.size - 2) if lags.size >= 3 else chosen
    rows = np.arange(n)
    if lags.size >= 3:
        a = r[rows, k - 1]
        b = r[rows, k]
        c = r[rows, k + 1]
        denom = a - 2 * b + c
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1.0, denom), 0.0)
        shift = np.clip(shift, -0.5, 0.5)
        lag_est = lags[k] + np.where((chosen == k), shift, 0.0)
    else:
        lag_est = lags[chosen].astype(np.float64)
    conf = np.clip(peak, -1.0, 1.0)
    voiced = (conf >= cfg.pitch_threshold) & (~silent)
    f0 = np.where(voiced, cfg.analysis_rate / np.maximum(lag_est, 1e-9), np.nan)
    return f0.astype(np.float64), conf.astype(np.float64)


# ---------------------------------------------------------------- VAD


def vad_frames(energy_db: np.ndarray, vad_db: float) -> np.ndarray:
    return np.asarray(energy_db) >= vad_db


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True values."""
    if flags.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flags.astype(np.int8), [0]))))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]


def vad_segments(speech: np.ndarray, cfg: AudioConfig) -> list[dict]:
    """Speech and silence segments in frame indexes, after gap merging and minimum durations."""
    hop_ms = cfg.hop_ms
    runs = _runs(speech)
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) * hop_ms <= cfg.vad_merge_gap_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    speech_runs = [(s, e) for s, e in merged if (e - s) * hop_ms >= cfg.vad_min_speech_ms]

    segments: list[dict] = []
    cursor = 0
    for s, e in speech_runs:
        if s > cursor and (s - cursor) * hop_ms >= cfg.vad_min_silence_ms:
            segments.append({"kind": "silence", "start_frame": cursor, "end_frame": s})
        segments.append({"kind": "speech", "start_frame": s, "end_frame": e})
        cursor = e
    total = speech.shape[0]
    if total > cursor and (total - cursor) * hop_ms >= cfg.vad_min_silence_ms:
        segments.append({"kind": "silence", "start_frame": cursor, "end_frame": total})
    return segments
