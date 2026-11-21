"""Audio preprocessing, timing and basic features for one session (H9.2).

Input: the H9.1 acquisition manifest at audio/manifest.json, and the source it describes.
Output: audio/features/, audio/vad/, audio/quality/ and audio/processing_manifest.json.

Preprocessing
- The source is verified against the H9.1 SHA-256 before anything is decoded. An embedded
  recording is read from recordings/raw/. An external file is read from its retained copy when the
  H9.1 retention policy kept one; otherwise the same file must be supplied again.
- FFmpeg decodes the selected stream and produces mono 32-bit float PCM at the analysis rate in a
  temporary folder, which is deleted on success and on failure:
    * downmix: equal-weight mean of all channels, written as an explicit pan expression;
      for mono sources there is no downmix. The LFE channel, if any, is included in the mean.
    * resampling: FFmpeg's swresample default resampler (windowed sinc). The FFmpeg version is
      recorded in the output metadata.
- The resampled signal is memory-mapped. Only per-frame arrays are held in memory.

Timing
- Analysis sample j has session time  base_us + round(j * 1e6 / analysis_rate), where
  base_us = audio_start_us + external_offset_us - origin_us, taken from the H9.1 manifest.
  Rounding is integer arithmetic, so it does not drift.
- Frame i covers samples [i * hop, i * hop + window). Its reference time is the window centre,
  sample i * hop + window // 2. start_sample, centre_sample and t_us are all written.

Publication
- All outputs are written to a staging folder inside audio/, validated, and then moved into place.
  processing_manifest.json is moved last. A regeneration keeps the previous outputs until the new
  ones are in place, and restores them if publication fails.
- The H9.1 acquisition lock is held for the whole run, so acquisition and processing cannot overlap.
- H9.1 outputs (manifest.json, source/) are read only.

Outputs (schema "audio-features-1")
- features/audio_features.csv: one row per frame. frame_index; start_sample (window start, analysis
  samples); centre_sample (window centre, the reference sample); t_us (session time of the centre,
  integer microseconds); rms (linear, full scale 1.0); energy_db (dBFS, floor -120); pitch_status
  (voiced / unvoiced / silent); f0_hz (empty when not voiced, never 0); f0_confidence (normalised
  correlation peak, -1 to 1).
- features/feature_metadata.json: effective configuration, timeline base, and the preprocessing record
  (source and analysis rates and counts, downmix expression, resampler, FFmpeg version).
- vad/vad_frames.csv: frame_index, centre_sample, t_us, energy_db, speech (1 or 0). Energy-based
  baseline decisions, not ground truth.
- vad/vad_segments.csv: segment_index, kind (speech / silence), start_sample, end_sample (exclusive),
  start_us, end_us, duration_ms. Boundaries are at hop resolution.
- vad/vad_metadata.json: the VAD rule and configuration.
- quality/audio_quality.json: technical measurements only (lengths, peak, RMS, clipping, silence and
  voiced ratios, VAD ratio). No subjective score; no accuracy claim.
- processing_manifest.json: written last. Lists every published file with its SHA-256 and size.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_EXTRACTION = Path(__file__).resolve().parent
if str(_EXTRACTION) not in sys.path:
    sys.path.insert(0, str(_EXTRACTION))
_ANALYSIS = _SRC / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from extraction import audio_acquisition as aa  # noqa: E402
from extraction import audio_signal as sig  # noqa: E402
from extraction.audio_signal import AudioConfig  # noqa: E402
from file_safety import atomic_write_text, check_writable  # noqa: E402
from project_paths import DATASETS_ROOT, RECORDINGS_RAW, is_valid_person, is_valid_session  # noqa: E402

SCHEMA = "audio-features-1"
ACQUISITION_SCHEMA = aa.SCHEMA
FEATURES_DIR, VAD_DIR, QUALITY_DIR = "features", "vad", "quality"
PROCESSING_MANIFEST = "processing_manifest.json"
FEATURE_CSV = "audio_features.csv"
FEATURE_META = "feature_metadata.json"
VAD_FRAMES_CSV = "vad_frames.csv"
VAD_SEGMENTS_CSV = "vad_segments.csv"
VAD_META = "vad_metadata.json"
QUALITY_JSON = "audio_quality.json"
FEATURE_COLUMNS = ["frame_index", "start_sample", "centre_sample", "t_us", "rms", "energy_db",
                   "pitch_status", "f0_hz", "f0_confidence"]
VAD_FRAME_COLUMNS = ["frame_index", "centre_sample", "t_us", "energy_db", "speech"]
VAD_SEGMENT_COLUMNS = ["segment_index", "kind", "start_sample", "end_sample", "start_us", "end_us", "duration_ms"]
MAX_ANALYSIS_BYTES = 1024**3  # float32 analysis signal: about 4 hours at 16 kHz
TIME_TOLERANCE_MS = 10
CLIP_LEVEL = 0.999
CHUNK_FRAMES = 4096
CHUNK_SAMPLES = 1 << 20


class FeatureError(aa.AudioError):
    """Feature processing cannot complete correctly. Nothing is published."""


# ---------------------------------------------------------------- inputs


def read_acquisition_manifest(audio_dir: Path) -> dict:
    path = Path(audio_dir) / aa.MANIFEST_NAME
    if not path.is_file():
        raise FeatureError("no H9.1 audio manifest for this session; run build_audio_session.py first")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FeatureError("the H9.1 audio manifest cannot be read") from exc
    if not isinstance(data, dict) or data.get("schema") != ACQUISITION_SCHEMA:
        raise FeatureError("the H9.1 audio manifest has an unknown schema")
    try:
        decoded, source, timeline = data["decoded"], data["source"], data["timeline"]
        selected, retention = data["streams"]["selected"], data["retention"]
        for key in ("sample_rate", "channels", "sample_count"):
            if int(decoded[key]) <= 0:
                raise ValueError(key)
        if source["type"] not in ("video_embedded", "external"):
            raise ValueError("source type")
        for key in ("audio_start_us", "origin_us", "external_offset_us", "sample_rate"):
            int(timeline[key])
        int(selected["index"])
        if not isinstance(retention["retained"], bool):
            raise ValueError("retained")
    except (KeyError, TypeError, ValueError) as exc:
        raise FeatureError("the H9.1 audio manifest is incomplete or invalid") from exc
    return data


def resolve_source(manifest: dict, audio_dir: Path, person: str, session: str,
                   recordings_root: Path, external: Path | None) -> Path:
    """The file to decode, verified against the SHA-256 recorded by H9.1."""
    if manifest["source"]["type"] == "video_embedded":
        path = Path(recordings_root) / person / f"{session}.mp4"
        aa._check_regular_file(path, "the recording")
        aa._require_inside(path, Path(recordings_root))
    elif manifest["retention"]["retained"]:
        stored = Path(audio_dir).parent / str(manifest["retention"]["stored_file"])
        aa._check_regular_file(stored, "the retained audio copy")
        aa._require_inside(stored, Path(audio_dir) / aa.SOURCE_DIR_NAME)
        path = stored
    elif external is not None:
        path = Path(external)
        aa._check_regular_file(path, "the external audio file")
    else:
        raise FeatureError("this external audio was not retained by H9.1; supply the same file with --external")
    if _sha256(path) != manifest["source"]["sha256"]:
        raise FeatureError("the source does not match the checksum recorded by H9.1; it has changed or is a different file")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- preprocessing


def _downmix_filter(channels: int) -> str | None:
    if channels == 1:
        return None
    weight = repr(1.0 / channels)
    return "pan=mono|c0=" + "+".join(f"{weight}*c{i}" for i in range(channels))


def _run_bounded(args: list[str], out: Path, limit: int, src: Path) -> None:
    """Run FFmpeg and stop it if its output grows past limit bytes. Same rules as the H9.1 decode:
    the size is watched while the process runs, not only after it has finished."""
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + aa.DECODE_TIMEOUT_S
    stderr = b""
    try:
        while True:
            try:
                _, stderr = proc.communicate(timeout=aa.DECODE_POLL_S)
                break
            except subprocess.TimeoutExpired:
                if out.exists() and out.stat().st_size > limit:
                    raise FeatureError("the analysis signal exceeds the size limit; resampling was stopped")
                if time.monotonic() > deadline:
                    raise FeatureError(f"resample and downmix {src.name}: timed out")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    if proc.returncode != 0:
        text = (stderr or b"").decode("utf-8", "replace").replace(str(src), src.name)
        last = text.strip().splitlines()[-1:] or ["no message"]
        raise FeatureError(f"resample and downmix {src.name}: exit code {proc.returncode}: {last[0][:300]}")
    if out.exists() and out.stat().st_size > limit:
        raise FeatureError("the analysis signal exceeds the size limit")


def decode_analysis_signal(src: Path, manifest: dict, cfg: AudioConfig, workdir: Path) -> tuple[Path, int, dict]:
    """Decode, downmix and resample to float32 mono raw PCM. Returns (path, samples, info)."""
    decoded = manifest["decoded"]
    src_rate, channels = int(decoded["sample_rate"]), int(decoded["channels"])
    src_count = int(decoded["sample_count"])
    rate = cfg.analysis_rate
    planned_samples = -(-src_count * rate // src_rate)
    if planned_samples * 4 > MAX_ANALYSIS_BYTES:
        raise FeatureError("the audio would be too long for the analysis limit at this analysis rate")
    out = Path(workdir) / "analysis.f32"
    args = [aa._tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-map", f"0:{int(manifest['streams']['selected']['index'])}", "-vn", "-sn", "-dn"]
    downmix = _downmix_filter(channels)
    if downmix:
        args += ["-af", downmix]
    args += ["-ar", str(rate), "-ac", "1", "-c:a", "pcm_f32le", "-f", "f32le", str(out)]
    _run_bounded(args, out, MAX_ANALYSIS_BYTES, src)
    size = out.stat().st_size if out.exists() else 0
    if size % 4 or size == 0:
        raise FeatureError("the decoded analysis signal is empty or truncated")
    if size > MAX_ANALYSIS_BYTES:
        raise FeatureError("the decoded analysis signal exceeds the size limit")
    n = size // 4
    expected = src_count * rate / src_rate
    diff_samples = n - expected
    if abs(diff_samples) * 1000 / rate > TIME_TOLERANCE_MS:
        raise FeatureError(f"resampled length differs from the source by {diff_samples:.1f} samples")
    info = {
        "source_sample_rate": src_rate,
        "source_channels": channels,
        "source_sample_count": src_count,
        "analysis_sample_rate": rate,
        "analysis_channels": 1,
        "analysis_sample_count": int(n),
        "expected_analysis_sample_count": round(expected),
        "length_difference_samples": round(diff_samples, 3),
        "downmix": "equal-weight mean of all channels" if downmix else "none (mono source)",
        "downmix_expression": downmix,
        "resampler": "FFmpeg swresample default (windowed sinc)",
        "ffmpeg": aa.ffmpeg_version(),
    }
    return out, int(n), info


# ---------------------------------------------------------------- analysis


class RawFloat32:
    """Read-only access to a raw little-endian float32 file in blocks.

    A plain file handle is used rather than a memory map, so the file is released when the
    context exits, including on failure. Windows cannot delete a file that a map still holds.
    """

    def __init__(self, path: Path, samples: int):
        self.path = Path(path)
        self.samples = int(samples)
        self._handle = None

    def __enter__(self):
        self._handle = open(self.path, "rb")
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False

    def __len__(self) -> int:
        return self.samples

    def block(self, start: int, stop: int) -> np.ndarray:
        start, stop = max(0, int(start)), min(self.samples, int(stop))
        if stop <= start:
            return np.empty(0, dtype=np.float32)
        self._handle.seek(start * 4)
        data = self._handle.read((stop - start) * 4)
        if len(data) != (stop - start) * 4:
            raise FeatureError("the analysis signal ended before its recorded length")
        return np.frombuffer(data, dtype="<f4")


def analyse(signal, cfg: AudioConfig, base_us: int) -> dict:
    """Frame features, VAD and whole-signal statistics. signal supports len() and block(start, stop)."""
    n = len(signal)
    frames_total, tail = sig.frame_grid(n, cfg)
    if frames_total == 0:
        raise FeatureError("the audio is shorter than one analysis window")
    w, hop = cfg.window_samples, cfg.hop_samples
    rate = cfg.analysis_rate

    rms = np.empty(frames_total)
    db = np.empty(frames_total)
    silent = np.empty(frames_total, dtype=bool)
    f0 = np.empty(frames_total)
    conf = np.empty(frames_total)
    for first in range(0, frames_total, CHUNK_FRAMES):
        last = min(frames_total, first + CHUNK_FRAMES)
        segment = signal.block(first * hop, (last - 1) * hop + w)
        frames = np.lib.stride_tricks.sliding_window_view(segment, w)[::hop]
        r, d, s = sig.energy(frames, cfg)
        p, c = sig.pitch(frames, cfg, s)
        rms[first:last], db[first:last], silent[first:last] = r, d, s
        f0[first:last], conf[first:last] = p, c

    peak, sumsq, clipped, nonfinite = 0.0, 0.0, 0, 0
    for start in range(0, n, CHUNK_SAMPLES):
        block = signal.block(start, start + CHUNK_SAMPLES).astype(np.float64)
        finite = np.isfinite(block)
        nonfinite += int(block.size - np.count_nonzero(finite))
        block = np.where(finite, block, 0.0)
        if block.size:
            peak = max(peak, float(np.max(np.abs(block))))
        sumsq += float(np.sum(block * block))
        clipped += int(np.count_nonzero(np.abs(block) >= CLIP_LEVEL))
    if nonfinite:
        raise FeatureError(f"{nonfinite} non-finite samples in the analysis signal")

    speech = sig.vad_frames(db, cfg.vad_db)
    segments = sig.vad_segments(speech, cfg)
    centre = np.arange(frames_total, dtype=np.int64) * hop + w // 2
    start_samples = np.arange(frames_total, dtype=np.int64) * hop
    status = np.where(silent, "silent", np.where(np.isnan(f0), "unvoiced", "voiced"))
    rms_all = float(np.sqrt(sumsq / n)) if n else 0.0
    return {
        "frames": frames_total,
        "tail_samples": tail,
        "start_sample": start_samples,
        "centre_sample": centre,
        "t_us": sig.sample_time_us(centre, rate, base_us),
        "rms": rms,
        "energy_db": db,
        "silent": silent,
        "speech": speech,
        "f0": f0,
        "conf": conf,
        "status": status,
        "segments": segments,
        "peak": peak,
        "rms_all": rms_all,
        "clipped": clipped,
        "signal_samples": n,
    }


# ---------------------------------------------------------------- output writing


def _fmt(value, digits):
    return "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"


def _write_csv(path: Path, columns, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def _write_features(stage: Path, a: dict) -> None:
    rows = (
        [i, int(a["start_sample"][i]), int(a["centre_sample"][i]), int(a["t_us"][i]),
         _fmt(a["rms"][i], 8), _fmt(a["energy_db"][i], 4), a["status"][i],
         _fmt(a["f0"][i], 4), _fmt(a["conf"][i], 4)]
        for i in range(a["frames"])
    )
    _write_csv(stage / FEATURES_DIR / FEATURE_CSV, FEATURE_COLUMNS, rows)


def _write_vad(stage: Path, a: dict, cfg: AudioConfig, base_us: int) -> None:
    frame_rows = (
        [i, int(a["centre_sample"][i]), int(a["t_us"][i]), _fmt(a["energy_db"][i], 4), int(a["speech"][i])]
        for i in range(a["frames"])
    )
    _write_csv(stage / VAD_DIR / VAD_FRAMES_CSV, VAD_FRAME_COLUMNS, frame_rows)
    seg_rows = []
    for idx, seg in enumerate(a["segments"]):
        start_sample = seg["start_frame"] * cfg.hop_samples
        end_sample = seg["end_frame"] * cfg.hop_samples
        start_us = int(sig.sample_time_us(start_sample, cfg.analysis_rate, base_us))
        end_us = int(sig.sample_time_us(end_sample, cfg.analysis_rate, base_us))
        seg_rows.append([idx, seg["kind"], start_sample, end_sample, start_us, end_us,
                         _fmt((end_sample - start_sample) * 1000 / cfg.analysis_rate, 3)])
    _write_csv(stage / VAD_DIR / VAD_SEGMENTS_CSV, VAD_SEGMENT_COLUMNS, seg_rows)


def _sha(path: Path) -> str:
    return _sha256(path)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def build_quality(a: dict, cfg: AudioConfig, info: dict) -> dict:
    frames = a["frames"]
    speech_frames = int(np.count_nonzero(a["speech"]))
    voiced = int(np.count_nonzero(~np.isnan(a["f0"])))
    silent = int(np.count_nonzero(a["silent"]))
    rate = cfg.analysis_rate
    peak_db = 20 * np.log10(a["peak"]) if a["peak"] > 0 else None
    rms_db = 20 * np.log10(a["rms_all"]) if a["rms_all"] > 0 else None
    return {
        "schema": SCHEMA,
        "kind": "technical measurements only; no subjective quality score",
        "source": {k: info[k] for k in ("source_sample_rate", "source_channels", "source_sample_count")},
        "analysis": {
            "sample_rate": rate,
            "channels": 1,
            "sample_count": info["analysis_sample_count"],
            "expected_sample_count": info["expected_analysis_sample_count"],
            "length_difference_samples": info["length_difference_samples"],
            "duration_ms": round(info["analysis_sample_count"] * 1000 / rate, 3),
            "tail_samples_not_framed": a["tail_samples"],
        },
        "amplitude": {
            "peak_linear": round(a["peak"], 6),
            "peak_dbfs": None if peak_db is None else round(float(peak_db), 3),
            "rms_dbfs": None if rms_db is None else round(float(rms_db), 3),
            "clipping_fraction": a["clipped"] / a["signal_samples"],
            "clipping_threshold": CLIP_LEVEL,
            "non_finite_samples": 0,
        },
        "frames": {
            "count": frames,
            "silent": silent,
            "silence_ratio": silent / frames,
            "voiced": voiced,
            "voiced_ratio": voiced / frames,
            "unvoiced": frames - silent - voiced,
            "missing_pitch_values": frames - voiced,
        },
        "vad": {
            "speech_frames": speech_frames,
            "speech_ratio": speech_frames / frames,
            "segments": len(a["segments"]),
            "is_ground_truth": False,
        },
    }


def _config_summary(cfg: AudioConfig) -> dict:
    return {**cfg.as_dict(), "frame_definition": "window of window_samples starting at i * hop_samples; "
                                                  "reference time is the window centre"}


def validate_stage(stage: Path, cfg: AudioConfig, a: dict) -> None:
    """Read back what will be published and check it against the in-memory results."""
    with open(stage / FEATURES_DIR / FEATURE_CSV, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != FEATURE_COLUMNS:
            raise FeatureError("feature CSV header is wrong")
        count = sum(1 for _ in reader)
    if count != a["frames"]:
        raise FeatureError("feature row count does not match the frame count")
    with open(stage / VAD_DIR / VAD_FRAMES_CSV, encoding="utf-8", newline="") as handle:
        vad_rows = sum(1 for _ in handle) - 1
    if vad_rows != a["frames"]:
        raise FeatureError("VAD frame row count does not match the frame count")
    starts = a["start_sample"]
    if a["frames"] > 1 and not np.all(np.diff(starts) == cfg.hop_samples):
        raise FeatureError("frame starts are not on the hop grid")
    t = a["t_us"]
    if a["frames"] > 1 and not np.all(np.diff(t) > 0):
        raise FeatureError("frame times are not strictly increasing")
    voiced = ~np.isnan(a["f0"])
    if np.any(a["f0"][voiced] < cfg.pitch_min_hz * 0.999) or np.any(a["f0"][voiced] > cfg.pitch_max_hz * 1.001):
        raise FeatureError("a voiced F0 value lies outside the configured pitch range")
    if np.any(a["silent"] & voiced):
        raise FeatureError("a silent frame was marked voiced")
    if not np.all(np.isfinite(a["energy_db"])) or not np.all(np.isfinite(a["rms"])):
        raise FeatureError("energy values are not finite")


def _publish(stage: Path, audio_dir: Path, outputs: list[str], manifest_entries: dict, regenerate: bool) -> None:
    token = uuid.uuid4().hex[:12]
    previous = audio_dir / f".h92-previous-{token}"
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        if regenerate:
            previous.mkdir()
            # The old completion marker is moved first, so an interrupted run cannot leave new files
            # beside an old marker that still appears to validate them.
            for name in [PROCESSING_MANIFEST] + outputs:
                target = audio_dir / name
                if target.exists():
                    os.replace(target, previous / name)
                    moved_old.append(name)
        for name in outputs:
            os.replace(stage / name, audio_dir / name)
            moved_new.append(name)
        _write_json(audio_dir / PROCESSING_MANIFEST, manifest_entries)
    except Exception:
        for name in moved_new:
            target = audio_dir / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        for name in moved_old:
            os.replace(previous / name, audio_dir / name)
        raise
    finally:
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)


# ---------------------------------------------------------------- entry point


def process_session_audio(
    person: str,
    session: str,
    *,
    config: AudioConfig | None = None,
    external: Path | None = None,
    regenerate: bool = False,
    recordings_root: Path = RECORDINGS_RAW,
    datasets_root: Path = DATASETS_ROOT,
) -> dict:
    """Run H9.2 for one session. Returns the processing manifest. Nothing is published on failure."""
    if not is_valid_person(person) or not is_valid_session(session):
        raise FeatureError("invalid person or session name")
    cfg = (config or AudioConfig()).validate()
    audio_dir = Path(datasets_root) / person / session / aa.AUDIO_DIR_NAME
    outputs = [FEATURES_DIR, VAD_DIR, QUALITY_DIR]

    with aa.session_lock(audio_dir):
        manifest = read_acquisition_manifest(audio_dir)
        check_writable([audio_dir / name for name in outputs + [PROCESSING_MANIFEST]], regenerate)
        src = resolve_source(manifest, audio_dir, person, session, Path(recordings_root), external)
        timeline = manifest["timeline"]
        base_us = int(timeline["audio_start_us"]) + int(timeline["external_offset_us"]) - int(timeline["origin_us"])

        stage = audio_dir / f".h92-stage-{uuid.uuid4().hex[:12]}"
        stage.mkdir()
        try:
            with tempfile.TemporaryDirectory(prefix="audio_features_") as work:
                raw, n, info = decode_analysis_signal(src, manifest, cfg, Path(work))
                with RawFloat32(raw, n) as signal:
                    a = analyse(signal, cfg, base_us)
            quality = build_quality(a, cfg, info)
            meta = {
                "schema": SCHEMA,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "config": _config_summary(cfg),
                "timeline": {"base_us": base_us, "formula": timeline.get("formula")},
                "preprocessing": info,
            }
            _write_features(stage, a)
            _write_vad(stage, a, cfg, base_us)
            _write_json(stage / FEATURES_DIR / FEATURE_META, meta)
            _write_json(stage / VAD_DIR / VAD_META, {**meta, "vad": {
                "rule": "frame is speech when energy_db >= vad_db; gaps <= vad_merge_gap_ms merged; "
                        "speech shorter than vad_min_speech_ms dropped; silence reported from vad_min_silence_ms",
                "is_ground_truth": False}})
            _write_json(stage / QUALITY_DIR / QUALITY_JSON, quality)
            validate_stage(stage, cfg, a)
            entries = {}
            for rel in [f"{FEATURES_DIR}/{FEATURE_CSV}", f"{FEATURES_DIR}/{FEATURE_META}",
                        f"{VAD_DIR}/{VAD_FRAMES_CSV}", f"{VAD_DIR}/{VAD_SEGMENTS_CSV}", f"{VAD_DIR}/{VAD_META}",
                        f"{QUALITY_DIR}/{QUALITY_JSON}"]:
                entries[rel] = {"sha256": _sha(stage / rel), "bytes": (stage / rel).stat().st_size}
            processing = {
                "schema": SCHEMA,
                "person": person,
                "session": session,
                "acquisition_schema": ACQUISITION_SCHEMA,
                "created_utc": meta["created_utc"],
                "frames": a["frames"],
                "files": entries,
                "validation": "read back and checked before publication",
            }
            _publish(stage, audio_dir, outputs, processing, regenerate)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return processing
