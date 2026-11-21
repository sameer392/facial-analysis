"""Transcription dataset layer (phase H9.3a): records, validation and atomic publication.

Works with any AsrEngine. The dataset contract does not depend on any runtime. A fake or real engine
gives an EngineResult; this module turns it into dataset records and publishes them.

Inputs
- The H9.1 manifest (audio/manifest.json), checked against the source's SHA-256.
- The H9.2 outputs (audio/processing_manifest.json and its files), whose SHA-256 values are re-checked.
- The H9.2 analysis signal (decoded again here, from the same verified source, at the analysis rate).

Outputs: audio/transcription/
    transcript.json       document: engine, model, decoding, language, raw and normalised text
    segments.csv          one row per segment, raw text and normalised text
    words.csv             one row per word
    transcription_metadata.json   written last: file list with SHA-256 values (completion marker)

Timing: every start and end is an integer analysis-sample index, computed once from the engine's float
seconds with asr_interface.seconds_to_sample. Microseconds come from audio_signal.sample_time_us using
the H9.1 timeline:  session_us = audio_start + external_offset - origin. Negative values are valid.

Text: the raw text is stored exactly as the engine returned it. Normalisation is stored separately and
is limited to Unicode NFC and whitespace. It does not change case, punctuation, spelling or script, and
it never translates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ANALYSIS = _SRC / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from extraction import audio_acquisition as aa  # noqa: E402
from extraction import audio_processing as ap  # noqa: E402
from extraction import audio_signal as sig  # noqa: E402
from extraction.asr_interface import (  # noqa: E402
    AsrEngine,
    AUTO,
    EngineResponseError,
    EngineResult,
    TranscriptionRequest,
    seconds_to_sample,
    validate_engine_result,
)
from extraction.audio_signal import AudioConfig  # noqa: E402
from file_safety import atomic_write_text, check_writable  # noqa: E402
from project_paths import DATASETS_ROOT, RECORDINGS_RAW, is_valid_person, is_valid_session  # noqa: E402

SCHEMA = "transcription-1"
TRANSCRIPTION_DIR = "transcription"
TRANSCRIPT_JSON = "transcript.json"
SEGMENTS_CSV = "segments.csv"
WORDS_CSV = "words.csv"
METADATA_JSON = "transcription_metadata.json"
SEGMENT_COLUMNS = ["segment_index", "start_sample", "end_sample", "start_us", "end_us", "language",
                   "text_raw", "text_normalized", "avg_logprob", "no_speech_prob",
                   "raw_start_s", "raw_end_s", "speaker"]
WORD_COLUMNS = ["segment_index", "word_index", "word_raw", "start_sample", "end_sample", "start_us", "end_us",
                "probability", "raw_start_s", "raw_end_s", "speaker"]


class TranscriptionError(aa.AudioError):
    """Transcription cannot be completed correctly. Nothing is published."""


# ---------------------------------------------------------------- text


def normalize_text(text: str) -> str:
    """Unicode NFC and whitespace only. No case, punctuation, spelling, script or language change."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return " ".join(text.split())


# ---------------------------------------------------------------- records (pure)


def build_records(result: EngineResult, request: TranscriptionRequest, rate: int, n_samples: int,
                  base_us: int) -> tuple[list[dict], list[dict]]:
    """Segment and word rows. Times are integer samples, from the engine's floats by seconds_to_sample."""
    segments, words = [], []
    previous_start = -1
    for s_index, seg in enumerate(result.segments):
        start = seconds_to_sample(seg.start_s, rate)
        end = seconds_to_sample(seg.end_s, rate)
        if end > n_samples:
            raise TranscriptionError(f"segment {s_index} ends beyond the analysis signal")
        if start < previous_start or end < start:
            raise TranscriptionError(f"segment {s_index} is out of order")
        previous_start = start
        segments.append({
            "segment_index": s_index,
            "start_sample": start,
            "end_sample": end,
            "start_us": int(sig.sample_time_us(start, rate, base_us)),
            "end_us": int(sig.sample_time_us(end, rate, base_us)),
            "language": result.language,
            "text_raw": seg.text,
            "text_normalized": normalize_text(seg.text),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "raw_start_s": repr(float(seg.start_s)),
            "raw_end_s": repr(float(seg.end_s)),
            "speaker": None,
        })
        for w_index, w in enumerate(seg.words):
            w_start = seconds_to_sample(w.start_s, rate)
            w_end = seconds_to_sample(w.end_s, rate)
            if w_start < start or w_end > end or w_end < w_start:
                raise TranscriptionError(f"word {w_index} of segment {s_index} lies outside its segment")
            words.append({
                "segment_index": s_index,
                "word_index": w_index,
                "word_raw": w.word,
                "start_sample": w_start,
                "end_sample": w_end,
                "start_us": int(sig.sample_time_us(w_start, rate, base_us)),
                "end_us": int(sig.sample_time_us(w_end, rate, base_us)),
                "probability": None if w.probability is None else float(w.probability),
                "raw_start_s": repr(float(w.start_s)),
                "raw_end_s": repr(float(w.end_s)),
                "speaker": None,
            })
    return segments, words


def validate_records(segments: list[dict], words: list[dict], rate: int, n_samples: int, base_us: int) -> None:
    """Checks on the finished records: ordering, bounds, microsecond consistency, probabilities."""
    last_start = -1
    for seg in segments:
        if seg["start_sample"] < last_start or seg["end_sample"] < seg["start_sample"]:
            raise TranscriptionError("segment timestamps are not ordered")
        if not 0 <= seg["start_sample"] <= seg["end_sample"] <= n_samples:
            raise TranscriptionError("a segment lies outside the analysis signal")
        if seg["start_us"] != int(sig.sample_time_us(seg["start_sample"], rate, base_us)) or \
                seg["end_us"] != int(sig.sample_time_us(seg["end_sample"], rate, base_us)):
            raise TranscriptionError("a segment's microseconds do not match its samples")
        last_start = seg["start_sample"]
    for w in words:
        if w["end_sample"] < w["start_sample"] or not 0 <= w["start_sample"] <= w["end_sample"] <= n_samples:
            raise TranscriptionError("a word lies outside the analysis signal")
        if w["start_us"] != int(sig.sample_time_us(w["start_sample"], rate, base_us)) or \
                w["end_us"] != int(sig.sample_time_us(w["end_sample"], rate, base_us)):
            raise TranscriptionError("a word's microseconds do not match its samples")
        p = w["probability"]
        if p is not None and not (isinstance(p, float) and 0.0 <= p <= 1.0):
            raise TranscriptionError("a word probability is outside 0 to 1")


def build_document(result: EngineResult, request: TranscriptionRequest, *, person: str, session: str,
                   rate: int, n_samples: int, base_us: int, timeline: dict, source_sha256: str,
                   acquisition_sha: str | None, config: AudioConfig, segments: list[dict],
                   words: list[dict]) -> dict:
    """transcript.json content. raw_text is the engine's text, unchanged."""
    return {
        "schema": SCHEMA,
        "person": person,
        "session": session,
        "engine": result.info.as_dict(),
        "decoding": request.options.as_dict(),
        "language": {
            "requested": request.language,
            "selected": result.language,
            "detected": request.language == AUTO,
            "probability": result.language_probability,
        },
        "raw_text": result.text,
        "normalized_text": normalize_text(result.text),
        "source": {"sha256": source_sha256, "acquisition_schema": aa.SCHEMA,
                   "acquisition_manifest_sha256": acquisition_sha},
        "analysis": {"sample_rate": rate, "sample_count": n_samples,
                     "source_sample_rate": timeline["source_sample_rate"],
                     "source_sample_count": timeline["source_sample_count"]},
        "timeline": {"base_us": base_us, "formula": "session_us = audio_start + external_offset - origin",
                     "audio_start_us": timeline["audio_start_us"],
                     "external_offset_us": timeline["external_offset_us"],
                     "origin_us": timeline["origin_us"]},
        "counts": {"segments": len(segments), "words": len(words)},
        "speaker": None,
        "processing": {"analysis_config": config.as_dict()},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- files


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Missing values (None) are written as empty cells, never as a number."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row[c] is None else row[c] for c in columns])


def write_outputs(stage: Path, doc: dict, segments: list[dict], words: list[dict]) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    atomic_write_text(stage / TRANSCRIPT_JSON, json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    _write_csv(stage / SEGMENTS_CSV, SEGMENT_COLUMNS, segments)
    _write_csv(stage / WORDS_CSV, WORD_COLUMNS, words)


def read_back_check(stage: Path, doc: dict, segments: list[dict], words: list[dict]) -> None:
    """Re-read what will be published and compare it with the records in memory."""
    loaded = json.loads((stage / TRANSCRIPT_JSON).read_text(encoding="utf-8"))
    if loaded != doc:
        raise TranscriptionError("transcript.json does not read back as written")
    for name, columns, rows in ((SEGMENTS_CSV, SEGMENT_COLUMNS, segments), (WORDS_CSV, WORD_COLUMNS, words)):
        with open(stage / name, encoding="utf-8", newline="") as handle:
            lines = list(csv.reader(handle))
        if lines[0] != columns or len(lines) - 1 != len(rows):
            raise TranscriptionError(f"{name} does not match its schema or row count")


def write_metadata(stage: Path, doc: dict, file_names: list[str]) -> dict:
    """Completion marker, written last: the file list with SHA-256 values."""
    files = {}
    for rel in file_names:
        path = stage / rel
        files[rel] = {"sha256": _sha(path), "bytes": path.stat().st_size}
    meta = {"schema": SCHEMA, "files": files, "counts": doc["counts"],
            "engine": doc["engine"], "language": doc["language"],
            "created_utc": doc["created_utc"], "validation": "read back and checked before publication"}
    atomic_write_text(stage / METADATA_JSON, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


# ---------------------------------------------------------------- publication


def publish(stage: Path, audio_dir: Path, regenerate: bool) -> None:
    """Move the staged folder into place. An existing transcription is kept until the new one is in place,
    and restored if publication fails. The completion marker is moved last."""
    target = audio_dir / TRANSCRIPTION_DIR
    previous = audio_dir / f".h93-previous-{uuid.uuid4().hex[:12]}"
    moved_old = False
    try:
        if regenerate and target.exists():
            os.replace(target, previous)
            moved_old = True
        os.replace(stage, target)
    except Exception:
        if target.exists() and moved_old:
            shutil.rmtree(target, ignore_errors=True)
        if moved_old and previous.exists():
            os.replace(previous, target)
        raise
    finally:
        if previous.exists():
            shutil.rmtree(previous, ignore_errors=True)


def verify_processing_outputs(audio_dir: Path) -> None:
    """H9.2 outputs must be present, and every file listed in their marker must match its SHA-256."""
    marker = audio_dir / ap.PROCESSING_MANIFEST
    if not marker.is_file():
        raise TranscriptionError("no H9.2 processing outputs for this session; run build_audio_features.py first")
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TranscriptionError("the H9.2 completion marker cannot be read") from exc
    for rel, entry in data.get("files", {}).items():
        path = audio_dir / rel
        if not path.is_file() or _sha(path) != entry.get("sha256"):
            raise TranscriptionError("an H9.2 output does not match its recorded checksum")


# ---------------------------------------------------------------- entry point


def process_transcription(
    person: str,
    session: str,
    engine: AsrEngine,
    request: TranscriptionRequest,
    *,
    config: AudioConfig | None = None,
    external=None,
    regenerate: bool = False,
    recordings_root: Path = RECORDINGS_RAW,
    datasets_root: Path = DATASETS_ROOT,
) -> dict:
    """Run one transcription through an engine and publish the dataset. Returns the metadata marker."""
    if not is_valid_person(person) or not is_valid_session(session):
        raise TranscriptionError("invalid person or session name")
    cfg = (config or AudioConfig()).validate()
    audio_dir = Path(datasets_root) / person / session / aa.AUDIO_DIR_NAME
    out_dir = audio_dir / TRANSCRIPTION_DIR

    with aa.session_lock(audio_dir):
        manifest = ap.read_acquisition_manifest(audio_dir)
        verify_processing_outputs(audio_dir)
        check_writable([out_dir], regenerate)
        src = ap.resolve_source(manifest, audio_dir, person, session, Path(recordings_root), external)
        timeline = manifest["timeline"]
        base_us = int(timeline["audio_start_us"]) + int(timeline["external_offset_us"]) - int(timeline["origin_us"])
        acquisition_sha = _sha(audio_dir / aa.MANIFEST_NAME)
        timeline = dict(timeline, source_sample_rate=int(manifest["decoded"]["sample_rate"]),
                        source_sample_count=int(manifest["decoded"]["sample_count"]))

        stage = audio_dir / f".h93-stage-{uuid.uuid4().hex[:12]}"
        try:
            with tempfile.TemporaryDirectory(prefix="asr_") as work:
                raw, n, _info = ap.decode_analysis_signal(src, manifest, cfg, Path(work))
                samples = np.fromfile(raw, dtype="<f4")
            if samples.size != n:
                raise TranscriptionError("the analysis signal length does not match its decode record")
            rate = cfg.analysis_rate
            try:
                result = engine.transcribe(samples, rate, request)
            except EngineResponseError:
                raise
            except Exception as exc:  # engine failure is reported, never published
                raise TranscriptionError(f"engine failed: {type(exc).__name__}") from exc
            result = validate_engine_result(result, request, n / rate)
            segments, words = build_records(result, request, rate, n, base_us)
            validate_records(segments, words, rate, n, base_us)
            doc = build_document(result, request, person=person, session=session, rate=rate, n_samples=n,
                                 base_us=base_us, timeline=timeline, source_sha256=manifest["source"]["sha256"],
                                 acquisition_sha=acquisition_sha, config=cfg, segments=segments, words=words)
            write_outputs(stage, doc, segments, words)
            read_back_check(stage, doc, segments, words)
            meta = write_metadata(stage, doc, [TRANSCRIPT_JSON, SEGMENTS_CSV, WORDS_CSV])
            publish(stage, audio_dir, regenerate)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return meta
