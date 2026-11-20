"""Audio acquisition for one session (H9.1).

Reads the audio of a session from one of two sources and writes audio/manifest.json inside the
session's dataset folder. The manifest is the only file this stage writes.

Sources
- video_embedded: an audio stream of recordings/raw/<person>/<session>.mp4 (never modified).
- external: a separately supplied WAV or MP3 file, with an explicit synchronisation offset.

Processing
- FFmpeg decodes the chosen stream to 16-bit PCM WAV in a temporary folder, at the source's own
  sample rate and channel count. Nothing is resampled or downmixed here.
- The decoded samples are counted, checksummed and compared with the container's duration.
- The temporary folder is deleted on success and on failure.

Retention
- Embedded audio is never copied: the original video stays the only stored copy.
- External audio is not copied by default. With retain=True a verified copy is stored under
  audio/source/. Without it, reanalysis needs the file to be supplied again.

Timing (integer microseconds on the session clock)
    session_us(k) = audio_start_us + round(k * 1e6 / sample_rate) + offset_us - origin_us
- k is the sample index from the start of the decoded stream.
- origin_us is the first video presentation time, or 0 when the source has no video.
- offset_us is an explicit input for external audio. Embedded audio is already on the video clock,
  so an offset is refused for it.
- FFmpeg's WAV output always starts at zero, so the stream's own start time is read from the
  probe and stored; it is not inferred from the decoded file.

Limits and locking
- Decoded PCM is capped at MAX_DECODED_PCM_BYTES. The size is planned from duration, rate, channels
  and sample width before decoding, the output is watched while FFmpeg runs, and the result is
  checked afterwards.
- One acquisition per session at a time: audio/.acquire.lock is created exclusively and holds the
  owner's process ID and host. A lock is replaced only if this machine wrote it and that process
  has exited; unreadable or foreign locks are never taken over.

The manifest never contains the external file's name; it records an opaque source_id and the extension.

Errors are raised as AudioError with the file name only, never a full path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import wave
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from file_safety import atomic_write_text, check_writable  # noqa: E402
from project_paths import (  # noqa: E402
    DATASETS_ROOT,
    RECORDINGS_RAW,
    is_valid_person,
    is_valid_session,
)

SCHEMA = "audio-acquisition-1"
AUDIO_DIR_NAME = "audio"
SOURCE_DIR_NAME = "source"
MANIFEST_NAME = "manifest.json"
SAMPLE_FORMAT = "pcm_s16le"
EXTERNAL_SUFFIXES = {"wav": ".wav", "mp3": ".mp3"}  # ffprobe format name -> expected file suffix
MAX_EXTERNAL_BYTES = 2 * 1024**3
MAX_DURATION_S = 4 * 3600
# Decoded 16-bit PCM limit. 4 hours of 48 kHz stereo is about 2.6 GiB, so 3 GiB admits every
# permitted duration at ordinary rates and refuses high-rate or many-channel sources.
MAX_DECODED_PCM_BYTES = 3 * 1024**3
SAMPLE_WIDTH = 2  # bytes per sample for pcm_s16le
WAV_HEADER_ALLOWANCE = 1 << 20  # slack for the WAV header and FFmpeg's own framing
DURATION_TOLERANCE_MS = 100
PROBE_TIMEOUT_S = 120
DECODE_TIMEOUT_S = 1800
DECODE_POLL_S = 0.25
LOCK_NAME = ".acquire.lock"
COPY_CHUNK = 1 << 20


class AudioError(Exception):
    """Audio cannot be acquired completely and correctly. The message names the file, never a path."""


@dataclass(frozen=True)
class AudioStream:
    index: int
    codec: str
    sample_rate: int
    channels: int
    start_us: int | None
    duration_s: float | None

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- FFmpeg tools


def _tool(name: str) -> str:
    """Locate ffmpeg or ffprobe. FACIAL_FFMPEG_DIR may name the bin folder; otherwise PATH is used."""
    folder = os.environ.get("FACIAL_FFMPEG_DIR")
    if folder:
        candidate = Path(folder).expanduser() / (name + (".exe" if os.name == "nt" else ""))
        if candidate.is_file():
            return str(candidate)
        raise AudioError(f"{name} was not found in FACIAL_FFMPEG_DIR")
    found = shutil.which(name)
    if not found:
        raise AudioError(f"{name} was not found. Install FFmpeg or set FACIAL_FFMPEG_DIR to its bin folder.")
    return found


def _run(args: list[str], label: str, timeout: int, names: list[Path]) -> subprocess.CompletedProcess:
    """Run a tool without a shell. Paths in its error output are replaced with file names."""
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"{label}: timed out after {timeout} s") from exc
    if proc.returncode != 0:
        text = proc.stderr.decode("utf-8", "replace")
        for p in names:
            text = text.replace(str(p), p.name)
        last = text.strip().splitlines()[-1:] or ["no message"]
        raise AudioError(f"{label}: exit code {proc.returncode}: {last[0][:300]}")
    return proc


def probe(path: Path) -> dict:
    """ffprobe's JSON description of the container and its streams."""
    proc = _run(
        [_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        f"probe {path.name}", PROBE_TIMEOUT_S, [path],
    )
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise AudioError(f"probe {path.name}: metadata could not be read") from exc


def ffmpeg_version() -> str:
    proc = _run([_tool("ffmpeg"), "-version"], "ffmpeg -version", PROBE_TIMEOUT_S, [])
    return proc.stdout.decode("utf-8", "replace").splitlines()[0].strip()


# ---------------------------------------------------------------- metadata (pure)


def _seconds_to_us(value) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return round(float(value) * 1_000_000)
    except (TypeError, ValueError):
        return None


def _is_cover_art(stream: dict) -> bool:
    return bool((stream.get("disposition") or {}).get("attached_pic"))


def parse_audio_streams(probe_data: dict) -> list[AudioStream]:
    """Audio streams in container order. Refuses an audio stream whose rate or channels are unknown."""
    result = []
    for s in probe_data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        try:
            rate = int(s["sample_rate"])
            channels = int(s["channels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AudioError(f"audio stream {s.get('index')} has no usable sample rate or channel count") from exc
        if rate <= 0 or channels < 1:
            raise AudioError(f"audio stream {s.get('index')} has an invalid sample rate or channel count")
        duration = s.get("duration")
        result.append(AudioStream(
            index=int(s["index"]),
            codec=str(s.get("codec_name", "unknown")),
            sample_rate=rate,
            channels=channels,
            start_us=_seconds_to_us(s.get("start_time")),
            duration_s=float(duration) if duration not in (None, "N/A") else None,
        ))
    return result


def first_video_stream(probe_data: dict) -> dict | None:
    """The first real video stream. Cover art attached to an audio file does not count."""
    for s in probe_data.get("streams", []):
        if s.get("codec_type") == "video" and not _is_cover_art(s):
            return s
    return None


def select_stream(streams: list[AudioStream], requested: int | None) -> AudioStream:
    """One audio stream is used automatically. With several, the caller must name the container index."""
    if not streams:
        raise AudioError("no audio stream was found")
    if requested is None:
        if len(streams) == 1:
            return streams[0]
        listing = ", ".join(f"{s.index} ({s.codec}, {s.sample_rate} Hz, {s.channels} ch)" for s in streams)
        raise AudioError(f"several audio streams; choose one with a stream index: {listing}")
    for s in streams:
        if s.index == requested:
            return s
    raise AudioError(f"audio stream {requested} was not found")


def session_timeline(*, audio_start_us: int, sample_rate: int, sample_count: int, offset_us: int,
                     origin_us: int) -> dict:
    """Integer-microsecond placement of the first and last decoded samples on the session clock."""
    if sample_rate <= 0 or sample_count < 1:
        raise AudioError("no decoded samples to place on the timeline")
    first = audio_start_us + offset_us - origin_us
    last_offset = round((sample_count - 1) * 1_000_000 / sample_rate)
    return {
        "formula": "session_us(k) = audio_start_us + round(k * 1e6 / sample_rate) + external_offset_us - origin_us",
        "origin_us": origin_us,
        "audio_start_us": audio_start_us,
        "external_offset_us": offset_us,
        "sample_rate": sample_rate,
        "first_sample_session_us": first,
        "last_sample_session_us": first + last_offset,
    }


# ---------------------------------------------------------------- files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_inside(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise AudioError("the source file is outside the allowed folder")


def _check_regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise AudioError(f"{label} is a symbolic link; links are refused")
    if not path.is_file():
        raise AudioError(f"{label} was not found")


def expected_pcm_bytes(duration_s: float, sample_rate: int, channels: int) -> int:
    """Size of the decoded 16-bit PCM data for a stream: samples x channels x 2 bytes."""
    return math.ceil(duration_s * sample_rate) * channels * SAMPLE_WIDTH


def _decode(src: Path, stream: AudioStream, workdir: Path, max_bytes: int | None = None) -> Path:
    """Decode one stream to PCM WAV inside workdir, stopping if the output grows past max_bytes.

    The output size is checked while FFmpeg runs, so a source that decodes far larger than its
    metadata claims is stopped early. The caller deletes workdir.
    """
    limit = (MAX_DECODED_PCM_BYTES + WAV_HEADER_ALLOWANCE) if max_bytes is None else max_bytes
    out = workdir / "decoded.wav"
    args = [_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-map", f"0:{stream.index}", "-vn", "-sn", "-dn", "-c:a", SAMPLE_FORMAT, "-f", "wav", str(out)]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + DECODE_TIMEOUT_S
    stderr = b""
    try:
        while True:
            try:
                _, stderr = proc.communicate(timeout=DECODE_POLL_S)
                break
            except subprocess.TimeoutExpired:
                if out.exists() and out.stat().st_size > limit:
                    raise AudioError(f"decoded audio of {src.name} exceeds the size limit; decoding was stopped")
                if time.monotonic() > deadline:
                    raise AudioError(f"decode {src.name}: timed out after {DECODE_TIMEOUT_S} s")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    if proc.returncode != 0:
        text = (stderr or b"").decode("utf-8", "replace").replace(str(src), src.name)
        last = text.strip().splitlines()[-1:] or ["no message"]
        raise AudioError(f"decode {src.name}: exit code {proc.returncode}: {last[0][:300]}")
    if out.exists() and out.stat().st_size > limit:
        raise AudioError(f"decoded audio of {src.name} exceeds the size limit")
    return out


def _measure_wav(path: Path, stream: AudioStream) -> dict:
    """Sample count, rate, channels and a checksum of the PCM data, read in chunks."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if width != 2:
            raise AudioError(f"decoded {path.name} is not 16-bit PCM")
        if rate != stream.sample_rate or channels != stream.channels:
            raise AudioError("decoded audio does not match the probed sample rate or channel count")
        if frames < 1:
            raise AudioError("no audio samples were decoded from the selected stream")
        digest = hashlib.sha256()
        w.rewind()
        while True:
            chunk = w.readframes(COPY_CHUNK // max(channels * width, 1))
            if not chunk:
                break
            digest.update(chunk)
    return {"sample_rate": rate, "channels": channels, "sample_count": frames, "pcm_sha256": digest.hexdigest()}


def _copy_verified(src: Path, dest: Path, expected_sha: str) -> None:
    """Copy to a temporary file beside dest, verify the checksum, then replace dest in one step."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as out, open(src, "rb") as inp:
            shutil.copyfileobj(inp, out, COPY_CHUNK)
        if _sha256_file(Path(tmp_name)) != expected_sha:
            raise AudioError("the retained copy does not match the source checksum")
        os.replace(tmp_name, dest)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ---------------------------------------------------------------- session lock


def _pid_alive(pid: int) -> bool:
    """True if the process exists. Uncertain answers count as alive, so a lock is never stolen by mistake."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: exists but protected; otherwise gone
        try:
            # A handle can still open on a process that has exited while another handle to it is held,
            # so the exit code decides: STILL_ACTIVE (259) means running.
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_stale(path: Path) -> bool:
    """A lock is stale only when this machine wrote it and its owning process no longer exists.

    Unreadable or foreign locks are never treated as stale: they may belong to a run in progress.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if record.get("host") != socket.gethostname() or not isinstance(record.get("pid"), int):
        return False
    return not _pid_alive(record["pid"])


def _read_token(path: Path) -> str | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("token")
    except (OSError, ValueError, AttributeError):
        return None


def _busy_message(path: Path) -> str:
    """Explain a refusal. A stale owner is reported with its details and the explicit recovery step."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and _lock_is_stale(path):
        return (f"audio acquisition for this session was stopped without releasing its lock "
                f"(process {record.get('pid')} on this machine, created {record.get('created_utc', 'unknown')}, "
                f"is no longer running). Confirm no acquisition is running, then clear it with: "
                f"build_audio_session.py <person> <session> --clear-stale-lock")
    return ("another audio acquisition is running for this session, or its lock belongs to another machine "
            "or cannot be read. The lock was not changed.")


def clear_stale_lock(audio_dir: Path) -> dict:
    """Explicit operator recovery. Removes the lock only if it is verifiably stale right now.

    Never called automatically. The run must be started by a person, after checking that no
    acquisition is running. Returns the owner record that was removed.
    """
    path = Path(audio_dir) / LOCK_NAME
    if not path.exists():
        raise AudioError("there is no audio lock for this session")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AudioError("the lock cannot be read, so it will not be cleared") from exc
    if not _lock_is_stale(path):
        raise AudioError("the lock's owner is still running, or the lock belongs to another machine; "
                         "it was not cleared")
    os.remove(path)
    return record


@contextmanager
def session_lock(audio_dir: Path):
    """Exclusive lock on one session's audio folder, held for the whole acquisition.

    Created with O_EXCL, so only one creator can succeed. A lock left by a process on this machine
    that no longer exists is removed and replaced. Release removes the lock only if it still carries
    this run's token, so a run never deletes another run's lock.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / LOCK_NAME
    token = uuid.uuid4().hex
    record = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "token": token,
                         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Never removed here. Two processes that both saw a stale lock could otherwise race: one
        # takes it over, the other deletes the new lock, and two holders run at once.
        raise AudioError(_busy_message(path)) from None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(record)
    try:
        yield path
    finally:
        if _read_token(path) == token:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------- main entry point


def acquire_session_audio(
    person: str,
    session: str,
    *,
    external: Path | None = None,
    stream: int | None = None,
    offset_us: int = 0,
    retain: bool = False,
    force: bool = False,
    recordings_root: Path = RECORDINGS_RAW,
    datasets_root: Path = DATASETS_ROOT,
) -> dict:
    """Acquire one session's audio under the session lock and write audio/manifest.json.

    Returns the manifest. Nothing is written unless every check and the decode succeed.
    """
    if not is_valid_person(person) or not is_valid_session(session):
        raise AudioError("invalid person or session name")
    audio_dir = Path(datasets_root) / person / session / AUDIO_DIR_NAME
    with session_lock(audio_dir):
        return _acquire_locked(
            person, session, external=external, stream=stream, offset_us=offset_us,
            retain=retain, force=force, recordings_root=recordings_root, datasets_root=datasets_root,
        )


def _acquire_locked(
    person: str,
    session: str,
    *,
    external: Path | None = None,
    stream: int | None = None,
    offset_us: int = 0,
    retain: bool = False,
    force: bool = False,
    recordings_root: Path = RECORDINGS_RAW,
    datasets_root: Path = DATASETS_ROOT,
) -> dict:
    """Acquire one session's audio and write audio/manifest.json. Returns the manifest.

    Nothing is written unless every check and the decode succeed. The manifest is written last.
    """
    if not is_valid_person(person) or not is_valid_session(session):
        raise AudioError("invalid person or session name")

    if external is None:
        if offset_us != 0:
            raise AudioError("an offset applies only to external audio; embedded audio shares the video's clock")
        if retain:
            raise AudioError("retention applies only to external audio; embedded audio is never copied")
        source_path = Path(recordings_root) / person / f"{session}.mp4"
        _check_regular_file(source_path, f"{session}.mp4")
        _require_inside(source_path, Path(recordings_root))
        source_type = "video_embedded"
    else:
        source_path = Path(external)
        _check_regular_file(source_path, "the external audio file")
        if source_path.suffix.lower() not in EXTERNAL_SUFFIXES.values():
            raise AudioError("external audio must be a .wav or .mp3 file")
        if source_path.stat().st_size > MAX_EXTERNAL_BYTES:
            raise AudioError("the external audio file is larger than the 2 GB limit")
        source_type = "external"
    source_id = uuid.uuid4().hex  # opaque provenance identifier; the file name is never written

    out_dir = Path(datasets_root) / person / session / AUDIO_DIR_NAME
    manifest_path = out_dir / MANIFEST_NAME
    check_writable([manifest_path], force)

    probe_data = probe(source_path)
    fmt = probe_data.get("format", {})
    container = str(fmt.get("format_name", ""))
    if source_type == "external":
        expected = {name for name, suffix in EXTERNAL_SUFFIXES.items() if suffix == source_path.suffix.lower()}
        if not expected or not set(container.split(",")) & expected:
            raise AudioError("the file's contents do not match its .wav or .mp3 extension")

    duration_s = fmt.get("duration")
    duration_s = float(duration_s) if duration_s not in (None, "N/A") else None
    if duration_s is not None and duration_s > MAX_DURATION_S:
        raise AudioError("the audio is longer than the 4 hour limit")

    streams = parse_audio_streams(probe_data)
    chosen = select_stream(streams, stream)
    if chosen.duration_s is None and duration_s is None:
        raise AudioError("the duration of the audio is unknown")
    # Refuse before decoding: the decoded size follows from the duration, rate, channels and sample width.
    planned_s = duration_s if duration_s is not None else chosen.duration_s
    planned_bytes = expected_pcm_bytes(planned_s, chosen.sample_rate, chosen.channels)
    if planned_bytes > MAX_DECODED_PCM_BYTES:
        raise AudioError(f"decoded audio would be {planned_bytes} bytes, above the "
                         f"{MAX_DECODED_PCM_BYTES} byte limit; nothing was decoded")

    video = first_video_stream(probe_data) if source_type == "video_embedded" else None
    video_start_us = None
    if video is not None:
        video_start_us = _seconds_to_us(video.get("start_time"))
        if video_start_us is None:
            raise AudioError("the video's start time is unknown; the audio cannot be placed on the session clock")

    source_sha = _sha256_file(source_path)

    with tempfile.TemporaryDirectory(prefix="audio_work_") as work:
        decoded = _decode(source_path, chosen, Path(work))
        measured = _measure_wav(decoded, chosen)
        actual_bytes = measured["sample_count"] * measured["channels"] * SAMPLE_WIDTH
        if actual_bytes > MAX_DECODED_PCM_BYTES:
            raise AudioError(f"decoded audio is {actual_bytes} bytes, above the {MAX_DECODED_PCM_BYTES} byte limit")
    # The decoded copy has been deleted here, on success and on every failure path above.

    expected_s = duration_s if duration_s is not None else chosen.duration_s
    decoded_s = measured["sample_count"] / measured["sample_rate"]
    difference_ms = round((decoded_s - expected_s) * 1000)
    if abs(difference_ms) > DURATION_TOLERANCE_MS:
        raise AudioError(f"decoded length differs from the container by {difference_ms} ms")

    origin_us = video_start_us or 0
    audio_start_us = chosen.start_us if chosen.start_us is not None else 0
    timeline = session_timeline(
        audio_start_us=audio_start_us,
        sample_rate=measured["sample_rate"],
        sample_count=measured["sample_count"],
        offset_us=offset_us,
        origin_us=origin_us,
    )
    timeline.update({
        "origin": "video_first_presentation" if video is not None else "source_start",
        "video_start_us": video_start_us,
        "audio_start_reported": chosen.start_us is not None,
        "audio_minus_video_us": (audio_start_us - video_start_us) if video_start_us is not None else None,
    })

    retention = {"retained": False, "stored_file": None}
    if source_type == "video_embedded":
        retention.update(policy="video_not_copied", reanalysis="from the original recording")
    elif retain:
        stored_name = f"{source_id}{source_path.suffix.lower()}"
        _copy_verified(source_path, out_dir / SOURCE_DIR_NAME / stored_name, source_sha)
        retention.update(
            policy="external_retained", retained=True,
            stored_file=f"{AUDIO_DIR_NAME}/{SOURCE_DIR_NAME}/{stored_name}",
            reanalysis="from the retained copy",
        )
    else:
        retention.update(policy="external_not_retained",
                         reanalysis="requires the file to be supplied again")

    manifest = {
        "schema": SCHEMA,
        "person": person,
        "session": session,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "type": source_type,
            "source_id": source_id,
            "extension": source_path.suffix.lower(),
            "container_format": container,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha,
        },
        "streams": {
            "selected": chosen.as_dict(),
            "all": [s.as_dict() for s in streams],
        },
        "video": None if video is None else {"stream_start_us": video_start_us},
        "decoded": {
            **measured,
            "sample_format": SAMPLE_FORMAT,
            "duration_ms": round(decoded_s * 1000),
        },
        "duration_check": {
            "container_duration_ms": round(expected_s * 1000),
            "difference_ms": difference_ms,
            "tolerance_ms": DURATION_TOLERANCE_MS,
            "status": "pass",
        },
        "timeline": timeline,
        "retention": retention,
        "tool": {"ffmpeg": ffmpeg_version()},
    }

    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
