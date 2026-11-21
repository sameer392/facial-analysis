"""Command-line entry point: audio preprocessing, timing, energy, pitch and baseline VAD (phase H9.2).

Usage:
    python src/extraction/build_audio_features.py <person> <session>
        [--external FILE.wav|FILE.mp3] [--regenerate]
        [--analysis-rate HZ] [--window-ms MS] [--hop-ms MS]
        [--pitch-min-hz HZ] [--pitch-max-hz HZ] [--pitch-threshold X]
        [--silence-db DB] [--vad-db DB]
        [--vad-min-speech-ms MS] [--vad-min-silence-ms MS] [--vad-merge-gap-ms MS]

Requires the H9.1 acquisition (build_audio_session.py) to have run for the session. This command
never runs acquisition itself. An external audio file that H9.1 did not retain must be supplied again
with --external; its SHA-256 must match the H9.1 manifest.

Exit codes:
    0  outputs published
    1  refused or failed (missing or invalid input, existing outputs without --regenerate, processing
       error); nothing is published
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction.audio_acquisition import AudioError  # noqa: E402
from extraction.audio_processing import process_session_audio  # noqa: E402
from extraction.audio_signal import AudioConfig, SignalError  # noqa: E402
from file_safety import OutputExistsError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    d = AudioConfig()
    parser = argparse.ArgumentParser(description="Audio features for one session (H9.2).")
    parser.add_argument("person")
    parser.add_argument("session")
    parser.add_argument("--external", type=Path, default=None)
    parser.add_argument("--regenerate", action="store_true", help="replace existing H9.2 outputs")
    parser.add_argument("--analysis-rate", type=int, default=d.analysis_rate)
    parser.add_argument("--window-ms", type=float, default=d.window_ms)
    parser.add_argument("--hop-ms", type=float, default=d.hop_ms)
    parser.add_argument("--pitch-min-hz", type=float, default=d.pitch_min_hz)
    parser.add_argument("--pitch-max-hz", type=float, default=d.pitch_max_hz)
    parser.add_argument("--pitch-threshold", type=float, default=d.pitch_threshold)
    parser.add_argument("--silence-db", type=float, default=d.silence_db)
    parser.add_argument("--vad-db", type=float, default=d.vad_db)
    parser.add_argument("--vad-min-speech-ms", type=float, default=d.vad_min_speech_ms)
    parser.add_argument("--vad-min-silence-ms", type=float, default=d.vad_min_silence_ms)
    parser.add_argument("--vad-merge-gap-ms", type=float, default=d.vad_merge_gap_ms)
    args = parser.parse_args(argv)

    try:
        cfg = AudioConfig(
            analysis_rate=args.analysis_rate, window_ms=args.window_ms, hop_ms=args.hop_ms,
            pitch_min_hz=args.pitch_min_hz, pitch_max_hz=args.pitch_max_hz,
            pitch_threshold=args.pitch_threshold, silence_db=args.silence_db, vad_db=args.vad_db,
            vad_min_speech_ms=args.vad_min_speech_ms, vad_min_silence_ms=args.vad_min_silence_ms,
            vad_merge_gap_ms=args.vad_merge_gap_ms,
        ).validate()
        result = process_session_audio(args.person, args.session, config=cfg, external=args.external,
                                       regenerate=args.regenerate)
    except (AudioError, OutputExistsError, SignalError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    print(f"audio features published: frames={result['frames']} "
          f"analysis_rate={cfg.analysis_rate} window_ms={cfg.window_ms} hop_ms={cfg.hop_ms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
