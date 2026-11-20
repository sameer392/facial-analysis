"""Command-line entry point: acquire a session's audio and write audio/manifest.json (phase H9.1).

Usage:
    python src/extraction/build_audio_session.py <person> <session>
        [--stream INDEX] [--external FILE.wav|FILE.mp3] [--offset-us N] [--retain] [--force]
    python src/extraction/build_audio_session.py <person> <session> --clear-stale-lock

By default the audio of recordings/raw/<person>/<session>.mp4 is used. --external uses a separately
supplied WAV or MP3 file instead; --offset-us then places it on the session clock, and --retain
keeps a verified copy under audio/source/. Without --retain, reanalysis needs the file again.

Exit codes:
    0  manifest written
    1  refused or failed (missing or invalid input, decode error, existing manifest without --force)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extraction.audio_acquisition import AUDIO_DIR_NAME, LOCK_NAME, AudioError, acquire_session_audio, clear_stale_lock  # noqa: E402
from file_safety import OutputExistsError  # noqa: E402
from project_paths import DATASETS_ROOT, is_valid_person, is_valid_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire one session's audio (H9.1).")
    parser.add_argument("person")
    parser.add_argument("session")
    parser.add_argument("--stream", type=int, default=None, help="container index of the audio stream to use")
    parser.add_argument("--external", type=Path, default=None, help="a WAV or MP3 file to use instead of the video")
    parser.add_argument("--offset-us", type=int, default=0, help="external audio offset on the session clock")
    parser.add_argument("--retain", action="store_true", help="keep a verified copy of external audio")
    parser.add_argument("--force", action="store_true", help="replace an existing audio manifest")
    parser.add_argument("--clear-stale-lock", action="store_true",
                        help="remove the session's audio lock, only if its owner process has stopped "
                             "(run only when you are sure no acquisition is running)")
    args = parser.parse_args(argv)

    if args.clear_stale_lock:
        if not is_valid_person(args.person) or not is_valid_session(args.session):
            print("refused: invalid person or session name", file=sys.stderr)
            return 1
        try:
            owner = clear_stale_lock(DATASETS_ROOT / args.person / args.session / AUDIO_DIR_NAME)
        except AudioError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print(f"stale lock cleared (process {owner.get('pid')} on this machine, {LOCK_NAME} removed)")
        return 0

    try:
        manifest = acquire_session_audio(
            args.person, args.session,
            external=args.external, stream=args.stream, offset_us=args.offset_us,
            retain=args.retain, force=args.force,
        )
    except (AudioError, OutputExistsError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    decoded = manifest["decoded"]
    print(f"audio manifest written: source={manifest['source']['type']} "
          f"stream={manifest['streams']['selected']['index']} "
          f"rate={decoded['sample_rate']} Hz channels={decoded['channels']} "
          f"samples={decoded['sample_count']} duration_check={manifest['duration_check']['difference_ms']} ms")
    print(f"retention: {manifest['retention']['policy']} ({manifest['retention']['reanalysis']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
