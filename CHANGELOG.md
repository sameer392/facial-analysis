# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- H9.1 audio acquisition failed for sources with more than two channels: FFmpeg writes a WAVE_FORMAT_EXTENSIBLE header, which the standard `wave` module cannot read. Decoding now produces raw 16-bit PCM and the decoded bytes are measured directly. Mono and stereo SHA-256 values are unchanged. Regression test added.

### Removed
- Streamlit dashboard (phase P7, approved in writing). `src/dashboard/app.py`, `common.py`, `dashboard_bootstrap.py`, `pages/` and `src/analysis/visualization.py` are deleted. Streamlit and Plotly are removed from `requirements.txt`. The Next.js dashboard with the Python API replaces it.
- `src/dashboard/` renamed to `src/viewer/`: `overlay.py` and `pose_viewer.py` are the framework-free helpers the API uses.

### Added
- Transcription interface and dataset H9.3a: an engine-independent `AsrEngine` adapter (`src/extraction/asr_interface.py`) with validation of every engine response, and a transcription dataset (`src/extraction/transcription.py`) under `audio/transcription/`. Raw engine text is kept unchanged beside a normalised text. Timing uses integer samples on the H9.1 timeline, with negative session times valid. Source checksum and H9.2 outputs are verified before use; publication is staged, read back, atomic and regeneration-safe. Tested with a deterministic fake engine only; no ASR runtime, model or accuracy claim.
- Audio acquisition H9.1: `src/extraction/build_audio_session.py` reads a session's audio from the recording's first audio stream, or from a WAV or MP3 file (`--external`), and writes `audio/manifest.json`. Original media is never modified. Per-session lock with a stale-lock policy that never removes a lock automatically (`--clear-stale-lock` is explicit). Limits: 2 GB source, 4 hours, 3 GiB decoded PCM. External audio is kept only with `--retain`; embedded audio is never copied.
- Audio preprocessing and basic features H9.2: `src/extraction/build_audio_features.py` resamples to a configurable analysis rate (default 16 kHz) with an equal-weight mean downmix, and writes 40 ms / 10 ms frame energy, an experimental pitch estimate (60 to 400 Hz), and an energy-based VAD baseline under `audio/features/`, `audio/vad/` and `audio/quality/`. Integer-microsecond timing from the H9.1 timeline. Staged, validated and atomically published. Synthetic tests only; no speech accuracy claimed. Documented in the README and in `src/extraction/audio_processing.py`.
- Video playback H8.2: secure video streaming and landmark windows. `GET .../video` streams the original recording with HTTP Range support. `GET .../landmarks/window` returns 1 to 30 frames of face, pose and hand landmarks in the HLW1 binary format (version 1). `GET .../video/timing` compares decoded frame times with the extracted timeline. All three are read-only, accept only local requests (Host and Sec-Fetch-Site checks), and refuse path traversal and symlink escapes.
- Video playback H8.3: native HTML5 video player on the Hands page, with play, pause, seeking, speed (0.25x to 2x), mute, volume, and frame-by-frame stepping. Frame review remains a separate mode.
- Video playback H8.4: facial, pose and left and right hand landmarks drawn on a transparent canvas over the video, synchronised with each presented frame by timestamp (never `currentTime x fps`). Per-stream visibility controls. Overlays are enabled only when the timing check reports `synchronized`. The HLW1 parser (`web/app/lib/hlw1.ts`) rejects malformed windows.
- Video playback H8.5: hardening. Eight-window landmark cache (at most 1.1 MB). No prefetch of windows past the last frame. The startup log names the token location in words, without the file path. Browser CPU, memory and rendering performance are not measured; a persistent timing-validation cache is not implemented.
- Hand Motion Analysis, phase H1 (branch `hand-motion`): hand detection and 21-point landmark extraction with MediaPipe Hand Landmarker, normalised and world coordinates kept separately, and a per-frame hand status file. Written into the hands stream of the pose data session by `src/extraction/build_hand_session.py`. Tracking, gestures, API and frontend are not yet implemented.
- Hand Motion Analysis, phase H2: temporal hand tracking (track IDs with interruption, reacquisition and termination events, uncertainty flags for label conflicts, close hands and ambiguous candidates) and image-space motion metrics (centroid, wrist and fingertip positions, displacement, velocity, acceleration, direction, joint angles, openness, orientation). Written to `hands/tracks.csv` and `hands/motion_metrics.csv` by `src/extraction/analyse_hand_session.py`. Units are image pixels, not physical units.
- Hand Motion Analysis, phase H3: rule-based gesture recognition (open palm, closed fist, pointing, thumbs up, peace, unknown) from hand-relative finger states, with noise-tolerant gesture events per track. Written to `hands/gesture_observations.csv`, `hands/gesture_events.csv` and the gesture summary by `src/extraction/classify_hand_gestures.py`. Thresholds are not calibrated on real footage; the geometric margin is not a probability.
- Hand Motion H4: read-only dataset validation (`src/extraction/validate_hand_session.py`) across H1, H2 and H3 outputs, with errors, warnings and quality metrics; manual annotation format and validation (`hand_annotation.py`); evaluation of H3 predictions against annotations with per-class scores, confusion matrix, unknown rate and visibility split (`hand_evaluation.py`). No calibration was performed: no labelled data exists yet.
- Hand Motion H5: `hand_build` job kind (runs the H1 to H3 pipeline and the read-only H4 validation through `src/extraction/build_hand_pipeline.py`), and read-only API routes under `/api/persons/{p}/sessions/{s}/hands/` (summary, landmarks, tracks, motion, gesture observations and events, quality, validation, annotated frame). Tables are paged; the annotation folder is git-ignored.
- Hand Motion H6: Hands page (`/hands`) with session selection, `hand_build` job control with per-stage state, frame viewer, motion charts, gesture filters and tables, dataset quality, on-demand validation. Browser check not yet done.
- Hand Motion H7 (hardening): all cached hand data, including the validation result, is refreshed when a hand job ends, so a regeneration cannot show results from the earlier dataset. README: hand motion section and corrected status.
- Compare people page (`/compare`): compare sessions from several people. `GET /api/comparison?items=person/session,...`.
- Interruptions chart and blink timeline on the Face analysis page.
- Blendshape values for the selected frame, movement by region, blendshape CSV download, landmark coordinate tables, and the 3D coordinates table.
- Rejected pose data folders with reasons on the Body Motion page.
- Blendshape means and landmark stability in the dataset comparison.
- Next.js Jobs page (`/jobs`): start a V2 build and cancel jobs, with the API token kept for the browser tab.
- Upload a recording from the Recordings page (choose the person) as well as from a person's page. One shared component.
- API token: kept in `config/api_token.txt` (same across restarts) and entered once on the Settings page. Other pages no longer show a token box.
- Live job progress: `GET /api/jobs/{id}/events` (Server-Sent Events, read only). The Jobs page updates each active job as its progress changes.
- Upload shows a progress bar (the upload is sent with XMLHttpRequest so progress can be reported).
- Rename a person or a recording from its row. Refused while processed data exists; delete the data, rename, then rebuild.
- Manage page: create people, upload MP4 recordings, delete recording data, videos or whole people with typed confirmation, and clear finished job history. Every action needs the API token.
- Face data is one job: extraction and analysis run together (`face_build`, `src/analysis/build_face_data.py`). The Jobs page offers Face data and Pose data. The separate extraction and analysis job kinds remain in the API.
- Naming cleanup: layout codes are `face`, `pose` and `pose_incomplete`; `--regenerate-v2` is `--regenerate`; the write, rollback and recovery functions, the constants and the exception names use face and pose names. The stored schema string `2.0` is unchanged.
- Naming migration, step 5: the older API paths (`/v1/…`, `/v2/…`), job codes, `.v2-` lock and temporary names, and the old class and module names are removed. Job rows saved with the older codes are updated when the job store opens.
- Naming migration, step 4: test files and test names use the new terms (for example `test_pose_writer.py`); product docs (README, AGENTS.md, dataset layout, validation) use feature names, with code names in backticks unchanged.
- Naming migration, step 3: the pose build script prints "pose data" wording, and the job parser reads the new markers only.
- Naming migration, step 2: feature paths (`/face/…`, `/pose/…`) and job codes (`pose_build`, `face_extract`, `face_analyse`) beside the older ones; sessions carry a `layout_name` display field. Older paths and codes still work. Error codes renamed to `no_face_data` and `face_data_exists`.
- Next.js migration P6: Audio and Hands placeholder pages, driven by a read-only layer status route.
- Next.js migration P5: Models, Logs and Settings pages and read-only API routes (model hash status, redacted job logs, settings without the token).
- Next.js migration P4: V1 face extraction and V1 analysis job kinds, per-frame progress for extraction, and a kind selector on the Jobs page.
- Next.js migration P4 (backend): job system for V2 builds with token-protected create and cancel routes, one active job per session, and restart handling that reports leftovers. Tested in `tests/test_api_p4.py`.
- Next.js dashboard P3: 3D facial page (head pose, depth, landmark projection) and Body Motion page (overlay from the stored video, detection timeline, joint table, quality summary). Read-only routes in `api/v2_data.py`, tested in `tests/test_api_p3.py`.
- Next.js dashboard (`web/`) P2 pages: facial analysis (tracking quality, motion, blendshapes,
  head rotation, blinks), diagnostics (face detection, neutral frames, landmarks for one frame)
  and dataset comparison across sessions. Charts use ECharts 6.1.0.
- Read-only API routes for version 1 data (`api/v1_data.py`): summary, series, blinks,
  landmarks and comparison, with parity tests against `facial_analyzer` (`tests/test_api_p2.py`).
- Apache-2.0 license, third-party notices and contributor guide.
- Synthetic demo data generator (`src/analysis/demo_data.py`) and dashboard screenshots built from it.
- `src/analysis/file_safety.py`: overwrite checks and atomic file writes.
- `src/analysis/validation.py`: landmark-count, coordinate and timestamp checks.
- `tests/test_fixtures.py`: tests on temporary synthetic datasets (JSON, CSV, validation,
  timestamps, dataset loading, overwrite protection, dataset immutability).
- `--force` option for `analyze_faces.py` and `facial_analyzer.py`.
- `FACIAL_DATA_HOME` environment variable to point the project at another data folder.
- GitHub Actions test workflows for Windows and Ubuntu on Python 3.11.
- Issue templates for bug reports and feature requests.

### Changed
- Recordings are stored as `recordings/raw/<person>/session-NNN.mp4`; datasets as
  `datasets/<person>/session-NNN/`.
- Dashboard: create and select people or projects, add or process recordings, delete
  session datasets, raw videos or whole people.
- Dashboard reloads project modules on each run, so helper edits show without a restart.
- Extraction writes `facial_data.json` atomically and refuses to overwrite a session that
  already has a dataset or frame images.
- `requirements.txt` installs `opencv-contrib-python` only (the build MediaPipe depends on).
  `opencv-python` is no longer installed alongside it.
- Personal names in code, tests and documentation replaced with `demo_person`.

### Fixed
- `opencv-python` and `opencv-contrib-python` were installed together, which conflicts.
- Analysis and extraction could leave partially written files if interrupted.
- Head-pose compensation used distorted normalised coordinates and no axis flip. It was worse
  than no compensation in two of three local sessions. It now converts to pixel-proportional
  units, uses the nearest proper rotation of the matrix, applies the axis flip, and normalises scale.
  See `docs/validation.md`.
- Acceleration divided by one time step, which is wrong for irregular timestamps. It now uses
  the time between velocity midpoints.
- Synthetic demo landmarks rotated in 2D while the matrix recorded 3D rotation. The demo now
  uses one consistent model.

### Added (Next.js migration, phase P1: read-only web shell)
- `web/`: Next.js 16 (App Router) with TypeScript and Tailwind CSS 4. Pages: Overview (API status), People, a person's sessions, and Recordings across people. Each page handles loading, empty and error states.
- The browser calls `/api/...` on the Next.js server, which proxies only `/api` to the local P0 API (127.0.0.1). The API itself is not opened to browsers, so no CORS is enabled.
- Components are hand-written in Tailwind for now (the shadcn/ui CLI is not used yet).
- CI: a Ubuntu job runs `npm ci`, `npm run typecheck` and `npm run build` for `web/`.
- Read-only: no uploads, jobs or deletes yet.

### Added (Next.js migration, phase P0: read-only API)
- `api/main.py` and `api/catalog.py`: `GET /api/health`, `GET /api/persons`, and `GET /api/persons/{person}/sessions`. Read-only; names are validated; errors have a code and status; no absolute paths are returned.
- `fastapi`, `uvicorn` and `pydantic` pinned in `requirements.txt`.
- Tests: `tests/test_api_p0.py` (11 synthetic tests, including a check that no request writes to disk).
- Migration proposal updated: Streamlit is removed completely at phase P7.

### Added (Phase 2 pose workflow, partially implemented)
- Shared video frame reader (`src/extraction/video_frames.py`), strict about truncated or empty videos.
- Model registry with SHA-256 verification before any model is loaded (`src/extraction/model_registry.py`).
- Pose Landmarker extractor with an injected detector and NaN rows for missing detections
  (`src/extraction/pose_extractor.py`). Single subject only.
- Builder that combines V1 facial data with the Pose output into a V2 session
  (`src/extraction/v2_session_builder.py`). Hands are recorded as an absent stream.
- `docs/dataset_layout.md`: V1 and V2 layouts, arrays, metadata and licence notes.
- CI and the documented test command now use test discovery (`tests/test_*.py`).
- Not validated on real recordings. The dashboard does not run this workflow.
- Command-line entry point `src/extraction/build_v2_session.py <person> <session>`: refuses bad
  input before touching the model, verifies the Pose model's SHA-256, refuses an existing V2
  dataset unless `--regenerate-v2` is given, always closes the detector, and checks that V1 files
  are unchanged. Tested with synthetic data and a fake detector.
- If a V1 file changes during a CLI run, the run's V2 output is rolled back with `rollback_v2`
  (exit code 4). The previous V2 dataset is restored if there was one. A refused or failed rollback
  is reported, and the files are left in place.

### Added (landmark overlay)
- Dashboard landmark overlay modes: Hybrid (default), Wireframe, Contours, Points and Off, with
  opacity, line-thickness and point-size controls. Implemented in `src/dashboard/overlay.py`.
- Connection topology taken from MediaPipe's official `FaceLandmarksConnections`, read once.
- `tests/test_overlay.py`: synthetic tests for topology, pixel mapping, each mode and invalid input.

### Added (validation)
- `tests/test_scientific.py`: known rotations, translations, scale, missing matrices, aspect
  ratio, axis flip, neutral calibration, and motion metrics on regular and irregular timestamps.
- `docs/validation.md`: conventions, formulas, the empirical convention check, and limitations.
- Dashboard page `06_Diagnostics`: tracking checks, interruptions, neutral-frame selection,
  raw versus compensated landmarks, head rotation, displacement, velocity and acceleration.
- `tracking_quality.json` reports interruptions, landmark completeness, matrix availability on
  detected frames, and timestamp issues. It states that these are tracking checks only.

### Not yet implemented
- Audio feature extraction (dashboard page is a placeholder).
- Body pose extraction (dashboard page is a placeholder).
- Neural network training and inference (`src/training/`, `src/inference/` are empty).

## Before this changelog

The project was developed locally before publication: MediaPipe face landmark extraction,
blendshape and transformation-matrix export to JSON and CSV, motion analysis, and a Streamlit
dashboard for a single presenter. No tagged release exists yet.

- Body Motion dashboard page: read-only viewer for existing V2 Pose sessions, with the pose
  overlay on the video frame, a visibility threshold that does not change stored values, a
  detection timeline from timeline.csv, a joint table, and tracking-quality figures. Discovery lists
  only V2 sessions that pass the metadata and manifest checks (`src/dashboard/v2_viewer.py`).
  Tested with synthetic data (22 tests). Playback is not implemented.
- Pose visibility gate: joints outside the frame with visibility below 0.5 are treated as not visible for analysis (see docs/validation.md section 9). The stored arrays are unchanged.

