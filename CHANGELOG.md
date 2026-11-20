# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Next.js Jobs page (`/jobs`): start a V2 build and cancel jobs, with the API token kept for the browser tab.
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

