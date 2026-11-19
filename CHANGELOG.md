# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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
