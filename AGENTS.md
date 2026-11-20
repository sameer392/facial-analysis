# AGENTS.md â€” Facial Analysis Project

This is the canonical source of shared project instructions for every coding agent working in this repository (Claude Code and OpenAI Codex). `CLAUDE.md` only adds Claude Code-specific operating notes and defers to this file.

## 1. Project Overview

This repository develops a multimodal facial and body motion analysis and dataset platform for future audio-driven digital-human generation.

Long-term objectives:

- Accurate lip synchronization
- Facial expressions and blendshapes
- Eye, eyebrow and mouth movements
- Head pose and movement
- Body and upper-body motion
- Hand gestures
- Audio and phoneme alignment
- Validated datasets for future model training

Scientific accuracy, reproducibility, data integrity and backward compatibility take priority over implementation speed or feature count.

**Repository:** `sameer392/facial-analysis`
**Local path:** `C:\AI\facial-analysis`
**Primary branch:** `main`
**Python:** 3.11

## 2. Agent Responsibilities

Act as a senior Python engineer, computer vision engineer and software architect.

Before making changes:

1. Inspect the relevant source code.
2. Read related documentation.
3. Review existing tests.
4. Check Git status, the current branch and the remote state (section 15).
5. Understand dependencies and data flow.
6. Identify possible regressions.

Do not assume a feature is missing, or present, until the implementation has been inspected. README text, earlier conversations and this file can be out of date.

Prefer incremental, modular and testable changes.

## 3. Architecture

The intended platform consists of four layers. Keep them separate.

### Layer 1 â€” Acquisition and Extraction

- Video decoding and timestamps
- Facial landmarks, blendshapes and transformation matrices
- Body pose landmarks
- Hand landmarks and handedness
- Audio extraction and alignment

### Layer 2 â€” Motion Analysis

- Facial and mouth movement
- Eye and eyebrow movement
- Head pose
- Body movement
- Hand gestures
- Velocity and acceleration
- Tracking and detection quality

### Layer 3 â€” Dataset and Training

- Standardized schemas
- Session metadata
- Shared timelines
- Data validation
- Dataset comparison
- Training-ready exports

### Layer 4 â€” Inference and Digital Human Generation

- Audio-driven lip synchronization
- Facial animation
- Expression generation
- Head movement
- Body and hand gesture generation

Do not combine extraction, analysis, dataset management, training and inference into a single module.

## 4. Repository Structure

Important paths (a reference, not a guarantee; inspect the working tree):

```text
.devcontainer/
  devcontainer.json           development container (see section 10: not for public deployment)
.github/
  workflows/tests.yml         CI: Windows and Ubuntu, Python 3.11 (section 16)
docs/
  validation.md               scientific validation and known limitations
  proposals/
    phase2_multimodal_schema.md
  screenshots/                dashboard images (may be outdated; see below)
src/
  analysis/                   extraction, analysis, schema and writer modules
    analyze_faces.py          face data analysis entry point
    demo_data.py              synthetic demo data generator
    extract_faces.py          face data facial extractor
    facial_3d_analyzer.py
    facial_analyzer.py
    file_safety.py            overwrite checks and atomic writes
    motion_analyzer.py
    project_paths.py          folder layout, FACIAL_DATA_HOME
    pose_schema.py              pose data schema
    pose_writer.py      pose data writer, recovery and rollback
    validation.py
    visualization.py
  dashboard/
    app.py
    common.py
    dashboard_bootstrap.py
    pages/
      01_Facial_Analysis.py
      02_3D_Facial_Analysis.py
      03_Audio_Analysis.py    placeholder
      04_Body_Motion.py       placeholder
      05_Dataset_Comparison.py
      06_Diagnostics.py
  api/                        Next.js migration, phase P0 (read-only): main.py (FastAPI app),
                              catalog.py (health, persons, sessions). Planned, not implemented:
                              the web app, write endpoints, job system (docs/proposals/nextjs_migration.md)
  extraction/                 video_frames.py, model_registry.py, pose_extractor.py,
                              pose_builder.py, build_pose_session.py (Pose data command-line entry point)
                              (hand and audio extraction: not implemented)
  training/                   placeholders only (.gitkeep)
  inference/                  placeholders only (.gitkeep)
tests/                        see section 13
models/
  mediapipe/                  model binaries are local and git-ignored
packages.txt                  Linux system packages (section 9)
requirements.txt              Python packages (section 9)
THIRD_PARTY_NOTICES.md
CHANGELOG.md
CONTRIBUTING.md
LICENSE                       Apache-2.0
README.md
```

### Screenshots

`docs/screenshots/` holds images made from synthetic data at an earlier stage. They may not match current behaviour. Do not treat them as evidence that a feature works now. Regenerate them, or label them with the commit they were made from, before relying on them.

## 5. Dataset Layouts and Backward Compatibility

Session folders are `datasets/<person>/<session>/`. The root is set by `project_paths.py`, and can be redirected with the `FACIAL_DATA_HOME` environment variable (the folder must contain `recordings/` and `datasets/`). Recordings are read from `recordings/raw/<person>/<session>.mp4`.

### Face data layout (session root)

- `facial_data.json`
- `facial_blendshapes.csv`
- `motion_metrics.csv`
- `blink_events.csv`
- `tracking_quality.json`
- `calibration.json`
- `frames/`

### Pose data layout (subdirectories in the same session folder)

- `metadata.json` (completion marker, written last)
- `manifest.json`
- `timeline.csv`
- `facial/`, `body/`, `hands/`, `quality/`

Note that `tracking_quality.json` exists in both layouts at different paths: `<session>/tracking_quality.json` (Face data) and `<session>/quality/tracking_quality.json` (Pose data).

### Coexistence rules

- face data and pose data coexist in the same session folder.
- pose data generation, regeneration, migration and recovery must not silently overwrite, move or delete face data files, or the frames folder.
- A session is treated as pose data only when `metadata.json` declares schema 2.x. Otherwise it is face data. A folder with pose data pieces but no `metadata.json` is incomplete and must be treated as absent.

## 6. Implementation Status

Do not describe a phase as implemented when only some of its steps are.

### Phase 1 â€” Facial Analysis: implemented

- MediaPipe Face Landmarker, 478 landmarks, 52 blendshape scores, transformation matrices
- JSON and CSV exports
- Head-pose compensation, neutral calibration, displacement, velocity and acceleration
- Blink detection and tracking-quality metrics
- Dashboard (Streamlit, removed in phase P7; the Next.js dashboard replaces it)

### Phase 2 â€” Multimodal Dataset Infrastructure: partially implemented

- **Step 1, pose data schema: implemented** (`src/analysis/pose_schema.py`).
- **Step 2, pose data session writer: implemented** (`src/analysis/pose_writer.py`), with staging, manifest, lock, recovery and rollback (section 7).
- **Stage 2, shared video frame reader: implemented** (`src/extraction/video_frames.py`). Strict: missing, unopenable, empty or truncated videos raise errors. Timestamps are `fps_derived`; decoded presentation times are not extracted yet.
- **Stage 3, model registry: implemented** (`src/extraction/model_registry.py`). Verifies SHA-256 before any model is loaded. Records licence status as evidence, not clearance.
- **Stage 4, Pose Landmarker extractor: implemented** (`src/extraction/pose_extractor.py`). Single subject only; missing detections are NaN rows. Tested with a fake detector; a one-off smoke test ran the real model on a blank synthetic video only.
- **Stage 5, pose data session assembly: implemented** (`src/extraction/pose_builder.py`). Combines face data (read only) with the Pose output and writes through the pose data writer. Hands are recorded as an absent stream. Tested with synthetic data.
- **Pose data command-line entry point: implemented** (`src/extraction/build_pose_session.py`). Verifies the model's SHA-256, refuses an existing pose dataset unless `--regenerate` is given, always closes the detector, and checks that face data files are unchanged. If a face data file changes, it rolls back that run's pose data output and reports the outcome honestly. Tested with synthetic data and a fake detector only.
- **Overall phase: incomplete.** Still pending:
  - hand extraction (the hand stream is currently recorded as absent)
  - validation on real recordings (no real session has been written through the Pose workflow yet; no recording has been approved for this)
  - the face-to-pose migration command
  - the Hand part of the Body and Hand Motion dashboard page, and a dashboard button that generates pose data (the Body Motion page is a read-only viewer of existing pose data sessions)
  - PTS timestamp extraction (timestamps are currently FPS-derived)
  - model licence clearance for commercial use or redistribution (Pose FULL is partly verified; Face and Hand are unverified)

### Next.js migration status

- **Phase P0, read-only Python API: implemented** (`api/main.py`, `api/catalog.py`). Routes: `GET /api/health`, `GET /api/persons`, `GET /api/persons/{person}/sessions`. Names are validated, errors carry a code and status, and no request writes to disk. Run with `uvicorn api.main:app --host 127.0.0.1 --port 8000`. Tested with synthetic data (11 tests).
- **Phase P1, read-only web shell: implemented** (`web/`). Next.js 16, React 19, TypeScript 5.9, Tailwind CSS 4, TanStack Query and Lucide, all pinned exactly in `web/package.json`. Pages: Overview, People, a person's sessions, Recordings. The browser calls `/api`, which Next.js proxies to the local API only. Checked by `npm run typecheck`, `npm run build` (CI, Ubuntu) and a headless browser screenshot on local data. Components are hand-written; shadcn/ui is not yet used.
- **Not yet started:** write endpoints (uploads, jobs, deletes), the job system, and parity pages for the analysis views (phases P2 to P7 in the proposal).
- **Phase P2, analysis pages: implemented.** `api/face_data.py` adds read-only routes under `/api/persons/{person}/sessions/{session}/face/` (`summary`, `series`, `blinks`, `landmarks/{index}`) and `/api/persons/{person}/comparison`. They read the face data files with the same analysis code as the dashboard, and `tests/test_api_p2.py` checks parity. The web app adds Facial analysis, Diagnostics and Compare pages with ECharts 6.1.0 (`web/app/components/chart.tsx`). No route writes to disk. Checked by typecheck and build; a browser check on local data is still pending.
- **Phase P3, 3D and Body Motion: implemented.** `api/pose_data.py` adds read-only routes: `pose` (validation, cached on metadata mtime; invalid sessions return 422 with the reason), `pose/timeline`, `pose/frames?start&end&threshold` (at most 300 frames per request), `pose/frames/{index}/quality`, `pose/frames/{index}/overlay.jpg` (drawn with `pose_viewer.draw_pose_overlay`), and `face/head-pose` (yaw, pitch, roll and depth span from `facial_3d_analyzer`). Tests: `tests/test_api_p3.py`. The web app adds `3d` and `body` pages. The 3D page is a projection with depth as colour; a rotatable 3D view needs `echarts-gl`, which is not installed and needs approval.
- **Phase P4, job system: implemented.** Kinds: `face_build` (`src/analysis/build_face_data.py`: runs extraction, then analysis, in one job; progress is per frame for extraction, then per stage), `pose_build` (`src/extraction/build_pose_session.py`). The single-step face kinds were removed; old job records keep their names. One active job per session across all kinds. Workers run with unbuffered output. A cancelled or interrupted extraction can leave a partly written `frames/` folder; it is reported and never removed. Note: the extraction script itself loads the face model without a hash check when run by hand; only the job path checks it. `api/jobs.py` runs the pose data build CLI (`src/extraction/build_pose_session.py`) as a separate process and stores job records in `logs/jobs.sqlite3` (git-ignored). Routes: `POST /api/jobs` (202, token required; refusals before queueing: bad names, missing video or face data, existing pose data without `regenerate`, writer lock present, one active job per session), `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel` (token required). Startup marks unfinished jobs `interrupted` and lists leftover `.pose-staging-*`, `.pose-previous-*` and `.pose-lock` items; nothing is removed automatically. The API token is kept in `config/api_token.txt` (created on first start, git-ignored, the same across restarts), or set with `FACIAL_API_TOKEN`. It is entered once on the Settings page and kept in the browser; no other page asks for it. The web app sends the saved token as `X-Facial-Token` on write actions (`web/app/lib/token.ts`). Not yet built: SSE progress, and per-frame progress for the analysis job. Tests: `tests/test_api_p4.py` (fake worker, no model).
- **Phase P5, models, logs and settings: implemented (read-only).** `api/system.py` provides `GET /api/models` (SHA-256 status for each registry entry, cached on size and modification time; models are never loaded; licence text is shown exactly as recorded), `GET /api/logs/jobs` and `GET /api/logs/jobs/{id}?lines=` (job worker logs, with the project and home folders replaced by `<project>` and `<home>`), and `GET /api/settings` (names only, token status only). The web pages are `/models`, `/logs` and `/settings`. Tests: `tests/test_api_p5.py`.
- **Phase P6, Audio and Hands placeholders: implemented.** `GET /api/layers` reports whether `src/extraction/audio_extractor.py` and `src/extraction/hand_extractor.py` exist; the web pages `/audio` and `/hands` show Not implemented until they do, and say what is needed. Tests in `tests/test_api_p5.py`.
- **Manage (people, recordings, datasets, job history): implemented.** `api/manage.py` and the `/manage` page. Every action needs the API token. Routes: `POST /api/persons` (create), `POST /api/persons/{p}/delete`, `PUT /api/persons/{p}/sessions/{s}/video` (raw MP4 body, up to 2 GiB; the MP4 signature is checked, the file is written to a temporary name and moved into place, and an existing session is never overwritten), `POST .../sessions/{s}/video/delete`, `POST .../sessions/{s}/dataset/delete`, `POST /api/jobs/history/clear` (finished jobs and their logs only). Deletes need the exact name typed as `confirm` (for job history, `clear job history`). A delete or upload is refused while a job is active on the session. Tests: `tests/test_api_manage.py`. The test that checks write routes now requires every write route to need the token.
- **Live job progress: implemented.** `GET /api/jobs/{id}/events` streams each job record as Server-Sent Events when its state, stage or progress changes, and closes when the job is no longer active (`api/job_events.py`). Read only, no token. The Jobs page subscribes to each active job; polling remains as a fallback. Tests: `tests/test_job_events.py`.
- **Rename (person and recording): implemented.** `POST /api/persons/{p}/rename` and `POST /api/persons/{p}/sessions/{s}/rename` with body `{"to": "new-name"}`. Token required. Refused while processed data exists (it records the names), while a job runs, or when the new name is taken. Tests: `tests/test_api_manage.py`.
- **Naming migration, step 5: done.** API paths are `/face/…` (face data) and `/pose/…` (pose data). Job codes are `face_build` and `pose_build`; rows saved with the older pose code are updated when the job store opens. Modules are `pose_schema`, `pose_writer`, `pose_viewer`, `pose_builder`, `build_pose_session.py`, `face_data.py` and `pose_data.py`. Pose data lock and temporary names are `.pose-lock`, `.pose-staging-*` and `.pose-previous-*`. Session rows include `layout_name`. Tests: `tests/test_naming_migration.py`. Plan: `docs/proposals/naming_migration.md`.
- CI (`.github/workflows/tests.yml`) runs on pushes to `main` and `migration-nextjs`, and on pull requests.
- **Phase P7, Streamlit removed: done** (written sign-off given). `src/dashboard/app.py`, `common.py`, `dashboard_bootstrap.py`, `pages/` and `src/analysis/visualization.py` are deleted. `src/dashboard/overlay.py` and `pose_viewer.py` remain because the API uses them. Streamlit and Plotly are removed from `requirements.txt`.

### Other status

- Audio Analysis dashboard page: placeholder.
- Body Motion dashboard page: read-only viewer of existing pose data sessions (`src/dashboard/pages/04_Body_Motion.py`, support code `src/dashboard/pose_viewer.py`). Only sessions that pass the metadata and manifest checks are listed. Playback is not implemented. Tested with synthetic data and headless runs.
- Training and inference layers: not implemented.

Do not claim that pose extraction, hand extraction, audio analysis, training or inference works until the code and tests show it.

### Face data extractor history

The face data extractor is `src/analysis/extract_faces.py`. Its documented changes are:

- Commit `40d5b75`: person and session arguments, and the person/session path layout.
- Commit `e8f032d`: overwrite protection and atomic JSON writes.

The `facial_data.json` format is unchanged, but the extractor and frame storage have changed.

Preserve face data compatibility. Do not treat the face data extractor as immutable. Changes to it need an explicit request, compatibility tests and documentation.

## 7. Pose data Session Writer Safety

Preserve these mechanisms. Do not simplify them to reduce code.

Conventions in a session folder:

- `.pose-staging-*`: temporary staging folders used while a pose dataset is generated.
- `.pose-previous-*`: the previous pose dataset, kept during regeneration so it can be restored.
- `.pose-lock`: exclusive lock that stops a second writer working on the same session.

Rules:

1. Validate generated data before publishing it.
2. Keep the manifest and file-integrity hashes. Verify the published files against the manifest before `metadata.json` is written.
3. Publish `metadata.json` last. It is the completion marker.
4. Do not bypass staging or validation.
5. Never remove or overwrite face data during pose data generation, regeneration, migration or recovery.
6. Do not remove or overwrite a previous pose dataset except through the regeneration workflow.
7. Recovery and rollback must refuse to remove any file whose size or hash no longer matches the manifest, and any manifest that lists a face data file.
8. `recover_incomplete_pose` clears the lock file (`.pose-lock`). Run it only after confirming that no writer is active. Running it during a live write defeats the lock. Never clear staging, previous or lock files by hand without understanding their state.
9. Test partial writes, crashes and recovery whenever this code changes.

## 8. Data Integrity and Scientific Requirements

Mandatory:

- Preserve frame order.
- Preserve timestamps and their provenance.
- Prefer valid decoded presentation timestamps (PTS). Use FPS-derived timestamps only as a fallback, and record which source was used. (At present, face data and the pose data writer rely on FPS-derived timestamps. Decoded PTS are not yet extracted.)
- Keep one shared timeline across modalities.
- Represent missing detections as NaN rows. Never silently drop frames.
- Validate array shapes, timestamps and data types.
- Document coordinate systems and units.
- Keep measured values, estimates and derived metrics distinct.
- Document assumptions and limitations.
- Do not claim accuracy without validation evidence.

### Scientific validation

The primary reference is `docs/validation.md`. It covers conventions, formulas, tests and limitations.

The head-pose axis-flip convention, `diag(-1, 1, 1)`, is **empirically supported by limited local recordings** (three sessions from two people). It is **not proven** as a universal MediaPipe coordinate-system guarantee, and it has not been checked against an independently measured head rotation. Do not describe it as universally correct.

Tracking-quality metrics measure how reliably the face was found and described. They are not proof of landmark or motion accuracy.

The system currently targets single-person recordings. Ambiguous multi-person detections must be rejected or flagged, not silently resolved.

Handedness labels and scores are kept separate from hand landmark arrays. Whether a model label matches anatomical left and right has not been verified.

## 9. Dependencies and Platform Support

Use Python 3.11.

### OpenCV pins (platform-specific)

- **Windows:** `opencv-contrib-python==5.1.0.95`
- **Linux:** `opencv-contrib-python==5.0.0.93`. The 5.1.0.95 release has no Linux wheel on PyPI.
- **macOS:** unsupported and untested. There is no supported project pin.

**Never install `opencv-python` and `opencv-contrib-python` in the same environment.** MediaPipe depends on the contrib build, and both packages provide overlapping OpenCV modules.

Before changing these pins, inspect `requirements.txt`, `packages.txt` and the CI workflow.

### System packages

`packages.txt` lists the Linux system packages required by OpenCV: `libgl1` and `libglib2.0-0`. The CI workflow installs the same packages on Ubuntu. Do not assume that `requirements.txt` alone is enough on every platform.

### Adding a dependency

1. Decide whether it is Python-level or system-level.
2. Update the matching file (`requirements.txt` or `packages.txt`).
3. Check Windows compatibility and Linux CI compatibility.
4. Avoid unnecessary additions and upgrades. Do not change working versions without a demonstrated need.
5. Document any special installation step.

## 10. Development Container and Deployment Security

`.devcontainer/devcontainer.json` is a development convenience. It runs on a Python 3.11 image and installs `packages.txt` and `requirements.txt`. It starts no server; start the API and `web/` as in the README.

For any public deployment:

- Use the default CORS and XSRF protections, or an equivalent reviewed configuration.
- Configure trusted origins explicitly.
- Validate the deployment separately. Do not assume the development configuration is production-safe.
- Keep private recordings and datasets out of the deployed artefact.

## 11. Model Licensing and External Assets

Before adding an external model:

1. Identify the official source.
2. Verify the model asset's own licence, separately from the framework licence.
3. Check commercial-use and redistribution terms.
4. Record model version, filename, source URL and verification date.
5. Define a SHA-256 verification procedure.
6. Document unresolved licence questions.

Never assume that a framework's Apache-2.0 licence covers its model weights.

Do not download models, add model binaries or add model dependencies without explicit approval. Model binaries must never be committed. `models/mediapipe/` is git-ignored.

## 12. Coding Standards

- Python 3.11 unless explicitly approved otherwise.
- Follow existing naming and directory conventions.
- Write modular, testable code. Avoid duplicating existing functionality.
- Avoid unnecessary dependencies.
- Preserve public interfaces and output formats unless a change is approved. Do not silently change schemas.
- Keep Windows and Linux compatibility.
- CPU processing is the baseline. GPU support needs explicit approval.
- Validate inputs. Handle missing, corrupt and unsupported files safely.
- Give clear error messages and logging.
- Keep focused tasks free of unrelated refactoring.

## 13. Tests and Test Retirement

The CI workflow is `.github/workflows/tests.yml`. The documented test command is:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

If test modules or the workflow change, update this command in the same change.

Tests should cover, where relevant: normal input, invalid input, missing detections, irregular timestamps, shape validation, face data compatibility, pose data schema validation, file integrity, and partial writes with recovery.

Never remove or weaken a test merely to make the suite pass.

### Retiring a test

A test may encode an incorrect scientific assumption or obsolete behaviour. To retire one:

1. Identify the specific incorrect assertion.
2. Explain why it is incorrect.
3. Provide technical or scientific evidence.
4. Propose a replacement test for the correct behaviour.
5. Keep the relevant regression coverage.
6. Record the rationale in the commit message.
7. Obtain the user's approval before removing or materially weakening the test.

## 14. Documentation

Keep documentation in step with the implementation. Relevant files:

- `README.md`
- `CHANGELOG.md`
- `CONTRIBUTING.md`
- `docs/validation.md`
- `docs/proposals/phase2_multimodal_schema.md`
- `THIRD_PARTY_NOTICES.md`

Update them when a change affects architecture, schemas, data formats, model dependencies, installation, extraction, validation or dashboard behaviour.

Label each capability as **implemented**, **partially implemented**, **planned**, **experimental** or **not validated**. Do not describe planned work as done.

## 15. Git Policy and Remote Synchronisation

The primary branch is `main`.

Mandatory rules:

1. Check `git status`, the current branch and the remote state before editing.
2. Do not discard, overwrite, reset, revert or clean uncommitted work.
3. Keep each change focused. Commit each completed task locally on `main`, with a meaningful message.
4. Run the relevant tests before committing. Review the full diff.
5. Never push without explicit user approval.
6. Never force-push. Never rewrite history without explicit approval.

### Remote synchronisation

- Before any synchronisation, run `git fetch` and compare local `main` with `origin/main`.
- If a remote-only commit exists, report its hash, author and summary first.
- Do not merge, rebase, reset or otherwise reconcile remote-only commits without explicit user approval.
- If the local state differs from what the user last reported, report the difference and do not synchronise.
- If unexpected working-tree changes exist, stop and report them before going further.

## 16. CI and Pushing

Before an explicitly approved push:

1. Run the documented test command locally and record the result.
2. Confirm that the GitHub Actions workflow passes on **both** Windows and Ubuntu for the commit being pushed.
3. If CI fails or is still pending, report it and wait for instructions.

Do not claim CI has passed without checking the actual workflow run for the latest pushed commit. Do not keep a hard-coded CI status in documentation. CI status must always be checked against the current commit on GitHub.

## 17. Collaboration Between Coding Agents

Claude Code and OpenAI Codex may both work in this repository.

- `AGENTS.md` is the canonical shared instruction file. `CLAUDE.md` defers to it and must not contradict it.
- Only one agent may actively modify the working tree at a time.
- Before taking over from another agent, check `git status`, the branch, recent commits and any uncommitted changes, and understand what the previous agent completed.
- Do not overwrite or discard another agent's work without review. Do not assume uncommitted changes are disposable.
- Use the existing code as the source of truth. Do not create duplicate implementations of functionality another agent has added.

When instructions conflict:

1. Follow the more restrictive safety or data-integrity requirement.
2. Treat `AGENTS.md` as authoritative for shared rules.
3. Ask the user when a conflict affects scope, architecture or authorisation.

## 18. Scope and Security Restrictions

Unless explicitly requested, do not:

- Modify unrelated repositories, ComfyUI or other local AI projects.
- Change face data extraction behaviour.
- Download models or add model binaries.
- Upload or push code to GitHub.
- Commit personal videos, biometric recordings, datasets, credentials, tokens, passwords or private keys.
- Include private recordings in public examples.

Use synthetic data for public examples and automated tests.

## 19. Task Workflow

1. Understand the requirements.
2. Inspect the relevant source, tests and documentation.
3. Check branch, working-tree and remote state.
4. Identify affected modules and regression risks.
5. For significant work, present a short plan.
6. Implement only the approved scope.
7. Add or update tests.
8. Run the relevant tests.
9. Review the full diff.
10. Update the documentation.
11. Commit locally on `main`.
12. Report the results and limitations.
13. Wait for approval before pushing or starting dependent work.

## 20. Completion Report

At the end of each task, report:

- Summary of changes, and why
- Files created and modified
- Tests run, with exact results
- Compatibility impact
- Known limitations and unverified statements
- Outstanding risks
- Local commit hash, if committed
- Recommended next step

Never claim that a feature, test, model or external resource works unless it has actually been verified.

## Primary Engineering Principle

Build a scientifically reliable, reproducible and maintainable multimodal analysis platform. Data integrity, backward compatibility, transparent validation and licence compliance take priority over implementation speed or feature count.
