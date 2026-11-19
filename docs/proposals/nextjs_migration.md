# Proposal: migrate the dashboard from Streamlit to Next.js (with a Python API)

Status: **proposal for review. No application code has been changed.**
Scope: the dashboard only. Python stays the source of truth for extraction, analysis and dataset
writing.

## 1. Current state (inspected)

| Area | Module(s) | Notes |
|---|---|---|
| Dashboard | `src/dashboard/app.py`, `common.py`, `dashboard_bootstrap.py`, pages 01–06 | Streamlit only; `dashboard_bootstrap` re-imports modules on each run, which is a Streamlit-specific workaround |
| Extraction | `extract_faces.py` (V1, script with module-level code), `video_frames.py`, `pose_extractor.py`, `model_registry.py`, `v2_session_builder.py`, `build_v2_session.py` | Pose and video code is pure Python with an injected detector |
| Analysis | `facial_analyzer.py`, `motion_analyzer.py`, `facial_3d_analyzer.py`, `analyze_faces.py`, `visualization.py` (Plotly), `validation.py`, `pose_gating.py` | Plotly figures are built in Python; the Next.js side would need JSON |
| Dataset | `schema_v2.py`, `session_writer_v2.py`, `file_safety.py`, `project_paths.py`, `demo_data.py` | Integrity, staging, lock, rollback and recovery are in the writer |
| Viewer support | `v2_viewer.py` (no Streamlit), `overlay.py` (OpenCV drawing) | `v2_viewer` is already framework-free |
| Tests | 289 tests across 14 modules | Discovery-based; dashboard tests use `streamlit.testing` |

## 2. Reuse versus adapters

**Reusable directly (import into FastAPI):**
- `project_paths`: names, folder layout, `FACIAL_DATA_HOME`. Validation rules are already in place.
- `schema_v2`, `session_writer_v2`, `file_safety`: the writer and its safety rules. Must not be bypassed.
- `v2_session_builder.build_payload`, `pose_extractor`, `video_frames`, `model_registry`.
- `facial_analyzer` (sessions, quality, blinks, per-frame metrics), `motion_analyzer`, `facial_3d_analyzer`, `pose_gating`, `validation`, `v2_viewer` (discovery, joint states, quality summary, landmark table).

**Needs an adapter:**
- `extract_faces.py` and `analyze_faces.py` run code at import time and read `sys.argv`. Call them as subprocesses with the same arguments, or extract their functions into importable modules later. The V1 extractor must not change behaviour (AGENTS.md section 6), so the subprocess route is preferred for now.
- `visualization.py` builds Plotly figures in Python. The API should return data, and the frontend should draw. Figures are not needed on the server.
- `overlay.py` and `v2_viewer.draw_pose_overlay` draw with OpenCV. The browser can draw the same overlay on canvas or SVG from JSON coordinates; server-side drawing remains available for exports.
- `build_v2_session.main()` returns exit codes and prints. The job runner should call `build_payload` and `write_v2_session` directly and map exceptions to job states. The CLI stays for command-line use.

**Streamlit-only (retired with the dashboard):** `app.py`, `common.py`, `dashboard_bootstrap.py`, `pages/*`, `st.cache_data` usage.

## 3. Folder structure (proposed)

```text
src/                        existing Python (unchanged in phase 1)
  analysis/  extraction/  dashboard/ (Streamlit, kept until retirement)
api/                        FastAPI application (new)
  main.py                   app factory, routers, CORS, token check
  routers/                  people, sessions, recordings, datasets, jobs, models, system
  services/                 thin wrappers over src/ modules (no duplicated logic)
  jobs/                     worker, job store, locks, progress events
  schemas/                  Pydantic models mirroring the API contract
  db/                       SQLite schema and migrations (jobs and audit only)
web/                        Next.js application (new)
  app/                      App Router routes (one per module in section 5)
  components/ui/            shadcn/ui components
  components/              domain components (Pose overlay, timeline, 3D viewer)
  lib/api/                  typed client generated from the OpenAPI schema
  lib/hooks/                data hooks (TanStack Query), SSE hook
  styles/                   Tailwind config and design tokens
tests/                      existing Python tests, plus api/ tests
docs/proposals/             this document
```

Recordings and datasets stay where they are (`recordings/raw/`, `datasets/`, `models/mediapipe/`).
The API and the web app never store media in their own folders.

## 4. Backend (FastAPI) architecture

- **Framework:** FastAPI with Pydantic models, served with uvicorn on `127.0.0.1` only.
- **Routers:** one per domain. Each router calls a service, and each service calls the existing `src/` functions.
- **Responses:** JSON for metadata, timelines and tables. Numeric arrays are sent as slices (for example frames `from` to `to`), and as binary `Float32Array` buffers for the 3D and pose views, to avoid very large JSON.
- **Video frames:** `GET .../frames/{index}.jpg` decodes one frame with `video_frames`, encodes it to JPEG at a fixed quality, and caches the result on disk in a cache folder outside `datasets/` (for example `cache/frames/`).
- **Errors:** a single error shape, `{ "code": "...", "message": "...", "details": {...} }`. Codes map from the existing exceptions (`V2ViewerError`, `VideoError`, `ModelIntegrityError`, `V2ExistsError`, and so on).
- **Logging:** structured logs to `logs/` and to the Logs page. No personal data in logs beyond session identifiers.

## 5. Frontend (Next.js) architecture

- **Stack:** Next.js (App Router) with TypeScript, Tailwind CSS, shadcn/ui, Lucide icons.
- **Charts:** ECharts for timelines and distributions (good performance on long recordings). Plotly.js is an alternative if the team prefers the current figure style.
- **3D:** Three.js through React Three Fiber, with `@react-three/drei` for orbit controls. Mesh and points come from the API. Topology is not defined here; the facial mesh uses the approved edge-only wireframe until a triangle source is cleared.
- **Data:** TanStack Query for server state, with keys that include the session and the metadata modification time, so regenerated data is refetched.
- **Typed API:** the OpenAPI schema from FastAPI generates the TypeScript client, so the contract is checked at build time.
- **Live progress:** Server-Sent Events from `/api/jobs/{id}/events`, wrapped in a small hook.

## 6. UI and design system

- **Components:** shadcn/ui on Radix primitives (dialogs, selects, sliders, tables, toasts, tabs). Tables support search, sorting, filtering and pagination.
- **Layout:** persistent sidebar with grouped navigation (Overview; Data: People, Sessions, Recordings, Dataset Explorer, Dataset Validation; Analysis: Facial, 3D, Body, Hand, Audio, Multimodal Timeline, Dataset Comparison; Processing: Jobs, History; Models: Registry, Integrity and Licensing; System: Settings, Logs and Diagnostics). A top bar holds the current person and session and contextual actions.
- **States:** every data view has loading, empty, success and error states.
- **Accessibility:** keyboard navigation, visible focus, labels on every control, colour that is not the only signal (for example joint states use shape or text as well as colour), and reduced-motion support.
- **Tokens:** colour, spacing, radius and type scales defined once in Tailwind, with light and dark variants.
- **Responsive:** sidebar collapses below 1024 px; tables become cards on narrow screens.

## 7. Job lifecycle and progress

Long work (extraction, analysis, V2 build, regeneration) never runs inside an HTTP request.

States: `queued` → `running` → `succeeded` | `failed` | `cancelled`. A `refused` state is recorded for requests rejected before work starts.

- **Create:** `POST /api/jobs` returns `202 Accepted` with a job id. The request is validated first (names, existing outputs, `regenerate` flag, model hash check). Refusals are returned at once and not queued.
- **Worker:** a separate worker process runs MediaPipe, so a crash does not take down the API. The worker reports progress as the fraction of frames processed, plus the stage name (`validate`, `extract`, `assemble`, `publish`, `verify`).
- **Progress:** written to SQLite and streamed over SSE. The client reconnects using the last event id.
- **Cancel:** cooperative. The worker checks a flag between frames. A cancelled V2 build removes only its own staging folder through the existing recovery path.
- **Concurrency:** a unique active-job constraint on `(person, session, job_kind)` in SQLite, plus the existing `.v2-lock` in the writer. A second job on the same session is refused with `409 Conflict`.
- **Restart:** on API start, jobs left `running` are marked `interrupted`. Their staging or lock files are reported, and cleanup needs explicit user action, using the existing `recover_incomplete_v2`.

## 8. API endpoint specification (summary)

All paths are under `/api`. Identifiers are validated by `project_paths`. No endpoint accepts a filesystem path.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, versions, data root name (not the full path) |
| GET | `/persons` | list people (valid names only) |
| POST | `/persons` | create a person folder (name validated) |
| GET | `/persons/{person}/sessions` | sessions with V1/V2 status and detection summaries |
| POST | `/persons/{person}/sessions/{session}/video` | upload an MP4 (size limit, MP4 signature check); refuses an existing session |
| GET | `/persons/{person}/sessions/{session}` | session summary, file list, layout |
| GET | `/persons/{person}/sessions/{session}/v1/frames?from&to` | V1 per-frame records (slice) |
| GET | `/persons/{person}/sessions/{session}/v1/metrics` | motion metrics, blinks, quality, calibration |
| GET | `/persons/{person}/sessions/{session}/v2` | V2 validation result (manifest, metadata, arrays) |
| GET | `/persons/{person}/sessions/{session}/v2/timeline` | timeline rows |
| GET | `/persons/{person}/sessions/{session}/v2/pose?from&to&threshold` | pose slice; binary option |
| GET | `/persons/{person}/sessions/{session}/frames/{index}.jpg` | one video frame (cached) |
| POST | `/jobs` | create extraction, analysis or V2 job (body: kind, person, session, regenerate) |
| GET | `/jobs`, `/jobs/{id}` | list and status |
| GET | `/jobs/{id}/events` | SSE progress stream |
| POST | `/jobs/{id}/cancel` | request cancel |
| DELETE | `/persons/{person}/sessions/{session}/datasets` | delete a dataset (confirmation token required) |
| GET | `/models` | registry entries: filename, SHA-256 status, version, licence status |
| GET | `/logs?level&limit` | recent log entries (no secrets) |

## 9. Dataset and file access strategy

- **Read paths:** resolved only from validated person and session names through `project_paths`. Path traversal is impossible because no client-supplied path reaches the filesystem layer.
- **Writes:** only through the existing writers (`write_v2_session`, `save_session_outputs`, the V1 scripts). The API never writes files directly.
- **Uploads:** written to a temporary name in the session folder, size-checked, MP4-signature-checked, and then renamed. Existing sessions are refused.
- **Caching:** cache keys include the metadata modification time, and a regenerated session invalidates its cache entries.
- **Large arrays:** served as slices. The full pose array (about 1,273 × 33 × 5 floats here) is small, but longer recordings are paged.
- **Deletes:** keep the current confirmation flow, requiring the typed name, and add a server-side confirmation token.

## 10. Authentication and local security

This is a single-user local application, but the risks are real because it exposes file operations.

- **Bind address:** `127.0.0.1` only. Not `0.0.0.0`.
- **Token:** a random token is generated at API start and printed to the console. Every mutating endpoint (uploads, jobs, deletes, folder creation) requires it in a header. The web app receives it through a local settings step, never from a URL.
- **CORS:** allow only the local web origin.
- **CSRF:** the token header rules out simple cross-site form posts.
- **Upload limits:** size limit and MP4 check.
- **No shell:** job commands are executed as argument lists, never through a shell string.
- **Secrets:** no tokens or keys are logged or stored in the repository.
- **Public deployment:** not in scope. Any future deployment needs a separate security review (AGENTS.md section 10). The current Streamlit development flags must not carry over.

## 11. Migration phases and acceptance criteria

Streamlit remains available on port 8501 throughout. The Next.js app runs on port 3000. Both read the same datasets.

| Phase | Work | Acceptance criteria |
|---|---|---|
| P0 | API skeleton, health, token, persons and sessions (read-only) | Endpoints return the same person and session lists as `project_paths`; tests against synthetic data; no file writes |
| P1 | Web shell, navigation, Overview, People, Sessions, Recordings | Pages render on synthetic data; keyboard navigation; empty and error states; Streamlit unchanged |
| P2 | Facial Analysis, Diagnostics, Dataset Comparison, V1 data endpoints | Numeric parity with the Streamlit pages on the same session (same values to a stated tolerance) |
| P3 | 3D Facial Analysis and Body Motion (V2 viewer) | Parity with `v2_viewer` outputs; V2 validation shown; threshold behaviour matches the Python tests |
| P4 | Job system: extraction, analysis, V2 build, progress, cancel, locks | Synthetic job tests: success, failure, cancel, concurrent refusal, restart handling; no HTTP request blocks |
| P5 | Model registry, integrity and licensing page, logs, settings | Hash checks match `model_registry`; licence status shown as recorded, never as cleared |
| P6 | Audio and Hand pages (placeholders, then real pages when those layers exist) | Shown as not implemented until the Python layers exist |
| P7 | Remove Streamlit completely | Decision: the end state is Next.js only, with no Streamlit kept in the project. Removal happens only after P0–P5 pass, with a written sign-off. The Streamlit code, dependencies and documentation are then deleted in one reviewed change. |

Each phase is committed locally, with its tests and documentation, and is pushed only with approval.

## 12. Dependencies and installation

Python additions (require approval under AGENTS.md section 9 before adding to `requirements.txt`):
- `fastapi`, `uvicorn`, `sse-starlette` (or FastAPI's `StreamingResponse`), `pydantic` (already a dependency of several packages in the venv, but must be pinned explicitly).
- `python-multipart` is already in the environment through Streamlit. It needs an explicit pin if the API uses uploads.

Node additions (in `web/package.json`, not in `requirements.txt`):
- Node.js 20 LTS, npm or pnpm.
- `next`, `react`, `react-dom`, `typescript`, `tailwindcss`, `lucide-react`, shadcn/ui components (copied into the repo, MIT licence).
- `@tanstack/react-query`, `zod`, `echarts` (or `plotly.js`), `three`, `@react-three/fiber`, `@react-three/drei`.

Installation (Windows 11): `npm install` in `web/`, then `uvicorn api.main:app --host 127.0.0.1 --port 8000`, then `npm run dev`. Each step is documented in the README when implemented.

Licences of the frontend packages are mostly MIT or Apache-2.0 and must be recorded in `THIRD_PARTY_NOTICES.md` as they are added. Model licences are unchanged by this migration.

## 13. Risks and backward compatibility

- **Two writers during coexistence.** Streamlit writes (upload, process, delete) and API jobs could touch the same session. Mitigation: the writer lock (`.v2-lock`) and the V1 scripts' overwrite refusals stay in force. Until P4, Streamlit remains the only writer for V1 operations, and the API is read-only.
- **Behaviour drift.** A rewrite can change numbers or labels. Mitigation: parity tests against the Python outputs (P2 and P3).
- **Windows subprocesses and MediaPipe.** Worker processes must start reliably on Windows. Test early in P4.
- **Large data in the browser.** Pose and face arrays for long recordings. Mitigation: slices, binary transfer, and virtualised tables.
- **Scope creep.** The web migration must not add features that the Python layers do not support. Audio and hands stay as placeholders until their extractors exist.
- **Compatibility.** No V1 or V2 file format changes. The writer, the manifest and the rollback rules are unchanged. The new API reads the same folders and needs no migration of existing data.
- **Licensing.** The Pose model's commercial status is unchanged by moving the UI. The dashboards must keep showing the licence status as recorded.

## 14. Recommended first step

Approve phase P0 (read-only API skeleton with synthetic-data tests). It adds no writes and no new Python dependencies beyond FastAPI and uvicorn, and it lets us check that the API and the Python code agree before any UI work begins.
