# Proposal: remove version names from the product (naming migration)

Status: Steps 0 to 5 are done. Step 5 removed the old names as requested, before the one-phase wait this plan suggested; the dataset check was run at Step 1 and the datasets were rebuilt from raw videos with the new code. Step 3 (printed output and the job parser) is done: the scripts print the new wording, and the parser reads only the new markers. Step 4 (docs and test names) is done. Step 5 (remove the old names) is not started.

## 1. Goal

The product will grow past two data versions. Names such as "V1", "V2", "version 1" and `_v2` in code, routes and
file names will stop meaning anything to users. This plan renames them to feature names (face data, pose data,
face analysis, pose build) without changing any dataset that already exists.

Already done: every visible label, badge, heading, help text and API error message (commit `fdbd34d`).

## 2. What is in scope, with counts

Counts come from the repository at commit `fdbd34d`, excluding datasets, recordings, models and logs.

| Area | Examples | Where | Count |
|---|---|---|---|
| Python module names | `schema_v2.py`, `session_writer_v2.py`, `v2_viewer.py`, `v2_session_builder.py`, `build_v2_session.py`, `api/v1_data.py`, `api/v2_data.py` | `src/`, `api/`, `tests/` | 7 modules |
| Python identifiers | `V2Error`, `V2ViewerError`, `V2ExistsError`, `V2Payload`, `V2Session`, `V2_STAGES`, `V2_MARKERS`, `write_v2_session` | `src/`, `api/`, `tests/` | about 130 occurrences of `V2` in code |
| API routes | `/v1/summary`, `/v2`, `/v2/timeline`, `/v2/pose`, `/v2/frames/{i}/overlay.jpg`, `/v1/head-pose` | `api/main.py`, web `api.ts` | 11 routes |
| API job kind codes | `v2_build`, `v1_extract`, `v1_analyse` | `api/jobs.py`, web, tests, job database rows | 3 codes |
| Layout codes | `v1`, `v2`, `incomplete_v2` | `api/catalog.py`, web, tests | 3 codes |
| Temporary and lock names on disk | `.v2-staging-<id>/`, `.v2-previous-<id>/`, `.v2-lock` | writer, readers, recovery, tests | 3 names |
| Script output that the job parser reads | "Extracting pose landmarks", "V2 session published" | `src/extraction/*.py`, `api/jobs.py` | about 8 lines |
| Documentation | README, AGENTS.md, CHANGELOG, `docs/` | repository root and `docs/` | about 8 files |
| Web query keys | `v1summary`, `v2info`, … | `web/app/**` | about 25 keys |

**Not in scope (kept as they are):**

- **Dataset file names.** `facial_data.json`, `facial_blendshapes.csv`, `motion_metrics.csv`, `blink_events.csv`,
  `tracking_quality.json`, `calibration.json`, `frames/`, `timeline.csv`, `manifest.json`, `metadata.json`,
  `facial/`, `body/`, `hands/`, `quality/`. None of these contain a version name.
- **The stored schema string.** `metadata.json` holds `"schema_version": "2.0"`, and readers reject any other major
  version. Metadata also records the SHA-256 of `manifest.json`. Changing the string would mean rewriting every
  completed dataset and its checksums. That is a data migration with real risk, and it gives the user nothing
  visible, so this plan keeps it. Section 6 explains the option of changing it later.
- **Historical records.** Git history, the CHANGELOG entries for past releases, and job logs already written.
- **Streamlit.** The Streamlit dashboard is removed in P7. Renaming there is wasted work, so it is excluded.

## 3. Current data on disk

Checked on this machine (`datasets/`):

| Person | Session | Layout | Notes |
|---|---|---|---|
| sameer_gupta | session-001 | completed pose data | rebuilt on 2026-11-19 from the job system |
| sameer_gupta | session-002 | completed pose data | built on 2026-11-20 |
| sameer_gupta | session-003 | completed pose data | validated on a real recording earlier |
| vajinder_kumar | session-001 | completed pose data | built on 2026-11-20 |

No dataset contains a `.v2-staging-*`, `.v2-previous-*` or `.v2-lock` item. The plan must still handle them if one
appears, for example after an interrupted build.

## 4. Approach

The change is made in small steps. Each step is committed on its own, and every step leaves the software working.
Old names are read for one full phase after the rename, so nothing breaks in the middle.

### Step 0. Safety first (no code change)

1. Back up `datasets/` to a folder outside the repository (a zip with a SHA-256 list of every file).
2. Record a manifest of every dataset file: path, size and SHA-256. Keep it outside the repository.
3. Run the full test suite and the web build, and record the results.

### Step 1. Read both names, write the new names (code only)

1. Add the new module names as thin re-export modules, for example `pose_schema.py` re-exporting `schema_v2.py`.
   The old names keep working.
2. Rename the Python identifiers (`V2Error` to `PoseDataError`, and so on). Keep old names as aliases.
3. Accept the new temporary and lock names in the writer, the readers and recovery:
   - new: `.pose-staging-<id>/`, `.pose-previous-<id>/`, `.pose-lock`
   - old: `.v2-…` names are still recognised, so an existing lock or staging folder is handled, and the writer
     refuses to run while an old lock exists.
4. The writer writes the new names only. Readers and recovery accept both.

Acceptance: the full test suite passes. Test fixtures cover both name sets. A dataset with an old-name lock
is refused, and a dataset with an old-name staging folder is reported by recovery.

### Step 2. Route, kind and layout names (API and web)

1. Add the new routes beside the old ones. For example `/api/persons/{p}/sessions/{s}/pose/…` and `/face/…`.
   The old routes stay for one phase, and each returns the same JSON.
2. Add the new job kind codes (`pose_build`, `face_extract`, `face_analyse`). The job database rows that use the
   old codes are mapped to the new ones when they are read. Nothing is rewritten in the database.
3. Layout codes stay in the API response, with a new `display_name` field. The web app switches to the new field.
   The old field stays for one phase.
4. Update the web app to the new routes, codes and query keys. Remove the old ones in the last step.

Acceptance: the web app works against both sets of routes. The live check passes on the real datasets.

### Step 3. Script output and the job parser

1. The scripts print the new wording, and the job parser accepts both the old and new markers.
2. Once the job log tests pass with both, the old markers are removed from the parser.

### Step 4. Documentation and test names

Rename test files and test functions, and update the README, AGENTS.md, CHANGELOG and the docs. The CHANGELOG entries for past
releases stay as they were written.

### Step 5. Remove the old names (last)

Only after one full phase with both names in use, and after the dataset checks in Step 6 pass:

- remove the old aliases, the old routes, the old job codes and the old temporary names
- the readers still refuse nothing that the writer wrote before this change

### Step 6. Dataset verification (after each step that touches the writer)

1. Re-check every dataset against the manifest from Step 0. No file may change.
2. Validate each pose dataset through the API (`/api/.../pose`) and the command-line validator.
3. Confirm the rebuilt files are byte-identical to the backup where the code path is unchanged.

## 5. Rollback

- Every step is one or more commits. `git revert` of the step restores the earlier behaviour.
- The dataset backup from Step 0 restores any file that was changed by mistake.
- No step changes the stored schema string, so an older version of the code can still read a dataset written by the new
  code. The only new names on disk are the temporary and lock names, which older code does not write. Older code does not
  recognise them, so it must not run against a session with an active lock: that is covered by the lock check in Step 1.

## 6. Options and what I recommend

| Option | Effect | Risk | Recommendation |
|---|---|---|---|
| A. Keep the stored `"2.0"` string, rename everything else (this plan) | Users see no version. Datasets need no rewrite. | Low | **Recommended** |
| B. Change the stored string to a format name (for example `"pose-1"`) | Metadata reads better. | Every completed dataset must be rewritten, its manifest checksum recomputed and every file re-verified. | Defer. Only if a format change is planned anyway. |
| C. Also rename dataset file names | Not needed: none of the names contain a version. | Breaks every existing dataset. | Do not do this. |

## 7. What I need from you

1. Approve option A, or choose B.
2. Confirm that the stored `"2.0"` string stays, and that the dataset file names stay.
3. Approve Step 0 (backup) before any code change. The backup is a copy of personal recordings' derived data, so the
   destination folder should be one you choose.
4. Approve the order of steps. Each step will be committed locally and pushed only after your approval.

## 8. Estimate

Each step is roughly one commit and one test run. Steps 1 and 2 are the largest. Step 1 changes the writer, so it gets a
full dataset verification before it is pushed.
