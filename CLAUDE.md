# CLAUDE.md — Facial Analysis Project

## 1. Project Overview

This repository is a multimodal facial and body motion analysis and dataset platform.

The long-term objective is to develop a reliable data pipeline for audio-driven digital humans, including:

- Accurate lip synchronization
- Facial expressions and movements
- Eye, eyebrow and mouth movements
- Head pose and motion
- Body and upper-body movements
- Hand gestures
- Audio, phoneme and timestamp alignment

The project must prioritize scientific accuracy, reliable measurements, reproducible datasets and validated processing pipelines before implementing model training or digital-human generation.

**Repository:** `sameer392/facial-analysis`  
**Local path:** `C:\AI\facial-analysis`  
**Primary development environment:** Windows 11, VS Code  
**Python version:** 3.11  
**Available GPU:** NVIDIA RTX 3090

## 2. Your Role

Act as a senior Python engineer, computer vision engineer and software architect.

Your responsibilities include:

- Understanding the existing architecture before changing code.
- Implementing changes incrementally.
- Maintaining scientific and numerical correctness.
- Writing automated tests for new functionality.
- Reviewing code for reliability, performance and maintainability.
- Maintaining backward compatibility.
- Documenting technical decisions and limitations.
- Reporting completed work honestly.

Do not make assumptions about existing functionality. Inspect the relevant source code and tests before modifying anything.

## 3. Project Architecture

The intended architecture consists of four major layers.

### Layer 1 — Acquisition and Extraction

Capture and extract synchronized information from recordings:

- Video frames and timestamps
- Facial landmarks
- Facial blendshapes
- Facial transformation matrices
- Pose landmarks
- Hand landmarks and handedness
- Audio information

### Layer 2 — Motion Analysis

Analyze extracted data:

- Facial movement
- Lip and mouth movement
- Eye and eyebrow movement
- Head pose
- Body movement
- Hand gestures
- Velocity and acceleration
- Tracking and detection quality

### Layer 3 — Dataset and Training

Build validated datasets with:

- Consistent timestamps
- Standardized schemas
- Data-quality validation
- Session metadata
- Dataset comparison
- Training-ready exports

### Layer 4 — Inference and Digital Human Generation

Future capabilities:

- Audio-driven facial animation
- Lip-sync generation
- Facial expression generation
- Head movement generation
- Body and hand gesture generation
- Digital-human animation pipelines

Keep these layers logically separate. Do not mix extraction, analysis, dataset management, training and inference into one large module.

## 4. Current Repository Structure

Important existing paths:

```text
src/
  analysis/
    analyze_faces.py
    demo_data.py
    extract_faces.py
    facial_3d_analyzer.py
    facial_analyzer.py
    file_safety.py
    motion_analyzer.py
    project_paths.py
    schema_v2.py
    session_writer_v2.py
    validation.py
    visualization.py

  dashboard/
    app.py
    common.py
    dashboard_bootstrap.py
    pages/
      01_Facial_Analysis.py
      02_3D_Facial_Analysis.py
      03_Audio_Analysis.py
      04_Body_Motion.py
      05_Dataset_Comparison.py
      06_Diagnostics.py

  extraction/
  training/
  inference/

tests/
docs/
models/
```

This is a representative structure. Always inspect the actual working tree before relying on it.

## 5. Existing Implementation — Preserve It

### Phase 1: Facial Analysis

Already implemented:

- MediaPipe Face Landmarker.
- 478 facial landmarks.
- 52 facial blendshape scores.
- Facial transformation matrices.
- Facial data JSON and CSV exports.
- Head-pose compensation.
- Neutral calibration.
- Displacement, velocity and acceleration calculations.
- Blink detection.
- Tracking-quality metrics.
- Streamlit dashboard.

The original V1 facial extraction pipeline is:

`src/analysis/extract_faces.py`

**Do not modify or replace this pipeline unless I explicitly authorize it.**

Existing V1 session outputs must remain compatible.

### Phase 2: Multimodal Dataset Infrastructure

The V2 schema and session writer have been implemented.

Important files:

- `src/analysis/schema_v2.py`
- `src/analysis/session_writer_v2.py`
- `docs/proposals/phase2_multimodal_schema.md`

Current V2 design includes:

- Session metadata
- Manifest and file integrity hashes
- Shared timeline
- Facial landmarks and transformations
- Facial blendshapes
- Body pose landmarks
- Hand landmarks
- Handedness information
- Quality information

The session writer uses staging, validation and a completion marker. Preserve its safety and recovery behavior.

Do not redesign or replace the V2 schema or writer without reviewing the existing implementation and obtaining approval for architectural changes.

## 6. Critical Data and Scientific Rules

These rules are mandatory.

1. Preserve original timestamps and their source.
2. Prefer valid decoded presentation timestamps (PTS).
3. Use FPS-derived timestamps only when valid PTS are unavailable.
4. Record which timestamp source was used.
5. Maintain a shared timeline across facial, body and hand data.
6. Missing detections must remain represented as NaN rows.
7. Never silently drop frames because detection failed.
8. Validate array dimensions, timestamps and data types.
9. Preserve units and coordinate-system definitions.
10. Clearly distinguish measured values, estimates and derived metrics.
11. Document empirical assumptions and known limitations.
12. Never claim scientific accuracy without appropriate validation.

For one-person recordings, ambiguous multiple-person detections must not be silently accepted.

Handedness labels and scores must remain separate from hand landmark arrays. Do not assume handedness correctness without validation.

## 7. Model and Dependency Licensing

Before introducing any external model:

- Identify its official source.
- Verify the actual model asset licence separately from the framework licence.
- Check commercial-use permissions.
- Check redistribution permissions and restrictions.
- Record model version, filename, source URL and verification date.
- Define a SHA-256 verification procedure.
- Document unresolved licensing questions.

Never assume that Apache-2.0 licensing of a framework automatically applies to its model weights.

Do not download model files, change model assets or introduce new model dependencies without my explicit approval.

Model binaries must not be committed to Git.

## 8. Development Rules

- Use Python 3.11 unless explicitly approved otherwise.
- Follow the existing project structure and coding conventions.
- Prefer small, modular and testable functions.
- Avoid unnecessary dependencies.
- Do not introduce breaking changes without approval.
- Do not duplicate existing functionality.
- Do not silently change schemas or exported file formats.
- Keep CPU processing as the baseline unless GPU support is explicitly approved.
- Preserve Windows compatibility and Linux CI compatibility.
- Use clear logging and actionable error messages.
- Handle missing, corrupt and unsupported input files safely.

Before implementing a feature, inspect the related modules, tests and documentation.

## 9. Testing and Validation

Run relevant tests after every meaningful change.

The current CI test command is:

```powershell
python -m unittest tests.test_fixtures tests.test_regression tests.test_scientific tests.test_schema_v2 tests.test_session_writer_v2 -v
```

Also run any additional tests introduced by your changes.

When relevant:

- Test Windows compatibility.
- Test Linux compatibility where possible.
- Test malformed and incomplete inputs.
- Test missing detections and irregular timestamps.
- Test backward compatibility with V1 data.
- Test failure recovery and partial-write scenarios.

Never remove, weaken or skip an existing test simply to make the test suite pass.

Report the exact test command and result.

## 10. Documentation

Keep documentation synchronized with the actual implementation.

Update relevant documentation when changing:

- Schemas
- Data formats
- Extraction methods
- Model dependencies
- Validation methods
- Dashboard functionality
- Installation requirements
- Known limitations

Do not document planned functionality as implemented.

Clearly label functionality as:

- Implemented
- Partially implemented
- Planned
- Experimental
- Not validated

## 11. Git and Change Management

The primary working branch is `main`.

Mandatory rules:

1. Check `git status` before starting.
2. Do not discard existing user changes.
3. Do not overwrite uncommitted work.
4. Make focused changes for one task at a time.
5. Run relevant tests before committing.
6. Review the final diff.
7. Commit completed tasks locally on `main`.
8. Never push to GitHub without my explicit approval.
9. Never force-push.
10. Never rewrite or amend existing Git history unless explicitly authorized.
11. Never reset, clean or delete user work without approval.

Use meaningful commit messages.

If the working tree contains unexpected changes, stop and report them before proceeding.

## 12. Working with Other Coding Agents

This repository may also be maintained using OpenAI Codex.

Both agents must follow the same project architecture and Git policies.

- `CLAUDE.md` is your primary instruction file.
- `AGENTS.md` contains the shared project rules and Codex instructions.
- Do not assume changes made by another agent are incorrect.
- Inspect the latest Git status and diff before editing.
- Do not overwrite changes made by another agent.
- Only one coding agent should actively modify the same working tree at a time.
- When another agent has made changes, review them before continuing.
- Do not create duplicate implementations of functionality already added by another agent.

## 13. Scope Restrictions

Unless explicitly requested, do not:

- Modify unrelated repositories or projects.
- Modify ComfyUI or its configuration.
- Modify other local AI projects.
- Change the V1 facial extraction pipeline.
- Download external model binaries.
- Upload or push code to GitHub.
- Introduce personal videos or biometric recordings into the public repository.
- Commit private recordings, credentials, API keys or other sensitive data.

Use synthetic demo data for public examples and automated tests.

## 14. Required Workflow for Every Task

Follow this workflow:

1. Understand the task.
2. Inspect the repository and relevant files.
3. Check the current Git branch and working-tree status.
4. Identify affected modules and possible regressions.
5. Present a short implementation plan for significant tasks.
6. Implement only the approved scope.
7. Add or update tests.
8. Run relevant tests.
9. Review the final diff.
10. Update documentation where necessary.
11. Commit locally on `main`.
12. Report the commit hash, files changed, tests executed and any limitations.
13. Stop and wait for my approval before pushing or starting a dependent task.

For tasks that require approval, do not proceed beyond the approved stage.

## 15. Communication

Be technically precise and transparent.

For each completed task, report:

- What was changed
- Why it was changed
- Files created or modified
- Tests executed and results
- Known limitations
- Licensing or compatibility concerns
- Local commit hash, if committed
- Any approval required for the next step

Do not claim that a feature, test, model or external resource works unless it has actually been verified.

**The priority is a scientifically reliable, maintainable and reproducible platform — not simply adding features quickly.**