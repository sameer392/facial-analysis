# AGENTS.md — Facial Analysis Project

## 1. Project Instructions

This repository develops a multimodal facial and body motion analysis and dataset platform for future audio-driven digital-human generation.

The long-term objectives include:

- Accurate lip synchronization
- Facial expressions and blendshapes
- Eye, eyebrow and mouth movements
- Head pose and movement
- Body and upper-body motion
- Hand gestures
- Audio and phoneme alignment
- Reproducible datasets for future model training

The project prioritizes scientific correctness, validated data, reproducibility and backward compatibility.

**Repository:** `sameer392/facial-analysis`  
**Local path:** `C:\AI\facial-analysis`  
**Primary branch:** `main`  
**Python:** 3.11

## 2. Agent Responsibilities

Act as a senior Python, computer vision and software architecture engineer.

Before making changes:

- Inspect the existing implementation.
- Read relevant documentation.
- Review existing tests.
- Check Git status and current branch.
- Understand the data flow and dependencies.
- Identify possible regressions.

Do not assume that a feature is missing until you have inspected the repository.

Prefer incremental, testable changes over large architectural rewrites.

## 3. Architecture

The intended platform has four separate layers.

### A. Acquisition and Extraction

- Video decoding and timestamps
- Facial landmarks
- Facial blendshapes
- Facial transformation matrices
- Body pose landmarks
- Hand landmarks and handedness
- Audio extraction and alignment

### B. Motion Analysis

- Facial motion
- Lip and mouth movement
- Eye and eyebrow movement
- Head pose
- Body motion
- Hand gestures
- Velocity and acceleration
- Tracking-quality analysis

### C. Dataset and Training

- Standardized schemas
- Session metadata
- Shared timelines
- Data validation
- Dataset comparison
- Training-ready exports

### D. Inference and Digital Human Generation

- Audio-driven lip synchronization
- Facial animation
- Expression generation
- Head movement
- Body and hand gesture generation

Keep these layers modular. Avoid coupling extraction, analysis, dataset writing, training and inference.

## 4. Existing Implementation

### Phase 1 — Facial Analysis

Implemented:

- MediaPipe Face Landmarker
- 478 facial landmarks
- 52 blendshape scores
- Facial transformation matrices
- JSON and CSV exports
- Head-pose compensation
- Neutral calibration
- Motion metrics
- Blink detection
- Tracking-quality metrics
- Streamlit dashboard

The original V1 facial extractor is:

`src/analysis/extract_faces.py`

**Do not modify the V1 extraction pipeline or change its output format without explicit authorization.**

Preserve existing V1 datasets and compatibility.

### Phase 2 — Multimodal Dataset Infrastructure

Implemented:

- V2 schema
- V2 session writer
- Shared timeline structure
- Metadata and manifest
- File integrity hashes
- Facial, body and hand data layouts
- Quality information
- Staging and recovery mechanisms

Important files:

- `src/analysis/schema_v2.py`
- `src/analysis/session_writer_v2.py`
- `docs/proposals/phase2_multimodal_schema.md`

Inspect the actual code before relying on documentation. If documentation is stale, report the discrepancy rather than silently changing the architecture.

## 5. Data Integrity and Scientific Requirements

These are mandatory requirements.

- Preserve frame ordering.
- Preserve timestamps and timestamp provenance.
- Prefer valid decoded PTS.
- Use FPS-derived timestamps only as a fallback.
- Record the selected timestamp source.
- Use one shared timeline for all modalities.
- Preserve missing detections as NaN rows.
- Never silently discard frames.
- Validate array shapes and data types.
- Document coordinate systems and measurement units.
- Distinguish raw observations from derived calculations.
- Document assumptions and limitations.
- Do not claim accuracy without validation.

The system currently targets single-person recordings. Ambiguous multiple-person detections must be handled explicitly rather than silently choosing a person.

Hand landmarks and handedness information are separate. Do not merge handedness labels into landmark coordinate arrays.

## 6. Model Licensing and External Assets

Before adding an external model:

1. Locate its official source.
2. Verify the model asset's own licence.
3. Distinguish model licensing from framework licensing.
4. Verify commercial use and redistribution terms.
5. Record version, filename, URL and verification date.
6. Define SHA-256 verification.
7. Document any uncertainty or restrictions.

Do not assume that a framework's Apache-2.0 licence covers its model weights.

**Do not download models or introduce model binaries without explicit user approval.**

Never commit model binaries to the repository.

## 7. Coding Standards

- Python 3.11 is the current baseline.
- Follow existing naming and directory conventions.
- Write modular, reusable and testable code.
- Avoid unnecessary dependencies.
- Preserve existing public interfaces unless a change is approved.
- Maintain Windows and Linux compatibility.
- Prefer CPU-compatible processing as the baseline.
- Provide clear error handling and logging.
- Validate input files before processing.
- Do not silently change schemas or output formats.
- Do not duplicate existing functionality.

Do not introduce unrelated refactoring into a focused task.

## 8. Tests

Run the existing test suite:

```powershell
python -m unittest tests.test_fixtures tests.test_regression tests.test_scientific tests.test_schema_v2 tests.test_session_writer_v2 -v
```

Run additional relevant tests for modified modules.

Tests should cover, where applicable:

- Normal input
- Invalid input
- Missing detections
- Irregular timestamps
- Data shape validation
- V1 compatibility
- V2 schema validation
- File integrity
- Partial writes and recovery

Never delete or weaken tests just to achieve a passing result.

Report actual test results. Do not claim that tests passed if they were not run.

## 9. Documentation

Update documentation when implementation changes affect:

- Architecture
- Schemas
- Data formats
- Model dependencies
- Installation
- Extraction
- Validation
- Dashboard behavior

Clearly distinguish implemented, experimental, planned and unvalidated functionality.

Do not describe planned features as completed.

## 10. Git Policy

The working branch is `main`.

Mandatory rules:

- Check `git status` before editing.
- Do not discard user changes.
- Do not overwrite uncommitted work.
- Keep changes focused.
- Run tests before committing.
- Review the final diff.
- Commit completed tasks locally on `main`.
- Never push without explicit approval.
- Never force-push.
- Never rewrite Git history without explicit approval.
- Do not use destructive reset or clean operations without approval.

If unexpected working-tree changes exist, stop and report them.

Use meaningful commit messages.

## 11. Collaboration with Other Agents

The repository may be edited by Claude Code as well as Codex.

Follow these rules:

- Respect `AGENTS.md` as the shared project instruction file.
- Claude Code also uses `CLAUDE.md`.
- Keep both instruction files consistent.
- Only one agent should actively write to the same working tree at a time.
- Before taking over, inspect `git status`, branch and recent commits.
- Review changes made by the previous agent.
- Never overwrite another agent's work without understanding it.
- Do not assume that an uncommitted change is disposable.
- Avoid duplicate implementations.

When switching agents, treat the current repository state as the source of truth.

## 12. Security and Scope Restrictions

Unless explicitly requested, do not:

- Modify unrelated projects.
- Modify ComfyUI or its configuration.
- Modify the V1 extraction pipeline.
- Download external model files.
- Upload or push code to GitHub.
- Commit personal videos or biometric recordings.
- Commit credentials, tokens, passwords or private keys.
- Add private or identifiable recordings to public examples.

Use synthetic data for public examples and automated tests.

## 13. Task Execution Protocol

For each task:

1. Read the task requirements.
2. Inspect the relevant source files.
3. Check Git status and branch.
4. Identify affected modules.
5. Prepare an implementation plan for significant work.
6. Implement only the requested scope.
7. Add or update tests.
8. Run relevant tests.
9. Review the complete diff.
10. Update documentation if required.
11. Commit locally on `main`.
12. Report the outcome.

Do not proceed to dependent work if the task requires approval.

## 14. Completion Report

At the end of each task, provide:

- Summary of implementation
- Files created
- Files modified
- Tests executed
- Test results
- Compatibility impact
- Known limitations
- Outstanding risks or questions
- Local commit hash, if committed
- Recommended next step

Never claim an action was completed unless it was actually performed and verified.

## 15. Primary Engineering Principle

Build a scientifically reliable, reproducible and maintainable multimodal analysis platform.

Data integrity, backward compatibility, transparent validation and licensing compliance take priority over implementation speed or feature count.