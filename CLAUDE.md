# CLAUDE.md — Claude Code Instructions

## 1. Shared rules

Before any task, read and follow **`AGENTS.md`**. It is the canonical source for architecture, data-integrity and scientific rules, dataset layouts, pose data recovery safety, dependency pins, testing, documentation, Git policy and scope restrictions. This file does not repeat those rules.

If this file ever conflicts with `AGENTS.md`, follow the more restrictive requirement, treat `AGENTS.md` as authoritative, and ask the user when the conflict affects scope, architecture or authorisation.

## 2. Claude Code operating notes

- Inspect the actual source, tests and `git status` before editing. Do not rely on the README or on earlier conversation.
- Make small, reviewable changes. For significant work, give a short plan (understanding, files, approach, tests, risks) and wait for approval where the task requires it.
- Do not expand a task into refactoring or extra features.
- Do not overwrite uncommitted user changes, or changes made by another agent.
- Only one coding agent should modify the working tree at a time. When switching to or from Codex, review the working tree and recent commits first.
- Do not push, merge, rebase or reset without explicit approval (see `AGENTS.md` section 15).
- Do not download models or add model binaries without explicit approval.

## 3. Verification and reporting

- Run the documented test command in `AGENTS.md` section 13, plus any tests the change needs.
- Report exact test results. Do not claim tests passed if they were not run.
- Check CI only against the commit actually pushed, and only when a push has been approved (`AGENTS.md` section 16).
- For each completed task, report what changed and why, the files touched, the tests and their results, known limitations, the local commit hash if committed, and any approval needed for the next step.
