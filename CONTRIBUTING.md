# Contributing

Thank you for considering a contribution. This project handles facial data, so please read the
privacy rules below before you open a pull request.

## Privacy rules (required)

- **Never commit personal material.** That includes recordings, photographs, facial landmark
  datasets, voice samples, extracted frames, or anything derived from a real person.
- Use the synthetic demo data (`python src/analysis/demo_data.py <folder>`) for examples,
  tests, screenshots and bug reports.
- Do not paste real landmark or blendshape data into issues. Describe the problem, or
  attach a synthetic reproduction.
- Do not commit credentials, API keys or tokens.

The `.gitignore` already excludes `.venv/`, `datasets/`, `recordings/raw/**/*.mp4` and the
model file. Check `git status` before each commit.

## Development setup

Python 3.11 is the supported version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest tests.test_fixtures tests.test_regression -v
```

Tests in `test_fixtures.py` use temporary synthetic data and run anywhere. Tests in
`test_regression.py` also check your local datasets when they exist, and skip those checks
when they do not.

## Coding rules

- Keep the extraction logic in `src/analysis/extract_faces.py` unchanged unless the change is
  discussed in an issue first.
- Outputs must never overwrite existing data without `--force`. Original recordings and
  `facial_data.json` must never be written by analysis code.
- Add a test for every behaviour you change. Tests must not depend on local recordings.
- Keep Windows, Linux and macOS paths using `pathlib`, not hard-coded separators.

## Pull requests

1. Open an issue first for anything larger than a small fix.
2. Keep each pull request focused on one change.
3. Describe what you tested and on which operating system.
4. Make sure the GitHub Actions checks pass.

## Licensing of contributions

By submitting a contribution you agree that it is licensed under the Apache License 2.0,
the same license as the project (see [LICENSE](LICENSE)).
