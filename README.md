# Facial Analysis

Personal facial-motion pipeline for a talking digital human. It extracts MediaPipe face landmarks, blendshapes and transformation matrices from presenter videos, analyses them, and shows them in a Streamlit dashboard. It is the foundation for later audio-to-motion training.

## Status

This project is being reorganised from `C:\AI\facial`. The folder structure and the copied scripts are in place. The scripts still use the old relative paths (`recordings/…`, `output/…`), so they must be run from `C:\AI\facial` until they are refactored. No recordings or datasets have been moved yet.

## Project structure

```text
facial-analysis/
├── .venv/                  Python virtual environment (not committed)
├── README.md
├── requirements.txt        Python packages
├── .gitignore
├── models/mediapipe/       MediaPipe face_landmarker.task (planned)
├── recordings/
│   ├── raw/                Original presenter videos (planned)
│   └── processed/
├── datasets/
│   ├── presenter-01/       metadata.json, facial_data.json, facial_blendshapes.csv, landmarks.npy, frames/ (planned)
│   └── presenter-02/       same layout (planned)
├── src/
│   ├── extraction/         Face, audio and body extraction (planned)
│   ├── analysis/           extract_faces.py, analyze_faces.py (copied, not yet refactored)
│   ├── dashboard/          Streamlit app (facial_motion_dashboard.py copied as app.py)
│   ├── training/           Dataset, model, losses, training (planned)
│   └── inference/          Generation (planned)
├── scripts/                Command-line entry points (planned)
├── checkpoints/            Model checkpoints
├── logs/
└── tests/                  test_extraction.py, test_dataset.py (planned)
```

Empty folders contain a `.gitkeep` file so that git keeps them.

## Requirements

- Windows 11
- Python 3.11
- NVIDIA GPU (RTX 3090 used for development). Not required for the current extraction code, which runs on the CPU.

Pinned packages are listed in `requirements.txt`:

| Package | Version |
|---|---|
| mediapipe | 1.2.3 |
| opencv-python | 5.0.0.93 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| streamlit | 1.67.0 |
| plotly | 7.1.0 |

## Installation

Run these from `C:\AI\facial-analysis` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` only installs packages. The `python -m venv` and `Activate.ps1` steps create and enable the virtual environment. Each new terminal needs the activation step again.

To reuse the existing environment in `C:\AI\facial`, activate it instead:

```powershell
C:\AI\facial\.venv\Scripts\Activate.ps1
```

## Dataset preparation

1. Place each presenter video in `recordings/raw/` as `presenter-01.mp4`, `presenter-02.mp4`, and so on.
2. Put the MediaPipe model at `models/mediapipe/face_landmarker.task`.
3. Run the extraction for each recording. Each recording gets its own folder under `datasets/`, and existing datasets are never overwritten.

## Facial extraction

The extraction script is currently hardcoded to presenter-01. Until it is refactored, run it from the old project:

```powershell
cd C:\AI\facial
python extract_faces.py
```

Planned once refactored:

```powershell
python scripts/extract_faces.py --video presenter-01
python scripts/extract_faces.py --video presenter-02
```

Extraction produces, per frame: 478 facial landmarks, 52 blendshape scores, a facial transformation matrix, a timestamp, and an annotated image.

## Analysis

```powershell
cd C:\AI\facial
python analyze_faces.py
```

This writes `facial_blendshapes_presenter-01.csv` from the extraction JSON.

## Dashboard

Until the dashboard is refactored, launch it from the old project:

```powershell
cd C:\AI\facial
streamlit run facial_motion_dashboard.py
```

The dashboard is meant to switch between presenters. That is not yet implemented in the copied version.

## Dataset files

| File | Contents |
|---|---|
| `facial_data.json` | One entry per frame: frame number, timestamp in ms, whether a face was detected, 478 landmarks (`id`, `x`, `y`, `z`), 52 blendshapes, and the transformation matrix |
| `facial_blendshapes.csv` | One row per frame: frame, timestamp, face detected, then one column per blendshape score |
| `frames/` | Annotated JPEG frames with landmarks drawn in green |
| `metadata.json` | Planned: source video, FPS, frame count, extraction settings |
| `landmarks.npy` | Planned: landmarks as a NumPy array for training |

## Planned: future training architecture

- **Audio features** (`src/extraction/audio_extractor.py`): speech features aligned to video frames.
- **Facial motion dataset** (`src/training/dataset.py`): pairs audio features with blendshape and landmark targets.
- **Body pose dataset** (`src/extraction/body_extractor.py`): body pose targets.
- **Audio-to-motion model** (`src/training/model.py`, `losses.py`): not implemented yet.
- **Training and validation** (`src/training/train.py`): not implemented yet. Checkpoints go in `checkpoints/`.
- **Inference** (`src/inference/generate.py`): not implemented yet.

No training code exists yet, and no model has been trained.

## Tests

Planned: `tests/test_extraction.py` and `tests/test_dataset.py`. Not written yet.
