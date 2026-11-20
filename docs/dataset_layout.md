# Dataset layout

Status: the face data layout is **implemented** and used by the dashboard. The pose data layout is **partially implemented**: the schema and writer are implemented, and the Pose workflow writes into it (tested with synthetic data). Hand extraction and real-recording validation are pending.

## Folders

```text
recordings/raw/<person>/<session>.mp4     original video (never modified)
datasets/<person>/<session>/              derived outputs for one session
datasets/<person>/dataset_metadata.json   person-level summary (Face data)
```

`<person>` and `<session>` are validated by `src/analysis/project_paths.py`. Session names look like `session-001`. The data root can be redirected with the `FACIAL_DATA_HOME` environment variable. It must contain `recordings/` and `datasets/`.

## Face data (session root)

| File | Contents |
|---|---|
| `facial_data.json` | One object per frame: `frame`, `timestamp_ms`, `face_detected`, and when a face is found `landmarks` (478 items with `id`, `x`, `y`, `z`), `blendshapes` (52 names to scores) and `transformation_matrix` (4×4). Written by extraction and not modified afterwards. |
| `facial_blendshapes.csv` | One row per frame: `frame`, `timestamp_ms`, `face_detected`, then the blendshape columns. |
| `motion_metrics.csv` | Per-frame motion measures (`session_id` on every row). |
| `blink_events.csv` | One row per detected blink. |
| `tracking_quality.json` | Tracking-quality measures only. |
| `calibration.json` | Neutral-frame count and the calibration method. |
| `frames/` | Annotated JPEG frames (created by extraction). |

## Pose data (subfolders in the same session folder)

Face data and pose data files coexist. Nothing in pose data writes, moves or deletes face data files or `frames/`. A session is pose data only when `metadata.json` declares schema 2.x. A folder with pose data pieces but no `metadata.json` is incomplete and must be treated as absent.

```text
metadata.json                 completion marker, written last
manifest.json                 every pose data file with size, SHA-256, and shape and dtype for arrays
timeline.csv                  frame, timestamp_ms, face_detected, pose_detected, left_hand_detected, right_hand_detected
facial/landmarks.npy          float32 (T, 478, 3)   x, y normalised; z relative on the x scale
facial/transformation_matrices.npy   float32 (T, 4, 4)
facial/blendshapes.csv        frame, timestamp_ms, face_detected, 52 blendshape columns
body/pose_landmarks.npy       float32 (T, 33, 5)    x, y, z, visibility, presence (image coordinates)
body/pose_world_landmarks.npy float32 (T, 33, 5)    same channels in world coordinates (metres, hip centre)
hands/left_hand.npy           float32 (T, 21, 3)    x, y, z (absent stream: all NaN for now)
hands/right_hand.npy          float32 (T, 21, 3)
hands/handedness.csv          per frame: detection flags, model label and score per hand
hands/left_hand_world.npy     optional, float32 (T, 21, 3)
hands/right_hand_world.npy    optional, float32 (T, 21, 3)
quality/tracking_quality.json tracking quality for the face, body and hands streams
```

`T` is the frame count, the same for every stream. Row `i` of every array is frame `i`. Missing detections are NaN rows, never dropped frames.

Pose data files that share a name with a face data file are placed in a subfolder, so they do not collide. The face data `tracking_quality.json` is at the session root; the pose data one is `quality/tracking_quality.json`.

### Timestamps

`timestamp_ms` is an integer count of milliseconds from the start of the video. It is `round(frame_index * 1000 / fps)` when no decoded presentation timestamps are available. The `metadata.json` `timeline.timestamp_source` field records which source was used. Currently it is always `fps_derived`.

### Coordinate conventions

- **Image landmarks:** MediaPipe normalised coordinates. `x` is divided by the frame width, `y` by the frame height, and `z` is a relative depth on the scale of `x`. Origin is top-left, with `y` pointing down.
- **World landmarks:** metres, with the origin at the hip centre (MediaPipe world coordinates).
- **Transformation matrix:** the top-left 3×3 block is the head rotation. The rotation convention, including the axis flip, is described in `docs/validation.md` and is **empirically supported, not proven**.

## Metadata (`metadata.json`)

Required sections (checked by `schema_v2.validate_metadata`):

- `schema_version`: `2.x`
- `identity`: person and session identifiers
- `source`: video filename, FPS, frame count, width and height
- `timeline`: reference, unit (`milliseconds`), `timestamp_source`, frame index base (0)
- `extraction`: software, software version, Python version, MediaPipe version, running mode
- `models`: for each model, file name, SHA-256 (lowercase), version, source, `licence_status` (`verified` or `to verify`) and `licence_note`
- `configuration`: face, pose and hand settings. Pose uses `num_poses: 1` (single subject only).
- `coordinate_conventions`
- `streams`: status for `facial`, `body` and `hands` (`complete`, `partial`, `absent` or `failed`), with detection counts
- `outputs`: manifest file name and its SHA-256, written by the writer

`streams.<name>.status: complete` means the stream was processed for every frame. It does not mean every frame had a detection. Detection rates are in `quality/tracking_quality.json`.

## Model licence notes

Copied from `THIRD_PARTY_NOTICES.md`, which is the authoritative record:

- **Pose Landmarker FULL:** partly verified. A one-line MediaPipe maintainer statement says the file is released under Apache 2.0. Commercial use, redistribution, notices and attribution are not confirmed. Not cleared for commercial use or redistribution.
- **Face and Hand Landmarker:** unverified. No model-specific licence has been found.

Model files are kept in `models/mediapipe/`, which git ignores. They are never committed.

## Command-line entry point

```powershell
python src/extraction/build_v2_session.py <person> <session> [--regenerate-v2] [--models-dir DIR]
```

It processes an existing face data session (which must already have `facial_data.json`) into a pose data session. It verifies the Pose FULL model's SHA-256 before loading it, and it never writes to the raw video or the face data files. An existing pose dataset is refused unless `--regenerate-v2` is given, and the previous one is kept for rollback.

Exit codes: `0` published; `1` processing failed, nothing published; `2` refused before processing; `3` model missing or hash mismatch; `4` a face data file changed during the run. The pose data output of that run is rolled back with `rollback_v2`, which restores the previous pose dataset if there was one. If the rollback is refused (for example, because a pose data file was edited), the output says so and the files are left in place. Face data files are never changed by the rollback.

This has been tested with synthetic data and a fake detector only. It has not been run on a real recording.

## Using the Pose workflow from code

The pieces are separate, and nothing runs them automatically yet. Example, using a session folder that already has face data facial data:

```python
from pathlib import Path
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "src/analysis")

from extraction.pose_extractor import extract_pose_video, load_pose_detector
from extraction.v2_session_builder import build_payload
from extraction.video_frames import open_video_info
from session_writer_v2 import write_v2_session

video = Path("recordings/raw/demo_person/session-001.mp4")
session_dir = Path("datasets/demo_person/session-001")
info = open_video_info(video)  # real width, height and frame rate

detector = load_pose_detector()   # verifies the Pose FULL SHA-256 first; raises if it does not match
try:
    pose = extract_pose_video(video, detector=detector)
finally:
    detector.close()

payload = build_payload(dataset_dir=session_dir, pose=pose, fps=info.fps,
                        person="demo_person", session_id="session-001",
                        video_filename=video.name, video_size=(info.width, info.height))
write_v2_session(session_dir, payload)
```

Notes on the example:

- `fps` and the video size come from the video itself, so they are correct for any recording.
- `write_v2_session` refuses to overwrite an existing pose dataset unless `regenerate=True` is passed.
- The example has not been run on a real recording. The synthetic tests cover the same steps with a fake detector.

## Limitations

- Hand extraction is not implemented, so the hand stream is always absent.
- The command-line entry point exists (see above). The dashboard's Body Motion page reads existing pose data sessions for viewing, but it does not generate them.
- No real recording has been processed through this workflow.
- The workflow has been tested with synthetic data and a fake detector. A real recording has not yet been written through it.
- Timestamps are `fps_derived`. Decoded presentation timestamps are not extracted.
