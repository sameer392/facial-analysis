# Phase 2 proposal: multimodal motion dataset (face, body, hands)

Status: **proposal for review. No extractor has been written.** Implementation starts only after
approval of this schema and the migration strategy.

## 1. Architecture

Every stream is extracted from the same decoded video frames in one pass. A single frame index
and a single timestamp reference link all streams, so alignment is by construction, not by
matching timestamps afterwards.

```text
recordings/raw/<person>/session-NNN.mp4          (unchanged)
        │  one decode pass, frame index i, timestamp t_i = round(i * 1000 / fps)
        ├── FaceLandmarker  ──► facial/
        ├── PoseLandmarker  ──► body/
        └── HandLandmarker  ──► hands/
                                  │
                       quality/ and metadata.json  (summaries, provenance)
```

Each session is written to a staging folder and moved into place only when complete (section 6).

### Proposed session folder (version 2)

```text
datasets/<person>/<session>/
  metadata.json                      schema version, provenance, stream status (section 3)
  timeline.csv                       one row per frame: frame, timestamp_ms, detection flags  [addition]
  facial/
    landmarks.npy                    float32 (T, 478, 3)
    blendshapes.csv                  frame, timestamp_ms, face_detected, 52 category columns
    transformation_matrices.npy      float32 (T, 4, 4)
  body/
    pose_landmarks.npy               float32 (T, 33, 5)
    pose_world_landmarks.npy         float32 (T, 33, 5)
  hands/
    left_hand.npy                    float32 (T, 21, 4)
    right_hand.npy                   float32 (T, 21, 4)
    left_hand_world.npy              float32 (T, 21, 3)   [optional, when the model provides it]
    right_hand_world.npy             float32 (T, 21, 3)   [optional]
  quality/
    tracking_quality.json            per-stream quality (section 5)
```

`timeline.csv` is an addition to your list. Without it, detection status per stream and the
timestamp of each frame would only be recoverable from arrays and metadata. It is small and
human-readable, and it is the single place to check alignment.

### Existing version 1 layout (unchanged)

```text
datasets/<person>/<session>/
  facial_data.json  facial_blendshapes.csv  motion_metrics.csv
  blink_events.csv  tracking_quality.json   calibration.json  frames/
```

Version 1 files are not deleted, renamed or overwritten. Version 2 sits alongside them in the
`facial/`, `body/`, `hands/`, `quality/` subfolders and `metadata.json`, so the two can coexist
in one session folder. The session is version 2 when `metadata.json` declares `schema_version`
2.x; otherwise it is version 1.

## 2. Conventions

| Item | Convention |
|---|---|
| Frame index `i` | 0-based index of decoded frames, as in the existing extraction |
| Timestamp `t_i` | `round(i * 1000 / fps)`, in **milliseconds from the start of the video**, integer |
| Timestamp passed to MediaPipe | the same `t_i`, in ms, for `detect_for_video` (monotonic) |
| Missing value | `NaN` in float arrays. Never zero, never a dropped row |
| Detection flags | in `timeline.csv` (`face_detected`, `pose_detected`, `left_hand_detected`, `right_hand_detected`) |
| Image landmarks | MediaPipe normalised: `x` ÷ width, `y` ÷ height, `z` relative depth on the x scale |
| World landmarks | metres, origin at the hip centre (MediaPipe world coordinate system) |
| Pose channels | `x, y, z, visibility, presence` |
| Hand channels | `x, y, z, handedness_score` (image landmarks); world arrays are `x, y, z` |
| Coordinate convention for the image frame | origin top-left, `y` down (as in the existing data) |

## 3. Example `metadata.json`

```json
{
  "schema_version": "2.0",
  "identity": {"person_id": "demo_person", "session_id": "session-001"},
  "source": {
    "video_filename": "session-001.mp4",
    "fps": 30.0,
    "frame_count": 240,
    "duration_s": 8.0,
    "width": 640,
    "height": 480,
    "sha256": "<hash of the source video>"
  },
  "timeline": {
    "reference": "start of video",
    "timestamp_unit": "milliseconds",
    "timestamp_rule": "round(frame_index * 1000 / fps)",
    "frame_index_base": 0
  },
  "extraction": {
    "started_utc": "2026-11-18T10:00:00Z",
    "finished_utc": "2026-11-18T10:03:12Z",
    "software": "facial-analysis",
    "software_version": "2.0.0-dev",
    "python_version": "3.11",
    "mediapipe_version": "1.2.3",
    "running_mode": "VIDEO"
  },
  "models": {
    "face_landmarker": {"file": "face_landmarker.task", "sha256": "<hash>", "source": "<official URL>", "licence_status": "to verify"},
    "pose_landmarker": {"file": "pose_landmarker_full.task", "sha256": "<hash>", "source": "<official URL>", "licence_status": "to verify"},
    "hand_landmarker": {"file": "hand_landmarker.task", "sha256": "<hash>", "source": "<official URL>", "licence_status": "to verify"}
  },
  "configuration": {
    "face": {"num_faces": 1, "output_blendshapes": true, "output_transformation_matrices": true},
    "pose": {"num_poses": 1, "min_pose_detection_confidence": 0.5, "min_tracking_confidence": 0.5},
    "hands": {"num_hands": 2, "min_hand_detection_confidence": 0.5, "min_tracking_confidence": 0.5}
  },
  "coordinate_conventions": {
    "image_landmarks": "normalised; x/width, y/height; z relative on the x scale; origin top-left, y down",
    "world_landmarks": "metres; origin at hip centre",
    "transformation_matrix": "4x4; top-left 3x3 is head rotation; see docs/validation.md (axis-flip convention empirically supported)"
  },
  "streams": {
    "facial": {"status": "complete", "detected_frames": 240, "landmarks_per_face": 478, "blendshape_categories": 52},
    "body": {"status": "complete", "detected_frames": 230, "landmarks_per_pose": 33, "max_people": 1},
    "hands": {"status": "complete", "left_detected_frames": 61, "right_detected_frames": 0, "landmarks_per_hand": 21}
  },
  "extensibility": {"future_audio": "audio/ folder reserved; will use the same timestamp_ms reference", "future_phonemes": "reserved"}
}
```

Metadata values in the example are placeholders (hashes, URLs, counts).

## 4. Array shapes and types

| File | dtype | shape | Meaning of missing |
|---|---|---|---|
| `facial/landmarks.npy` | float32 | (T, 478, 3) | all `NaN` for that frame |
| `facial/transformation_matrices.npy` | float32 | (T, 4, 4) | all `NaN` |
| `facial/blendshapes.csv` | text | T rows, 3 + 52 columns | empty cells |
| `body/pose_landmarks.npy` | float32 | (T, 33, 5) | all `NaN` for that frame |
| `body/pose_world_landmarks.npy` | float32 | (T, 33, 5) | all `NaN` |
| `hands/left_hand.npy` | float32 | (T, 21, 4) | all `NaN` (no left hand) |
| `hands/right_hand.npy` | float32 | (T, 21, 4) | all `NaN` (no right hand) |
| `hands/left_hand_world.npy` | float32 | (T, 21, 3) | all `NaN` |
| `hands/right_hand_world.npy` | float32 | (T, 21, 3) | all `NaN` |

`T` is the number of decoded frames, identical for every stream. Row `i` of every array is frame
`i`, timestamp `timestamp_ms[i]` in `timeline.csv`. A test asserts this for every session.

Hand arrays hold at most one hand per label per frame. If MediaPipe returns two hands with the same
label, the higher-scoring one is kept and the other is recorded as `unassigned_duplicate_hands` in
`tracking_quality.json`, so nothing is silently dropped.

## 5. Quality data (`quality/tracking_quality.json`)

Per stream (`facial`, `body`, `hands.left`, `hands.right`):

- `detected_frames`, `detection_rate`
- `interruptions`: runs of frames without detection, with start frame, end frame, length and start time
- `completeness`: detected frames with all expected landmarks finite
- `mean_detection_score` where the model provides a score (hands: handedness; pose: presence)
- `timestamp_issues` (shared timeline)

This file reports tracking only. Motion accuracy is documented separately, as in `docs/validation.md`.

## 6. Atomic writes, interruption and regeneration

- **Staging:** extraction writes into `datasets/<person>/<session>.staging/`. The session folder
  appears only after every file is written and checked. An interrupted run leaves only a
  `.staging` folder, which is reported and never read as data.
- **Atomic files:** each file is written to a temporary name and renamed (the existing
  `file_safety.atomic_write_*` helpers, extended to `.npy`).
- **Overwrite protection:** extraction never overwrites a session that has any v1 or v2 output.
  Regeneration of v2 requires an explicit `--regenerate-v2` and first moves the existing v2
  folder to `<session>.previous/`, so the previous result can be restored by a single rename.
  Nothing is deleted automatically.
- **Raw data:** recordings and the v1 files are read only.

## 7. Migration strategy (explicit, reversible)

- **No automatic migration.** Existing version 1 sessions keep working unchanged.
- **Optional command:** `migrate_session.py <person> <session> --dry-run` lists what it would
  create. Without `--dry-run`, it writes version 2 files alongside v1 files. It copies
  `facial_data.json` into `landmarks.npy`, `transformation_matrices.npy` and `blendshapes.csv`
  where values can be reproduced exactly from the JSON, and checks them against the JSON.
- **Source of truth:** `facial_data.json` stays the source for facial data. v2 facial arrays are
  a derived copy and are verified against it.
- **Rollback:** delete only the v2 subfolders and `metadata.json` listed in the migration report.
  The command never touches v1 files.
- **Reader rule:** the dashboard reads v2 when `metadata.json` declares it, otherwise v1.

## 8. Dependencies and models

- **Python packages:** no new packages. `mediapipe` 1.2.3 provides `PoseLandmarker`,
  `PoseLandmarkerOptions`, `HandLandmarker` and `HandLandmarkerOptions` (checked in the project
  `.venv`). `numpy` already in use.
- **Model files (not yet downloaded):**
  - `pose_landmarker_*.task` (MediaPipe Pose Landmarker; variant to choose: lite, full or heavy)
  - `hand_landmarker.task` (MediaPipe Hand Landmarker)
  - Each model is stored under `models/mediapipe/`, which git ignores, with its SHA-256 in
    `metadata.json`.
- **Licence status:** the MediaPipe framework is Apache-2.0. **Model weights have their own terms.**
  Before any download, the official model card or licence page must be read and recorded in
  `THIRD_PARTY_NOTICES.md`. Nothing will be assumed from the framework licence.
- **Download:** requires your approval (per the original project rules).
- **Platforms:** CPU by default. The RTX 3090 is not needed for these models at this scale.
  GPU delegates are not part of this proposal.
- **OpenCV:** unchanged. Pose and hand processing use the same `cv2` calls already in use.

## 9. Design decisions to confirm

1. **Single person per frame in version 2.0.** MediaPipe Pose and Hands do not give persistent
   identity across frames, so multiple people could be merged silently. Version 2.0 sets
   `max_people: 1` and records it. Multi-person support needs a tracking method and is deferred.
2. **Handedness:** MediaPipe's left/right label may be mirrored relative to the person's own left and
   right. This must be checked against a known recording before the labels are called "left" and
   "right" in the dashboard. Until then the schema stores the model's label as `handedness_label`
   alongside the arrays, and the dashboard labels them "label A / label B" if the check is not done.
3. **`timeline.csv`:** an addition, see section 1.
4. **Dashboard:** the existing placeholder page `04_Body_Motion` becomes the Body and Hand page. The
   Audio placeholder stays. The Facial and Diagnostics pages are not changed.
5. **Hand world coordinates:** stored as optional, because the models may not provide them in every
   case.

## 10. Implementation plan (after approval)

1. Model acquisition: approved download, SHA-256 recorded, licence status recorded in notices.
2. `src/analysis/schema_v2.py`: schema constants, validation of arrays, metadata and timeline.
   Tests first.
3. `src/extraction/pose_extractor.py` and `hand_extractor.py`, sharing one frame loop with
   `extract_faces.py` logic kept unchanged. Tests with synthetic images and mocked landmarker output.
4. Session writer: staging, atomic `.npy` writes, `--regenerate-v2` with `.previous/`.
5. Migration command with dry run and rollback report, tested on a copy.
6. Dashboard: dataset browser reads v1 and v2; new Body and Hand page; existing pages unchanged.
7. Tests (section 11), README, CHANGELOG, THIRD_PARTY_NOTICES, CI on Windows and Ubuntu.

## 11. Test plan (synthetic only)

- Timestamp alignment: every stream has `T` rows and `timeline.csv` agrees with the array index.
- Array dimensions and dtypes for every `.npy` file.
- Missing pose: a frame without a pose gives `NaN` and `pose_detected = 0`.
- Missing hands: no hand gives `NaN` for that label and a missing-hand interval in quality.
- Left/right identity: a synthetic hand labelled A stays in the same array across frames.
- Multiple people: rejected with a clear message while `max_people` is 1.
- Schema validation: missing fields, wrong shapes and wrong dtypes are reported.
- Metadata completeness: every field in section 3 is present.
- Existing dataset immutability: v1 files hash-identical before and after v2 extraction and migration.
- Interrupted extraction: a staging folder is left and not read; regeneration needs `--regenerate-v2`.
- Backward compatibility: v1 sessions load unchanged in the dashboard and in the existing tests.

## 12. Open questions for you

1. Approve the version 2 folder layout and the additional `timeline.csv`.
2. Approve the model downloads (which Pose variant: lite, full or heavy), after checking their licences.
3. Accept single-person-per-frame for version 2.0.
4. Confirm the handedness check method: a known recording where left and right are labelled by you.
5. Confirm replacing the placeholder `04_Body_Motion` page.
