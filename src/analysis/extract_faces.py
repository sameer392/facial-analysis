 
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------------
# Recording to extract. Pass an ID on the command line, e.g. "python extract_faces.py presenter-02",
# or change the default below.
VIDEO_ID = sys.argv[1] if len(sys.argv) > 1 else "presenter-01"
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO = PROJECT_ROOT / "recordings" / "raw" / f"{VIDEO_ID}.mp4"
MODEL = PROJECT_ROOT / "models" / "mediapipe" / "face_landmarker.task"

OUTPUT = PROJECT_ROOT / "datasets" / VIDEO_ID
FRAMES_DIR = OUTPUT / "frames"

# Never overwrite an existing dataset
if (OUTPUT / "facial_data.json").exists():
    raise SystemExit(f"Dataset already exists, refusing to overwrite: {OUTPUT}")

FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MediaPipe
base_options = python.BaseOptions(
    model_asset_path=str(MODEL)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

landmarker = vision.FaceLandmarker.create_from_options(
    options
)

cap = cv2.VideoCapture(str(VIDEO))

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", total_frames)

all_data = []

frame_number = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    timestamp_ms = round(
        frame_number * 1000 / fps
    )

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    frame_data = {
        "frame": frame_number,
        "timestamp_ms": timestamp_ms,
        "face_detected": False
    }

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        points = []

        for idx, landmark in enumerate(landmarks):

            points.append({
                "id": idx,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            })

        frame_data["face_detected"] = True
        frame_data["landmarks"] = points

        # Facial blendshapes
        if result.face_blendshapes:

            blendshapes = {}

            for item in result.face_blendshapes[0]:

                blendshapes[item.category_name] = (
                    item.score
                )

            frame_data["blendshapes"] = blendshapes

        # Facial transformation matrix
        if result.facial_transformation_matrixes:

            frame_data["transformation_matrix"] = (
                np.asarray(
                    result.facial_transformation_matrixes[0]
                ).tolist()
            )

        # Draw facial landmarks
        for point in points:

            x = int(point["x"] * width)
            y = int(point["y"] * height)

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        cv2.imwrite(
            str(FRAMES_DIR / f"frame_{frame_number:06d}.jpg"),
            frame
        )

    all_data.append(frame_data)

    frame_number += 1

    if frame_number % 100 == 0:
        print("Processed:", frame_number)

cap.release()
landmarker.close()

# Save all extracted data
with open(
    OUTPUT / "facial_data.json",
    "w"
) as file:

    json.dump(
        all_data,
        file
    )

print("Processing completed!")

print("Frames processed:", frame_number)
print("Results:", OUTPUT)
