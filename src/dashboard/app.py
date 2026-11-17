import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Personal Facial Motion Analyzer",
    page_icon="🎭",
    layout="wide",
)

st.title("Personal Facial Motion Analyzer")
st.caption("Inspect video frames, MediaPipe landmarks, blendshapes, and facial movement over time.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = PROJECT_ROOT / "datasets"
VIDEOS_ROOT = PROJECT_ROOT / "recordings" / "raw"
EXTRACT_SCRIPT = PROJECT_ROOT / "src" / "analysis" / "extract_faces.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "src" / "analysis" / "analyze_faces.py"
ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,59}")


def suggest_id(filename):
    """Turn an uploaded file name into a safe recording ID, e.g. 'My Clip.mp4' -> 'my-clip'."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-")[:60]


def run_step(script, video_id):
    """Run one pipeline script for a recording using this app's Python. Returns (ok, output)."""
    proc = subprocess.run(
        [sys.executable, str(script), video_id],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


if "upload_message" in st.session_state:
    kind, text = st.session_state.pop("upload_message")
    (st.success if kind == "success" else st.error)(text)

with st.expander("Add or process a recording", expanded=False):
    uploaded = st.file_uploader("Upload a new video (MP4)", type=["mp4"])
    existing_videos = sorted(p.stem for p in VIDEOS_ROOT.glob("*.mp4")) if VIDEOS_ROOT.is_dir() else []
    unprocessed = [v for v in existing_videos if not (DATASETS_ROOT / v / "facial_data.json").is_file()]
    picked = st.selectbox(
        "Or process a video already in recordings/raw/",
        ["(none)"] + unprocessed,
        help="Lists videos that do not have a dataset yet.",
    )
    chosen = None if picked == "(none)" else picked

    if uploaded:
        new_id = st.text_input(
            "Recording ID",
            value=suggest_id(uploaded.name),
            help="Letters, numbers and hyphens. Used as the file and dataset name. Existing IDs cannot be reused.",
        )
    else:
        new_id = chosen or ""
        if chosen:
            st.caption(f"Recording ID: {chosen}")

    if st.button("Extract and analyse", disabled=not (new_id and (uploaded or chosen))):
        target_video = VIDEOS_ROOT / f"{new_id}.mp4"
        target_dataset = DATASETS_ROOT / new_id
        if not ID_PATTERN.fullmatch(new_id):
            st.error("Recording ID must use only lowercase letters, numbers and hyphens.")
        elif (target_dataset / "facial_data.json").exists() or (uploaded and target_video.exists()):
            st.error(f"'{new_id}' already exists. Choose a different recording ID; existing data is never overwritten.")
        else:
            VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
            if uploaded:
                target_video.write_bytes(uploaded.getbuffer())
            with st.status(f"Processing {new_id} (this can take several minutes)...", expanded=True) as status:
                failed = None
                for label, script in [("Extracting faces", EXTRACT_SCRIPT), ("Analysing blendshapes", ANALYZE_SCRIPT)]:
                    st.write(f"{label}...")
                    ok, output = run_step(script, new_id)
                    if not ok:
                        failed = (label, output)
                        break
                if failed:
                    status.update(label=f"Failed while: {failed[0]}", state="error")
                    st.code(failed[1][-4000:])
                    st.session_state["upload_message"] = ("error", f"Processing of '{new_id}' failed. The video was kept in recordings/raw/.")
                else:
                    status.update(label=f"'{new_id}' is ready", state="complete")
                    st.session_state["upload_message"] = ("success", f"'{new_id}' was added and analysed. It is now selectable in the sidebar.")
                    st.cache_data.clear()
                    st.rerun()


def discover_datasets():
    """Return dataset IDs that have a landmarks JSON and a blendshapes CSV."""
    if not DATASETS_ROOT.is_dir():
        return []
    return sorted(
        d.name
        for d in DATASETS_ROOT.iterdir()
        if d.is_dir()
        and (d / "facial_data.json").is_file()
        and (d / "facial_blendshapes.csv").is_file()
    )


available_ids = discover_datasets()

if not available_ids:
    st.warning("No datasets found in the datasets/ folder.")
    st.info(
        "Run extraction and analysis first, for example: "
        "python src/analysis/extract_faces.py, then python src/analysis/analyze_faces.py."
    )
    st.stop()

with st.sidebar:
    st.header("Recording")
    VIDEO_ID = st.selectbox("Select recording", available_ids)
    draw_landmarks = st.checkbox("Draw facial landmarks", value=True)
    point_radius = st.slider("Landmark point size", 1, 5, 2)

DATASET_DIR = DATASETS_ROOT / VIDEO_ID
video_path = str(VIDEOS_ROOT / f"{VIDEO_ID}.mp4")
json_path = str(DATASET_DIR / "facial_data.json")
csv_path = str(DATASET_DIR / "facial_blendshapes.csv")

@st.cache_data(show_spinner=False)
def load_csv(path_string):
    return pd.read_csv(path_string)

@st.cache_data(show_spinner=False)
def load_json(path_string):
    with open(path_string, "r", encoding="utf-8") as f:
        return json.load(f)

def read_video_frame(path, frame_index):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame, fps, frame_count

def draw_points(frame, frame_data, radius=2):
    h, w = frame.shape[:2]
    for point in frame_data.get("landmarks", []):
        x = int(float(point["x"]) * w)
        y = int(float(point["y"]) * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), radius, (0, 255, 0), -1, cv2.LINE_AA)
    return frame

# Validate paths
missing = [p for p in [video_path, json_path, csv_path] if not Path(p).is_file()]
if missing:
    st.warning("Some dataset files were not found. Update the paths in the sidebar.")
    st.code("\n".join(missing))
    st.info(
        f"Expected files: recordings/raw/{VIDEO_ID}.mp4, "
        f"datasets/{VIDEO_ID}/facial_data.json and datasets/{VIDEO_ID}/facial_blendshapes.csv."
    )
    st.stop()

try:
    df = load_csv(csv_path)
    frames_data = load_json(json_path)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

if df.empty or not frames_data:
    st.error("The CSV or JSON dataset is empty.")
    st.stop()

# Align by frame number, not by assuming that CSV row order always matches.
if "frame" in df.columns:
    df = df.sort_values("frame").reset_index(drop=True)
else:
    st.error("CSV must contain a 'frame' column.")
    st.stop()

frame_count_json = len(frames_data)
frame_count_csv = len(df)
max_frame = min(frame_count_json, frame_count_csv) - 1

if max_frame < 0:
    st.error("No matching frames were found in the JSON and CSV.")
    st.stop()

# Overview
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

duration = video_total / video_fps if video_fps else 0
detected_count = int(df["face_detected"].fillna(False).astype(bool).sum()) if "face_detected" in df else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Video FPS", f"{video_fps:.2f}" if video_fps else "Unknown")
m2.metric("Video frames", f"{video_total:,}")
m3.metric("Duration", f"{duration:.2f} sec")
m4.metric("Frames with face", f"{detected_count:,} / {len(df):,}")

st.divider()

# Frame inspection
st.subheader("Frame-by-frame inspection")
frame_index = st.slider("Select frame", 0, max_frame, 0, 1)
frame_data = frames_data[frame_index]
row = df.iloc[min(frame_index, len(df) - 1)]

try:
    original, fps, actual_frame_count = read_video_frame(video_path, frame_index)
    marked = original.copy()
    if draw_landmarks and frame_data.get("face_detected", False):
        marked = draw_points(marked, frame_data, point_radius)

    left, right = st.columns(2)
    with left:
        st.markdown("**Original frame**")
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), use_container_width=True)
    with right:
        st.markdown("**Facial landmarks overlay**")
        st.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), use_container_width=True)

    timestamp = row["timestamp_ms"] / 1000 if "timestamp_ms" in df.columns else (
        frame_index / video_fps if video_fps else 0
    )
    st.caption(
        f"Frame {frame_index} · {timestamp:.3f} seconds · "
        f"Face detected: {frame_data.get('face_detected', False)} · "
        f"Landmarks: {len(frame_data.get('landmarks', []))}"
    )
except Exception as exc:
    st.error(str(exc))

# Blendshape values for selected frame
st.subheader("Blendshape values at selected frame")
metadata = {"frame", "timestamp_ms", "face_detected"}
blend_cols = [
    c for c in df.columns
    if c not in metadata and pd.api.types.is_numeric_dtype(df[c])
]
if blend_cols:
    values = row[blend_cols].sort_values(ascending=False)
    values_df = values.rename_axis("Blendshape").reset_index(name="Score")
    values_df = values_df.head(15)
    fig = px.bar(
        values_df.sort_values("Score"),
        x="Score",
        y="Blendshape",
        orientation="h",
        range_x=[0, 1],
        title="Top 15 blendshape scores for this frame",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No numeric blendshape columns were found in the CSV.")

# Timeline plots
st.subheader("Facial movement over the complete recording")
preferred_groups = {
    "Mouth": ["jawOpen", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthSmileRight"],
    "Eyes": ["eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight"],
    "Eyebrows": ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
}
time_col = "timestamp_ms" if "timestamp_ms" in df.columns else None
if time_col:
    plot_df = df.copy()
    plot_df["time_seconds"] = plot_df[time_col] / 1000
    for group_name, candidates in preferred_groups.items():
        available = [c for c in candidates if c in plot_df.columns]
        if available:
            with st.expander(f"{group_name} movement", expanded=(group_name == "Mouth")):
                melted = plot_df[["time_seconds"] + available].melt(
                    id_vars="time_seconds",
                    var_name="Blendshape",
                    value_name="Score",
                )
                fig = px.line(
                    melted,
                    x="time_seconds",
                    y="Score",
                    color="Blendshape",
                    range_y=[0, 1],
                    title=f"{group_name} blendshapes over time",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)

# Landmark coordinate inspection
st.subheader("Landmark coordinates")
landmarks = frame_data.get("landmarks", [])
if landmarks:
    landmark_df = pd.DataFrame(landmarks)
    st.write(f"Detected {len(landmark_df)} landmarks in frame {frame_index}.")
    selected_ids = st.multiselect(
        "Inspect landmark IDs",
        options=landmark_df["id"].tolist(),
        default=[1, 13, 14, 33, 61, 199, 263, 291] if len(landmark_df) > 291 else [],
    )
    if selected_ids:
        st.dataframe(
            landmark_df[landmark_df["id"].isin(selected_ids)],
            use_container_width=True,
            hide_index=True,
        )
    with st.expander("Show all landmark coordinates"):
        st.dataframe(landmark_df, use_container_width=True, hide_index=True)
else:
    st.info("No landmarks were stored for this frame.")

# Download the currently loaded tabular dataset
st.download_button(
    "Download loaded blendshapes CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="facial_blendshapes.csv",
    mime="text/csv",
)

st.caption(
    "Note: this dashboard aligns the selected frame index across the MP4, JSON, "
    "and CSV. It is a frame-inspection tool; it does not yet synchronize a "
    "continuous video player with the charts."
)
