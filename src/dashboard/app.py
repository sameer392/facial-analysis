import json
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from common import select_person_and_session  # noqa: E402  (sets up sys.path for analysis modules)
from project_paths import (  # noqa: E402
    PROJECT_ROOT,
    create_person,
    is_valid_person,
    is_valid_session,
    list_persons,
    list_processed_sessions,
    list_sessions,
    next_session_id,
    person_recordings_dir,
    session_dataset_dir,
    session_video,
    suggest_person,
)

st.set_page_config(
    page_title="Personal Facial Motion Analyzer",
    page_icon="🎭",
    layout="wide",
)

st.title("Personal Facial Motion Analyzer")
st.caption("Inspect video frames, MediaPipe landmarks, blendshapes, and facial movement over time.")

EXTRACT_SCRIPT = PROJECT_ROOT / "src" / "analysis" / "extract_faces.py"
ANALYZE_SCRIPT = PROJECT_ROOT / "src" / "analysis" / "analyze_faces.py"


def run_step(script, person, session):
    """Run one pipeline script for a session with this app's Python. Returns (ok, output)."""
    proc = subprocess.run(
        [sys.executable, str(script), person, session],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


if "upload_message" in st.session_state:
    kind, text = st.session_state.pop("upload_message")
    (st.success if kind == "success" else st.error)(text)


# ---------------------------------------------------------------------------
# Person / project management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Person / project")
    with st.expander("Create a new person or project", expanded=not list_persons()):
        new_person_raw = st.text_input("Name", placeholder="e.g. sameer_gupta or my-project")
        if st.button("Create folder"):
            name = suggest_person(new_person_raw)
            if not is_valid_person(name):
                st.error("Enter a name with letters or numbers.")
            elif (person_recordings_dir(name)).exists():
                st.error(f"'{name}' already exists.")
            else:
                create_person(name)
                st.session_state["upload_message"] = ("success", f"Created '{name}'. Select it above.")
                st.rerun()


# ---------------------------------------------------------------------------
# Add a recording (upload or process an existing video) into a person's session
# ---------------------------------------------------------------------------
persons = list_persons()
with st.expander("Add or process a recording", expanded=not persons):
    if not persons:
        st.info("Create a person or project in the sidebar first.")
    else:
        target_person = st.selectbox("Person / project", persons, key="add_person")
        sessions_on_disk = list_sessions(target_person)
        processed = set(list_processed_sessions(target_person))
        unprocessed = [s for s in sessions_on_disk if s not in processed]

        uploaded = st.file_uploader("Upload a new video (MP4)", type=["mp4"])
        if uploaded:
            session_name = st.text_input(
                "Session name",
                value=next_session_id(target_person),
                help="Format session-NNN, e.g. session-001. The video is saved as <session name>.mp4.",
            )
        else:
            choice = st.selectbox(
                "Or process a video already in this folder",
                ["(none)"] + unprocessed,
                help="Videos in recordings/raw/<person>/ that do not have a dataset yet.",
            )
            session_name = None if choice == "(none)" else choice

        if st.button("Extract and analyse", disabled=not (uploaded or session_name)):
            if uploaded and not is_valid_session(session_name):
                st.error("Session name must look like session-001.")
            elif uploaded and session_video(target_person, session_name).exists():
                st.error(f"'{session_name}' already exists for {target_person}. Choose another session name.")
            elif session_dataset_dir(target_person, session_name).joinpath("facial_data.json").exists():
                st.error(f"A dataset already exists for {session_name}. Existing data is never overwritten.")
            else:
                if uploaded:
                    session_video(target_person, session_name).write_bytes(uploaded.getbuffer())
                with st.status(f"Processing {target_person} / {session_name} (this can take several minutes)...",
                               expanded=True) as status:
                    failed = None
                    for label, script in [("Extracting faces", EXTRACT_SCRIPT),
                                          ("Analysing blendshapes", ANALYZE_SCRIPT)]:
                        st.write(f"{label}...")
                        ok, output = run_step(script, target_person, session_name)
                        if not ok:
                            failed = (label, output)
                            break
                    if failed:
                        status.update(label=f"Failed while: {failed[0]}", state="error")
                        st.code(failed[1][-4000:])
                        st.session_state["upload_message"] = (
                            "error", f"Processing of {target_person}/{session_name} failed. The video was kept.")
                    else:
                        status.update(label=f"{session_name} is ready", state="complete")
                        st.session_state["upload_message"] = (
                            "success", f"{target_person}/{session_name} was added and analysed.")
                        st.cache_data.clear()
                        st.rerun()


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Recording")

person_for_view, VIDEO_ID = select_person_and_session("view_")

with st.sidebar:
    draw_landmarks = st.checkbox("Draw facial landmarks", value=True)
    point_radius = st.slider("Landmark point size", 1, 5, 2)

DATASET_DIR = session_dataset_dir(person_for_view, VIDEO_ID)
video_path = str(session_video(person_for_view, VIDEO_ID))
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


missing = [p for p in [video_path, json_path, csv_path] if not Path(p).is_file()]
if missing:
    st.warning("Some dataset files were not found for this session.")
    st.code("\n".join(missing))
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

cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

duration = video_total / video_fps if video_fps else 0
detected_count = int(df["face_detected"].fillna(False).astype(bool).sum()) if "face_detected" in df else 0

st.caption(f"Viewing {person_for_view} / {VIDEO_ID}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Video FPS", f"{video_fps:.2f}" if video_fps else "Unknown")
m2.metric("Video frames", f"{video_total:,}")
m3.metric("Duration", f"{duration:.2f} sec")
m4.metric("Frames with face", f"{detected_count:,} / {len(df):,}")

st.divider()

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
        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), width="stretch")
    with right:
        st.markdown("**Facial landmarks overlay**")
        st.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), width="stretch")

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

st.subheader("Blendshape values at selected frame")
metadata = {"frame", "timestamp_ms", "face_detected"}
blend_cols = [
    c for c in df.columns
    if c not in metadata and pd.api.types.is_numeric_dtype(df[c])
]
if blend_cols:
    values = row[blend_cols].sort_values(ascending=False)
    values_df = values.rename_axis("Blendshape").reset_index(name="Score").head(15)
    fig = px.bar(
        values_df.sort_values("Score"),
        x="Score",
        y="Blendshape",
        orientation="h",
        range_x=[0, 1],
        title="Top 15 blendshape scores for this frame",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No numeric blendshape columns were found in the CSV.")

st.subheader("Facial movement over the complete recording")
preferred_groups = {
    "Mouth": ["jawOpen", "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthSmileRight"],
    "Eyes": ["eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight"],
    "Eyebrows": ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
}
if "timestamp_ms" in df.columns:
    plot_df = df.copy()
    plot_df["time_seconds"] = plot_df["timestamp_ms"] / 1000
    for group_name, candidates in preferred_groups.items():
        available = [c for c in candidates if c in plot_df.columns]
        if available:
            with st.expander(f"{group_name} movement", expanded=(group_name == "Mouth")):
                melted = plot_df[["time_seconds"] + available].melt(
                    id_vars="time_seconds", var_name="Blendshape", value_name="Score",
                )
                fig = px.line(
                    melted, x="time_seconds", y="Score", color="Blendshape",
                    range_y=[0, 1], title=f"{group_name} blendshapes over time",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, width="stretch")

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
        st.dataframe(landmark_df[landmark_df["id"].isin(selected_ids)], width="stretch", hide_index=True)
    with st.expander("Show all landmark coordinates"):
        st.dataframe(landmark_df, width="stretch", hide_index=True)
else:
    st.info("No landmarks were stored for this frame.")

st.download_button(
    "Download loaded blendshapes CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{person_for_view}_{VIDEO_ID}_facial_blendshapes.csv",
    mime="text/csv",
)

st.caption(
    "Note: this dashboard aligns the selected frame index across the MP4, JSON, and CSV. "
    "It is a frame-inspection tool; it does not yet synchronize a continuous video player with the charts."
)
