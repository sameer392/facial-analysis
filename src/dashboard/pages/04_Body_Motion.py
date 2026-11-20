"""Body Motion page: inspect existing Pose Landmarker output. Read only; nothing is generated here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard_bootstrap import reload_project_modules  # noqa: E402
reload_project_modules()

import project_paths as pp  # noqa: E402
import pose_viewer as vv  # noqa: E402
from pose_schema import session_layout  # noqa: E402

DEFAULT_THRESHOLD = 0.5  # the same empirical default as the pose visibility gate

st.set_page_config(page_title="Body Motion", layout="wide")
st.title("Body motion (pose data)")
st.caption("Inspects existing Pose Landmarker pose data. Tracking figures only; not a measure of motion accuracy.")


@st.cache_data(show_spinner="Checking pose data session...")
def checked_session(session_path: str, metadata_mtime_ns: int, person: str, session_id: str):
    """Validated pose data session, or the reason it was rejected. Keyed on metadata mtime, so regeneration refreshes it."""
    try:
        return vv.validate_pose_session(Path(session_path), person, session_id), None
    except vv.PoseDataViewerError as exc:
        return None, str(exc)


@st.cache_data(show_spinner="Loading pose data...")
def load_arrays(session_path: str, metadata_mtime_ns: int):
    session = Path(session_path)
    pose = vv.load_pose(session)
    timeline = vv.load_timeline(session, expected_frames=pose.shape[0])
    return pose, timeline


@st.cache_data(show_spinner=False)
def read_video_frame(video_path: str, video_mtime_ns: int, index: int):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
    finally:
        cap.release()
    return frame if ok else None


# ---- discovery -------------------------------------------------------------------------------
candidates = []
for person_dir in sorted(p for p in pp.DATASETS_ROOT.glob("*") if p.is_dir() and pp.is_valid_person(p.name)):
    for session_dir in sorted(p for p in person_dir.glob("*") if p.is_dir() and pp.is_valid_session(p.name)):
        if session_layout(session_dir) in ("face", "none"):
            continue
        candidates.append((person_dir.name, session_dir.name, session_dir))

valid, rejected = [], []
for person, session_id, session_dir in candidates:
    meta_path = session_dir / "metadata.json"
    mtime = meta_path.stat().st_mtime_ns if meta_path.is_file() else 0
    info, reason = checked_session(str(session_dir), mtime, person, session_id)
    if info is None:
        rejected.append((session_dir, reason))
    else:
        valid.append(info)

if rejected:
    with st.expander(f"{len(rejected)} pose data folder(s) not shown (failed checks)"):
        for path, reason in rejected:
            st.write(f"`{path}`: {reason}")

if not valid:
    st.info(
        "No valid pose data sessions were found. Create one with the command line: "
        "`python src/extraction/build_pose_session.py <person> <session>`. "
        "Sessions appear here only after their metadata, manifest and arrays pass the checks."
    )
    st.stop()

# ---- selection and controls -----------------------------------------------------------------
with st.sidebar:
    st.header("Session")
    labels = {f"{v.person} / {v.session_id}": v for v in valid}
    chosen = labels[st.selectbox("pose data session", list(labels))]
    threshold = st.slider("Visibility threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05,
                          help="Joints below this visibility are drawn grey and counted as low visibility. "
                               "Stored values are not changed.")
    show_points = st.checkbox("Show joints", value=True)
    show_skeleton = st.checkbox("Show skeleton", value=True)

meta_mtime = (chosen.path / "metadata.json").stat().st_mtime_ns
pose, timeline = load_arrays(str(chosen.path), meta_mtime)
n = pose.shape[0]

frame_key = f"frame::{chosen.person}::{chosen.session_id}"
st.session_state.setdefault(frame_key, 0)
prev_col, slider_col, next_col = st.columns([1, 8, 1])
if prev_col.button("◀ Previous", disabled=st.session_state[frame_key] <= 0):
    st.session_state[frame_key] -= 1
if next_col.button("Next ▶", disabled=st.session_state[frame_key] >= n - 1):
    st.session_state[frame_key] += 1
frame_index = slider_col.slider("Frame", 0, n - 1, key=frame_key)

ts_ms = int(timeline.loc[frame_index, "timestamp_ms"])
face_flag = bool(timeline.loc[frame_index, "face_detected"])
pose_flag = bool(timeline.loc[frame_index, "pose_detected"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Frame", f"{frame_index} / {n - 1}")
m2.metric("Timestamp", f"{ts_ms / 1000:.3f} s")
m3.metric("Face detected", "yes" if face_flag else "no")
m4.metric("Pose detected", "yes" if pose_flag else "no")

# ---- video and overlay ------------------------------------------------------------------------
video_path = pp.session_video(chosen.person, chosen.session_id)
frame_bgr = read_video_frame(str(video_path), video_path.stat().st_mtime_ns, frame_index) if video_path.is_file() else None
left, right = st.columns(2)
if frame_bgr is None:
    st.warning(f"Video frame unavailable: {video_path.name} was not found or could not be read. "
               "The pose data is still shown in the table below.")
else:
    h, w = frame_bgr.shape[:2]
    if (w, h) != (chosen.width, chosen.height):
        st.warning(f"The video is {w}x{h} but the pose data metadata records {chosen.width}x{chosen.height}. "
                   "The overlay uses the video size, so check the alignment.")
    overlay = vv.draw_pose_overlay(frame_bgr, pose[frame_index], threshold,
                                   show_points=show_points, show_skeleton=show_skeleton)
    with left:
        st.markdown("**Original frame**")
        st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), width="stretch")
    with right:
        st.markdown("**Pose overlay**")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width="stretch")
    st.caption("Green: joint shown (visibility at or above the threshold, inside the frame). "
               "Grey: below the threshold. Off-frame and missing joints are not drawn. "
               "Coordinates are normalised to the video: x / width, y / height, origin top-left.")

# ---- timeline ---------------------------------------------------------------------------------
st.subheader("Detection timeline (timeline.csv)")
tl = timeline.assign(time_s=timeline["timestamp_ms"] / 1000.0,
                     face=timeline["face_detected"].astype(int),
                     pose=timeline["pose_detected"].astype(int))
fig = px.scatter(tl, x="time_s", y=["face", "pose"], labels={"time_s": "time (s)", "value": "detected (1 = yes)"},
                 title="Face and pose detection per frame")
fig.add_vline(x=ts_ms / 1000.0, line_dash="dash")
fig.update_layout(height=280, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")
missing = tl.loc[tl["pose"] == 0, ["frame", "time_s"]]
if len(missing):
    with st.expander(f"Frames without a pose detection ({len(missing)})"):
        st.dataframe(missing, width="stretch", hide_index=True)

# ---- landmark inspection ----------------------------------------------------------------------
st.subheader(f"Joints for frame {frame_index}")
table = vv.landmark_table(pose[frame_index], threshold)
st.dataframe(table, width="stretch", hide_index=True)

# ---- quality summary --------------------------------------------------------------------------
st.subheader("Pose tracking quality")
summary = vv.quality_summary(pose, timeline, threshold, frame_index)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Pose detection rate", f"{summary['pose_detection_rate'] * 100:.1f} %")
q2.metric("Frames without pose", len(summary["missing_pose_frames"]))
q3.metric("Median visibility", "n/a" if summary["median_visibility"] is None else f"{summary['median_visibility']:.2f}")
q4.metric(f"Valid joints (frame {frame_index})", f"{summary['valid_joints_selected_frame']} / 33")
st.caption(f"Share of joints shown across detected frames at threshold {threshold:.2f}: "
           f"{summary['valid_joint_share_detected_frames'] * 100:.1f} %. {summary['scope']}.")
