import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard_bootstrap import reload_project_modules  # noqa: E402
reload_project_modules()

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from common import cached_analysis, cached_dataset, select_person_and_session  # noqa: E402
from facial_3d_analyzer import frame_points  # noqa: E402
from visualization import landmarks_3d_figure, line_figure  # noqa: E402

st.set_page_config(page_title="Diagnostics", layout="wide")
st.title("Diagnostics")

person, session_id = select_person_and_session()
st.caption(f"{person} / {session_id}")
analysis = cached_analysis(person, session_id)
dataset = cached_dataset(person, session_id)
table = analysis.table
quality = analysis.quality

st.info(
    "Two different things are shown here. **Tracking checks** say whether the face was found and "
    "described completely. **Motion values** describe movement, and are only as good as the tracking "
    "and the head-pose estimate behind them. Good tracking does not prove the motion is accurate."
)

st.subheader("Tracking checks")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Face detected", f"{quality['detection_rate'] * 100:.1f} %")
c2.metric("Transform matrix available", f"{quality['transform_matrix_on_detected'] * 100:.1f} %")
c3.metric("Landmarks complete (detected frames)", f"{quality['landmark_completeness_on_detected'] * 100:.1f} %")
c4.metric("Tracking interruptions", quality["interruption_count"])
if quality["timestamp_issues"]:
    st.warning("Timestamp issues: " + "; ".join(quality["timestamp_issues"]))
else:
    st.success("Timestamps increase and step at the frame rate.")
if not quality["frame_size_known"]:
    st.warning("Video frame size unavailable. Head-pose compensation assumed square pixels.")

st.subheader("Tracking interruptions")
st.plotly_chart(
    px.scatter(table, x="time_s", y="face_detected", title="Face detected per frame (0 = no face)",
               labels={"time_s": "time (s)", "face_detected": "face detected"}).update_traces(
        marker=dict(size=4)),
    width="stretch",
)
if quality["interruptions"]:
    st.dataframe(pd.DataFrame(quality["interruptions"]), width="stretch", hide_index=True)
else:
    st.info("No interruptions: a face was found in every frame.")

st.subheader("Neutral-frame selection")
st.caption(
    "Neutral frames are the tracked frames with the lowest mouth, jaw, eye and brow activity. "
    "They set the reference shape for neutral-face offset."
)
neutral_df = table.assign(neutral=table["is_neutral"].fillna(False).astype(int))
st.plotly_chart(
    px.bar(neutral_df, x="time_s", y="neutral", title=f"Neutral frames selected: {analysis.neutral_frames}",
           labels={"time_s": "time (s)", "neutral": "1 = neutral"}),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["neutral_offset"], "Distance from neutral shape", "Face-size units"),
    width="stretch",
)

st.subheader("Raw versus head-pose compensated landmarks")
max_frame = len(table) - 1
frame_index = st.slider("Frame", 0, max_frame, 0, 1, key="diag_frame")
raw_frame = frame_points(dataset.landmarks, frame_index).dropna()
comp_frame = frame_points(analysis.compensated, frame_index).dropna()
left, right = st.columns(2)
with left:
    if raw_frame.empty:
        st.info("No face in this frame.")
    else:
        st.plotly_chart(landmarks_3d_figure(raw_frame, f"Raw (normalised image units), frame {frame_index}"),
                        width="stretch")
with right:
    if comp_frame.empty:
        st.info("No compensated shape for this frame.")
    else:
        st.plotly_chart(landmarks_3d_figure(comp_frame, f"Compensated (face-size units), frame {frame_index}"),
                        width="stretch")

st.subheader("Head rotation (MediaPipe matrix, degrees)")
st.plotly_chart(
    line_figure(table, "time_s", ["yaw_deg", "pitch_deg", "roll_deg"], "Head rotation", "Degrees"),
    width="stretch",
)
st.caption(
    "Angles come from the transformation matrix. The sign convention of these angles has not been "
    "checked against an external reference, so read the direction of rotation with care."
)

st.subheader("Facial displacement, velocity and acceleration")
st.plotly_chart(
    line_figure(table, "time_s", ["displacement"], "Mean landmark displacement per frame", "Face-size units"),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["velocity"], "Speed (magnitude)", "Face-size units / s"),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["acceleration"], "Acceleration (magnitude of change in speed)",
                "Face-size units / s²"),
    width="stretch",
)
st.caption("Velocity and acceleration are speed magnitudes. They do not show direction.")
