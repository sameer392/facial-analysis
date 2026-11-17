import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from dashboard_bootstrap import reload_project_modules  # noqa: E402
reload_project_modules()

from common import cached_analysis, cached_dataset, select_person_and_session  # noqa: E402
from facial_3d_analyzer import depth_range_per_frame, frame_points, head_pose_angles  # noqa: E402
from visualization import landmarks_3d_figure, line_figure  # noqa: E402

st.set_page_config(page_title="3D Facial Analysis", layout="wide")
st.title("3D Facial Landmarks")

person, session_id = select_person_and_session()
st.caption(f"{person} / {session_id}")
analysis = cached_analysis(person, session_id)
dataset = cached_dataset(person, session_id)
raw = dataset.landmarks
compensated = analysis.compensated

view = st.sidebar.radio("Landmark view", ["Head-pose compensated", "Raw as extracted"])
frame_index = st.slider("Frame", 0, len(analysis.table) - 1, 0, 1)

points = compensated if view == "Head-pose compensated" else raw
frame = frame_points(points, frame_index)

if frame["x"].isna().all():
    st.warning(f"No face was tracked in frame {frame_index}.")
else:
    st.plotly_chart(
        landmarks_3d_figure(frame.dropna(), f"{session_id} · frame {frame_index} · {view.lower()}"),
        width="stretch",
    )

angles = head_pose_angles(dataset.matrices)
row = angles.iloc[frame_index]
p1, p2, p3 = st.columns(3)
p1.metric("Yaw", f"{row['yaw_deg']:.1f}°")
p2.metric("Pitch", f"{row['pitch_deg']:.1f}°")
p3.metric("Roll", f"{row['roll_deg']:.1f}°")

st.subheader("Head pose over the session")
st.plotly_chart(
    line_figure(angles.assign(time_s=analysis.table["time_s"]), "time_s",
                ["yaw_deg", "pitch_deg", "roll_deg"], "Head rotation", "Degrees"),
    width="stretch",
)

st.subheader("Depth")
depth = depth_range_per_frame(raw)
st.plotly_chart(
    line_figure(analysis.table.assign(depth_range=depth), "time_s", ["depth_range"],
                "Face depth span (max z minus min z)", "Relative depth"),
    width="stretch",
)

with st.expander("Landmark coordinates for this frame"):
    st.dataframe(frame, width="stretch", hide_index=True)
