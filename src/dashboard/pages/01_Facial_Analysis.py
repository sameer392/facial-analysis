import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from dashboard_bootstrap import reload_project_modules  # noqa: E402
reload_project_modules()

from common import cached_analysis, select_person_and_session  # noqa: E402
from visualization import blink_timeline_figure, line_figure  # noqa: E402

st.set_page_config(page_title="Facial Analysis", layout="wide")
st.title("Facial Motion Intelligence")

person, session_id = select_person_and_session()
st.caption(f"{person} / {session_id}")
analysis = cached_analysis(person, session_id)
table = analysis.table
quality = analysis.quality

st.subheader("Tracking quality")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Quality score", f"{quality['quality_score']:.1f} / 100")
q2.metric("Face detection", f"{quality['detection_rate'] * 100:.1f} %")
q3.metric("Longest missing run", f"{quality['longest_missing_run_frames']} frames")
q4.metric("Median landmark jitter", f"{quality['median_landmark_jitter']:.5f}")

c1, c2 = st.columns(2)
c1.metric("Neutral calibration frames", analysis.neutral_frames)
c2.metric("Blink events", len(analysis.blinks))

st.subheader("Motion over time (head-pose compensated)")
st.plotly_chart(
    line_figure(table, "time_s", ["displacement"], "Mean landmark displacement per frame", "Normalised units"),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["velocity"], "Facial movement velocity", "Units / second"),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["acceleration"], "Facial movement acceleration", "Units / second²"),
    width="stretch",
)
st.plotly_chart(
    line_figure(table, "time_s", ["neutral_offset"], "Distance from neutral face", "Normalised units"),
    width="stretch",
)

st.subheader("Blinks")
st.plotly_chart(blink_timeline_figure(table, analysis.blinks), width="stretch")
if len(analysis.blinks):
    st.dataframe(analysis.blinks, width="stretch", hide_index=True)
else:
    st.info("No blinks detected in this session at the 0.5 threshold.")

st.subheader("Head pose (for reference)")
st.plotly_chart(
    line_figure(table, "time_s", ["yaw_deg", "pitch_deg", "roll_deg"], "Head rotation", "Degrees"),
    width="stretch",
)

st.subheader("Jaw and mouth")
st.plotly_chart(
    line_figure(table, "time_s", ["jaw_open", "mouth_activity"], "Jaw opening and mouth activity", "Blendshape score"),
    width="stretch",
)

st.caption(
    "Velocity and acceleration are computed from head-pose compensated landmarks. Translation and scale are not "
    "compensated, so these are shape-change measures. Derived files are in datasets/<person>/<session>/."
)
