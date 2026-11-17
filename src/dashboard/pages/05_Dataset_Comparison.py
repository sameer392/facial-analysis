import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard_bootstrap import reload_project_modules  # noqa: E402
reload_project_modules()

from common import cached_analysis, cached_dataset, persons_with_data  # noqa: E402
from project_paths import list_processed_sessions, session_video  # noqa: E402
from visualization import bar_figure, box_figure, distribution_figure  # noqa: E402

st.set_page_config(page_title="Dataset Comparison", layout="wide")
st.title("Dataset Comparison")

people = persons_with_data()
if not people:
    st.info("No processed sessions yet. Add a recording on the main page first.")
    st.stop()

person = st.sidebar.selectbox("Person / project", people)
all_sessions = list_processed_sessions(person)
session_ids = st.sidebar.multiselect("Sessions to compare", all_sessions, default=all_sessions)
if len(session_ids) < 1:
    st.info("Select at least one session in the sidebar.")
    st.stop()

analyses = {sid: cached_analysis(person, sid) for sid in session_ids}
datasets = {sid: cached_dataset(person, sid) for sid in session_ids}

# Summary table
rows = []
for sid in session_ids:
    a = analyses[sid]
    ds = datasets[sid]
    stability = float(np.nanmedian(np.nanstd(a.compensated, axis=0)))
    rows.append({
        "person": person,
        "session_id": sid,
        "recording": session_video(person, sid).name,
        "frames": a.quality["frames"],
        "quality_score": a.quality["quality_score"],
        "detection_rate": a.quality["detection_rate"],
        "longest_missing_run": a.quality["longest_missing_run_frames"],
        "median_jitter": round(a.quality["median_landmark_jitter"], 5),
        "landmark_stability": round(stability, 5),
        "blink_events": len(a.blinks),
        "neutral_frames": a.neutral_frames,
        "duration_s": round(float(ds.timestamps_s[-1]) if len(ds.timestamps_s) else 0.0, 2),
    })
summary = pd.DataFrame(rows)

st.subheader("Session summary and tracking quality")
st.dataframe(summary, width="stretch", hide_index=True)
st.caption(
    "Landmark stability is the median per-landmark standard deviation of head-pose compensated positions. "
    "Lower values mean the face shape was steadier across the session."
)

st.subheader("Blendshape statistics (mean score per session)")
blend_names = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel",
               "eyeBlinkLeft", "eyeBlinkRight", "eyeWideLeft", "eyeWideRight",
               "browDownLeft", "browDownRight", "browInnerUp"]
means = {}
for sid in session_ids:
    cols = [c for c in blend_names if c in datasets[sid].blendshapes]
    means[sid] = datasets[sid].blendshapes[cols].mean()
stats = pd.DataFrame(means).reindex(blend_names).dropna(how="all")
st.dataframe(stats.round(4), width="stretch")

st.subheader("Movement distributions")
tables = {sid: analyses[sid].table for sid in session_ids}

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        distribution_figure({sid: t["jaw_open"] for sid, t in tables.items()},
                            "Jaw opening distribution", "jawOpen score"),
        width="stretch",
    )
with c2:
    st.plotly_chart(
        distribution_figure({sid: t["mouth_activity"] for sid, t in tables.items()},
                            "Mouth movement distribution", "Summed mouth/jaw activity"),
        width="stretch",
    )

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(
        distribution_figure({sid: t["eye_blink"] for sid, t in tables.items()},
                            "Eye blink score distribution", "Mean eyeBlink score"),
        width="stretch",
    )
with c4:
    st.plotly_chart(
        distribution_figure({sid: t["brow_inner_up"] for sid, t in tables.items()},
                            "Eyebrow movement distribution", "browInnerUp score"),
        width="stretch",
    )

st.subheader("Landmark motion")
c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(
        box_figure({sid: t["velocity"] for sid, t in tables.items()},
                   "Facial movement velocity", "Units / second"),
        width="stretch",
    )
with c6:
    st.plotly_chart(
        box_figure({sid: t["acceleration"] for sid, t in tables.items()},
                   "Facial movement acceleration", "Units / second²"),
        width="stretch",
    )

st.plotly_chart(
    bar_figure(summary.set_index("session_id")["landmark_stability"], "Landmark stability (lower is steadier)"),
    width="stretch",
)
st.plotly_chart(
    bar_figure(summary.set_index("session_id")["quality_score"], "Tracking quality score (0-100)"),
    width="stretch",
)
