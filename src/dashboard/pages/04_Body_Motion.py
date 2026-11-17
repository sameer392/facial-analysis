from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTRACTOR = PROJECT_ROOT / "src" / "extraction" / "body_extractor.py"

st.set_page_config(page_title="Body Motion", layout="wide")
st.title("Body Motion")

if not EXTRACTOR.is_file():
    st.info(
        "Not available yet. Body pose extraction has not been implemented, so there is no body motion data. "
        "This page will show upper-body pose and gesture metrics per session once "
        "src/extraction/body_extractor.py exists."
    )
    st.markdown(
        "**Needed next:** a pose extractor (for example MediaPipe Pose Landmarker, which requires a model file "
        "that is not in the project yet) writing per-frame pose data keyed by `session_id`."
    )
else:
    st.success("Body extractor found, but no pose data has been generated for this page yet.")
