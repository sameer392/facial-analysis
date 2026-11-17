from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTRACTOR = PROJECT_ROOT / "src" / "extraction" / "audio_extractor.py"

st.set_page_config(page_title="Audio Analysis", layout="wide")
st.title("Audio Analysis")

if not EXTRACTOR.is_file():
    st.info(
        "Not available yet. Audio extraction has not been implemented, so there is no audio data to analyse. "
        "This page will show speech features (energy, pitch, speech/silence timing) aligned to each session's "
        "frames once src/extraction/audio_extractor.py exists."
    )
    st.markdown(
        "**Needed next:** an audio extractor that reads each session's `recordings/raw/<person>/session-NNN.mp4`, "
        "writes features keyed by session into `datasets/<person>/<session>/`, "
        "and aligns them to the video frame timestamps."
    )
else:
    st.success("Audio extractor found, but no audio features have been generated for this page yet.")
