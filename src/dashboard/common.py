"""Shared helpers for the dashboard (app.py and pages/ import from here)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from facial_analyzer import analyse_session, load_session, make_session  # noqa: E402
from project_paths import list_persons, list_processed_sessions  # noqa: E402


def persons_with_data() -> list[str]:
    return [p for p in list_persons() if list_processed_sessions(p)]


def select_person_and_session(key_prefix: str = ""):
    """Sidebar selectors for a processed (person, session). Stops the page if nothing is processed yet."""
    persons = persons_with_data()
    if not persons:
        st.info(
            "No processed sessions yet. Add a recording on the main page (Add or process a recording), "
            "or run extract_faces.py and analyze_faces.py for a session."
        )
        st.stop()
    person = st.sidebar.selectbox("Person / project", persons, key=f"{key_prefix}person")
    sessions = list_processed_sessions(person)
    session_id = st.sidebar.selectbox("Session", sessions, key=f"{key_prefix}session")
    return person, session_id


@st.cache_data(show_spinner="Analysing session...")
def cached_analysis(person: str, session_id: str):
    return analyse_session(make_session(person, session_id))


@st.cache_data(show_spinner="Loading dataset...")
def cached_dataset(person: str, session_id: str):
    """Original (raw) extracted data for a session, loaded once per server session."""
    return load_session(make_session(person, session_id))
