"""Small shared UI helpers for the frontend."""

from pathlib import Path

import streamlit as st


def load_css(filename="styles.css"):
    """Inject the project stylesheet into the Streamlit page."""
    css_path = Path(__file__).parent / filename
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
