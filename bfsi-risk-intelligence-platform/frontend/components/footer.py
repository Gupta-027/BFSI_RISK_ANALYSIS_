"""Page footer."""

import streamlit as st


def render_footer():
    st.markdown(
        '<div class="footer">BFSI Risk Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
