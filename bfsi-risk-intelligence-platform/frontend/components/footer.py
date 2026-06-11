"""Page footer."""

import streamlit as st


def render_footer():
    st.markdown(
        '<div class="footer">BFSI Risk Intelligence Platform '
        '&nbsp;&middot;&nbsp; Built by <b>Gupta</b></div>',
        unsafe_allow_html=True,
    )
