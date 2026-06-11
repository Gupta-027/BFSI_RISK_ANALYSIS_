"""Hero banner for the landing page."""

import streamlit as st


def render_hero(title, subtitle, pill="Enterprise BFSI Analytics"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">{pill}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
