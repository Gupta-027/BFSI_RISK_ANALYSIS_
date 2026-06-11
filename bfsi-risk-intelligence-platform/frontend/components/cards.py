"""
Reusable card components for the home page.

Each function renders one card into the current Streamlit container, so they
can be dropped straight into st.columns() for a responsive grid.
"""

import streamlit as st


def module_card(icon, title, description, tag):
    """A module summary card (used for the four ML modules)."""
    st.markdown(
        f"""
        <div class="module-card">
            <div class="module-icon">{icon}</div>
            <div class="module-title">{title}</div>
            <div class="module-desc">{description}</div>
            <div class="module-tag">{tag}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(value, label):
    """A single KPI / stat card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def why_card(title, description):
    """A 'why BFSI companies use this' card with an accent border."""
    st.markdown(
        f"""
        <div class="why-card">
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
