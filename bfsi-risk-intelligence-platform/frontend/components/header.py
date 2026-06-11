"""Sidebar branding and navigation."""

import streamlit as st

NAV_ITEMS = [
    "Home",
    "Credit Risk Prediction",
    "Fraud Detection",
    "Customer Churn Prediction",
    "Loan Collection Risk",
    "Batch Prediction",
    "Model Information",
]


def render_sidebar():
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">&#127974; BFSI <span>Risk</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sidebar-tag">Risk Intelligence Platform</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
        st.divider()
        st.markdown(
            '<div class="sidebar-tag">Built by <b>Gupta</b></div>',
            unsafe_allow_html=True,
        )
    return page
