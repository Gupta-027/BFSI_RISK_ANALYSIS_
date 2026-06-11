"""
Prediction result panel.

Renders a single, self-contained result card for every prediction page:
a probability figure, a coloured progress bar, a risk badge, the recommended
business action and a short plain-English explanation.
"""

import streamlit as st

from frontend.components.risk_badges import badge_html, badge_level


def render_result(probability, category, recommendation, explanation,
                  prob_label="Predicted probability"):
    """Render the populated result card."""
    level = badge_level(category)
    pct = round(float(probability) * 100)

    st.markdown(
        f"""
        <div class="result-panel">
            <div class="result-header">Prediction Result</div>
            <div class="result-prob">{pct}<span>%</span></div>
            <div class="result-prob-label">{prob_label}</div>
            <div class="bar">
                <div class="bar-fill bar-{level}" style="width:{pct}%"></div>
            </div>
            <div class="result-row">
                <span class="label">Risk Category</span>
                {badge_html(category)}
            </div>
            <div class="result-row">
                <span class="label">Recommended Action</span>
                <span class="result-action">{recommendation}</span>
            </div>
            <div class="result-explain">&#8505;&nbsp; {explanation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(message="Fill in the details and run the model to see "
                               "the risk assessment here."):
    """Render the empty-state card shown before a prediction is made."""
    st.markdown(
        f"""
        <div class="result-panel placeholder">
            <div style="font-size:2rem; margin-bottom:10px;">&#128202;</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
