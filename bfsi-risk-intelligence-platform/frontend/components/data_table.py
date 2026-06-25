"""
Result-display helpers for the Batch Prediction page: summary KPI cards and
the scored results table.

No ML logic here — these take an already-scored DataFrame and render it.
"""

import streamlit as st

from src.utils.risk_rules import classify


def _stat_card(variant, label, value):
    """One coloured KPI/stat card (rendered as styled HTML)."""
    st.markdown(
        f"""
        <div class="stat-card is-{variant}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summary_kpis(result_df, module_name):
    """Render five KPI cards summarising a scored batch."""
    probabilities = result_df["prediction_probability"]
    levels = [classify(module_name, p)[0] for p in probabilities]

    total = len(result_df)
    low = levels.count("low")
    medium = levels.count("medium")
    high = levels.count("high")
    avg_prob = float(probabilities.mean()) if total else 0.0

    cols = st.columns(5, gap="medium")
    cards = [
        ("total", "Total Records", total),
        ("low", "Low Risk", low),
        ("medium", "Medium Risk", medium),
        ("high", "High Risk", high),
        ("avg", "Avg Risk Probability", f"{avg_prob:.0%}"),
    ]
    for col, (variant, label, value) in zip(cols, cards):
        with col:
            _stat_card(variant, label, value)


def results_table(df):
    """Render the (optionally filtered) scored results as a sortable table."""
    st.dataframe(df, use_container_width=True, hide_index=True)
