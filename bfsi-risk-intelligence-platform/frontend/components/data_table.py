"""
Result-display helpers for the Batch Prediction page: summary KPI cards and
the scored results table.

No ML logic here — these take an already-scored DataFrame and render it.
"""

import streamlit as st

from src.utils.risk_rules import classify


def summary_kpis(result_df, module_name):
    """Render five KPI cards summarising a scored batch."""
    probabilities = result_df["prediction_probability"]
    levels = [classify(module_name, p)[0] for p in probabilities]

    total = len(result_df)
    low = levels.count("low")
    medium = levels.count("medium")
    high = levels.count("high")
    avg_prob = float(probabilities.mean()) if total else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", total)
    c2.metric("Low Risk", low)
    c3.metric("Medium Risk", medium)
    c4.metric("High Risk", high)
    c5.metric("Avg Risk Probability", f"{avg_prob:.0%}")


def results_table(df):
    """Render the (optionally filtered) scored results as a sortable table."""
    st.dataframe(df, use_container_width=True, hide_index=True)
