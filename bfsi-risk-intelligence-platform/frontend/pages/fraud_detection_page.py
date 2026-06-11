"""Fraud Detection page."""

import streamlit as st

from src.prediction.predict_fraud import predict_fraud
from frontend.components.result_panel import render_result, render_placeholder


def render():
    st.markdown('<div class="section-title">&#128680; Fraud Detection</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Score a transaction for fraud risk and decide '
        'whether to allow, verify or block it.</div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("fraud_form"):
            a, b = st.columns(2)
            with a:
                transaction_amount = st.number_input("Transaction Amount (Rs)", 1, 1000000, 500)
                transaction_hour = st.slider("Transaction Hour", 0, 23, 10)
                old_balance = st.number_input("Old Balance (Rs)", 0, 10000000, 10000)
                new_balance = st.number_input("New Balance (Rs)", 0, 10000000, 9500)
                merchant_risk_score = st.slider("Merchant Risk Score", 0, 100, 20)
            with b:
                failed_attempts = st.number_input("Failed Attempts (24h)", 0, 10, 0)
                device_changed = st.checkbox("Device changed")
                location_changed = st.checkbox("Location changed")
                is_new_beneficiary = st.checkbox("New beneficiary")
            submitted = st.form_submit_button("Run Fraud Model", use_container_width=True)

    with result_col:
        if submitted:
            data = {
                "transaction_amount": transaction_amount,
                "transaction_hour": transaction_hour,
                "old_balance": old_balance,
                "new_balance": new_balance,
                "merchant_risk_score": merchant_risk_score,
                "device_changed": int(device_changed),
                "location_changed": int(location_changed),
                "is_new_beneficiary": int(is_new_beneficiary),
                "failed_attempts_last_24h": failed_attempts,
            }
            result = predict_fraud(data)
            if result:
                render_result(
                    result["fraud_probability"],
                    result["fraud_category"],
                    result["action"],
                    "The model looks at amount, time of day, balance change, "
                    "merchant risk and signals like device or location change "
                    "and failed attempts.",
                    prob_label="Fraud probability",
                )
            else:
                render_placeholder("Could not generate a prediction. Train the "
                                   "models first with: python run_all.py")
        else:
            render_placeholder()
