"""Loan Collection Risk page."""

import streamlit as st

from src.prediction.predict_collection_risk import predict_collection_risk
from frontend.components.result_panel import render_result, render_placeholder


def render():
    st.markdown('<div class="section-title">&#128176; Loan Collection Risk</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Prioritise collection effort by predicting how '
        'likely an EMI is to be delayed.</div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("collection_form"):
            a, b = st.columns(2)
            with a:
                loan_amount = st.number_input("Loan Amount (Rs)", 10000, 10000000, 500000)
                emi_amount = st.number_input("EMI Amount (Rs)", 100, 1000000, 10000)
                monthly_income = st.number_input("Monthly Income (Rs)", 1000, 10000000, 100000)
                credit_score = st.number_input("Credit Score", 300, 850, 700)
                past_due_days = st.number_input("Past Due Days", 0, 365, 0)
            with b:
                missed_payments = st.number_input("Missed Payments", 0, 10, 0)
                loan_tenure = st.number_input("Loan Tenure (months)", 6, 360, 60)
                existing_loans = st.number_input("Existing Loans Count", 0, 10, 1)
                employment_type = st.selectbox("Employment Type",
                                               ["Salaried", "Self-employed", "Contractual"])
                repayment_ratio = st.slider("Repayment Ratio (EMI / Income)", 0.0, 1.0, 0.1, step=0.05)
            submitted = st.form_submit_button("Run Collection Model", use_container_width=True)

    with result_col:
        if submitted:
            data = {
                "loan_amount": loan_amount, "emi_amount": emi_amount,
                "monthly_income": monthly_income, "credit_score": credit_score,
                "past_due_days": past_due_days, "missed_payments": missed_payments,
                "loan_tenure_months": loan_tenure, "employment_type": employment_type,
                "existing_loans_count": existing_loans, "repayment_ratio": repayment_ratio,
            }
            result = predict_collection_risk(data)
            if result:
                render_result(
                    result["emi_delay_probability"],
                    result["collection_priority"],
                    result["action"],
                    "The model uses past due days, missed payments, credit score "
                    "and the EMI-to-income ratio to rank collection priority.",
                    prob_label="EMI delay probability",
                )
            else:
                render_placeholder("Could not generate a prediction. Train the "
                                   "models first with: python run_all.py")
        else:
            render_placeholder()
