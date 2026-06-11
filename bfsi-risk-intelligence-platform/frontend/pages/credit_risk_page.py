"""Credit Risk Prediction page."""

import streamlit as st

from src.prediction.predict_credit_risk import predict_credit_risk
from frontend.components.result_panel import render_result, render_placeholder


def render():
    st.markdown('<div class="section-title">&#128202; Credit Risk Prediction</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Assess a loan applicant\'s credit risk and '
        'get a lending recommendation.</div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("credit_form"):
            a, b = st.columns(2)
            with a:
                age = st.number_input("Age", 18, 100, 35)
                credit_amount = st.number_input("Credit Amount (Rs)", 100, 100000, 5000, step=100)
                duration = st.number_input("Duration (months)", 1, 72, 24)
                job = st.selectbox("Job", ["admin", "technician", "services", "management",
                                           "retired", "self-employed", "unemployed", "unknown"])
            with b:
                purpose = st.selectbox("Purpose", ["housing", "car", "education",
                                                   "personal", "business", "unknown"])
                housing = st.selectbox("Housing", ["own", "rent", "other"])
                saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "unknown"])
                checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "unknown"])
            submitted = st.form_submit_button("Run Credit Risk Model", use_container_width=True)

    with result_col:
        if submitted:
            data = {
                "age": age, "job": job, "credit_amount": credit_amount,
                "duration": duration, "purpose": purpose, "housing": housing,
                "saving_accounts": saving_accounts, "checking_account": checking_account,
            }
            result = predict_credit_risk(data)
            if result:
                render_result(
                    result["risk_probability"],
                    result["risk_category"],
                    result["business_recommendation"],
                    "The model weighs loan size, duration, savings, checking balance, "
                    "job stability and housing to estimate repayment risk.",
                    prob_label="Confidence in predicted category",
                )
            else:
                render_placeholder("Could not generate a prediction. Train the "
                                   "models first with: python run_all.py")
        else:
            render_placeholder()
