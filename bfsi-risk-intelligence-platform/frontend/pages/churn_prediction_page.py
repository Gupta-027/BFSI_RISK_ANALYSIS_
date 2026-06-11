"""Customer Churn Prediction page."""

import streamlit as st

from src.prediction.predict_churn import predict_churn
from frontend.components.result_panel import render_result, render_placeholder


def render():
    st.markdown('<div class="section-title">&#128101; Customer Churn Prediction</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Predict how likely a customer is to leave and '
        'suggest a retention action.</div>',
        unsafe_allow_html=True,
    )

    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        with st.form("churn_form"):
            a, b = st.columns(2)
            with a:
                credit_score = st.number_input("Credit Score", 300, 850, 700)
                age = st.number_input("Age", 18, 100, 42)
                tenure = st.slider("Tenure (years)", 0, 10, 5)
                balance = st.number_input("Account Balance (Rs)", 0, 10000000, 100000)
                number_of_products = st.slider("Number of Products", 1, 4, 2)
            with b:
                estimated_salary = st.number_input("Estimated Salary (Rs)", 0, 10000000, 150000)
                complaints = st.number_input("Complaints (past year)", 0, 5, 0)
                satisfaction_score = st.slider("Satisfaction Score", 1, 5, 4)
                has_credit_card = st.checkbox("Has credit card", value=True)
                is_active_member = st.checkbox("Active member", value=True)
            submitted = st.form_submit_button("Run Churn Model", use_container_width=True)

    with result_col:
        if submitted:
            data = {
                "credit_score": credit_score, "age": age, "tenure": tenure,
                "balance": balance, "number_of_products": number_of_products,
                "has_credit_card": int(has_credit_card),
                "is_active_member": int(is_active_member),
                "estimated_salary": estimated_salary, "complaints": complaints,
                "satisfaction_score": satisfaction_score,
            }
            result = predict_churn(data)
            if result:
                render_result(
                    result["churn_probability"],
                    result["churn_risk_category"],
                    result["retention_suggestion"],
                    "The model considers tenure, activity, products held, "
                    "complaints and satisfaction to estimate churn risk.",
                    prob_label="Churn probability",
                )
            else:
                render_placeholder("Could not generate a prediction. Train the "
                                   "models first with: python run_all.py")
        else:
            render_placeholder()
