"""Model Information page - one clean card per model."""

import streamlit as st

MODELS = [
    {
        "icon": "&#128202;",
        "name": "Credit Risk Prediction",
        "algo": "Logistic Regression / Random Forest",
        "use": "Decide whether to approve, review or reject a loan application.",
        "inputs": "age, job, credit amount, duration, purpose, housing, savings, checking account",
        "output": "Risk category (Low / Medium / High) + recommendation",
        "metric": "F1-Score (balanced across classes)",
        "why": "Reduces defaults by scoring creditworthiness before lending.",
    },
    {
        "icon": "&#128680;",
        "name": "Fraud Detection",
        "algo": "Logistic Regression / Random Forest (balanced)",
        "use": "Detect fraudulent transactions in real time.",
        "inputs": "amount, hour, old/new balance, merchant risk, device/location change, failed attempts",
        "output": "Fraud category (Safe / Suspicious / High) + action",
        "metric": "Precision, Recall, F1-Score, ROC-AUC",
        "why": "Fraud is rare, so recall and precision matter far more than raw accuracy.",
    },
    {
        "icon": "&#128101;",
        "name": "Customer Churn Prediction",
        "algo": "Logistic Regression / Random Forest",
        "use": "Find customers likely to leave and act before they do.",
        "inputs": "credit score, age, tenure, balance, products, activity, salary, complaints, satisfaction",
        "output": "Churn risk (Low / Medium / High) + retention action",
        "metric": "F1-Score",
        "why": "Retaining an existing customer is far cheaper than acquiring a new one.",
    },
    {
        "icon": "&#128176;",
        "name": "Loan Collection Risk",
        "algo": "Logistic Regression / Random Forest (multi-class)",
        "use": "Prioritise which overdue accounts collection teams chase first.",
        "inputs": "loan & EMI amount, income, credit score, past due days, missed payments, tenure, repayment ratio",
        "output": "Collection priority (Low / Medium / High) + action",
        "metric": "F1-Score",
        "why": "Focuses limited collection capacity on the accounts most likely to default.",
    },
]


def _model_card(m):
    st.markdown(
        f"""
        <div class="model-card">
            <div class="mc-title">{m['icon']} {m['name']}</div>
            <div class="mc-tag">{m['algo']}</div>
            <p><b>Business use case:</b> {m['use']}</p>
            <p><b>Input features:</b> {m['inputs']}</p>
            <p><b>Output:</b> {m['output']}</p>
            <p><b>Evaluation metric:</b> {m['metric']}</p>
            <p><b>Why it matters:</b> {m['why']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="section-title">Model Information</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Each module is a separate scikit-learn model '
        'trained and saved independently with joblib.</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2, gap="medium")
    for i, model in enumerate(MODELS):
        with cols[i % 2]:
            _model_card(model)
