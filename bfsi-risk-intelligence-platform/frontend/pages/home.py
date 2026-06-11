"""Landing / home page."""

import streamlit as st

from frontend.components.hero import render_hero
from frontend.components.cards import module_card, kpi_card, why_card


def render():
    render_hero(
        "BFSI Risk Intelligence Platform",
        "AI-powered risk analytics for banking, financial services, fintech, "
        "NBFC, and insurance use cases.",
    )

    # ---- Module cards ----
    st.markdown('<div class="section-title">Risk Modules</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Four independent machine learning models, '
        'each focused on a core BFSI risk decision.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        module_card("&#128202;", "Credit Risk Prediction",
                    "Score loan applicants as Low, Medium or High risk and "
                    "recommend approve, review or reject.", "Classification")
    with c2:
        module_card("&#128680;", "Fraud Detection",
                    "Flag suspicious transactions in real time and decide "
                    "allow, OTP or block.", "Imbalanced data")
    with c3:
        module_card("&#128101;", "Customer Churn Prediction",
                    "Identify customers likely to leave and suggest the right "
                    "retention action.", "Classification")
    with c4:
        module_card("&#128176;", "Loan Collection Risk",
                    "Prioritise EMI collection effort and route accounts to "
                    "the right team.", "Multi-class")

    # ---- KPI cards ----
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3, gap="medium")
    with k1:
        kpi_card("4", "ML Risk Modules")
    with k2:
        kpi_card("Real-time", "Prediction UI")
    with k3:
        kpi_card("Explainable", "Business Decisions")

    # ---- Why BFSI ----
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Why BFSI companies use this</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Each module turns a model output into a '
        'concrete business action.</div>',
        unsafe_allow_html=True,
    )
    w1, w2, w3, w4 = st.columns(4, gap="medium")
    with w1:
        why_card("Loan Approval Risk",
                 "Assess creditworthiness before lending to reduce defaults "
                 "and bad loans.")
    with w2:
        why_card("Transaction Fraud",
                 "Catch fraudulent payments early to cut financial loss and "
                 "protect customers.")
    with w3:
        why_card("Customer Retention",
                 "Spot at-risk customers and act before they churn to protect "
                 "revenue.")
    with w4:
        why_card("EMI Collection",
                 "Focus collection teams on the riskiest accounts to improve "
                 "recovery rates.")
