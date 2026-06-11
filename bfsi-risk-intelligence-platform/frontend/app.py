"""
BFSI Risk Intelligence Platform - Streamlit entry point.

This file stays intentionally small: it sets up the page, loads the custom
stylesheet, renders the sidebar navigation and routes to a page module.
All UI lives in frontend/components and frontend/pages; all ML logic lives in
src/prediction. Run with:  streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make the project root importable so we can use `src` and `frontend` packages.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.ui_helpers import load_css
from frontend.components.header import render_sidebar
from frontend.components.footer import render_footer
from frontend.pages import (
    home,
    credit_risk_page,
    fraud_detection_page,
    churn_prediction_page,
    collection_risk_page,
    batch_prediction_page,
    model_info_page,
)

st.set_page_config(
    page_title="BFSI Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()

# Sidebar label -> page render function
ROUTES = {
    "Home": home.render,
    "Credit Risk Prediction": credit_risk_page.render,
    "Fraud Detection": fraud_detection_page.render,
    "Customer Churn Prediction": churn_prediction_page.render,
    "Loan Collection Risk": collection_risk_page.render,
    "Batch Prediction": batch_prediction_page.render,
    "Model Information": model_info_page.render,
}

selected = render_sidebar()
ROUTES[selected]()
render_footer()
