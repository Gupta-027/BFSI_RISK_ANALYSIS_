"""
Configuration file for BFSI Risk Intelligence Platform
Contains paths, model parameters, and dataset configurations
"""

import os
import sys

# Ensure Unicode console output (e.g. ✓ / ✗) works on Windows terminals
# that default to the cp1252 codec. Imported by every script, so this is a
# single place that fixes encoding for the whole project.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_INPUTS_DIR = os.path.join(DATA_DIR, "sample_inputs")

# Artifacts paths
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_INPUTS_DIR, ARTIFACTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8

# File names
CREDIT_RISK_RAW_FILE = os.path.join(RAW_DATA_DIR, "credit_risk_data.csv")
FRAUD_DETECTION_RAW_FILE = os.path.join(RAW_DATA_DIR, "fraud_detection_data.csv")
CHURN_PREDICTION_RAW_FILE = os.path.join(RAW_DATA_DIR, "churn_prediction_data.csv")
COLLECTION_RISK_RAW_FILE = os.path.join(RAW_DATA_DIR, "collection_risk_data.csv")

CREDIT_RISK_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "credit_risk_processed.csv")
FRAUD_DETECTION_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "fraud_detection_processed.csv")
CHURN_PREDICTION_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "churn_prediction_processed.csv")
COLLECTION_RISK_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, "collection_risk_processed.csv")

# Model and preprocessor artifact files
CREDIT_RISK_MODEL = os.path.join(ARTIFACTS_DIR, "credit_risk_model.pkl")
CREDIT_RISK_PREPROCESSOR = os.path.join(ARTIFACTS_DIR, "credit_preprocessor.pkl")

FRAUD_DETECTION_MODEL = os.path.join(ARTIFACTS_DIR, "fraud_model.pkl")
FRAUD_DETECTION_PREPROCESSOR = os.path.join(ARTIFACTS_DIR, "fraud_preprocessor.pkl")

CHURN_PREDICTION_MODEL = os.path.join(ARTIFACTS_DIR, "churn_model.pkl")
CHURN_PREDICTION_PREPROCESSOR = os.path.join(ARTIFACTS_DIR, "churn_preprocessor.pkl")

COLLECTION_RISK_MODEL = os.path.join(ARTIFACTS_DIR, "collection_model.pkl")
COLLECTION_RISK_PREPROCESSOR = os.path.join(ARTIFACTS_DIR, "collection_preprocessor.pkl")

# Feature columns for each module
CREDIT_RISK_FEATURES = [
    'age', 'job', 'credit_amount', 'duration', 'purpose',
    'housing', 'saving_accounts', 'checking_account'
]

FRAUD_DETECTION_FEATURES = [
    'transaction_amount', 'transaction_hour', 'old_balance', 'new_balance',
    'merchant_risk_score', 'device_changed', 'location_changed',
    'is_new_beneficiary', 'failed_attempts_last_24h'
]

CHURN_PREDICTION_FEATURES = [
    'credit_score', 'age', 'tenure', 'balance', 'number_of_products',
    'has_credit_card', 'is_active_member', 'estimated_salary',
    'complaints', 'satisfaction_score'
]

COLLECTION_RISK_FEATURES = [
    'loan_amount', 'emi_amount', 'monthly_income', 'credit_score',
    'past_due_days', 'missed_payments', 'loan_tenure_months',
    'employment_type', 'existing_loans_count', 'repayment_ratio'
]
