"""
Batch prediction engine for the BFSI Risk Intelligence Platform.

Given a module name and a DataFrame of many records, this loads the correct
trained model + preprocessor once, scores every row, and returns the input
DataFrame with four extra columns:

    prediction_probability  - risk probability (0-1) for the row
    risk_category           - business risk label (from risk_rules)
    business_action         - recommended action (from risk_rules)
    model_module            - which module produced the score

No UI code lives here. The Streamlit page calls run_batch_prediction() and
displays / filters / downloads the returned DataFrame.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CREDIT_RISK_MODEL, CREDIT_RISK_PREPROCESSOR, CREDIT_RISK_FEATURES,
    FRAUD_DETECTION_MODEL, FRAUD_DETECTION_PREPROCESSOR, FRAUD_DETECTION_FEATURES,
    CHURN_PREDICTION_MODEL, CHURN_PREDICTION_PREPROCESSOR, CHURN_PREDICTION_FEATURES,
    COLLECTION_RISK_MODEL, COLLECTION_RISK_PREPROCESSOR, COLLECTION_RISK_FEATURES,
    SAMPLE_INPUTS_DIR,
)
from utils.model_io import load_model, load_preprocessor
from utils.risk_rules import classify


# Central registry: everything the batch flow needs to know about a module.
#
#   features        - columns the model expects (and the required columns)
#   model / preproc - saved artifact paths
#   risk_class      - the model class whose probability is the "risk" score.
#                     Binary models use 1; multi-class models use the adverse
#                     label so a high probability always means higher risk.
#   download_name   - filename used by the "Download Prediction Results" button
#   sample_file     - template CSV shipped in data/sample_inputs/
MODULES = {
    "Credit Risk": {
        "features": CREDIT_RISK_FEATURES,
        "model": CREDIT_RISK_MODEL,
        "preprocessor": CREDIT_RISK_PREPROCESSOR,
        "risk_class": "High",
        "download_name": "credit_risk_predictions.csv",
        "sample_file": os.path.join(SAMPLE_INPUTS_DIR, "credit_risk_sample.csv"),
    },
    "Fraud Detection": {
        "features": FRAUD_DETECTION_FEATURES,
        "model": FRAUD_DETECTION_MODEL,
        "preprocessor": FRAUD_DETECTION_PREPROCESSOR,
        "risk_class": 1,
        "download_name": "fraud_predictions.csv",
        "sample_file": os.path.join(SAMPLE_INPUTS_DIR, "fraud_sample.csv"),
    },
    "Customer Churn": {
        "features": CHURN_PREDICTION_FEATURES,
        "model": CHURN_PREDICTION_MODEL,
        "preprocessor": CHURN_PREDICTION_PREPROCESSOR,
        "risk_class": 1,
        "download_name": "churn_predictions.csv",
        "sample_file": os.path.join(SAMPLE_INPUTS_DIR, "churn_sample.csv"),
    },
    "Loan Collection Risk": {
        "features": COLLECTION_RISK_FEATURES,
        "model": COLLECTION_RISK_MODEL,
        "preprocessor": COLLECTION_RISK_PREPROCESSOR,
        "risk_class": "High Priority",
        "download_name": "collection_risk_predictions.csv",
        "sample_file": os.path.join(SAMPLE_INPUTS_DIR, "collection_risk_sample.csv"),
    },
}

# Display order for the module dropdown.
MODULE_NAMES = list(MODULES.keys())


def get_module_config(module_name):
    """Return the registry entry for a module (raises KeyError if unknown)."""
    return MODULES[module_name]


def _risk_probabilities(model, processed_X, risk_class):
    """Probability of the risk class for every row, as a 1-D array."""
    proba = model.predict_proba(processed_X)
    class_index = list(model.classes_).index(risk_class)
    return proba[:, class_index]


def run_batch_prediction(module_name, input_df):
    """
    Score every row in ``input_df`` with the chosen module's model.

    Parameters
    ----------
    module_name : str
        One of MODULE_NAMES.
    input_df : pandas.DataFrame
        Uploaded records. Must contain the module's required columns; extra
        columns are kept in the output but ignored by the model.

    Returns
    -------
    pandas.DataFrame
        A copy of the input with the four result columns appended.
    """
    config = MODULES[module_name]
    features = config["features"]

    missing = [col for col in features if col not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Load artifacts once for the whole batch (not per row).
    model = load_model(config["model"])
    preprocessor = load_preprocessor(config["preprocessor"])

    # Select features in the training order and score all rows at once.
    X = input_df[features]
    X_processed = preprocessor.transform(X)
    probabilities = _risk_probabilities(model, X_processed, config["risk_class"])

    # Map each probability to a business category + action.
    categories, actions = [], []
    for prob in probabilities:
        _level, category, action = classify(module_name, prob)
        categories.append(category)
        actions.append(action)

    result = input_df.copy()
    result["prediction_probability"] = probabilities.round(4)
    result["risk_category"] = categories
    result["business_action"] = actions
    result["model_module"] = module_name

    return result
