"""
Train credit risk prediction model
Trains Logistic Regression and Random Forest, selects best model
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CREDIT_RISK_PROCESSED_FILE, CREDIT_RISK_MODEL, RANDOM_STATE, TEST_SIZE
)
from utils.common import load_data
from utils.metrics import evaluate_classification_model
from utils.model_io import save_model
from preprocessing.credit_preprocessing import preprocess_credit_data


def train_credit_risk_model():
    """
    Train and evaluate credit risk models
    
    Returns:
    --------
    best_model : sklearn model
        Best trained model
    """
    
    print("\n" + "="*50)
    print("Training Credit Risk Model")
    print("="*50)
    
    # Run preprocessing first so the processed data and saved preprocessor are
    # always in sync with the current raw data.
    preprocess_credit_data()

    # Load preprocessed data
    df = load_data(CREDIT_RISK_PROCESSED_FILE)
    if df is None:
        print("✗ Preprocessed data not found. Run preprocessing first.")
        return None
    
    # Separate features and target
    X = df.drop('risk', axis=1)
    y = df['risk']
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train Logistic Regression
    print(f"\n{'='*50}")
    print("Model 1: Logistic Regression")
    print('='*50)
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_classification_model(y_test, lr_pred, lr_pred_proba, "Logistic Regression")
    
    # Train Random Forest
    print(f"\n{'='*50}")
    print("Model 2: Random Forest Classifier")
    print('='*50)
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_classification_model(y_test, rf_pred, rf_pred_proba, "Random Forest")
    
    # Select best model based on F1 score
    print(f"\n{'='*50}")
    print("Model Comparison")
    print('='*50)
    print(f"Logistic Regression F1: {lr_metrics['f1']:.4f}")
    print(f"Random Forest F1: {rf_metrics['f1']:.4f}")
    
    if rf_metrics['f1'] >= lr_metrics['f1']:
        best_model = rf_model
        best_name = "Random Forest"
        print(f"\n✓ Selected: {best_name}")
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        print(f"\n✓ Selected: {best_name}")
    
    # Save best model
    save_model(best_model, CREDIT_RISK_MODEL)
    
    return best_model


if __name__ == "__main__":
    train_credit_risk_model()
