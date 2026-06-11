"""
Train fraud detection model
Trains models with emphasis on precision, recall, and F1 for imbalanced data
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FRAUD_DETECTION_PROCESSED_FILE, FRAUD_DETECTION_MODEL, RANDOM_STATE, TEST_SIZE
)
from utils.common import load_data
from utils.metrics import evaluate_imbalanced_classification
from utils.model_io import save_model
from preprocessing.fraud_preprocessing import preprocess_fraud_data


def train_fraud_model():
    """
    Train and evaluate fraud detection models
    For imbalanced data, emphasizes recall and precision over accuracy
    
    Returns:
    --------
    best_model : sklearn model
        Best trained model
    """
    
    print("\n" + "="*50)
    print("Training Fraud Detection Model")
    print("="*50)
    
    # Run preprocessing first to keep processed data and preprocessor in sync.
    preprocess_fraud_data()

    # Load preprocessed data
    df = load_data(FRAUD_DETECTION_PROCESSED_FILE)
    if df is None:
        print("✗ Preprocessed data not found. Run preprocessing first.")
        return None
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Class balance: {y.mean()*100:.2f}% fraud")
    
    # Train-test split (stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape} - Fraud: {y_train.mean()*100:.2f}%")
    print(f"Test set: {X_test.shape} - Fraud: {y_test.mean()*100:.2f}%")
    
    # Train Logistic Regression with class weight balancing
    print(f"\n{'='*50}")
    print("Model 1: Logistic Regression (Balanced)")
    print('='*50)
    lr_model = LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)
    lr_metrics = evaluate_imbalanced_classification(y_test, lr_pred, lr_pred_proba, "Logistic Regression")
    
    # Train Random Forest with class weight balancing
    print(f"\n{'='*50}")
    print("Model 2: Random Forest (Balanced)")
    print('='*50)
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    rf_metrics = evaluate_imbalanced_classification(y_test, rf_pred, rf_pred_proba, "Random Forest")
    
    # Select best model based on F1 score (better for imbalanced data)
    print(f"\n{'='*50}")
    print("Model Comparison (Imbalanced Data)")
    print('='*50)
    print(f"Logistic Regression - Precision: {lr_metrics['precision']:.4f}, Recall: {lr_metrics['recall']:.4f}, F1: {lr_metrics['f1']:.4f}")
    print(f"Random Forest - Precision: {rf_metrics['precision']:.4f}, Recall: {rf_metrics['recall']:.4f}, F1: {rf_metrics['f1']:.4f}")
    
    if rf_metrics['f1'] >= lr_metrics['f1']:
        best_model = rf_model
        best_name = "Random Forest"
        print(f"\n✓ Selected: {best_name}")
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        print(f"\n✓ Selected: {best_name}")
    
    # Save best model
    save_model(best_model, FRAUD_DETECTION_MODEL)
    
    return best_model


if __name__ == "__main__":
    train_fraud_model()
