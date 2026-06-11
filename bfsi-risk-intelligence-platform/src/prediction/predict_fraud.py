"""
Fraud detection prediction - Load model and make predictions
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FRAUD_DETECTION_MODEL, FRAUD_DETECTION_PREPROCESSOR, FRAUD_DETECTION_FEATURES
)
from utils.model_io import load_model, load_preprocessor


def predict_fraud(input_data):
    """
    Predict transaction fraud risk
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Transaction features
        Example:
        {
            'transaction_amount': 100,
            'transaction_hour': 2,
            'old_balance': 5000,
            'new_balance': 4900,
            'merchant_risk_score': 15,
            'device_changed': 0,
            'location_changed': 0,
            'is_new_beneficiary': 0,
            'failed_attempts_last_24h': 0
        }
    
    Returns:
    --------
    dict : Prediction results with fraud probability, category, and action
    """
    
    try:
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Load model and preprocessor
        model = load_model(FRAUD_DETECTION_MODEL)
        preprocessor = load_preprocessor(FRAUD_DETECTION_PREPROCESSOR)
        
        # Ensure all features are present
        for feature in FRAUD_DETECTION_FEATURES:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select only required features
        X = df[FRAUD_DETECTION_FEATURES]
        
        # Preprocess (pass the transformed output straight to the model)
        X_processed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Get fraud probability
        fraud_probability = probability[1]  # Probability of fraud (class 1)
        
        # Determine fraud category based on probability
        if fraud_probability < 0.33:
            fraud_category = 'Safe'
        elif fraud_probability < 0.67:
            fraud_category = 'Suspicious'
        else:
            fraud_category = 'High Fraud Risk'
        
        # Determine action
        if fraud_category == 'Safe':
            action = 'Allow Transaction'
        elif fraud_category == 'Suspicious':
            action = 'Send OTP / Manual Review'
        else:
            action = 'Block Temporarily'
        
        result = {
            'is_fraud': prediction,
            'fraud_probability': float(fraud_probability),
            'fraud_category': fraud_category,
            'action': action
        }
        
        return result
    
    except Exception as e:
        print(f"✗ Error in prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test with sample inputs
    sample_safe = {
        'transaction_amount': 50,
        'transaction_hour': 10,
        'old_balance': 5000,
        'new_balance': 4950,
        'merchant_risk_score': 10,
        'device_changed': 0,
        'location_changed': 0,
        'is_new_beneficiary': 0,
        'failed_attempts_last_24h': 0
    }
    
    sample_suspicious = {
        'transaction_amount': 500,
        'transaction_hour': 3,
        'old_balance': 5000,
        'new_balance': 4500,
        'merchant_risk_score': 75,
        'device_changed': 1,
        'location_changed': 1,
        'is_new_beneficiary': 1,
        'failed_attempts_last_24h': 2
    }
    
    print("\n" + "="*50)
    print("Fraud Detection Prediction - Test")
    print("="*50)
    
    for sample_name, sample_input in [("Safe Transaction", sample_safe), ("Suspicious Transaction", sample_suspicious)]:
        print(f"\n{sample_name}:")
        result = predict_fraud(sample_input)
        if result:
            print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"  Fraud Category: {result['fraud_category']}")
            print(f"  Action: {result['action']}")
