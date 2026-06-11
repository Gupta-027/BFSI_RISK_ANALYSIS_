"""
Credit risk prediction - Load model and make predictions
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CREDIT_RISK_MODEL, CREDIT_RISK_PREPROCESSOR, CREDIT_RISK_FEATURES
)
from utils.model_io import load_model, load_preprocessor


def predict_credit_risk(input_data):
    """
    Predict credit risk for new customer
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Customer features
        Example:
        {
            'age': 35,
            'job': 'admin',
            'credit_amount': 5000,
            'duration': 24,
            'purpose': 'car',
            'housing': 'own',
            'saving_accounts': 'moderate',
            'checking_account': 'little'
        }
    
    Returns:
    --------
    dict : Prediction results with probability, category, and recommendation
    """
    
    try:
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Load model and preprocessor
        model = load_model(CREDIT_RISK_MODEL)
        preprocessor = load_preprocessor(CREDIT_RISK_PREPROCESSOR)
        
        # Ensure all features are present
        for feature in CREDIT_RISK_FEATURES:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select only required features
        X = df[CREDIT_RISK_FEATURES]
        
        # Preprocess (the fitted preprocessor expands categoricals via one-hot
        # encoding, so we pass its transformed output straight to the model)
        X_processed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Get probability for each class
        classes = model.classes_
        class_probabilities = {cls: prob for cls, prob in zip(classes, probability)}
        
        # Determine business recommendation
        if prediction == 'Low':
            recommendation = 'Approve'
        elif prediction == 'Medium':
            recommendation = 'Manual Review'
        else:
            recommendation = 'Reject'
        
        result = {
            'risk_category': prediction,
            'risk_probability': float(class_probabilities.get(prediction, 0)),
            'all_probabilities': class_probabilities,
            'business_recommendation': recommendation
        }
        
        return result
    
    except Exception as e:
        print(f"✗ Error in prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test with sample input
    sample_input = {
        'age': 35,
        'job': 'admin',
        'credit_amount': 5000,
        'duration': 24,
        'purpose': 'car',
        'housing': 'own',
        'saving_accounts': 'moderate',
        'checking_account': 'little'
    }
    
    print("\n" + "="*50)
    print("Credit Risk Prediction - Test")
    print("="*50)
    print(f"Input: {sample_input}")
    
    result = predict_credit_risk(sample_input)
    if result:
        print(f"\nPrediction Results:")
        print(f"Risk Category: {result['risk_category']}")
        print(f"Risk Probability: {result['risk_probability']:.4f}")
        print(f"Business Recommendation: {result['business_recommendation']}")
        print(f"\nAll Probabilities:")
        for category, prob in result['all_probabilities'].items():
            print(f"  {category}: {prob:.4f}")
