"""
Loan collection risk prediction - Load model and make predictions
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    COLLECTION_RISK_MODEL, COLLECTION_RISK_PREPROCESSOR, COLLECTION_RISK_FEATURES
)
from utils.model_io import load_model, load_preprocessor


def predict_collection_risk(input_data):
    """
    Predict EMI delay and collection priority
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Loan features
        Example:
        {
            'loan_amount': 500000,
            'emi_amount': 10000,
            'monthly_income': 100000,
            'credit_score': 700,
            'past_due_days': 15,
            'missed_payments': 1,
            'loan_tenure_months': 60,
            'employment_type': 'Salaried',
            'existing_loans_count': 1,
            'repayment_ratio': 0.1
        }
    
    Returns:
    --------
    dict : Prediction results with delay probability, collection priority, and action
    """
    
    try:
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Load model and preprocessor
        model = load_model(COLLECTION_RISK_MODEL)
        preprocessor = load_preprocessor(COLLECTION_RISK_PREPROCESSOR)
        
        # Ensure all features are present
        for feature in COLLECTION_RISK_FEATURES:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select only required features
        X = df[COLLECTION_RISK_FEATURES]
        
        # Preprocess (pass the transformed output straight to the model)
        X_processed = preprocessor.transform(X)
        
        # The model predicts the priority label directly (e.g. "High Priority")
        collection_priority = model.predict(X_processed)[0]

        # Probability the model assigned to the predicted class
        probability = model.predict_proba(X_processed)[0]
        classes = list(model.classes_)
        delay_probability = probability[classes.index(collection_priority)]
        
        # Determine action
        if collection_priority == 'Low Priority':
            action = 'Normal reminder'
        elif collection_priority == 'Medium Priority':
            action = 'Follow-up call'
        else:
            action = 'Collection team escalation'
        
        result = {
            'emi_delay_probability': float(delay_probability),
            'collection_priority': collection_priority,
            'action': action
        }
        
        return result
    
    except Exception as e:
        print(f"✗ Error in prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test with sample inputs
    sample_low_risk = {
        'loan_amount': 500000,
        'emi_amount': 5000,
        'monthly_income': 150000,
        'credit_score': 750,
        'past_due_days': 0,
        'missed_payments': 0,
        'loan_tenure_months': 60,
        'employment_type': 'Salaried',
        'existing_loans_count': 1,
        'repayment_ratio': 0.05
    }
    
    sample_high_risk = {
        'loan_amount': 1000000,
        'emi_amount': 50000,
        'monthly_income': 80000,
        'credit_score': 400,
        'past_due_days': 45,
        'missed_payments': 3,
        'loan_tenure_months': 120,
        'employment_type': 'Self-employed',
        'existing_loans_count': 3,
        'repayment_ratio': 0.8
    }
    
    print("\n" + "="*50)
    print("Collection Risk Prediction - Test")
    print("="*50)
    
    for sample_name, sample_input in [("Low Risk Loan", sample_low_risk), ("High Risk Loan", sample_high_risk)]:
        print(f"\n{sample_name}:")
        result = predict_collection_risk(sample_input)
        if result:
            print(f"  EMI Delay Probability: {result['emi_delay_probability']:.4f}")
            print(f"  Collection Priority: {result['collection_priority']}")
            print(f"  Action: {result['action']}")
