"""
Customer churn prediction - Load model and make predictions
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHURN_PREDICTION_MODEL, CHURN_PREDICTION_PREPROCESSOR, CHURN_PREDICTION_FEATURES
)
from utils.model_io import load_model, load_preprocessor


def predict_churn(input_data):
    """
    Predict customer churn risk
    
    Parameters:
    -----------
    input_data : dict or pd.DataFrame
        Customer features
        Example:
        {
            'credit_score': 750,
            'age': 42,
            'tenure': 5,
            'balance': 100000,
            'number_of_products': 2,
            'has_credit_card': 1,
            'is_active_member': 1,
            'estimated_salary': 150000,
            'complaints': 0,
            'satisfaction_score': 5
        }
    
    Returns:
    --------
    dict : Prediction results with churn probability, category, and retention suggestion
    """
    
    try:
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Load model and preprocessor
        model = load_model(CHURN_PREDICTION_MODEL)
        preprocessor = load_preprocessor(CHURN_PREDICTION_PREPROCESSOR)
        
        # Ensure all features are present
        for feature in CHURN_PREDICTION_FEATURES:
            if feature not in df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Select only required features
        X = df[CHURN_PREDICTION_FEATURES]
        
        # Preprocess (pass the transformed output straight to the model)
        X_processed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Get churn probability
        churn_probability = probability[1]  # Probability of churn (class 1)
        
        # Determine churn risk category
        if churn_probability < 0.33:
            churn_risk = 'Low'
        elif churn_probability < 0.67:
            churn_risk = 'Medium'
        else:
            churn_risk = 'High'
        
        # Determine retention suggestion
        if churn_risk == 'Low':
            retention_suggestion = 'No action'
        elif churn_risk == 'Medium':
            retention_suggestion = 'Send offer'
        else:
            retention_suggestion = 'Priority relationship manager call'
        
        result = {
            'will_churn': prediction,
            'churn_probability': float(churn_probability),
            'churn_risk_category': churn_risk,
            'retention_suggestion': retention_suggestion
        }
        
        return result
    
    except Exception as e:
        print(f"✗ Error in prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test with sample inputs
    sample_loyal = {
        'credit_score': 800,
        'age': 45,
        'tenure': 8,
        'balance': 150000,
        'number_of_products': 3,
        'has_credit_card': 1,
        'is_active_member': 1,
        'estimated_salary': 200000,
        'complaints': 0,
        'satisfaction_score': 5
    }
    
    sample_at_risk = {
        'credit_score': 500,
        'age': 30,
        'tenure': 1,
        'balance': 0,
        'number_of_products': 1,
        'has_credit_card': 0,
        'is_active_member': 0,
        'estimated_salary': 50000,
        'complaints': 2,
        'satisfaction_score': 2
    }
    
    print("\n" + "="*50)
    print("Customer Churn Prediction - Test")
    print("="*50)
    
    for sample_name, sample_input in [("Loyal Customer", sample_loyal), ("At-Risk Customer", sample_at_risk)]:
        print(f"\n{sample_name}:")
        result = predict_churn(sample_input)
        if result:
            print(f"  Churn Probability: {result['churn_probability']:.4f}")
            print(f"  Churn Risk Category: {result['churn_risk_category']}")
            print(f"  Retention Suggestion: {result['retention_suggestion']}")
