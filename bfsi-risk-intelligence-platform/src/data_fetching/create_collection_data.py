"""
Create loan collection risk dataset
Synthetic data based on loan repayment behavior patterns
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTION_RISK_RAW_FILE, RANDOM_STATE
from utils.common import save_data


def create_synthetic_collection_data(n_samples=1000):
    """
    Create synthetic collection risk dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of loan records to generate
    
    Returns:
    --------
    df : pandas DataFrame
        Synthetic collection risk data
    """
    np.random.seed(RANDOM_STATE)
    
    # Generate features
    data = {
        'loan_amount': np.random.exponential(100000, n_samples),
        'emi_amount': np.random.exponential(5000, n_samples),
        'monthly_income': np.random.exponential(50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'past_due_days': np.random.exponential(5, n_samples).astype(int),
        'missed_payments': np.random.randint(0, 6, n_samples),
        'loan_tenure_months': np.random.randint(6, 360, n_samples),
        'employment_type': np.random.choice(['Salaried', 'Self-employed', 'Contractual'], n_samples),
        'existing_loans_count': np.random.randint(0, 5, n_samples),
        'repayment_ratio': np.random.uniform(0.1, 1.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create collection risk based on patterns
    # Higher past_due_days, missed_payments, and lower credit_score = higher risk
    collection_risk_prob = (
        (df['past_due_days'] > 30) * 0.5 +
        (df['missed_payments'] > 2) * 0.4 +
        (df['credit_score'] < 550) * 0.35 +
        (df['repayment_ratio'] < 0.5) * 0.3 +
        (df['emi_amount'] > df['monthly_income'] * 0.5) * 0.25
    )
    
    # Bins span the full possible score range (0 up to ~1.8) and include the
    # lower edge, so no rows fall outside and become NaN.
    df['collection_risk'] = pd.cut(
        collection_risk_prob,
        bins=[-0.01, 0.33, 0.66, 2.0],
        labels=['Low Priority', 'Medium Priority', 'High Priority']
    )
    
    print(f"✓ Created synthetic collection risk data with {n_samples} samples")
    print(f"  Collection Risk Distribution:")
    print(df['collection_risk'].value_counts())
    
    return df


def create_collection_data():
    """
    Main function to create collection risk data
    
    Returns:
    --------
    df : pandas DataFrame
        Collection risk data
    """
    print("\n" + "="*50)
    print("Creating Collection Risk Data")
    print("="*50)
    
    print("Using synthetic collection risk data...")
    df = create_synthetic_collection_data(n_samples=1000)
    
    # Save the data
    save_data(df, COLLECTION_RISK_RAW_FILE)
    
    print(f"\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData Info:")
    print(df.info())
    
    return df


if __name__ == "__main__":
    create_collection_data()
