"""
Fetch and create customer churn prediction dataset
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHURN_PREDICTION_RAW_FILE, RANDOM_STATE
from utils.common import save_data


def create_synthetic_churn_data(n_samples=1000):
    """
    Create synthetic customer churn dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate
    
    Returns:
    --------
    df : pandas DataFrame
        Synthetic churn data
    """
    np.random.seed(RANDOM_STATE)
    
    # Generate features
    data = {
        'credit_score': np.random.randint(300, 850, n_samples),
        'age': np.random.randint(18, 92, n_samples),
        'tenure': np.random.randint(0, 10, n_samples),
        'balance': np.random.exponential(50000, n_samples),
        'number_of_products': np.random.randint(1, 5, n_samples),
        'has_credit_card': np.random.choice([0, 1], n_samples),
        'is_active_member': np.random.choice([0, 1], n_samples),
        'estimated_salary': np.random.exponential(75000, n_samples),
        'complaints': np.random.randint(0, 3, n_samples),
        'satisfaction_score': np.random.randint(1, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target with realistic patterns
    # Inactive members, low satisfaction, complaints increase churn likelihood
    churn_prob = (
        (1 - df['is_active_member']) * 0.4 +
        (df['complaints'] > 0) * 0.3 +
        (df['satisfaction_score'] < 3) * 0.35 +
        (df['tenure'] < 2) * 0.2 +
        (df['balance'] < 20000) * 0.1
    )
    
    df['exited'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    print(f"✓ Created synthetic churn data with {n_samples} samples")
    print(f"  - Churned: {df['exited'].sum()} ({df['exited'].sum()/n_samples*100:.1f}%)")
    print(f"  - Retained: {(1-df['exited']).sum()} ({(1-df['exited']).sum()/n_samples*100:.1f}%)")
    
    return df


def fetch_churn_data():
    """
    Main function to fetch or create churn prediction data
    
    Returns:
    --------
    df : pandas DataFrame
        Churn prediction data
    """
    print("\n" + "="*50)
    print("Fetching Customer Churn Data")
    print("="*50)
    
    print("Using synthetic churn data as fallback...")
    df = create_synthetic_churn_data(n_samples=1000)
    
    # Save the data
    save_data(df, CHURN_PREDICTION_RAW_FILE)
    
    print(f"\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nChurn Distribution:")
    print(df['exited'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    fetch_churn_data()
