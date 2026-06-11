"""
Fetch and create fraud detection dataset
Attempts to download from Kaggle or creates synthetic imbalanced data as fallback
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FRAUD_DETECTION_RAW_FILE, RANDOM_STATE
from utils.common import save_data


def create_synthetic_fraud_data(n_samples=5000, fraud_ratio=0.05):
    """
    Create synthetic fraud detection dataset with class imbalance
    
    Parameters:
    -----------
    n_samples : int
        Total number of records to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (0.05 = 5%)
    
    Returns:
    --------
    df : pandas DataFrame
        Synthetic fraud data
    """
    np.random.seed(RANDOM_STATE)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legitimate transactions
    legit_data = {
        'transaction_amount': np.random.exponential(100, n_legit),
        'transaction_hour': np.random.randint(0, 24, n_legit),
        'old_balance': np.random.exponential(5000, n_legit),
        'new_balance': np.random.exponential(5000, n_legit),
        'merchant_risk_score': np.random.uniform(0, 30, n_legit),
        'device_changed': np.random.choice([0, 1], n_legit, p=[0.95, 0.05]),
        'location_changed': np.random.choice([0, 1], n_legit, p=[0.98, 0.02]),
        'is_new_beneficiary': np.random.choice([0, 1], n_legit, p=[0.9, 0.1]),
        'failed_attempts_last_24h': np.random.randint(0, 2, n_legit),
        'is_fraud': 0
    }
    
    # Fraudulent transactions
    fraud_data = {
        'transaction_amount': np.random.exponential(500, n_fraud),  # Higher amounts
        'transaction_hour': np.random.choice([2, 3, 4, 5], n_fraud),  # Unusual hours
        'old_balance': np.random.exponential(2000, n_fraud),
        'new_balance': np.random.exponential(2000, n_fraud),
        'merchant_risk_score': np.random.uniform(50, 100, n_fraud),  # Higher risk
        'device_changed': np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),  # Often changed
        'location_changed': np.random.choice([0, 1], n_fraud, p=[0.4, 0.6]),  # Often changed
        'is_new_beneficiary': np.random.choice([0, 1], n_fraud, p=[0.2, 0.8]),  # Often new
        'failed_attempts_last_24h': np.random.randint(1, 5, n_fraud),  # Multiple failures
        'is_fraud': 1
    }
    
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    print(f"✓ Created synthetic fraud data with {n_samples} samples")
    print(f"  - Legitimate: {n_legit} ({100-fraud_ratio*100:.1f}%)")
    print(f"  - Fraudulent: {n_fraud} ({fraud_ratio*100:.1f}%)")
    
    return df


def fetch_fraud_data():
    """
    Main function to fetch or create fraud detection data
    
    Returns:
    --------
    df : pandas DataFrame
        Fraud detection data
    """
    print("\n" + "="*50)
    print("Fetching Fraud Detection Data")
    print("="*50)
    
    print("Using synthetic fraud detection data as fallback...")
    df = create_synthetic_fraud_data(n_samples=5000, fraud_ratio=0.05)
    
    # Save the data
    save_data(df, FRAUD_DETECTION_RAW_FILE)
    
    print(f"\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nFraud Distribution:")
    print(df['is_fraud'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    fetch_fraud_data()
