"""
Fetch and create credit risk dataset
Attempts to download from UCI or creates synthetic data as fallback
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CREDIT_RISK_RAW_FILE, RANDOM_STATE
from utils.common import save_data


def create_synthetic_credit_data(n_samples=1000):
    """
    Create synthetic credit risk dataset for demonstration
    
    Parameters:
    -----------
    n_samples : int
        Number of records to generate
    
    Returns:
    --------
    df : pandas DataFrame
        Synthetic credit data
    """
    np.random.seed(RANDOM_STATE)

    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'job': np.random.choice(['admin', 'technician', 'services', 'management', 'retired', 'self-employed', 'unemployed', 'unknown'], n_samples),
        'credit_amount': np.random.randint(250, 20000, n_samples),
        'duration': np.random.randint(4, 72, n_samples),
        'purpose': np.random.choice(['housing', 'car', 'education', 'personal', 'business', 'unknown'], n_samples),
        'housing': np.random.choice(['own', 'rent', 'other'], n_samples),
        'saving_accounts': np.random.choice(['little', 'moderate', 'rich', 'unknown'], n_samples),
        'checking_account': np.random.choice(['little', 'moderate', 'rich', 'unknown'], n_samples),
    })

    # Build a realistic risk score from the features so the model has a real
    # signal to learn (a higher score means a riskier customer).
    savings_weight = {'little': 2, 'unknown': 1, 'moderate': 0, 'rich': -1}
    checking_weight = {'little': 2, 'unknown': 1, 'moderate': 0, 'rich': -1}
    job_weight = {'unemployed': 2, 'unknown': 1, 'self-employed': 1, 'services': 0,
                  'technician': 0, 'admin': 0, 'retired': 1, 'management': -1}
    housing_weight = {'own': -1, 'rent': 1, 'other': 1}

    score = (
        (df['credit_amount'] / 5000)                       # larger loans are riskier
        + (df['duration'] / 18)                            # longer tenures are riskier
        + df['saving_accounts'].map(savings_weight)
        + df['checking_account'].map(checking_weight)
        + df['job'].map(job_weight)
        + df['housing'].map(housing_weight)
        + (df['age'] < 25).astype(int)                     # very young borrowers
        + np.random.normal(0, 1.0, n_samples)              # noise so it is not perfectly separable
    )

    # Convert the continuous score into Low / Medium / High using tertiles,
    # which keeps the classes reasonably balanced.
    low_cut, high_cut = score.quantile([0.55, 0.82])
    df['risk'] = np.where(score <= low_cut, 'Low',
                  np.where(score <= high_cut, 'Medium', 'High'))

    print(f"✓ Created synthetic credit risk data with {n_samples} samples")
    print(f"  Risk distribution: {df['risk'].value_counts().to_dict()}")
    return df


def fetch_credit_data():
    """
    Main function to fetch or create credit risk data
    
    Returns:
    --------
    df : pandas DataFrame
        Credit risk data
    """
    print("\n" + "="*50)
    print("Fetching Credit Risk Data")
    print("="*50)
    
    # Try to download from UCI or Kaggle (commented as it requires API keys)
    # For this project, we use synthetic data
    
    print("Using synthetic credit risk data as fallback...")
    df = create_synthetic_credit_data(n_samples=1000)
    
    # Save the data
    save_data(df, CREDIT_RISK_RAW_FILE)
    
    print(f"\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData Info:")
    print(df.info())
    
    return df


if __name__ == "__main__":
    fetch_credit_data()
