"""
Preprocessing for fraud detection
Handles missing values, encoding, and scaling for imbalanced data
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FRAUD_DETECTION_RAW_FILE, FRAUD_DETECTION_PROCESSED_FILE,
    FRAUD_DETECTION_PREPROCESSOR, FRAUD_DETECTION_FEATURES
)
from utils.common import load_data, save_data, handle_missing_values, remove_outliers_iqr
from utils.model_io import save_preprocessor


def preprocess_fraud_data():
    """
    Preprocess fraud detection data for model training
    Special care for imbalanced data
    
    Returns:
    --------
    X : pandas DataFrame
        Features
    y : pandas Series
        Target variable
    preprocessor : ColumnTransformer
        Preprocessing pipeline
    """
    
    print("\n" + "="*50)
    print("Preprocessing Fraud Detection Data")
    print("="*50)
    
    # Load raw data
    df = load_data(FRAUD_DETECTION_RAW_FILE)
    if df is None:
        print("✗ Failed to load data")
        return None, None, None
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    print(f"\nClass Distribution (Imbalanced):")
    print(df['is_fraud'].value_counts())
    print(f"Fraud Ratio: {df['is_fraud'].mean()*100:.2f}%")
    
    # Skip aggressive outlier removal for fraud data to preserve fraud samples
    # Only remove extreme outliers from transaction_amount and balances
    numeric_cols_list = ['transaction_amount', 'old_balance', 'new_balance']
    for col in numeric_cols_list:
        Q1 = df[col].quantile(0.01)  # 1st percentile instead of 25th for fraud data
        Q3 = df[col].quantile(0.99)  # 99th percentile instead of 75th
        IQR = Q3 - Q1
        lower_bound = Q1 - 10 * IQR  # 10x multiplier instead of 1.5x
        upper_bound = Q3 + 10 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    print(f"\n✓ Applied conservative outlier removal (preserved fraud cases)")
    
    # Verify fraud samples still exist
    if df['is_fraud'].sum() == 0:
        print("⚠ Warning: All fraud samples were removed! Using all data.")
    
    # Separate features and target
    X = df[FRAUD_DETECTION_FEATURES].copy()
    y = df['is_fraud'].copy()
    
    # All features are numeric for fraud detection
    numeric_cols_all = X.columns.tolist()
    
    print(f"\nNumeric features: {numeric_cols_all}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numeric_cols_all)
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    X_processed = pd.DataFrame(X_processed, columns=numeric_cols_all)
    
    print(f"\n✓ Preprocessing completed")
    print(f"  Input shape: {df.shape}")
    print(f"  Features shape: {X_processed.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Combine for saving
    processed_df = X_processed.copy()
    processed_df['is_fraud'] = y.values
    
    # Save processed data
    save_data(processed_df, FRAUD_DETECTION_PROCESSED_FILE)
    
    # Save preprocessor
    save_preprocessor(preprocessor, FRAUD_DETECTION_PREPROCESSOR)
    
    return X_processed, y, preprocessor


if __name__ == "__main__":
    preprocess_fraud_data()
