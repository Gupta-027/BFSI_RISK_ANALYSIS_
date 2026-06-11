"""
Preprocessing for customer churn prediction
Handles missing values and scaling
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CHURN_PREDICTION_RAW_FILE, CHURN_PREDICTION_PROCESSED_FILE,
    CHURN_PREDICTION_PREPROCESSOR, CHURN_PREDICTION_FEATURES
)
from utils.common import load_data, save_data, handle_missing_values, remove_outliers_iqr
from utils.model_io import save_preprocessor


def preprocess_churn_data():
    """
    Preprocess customer churn data for model training
    
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
    print("Preprocessing Customer Churn Data")
    print("="*50)
    
    # Load raw data
    df = load_data(CHURN_PREDICTION_RAW_FILE)
    if df is None:
        print("✗ Failed to load data")
        return None, None, None
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Remove outliers from numeric columns
    numeric_outlier_cols = ['balance', 'estimated_salary']
    df = remove_outliers_iqr(df, numeric_outlier_cols)
    
    # Separate features and target
    X = df[CHURN_PREDICTION_FEATURES].copy()
    y = df['exited'].copy()
    
    print(f"\nTarget Distribution (Churn):")
    print(y.value_counts())
    print(f"Churn Rate: {y.mean()*100:.2f}%")
    
    # All features are numeric for churn prediction
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
    processed_df['exited'] = y.values
    
    # Save processed data
    save_data(processed_df, CHURN_PREDICTION_PROCESSED_FILE)
    
    # Save preprocessor
    save_preprocessor(preprocessor, CHURN_PREDICTION_PREPROCESSOR)
    
    return X_processed, y, preprocessor


if __name__ == "__main__":
    preprocess_churn_data()
