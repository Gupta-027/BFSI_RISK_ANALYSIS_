"""
Preprocessing for credit risk prediction
Handles missing values, encoding, and scaling
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CREDIT_RISK_RAW_FILE, CREDIT_RISK_PROCESSED_FILE,
    CREDIT_RISK_PREPROCESSOR, CREDIT_RISK_FEATURES
)
from utils.common import load_data, save_data, handle_missing_values, remove_outliers_iqr
from utils.model_io import save_preprocessor


def preprocess_credit_data():
    """
    Preprocess credit risk data for model training
    
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
    print("Preprocessing Credit Risk Data")
    print("="*50)
    
    # Load raw data
    df = load_data(CREDIT_RISK_RAW_FILE)
    if df is None:
        print("✗ Failed to load data")
        return None, None, None
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Remove outliers from numeric columns
    numeric_cols = ['age', 'credit_amount', 'duration']
    df = remove_outliers_iqr(df, numeric_cols)
    
    # Separate features and target
    X = df[CREDIT_RISK_FEATURES].copy()
    y = df['risk'].copy()
    
    print(f"\nTarget Distribution:")
    print(y.value_counts())
    
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {categorical_cols}")
    print(f"Numeric columns: {numeric_cols}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols) if categorical_cols else ('cat', 'passthrough', []),
            ('num', StandardScaler(), numeric_cols) if numeric_cols else ('num', 'passthrough', [])
        ],
        remainder='passthrough'
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = []
    # Add one-hot encoded feature names
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_features = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
    # Add numeric feature names
    if numeric_cols:
        feature_names.extend(numeric_cols)
    
    # If no features generated (shouldn't happen), create dummy names
    if len(feature_names) == 0:
        feature_names = [f'feature_{i}' for i in range(X_processed.shape[1])]
    
    # Ensure feature names match X_processed dimensions
    while len(feature_names) < X_processed.shape[1]:
        feature_names.append(f'feature_{len(feature_names)}')
    feature_names = feature_names[:X_processed.shape[1]]
    
    # Convert to DataFrame
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    print(f"\n✓ Preprocessing completed")
    print(f"  Input shape: {df.shape}")
    print(f"  Features shape: {X_processed.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Combine for saving
    processed_df = X_processed.copy()
    processed_df['risk'] = y.values
    
    # Save processed data
    save_data(processed_df, CREDIT_RISK_PROCESSED_FILE)
    
    # Save preprocessor
    save_preprocessor(preprocessor, CREDIT_RISK_PREPROCESSOR)
    
    return X_processed, y, preprocessor


if __name__ == "__main__":
    preprocess_credit_data()
