"""
Preprocessing for loan collection risk prediction
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
    COLLECTION_RISK_RAW_FILE, COLLECTION_RISK_PROCESSED_FILE,
    COLLECTION_RISK_PREPROCESSOR, COLLECTION_RISK_FEATURES
)
from utils.common import load_data, save_data, handle_missing_values, remove_outliers_iqr
from utils.model_io import save_preprocessor


def preprocess_collection_data():
    """
    Preprocess collection risk data for model training
    
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
    print("Preprocessing Collection Risk Data")
    print("="*50)
    
    # Load raw data
    df = load_data(COLLECTION_RISK_RAW_FILE)
    if df is None:
        print("✗ Failed to load data")
        return None, None, None
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Remove outliers from numeric columns
    numeric_outlier_cols = ['loan_amount', 'emi_amount', 'monthly_income']
    df = remove_outliers_iqr(df, numeric_outlier_cols)
    
    print(f"\nTarget Distribution (Collection Risk):")
    print(df['collection_risk'].value_counts())
    
    # Separate features and target. The target is kept as readable string
    # labels (Low/Medium/High Priority) so the model predicts the category
    # directly and the prediction layer needs no index-to-label mapping.
    X = df[COLLECTION_RISK_FEATURES].copy()
    y = df['collection_risk'].copy()

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
    
    # Combine for saving (use .values so the string labels align positionally
    # with the reset-index feature rows rather than by the original index)
    processed_df = X_processed.copy()
    processed_df['collection_risk'] = y.values
    
    # Save processed data
    save_data(processed_df, COLLECTION_RISK_PROCESSED_FILE)
    
    # Save preprocessor
    save_preprocessor(preprocessor, COLLECTION_RISK_PREPROCESSOR)
    
    return X_processed, y, preprocessor


if __name__ == "__main__":
    preprocess_collection_data()
