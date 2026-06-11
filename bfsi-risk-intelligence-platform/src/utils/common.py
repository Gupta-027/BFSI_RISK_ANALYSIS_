"""
Common utility functions for data handling and processing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load CSV data from file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    
    Returns:
    --------
    df : pandas DataFrame
        Loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Data loaded from: {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        return None


def save_data(df, filepath):
    """
    Save DataFrame to CSV
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data to save
    filepath : str
        Path to save the CSV
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to: {filepath}")


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    strategy : str
        'drop' - drop rows with missing values
        'mean' - fill numeric with mean
        'median' - fill numeric with median
        'forward_fill' - forward fill
    
    Returns:
    --------
    df : pandas DataFrame
        Data with missing values handled
    """
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("✓ No missing values found")
        return df
    
    print(f"Found {missing_count} missing values")
    
    if strategy == 'drop':
        df = df.dropna()
        print("✓ Dropped rows with missing values")
    elif strategy == 'mean':
        df = df.fillna(df.mean())
        print("✓ Filled missing numeric values with mean")
    elif strategy == 'median':
        df = df.fillna(df.median())
        print("✓ Filled missing numeric values with median")
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill')
        print("✓ Forward filled missing values")
    
    return df


def remove_outliers_iqr(df, columns, multiplier=1.5):
    """
    Remove outliers using Interquartile Range (IQR) method
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    columns : list
        Numeric columns to check
    multiplier : float
        IQR multiplier (1.5 is standard)
    
    Returns:
    --------
    df : pandas DataFrame
        Data without outliers
    """
    initial_size = len(df)
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    removed = initial_size - len(df)
    if removed > 0:
        print(f"✓ Removed {removed} outlier rows")
    
    return df


def encode_categorical(df, categorical_cols):
    """
    Encode categorical variables using LabelEncoder
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    categorical_cols : list
        Columns to encode
    
    Returns:
    --------
    df : pandas DataFrame
        Data with encoded categorical variables
    encoders : dict
        Dictionary of encoders for each column
    """
    encoders = {}
    
    for col in categorical_cols:
        if col in df.columns and df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    if encoders:
        print(f"✓ Encoded {len(encoders)} categorical columns")
    
    return df, encoders


def get_data_info(df):
    """
    Print basic information about the dataset
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input data
    """
    print(f"\n{'='*50}")
    print("Dataset Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
