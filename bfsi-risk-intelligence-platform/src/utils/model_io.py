"""
Model input/output utilities for saving and loading models
"""

import joblib
import os
from pathlib import Path


def save_model(model, filepath):
    """
    Save a trained model using joblib
    
    Parameters:
    -----------
    model : sklearn model object
        Trained model to save
    filepath : str
        Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath):
    """
    Load a saved model using joblib
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
    
    Returns:
    --------
    model : sklearn model object
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✓ Model loaded from: {filepath}")
    return model


def save_preprocessor(preprocessor, filepath):
    """
    Save preprocessing pipeline (e.g., ColumnTransformer, StandardScaler)
    
    Parameters:
    -----------
    preprocessor : sklearn transformer object
        Preprocessing pipeline
    filepath : str
        Path to save the preprocessor
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"✓ Preprocessor saved to: {filepath}")


def load_preprocessor(filepath):
    """
    Load a saved preprocessor pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to the saved preprocessor
    
    Returns:
    --------
    preprocessor : sklearn transformer object
        Loaded preprocessor
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
    
    preprocessor = joblib.load(filepath)
    print(f"✓ Preprocessor loaded from: {filepath}")
    return preprocessor


def check_model_exists(filepath):
    """
    Check if a model file exists
    
    Parameters:
    -----------
    filepath : str
        Path to check
    
    Returns:
    --------
    bool : True if model exists, False otherwise
    """
    return os.path.exists(filepath)
