"""
Model evaluation metrics for BFSI models
"""

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
import pandas as pd


def evaluate_classification_model(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Evaluate a classification model with standard metrics
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities (for ROC-AUC)
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Add ROC-AUC if probabilities are available and binary classification
    if y_pred_proba is not None and len(set(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{model_name} - Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\nMetrics Summary:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return metrics


def evaluate_imbalanced_classification(y_true, y_pred, y_pred_proba=None, model_name="Model"):
    """
    Special evaluation for imbalanced datasets (e.g., fraud detection)
    Emphasizes precision, recall, F1, and ROC-AUC over accuracy
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    
    metrics = {
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    # ROC-AUC for binary classification
    if y_pred_proba is not None and len(set(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    print(f"\n{'='*50}")
    print(f"{model_name} - Imbalanced Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\nKey Metrics for Imbalanced Data:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return metrics
