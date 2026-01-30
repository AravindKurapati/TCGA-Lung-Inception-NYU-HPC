"""Evaluation metrics"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

def compute_auc(model, images, labels, dataset_name=""):
    """Compute AUC score"""
    
    if len(np.unique(labels)) <= 1:
        print(f"{dataset_name} labels contain only one class - cannot compute AUC")
        return None
    
    predictions = model.predict(images).ravel()
    auc = roc_auc_score(labels, predictions)
    print(f"{dataset_name} AUC: {auc:.4f}")
    
    return auc

def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute comprehensive metrics"""
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # AUC (if probabilities provided)
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics
