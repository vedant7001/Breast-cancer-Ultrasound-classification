import torch
import numpy as np
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(labels, predictions, probabilities=None):
    """
    Compute classification metrics
    
    Args:
        labels (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Predicted labels
        probabilities (numpy.ndarray, optional): Predicted class probabilities
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Basic metrics
    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions, average='weighted', zero_division=0)),
        'recall': float(recall_score(labels, predictions, average='weighted', zero_division=0)),
        'f1': float(f1_score(labels, predictions, average='weighted', zero_division=0))
    }
    
    # ROC-AUC (if probabilities are provided)
    if probabilities is not None:
        if probabilities.shape[1] == 2:  # Binary classification
            metrics['roc_auc'] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:  # Multi-class classification
            metrics['roc_auc'] = float(roc_auc_score(
                np.eye(probabilities.shape[1])[labels.astype(int)],
                probabilities,
                multi_class='ovr',
                average='weighted'
            ))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Detailed classification report
    report = classification_report(labels, predictions, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


def save_metrics(metrics, save_path):
    """
    Save metrics to a JSON file
    
    Args:
        metrics (dict): Dictionary with metrics
        save_path (str): Path to save the metrics
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def load_metrics(metrics_path):
    """
    Load metrics from a JSON file
    
    Args:
        metrics_path (str): Path to the metrics file
        
    Returns:
        dict: Dictionary with metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def calculate_params_flops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Calculate model parameters and FLOPS
    
    Args:
        model (torch.nn.Module): Model to evaluate
        input_size (tuple): Input tensor shape
        device (str): Device to use
        
    Returns:
        tuple: (num_params, num_flops)
    """
    try:
        from thop import profile
        
        device = torch.device(device)
        model.to(device)
        model.eval()
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        input_tensor = torch.randn(input_size).to(device)
        flops, _ = profile(model, inputs=(input_tensor,))
        
        return num_params, flops
    
    except ImportError:
        print("thop package is required for FLOPS calculation.")
        print("Install it with: pip install thop")
        
        # Just return parameters count if thop is not available
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_params, None


def k_fold_cross_validation_indices(n_samples, n_folds=5, random_state=42):
    """
    Generate indices for k-fold cross-validation
    
    Args:
        n_samples (int): Number of samples
        n_folds (int): Number of folds
        random_state (int): Random seed
        
    Returns:
        list: List of (train_indices, val_indices) tuples
    """
    np.random.seed(random_state)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_folds
    folds = []
    
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])
        folds.append((train_indices, val_indices))
    
    return folds 