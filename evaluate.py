import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from models import get_model
from dataloader import get_dataloaders


def load_model(model_path, model_name, variant=None, num_classes=2, device=None):
    """
    Load a trained model from disk
    
    Args:
        model_path (str): Path to the model weights (.pt file)
        model_name (str): Name of the model architecture
        variant (str, optional): Specific variant of the model
        num_classes (int): Number of classes
        device (str, optional): Device to load model to
        
    Returns:
        torch.nn.Module: Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Create model architecture
    model = get_model(model_name, num_classes=num_classes, pretrained=False, variant=variant)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device=None):
    """
    Evaluate model on a dataset
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data
        device (str, optional): Device to use for evaluation
        
    Returns:
        tuple: (metrics_dict, predictions, labels, probabilities)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    # Calculate ROC-AUC score (only for binary classification)
    if len(np.unique(all_labels)) == 2:
        metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        # For multi-class, we compute one-vs-rest ROC AUC and average
        metrics['roc_auc'] = roc_auc_score(
            np.eye(len(np.unique(all_labels)))[all_labels],
            all_probs,
            multi_class='ovr',
            average='weighted'
        )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Generate detailed classification report
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    metrics['classification_report'] = class_report
    
    return metrics, all_preds, all_labels, all_probs


def save_evaluation_results(metrics, preds, labels, probs, save_dir, model_name):
    """
    Save evaluation results to disk
    
    Args:
        metrics (dict): Dictionary with evaluation metrics
        preds (numpy.ndarray): Model predictions
        labels (numpy.ndarray): Ground truth labels
        probs (numpy.ndarray): Probability outputs
        save_dir (str): Directory to save results
        model_name (str): Name of the model for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions, labels, and probabilities as numpy array
    predictions_path = os.path.join(save_dir, f'{model_name}_predictions.npz')
    np.savez(
        predictions_path,
        predictions=preds,
        labels=labels,
        probabilities=probs
    )
    
    print(f"Saved evaluation results to {save_dir}")
    print(f"Metrics summary:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"  {metric}: {metrics[metric]:.4f}")


def evaluate_multiple_models(model_paths, data_dir, save_dir='results', batch_size=32):
    """
    Evaluate multiple models on the same test dataset
    
    Args:
        model_paths (dict): Dictionary mapping model names to (weight_path, model_type, variant)
        data_dir (str): Path to dataset directory
        save_dir (str): Directory to save results
        batch_size (int): Batch size for evaluation
        
    Returns:
        dict: Dictionary with evaluation metrics for each model
    """
    # Get dataloaders
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size)
    test_dataloader = dataloaders['test']
    
    all_metrics = {}
    
    for model_name, (weight_path, model_type, variant) in model_paths.items():
        print(f"Evaluating {model_name}...")
        
        # Load model
        model = load_model(weight_path, model_type, variant)
        
        # Evaluate
        metrics, preds, labels, probs = evaluate_model(model, test_dataloader)
        
        # Save results
        save_evaluation_results(metrics, preds, labels, probs, save_dir, model_name)
        
        # Store metrics
        all_metrics[model_name] = metrics
    
    # Save comparative metrics
    comparative_path = os.path.join(save_dir, 'comparative_metrics.json')
    with open(comparative_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    return all_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models for breast ultrasound classification')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model weights (.pt file)')
    parser.add_argument('--model_name', type=str, required=True, choices=['densenet', 'resnet', 'efficientnet'],
                        help='Model architecture name')
    parser.add_argument('--variant', type=str, default=None,
                        help='Specific model variant (e.g., densenet121, resnet50, efficientnet_b0)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        model_name=args.model_name,
        variant=args.variant,
        num_classes=args.num_classes,
        device=device
    )
    
    # Get dataloaders
    dataloaders = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    test_dataloader = dataloaders['test']
    
    # Evaluate model
    metrics, preds, labels, probs = evaluate_model(model, test_dataloader, device=device)
    
    # Save results
    save_evaluation_results(
        metrics=metrics,
        preds=preds,
        labels=labels,
        probs=probs,
        save_dir=args.save_dir,
        model_name=f"{args.model_name}_{args.variant or ''}"
    ) 