import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import torchvision.transforms as transforms


def plot_training_curves(history_file, save_dir='results'):
    """
    Plot training and validation loss/accuracy curves
    
    Args:
        history_file (str): Path to the training history JSON file
        save_dir (str): Directory to save the plots
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load history data
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Plot accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the figure
    model_name = os.path.basename(history_file).replace('_history.json', '')
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(predictions_files, save_dir='results'):
    """
    Plot ROC curves for multiple models
    
    Args:
        predictions_files (dict): Dictionary mapping model names to prediction NPZ files
        save_dir (str): Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    for model_name, pred_file in predictions_files.items():
        # Load predictions
        data = np.load(pred_file)
        labels = data['labels']
        probabilities = data['probabilities']
        
        # For binary classification
        if probabilities.shape[1] == 2:
            # Compute ROC curve and ROC area
            fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(predictions_file, save_dir='results'):
    """
    Plot confusion matrix for a model
    
    Args:
        predictions_file (str): Path to the predictions NPZ file
        save_dir (str): Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load predictions
    data = np.load(predictions_file)
    labels = data['labels']
    predictions = data['predictions']
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Get number of classes
    num_classes = len(np.unique(labels))
    
    # Set up plot
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    
    # Set plot details
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    model_name = os.path.basename(predictions_file).replace('_predictions.npz', '')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_files, metrics_names=None, save_dir='results'):
    """
    Plot bar charts comparing metrics across models
    
    Args:
        metrics_files (dict): Dictionary mapping model names to metrics JSON files
        metrics_names (list, optional): List of metric names to compare
        save_dir (str): Directory to save the plot
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if metrics_names is None:
        metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Load metrics for each model
    models_metrics = {}
    for model_name, metrics_file in metrics_files.items():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        models_metrics[model_name] = {k: metrics[k] for k in metrics_names if k in metrics}
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.15
    index = np.arange(len(metrics_names))
    
    # Plot bars for each model
    for i, (model_name, metrics) in enumerate(models_metrics.items()):
        values = [metrics.get(metric, 0) for metric in metrics_names]
        ax.bar(index + i * bar_width, values, bar_width, label=model_name)
    
    # Set plot details
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(index + bar_width * (len(models_metrics) - 1) / 2)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Save the figure
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_grad_cam(model, image_path, target_layer_name, class_idx=None, save_dir='results'):
    """
    Visualize model activation using Grad-CAM
    
    Args:
        model (torch.nn.Module): Model to visualize
        image_path (str): Path to the input image
        target_layer_name (str): Name of the target layer for Grad-CAM
        class_idx (int, optional): Target class index (default: None, uses predicted class)
        save_dir (str): Directory to save the visualization
        
    Returns:
        tuple: (original_image, grad_cam_image)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("pytorch-grad-cam package is required for Grad-CAM visualization.")
        print("Install it with: pip install pytorch-grad-cam")
        return None, None
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = next(model.parameters()).device
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Save original image for overlay
    rgb_img = np.array(Image.open(image_path).convert('RGB').resize((224, 224))) / 255.0
    
    # Get target layer
    if hasattr(model, 'features'):
        if target_layer_name == 'last':
            target_layers = [list(model.features.children())[-1]]
        else:
            target_layers = [model.features.__getattr__(target_layer_name)]
    elif hasattr(model, 'conv_layers'):
        if target_layer_name == 'last':
            target_layers = [list(model.conv_layers.children())[-1]]
        else:
            # This is model-specific and may need adjustment
            layers = list(model.conv_layers.children())
            target_layers = [layers[int(target_layer_name)]]
    else:
        raise ValueError("Cannot find target layer in model. Adjust implementation for this model architecture.")
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device.type=='cuda')
    
    # Generate class activation map
    grayscale_cam = cam(input_tensor=input_tensor, target_category=class_idx)
    grayscale_cam = grayscale_cam[0, :]
    
    # Create visualization
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save visualization
    img_name = os.path.basename(image_path)
    save_path = os.path.join(save_dir, f'gradcam_{img_name}')
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title('Grad-CAM Visualization')
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rgb_img, visualization


def create_full_report(experiment_dir, models, save_dir='results'):
    """
    Create a comprehensive visualization report for all models
    
    Args:
        experiment_dir (str): Directory containing experiment outputs
        models (list): List of model names
        save_dir (str): Directory to save the report figures
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot training curves for each model
    for model in models:
        history_file = os.path.join(experiment_dir, 'checkpoints', f'{model}_history.json')
        if os.path.exists(history_file):
            plot_training_curves(history_file, save_dir)
    
    # 2. Plot ROC curves comparison
    predictions_files = {}
    for model in models:
        pred_file = os.path.join(experiment_dir, 'results', f'{model}_predictions.npz')
        if os.path.exists(pred_file):
            predictions_files[model] = pred_file
    
    if predictions_files:
        plot_roc_curve(predictions_files, save_dir)
    
    # 3. Plot confusion matrices
    for model in models:
        pred_file = os.path.join(experiment_dir, 'results', f'{model}_predictions.npz')
        if os.path.exists(pred_file):
            plot_confusion_matrix(pred_file, save_dir)
    
    # 4. Plot metrics comparison
    metrics_files = {}
    for model in models:
        metrics_file = os.path.join(experiment_dir, 'results', f'{model}_metrics.json')
        if os.path.exists(metrics_file):
            metrics_files[model] = metrics_file
    
    if metrics_files:
        plot_metrics_comparison(metrics_files, save_dir=save_dir)
    
    print(f"Visualization report completed. Results saved to {save_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize results for breast ultrasound classification')
    parser.add_argument('--experiment_dir', type=str, required=True,
                        help='Directory containing experiment outputs')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model names to include in visualizations')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    create_full_report(
        experiment_dir=args.experiment_dir,
        models=args.models,
        save_dir=args.save_dir
    ) 