import os
import argparse
import json
import torch
import numpy as np
from datetime import datetime

from dataloader import get_dataloaders, load_standard_dataset
from models import get_model
from train import train_model
from evaluate import evaluate_model, evaluate_multiple_models
from visualize import create_full_report, plot_metrics_comparison
from utils import calculate_params_flops, get_target_layer, visualize_multiple_samples


def run_experiment(config_file):
    """
    Run a breast ultrasound classification experiment from a config file
    
    Args:
        config_file (str): Path to the configuration JSON file
    """
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = config.get('experiment_name', f"experiment_{timestamp}")
    experiment_dir = os.path.join(config.get('output_dir', 'experiments'), experiment_name)
    
    # Create subdirectories
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    results_dir = os.path.join(experiment_dir, 'results')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save config to experiment directory
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    print(f"Using device: {device}")
    
    # Get dataset
    data_config = config.get('data', {})
    if data_config.get('use_standard_dataset', False):
        dataloaders = load_standard_dataset(
            dataset_name=data_config.get('dataset_name', 'BUSI'),
            batch_size=data_config.get('batch_size', 32),
            img_size=data_config.get('img_size', 224),
            num_workers=data_config.get('num_workers', 4)
        )
    else:
        dataloaders = get_dataloaders(
            data_dir=data_config.get('data_dir'),
            batch_size=data_config.get('batch_size', 32),
            img_size=data_config.get('img_size', 224),
            num_workers=data_config.get('num_workers', 4)
        )
    
    # Extract dataset info
    num_classes = len(dataloaders['train'].dataset.classes)
    class_names = dataloaders['train'].dataset.classes
    print(f"Dataset loaded: {num_classes} classes: {class_names}")
    print(f"Train: {len(dataloaders['train'].dataset)} samples")
    print(f"Validation: {len(dataloaders['val'].dataset)} samples")
    print(f"Test: {len(dataloaders['test'].dataset)} samples")
    
    # Train models
    model_results = {}
    model_configs = config.get('models', [])
    for model_config in model_configs:
        model_name = model_config.get('name')
        variant = model_config.get('variant')
        full_name = f"{model_name}_{variant}" if variant else model_name
        
        print(f"\n{'=' * 50}")
        print(f"Training {full_name}")
        print(f"{'=' * 50}")
        
        # Train model
        model, trainer, metrics = train_model(
            model_name=model_name,
            data_dir=data_config.get('data_dir'),
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True),
            variant=variant,
            batch_size=data_config.get('batch_size', 32),
            num_epochs=model_config.get('epochs', 50),
            learning_rate=model_config.get('learning_rate', 1e-4),
            use_scheduler=model_config.get('use_scheduler', True),
            device=device
        )
        
        # Calculate and save model parameters
        num_params, flops = calculate_params_flops(model, device=device)
        model_info = {
            'name': model_name,
            'variant': variant,
            'parameters': num_params,
            'flops': flops if flops is not None else "Not available",
            'metrics': metrics
        }
        
        with open(os.path.join(results_dir, f"{full_name}_info.json"), 'w') as f:
            json.dump(model_info, f, indent=4)
        
        model_results[full_name] = model_info
    
    # Create comparative results
    with open(os.path.join(results_dir, 'comparative_results.json'), 'w') as f:
        json.dump(model_results, f, indent=4)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_full_report(
        experiment_dir=experiment_dir,
        models=[m for m in model_results.keys()],
        save_dir=os.path.join(results_dir, 'visualizations')
    )
    
    # Generate Grad-CAM visualizations if sample images are provided
    if 'sample_images' in config:
        for model_name, model_info in model_results.items():
            model_path = os.path.join(checkpoints_dir, f"{model_name}_best.pt")
            model = get_model(
                model_info['name'], 
                num_classes=num_classes, 
                pretrained=False, 
                variant=model_info['variant']
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            # Get target layer for Grad-CAM
            target_layer = get_target_layer(model, model_info['name'])
            
            # Visualize sample images
            visualize_multiple_samples(
                model=model,
                img_paths=config['sample_images'],
                target_layer=target_layer,
                class_names=class_names,
                device=device,
                save_dir=os.path.join(results_dir, 'visualizations', f"{model_name}_gradcam")
            )
    
    print(f"\nExperiment completed. Results saved to {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(description='Breast Ultrasound Classification')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration JSON file')
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == '__main__':
    main() 