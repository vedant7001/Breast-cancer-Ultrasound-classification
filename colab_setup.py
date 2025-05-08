"""
Google Colab setup script for Breast Ultrasound Classification project.
Run this script at the beginning of your Colab notebook.
"""

import os
import json
import urllib.request
import zipfile
import sys

def setup_colab():
    """Setup the Colab environment for the project."""
    # Check if running in Colab
    try:
        import google.colab
        is_colab = True
        print("Running in Google Colab")
    except ImportError:
        is_colab = False
        print("Not running in Google Colab")
        return
    
    # Install required packages
    print("\nInstalling required packages...")
    !pip install -q torch torchvision numpy scikit-learn matplotlib seaborn tqdm Pillow pytorch-grad-cam thop
    
    # Create project directories
    print("\nCreating project directories...")
    os.makedirs('data/BUSI', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('experiments', exist_ok=True)
    
    # Update config.json for Colab paths
    print("\nUpdating configuration for Colab...")
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Update paths for Colab
        config['data']['data_dir'] = '/content/data/BUSI'
        config['output_dir'] = '/content/experiments'
        
        # Update sample images paths if they exist
        if 'sample_images' in config:
            for i in range(len(config['sample_images'])):
                config['sample_images'][i] = os.path.join('/content', config['sample_images'][i])
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration updated for Colab paths")
    
    # Add option to download the BUSI dataset
    download_dataset = input("\nDownload BUSI dataset? (y/n): ")
    if download_dataset.lower() == 'y':
        print("\nDownloading BUSI dataset...")
        # Note: You would need to provide the actual download link for the dataset
        # This is just a placeholder - replace with actual dataset link
        dataset_url = "https://kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/download"
        
        try:
            # For Kaggle datasets, you would typically need to log in and use the Kaggle API
            # Here's a simplified approach:
            print("Please use this alternative method to download the BUSI dataset:")
            print("1. Go to https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
            print("2. Download the dataset")
            print("3. Upload it to your Colab session using the file browser")
            print("4. Extract it to the data/BUSI directory")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    
    print("\nSetup complete! You can now run the breast ultrasound classification project in Colab.")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"\nGPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nGPU is not available. Training will be slower.")

if __name__ == "__main__":
    setup_colab() 