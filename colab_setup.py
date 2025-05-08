"""
Google Colab setup script for Breast Ultrasound Classification project.
Run this script at the beginning of your Colab notebook.
"""

import os
import json
import urllib.request
import zipfile
import sys
import subprocess

def clone_repository():
    """Clone the GitHub repository if not already cloned."""
    repo_url = "https://github.com/vedant7001/Breast-cancer-Ultrasound-classification.git"
    repo_dir = "Breast-cancer-Ultrasound-classification"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repository from {repo_url}...")
        # Using subprocess instead of ! magic
        subprocess.run(["git", "clone", repo_url], check=True)
        # Change directory
        os.chdir(repo_dir)
    else:
        print(f"Repository already exists at {repo_dir}")
        # Change directory
        os.chdir(repo_dir)
        # Pull the latest changes
        subprocess.run(["git", "pull"], check=True)
    
    print(f"Working directory: {os.getcwd()}")

def install_packages():
    """Install required packages."""
    print("\nInstalling required packages...")
    packages = [
        "torch", "torchvision", "numpy", "scikit-learn", 
        "matplotlib", "seaborn", "tqdm", "Pillow", 
        "pytorch-grad-cam", "thop"
    ]
    
    # Use subprocess to run pip install
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + packages
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

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
    install_packages()
    
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
    print("\nTo download the BUSI dataset:")
    print("1. Go to https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print("2. Download the dataset")
    print("3. Upload it to your Colab session using the file browser")
    print("4. Extract it to the data/BUSI directory")
    
    # Instead of using input() which can block in Colab
    print("\nSetup complete! You can now run the breast ultrasound classification project in Colab.")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"\nGPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nGPU is not available. Training will be slower.")

def main():
    """Main function to set up the Colab environment."""
    # Clone the repository
    clone_repository()
    
    # Set up the Colab environment
    setup_colab()

if __name__ == "__main__":
    main() 