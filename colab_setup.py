"""
Google Colab setup script for Breast Ultrasound Classification project.
Run this script at the beginning of your Colab notebook.
"""

import os
import json
import sys
import subprocess
import shutil

def clean_repository_setup():
    """Clean up any existing repository and set up properly."""
    repo_url = "https://github.com/vedant7001/Breast-cancer-Ultrasound-classification.git"
    repo_dir = "Breast-cancer-Ultrasound-classification"
    
    # Check if we're already in a repository directory
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    
    # Count how many times the repo name appears in the path
    repo_count = path_parts.count(repo_dir)
    
    if repo_count > 0:
        print(f"Detected nested repository directories ({repo_count} levels deep)")
        
        # Go back to /content directory
        if "/content" in current_path:
            print("Moving back to /content directory...")
            os.chdir("/content")
            
            # Remove existing repository folders
            if os.path.exists(repo_dir):
                print(f"Removing existing repository at {repo_dir}...")
                try:
                    shutil.rmtree(repo_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove directory: {e}")
    
    # Clone the repository fresh
    if not os.path.exists(repo_dir):
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(["git", "clone", repo_url], check=True)
    
    # Change to repository directory
    os.chdir(repo_dir)
    print(f"Working directory: {os.getcwd()}")

def install_packages():
    """Install required packages."""
    print("\nInstalling required packages...")
    
    # First, upgrade pip
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    except Exception as e:
        print(f"Warning: Failed to upgrade pip: {e}")
    
    # Install packages individually
    packages = [
        "torch", 
        "torchvision", 
        "numpy", 
        "scikit-learn", 
        "matplotlib", 
        "seaborn", 
        "tqdm", 
        "Pillow", 
        "thop"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
        except Exception as e:
            print(f"Warning: Failed to install {package}: {e}")
    
    # Try multiple versions of grad-cam
    grad_cam_options = ["pytorch-grad-cam", "grad-cam", "pytorch_grad_cam"]
    grad_cam_installed = False
    
    for option in grad_cam_options:
        try:
            print(f"Trying to install {option}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", option], check=False)
            if result.returncode == 0:
                print(f"Successfully installed {option}")
                grad_cam_installed = True
                break
        except Exception as e:
            print(f"Warning: Failed to install {option}: {e}")
    
    if not grad_cam_installed:
        print("WARNING: Could not install any version of grad-cam.")
        print("Manual installation of pytorch-grad-cam from source may be required.")
        print("You can try running: !pip install git+https://github.com/jacobgil/pytorch-grad-cam.git")

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
        config['data']['data_dir'] = '/content/Breast-cancer-Ultrasound-classification/data/BUSI'
        config['output_dir'] = '/content/Breast-cancer-Ultrasound-classification/experiments'
        
        # Update sample images paths if they exist
        if 'sample_images' in config:
            for i in range(len(config['sample_images'])):
                config['sample_images'][i] = os.path.join('/content/Breast-cancer-Ultrasound-classification', 
                                                        config['sample_images'][i])
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration updated for Colab paths")
    
    # Instructions for downloading BUSI dataset
    print("\nTo download the BUSI dataset:")
    print("1. Go to https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset")
    print("2. Download the dataset")
    print("3. Upload it to your Colab session using the file browser")
    print("4. Extract it to the data/BUSI directory")
    
    print("\nSetup complete! You can now run the breast ultrasound classification project in Colab.")
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"\nGPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nGPU is not available. Training will be slower.")

def main():
    """Main function to set up the Colab environment."""
    # Clean repository setup
    clean_repository_setup()
    
    # Set up the Colab environment
    setup_colab()

if __name__ == "__main__":
    main() 