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
    
    # If standard methods fail, try installing from GitHub source
    if not grad_cam_installed:
        try:
            print("Installing pytorch-grad-cam from GitHub source...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "git+https://github.com/jacobgil/pytorch-grad-cam.git"],
                check=False
            )
            if result.returncode == 0:
                print("Successfully installed pytorch-grad-cam from source")
                grad_cam_installed = True
        except Exception as e:
            print(f"Warning: Failed to install pytorch-grad-cam from source: {e}")
    
    if not grad_cam_installed:
        print("WARNING: Could not install any version of grad-cam.")
        print("You may need to manually install it later if needed for visualization.")

def create_project_directories():
    """Create necessary project directories."""
    print("\nCreating project directories...")
    
    # Define directories to create
    directories = [
        'data',
        'data/BUSI',
        'checkpoints',
        'results',
        'experiments'
    ]
    
    # Create each directory and verify creation
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            if os.path.exists(directory):
                print(f"✓ Successfully created directory: {directory}")
            else:
                print(f"✗ Failed to create directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
    
    # Additional confirmation for the most important directory
    if os.path.exists('data/BUSI'):
        print("\n✓ CONFIRMED: data/BUSI directory is ready for dataset extraction")
    else:
        print("\n✗ ERROR: Failed to create data/BUSI directory. Please create it manually before extraction.")
        # Try an alternative method to create the directory
        try:
            subprocess.run(["mkdir", "-p", "data/BUSI"], check=False)
            print("Attempted alternative directory creation method.")
        except Exception:
            pass

def update_config_for_colab():
    """Update config.json paths for Google Colab environment."""
    print("\nUpdating configuration for Colab...")
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Update paths for Colab
        config['data']['data_dir'] = '/content/Breast-cancer-Ultrasound-classification/data/BUSI'
        config['output_dir'] = '/content/Breast-cancer-Ultrasound-classification/experiments'
        
        # Update sample images paths
        if 'sample_images' in config:
            for i in range(len(config['sample_images'])):
                config['sample_images'][i] = os.path.join('/content/Breast-cancer-Ultrasound-classification', 
                                                        config['sample_images'][i])
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration updated for Colab")

def check_gpu_availability():
    """Check and print GPU availability information."""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU is available: {device_name}")
            # Try to get more GPU info using nvidia-smi
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    print("\nGPU Details:")
                    print(result.stdout)
            except:
                pass  # If nvidia-smi fails, just continue
        else:
            print("✗ GPU is not available. Training will be slower on CPU.")
    except ImportError:
        print("Could not import torch. GPU check skipped.")

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
    
    # Clean up and clone repository
    clean_repository_setup()
    
    # Install required packages
    install_packages()
    
    # Create project directories
    create_project_directories()
    
    # Update config.json for Colab paths
    update_config_for_colab()
    
    # Check GPU availability
    check_gpu_availability()
    
    # Print dataset instructions
    print("\n" + "="*50)
    print("DATASET INSTRUCTIONS")
    print("="*50)
    print("To download the BUSI dataset:")
    print("1. Run the dataset_utils.py script:")
    print("   !python dataset_utils.py")
    print("2. OR use the notebook cell dedicated to dataset download")
    print("3. The script will guide you through downloading from Kaggle")
    print("\nSetup complete! You can now run the breast ultrasound classification project in Colab.")

if __name__ == "__main__":
    setup_colab() 