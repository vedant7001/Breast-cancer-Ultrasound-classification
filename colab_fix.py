"""
One-cell Google Colab fix script for severe directory issues.
This script is designed to be copied directly into a Google Colab cell.
"""

# Fix for severe directory issues in Google Colab
import os
import subprocess
import shutil
import sys
import json
import time

def fix_colab_directory_issues():
    """Reset directories and fix severe directory access issues."""
    print("Starting emergency directory fix...")
    
    # Force move to /content regardless of current state
    try:
        os.chdir("/content")
        print("✓ Changed to /content directory")
    except Exception as e:
        print(f"❌ Error changing to /content: {e}")
        print("This is a severe issue with Colab runtime. Try restarting the runtime.")
        return False
    
    # Clean up any nested repositories that might exist
    repo_name = "Breast-cancer-Ultrasound-classification"
    clean_paths = [
        f"/content/{repo_name}",
        f"/content/{repo_name}/{repo_name}",
        f"/content/{repo_name}/{repo_name}/{repo_name}",
        f"/content/{repo_name}/{repo_name}/{repo_name}/{repo_name}",
        f"/content/{repo_name}/{repo_name}/{repo_name}/{repo_name}/{repo_name}"
    ]
    
    for path in clean_paths:
        if os.path.exists(path):
            try:
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
            except Exception as e:
                print(f"❌ Error removing {path}: {e}")
    
    # Verify working directory
    try:
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")
        if cwd != "/content":
            os.chdir("/content")
            print(f"Fixed working directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Error verifying working directory: {e}")
        return False
    
    # Attempt to clone the repository
    repo_url = "https://github.com/vedant7001/Breast-cancer-Ultrasound-classification.git"
    try:
        if os.path.exists(repo_name):
            print(f"Removing existing repository at {repo_name}")
            shutil.rmtree(repo_name)
        
        print(f"Cloning repository from {repo_url}")
        subprocess.run(["git", "clone", repo_url], check=True)
        
        # Verify clone was successful
        if not os.path.exists(repo_name):
            print("❌ Clone seemed to work but directory doesn't exist")
            return False
        
        print(f"✓ Repository cloned successfully to {repo_name}")
    except Exception as e:
        print(f"❌ Error cloning repository: {e}")
        return False
    
    # Change to repository directory
    try:
        os.chdir(repo_name)
        print(f"✓ Changed to repository directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Error changing to repository directory: {e}")
        return False
    
    return True

def setup_environment():
    """Install required packages for the project."""
    print("\nSetting up environment...")
    
    packages = [
        "torch", 
        "torchvision", 
        "numpy", 
        "scikit-learn", 
        "matplotlib", 
        "seaborn", 
        "tqdm", 
        "Pillow", 
        "kaggle",
        "thop"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
        except Exception as e:
            print(f"⚠️ Warning: Failed to install {package}: {e}")
    
    # Try to install grad-cam directly from source
    try:
        print("Installing pytorch-grad-cam from source...")
        subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/jacobgil/pytorch-grad-cam.git"], check=False)
    except Exception as e:
        print(f"⚠️ Warning: Failed to install pytorch-grad-cam: {e}")

def create_directories():
    """Create necessary project directories."""
    print("\nCreating project directories...")
    
    directories = [
        'data',
        'data/BUSI',
        'checkpoints',
        'results',
        'experiments'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to create directory {directory}: {e}")

def update_config():
    """Update config.json for Colab."""
    print("\nUpdating configuration...")
    
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Get the full path to the current directory
            repo_dir = os.getcwd()
            
            # Update paths for Colab
            config['data']['data_dir'] = os.path.join(repo_dir, 'data/BUSI')
            config['output_dir'] = os.path.join(repo_dir, 'experiments')
            
            # Update sample images
            if 'sample_images' in config:
                for i in range(len(config['sample_images'])):
                    config['sample_images'][i] = os.path.join(repo_dir, config['sample_images'][i])
            
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            
            print("✓ Configuration updated successfully")
        except Exception as e:
            print(f"⚠️ Warning: Failed to update config.json: {e}")

def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU is available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ GPU is not available. Training will be slower on CPU.")
            return False
    except Exception as e:
        print(f"⚠️ Warning: Failed to check GPU: {e}")
        return False

def download_dataset():
    """Handle dataset download."""
    print("\nSetting up dataset download...")
    
    # Create dataset directory
    os.makedirs('data/BUSI', exist_ok=True)
    
    # Create a helper script for dataset download
    dataset_script = """
import os
import sys
import subprocess
import zipfile

def install_kaggle():
    try:
        import kaggle
        print("Kaggle API already installed")
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        try:
            import kaggle
            print("Kaggle API installed successfully")
        except ImportError:
            print("Failed to install Kaggle API")
            return False
    return True

def setup_credentials():
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_file = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_file):
        print("Kaggle credentials already exist")
        return True
    
    try:
        from google.colab import files
        print("Please upload your kaggle.json file")
        print("To get this file, go to https://www.kaggle.com/account")
        print("Then click 'Create New API Token'")
        
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            with open(kaggle_file, "wb") as f:
                f.write(uploaded["kaggle.json"])
            os.chmod(kaggle_file, 0o600)
            print("Credentials saved successfully")
            return True
        else:
            print("No kaggle.json file was uploaded")
            return False
    except Exception as e:
        print(f"Error setting up credentials: {e}")
        return False

def download_dataset():
    target_dir = "data/BUSI"
    os.makedirs(target_dir, exist_ok=True)
    
    if not install_kaggle():
        return False
    
    if not setup_credentials():
        return False
    
    try:
        print("Downloading BUSI dataset...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "aryashah2k/breast-ultrasound-images-dataset"], check=True)
        
        if not os.path.exists("breast-ultrasound-images-dataset.zip"):
            print("Download failed - zip file not found")
            return False
        
        print(f"Extracting to {target_dir}...")
        with zipfile.ZipFile("breast-ultrasound-images-dataset.zip", "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        os.remove("breast-ultrasound-images-dataset.zip")
        print("Dataset downloaded and extracted successfully")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
"""
    
    try:
        with open("download_dataset.py", "w") as f:
            f.write(dataset_script)
        print("✓ Created dataset download script")
    except Exception as e:
        print(f"⚠️ Warning: Failed to create dataset script: {e}")

def main():
    """Main function to fix directory issues and set up the project."""
    print("=" * 60)
    print("EMERGENCY COLAB DIRECTORY FIX AND PROJECT SETUP")
    print("=" * 60)
    
    # Fix directory issues
    if not fix_colab_directory_issues():
        print("\n❌ Failed to fix directory issues. Please restart the runtime and try again.")
        return
    
    # Setup environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Update configuration
    update_config()
    
    # Check GPU
    check_gpu()
    
    # Create dataset download script
    download_dataset()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("To download the dataset, run:")
    print("!python download_dataset.py")
    print("\nTo train models, run:")
    print("!python main.py --config config.json")
    print("\nIf you encounter any further directory issues, restart the runtime and run this script again.")
    print("=" * 60)

if __name__ == "__main__":
    main() 