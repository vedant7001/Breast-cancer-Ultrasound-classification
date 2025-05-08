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
    
    # First ensure we're in a valid working directory
    # Colab sometimes has issues with working directory
    try:
        os.getcwd()
    except:
        # If current directory is inaccessible, move to /content which should always exist in Colab
        print("Current directory is inaccessible, moving to /content...")
        os.chdir("/content")
    
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
    
    # Ensure we're in a valid working directory
    try:
        cwd = os.getcwd()
        print(f"Current working directory: {cwd}")
    except:
        print("Still having issues with working directory, moving to /content...")
        os.chdir("/content")
    
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
    
    # Get the absolute path to the repository directory
    repo_dir = os.getcwd()
    
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Update paths for Colab using absolute paths
        config['data']['data_dir'] = os.path.join(repo_dir, 'data/BUSI')
        config['output_dir'] = os.path.join(repo_dir, 'experiments')
        
        # Update sample images paths
        if 'sample_images' in config:
            for i in range(len(config['sample_images'])):
                config['sample_images'][i] = os.path.join(repo_dir, config['sample_images'][i])
        
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration updated for Colab")
        print(f"Data directory set to: {config['data']['data_dir']}")
        print(f"Output directory set to: {config['output_dir']}")

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

def create_test_setup_script():
    """Create the test_colab_setup.py script in the current directory."""
    script_content = """
# Test script to verify the Colab setup for the Breast Ultrasound Classification project
import os
import sys
import importlib
import json

# Test packages and dependencies
def test_dependencies():
    \"\"\"Test that required Python packages are installed and working.\"\"\"
    print("\\nTesting Python dependencies...")
    
    # List of required packages
    required_packages = [
        "torch", "torchvision", "numpy", "sklearn", 
        "matplotlib", "seaborn", "tqdm", "PIL", "thop"
    ]
    
    # Optional packages
    optional_packages = ["pytorch_grad_cam", "kaggle"]
    
    # Test required packages
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
    
    # Test optional packages
    print("\\nOptional packages:")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"⚠ {package} is NOT installed (but may not be required)")
    
    # Test PyTorch with GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"\\nPyTorch GPU available: {'Yes' if gpu_available else 'No'}")
        if gpu_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("\\nCould not test PyTorch GPU availability")

# Test directory structure
def test_directories():
    \"\"\"Test that required directories exist.\"\"\"
    print("\\nTesting directory structure...")
    
    required_dirs = [
        "data",
        "data/BUSI",
        "checkpoints",
        "results",
        "experiments"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does NOT exist")

# Test configuration
def test_config():
    \"\"\"Test that the configuration file exists and is valid.\"\"\"
    print("\\nTesting configuration...")
    
    config_file = "config.json"
    
    if not os.path.exists(config_file):
        print(f"✗ {config_file} does not exist")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✓ {config_file} is valid JSON")
        
        # Check for required config keys
        required_keys = ["data", "output_dir", "models"]
        for key in required_keys:
            if key in config:
                print(f"✓ Config contains '{key}'")
            else:
                print(f"✗ Config missing '{key}'")
        
        # Check data directory in config
        if "data" in config and "data_dir" in config["data"]:
            data_dir = config["data"]["data_dir"]
            print(f"Config data_dir: {data_dir}")
            if os.path.exists(data_dir):
                print(f"✓ Data directory exists")
            else:
                print(f"✗ Data directory does NOT exist")
    except Exception as e:
        print(f"✗ Error reading config: {e}")

def main():
    \"\"\"Run all tests.\"\"\"
    print("=" * 50)
    print("COLAB SETUP VERIFICATION")
    print("=" * 50)
    
    # Get current directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if running in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("⚠ Not running in Google Colab")
    
    # Run tests
    test_dependencies()
    test_directories()
    test_config()
    
    print("\\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
"""
    try:
        with open('test_colab_setup.py', 'w') as f:
            f.write(script_content)
        print("\n✓ Created test_colab_setup.py script in the current directory")
    except Exception as e:
        print(f"✗ Error creating test_colab_setup.py: {e}")

def create_dataset_utils():
    """Create the dataset_utils.py script in the current directory."""
    script_content = """
# Utility functions for dataset handling in Google Colab
import os
import sys
import subprocess
import zipfile

def install_kaggle_api():
    \"\"\"Install Kaggle API if not already installed.\"\"\"
    try:
        import kaggle
        print("Kaggle API is already installed.")
    except ImportError:
        print("Installing Kaggle API...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)
        try:
            import kaggle
            print("Kaggle API installed successfully.")
        except ImportError:
            print("Failed to install Kaggle API. Please try manual installation.")
            return False
    return True

def setup_kaggle_credentials():
    \"\"\"Set up Kaggle credentials for API access.\"\"\"
    # Create Kaggle directory
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Check if credentials already exist
    kaggle_cred = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_cred):
        print("Kaggle credentials already exist.")
        return True
    
    # If in Colab, prompt for upload
    try:
        from google.colab import files
        print("Please upload your kaggle.json file.")
        print("If you don't have one, go to https://www.kaggle.com/account")
        print("Then click 'Create New API Token' to download kaggle.json")
        
        uploaded = files.upload()
        if "kaggle.json" in uploaded:
            # Save the uploaded credentials
            with open(kaggle_cred, "wb") as f:
                f.write(uploaded["kaggle.json"])
            os.chmod(kaggle_cred, 0o600)
            print("Kaggle credentials configured successfully.")
            return True
        else:
            print("No kaggle.json file was uploaded.")
            return False
    except ImportError:
        print("Not running in Google Colab or files.upload() not available.")
        return False

def download_busi_dataset(target_dir="data/BUSI"):
    \"\"\"Download the BUSI dataset from Kaggle.\"\"\"
    # Create the target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Check if kaggle API is installed
    if not install_kaggle_api():
        return False
    
    # Set up kaggle credentials if needed
    if not setup_kaggle_credentials():
        return False
    
    # Download dataset
    try:
        print("Downloading BUSI dataset from Kaggle...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "aryashah2k/breast-ultrasound-images-dataset"], 
                      check=True)
        
        # Extract dataset
        print(f"Extracting dataset to {target_dir}...")
        with zipfile.ZipFile("breast-ultrasound-images-dataset.zip", "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up
        os.remove("breast-ultrasound-images-dataset.zip")
        print("Dataset downloaded and extracted successfully.")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    data_dir = "data/BUSI"
    os.makedirs(data_dir, exist_ok=True)
    download_busi_dataset(data_dir)
"""
    try:
        with open('dataset_utils.py', 'w') as f:
            f.write(script_content)
        print("✓ Created dataset_utils.py script in the current directory")
    except Exception as e:
        print(f"✗ Error creating dataset_utils.py: {e}")

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
    
    # Create helper scripts directly
    create_test_setup_script()
    create_dataset_utils()
    
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