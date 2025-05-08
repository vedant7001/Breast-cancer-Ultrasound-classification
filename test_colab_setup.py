"""
Test script to verify the Colab setup for the Breast Ultrasound Classification project.
Run this script after colab_setup.py to verify the environment.
"""

import os
import sys
import importlib
import json

# Test packages and dependencies
def test_dependencies():
    """Test that required Python packages are installed and working."""
    print("\nTesting Python dependencies...")
    
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
    print("\nOptional packages:")
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
        print(f"\nPyTorch GPU available: {'Yes' if gpu_available else 'No'}")
        if gpu_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except:
        print("\nCould not test PyTorch GPU availability")

# Test directory structure
def test_directories():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
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
    """Test that the configuration file exists and is valid."""
    print("\nTesting configuration...")
    
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

# Test dataset
def test_dataset():
    """Test that the dataset is available and properly structured."""
    print("\nTesting dataset...")
    
    # Use dataset_utils if available
    try:
        import dataset_utils
        dataset_utils.verify_busi_dataset()
    except ImportError:
        # Manual check if dataset_utils is not available
        data_dir = "data/BUSI"
        expected_classes = ["benign", "malignant", "normal"]
        
        if not os.path.exists(data_dir):
            print(f"✗ Dataset directory not found: {data_dir}")
            return
        
        # Check classes
        print("Looking for dataset classes...")
        classes = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
        
        print(f"Found classes: {classes}")
        missing_classes = [c for c in expected_classes if c not in classes]
        if missing_classes:
            print(f"✗ Missing expected classes: {missing_classes}")
        else:
            print(f"✓ All expected classes found")
            
            # Count images in each class
            for class_name in classes:
                class_path = os.path.join(data_dir, class_name)
                try:
                    image_files = [f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    print(f"  {class_name}: {len(image_files)} images")
                except Exception as e:
                    print(f"  Error counting {class_name} images: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("COLAB SETUP VERIFICATION")
    print("=" * 50)
    
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
    test_dataset()
    
    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 