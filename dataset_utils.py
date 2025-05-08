"""
Utility functions for dataset handling in Google Colab.
This script provides functions to download and prepare the BUSI dataset.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import zipfile

def install_kaggle_api():
    """Install Kaggle API if not already installed."""
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

def setup_kaggle_credentials(kaggle_json_path=None):
    """
    Set up Kaggle credentials for API access.
    
    Args:
        kaggle_json_path (str, optional): Path to kaggle.json file. 
                                         If None, prompts for upload.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create Kaggle directory
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Check if credentials already exist
    kaggle_cred = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_cred):
        print("Kaggle credentials already exist.")
        return True
    
    # Process provided kaggle.json file
    if kaggle_json_path and os.path.exists(kaggle_json_path):
        shutil.copy(kaggle_json_path, kaggle_cred)
        os.chmod(kaggle_cred, 0o600)  # Set permissions
        print(f"Copied kaggle.json from {kaggle_json_path}")
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
        print(f"Please manually place kaggle.json in {kaggle_dir}")
        return False

def download_busi_dataset(target_dir="data/BUSI"):
    """
    Download the BUSI dataset from Kaggle.
    
    Args:
        target_dir (str): Directory to extract dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
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
        
        # Verify dataset structure
        verify_busi_dataset(target_dir)
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def extract_zip_to_busi(zip_file_path, target_dir="data/BUSI"):
    """
    Extract a zip file to the BUSI dataset directory.
    
    Args:
        zip_file_path (str): Path to the zip file
        target_dir (str): Directory to extract dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(zip_file_path):
        print(f"Zip file not found: {zip_file_path}")
        return False
    
    # Create the target directory
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        print(f"Extracting {zip_file_path} to {target_dir}...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        print("Dataset extracted successfully.")
        
        # Verify dataset structure
        verify_busi_dataset(target_dir)
        return True
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False

def verify_busi_dataset(data_dir="data/BUSI"):
    """
    Verify the BUSI dataset structure and count images.
    
    Args:
        data_dir (str): Directory containing the dataset
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Expected classes
    expected_classes = ["benign", "malignant", "normal"]
    
    # Check directory existence
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        return False
    
    # Check classes
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    missing_classes = [c for c in expected_classes if c not in classes]
    if missing_classes:
        print(f"Missing classes: {missing_classes}")
        print("Dataset structure may be incorrect.")
        return False
    
    # Count images in each class
    class_counts = {}
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)
        total_images += len(image_files)
    
    # Print dataset statistics
    print(f"\nBUSI Dataset Statistics:")
    print(f"Total images: {total_images}")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images ({count/total_images*100:.1f}%)")
    
    if total_images < 10:
        print("WARNING: Very few images found. Dataset may not be extracted properly.")
        return False
    
    return True

if __name__ == "__main__":
    # If run directly, download and verify the dataset
    data_dir = "data/BUSI"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        
    if os.path.exists(data_dir) and verify_busi_dataset(data_dir):
        print(f"BUSI dataset already exists at {data_dir}")
    else:
        download_busi_dataset(data_dir) 