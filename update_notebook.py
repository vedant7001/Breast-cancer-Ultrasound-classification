"""
Script to update the breast_ultrasound_classification.ipynb notebook.
Adds references to test_colab_setup.py and dataset_utils.py.
"""

import json
import os

def update_notebook(notebook_path):
    """Update the Jupyter notebook with improved setup code."""
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find the setup cell
        setup_cell_index = None
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'setup_code' in cell['metadata'].get('id', ''):
                setup_cell_index = i
                break
        
        if setup_cell_index is not None:
            # Update the setup cell with improved code
            notebook['cells'][setup_cell_index]['source'] = [
                "# Clone the repository\n",
                "!git clone https://github.com/vedant7001/Breast-cancer-Ultrasound-classification.git\n",
                "%cd Breast-cancer-Ultrasound-classification\n",
                "\n",
                "# Run the setup script\n",
                "!python colab_setup.py\n",
                "\n",
                "# Verify setup\n",
                "!python test_colab_setup.py"
            ]
            
            # Find the dataset download cell
            dataset_cell_index = None
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] == 'code' and 'dataset_download' in cell['metadata'].get('id', ''):
                    dataset_cell_index = i
                    break
            
            if dataset_cell_index is not None:
                # Update the dataset download cell
                notebook['cells'][dataset_cell_index]['source'] = [
                    "# Use our dataset utility script to download and set up the dataset\n",
                    "!python dataset_utils.py\n",
                    "\n",
                    "# If that fails, try the manual method\n",
                    "# Uncomment the following lines if dataset_utils.py fails\n",
                    "\n",
                    "# Install kaggle API if needed\n",
                    "# !pip install -q kaggle\n",
                    "# \n",
                    "# # Upload your kaggle.json file\n",
                    "# from google.colab import files\n",
                    "# print(\"Upload your kaggle.json file (from your Kaggle account settings)\")\n",
                    "# uploaded = files.upload()\n",
                    "# \n",
                    "# # Configure kaggle credentials\n",
                    "# !mkdir -p ~/.kaggle\n",
                    "# !cp kaggle.json ~/.kaggle/\n",
                    "# !chmod 600 ~/.kaggle/kaggle.json\n",
                    "# \n",
                    "# # Create data directory if it doesn't exist\n",
                    "# !mkdir -p data/BUSI\n",
                    "# \n",
                    "# # Download BUSI dataset\n",
                    "# !kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset\n",
                    "# !unzip -q breast-ultrasound-images-dataset.zip -d data/BUSI\n",
                    "# !rm breast-ultrasound-images-dataset.zip\n",
                    "# \n",
                    "# # Check dataset structure\n",
                    "# !ls -la data/BUSI"
                ]
            
            # Save the updated notebook with a backup of the original
            backup_path = notebook_path + '.backup'
            if not os.path.exists(backup_path):
                os.rename(notebook_path, backup_path)
                print(f"Created backup at {backup_path}")
            
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2)
            
            print(f"Successfully updated {notebook_path}")
            return True
        else:
            print("Could not find setup code cell in the notebook")
            return False
    
    except Exception as e:
        print(f"Error updating notebook: {e}")
        return False

if __name__ == "__main__":
    notebook_path = "breast_ultrasound_classification.ipynb"
    update_notebook(notebook_path) 