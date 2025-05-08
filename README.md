# Breast Ultrasound Image Classification

This project implements deep learning models for breast ultrasound image classification using PyTorch. The implementation includes three model architectures (DenseNet, ResNet, and EfficientNet), with comprehensive training, evaluation, and visualization functionality.

## Features

- **Data Processing**: 
  - Loading and preprocessing of breast ultrasound images
  - Data augmentation for training
  - Train/validation/test splitting (70/15/15)
  - Support for both custom and standard datasets (BUSI, mini-MIAS)

- **Model Architectures**:
  - DenseNet121/169
  - ResNet18/50/101
  - EfficientNetB0/B3
  - Pretrained weights (ImageNet) or training from scratch

- **Training**:
  - Binary or multi-class classification
  - CrossEntropyLoss
  - Adam optimizer
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing

- **Evaluation**:
  - Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, AUC
  - Confusion matrices
  - Classification reports

- **Visualization**:
  - Training/validation curves
  - ROC curves for model comparison
  - Confusion matrix heatmaps
  - Bar charts for metric comparison
  - Grad-CAM visualization for model interpretability

## Project Structure

```
├── config.json               # Example configuration file
├── dataloader.py             # Data loading and preprocessing
├── models.py                 # Model architectures
├── train.py                  # Training pipeline
├── evaluate.py               # Evaluation functions
├── visualize.py              # Visualization utilities
├── main.py                   # Main script to run experiments
├── utils/                    # Utility functions
│   ├── __init__.py           # Package initialization
│   ├── metrics.py            # Metrics calculation
│   └── grad_cam.py           # Grad-CAM implementation
├── colab_setup.py            # Google Colab setup script
├── breast_ultrasound_classification.ipynb  # Colab notebook
├── data/                     # Data directory
│   ├── BUSI/                 # BUSI dataset (if used)
│   └── mini-MIAS/            # mini-MIAS dataset (if used)
├── checkpoints/              # Model checkpoints
└── results/                  # Evaluation results and visualizations
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- NumPy
- scikit-learn
- matplotlib
- seaborn
- tqdm
- Pillow
- pytorch-grad-cam (for Grad-CAM visualization)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd breast-ultrasound-classification
   ```

2. Install dependencies:
   ```
   pip install torch torchvision numpy scikit-learn matplotlib seaborn tqdm Pillow pytorch-grad-cam
   ```

3. Prepare your dataset:
   - For the BUSI dataset, organize it as follows:
     ```
     data/BUSI/
     ├── benign/
     ├── malignant/
     └── normal/
     ```
   - For custom datasets, ensure a similar structure with class folders containing images.

## Usage

### Basic Usage

Run an experiment using a configuration file:

```bash
python main.py --config config.json
```

### Training a Single Model

```bash
python train.py --model densenet --variant densenet121 --data_dir data/BUSI --num_classes 3 --batch_size 32 --epochs 50 --lr 1e-4
```

### Evaluating a Model

```bash
python evaluate.py --model_path checkpoints/densenet_densenet121_best.pt --model_name densenet --variant densenet121 --data_dir data/BUSI --num_classes 3
```

### Visualizing Results

```bash
python visualize.py --experiment_dir experiments/breast_ultrasound_classification --models densenet_densenet121 resnet_resnet50 efficientnet_efficientnet_b0
```

## Google Colab Usage

This project supports running on Google Colab for easy access to GPU resources. You can use the provided notebook or follow these steps:

1. Open the `breast_ultrasound_classification.ipynb` notebook in Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload the notebook from this repository or open it directly from GitHub

2. Alternatively, create a new notebook and run:
   ```python
   # Clone repository
   !git clone https://github.com/vedant7001/Breast-cancer-Ultrasound-classification.git
   %cd Breast-cancer-Ultrasound-classification

   # Execute setup script
   !python colab_setup.py
   ```

3. The `colab_setup.py` script will:
   - Install all required dependencies
   - Create necessary directories
   - Update configuration for the Colab environment

4. Download the BUSI dataset using Kaggle API:
   ```python
   # Install Kaggle API
   !pip install -q kaggle
   
   # Upload your Kaggle API token
   from google.colab import files
   print("Upload your kaggle.json file (from your Kaggle account settings)")
   uploaded = files.upload()
   
   # Set up Kaggle credentials
   !mkdir -p ~/.kaggle
   !cp kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   
   # Download and extract dataset
   !kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
   !unzip -q breast-ultrasound-images-dataset.zip -d data/BUSI
   ```

5. Run training and evaluation:
   ```python
   # Train models
   !python main.py --config config.json
   
   # Evaluate models
   !python evaluate.py --model_path checkpoints/densenet_densenet121_best.pt --model_name densenet --variant densenet121 --data_dir data/BUSI --num_classes 3
   ```

6. For more detailed usage, refer to the `breast_ultrasound_classification.ipynb` notebook provided in the repository.

## Configuration

The `config.json` file specifies the experiment parameters:

```json
{
    "experiment_name": "breast_ultrasound_classification",
    "output_dir": "experiments",
    "data": {
        "data_dir": "data/BUSI",
        "batch_size": 32,
        "img_size": 224,
        "num_workers": 4
    },
    "models": [
        {
            "name": "densenet",
            "variant": "densenet121",
            "pretrained": true,
            "epochs": 50,
            "learning_rate": 1e-4,
            "use_scheduler": true
        },
        {
            "name": "resnet",
            "variant": "resnet50",
            "pretrained": true,
            "epochs": 50,
            "learning_rate": 1e-4,
            "use_scheduler": true
        },
        {
            "name": "efficientnet",
            "variant": "efficientnet_b0",
            "pretrained": true,
            "epochs": 50,
            "learning_rate": 1e-4,
            "use_scheduler": true
        }
    ],
    "sample_images": [
        "data/BUSI/benign/benign (1).png",
        "data/BUSI/malignant/malignant (1).png",
        "data/BUSI/normal/normal (1).png"
    ]
}
```

## Troubleshooting

### Common Issues in Google Colab

1. **Package Installation Problems**:
   - If `pytorch-grad-cam` fails to install, try: `!pip install git+https://github.com/jacobgil/pytorch-grad-cam.git`
   - For other packages, install them individually: `!pip install package-name`

2. **Dataset Directory Issues**:
   - Ensure the data directories exist: `!mkdir -p data/BUSI`
   - Check the extracted dataset structure: `!ls -la data/BUSI`

3. **GPU Not Available**:
   - In Colab, go to Runtime → Change runtime type → Hardware accelerator → GPU

4. **Memory Issues**:
   - Reduce batch size in `config.json`
   - Use a smaller model variant (e.g., EfficientNetB0 instead of larger variants)

## Results

The experiment results are saved in the specified output directory:

- Model checkpoints in the `checkpoints/` directory
- Evaluation metrics and visualization in the `results/` directory
- Training history and metrics as JSON files
- Visualization images (ROC curves, confusion matrices, etc.)

## Extending the Project

- **Adding New Models**: Extend the `get_model()` function in `models.py`
- **Custom Datasets**: Follow the structure of `BreastUltrasoundDataset` in `dataloader.py`
- **New Metrics**: Add to the `compute_metrics()` function in `utils/metrics.py`
- **Custom Visualizations**: Extend the visualization functions in `visualize.py`

## License

[MIT License](LICENSE)

## Acknowledgements

- Dataset sources:
  - BUSI Dataset: [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
  - mini-MIAS: [The Mini-MIAS Database of Mammograms](http://peipa.essex.ac.uk/info/mias.html)
- PyTorch and torchvision for deep learning framework
- pytorch-grad-cam for visualization 