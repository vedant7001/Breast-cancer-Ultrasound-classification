import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split

class BreastUltrasoundDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', test_size=0.15, val_size=0.15, random_state=42):
        """
        Args:
            data_dir (str): Path to the dataset directory.
            transform (callable, optional): Transform to be applied to the images.
            split (str): One of 'train', 'val', or 'test'.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of data to use for validation.
            random_state (int): Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load dataset
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.image_paths, self.labels, test_size=test_size, stratify=self.labels, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size proportion
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, stratify=y_train_val, random_state=random_state
        )
        
        # Select appropriate data split
        if split == 'train':
            self.image_paths, self.labels = X_train, y_train
        elif split == 'val':
            self.image_paths, self.labels = X_val, y_val
        elif split == 'test':
            self.image_paths, self.labels = X_test, y_test
        else:
            raise ValueError(f"Split '{split}' not recognized. Use 'train', 'val', or 'test'.")
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size=224):
    """
    Returns image transforms for training and validation/testing
    
    Args:
        img_size (int): Target image size (default: 224)
    
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' transforms
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform
    }


def get_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Creates train, validation, and test data loaders
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size
        img_size (int): Target image size
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' data loaders
    """
    transforms_dict = get_transforms(img_size)
    
    datasets_dict = {
        split: BreastUltrasoundDataset(
            data_dir=data_dir,
            transform=transforms_dict[split],
            split=split
        ) for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        split: DataLoader(
            dataset=datasets_dict[split],
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers,
            pin_memory=True
        ) for split in ['train', 'val', 'test']
    }
    
    return dataloaders


def load_standard_dataset(dataset_name, batch_size=32, img_size=224, num_workers=4):
    """
    Loads a standard dataset (BUSI or mini-MIAS) from torchvision or local path
    
    Args:
        dataset_name (str): Name of the dataset ('BUSI' or 'mini-MIAS')
        batch_size (int): Batch size
        img_size (int): Target image size
        num_workers (int): Number of worker threads for data loading
        
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' data loaders
    """
    # This is a placeholder for loading standard datasets
    # For actual implementation, you'd need to download or point to these datasets
    if dataset_name.lower() == 'busi':
        data_dir = './data/BUSI'  # Adjust path as needed
    elif dataset_name.lower() == 'mini-mias':
        data_dir = './data/mini-MIAS'  # Adjust path as needed
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized. Use 'BUSI' or 'mini-MIAS'.")
    
    return get_dataloaders(data_dir, batch_size, img_size, num_workers)


if __name__ == '__main__':
    # Example usage:
    dataloaders = get_dataloaders(data_dir='./data/sample_dataset', batch_size=16)
    
    # Print dataset info
    for split in ['train', 'val', 'test']:
        dataset = dataloaders[split].dataset
        print(f"{split} set: {len(dataset)} images, {len(dataset.classes)} classes")
        
        # Sample batch
        images, labels = next(iter(dataloaders[split]))
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}") 