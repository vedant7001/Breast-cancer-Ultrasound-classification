import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataloader import get_dataloaders, load_standard_dataset
from models import get_model


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, scheduler=None, 
                 device='cuda', model_name='model', save_dir='checkpoints'):
        """
        Trainer class for model training and evaluation
        
        Args:
            model (nn.Module): Model to train
            dataloaders (dict): Dictionary with 'train', 'val', 'test' dataloaders
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device (str): Device to use ('cuda' or 'cpu')
            model_name (str): Name of the model for saving
            save_dir (str): Directory to save checkpoints
        """
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'best_epoch': 0,
            'best_val_acc': 0
        }
        
    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(self.dataloaders['train'], desc='Training'):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Track predictions
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """
        Validate the model
        
        Returns:
            tuple: (loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders['val'], desc='Validating'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item() * inputs.size(0)
                
                # Track predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.dataloaders['val'].dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs=50, patience=10):
        """
        Train the model
        
        Args:
            num_epochs (int): Number of epochs to train
            patience (int): Early stopping patience
        
        Returns:
            dict: Training history
        """
        start_time = time.time()
        best_val_acc = 0.0
        best_model_wts = self.model.state_dict()
        no_improve_counter = 0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 20)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, _, _ = self.validate_epoch()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch results
            print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_wts = self.model.state_dict()
                self.history['best_epoch'] = epoch
                self.history['best_val_acc'] = best_val_acc
                no_improve_counter = 0
                
                # Save best model
                torch.save(best_model_wts, os.path.join(self.save_dir, f'{self.model_name}_best.pt'))
                
                # Save history
                with open(os.path.join(self.save_dir, f'{self.model_name}_history.json'), 'w') as f:
                    json.dump(self.history, f)
            else:
                no_improve_counter += 1
            
            # Early stopping
            if no_improve_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            print()
        
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_val_acc:.4f} at epoch {self.history["best_epoch"]+1}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        return self.history
    
    def evaluate(self, dataset_split='test'):
        """
        Evaluate the model on test data
        
        Args:
            dataset_split (str): Which dataset split to evaluate on (default: 'test')
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders[dataset_split], desc=f'Evaluating on {dataset_split}'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                # Track predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        # Calculate ROC-AUC score (only for binary classification)
        if len(np.unique(all_labels)) == 2:
            metrics['roc_auc'] = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # For multi-class, we compute one-vs-rest ROC AUC and average
            metrics['roc_auc'] = roc_auc_score(
                np.eye(len(np.unique(all_labels)))[all_labels],
                all_probs,
                multi_class='ovr',
                average='weighted'
            )
        
        # Print metrics
        print(f"\nEvaluation on {dataset_split} data:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Save metrics
        with open(os.path.join(self.save_dir, f'{self.model_name}_{dataset_split}_metrics.json'), 'w') as f:
            json.dump(metrics, f)
        
        # Save predictions and labels for later analysis
        np.savez(
            os.path.join(self.save_dir, f'{self.model_name}_{dataset_split}_predictions.npz'),
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs
        )
        
        return metrics, all_preds, all_labels, all_probs


def train_model(model_name, data_dir, num_classes=2, pretrained=True, variant=None, 
                batch_size=32, num_epochs=50, learning_rate=1e-4, use_scheduler=True, device=None):
    """
    Train and evaluate a model
    
    Args:
        model_name (str): Model name ('densenet', 'resnet', or 'efficientnet')
        data_dir (str): Path to dataset directory
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
        variant (str, optional): Specific model variant
        batch_size (int): Batch size
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        use_scheduler (bool): Whether to use learning rate scheduler
        device (str, optional): Device to use ('cuda' or 'cpu')
        
    Returns:
        tuple: (model, trainer, metrics)
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Get dataloaders
    dataloaders = get_dataloaders(data_dir, batch_size=batch_size)
    
    # Create model
    model = get_model(model_name, num_classes=num_classes, pretrained=pretrained, variant=variant)
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define scheduler
    scheduler = None
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_name=f"{model_name}_{variant or ''}"
    )
    
    # Train model
    trainer.train(num_epochs=num_epochs)
    
    # Evaluate model
    metrics, _, _, _ = trainer.evaluate(dataset_split='test')
    
    return model, trainer, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model for breast ultrasound classification')
    parser.add_argument('--model', type=str, default='densenet', choices=['densenet', 'resnet', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--variant', type=str, default=None,
                        help='Specific model variant (e.g., densenet121, resnet50, efficientnet_b0)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes in the dataset')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--no_scheduler', action='store_true',
                        help='Disable learning rate scheduler')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    model, trainer, metrics = train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        variant=args.variant,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_scheduler=not args.no_scheduler,
        device='cpu' if args.cpu else None
    ) 