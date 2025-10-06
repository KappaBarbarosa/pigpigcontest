"""
Training utilities for object detection models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, List, Tuple


class ObjectDetectionTrainer:
    """Trainer class for object detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Setup optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = self._get_loss_function()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def _get_loss_function(self):
        """Get loss function for object detection."""
        return nn.MSELoss()  # Simplified loss for demonstration
    
    def _format_targets(self, targets: List[Dict]) -> torch.Tensor:
        """
        Format targets for training.
        
        Args:
            targets: List of target dictionaries
        
        Returns:
            Formatted target tensor
        """
        # This is a simplified version - in practice, you'd need proper target formatting
        batch_size = len(targets)
        num_classes = 1  # Assuming single class for now
        
        formatted_targets = torch.zeros(batch_size, num_classes, 5)
        
        for i, target_list in enumerate(targets):
            if target_list:
                # Use the first detection as target (simplified)
                target = target_list[0]
                bbox = target['bbox']
                # Convert to [x, y, w, h, confidence] format
                formatted_targets[i, 0, :4] = torch.tensor(bbox, dtype=torch.float32)
                formatted_targets[i, 0, 4] = 1.0  # Confidence
        
        return formatted_targets.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            images = batch['images']
            targets = batch['targets']
            
            # Convert images to tensor if needed
            if not isinstance(images, torch.Tensor):
                images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
                                    for img in images])
            
            images = images.to(self.device)
            
            # Format targets
            formatted_targets = self._format_targets(targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Compute loss
            loss = self.criterion(predictions, formatted_targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                images = batch['images']
                targets = batch['targets']
                
                # Convert images to tensor if needed
                if not isinstance(images, torch.Tensor):
                    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 
                                        for img in images])
                
                images = images.to(self.device)
                
                # Format targets
                formatted_targets = self._format_targets(targets)
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                loss = self.criterion(predictions, formatted_targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        num_epochs: int = 10,
        save_dir: str = './checkpoints'
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pth'))
                    print("New best model saved!")
            
            # Save checkpoint
            self.save_checkpoint(os.path.join(save_dir, f'epoch_{epoch + 1}.pth'))
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {filepath}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cpu',
    save_dir: str = './checkpoints'
):
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    trainer = ObjectDetectionTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=save_dir
    )
    
    return trainer
