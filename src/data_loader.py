"""
Data loading utilities for the computer vision project.
"""
import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


class ObjectDetectionDataset(Dataset):
    """Custom dataset for object detection tasks."""
    
    def __init__(self, img_dir: str, gt_file: str = None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_dir: Directory containing images
            gt_file: Path to ground truth file (optional)
            transform: Optional transform to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        
        # Load ground truth if provided
        self.annotations = None
        if gt_file and os.path.exists(gt_file):
            self.annotations = self._load_annotations(gt_file)
    
    def _load_annotations(self, gt_file: str) -> dict:
        """Load ground truth annotations from file."""
        annotations = {}
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    img_id = int(parts[0])
                    x, y, w, h = map(int, parts[1:5])
                    class_id = int(parts[5]) if len(parts) > 5 else 0  # Default to class 0
                    
                    if img_id not in annotations:
                        annotations[img_id] = []
                    
                    # Convert to [x1, y1, x2, y2] format for consistency
                    bbox = [x, y, x + w, y + h]
                    annotations[img_id].append({
                        'bbox': bbox,
                        'class_id': class_id
                    })
        return annotations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get item by index."""
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get image ID from filename
        img_id = int(os.path.splitext(img_name)[0])
        
        # Get annotations if available
        targets = None
        if self.annotations and img_id in self.annotations:
            targets = self.annotations[img_id]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_id': img_id,
            'targets': targets,
            'image_name': img_name
        }


def create_data_loaders(
    train_img_dir: str,
    train_gt_file: str,
    test_img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test data loaders.
    
    Args:
        train_img_dir: Training images directory
        train_gt_file: Training ground truth file
        test_img_dir: Test images directory
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = ObjectDetectionDataset(train_img_dir, train_gt_file)
    test_dataset = ObjectDetectionDataset(test_img_dir)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    image_ids = []
    targets = []
    image_names = []
    
    for item in batch:
        images.append(item['image'])
        image_ids.append(item['image_id'])
        targets.append(item['targets'])
        image_names.append(item['image_name'])
    
    return {
        'images': images,
        'image_ids': image_ids,
        'targets': targets,
        'image_names': image_names
    }
