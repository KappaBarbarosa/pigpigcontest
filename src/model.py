"""
Model definitions for object detection.
"""
import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, List, Tuple


class ObjectDetectionModel(nn.Module):
    """Object detection model based on pre-trained backbone."""
    
    def __init__(self, num_classes: int = 1, backbone: str = 'resnet50'):
        """
        Initialize the model.
        
        Args:
            num_classes: Number of object classes
            backbone: Backbone architecture ('resnet50', 'resnet101', 'efficientnet')
        """
        super(ObjectDetectionModel, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pre-trained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            feature_dim = 2048
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=True)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5 * num_classes)  # 5 = x, y, w, h, confidence
        )
    
    def forward(self, x):
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        # Detection predictions
        predictions = self.detection_head(features)
        
        # Reshape to [batch_size, num_classes, 5]
        batch_size = x.size(0)
        predictions = predictions.view(batch_size, self.num_classes, 5)
        
        return predictions


class YOLOv10Wrapper:
    """Wrapper for YOLOv10 model."""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        """
        Initialize YOLOv10 wrapper.
        
        Args:
            model_path: Path to pre-trained model or model name
            device: Device to run inference on
        """
        self.device = device
        self.model = None
        
        # Try to import ultralytics with YOLOv10
        try:
            from ultralytics import YOLOv10
            if model_path:
                if model_path.startswith('jameslahm/yolov10'):
                    # Use YOLOv10.from_pretrained for Hugging Face models
                    print(f"Loading YOLOv10 model from Hugging Face: {model_path}")
                    self.model = YOLOv10.from_pretrained(model_path)
                else:
                    # Use regular YOLOv10 loading for local models
                    self.model = YOLOv10(model_path)
            else:
                # Use default YOLOv10 model
                self.model = YOLOv10('yolo10n.pt')
        except ImportError:
            print("Warning: ultralytics with YOLOv10 not available. YOLOv10 functionality disabled.")
    
    def predict(self, image_path: str) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            List of detection results
        """
        if self.model is None:
            return []
        
        # Use YOLOv10's predict method
        results = self.model.predict(image_path, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                        'confidence': boxes.conf[i].cpu().numpy().item(),
                        'class_id': int(boxes.cls[i].cpu().numpy().item())
                    }
                    detections.append(detection)
        
        return detections
    
    def __call__(self, image_tensor):
        """Make the wrapper callable for compatibility with inference code."""
        # This is a placeholder - the actual inference should use predict() method
        # with image paths, not tensors
        raise NotImplementedError("Use predict() method with image path instead")
    
    def predict_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of detection results for each image
        """
        if self.model is None:
            return [[] for _ in image_paths]
        
        # Use YOLOv10's predict method for batch inference
        results = self.model.predict(image_paths, verbose=False)
        
        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                        'confidence': boxes.conf[i].cpu().numpy().item(),
                        'class_id': int(boxes.cls[i].cpu().numpy().item())
                    }
                    detections.append(detection)
            all_detections.append(detections)
        
        return all_detections


def load_model(model_type: str = 'yolov10', **kwargs):
    """
    Load a model for object detection.
    
    Args:
        model_type: Type of model to load ('yolov10', 'yolov10x', 'custom')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Loaded model
    """
    if model_type == 'yolov10':
        return YOLOv10Wrapper(**kwargs)
    elif model_type == 'yolov10x':
        return YOLOv10Wrapper(model_path='jameslahm/yolov10x', **kwargs)
    elif model_type == 'custom':
        return ObjectDetectionModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
