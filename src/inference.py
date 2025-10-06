"""
Inference utilities for object detection models.
"""
import torch
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os


class ObjectDetectionInference:
    """Inference class for object detection models."""
    
    def __init__(self, model, device: str = 'cpu', confidence_threshold: float = 0.5):
        """
        Initialize inference.
        
        Args:
            model: Trained model
            device: Device to run inference on
            confidence_threshold: Confidence threshold for detections
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        if hasattr(model, 'eval'):
            self.model.eval()
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array
            target_size: Target size for resizing
        
        Returns:
            Preprocessed image tensor
        """
        # Resize image
        image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_predictions(
        self,
        predictions: torch.Tensor,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int] = (640, 640)
    ) -> List[Dict]:
        """
        Postprocess model predictions.
        
        Args:
            predictions: Raw model predictions
            original_size: Original image size (height, width)
            target_size: Target size used for preprocessing
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Convert predictions to numpy
        pred_np = predictions.cpu().numpy()
        
        # Scale predictions back to original image size
        scale_x = original_size[1] / target_size[1]
        scale_y = original_size[0] / target_size[0]
        
        for pred in pred_np[0]:  # Assuming batch size of 1
            confidence = pred[4]
            
            if confidence > self.confidence_threshold:
                # Convert from [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = pred[:4]
                x1 = (x - w / 2) * scale_x
                y1 = (y - h / 2) * scale_y
                x2 = (x + w / 2) * scale_x
                y2 = (y + h / 2) * scale_y
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': 0  # Assuming single class
                }
                detections.append(detection)
        
        return detections
    
    def predict_image(self, image_path: str) -> List[Dict]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
        
        Returns:
            List of detection results
        """
        # Check if model has predict method (YOLOv10)
        if hasattr(self.model, 'predict'):
            return self.model.predict(image_path)
        
        # Fallback to tensor-based inference for custom models
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return []
        
        original_size = image.shape[:2]  # (height, width)
        
        # Preprocess
        image_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Postprocess
        detections = self.postprocess_predictions(predictions, original_size)
        
        return detections
    
    def predict_batch(self, image_paths: List[str]) -> List[List[Dict]]:
        """
        Run inference on multiple images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of detection results for each image
        """
        all_detections = []
        
        for image_path in image_paths:
            detections = self.predict_image(image_path)
            all_detections.append(detections)
        
        return all_detections


def format_predictions_for_submission(
    image_ids: List[int],
    all_detections: List[List[Dict]],
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Format predictions for submission.
    
    Args:
        image_ids: List of image IDs
        all_detections: List of detection results for each image
        confidence_threshold: Confidence threshold for filtering
    
    Returns:
        DataFrame formatted for submission
    """
    submission_data = []
    
    for img_id, detections in zip(image_ids, all_detections):
        # Filter by confidence threshold
        filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
        
        if not filtered_detections:
            # No detections above threshold
            submission_data.append({
                'Image_ID': img_id,
                'PredictionString': ''
            })
        else:
            # Format predictions as string
            pred_strings = []
            for det in filtered_detections:
                bbox = det['bbox']
                conf = det['confidence']
                class_id = det['class_id']
                
                # Format: confidence x1 y1 x2 y2 class_id
                pred_string = f"{conf:.6f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} {class_id}"
                pred_strings.append(pred_string)
            
            prediction_string = ' '.join(pred_strings)
            submission_data.append({
                'Image_ID': img_id,
                'PredictionString': prediction_string
            })
    
    return pd.DataFrame(submission_data)


def run_inference_on_test_set(
    model,
    test_img_dir: str,
    output_file: str = 'predictions.csv',
    device: str = 'cpu',
    confidence_threshold: float = 0.5
):
    """
    Run inference on the entire test set.
    
    Args:
        model: Trained model
        test_img_dir: Directory containing test images
        output_file: Output file for predictions
        device: Device to run inference on
        confidence_threshold: Confidence threshold for detections
    """
    # Initialize inference
    inference = ObjectDetectionInference(
        model=model,
        device=device,
        confidence_threshold=confidence_threshold
    )
    
    # Get test image paths
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.jpg')])
    test_paths = [os.path.join(test_img_dir, img) for img in test_images]
    
    # Get image IDs
    image_ids = [int(os.path.splitext(img)[0]) for img in test_images]
    
    print(f"Running inference on {len(test_images)} test images...")
    
    # Run inference
    all_detections = inference.predict_batch(test_paths)
    
    # Format for submission
    submission_df = format_predictions_for_submission(
        image_ids=image_ids,
        all_detections=all_detections,
        confidence_threshold=confidence_threshold
    )
    
    # Save predictions
    submission_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return submission_df


def visualize_predictions(
    image_path: str,
    detections: List[Dict],
    output_path: str = None,
    confidence_threshold: float = 0.5
):
    """
    Visualize predictions on an image.
    
    Args:
        image_path: Path to input image
        detections: List of detection results
        output_path: Path to save visualization (optional)
        confidence_threshold: Confidence threshold for visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Draw detections
    for det in detections:
        if det['confidence'] >= confidence_threshold:
            bbox = det['bbox']
            conf = det['confidence']
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"{conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save or display image
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to {output_path}")
    else:
        cv2.imshow('Predictions', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
