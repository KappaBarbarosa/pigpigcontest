"""
Example script demonstrating how to use the object detection project.
"""

import os
import torch
from src.data_loader import ObjectDetectionDataset, create_data_loaders
from src.model import load_model
from src.inference import ObjectDetectionInference, visualize_predictions


def example_data_loading():
    """Example of loading and exploring the dataset."""
    print("=== Data Loading Example ===")
    
    # Create dataset
    train_dataset = ObjectDetectionDataset(
        img_dir='train/img',
        gt_file='train/gt.txt'
    )
    
    print(f"Number of training images: {len(train_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample image ID: {sample['image_id']}")
    print(f"Number of annotations: {len(sample['targets']) if sample['targets'] else 0}")
    
    if sample['targets']:
        print(f"First annotation: {sample['targets'][0]}")


def example_model_loading():
    """Example of loading different models."""
    print("\n=== Model Loading Example ===")
    
    # Load custom model
    try:
        custom_model = load_model(
            model_type='custom',
            num_classes=1,
            backbone='resnet50'
        )
        print("Custom model loaded successfully")
        print(f"Model type: {type(custom_model)}")
    except Exception as e:
        print(f"Error loading custom model: {e}")
    
    # Load YOLOv10 model
    try:
        yolov10_model = load_model(model_type='yolov10')
        print("YOLOv10 model loaded successfully")
        print(f"Model type: {type(yolov10_model)}")
    except Exception as e:
        print(f"Error loading YOLOv10 model: {e}")


def example_inference():
    """Example of running inference."""
    print("\n=== Inference Example ===")
    
    # Check if test images exist
    test_img_dir = 'test/img'
    if not os.path.exists(test_img_dir):
        print(f"Test directory {test_img_dir} not found")
        return
    
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    if not test_images:
        print("No test images found")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Load model for inference
    try:
        model = load_model(model_type='yolov10')
        inference = ObjectDetectionInference(
            model=model,
            device='cpu',
            confidence_threshold=0.5
        )
        
        # Run inference on first test image
        test_image_path = os.path.join(test_img_dir, test_images[0])
        print(f"Running inference on: {test_images[0]}")
        
        detections = inference.predict_image(test_image_path)
        print(f"Found {len(detections)} detections")
        
        for i, det in enumerate(detections):
            print(f"Detection {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
        
    except Exception as e:
        print(f"Error during inference: {e}")


def example_data_visualization():
    """Example of visualizing data and predictions."""
    print("\n=== Data Visualization Example ===")
    
    # Check if we have training images
    train_img_dir = 'train/img'
    if not os.path.exists(train_img_dir):
        print(f"Training directory {train_img_dir} not found")
        return
    
    train_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    if not train_images:
        print("No training images found")
        return
    
    # Load a sample image
    sample_image_path = os.path.join(train_img_dir, train_images[0])
    print(f"Loading sample image: {train_images[0]}")
    
    # Load dataset to get annotations
    dataset = ObjectDetectionDataset(
        img_dir=train_img_dir,
        gt_file='train/gt.txt'
    )
    
    # Get sample data
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Annotations: {sample['targets']}")
    
    # Try to visualize (this would require matplotlib/opencv)
    print("Note: To visualize images, you can use the visualize_predictions function")


def main():
    """Run all examples."""
    print("TAICA CVPDL 2025 Homework 1 - Example Usage")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('train') or not os.path.exists('test'):
        print("Error: Please run this script from the project root directory")
        print("Make sure 'train' and 'test' directories exist")
        return
    
    # Run examples
    example_data_loading()
    example_model_loading()
    example_inference()
    example_data_visualization()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run training:")
    print("python main.py --mode train --model_type custom --epochs 5")
    print("\nTo run inference:")
    print("python main.py --mode inference --model_type yolov10")


if __name__ == "__main__":
    main()
