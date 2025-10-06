#!/usr/bin/env python3
"""
Test script for YOLOv10x model inference.
"""

import os
import time
from src.model import load_model
from src.inference import ObjectDetectionInference, run_inference_on_test_set
import pandas as pd


def test_single_image():
    """Test inference on a single image."""
    print("=== Testing Single Image Inference ===")
    
    # Find a test image
    test_img_dir = 'test/img'
    if not os.path.exists(test_img_dir):
        print(f"Test directory {test_img_dir} not found")
        return
    
    test_images = [f for f in os.listdir(test_img_dir) if f.endswith('.jpg')]
    if not test_images:
        print("No test images found")
        return
    
    # Load YOLOv10x model
    print("Loading YOLOv10x model...")
    try:
        model = load_model(model_type='yolov10x', device='cpu')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test on first image
    test_image_path = os.path.join(test_img_dir, test_images[0])
    print(f"Testing on: {test_images[0]}")
    
    # Create inference object
    inference = ObjectDetectionInference(
        model=model,
        device='cpu',
        confidence_threshold=0.3
    )
    
    # Run inference
    start_time = time.time()
    detections = inference.predict_image(test_image_path)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Number of detections: {len(detections)}")
    
    # Print detections
    for i, det in enumerate(detections):
        print(f"Detection {i+1}:")
        print(f"  Confidence: {det['confidence']:.4f}")
        print(f"  Bounding box: {det['bbox']}")
        print(f"  Class ID: {det['class_id']}")


def test_batch_inference():
    """Test batch inference on multiple images."""
    print("\n=== Testing Batch Inference ===")
    
    # Load YOLOv10x model
    print("Loading YOLOv10x model...")
    try:
        model = load_model(model_type='yolov10x', device='cpu')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run inference on test set
    print("Running inference on test set...")
    start_time = time.time()
    
    try:
        submission_df = run_inference_on_test_set(
            model=model,
            test_img_dir='test/img',
            output_file='yolov10x_predictions.csv',
            device='cpu',
            confidence_threshold=0.3
        )
        
        total_time = time.time() - start_time
        print(f"✅ Batch inference completed in {total_time:.2f} seconds")
        print(f"Processed {len(submission_df)} images")
        
        # Show some statistics
        total_detections = 0
        for _, row in submission_df.iterrows():
            if row['PredictionString']:
                detections = row['PredictionString'].split()
                total_detections += len(detections) // 6  # 6 values per detection
        
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(submission_df):.2f}")
        
        # Show first few predictions
        print("\nFirst 5 predictions:")
        for i, (_, row) in enumerate(submission_df.head().iterrows()):
            print(f"Image {row['Image_ID']}: {row['PredictionString'][:100]}...")
            if i >= 4:
                break
                
    except Exception as e:
        print(f"❌ Error during batch inference: {e}")


def test_model_info():
    """Test model information and capabilities."""
    print("\n=== Testing Model Information ===")
    
    try:
        model = load_model(model_type='yolov10x', device='cpu')
        print("✅ Model loaded successfully")
        
        # Try to get model info
        if hasattr(model, 'model') and model.model is not None:
            print(f"Model type: {type(model.model)}")
            
            # Try to get model parameters
            if hasattr(model.model, 'model'):
                print(f"Model architecture: {type(model.model.model)}")
            
            print("Model is ready for inference")
        else:
            print("Model object created but model not loaded")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main test function."""
    print("🚀 YOLOv10x Model Testing")
    print("=" * 50)
    
    # Check if test directory exists
    if not os.path.exists('test/img'):
        print("❌ Test directory 'test/img' not found")
        print("Please make sure you're running from the project root directory")
        return
    
    # Run tests
    test_model_info()
    test_single_image()
    test_batch_inference()
    
    print("\n🎉 Testing completed!")
    print("\nTo evaluate the results:")
    print("uv run python evaluate.py --pred_file yolov10x_predictions.csv --gt_file train/gt.txt")


if __name__ == "__main__":
    main()
