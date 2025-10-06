"""
TAICA CVPDL 2025 Homework 1 - Object Detection Project
Main execution script for training and inference.
"""

import os
import torch
import argparse
from src.data_loader import create_data_loaders
from src.model import load_model
from src.train import train_model
from src.inference import run_inference_on_test_set


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Object Detection Training and Inference')
    parser.add_argument('--mode', choices=['train', 'inference'], required=True,
                       help='Mode: train or inference')
    parser.add_argument('--model_type', default='yolov10', choices=['yolov10', 'yolov10x', 'custom'],
                       help='Type of model to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', default='auto',
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for inference')
    parser.add_argument('--output_file', default='predictions.csv',
                       help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        print("Starting training...")
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            train_img_dir='train/img',
            train_gt_file='train/gt.txt',
            test_img_dir='test/img',
            batch_size=args.batch_size
        )
        
        # Load model
        if args.model_type == 'custom':
            model = load_model(
                model_type='custom',
                num_classes=1,
                backbone='resnet50'
            )
        else:
            model = load_model(model_type='yolov10')
        
        # Train model
        if args.model_type == 'custom':
            trainer = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=device
            )
        else:
            print("YOLOv10 training not implemented in this example.")
            print("Please use the YOLOv10 training scripts directly.")
    
    elif args.mode == 'inference':
        print("Starting inference...")
        
        # Load model
        model = load_model(model_type=args.model_type)
        
        # Run inference
        run_inference_on_test_set(
            model=model,
            test_img_dir='test/img',
            output_file=args.output_file,
            device=device,
            confidence_threshold=args.confidence_threshold
        )
        
        print(f"Inference completed. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
