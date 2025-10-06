#!/usr/bin/env python3
"""
Evaluation script for object detection predictions.
"""

import argparse
import os
from src.evaluation import evaluate_submission, print_evaluation_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate object detection predictions')
    parser.add_argument('--gt_file', default='train/gt.txt',
                       help='Ground truth file path')
    parser.add_argument('--pred_file', required=True,
                       help='Prediction file path (CSV format)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for evaluation (default: 0.5)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for filtering predictions (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.gt_file):
        print(f"Error: Ground truth file {args.gt_file} not found")
        return
    
    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file {args.pred_file} not found")
        return
    
    print(f"Evaluating predictions from: {args.pred_file}")
    print(f"Against ground truth from: {args.gt_file}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    
    # Run evaluation
    try:
        metrics = evaluate_submission(
            ground_truth_file=args.gt_file,
            prediction_file=args.pred_file,
            iou_threshold=args.iou_threshold,
            confidence_threshold=args.confidence_threshold
        )
        
        print_evaluation_results(metrics)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main()
