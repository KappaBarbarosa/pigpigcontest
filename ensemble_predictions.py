#!/usr/bin/env python3
"""
Ensemble Predictions Script
Combines two CSV prediction files using weighted box fusion (WBF)
"""

import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import warnings
from ensemble_boxes import weighted_boxes_fusion
warnings.filterwarnings('ignore')


def parse_prediction_string(pred_string: str) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Parse prediction string into list of (confidence, x, y, w, h, class_id) tuples
    """
    if pd.isna(pred_string) or pred_string == '':
        return []
    
    values = pred_string.strip().split()
    predictions = []
    
    # Each prediction has 6 values: confidence, x, y, w, h, class_id
    for i in range(0, len(values), 6):
        if i + 5 < len(values):
            try:
                conf = float(values[i])
                x = float(values[i + 1])
                y = float(values[i + 2])
                w = float(values[i + 3])
                h = float(values[i + 4])
                class_id = int(values[i + 5])
                predictions.append((conf, x, y, w, h, class_id))
            except (ValueError, IndexError):
                continue
    
    return predictions


def weighted_box_fusion(predictions: List[Tuple[float, float, float, float, float, int]], 
                       iou_threshold: float = 0.5, 
                       skip_box_threshold: float = 0.0001) -> List[Tuple[float, float, float, float, float, int]]:
    """
    Apply Weighted Box Fusion (WBF) using ensemble_boxes library
    """
    if not predictions:
        return []
    
    # Filter out low confidence predictions
    filtered_predictions = [pred for pred in predictions if pred[0] >= skip_box_threshold]
    
    if not filtered_predictions:
        return []
    
    # Group predictions by class
    class_groups = {}
    for pred in filtered_predictions:
        class_id = pred[5]
        if class_id not in class_groups:
            class_groups[class_id] = []
        class_groups[class_id].append(pred)
    
    fused_predictions = []
    
    for class_id, class_predictions in class_groups.items():
        if not class_predictions:
            continue
            
        # Prepare data for ensemble_boxes
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for conf, x, y, w, h, _ in class_predictions:
            # Convert to (x1, y1, x2, y2) format for ensemble_boxes
            # CSV coordinates are already scaled back to original image size (640x360)
            img_width, img_height = 640, 360
            x1, y1 = x / img_width, y / img_height
            x2, y2 = (x + w) / img_width, (y + h) / img_height
            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(conf)
            labels_list.append(class_id)
        
        if not boxes_list:
            continue
        
        # Apply weighted box fusion
        try:
            boxes, scores, labels = weighted_boxes_fusion(
                [boxes_list],
                [scores_list], 
                [labels_list],
                weights=None,
                iou_thr=iou_threshold,
                skip_box_thr=skip_box_threshold
            )
            
            # Convert back to our format
            img_width, img_height = 640, 360
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                # Convert back to pixel coordinates (original image size)
                x1, y1 = x1 * img_width, y1 * img_height
                x2, y2 = x2 * img_width, y2 * img_height
                w = x2 - x1
                h = y2 - y1
                conf = scores[i]
                label = int(labels[i])
                fused_predictions.append((conf, x1, y1, w, h, label))
                
        except Exception as e:
            print(f"Warning: Error in weighted box fusion for class {class_id}: {e}")
            # Fallback: keep original predictions
            fused_predictions.extend(class_predictions)
    
    # Sort by confidence and return
    return sorted(fused_predictions, key=lambda x: x[0], reverse=True)


def format_prediction_string(predictions: List[Tuple[float, float, float, float, float, int]]) -> str:
    """
    Format predictions back to string format
    """
    if not predictions:
        return ""
    
    formatted_predictions = []
    for conf, x, y, w, h, class_id in predictions:
        formatted_predictions.append(f"{conf:.6f} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {class_id}")
    
    return " ".join(formatted_predictions)


def ensemble_predictions(csv1_path: str, csv2_path: str, output_path: str, 
                        iou_threshold: float = 0.5, skip_box_threshold: float = 0.0001):
    """
    Ensemble predictions from two CSV files using weighted box fusion
    """
    print(f"Loading predictions from {csv1_path} and {csv2_path}")
    
    # Load CSV files
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Ensure both have the same Image_IDs
    common_ids = set(df1['Image_ID']) & set(df2['Image_ID'])
    print(f"Found {len(common_ids)} common images")
    
    # Create result dataframe
    result_data = []
    
    for image_id in sorted(common_ids):
        # Get predictions from both files
        pred1_str = df1[df1['Image_ID'] == image_id]['PredictionString'].iloc[0]
        pred2_str = df2[df2['Image_ID'] == image_id]['PredictionString'].iloc[0]
        
        # Parse predictions
        pred1_list = parse_prediction_string(pred1_str)
        pred2_list = parse_prediction_string(pred2_str)
        
        # Combine all predictions
        all_predictions = pred1_list + pred2_list
        
        # Apply weighted box fusion
        fused_predictions = weighted_box_fusion(all_predictions, iou_threshold, skip_box_threshold)
        
        # Format back to string
        ensemble_pred_str = format_prediction_string(fused_predictions)
        
        result_data.append({
            'Image_ID': image_id,
            'PredictionString': ensemble_pred_str
        })
    
    # Create output dataframe
    result_df = pd.DataFrame(result_data)
    
    # Save to output file
    result_df.to_csv(output_path, index=False)
    print(f"Ensemble predictions saved to {output_path}")
    print(f"Total images processed: {len(result_df)}")


def main():
    parser = argparse.ArgumentParser(description='Ensemble predictions using weighted box fusion')
    parser.add_argument('csv1', help='Path to first CSV prediction file')
    parser.add_argument('csv2', help='Path to second CSV prediction file')
    parser.add_argument('--output', '-o', default='ensemble_predictions.csv', 
                       help='Output CSV file path (default: ensemble_predictions.csv)')
    parser.add_argument('--iou-threshold', type=float, default=0.8,
                       help='IoU threshold for box fusion (default: 0.8)')
    parser.add_argument('--skip-threshold', type=float, default=0.0001,
                       help='Skip boxes with confidence below this threshold (default: 0.0001)')
    
    args = parser.parse_args()
    
    ensemble_predictions(
        args.csv1, 
        args.csv2, 
        args.output,
        args.iou_threshold,
        args.skip_threshold
    )


if __name__ == "__main__":
    main()
