"""
Evaluation metrics for object detection tasks.
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(precision: List[float], recall: List[float]) -> float:
    """
    Calculate Average Precision (AP) from precision and recall arrays.
    
    Args:
        precision: List of precision values
        recall: List of recall values
    
    Returns:
        Average Precision value
    """
    # Add sentinel values at the end
    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [0]))
    
    # Compute the precision envelope
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    # Find points where recall changes
    indices = np.where(recall[1:] != recall[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap


def evaluate_detections(
    ground_truth: List[Dict],
    predictions: List[Dict],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate object detection predictions against ground truth.
    
    Args:
        ground_truth: List of ground truth detections
        predictions: List of predicted detections
        iou_threshold: IoU threshold for considering a detection as correct
        confidence_threshold: Confidence threshold for filtering predictions
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Filter predictions by confidence threshold
    filtered_predictions = [p for p in predictions if p['confidence'] >= confidence_threshold]
    
    if not filtered_predictions:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'ap': 0.0,
            'num_gt': len(ground_truth),
            'num_pred': 0,
            'num_tp': 0,
            'num_fp': 0,
            'num_fn': len(ground_truth)
        }
    
    # Sort predictions by confidence (descending)
    filtered_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Initialize tracking arrays
    tp = np.zeros(len(filtered_predictions))
    fp = np.zeros(len(filtered_predictions))
    gt_matched = np.zeros(len(ground_truth), dtype=bool)
    
    # For each prediction, find the best matching ground truth
    for i, pred in enumerate(filtered_predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truth):
            if gt_matched[j]:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp[i] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[i] = 1
    
    # Calculate cumulative precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(ground_truth)
    
    # Calculate metrics
    num_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
    num_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
    num_fn = len(ground_truth) - num_tp
    
    final_precision = precision[-1] if len(precision) > 0 else 0.0
    final_recall = recall[-1] if len(recall) > 0 else 0.0
    
    f1_score = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0.0
    
    # Calculate Average Precision
    ap = calculate_ap(precision, recall)
    
    return {
        'precision': final_precision,
        'recall': final_recall,
        'f1_score': f1_score,
        'ap': ap,
        'num_gt': len(ground_truth),
        'num_pred': len(filtered_predictions),
        'num_tp': num_tp,
        'num_fp': num_fp,
        'num_fn': num_fn
    }


def evaluate_submission(
    ground_truth_file: str,
    prediction_file: str,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate a submission file against ground truth.
    
    Args:
        ground_truth_file: Path to ground truth file
        prediction_file: Path to prediction file
        iou_threshold: IoU threshold for evaluation
        confidence_threshold: Confidence threshold for filtering
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load ground truth
    gt_data = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                img_id = int(parts[0])
                x, y, w, h = map(int, parts[1:5])
                
                if img_id not in gt_data:
                    gt_data[img_id] = []
                
                # Convert to [x1, y1, x2, y2] format
                bbox = [x, y, x + w, y + h]
                gt_data[img_id].append({'bbox': bbox, 'class_id': 0})
    
    # Load predictions
    pred_data = pd.read_csv(prediction_file)
    
    all_metrics = []
    
    for _, row in pred_data.iterrows():
        img_id = row['Image_ID']
        pred_string = row['PredictionString']
        
        if img_id not in gt_data:
            continue
        
        # Parse prediction string
        predictions = []
        if pred_string and pred_string.strip():
            parts = pred_string.strip().split()
            for i in range(0, len(parts), 6):  # 6 values per detection
                if i + 5 < len(parts):
                    conf = float(parts[i])
                    x1, y1, x2, y2 = map(float, parts[i+1:i+5])
                    class_id = int(parts[i+5])
                    
                    predictions.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': class_id
                    })
        
        # Evaluate this image
        metrics = evaluate_detections(
            ground_truth=gt_data[img_id],
            predictions=predictions,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold
        )
        all_metrics.append(metrics)
    
    # Calculate average metrics
    if not all_metrics:
        return {
            'mAP': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'num_images': 0
        }
    
    avg_metrics = {
        'mAP': np.mean([m['ap'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'num_images': len(all_metrics)
    }
    
    return avg_metrics


def compute_map_50_95(
    gt_by_image: Dict[int, List[Dict]],
    preds_by_image: Dict[int, List[Dict]],
    iou_thresholds: List[float] | None = None,
    confidence_threshold: float = 0.0,
) -> Dict[str, float]:
    """
    Compute mAP@[.5:.95] given ground truth and predictions grouped by image.

    Args:
        gt_by_image: mapping image_id -> list of gt dicts with 'bbox' in [x1,y1,x2,y2]
        preds_by_image: mapping image_id -> list of pred dicts with 'bbox', 'confidence'
        iou_thresholds: list of IoU thresholds to evaluate. Defaults to 0.50:0.95 step 0.05
        confidence_threshold: filter predictions below this confidence

    Returns:
        Dict with mAP, AP50, AP75 and counts
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]

    # Flatten predictions across images, maintaining image_id
    flat_preds = []  # (confidence, image_id, bbox)
    num_gt = 0
    for img_id, gts in gt_by_image.items():
        num_gt += len(gts)
        preds = preds_by_image.get(img_id, [])
        for p in preds:
            if p.get('confidence', 0.0) >= confidence_threshold:
                flat_preds.append((float(p['confidence']), img_id, p['bbox']))

    if num_gt == 0:
        return {'mAP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'num_images': len(gt_by_image), 'num_gt': 0}

    # Sort predictions by confidence (desc)
    flat_preds.sort(key=lambda t: t[0], reverse=True)

    aps = []
    ap50 = 0.0
    ap75 = 0.0

    for t in iou_thresholds:
        # For each IoU threshold compute PR and AP
        tp_flags = np.zeros(len(flat_preds))
        fp_flags = np.zeros(len(flat_preds))

        # Track matched GT per image to avoid double matching
        matched_per_image = {img_id: np.zeros(len(gt_by_image[img_id]), dtype=bool) for img_id in gt_by_image}

        # Precompute gt bboxes arrays for faster IoU computation
        gt_arrays = {img_id: np.array([gt['bbox'] for gt in gts], dtype=float) if gts else np.zeros((0, 4))
                     for img_id, gts in gt_by_image.items()}

        for i, (_, img_id, pred_bbox) in enumerate(flat_preds):
            gt_bboxes = gt_arrays.get(img_id, None)
            if gt_bboxes is None or gt_bboxes.shape[0] == 0:
                fp_flags[i] = 1
                continue

            # Compute IoU with all unmatched GTs
            best_iou = 0.0
            best_j = -1

            for j, gt_bbox in enumerate(gt_bboxes):
                if matched_per_image[img_id][j]:
                    continue
                iou = calculate_iou(pred_bbox, gt_bbox.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= t and best_j != -1:
                tp_flags[i] = 1
                matched_per_image[img_id][best_j] = True
            else:
                fp_flags[i] = 1

        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)
        recall = tp_cum / max(num_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        ap = calculate_ap(precision, recall)
        aps.append(ap)
        if abs(t - 0.5) < 1e-6:
            ap50 = ap
        if abs(t - 0.75) < 1e-6:
            ap75 = ap

    mAP = float(np.mean(aps)) if aps else 0.0

    return {
        'mAP': mAP,
        'AP50': float(ap50),
        'AP75': float(ap75),
        'num_images': len(gt_by_image),
        'num_gt': num_gt,
    }


def print_evaluation_results(metrics: Dict[str, float]):
    """Print evaluation results in a formatted way."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of images evaluated: {metrics['num_images']}")
    print(f"Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    print(f"Average Precision: {metrics['precision']:.4f}")
    print(f"Average Recall: {metrics['recall']:.4f}")
    print(f"Average F1 Score: {metrics['f1_score']:.4f}")
    print("="*50)


# Example usage
if __name__ == "__main__":
    # Example evaluation
    gt = [
        {'bbox': [100, 100, 200, 200], 'class_id': 0},
        {'bbox': [300, 300, 400, 400], 'class_id': 0}
    ]
    
    pred = [
        {'bbox': [105, 105, 195, 195], 'confidence': 0.9, 'class_id': 0},
        {'bbox': [310, 310, 390, 390], 'confidence': 0.8, 'class_id': 0},
        {'bbox': [500, 500, 600, 600], 'confidence': 0.7, 'class_id': 0}  # False positive
    ]
    
    results = evaluate_detections(gt, pred)
    print_evaluation_results(results)
