"""
Evaluation metrics for object detection.
Implements mAP@[.5:.95] (COCO-style) evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in xyxy format.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0-1)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def compute_ap_at_iou(
    gt_by_image: Dict[int, List[Dict]],
    preds_by_image: Dict[int, List[Dict]],
    iou_threshold: float,
    confidence_threshold: float = 0.01
) -> float:
    """
    Compute Average Precision at a specific IoU threshold.

    Args:
        gt_by_image: Dict mapping image_id to list of ground truth boxes
            Each box: {'bbox': [x1, y1, x2, y2], 'class_id': int}
        preds_by_image: Dict mapping image_id to list of predictions
            Each pred: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}
        iou_threshold: IoU threshold for matching (e.g., 0.5, 0.75, 0.95)
        confidence_threshold: Minimum confidence to consider a prediction

    Returns:
        AP score (0-1)
    """
    # Collect all predictions across images
    all_predictions = []
    for img_id, preds in preds_by_image.items():
        for pred in preds:
            if pred['confidence'] >= confidence_threshold:
                all_predictions.append({
                    'img_id': img_id,
                    'bbox': pred['bbox'],
                    'confidence': pred['confidence'],
                    'class_id': pred.get('class_id', 0)
                })

    # Sort predictions by confidence (descending)
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    # Count total ground truth boxes
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())

    if total_gt == 0:
        return 0.0

    if len(all_predictions) == 0:
        return 0.0

    # Track which GT boxes have been matched
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in gt_by_image.items()}

    # Compute precision and recall for each prediction
    tp = []  # true positives
    fp = []  # false positives

    for pred in all_predictions:
        img_id = pred['img_id']
        pred_bbox = pred['bbox']

        # Get ground truth boxes for this image
        gt_boxes = gt_by_image.get(img_id, [])

        # Find best matching GT box
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            # Skip if already matched
            if gt_matched[img_id][gt_idx]:
                continue

            iou = compute_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if this is a true positive
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[img_id][best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)

    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute precision and recall
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Add sentinel values at the end for interpolation
    recalls = np.concatenate([[0], recalls, [1]])
    precisions = np.concatenate([[1], precisions, [0]])

    # Interpolate precision: make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Compute AP as area under the interpolated PR curve
    # Using 101-point interpolation (COCO-style)
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 101):
        # Find all precisions at recall >= recall_threshold
        indices = recalls >= recall_threshold
        if np.any(indices):
            ap += np.max(precisions[indices])
    ap /= 101

    return ap


def compute_map_50_95(
    gt_by_image: Dict[int, List[Dict]],
    preds_by_image: Dict[int, List[Dict]],
    confidence_threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute mAP@[.5:.95] (COCO-style evaluation).

    Args:
        gt_by_image: Dict mapping image_id to list of ground truth boxes
            Each box: {'bbox': [x1, y1, x2, y2], 'class_id': int}
        preds_by_image: Dict mapping image_id to list of predictions
            Each pred: {'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int}
        confidence_threshold: Minimum confidence to consider a prediction

    Returns:
        Dict with metrics:
            - 'mAP': mAP@[.5:.95] (average over IoU thresholds 0.5 to 0.95)
            - 'AP50': AP at IoU=0.5
            - 'AP75': AP at IoU=0.75
            - 'AP_per_iou': List of AP at each IoU threshold
    """
    # IoU thresholds from 0.5 to 0.95 with step 0.05 (COCO-style)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    aps = []
    for iou_thresh in iou_thresholds:
        ap = compute_ap_at_iou(
            gt_by_image=gt_by_image,
            preds_by_image=preds_by_image,
            iou_threshold=iou_thresh,
            confidence_threshold=confidence_threshold
        )
        aps.append(ap)

    # Compute mAP as average over all IoU thresholds
    map_50_95 = np.mean(aps)

    # Extract specific metrics
    ap50 = aps[0]   # IoU=0.5
    ap75 = aps[5]   # IoU=0.75 (index 5: 0.5 + 5*0.05 = 0.75)

    return {
        'mAP': float(map_50_95),
        'AP50': float(ap50),
        'AP75': float(ap75),
        'AP_per_iou': [float(ap) for ap in aps],
        'iou_thresholds': iou_thresholds.tolist()
    }


if __name__ == '__main__':
    # Simple test
    gt_by_image = {
        1: [{'bbox': [10, 10, 50, 50], 'class_id': 0}],
        2: [{'bbox': [20, 20, 60, 60], 'class_id': 0}]
    }

    preds_by_image = {
        1: [{'bbox': [12, 12, 48, 48], 'confidence': 0.9, 'class_id': 0}],
        2: [{'bbox': [22, 22, 58, 58], 'confidence': 0.85, 'class_id': 0}]
    }

    metrics = compute_map_50_95(gt_by_image, preds_by_image)
    print(f"Test mAP@[.5:.95]: {metrics['mAP']:.4f}")
    print(f"Test AP50: {metrics['AP50']:.4f}")
    print(f"Test AP75: {metrics['AP75']:.4f}")
