#!/usr/bin/env python3
import os
import shutil
import argparse
from typing import List, Tuple, Dict
import cv2
import numpy as np
import torch
import albumentations as A
from pathlib import Path
import glob
import tempfile

from ultralytics import YOLO


def list_images_sorted(img_dir: str) -> List[str]:
    """List all .jpg images in directory, sorted numerically."""
    files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return files


def rgb_to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale but keep 3 channels (R=G=B)."""
    if len(image.shape) == 2:
        # Already grayscale, expand to 3 channels
        gray = image
    else:
        # Convert to grayscale using standard weights
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Stack to 3 channels
    return np.stack([gray, gray, gray], axis=-1)


class GrayscaleTransform(A.ImageOnlyTransform):
    """Convert image to grayscale (RGB format with R=G=B)."""

    def apply(self, img, **params):
        return rgb_to_grayscale_rgb(img)

    def get_transform_init_args_names(self):
        return ()


def get_domain_augmentations(imgsz: int = 640, mode: str = 'color'):
    """
    Gentle augmentations for domain adaptation (PRETRAIN stage).
    Focus on: perspective, optical distortion, lighting, blur, compression.
    No heavy distortions (CoarseDropout, GridDropout, etc.)

    NOTE: For grayscale mode, images should already be converted to grayscale
    in the dataset, so no GrayscaleTransform is applied here.

    For color mode, random grayscale augmentation is applied to improve robustness.
    """
    transforms = [
        # 1. Geometric - simulate camera angles
        A.Perspective(scale=(0.05, 0.15), p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.5
        ),

        # 2. Lighting adjustments
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # 4. Blur and noise - simulate CCTV quality
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        # 6. Compression artifacts - simulate CCTV compression
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

        # 7. Flip
        A.HorizontalFlip(p=0.5),
    ]

    # For color mode: add random grayscale to improve model robustness
    if mode == 'color':
        transforms.append(A.ToGray(p=0.3))

    # Must resize to imgsz
    transforms.insert(0, A.Resize(height=imgsz, width=imgsz, p=1.0))

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=100,
            min_visibility=0.3
        )
    )


def get_finetune_augmentations(imgsz: int = 640):
    """
    Minimal augmentations for FINE-TUNE stage on small dataset (50 images).

    Key principle: Keep augmentations minimal to preserve target domain characteristics.
    Only use flip and very light adjustments to avoid overfitting on augmented data.
    """
    transforms = [
        A.Resize(height=imgsz, width=imgsz, p=1.0),

        # Only horizontal flip - most reliable augmentation
        A.HorizontalFlip(p=0.5),

        # Very light lighting adjustments only
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),

        # Slight blur to simulate different focus
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=100,
            min_visibility=0.3
        )
    )


def create_domain_adaptation_dataset(
    img_dir: str,
    labels_dir: str,
    output_base: str,
    target_domain_start: int = 1207,
    target_domain_train_size: int = 50,
    mode: str = 'pretrain',  # 'pretrain' or 'finetune'
    color_mode: str = 'color',  # 'color' or 'grayscale'
    train_all: bool = False  # Use all images for training (no validation split)
):
    """
    Create dataset for domain adaptation.

    Args:
        mode: 'pretrain' - use all 1270 images (train/val split: 90%/10%)
              'finetune' - use only last 60 images (50 train / 10 val)
        color_mode: 'color' or 'grayscale' - if grayscale, convert images during copy
        train_all: if True, use all images for training (no validation split)
    """
    images = list_images_sorted(img_dir)
    n = len(images)

    if mode == 'pretrain':
        if train_all:
            # Train all: use all images for training, no validation split
            train_files = images
            val_files = []
            print(f'PRETRAIN mode (train_all=True): Total {n} images')
            print(f'  Training: {len(train_files)} images (ALL images)')
            print(f'  Validation: {len(val_files)} images (NO validation split)')
        else:
            # Pretrain: validation = first 10% + last 10 images (never used for training)
            first_10pct_size = int(n * 0.1)
            last_10_indices = set(range(n - 10, n))  # Last 10 images (indices)
            first_10pct_indices = set(range(first_10pct_size))

            val_indices = first_10pct_indices | last_10_indices
            train_files = [f for i, f in enumerate(images) if i not in val_indices]
            val_files = [f for i, f in enumerate(images) if i in val_indices]

            print(f'PRETRAIN mode: Total {n} images')
            print(f'  Validation: {len(val_files)} images (first 10% + last 10)')
            print(f'    - First 10%: indices 0-{first_10pct_size-1}')
            print(f'    - Last 10: indices {n-10}-{n-1}')
            print(f'  Training: {len(train_files)} images')

    elif mode == 'finetune':
        if train_all:
            # Train all: use all last 60 images for training, no validation split
            start_idx = target_domain_start - 1  # Convert to 0-indexed
            target_images = images[start_idx:]  # Last 60 images
            
            train_files = target_images  # All 60 images for training
            val_files = []  # No validation split
            
            print(f'FINETUNE mode (train_all=True): Using last 60 images (domain-similar)')
            print(f'  Training: {len(train_files)} images (ALL last 60 images, indices {target_domain_start}-{target_domain_start + 59})')
            print(f'  Validation: {len(val_files)} images (NO validation split)')
        else:
            # Fine-tune: use only last 60 images
            # Training: images at indices [target_domain_start : target_domain_start+50]
            # Validation: LAST 10 images (same as pretrain validation, never used for training)
            start_idx = target_domain_start - 1  # Convert to 0-indexed
            target_images = images[start_idx:]  # Last 60 images

            train_files = target_images[:target_domain_train_size]  # First 50 of last 60
            val_files = target_images[target_domain_train_size:]    # Last 10 images (global last 10)

            print(f'FINETUNE mode: Using last 60 images (domain-similar)')
            print(f'  Training: {len(train_files)} images (indices {target_domain_start}-{target_domain_start + target_domain_train_size - 1})')
            print(f'  Validation: {len(val_files)} images (LAST 10, indices {target_domain_start + target_domain_train_size}-{target_domain_start + 59})')
            print(f'  Note: Last 10 images are NEVER used for training in any stage')

    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Create directory structure
    train_img_dir = os.path.join(output_base, 'images', 'train')
    train_lbl_dir = os.path.join(output_base, 'labels', 'train')
    val_img_dir = os.path.join(output_base, 'images', 'val')
    val_lbl_dir = os.path.join(output_base, 'labels', 'val')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Copy train files (with grayscale conversion if needed)
    print(f'Copying {len(train_files)} training images and labels...')
    if color_mode == 'grayscale':
        print(f'  Converting to grayscale...')

    for fname in train_files:
        src_path = os.path.join(img_dir, fname)
        dst_path = os.path.join(train_img_dir, fname)

        # Convert to grayscale if needed
        if color_mode == 'grayscale':
            img = cv2.imread(src_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = rgb_to_grayscale_rgb(img_rgb)
            img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, img_gray_bgr)
        else:
            shutil.copy2(src_path, dst_path)

        # Copy labels
        label_fname = os.path.splitext(fname)[0] + '.txt'
        label_src = os.path.join(labels_dir, label_fname)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(train_lbl_dir, label_fname))

    # Copy val files (with grayscale conversion if needed)
    print(f'Copying {len(val_files)} validation images and labels...')
    for fname in val_files:
        src_path = os.path.join(img_dir, fname)
        dst_path = os.path.join(val_img_dir, fname)

        # Convert to grayscale if needed
        if color_mode == 'grayscale':
            img = cv2.imread(src_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = rgb_to_grayscale_rgb(img_rgb)
            img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_path, img_gray_bgr)
        else:
            shutil.copy2(src_path, dst_path)

        # Copy labels
        label_fname = os.path.splitext(fname)[0] + '.txt'
        label_src = os.path.join(labels_dir, label_fname)
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(val_lbl_dir, label_fname))

    return output_base


def write_data_yaml(dataset_path: str, out_yaml: str):
    """Create YOLO data.yaml config."""
    content = f"""# YOLO detection data config (Domain Adaptation)
path: {os.path.abspath(dataset_path)}
train: images/train
val: images/val
names:
  0: pig
nc: 1
""".lstrip()
    with open(out_yaml, 'w') as f:
        f.write(content)



def create_albumentations_callback(albu_transform):
    """Create callback to apply Albumentations to training batches."""

    def apply_albu_to_sample(im, lbl):
        """Apply Albumentations to a single image + labels."""
        # im: CHW torch tensor in [0,1]
        # lbl: (n, 5) tensor [cls, x, y, w, h] in YOLO format (normalized)

        # Convert to numpy uint8
        im_np = (im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Extract bboxes and classes
        if len(lbl) == 0:
            # No annotations
            try:
                t = albu_transform(image=im_np, bboxes=[], class_labels=[])
                im_out = torch.from_numpy(t['image']).permute(2, 0, 1).float() / 255.0
                return im_out, lbl
            except Exception as e:
                print(f'Albumentations failed (no boxes): {e}')
                return im, lbl

        bboxes = lbl[:, 1:].cpu().numpy().tolist()
        classes = lbl[:, 0].cpu().numpy().astype(int).tolist()

        try:
            t = albu_transform(image=im_np, bboxes=bboxes, class_labels=classes)
        except Exception as e:
            print(f'Albumentations failed: {e}')
            return im, lbl

        # Convert back
        im_out = torch.from_numpy(t['image']).permute(2, 0, 1).float() / 255.0

        if len(t['bboxes']) == 0:
            # All boxes cropped out - return original labels to avoid losing targets
            return im_out, lbl

        # Reconstruct labels
        new_lbl = torch.zeros((len(t['bboxes']), 5), dtype=lbl.dtype)
        new_lbl[:, 0] = torch.tensor(t['class_labels'], dtype=lbl.dtype)
        new_lbl[:, 1:] = torch.tensor(t['bboxes'], dtype=lbl.dtype)

        # Clip to valid range [0, 1]
        new_lbl[:, 1:] = torch.clamp(new_lbl[:, 1:], 0, 1)

        return im_out, new_lbl

    def on_preprocess_batch(trainer):
        """Apply Albumentations before training step."""
        if not trainer.model.training:
            return

        # Get batch
        batch = trainer.batch
        imgs = batch['img']  # (B, C, H, W)

        # Convert batch labels to per-image format
        batch_idx = batch.get('batch_idx', None)
        if batch_idx is None:
            return

        cls = batch.get('cls', None)
        bboxes = batch.get('bboxes', None)

        if cls is None or bboxes is None:
            return

        # Group by batch index
        B = imgs.shape[0]
        out_imgs = []
        out_labels = []

        for i in range(B):
            mask = batch_idx == i
            if mask.sum() == 0:
                # No labels for this image
                img_labels = torch.zeros((0, 5), dtype=cls.dtype, device=cls.device)
            else:
                img_cls = cls[mask].unsqueeze(1)
                img_bboxes = bboxes[mask]
                img_labels = torch.cat([img_cls, img_bboxes], dim=1)

            # Apply Albumentations
            im_out, lbl_out = apply_albu_to_sample(imgs[i], img_labels)
            out_imgs.append(im_out)
            out_labels.append(lbl_out)

        # Reconstruct batch
        trainer.batch['img'] = torch.stack(out_imgs, dim=0).to(imgs.device, dtype=imgs.dtype)

        # Reconstruct labels in batch format
        new_batch_idx = []
        new_cls = []
        new_bboxes = []

        for i, lbl in enumerate(out_labels):
            if len(lbl) > 0:
                new_batch_idx.append(torch.full((len(lbl),), i, dtype=torch.long, device=lbl.device))
                new_cls.append(lbl[:, 0])
                new_bboxes.append(lbl[:, 1:])

        if len(new_cls) > 0:
            trainer.batch['batch_idx'] = torch.cat(new_batch_idx)
            trainer.batch['cls'] = torch.cat(new_cls)
            trainer.batch['bboxes'] = torch.cat(new_bboxes)
        else:
            # No labels in batch
            trainer.batch['batch_idx'] = torch.zeros((0,), dtype=torch.long, device=imgs.device)
            trainer.batch['cls'] = torch.zeros((0,), dtype=cls.dtype, device=imgs.device)
            trainer.batch['bboxes'] = torch.zeros((0, 4), dtype=bboxes.dtype, device=imgs.device)

    return on_preprocess_batch

def load_yolo_labels(labels_dir: str, val_images: list) -> dict:
    """Load YOLO format labels and convert to evaluation format."""
    gt_by_image = {}

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        img_id = int(os.path.splitext(img_name)[0])

        # Read corresponding label file
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(label_path):
            continue

        # Read image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Parse YOLO format labels
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    box_w = float(parts[3]) * w
                    box_h = float(parts[4]) * h

                    # Convert to xyxy format
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    bbox = [x1, y1, x2, y2]
                    gt_by_image.setdefault(img_id, []).append({'bbox': bbox, 'class_id': cls})

    return gt_by_image


def evaluate_with_tta(model: YOLO, val_img_dir: str, val_label_dir: str,
                      device: str, imgsz: int, mode: str = 'color', use_tta: bool = True,
                      output_dir: str = None, visualize_limit: int = 5):
    """Evaluate validation set with optional TTA and save visualization images.

    Args:
        model: YOLO model instance
        val_img_dir: Directory of validation images (.jpg)
        val_label_dir: Directory of YOLO-format labels for validation images
        device: Device string
        imgsz: Inference size
        mode: 'color' or 'grayscale'
        use_tta: Whether to enable test-time augmentation
        output_dir: If provided, save per-image predictions to this directory
    """
    from evaluation import compute_map_50_95
    model.to(device)  # Ensure model is on correct device

    # Get validation images
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))
    print(f"Found {len(val_images)} validation images")

    # Load ground truth from YOLO labels
    print(f"Loading ground truth from: {val_label_dir}")
    gt_by_image = load_yolo_labels(val_label_dir, val_images)

    print(f"Loaded {len(gt_by_image)} images with ground truth")

    # Prepare visualization directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Run inference
    print(f"Running inference {'with TTA' if use_tta else 'without TTA'}...")
    preds_by_image = {}

    # Track how many images were visualized
    vis_count = 0

    for img_path in val_images:
        img_name = os.path.basename(img_path)
        img_id = int(os.path.splitext(img_name)[0])

        if mode == 'grayscale':
            # Load and convert to grayscale, save to temp file
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = rgb_to_grayscale_rgb(img_rgb)
            img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img_gray_bgr)

            # Use predict with temp file (supports TTA)
            # Use very low conf to collect all predictions, then filter by confidence_threshold in metrics
            results = model.predict(tmp_path, imgsz=imgsz, augment=use_tta, verbose=False, conf=0.01)

            # Clean up temp file
            os.unlink(tmp_path)

            preds = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    preds.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy())
                    })
            # Prepare original RGB for visualization
            img_vis_rgb = img_gray
        else:
            # Color mode: use file path with TTA support
            # Use very low conf to collect all predictions, then filter by confidence_threshold in metrics
            results = model.predict(img_path, imgsz=imgsz, augment=use_tta, verbose=False, conf=0.01)

            preds = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    preds.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy())
                    })
            # Load original for visualization
            img_bgr = cv2.imread(img_path)
            img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        preds_by_image[img_id] = preds

        # Draw and save visualization for a limited number of images
        if output_dir is not None and vis_count < visualize_limit:
            vis = img_vis_rgb.copy()

            # Draw GT boxes (blue)
            gt_items = gt_by_image.get(img_id, [])
            for gt in gt_items:
                x1, y1, x2, y2 = [int(v) for v in gt['bbox']]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 102, 255), 2)
                cv2.putText(vis, "GT", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 102, 255), 2)

            # Draw predicted boxes (green)
            for p in preds:
                x1, y1, x2, y2 = [int(v) for v in p['bbox']]
                conf = p['confidence']
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"pred {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Legend banner
            w = vis.shape[1]
            cv2.rectangle(vis, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(vis, f"ID:{img_id}  GT=blue  PRED=green  TTA={'on' if use_tta else 'off'}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            out_path = os.path.join(output_dir, f"{img_id}_gt_pred.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            vis_count += 1

    # Compute metrics for multiple confidence thresholds
    print("\nComputing mAP@[.5:.95] for different confidence thresholds...")
    confidence_thresholds = [0.01, 0.1, 0.2]
    metrics_by_conf = {}

    print(f"\n{'Conf':<8} {'mAP@[.5:.95]':<14} {'AP50':<8} {'AP75':<8}")
    print("-" * 40)

    for conf_thresh in confidence_thresholds:
        metrics = compute_map_50_95(
            gt_by_image=gt_by_image,
            preds_by_image=preds_by_image,
            confidence_threshold=conf_thresh
        )
        metrics_by_conf[conf_thresh] = metrics
        print(f"{conf_thresh:<8.2f} {metrics['mAP']:<14.4f} {metrics['AP50']:<8.4f} {metrics['AP75']:<8.4f}")

    # Return metrics at conf=0.01 (highest recall)
    return metrics_by_conf[0.01]


def evaluate_test_set_sample(model: YOLO, test_img_dir: str, device: str,
                              imgsz: int, mode: str = 'color', use_tta: bool = True,
                              sample_early: int = 3, sample_late: int = 2,
                              seed: int = 42, output_dir: str = None):
    """
    Evaluate on test samples: 3 from early images + 2 from last 60 (target domain).

    This allows comparing model performance on different domains:
    - Early images: non-target domain
    - Last 60 images: target domain (similar to training last 60)

    Args:
        model: Trained model
        test_img_dir: Directory containing test images
        device: Device to run inference on
        imgsz: Image size for inference
        mode: 'color' or 'grayscale'
        use_tta: Whether to use test-time augmentation
        sample_early: Number of images to sample from early images (default: 3)
        sample_late: Number of images to sample from last 60 (default: 2)
        seed: Random seed for reproducibility
        output_dir: Directory to save visualizations (default: same as model dir)
    """
    print(f"\n{'='*60}")
    print(f"🎲 TEST SET SAMPLE EVALUATION")
    print(f"   Early (non-target): {sample_early} images")
    print(f"   Last 60 (target): {sample_late} images")
    print(f"{'='*60}")

    # Create output directory for visualizations
    if output_dir is None:
        output_dir = os.path.join('visualizations', 'test_sample')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Visualizations will be saved to: {output_dir}")

    # Get all test images
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Total test images: {len(test_images)}")

    # Split into early and last 60
    early_images = test_images[:-60]  # All except last 60
    last_60_images = test_images[-60:]  # Last 60 images (target domain)

    print(f"\nEarly images (non-target domain): {len(early_images)} images")
    print(f"Last 60 images (target domain): {len(last_60_images)} images")

    # Random sample from both groups
    np.random.seed(seed)
    sampled_early = np.random.choice(early_images, size=min(sample_early, len(early_images)), replace=False)
    sampled_late = np.random.choice(last_60_images, size=min(sample_late, len(last_60_images)), replace=False)

    sampled_images = list(sampled_early) + list(sampled_late)

    print(f"\nSampled from early: {sorted([int(os.path.splitext(os.path.basename(p))[0]) for p in sampled_early])}")
    print(f"Sampled from last 60: {sorted([int(os.path.splitext(os.path.basename(p))[0]) for p in sampled_late])}")
    print(f"Total sampled: {len(sampled_images)} images (seed={seed})")

    # Load model
    model.to(device)

    # Run inference on sampled images
    print(f"Running inference {'with TTA' if use_tta else 'without TTA'}...")

    all_predictions = []
    for img_path in sampled_images:
        img_name = os.path.basename(img_path)
        img_id = int(os.path.splitext(img_name)[0])

        # Determine domain
        is_target_domain = img_path in sampled_late
        domain_label = "TARGET" if is_target_domain else "NON-TARGET"

        # Load original image for visualization
        img_orig = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        if mode == 'grayscale':
            # Convert to grayscale and save to temp file
            img_gray = rgb_to_grayscale_rgb(img_rgb)
            img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, img_gray_bgr)

            # Use predict with temp file (supports TTA)
            results = model.predict(tmp_path, imgsz=imgsz, augment=use_tta, verbose=False, conf=0.01)

            # Clean up temp file
            os.unlink(tmp_path)

            preds = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    preds.append({
                        'image_id': img_id,
                        'bbox': bbox,
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy())
                    })
            img_vis = img_gray.copy()
        else:
            # Color mode: use file path with TTA support
            results = model.predict(img_path, imgsz=imgsz, augment=use_tta, verbose=False, conf=0.01)

            preds = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    preds.append({
                        'image_id': img_id,
                        'bbox': bbox,
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy())
                    })

        # Visualize predictions on original image
            img_vis = img_rgb.copy()

        # Add domain label at top of image
        w = img_vis.shape[1]
        cv2.rectangle(img_vis, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img_vis, f"ID:{img_id} | {domain_label}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for pred in preds:
            x1, y1, x2, y2 = [int(v) for v in pred['bbox']]
            conf = pred['confidence']

            # Draw bounding box (green)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence label
            label = f"pig {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_vis, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img_vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Save visualization with domain prefix
        domain_prefix = "target" if is_target_domain else "nontarget"
        output_path = os.path.join(output_dir, f"{domain_prefix}_{img_id}_pred.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))

        all_predictions.extend(preds)
        print(f"  Image {img_id} ({domain_label}): {len(preds)} detections → {output_path}")

    print(f"\nTotal predictions: {len(all_predictions)}")
    print(f"✅ Test set sample evaluation completed!")
    print(f"📁 Visualizations saved to: {output_dir}")

    return all_predictions


def predict_test_set_to_csv(model: YOLO, test_img_dir: str, device: str, imgsz: int,
                            mode: str = 'color', use_tta: bool = True,
                            output_csv: str = 'predictions.csv',
                            vis_output_dir: str = None, conf: float = 0.01,
                            batch_size: int = 8):
    """Run inference on all test images, optionally save visualizations, and write CSV.

    CSV format matches sample_submission.csv:
    header: Image_ID,PredictionString
    PredictionString for each image is a space-separated sequence of detections:
    "score x y w h class_id" repeated for each detection.

    Args:
        batch_size: Number of images to process in parallel (default: 8)
    """
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    if vis_output_dir is not None:
        os.makedirs(vis_output_dir, exist_ok=True)

    model.to(device)

    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"Test images found: {len(test_images)}")
    print(f"Using batch size: {batch_size}")

    rows = [("Image_ID", "PredictionString")]

    # Process in batches for speed
    total_batches = (len(test_images) + batch_size - 1) // batch_size
    print(f"\n{'='*60}")
    print(f"Starting batch prediction: {total_batches} batches")
    print(f"{'='*60}")

    import time
    start_time = time.time()

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(test_images))
        batch_paths = test_images[start_idx:end_idx]

        elapsed = time.time() - start_time
        avg_time = elapsed / (batch_idx + 1) if batch_idx > 0 else 0
        eta = avg_time * (total_batches - batch_idx - 1) if batch_idx > 0 else 0

        print(f"[{batch_idx+1:>3}/{total_batches}] Processing {len(batch_paths):>2} images | Elapsed: {elapsed:>6.1f}s | ETA: {eta:>6.1f}s", end='\r' if batch_idx < total_batches - 1 else '\n')

        # Prepare batch based on color mode
        if mode == 'grayscale':
            # For grayscale, need to convert and use temp files
            batch_results = []
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_gray = rgb_to_grayscale_rgb(img_rgb)
                img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_RGB2BGR)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, img_gray_bgr)
                results = model.predict(tmp_path, imgsz=imgsz, augment=use_tta, verbose=False, conf=conf)
                batch_results.append((results, img_path))
                os.unlink(tmp_path)
        else:
            # For color mode, can use batch prediction directly
            batch_results = model.predict(batch_paths, imgsz=imgsz, augment=use_tta, verbose=False, conf=conf)

        # Process results for each image in batch
        for idx, img_path in enumerate(batch_paths):
            img_name = os.path.basename(img_path)
            img_id = int(os.path.splitext(img_name)[0])

            # Get result for this image
            if mode == 'grayscale':
                results, _ = batch_results[idx]
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_gray = rgb_to_grayscale_rgb(img_rgb)
                img_vis_rgb = img_gray
            else:
                results = batch_results[idx]
                img_bgr = cv2.imread(img_path)
                img_vis_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Build prediction string
            dets = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    w = x2 - x1
                    h = y2 - y1
                    score = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    dets.extend([f"{score:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}", str(cls_id)])

            prediction_string = " ".join(dets)
            rows.append((str(img_id), prediction_string))

            # Optional visualization
            if vis_output_dir is not None:
                vis = img_vis_rgb.copy()
                for i in range(0, len(dets), 6):
                    # dets chunk: score, x, y, w, h, cls
                    score = float(dets[i])
                    x = int(float(dets[i + 1]))
                    y = int(float(dets[i + 2]))
                    w = int(float(dets[i + 3]))
                    h = int(float(dets[i + 4]))
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"{score:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(vis, (x, y - label_size[1] - 5), (x + label_size[0], y), (0, 255, 0), -1)
                    cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                out_path = os.path.join(vis_output_dir, f"{img_id}_pred.jpg")
                cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Write CSV
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Writing CSV...")
    import csv
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    print(f"✅ Test predictions CSV written to: {output_csv}")
    print(f"Total images: {len(test_images)}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(test_images):.2f}s per image)")
    print(f"Speed improvement: ~{8.0 * batch_size / (batch_size + 7.0):.1f}x faster than sequential")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Domain Adaptation Training for Pig Detection',
        epilog="""
Examples:
  # Color mode (two-stage)
  python train_domain_adaptation.py --color_mode color --device cuda

  # Grayscale mode (two-stage)
  python train_domain_adaptation.py --color_mode grayscale --device cuda

  # Skip pretrain (only fine-tune on last 60)
  python train_domain_adaptation.py --skip_pretrain --color_mode color --device cuda

  # Train with all images (no validation split)
  python train_domain_adaptation.py --train_all --color_mode color --device cuda

  # Evaluation only with TTA
  python train_domain_adaptation.py --eval_only --checkpoint runs/domain_adapt_color/yolov10x_finetune/weights/best.pt --use_tta
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--img_dir', default='train/img', help='Directory of training images')
    parser.add_argument('--labels_dir', default='train/labels', help='Directory of label files (YOLO format)')
    parser.add_argument('--test_dir', default='test/img', help='Test images directory')

    # Training params
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Epochs for pretraining stage')
    parser.add_argument('--finetune_epochs', type=int, default=5, help='Epochs for fine-tuning stage (use 5-10, with early stopping)')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--model', default='yolov11x', help='YOLO model variant')

    # Domain adaptation params
    parser.add_argument('--color_mode', choices=['color', 'grayscale'], default='color',
                        help='Training mode: color or grayscale (RGB format)')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip pretraining, only fine-tune on last 60 images')
    parser.add_argument('--train_all', action='store_true',
                        help='Use all train and val images for training (no validation split)')
    parser.add_argument('--target_domain_start', type=int, default=1207,
                        help='Starting index of target domain images (1-indexed)')
    parser.add_argument('--target_train_size', type=int, default=50,
                        help='Number of target domain images for training')

    # Evaluation params
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint for evaluation')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation for evaluation')
    parser.add_argument('--eval_test_sample', action='store_true', help='Evaluate on test set sample (3 early + 2 from last 60)')
    parser.add_argument('--test_sample_early', type=int, default=10, help='Number of early test images to sample')
    parser.add_argument('--test_sample_late', type=int, default=5, help='Number of late (last 60) test images to sample')
    parser.add_argument('--test_sample_seed', type=int, default=42, help='Random seed for test sampling')
    # Test CSV export
    parser.add_argument('--export_test_csv', action='store_true', help='Run full test inference and export CSV')
    parser.add_argument('--test_csv_path', default=None, help='Output CSV path (default: runs/.../predictions.csv)')
    parser.add_argument('--test_vis', action='store_true', help='Also save test visualizations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for test set prediction (default: 8, larger=faster but more VRAM)')
    parser.add_argument('--name', default=None, help='Name of the run')

    args = parser.parse_args()

    # Set default name if not provided
    if args.name is None:
        if args.eval_only:
            args.name = 'eval'
        else:
            args.name = 'train'

    # Output directory
    out_dir = f'runs/domain_adapt_{args.color_mode}/{args.name}'
    os.makedirs(out_dir, exist_ok=True)

    # Model mapping
    vmap = {
        'yolov10n': 'yolov10n.yaml',
        'yolov10s': 'yolov10s.yaml',
        'yolov10m': 'yolov10m.yaml',
        'yolov10l': 'yolov10l.yaml',
        'yolov10x': 'yolov10x.yaml',
        'yolov11n': 'yolo11n.pt',  # YOLO11 uses pretrained weights
        'yolov11s': 'yolo11s.pt',
        'yolov11m': 'yolo11m.pt',
        'yolov11l': 'yolo11l.pt',
        'yolov11x': 'yolo11x.pt',
    }
    if args.model not in vmap:
        raise SystemExit(f'Unsupported model: {args.model}')

    cfg_yaml = vmap[args.model]

    if args.checkpoint:
        model = YOLO(args.checkpoint)
    else:
        model = YOLO(cfg_yaml)

    # ==================== EVALUATION ONLY ====================
    if args.eval_only:
        if not args.checkpoint:
            print("Error: --checkpoint is required for --eval_only")
            return

        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            return

        print(f"\n{'='*60}")
        print(f"🔍 EVALUATION MODE ({args.color_mode.upper()})")
        print(f"{'='*60}")

        # Create fine-tune dataset for validation
        dataset_dir_ft = f'yolo_dataset_finetune_{args.color_mode}'
        if not os.path.exists(dataset_dir_ft):
            create_domain_adaptation_dataset(
                args.img_dir, args.labels_dir, dataset_dir_ft,
                target_domain_start=args.target_domain_start,
                target_domain_train_size=args.target_train_size,
                mode='finetune',
                color_mode=args.color_mode,
                train_all=args.train_all
            )
        else:
            print(f"Using existing dataset: {dataset_dir_ft}")

        val_img_dir = os.path.join(dataset_dir_ft, 'images', 'val')
        val_label_dir = os.path.join(dataset_dir_ft, 'labels', 'val')

        # Evaluate WITHOUT TTA
        print("\n1. Validation Evaluation WITHOUT TTA:")
        metrics_no_tta = evaluate_with_tta(
            model=model,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            device=args.device,
            imgsz=args.imgsz,
            mode=args.color_mode,
            use_tta=False,
            output_dir=os.path.join(out_dir, 'eval_val_visualizations_no_tta'),
            visualize_limit=5
        )

        # Evaluate WITH TTA
        metrics_with_tta = None
        if args.model.startswith('yolov11'):
            print("\n2. Validation Evaluation WITH TTA:")
            metrics_with_tta = evaluate_with_tta(
                model=model,
                val_img_dir=val_img_dir,
                val_label_dir=val_label_dir,
                device=args.device,
                imgsz=args.imgsz,
                mode=args.color_mode,
                use_tta=True,
                output_dir=os.path.join(out_dir, 'eval_val_visualizations_with_tta'),
                visualize_limit=5
            )
        else:
            print(f"\n⚠️  TTA skipped: {args.model} does not support TTA")

        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ VALIDATION EVALUATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Mode: {args.color_mode.upper()}")
        print(f"Model: {args.checkpoint}")
        print(f"\nValidation Results (at conf=0.01):")
        print(f"  Without TTA - mAP@[.5:.95]: {metrics_no_tta['mAP']:.4f} | AP50: {metrics_no_tta['AP50']:.4f} | AP75: {metrics_no_tta['AP75']:.4f}")
        if metrics_with_tta:
            improvement = (metrics_with_tta['mAP'] - metrics_no_tta['mAP']) / metrics_no_tta['mAP'] * 100
            print(f"  With TTA    - mAP@[.5:.95]: {metrics_with_tta['mAP']:.4f} | AP50: {metrics_with_tta['AP50']:.4f} | AP75: {metrics_with_tta['AP75']:.4f}")
            print(f"  TTA Improvement: {improvement:+.2f}%")
        print(f"\nNote: See detailed metrics at different confidence thresholds (0.01, 0.1, 0.2) above.")
        print(f"{'='*60}")

        # Test set sample evaluation
        if args.eval_test_sample:
            print(f"\n{'='*60}")
            print("🎲 TEST SET SAMPLE EVALUATION")
            print(f"{'='*60}")

            # Evaluate WITHOUT TTA
            print("\n1. Test Sample WITHOUT TTA:")
            evaluate_test_set_sample(
                model=model,
                test_img_dir=args.test_dir,
                device=args.device,
                imgsz=args.imgsz,
                mode=args.color_mode,
                use_tta=False,
                sample_early=args.test_sample_early,
                sample_late=args.test_sample_late,
                seed=args.test_sample_seed,
                output_dir=os.path.join(out_dir, 'eval_test_sample_no_tta')
            )

            # Evaluate WITH TTA
            if args.model.startswith('yolov11'):
                print("\n2. Test Sample WITH TTA:")
                evaluate_test_set_sample(
                    model=model,
                    test_img_dir=args.test_dir,
                    device=args.device,
                    imgsz=args.imgsz,
                    mode=args.color_mode,
                    use_tta=True,
                    sample_early=args.test_sample_early,
                    sample_late=args.test_sample_late,
                    seed=args.test_sample_seed,
                    output_dir=os.path.join(out_dir, 'eval_test_sample_with_tta')
                )
            else:
                print(f"\n⚠️  TTA skipped for test samples: {args.model} does not support TTA")

        # Full test CSV export (use TTA if supported for best results)
        if args.export_test_csv:
            print(f"\n{'='*60}")
            print("📄 EXPORTING TEST PREDICTIONS TO CSV")
            print(f"{'='*60}")

            # Determine whether to use TTA
            use_tta_for_csv = args.model.startswith('yolov11')
            if use_tta_for_csv:
                print("Using TTA for final predictions (best performance)")
            else:
                print(f"TTA not available for {args.model}, using standard inference")

            csv_path = args.test_csv_path or os.path.join(out_dir, 'predictions.csv')
            vis_dir = os.path.join(out_dir, 'eval_test_visualizations') if args.test_vis else None
            predict_test_set_to_csv(
                model=model,
                test_img_dir=args.test_dir,
                device=args.device,
                imgsz=args.imgsz,
                mode=args.color_mode,
                use_tta=use_tta_for_csv,
                output_csv=csv_path,
                vis_output_dir=vis_dir,
                conf=0.01,
                batch_size=args.batch_size
            )

        return

    # ==================== TRAINING ====================
    print(f"\n{'='*60}")
    print(f"🚀 DOMAIN ADAPTATION TRAINING ({args.color_mode.upper()})")
    print(f"{'='*60}")
    print(f"Strategy: {'Pretrain → Fine-tune' if not args.skip_pretrain else 'Fine-tune only'}")
    if args.train_all:
        print(f"Mode: TRAIN ALL (no validation split)")
        print(f"Target domain: All last 60 images for training")
    else:
        print(f"Target domain: Last 60 images ({args.target_train_size} train / {60 - args.target_train_size} val)")
    print(f"{'='*60}\n")

    pretrain_model_path = None

    # ==================== STAGE 1: PRETRAIN ====================
    if not args.skip_pretrain:
        print(f"\n{'='*60}")
        print("STAGE 1: PRETRAIN ON ALL DATA")
        print(f"{'='*60}")

        # Create pretrain dataset
        dataset_dir_pretrain = f'yolo_dataset_pretrain_{args.color_mode}'
        if not os.path.exists(dataset_dir_pretrain):
            create_domain_adaptation_dataset(
                args.img_dir, args.labels_dir, dataset_dir_pretrain, mode='pretrain',
                color_mode=args.color_mode,
                train_all=args.train_all
            )

        data_yaml_pretrain = os.path.join('data_splits', f'data_pretrain_{args.color_mode}.yaml')
        os.makedirs('data_splits', exist_ok=True)
        write_data_yaml(dataset_dir_pretrain, data_yaml_pretrain)

        # Create augmentation pipeline
        albu_transform = get_domain_augmentations(imgsz=args.imgsz, mode=args.color_mode)

        # Train pretrain stage
        # Note: YOLOv11 models start from COCO pretrained weights, YOLOv10 trains from scratch
        if args.model.startswith('yolov11'):
            print(f"Training {args.model} from COCO pretrained weights (pretrain stage)...")
        else:
            print(f"Training {args.model} from scratch (pretrain stage)...")

        # Add Albumentations callback
        albu_callback = create_albumentations_callback(albu_transform)
        model.add_callback('on_preprocess_batch', albu_callback)


        print(f"✅ Callbacks registered: Albumentations + Domain Evaluation")

        model.train(
            data=data_yaml_pretrain,
            epochs=args.pretrain_epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=4,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            label_smoothing=0.05,
            project=out_dir,
            name=f'{args.model}_pretrain',
            exist_ok=True,
            val=not args.train_all,  # Disable validation when train_all=True
            plots=True,
            save=True,
            save_period=10,
        )

        pretrain_model_path = os.path.join(out_dir, f'{args.model}_pretrain', 'weights', 'best.pt')
        print(f"✅ Pretrain completed: {pretrain_model_path}")

        # Post-pretrain evaluation (skip if train_all=True)
        if not args.train_all:
            print(f"\n{'='*60}")
            print("📊 POST-PRETRAIN EVALUATION")
            print(f"{'='*60}")

            val_img_dir_pretrain = os.path.join(dataset_dir_pretrain, 'images', 'val')
            val_label_dir_pretrain = os.path.join(dataset_dir_pretrain, 'labels', 'val')

            print("\nEvaluation WITHOUT TTA:")
            metrics_pretrain_no_tta = evaluate_with_tta(
                model=model,
                val_img_dir=val_img_dir_pretrain,
                val_label_dir=val_label_dir_pretrain,
                device=args.device,
                imgsz=args.imgsz,
                mode=args.color_mode,
                use_tta=False,
                output_dir=os.path.join(out_dir, 'pretrain_val_visualizations_no_tta'),
                visualize_limit=5
            )

            # Evaluate WITH TTA (only for YOLOv11)
            if args.model.startswith('yolov11'):
                print("\nEvaluation WITH TTA:")
                metrics_pretrain_with_tta = evaluate_with_tta(
                    model=model,
                    val_img_dir=val_img_dir_pretrain,
                    val_label_dir=val_label_dir_pretrain,
                    device=args.device,
                    imgsz=args.imgsz,
                    mode=args.color_mode,
                    use_tta=True,
                    output_dir=os.path.join(out_dir, 'pretrain_val_visualizations_with_tta'),
                    visualize_limit=5
                )
                improvement = (metrics_pretrain_with_tta['mAP'] - metrics_pretrain_no_tta['mAP']) / metrics_pretrain_no_tta['mAP'] * 100
                print(f"\nPretrain Results Summary (at conf=0.01):")
                print(f"  Without TTA: mAP@[.5:.95]={metrics_pretrain_no_tta['mAP']:.4f}")
                print(f"  With TTA:    mAP@[.5:.95]={metrics_pretrain_with_tta['mAP']:.4f} ({improvement:+.2f}%)")
            else:
                print(f"\n⚠️  TTA skipped: {args.model} does not support TTA")
                print(f"Pretrain Results Summary (at conf=0.01):")
                print(f"  mAP@[.5:.95]: {metrics_pretrain_no_tta['mAP']:.4f}")
        else:
            print(f"\n⚠️  Post-pretrain evaluation skipped: train_all=True (no validation set)")

    # ==================== STAGE 2: FINE-TUNE ====================
    print(f"\n{'='*60}")
    print("STAGE 2: FINE-TUNE ON TARGET DOMAIN (LAST 60 IMAGES)")
    print(f"{'='*60}")

    # Create fine-tune dataset
    dataset_dir_finetune = f'yolo_dataset_finetune_{args.color_mode}'
    if not os.path.exists(dataset_dir_finetune):
        create_domain_adaptation_dataset(
            args.img_dir, args.labels_dir, dataset_dir_finetune,
            target_domain_start=args.target_domain_start,
            target_domain_train_size=args.target_train_size,
            mode='finetune',
            color_mode=args.color_mode,
            train_all=args.train_all
        )

    data_yaml_finetune = os.path.join('data_splits', f'data_finetune_{args.color_mode}.yaml')
    write_data_yaml(dataset_dir_finetune, data_yaml_finetune)

    # Create augmentation pipeline (MINIMAL for fine-tune on small dataset)
    albu_transform = get_finetune_augmentations(imgsz=args.imgsz)

    # Load pretrained model or start from scratch
    if pretrain_model_path and os.path.exists(pretrain_model_path):
        print(f"Loading pretrained model: {pretrain_model_path}")
        model = YOLO(pretrain_model_path)

    # Add Albumentations callback
    callback = create_albumentations_callback(albu_transform)
    model.add_callback('on_preprocess_batch', callback)

    # Fine-tune with MUCH lower learning rate and fewer epochs
    # Key: avoid overfitting on small dataset (50 images)
    model.train(
        data=data_yaml_finetune,
        epochs=args.finetune_epochs,
        imgsz=args.imgsz,
        batch=args.batch,  # Keep same batch size for stability
        device=args.device,
        workers=4,
        optimizer='AdamW',
        lr0=0.0001,  # Very low LR to preserve pretrained features
        lrf=0.00001,  # Aggressive decay to avoid overfitting
        label_smoothing=0.05,
        project=out_dir,
        name=f'{args.model}_finetune',
        exist_ok=True,
        val=not args.train_all,  # Disable validation when train_all=True
        plots=True,
        save=True,
        save_period=5,
        patience=10,  # Early stopping with patience
    )

    finetune_model_path = os.path.join(out_dir, f'{args.model}_finetune', 'weights', 'best.pt')
    print(f"✅ Fine-tune completed: {finetune_model_path}")

    # ==================== POST-TRAINING EVALUATION ====================
    if not args.train_all:
        print(f"\n{'='*60}")
        print("📊 POST-TRAINING EVALUATION")
        print(f"{'='*60}")

        val_img_dir = os.path.join(dataset_dir_finetune, 'images', 'val')
        val_label_dir = os.path.join(dataset_dir_finetune, 'labels', 'val')

        model = YOLO(finetune_model_path)

        # Evaluate without TTA
        print("\n1. Evaluation WITHOUT TTA:")
        metrics_no_tta = evaluate_with_tta(
            model=model,
            val_img_dir=val_img_dir,
            val_label_dir=val_label_dir,
            device=args.device,
            imgsz=args.imgsz,
            mode=args.color_mode,
            use_tta=False,
            output_dir=os.path.join(out_dir, 'train_val_visualizations_no_tta'),
            visualize_limit=5
        )

        # Evaluate WITH TTA (only for YOLOv11, as YOLOv10 doesn't support TTA)
        metrics_with_tta = None
        if args.model.startswith('yolov11'):
            print("\n2. Evaluation WITH TTA:")
            metrics_with_tta = evaluate_with_tta(
                model=model,
                val_img_dir=val_img_dir,
                val_label_dir=val_label_dir,
                device=args.device,
                imgsz=args.imgsz,
                mode=args.color_mode,
                use_tta=True,
                output_dir=os.path.join(out_dir, 'train_val_visualizations_with_tta'),
                visualize_limit=5
            )
        else:
            print(f"\n⚠️  TTA skipped: {args.model} does not support TTA")

        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Mode: {args.color_mode.upper()}")
        print(f"Final model: {finetune_model_path}")
        print(f"\nValidation Results (at conf=0.01):")
        print(f"  Without TTA - mAP@[.5:.95]: {metrics_no_tta['mAP']:.4f} | AP50: {metrics_no_tta['AP50']:.4f} | AP75: {metrics_no_tta['AP75']:.4f}")
        if metrics_with_tta:
            improvement = (metrics_with_tta['mAP'] - metrics_no_tta['mAP']) / metrics_no_tta['mAP'] * 100
            print(f"  With TTA    - mAP@[.5:.95]: {metrics_with_tta['mAP']:.4f} | AP50: {metrics_with_tta['AP50']:.4f} | AP75: {metrics_with_tta['AP75']:.4f}")
            print(f"  TTA Improvement: {improvement:+.2f}%")
        print(f"\nNote: See detailed metrics at different confidence thresholds (0.01, 0.1, 0.2) above.")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETED!")
        print(f"{'='*60}")
        print(f"Mode: {args.color_mode.upper()} (TRAIN ALL)")
        print(f"Final model: {finetune_model_path}")
        print(f"\n⚠️  No validation evaluation: train_all=True (no validation set)")
        print(f"{'='*60}")

    # Test set sample evaluation
    if args.eval_test_sample:
        evaluate_test_set_sample(
            model=model,
            test_img_dir=args.test_dir,
            device=args.device,
            imgsz=args.imgsz,
            mode=args.color_mode,
            use_tta=True,  # Always use TTA for final test evaluation
            sample_early=args.test_sample_early,
            sample_late=args.test_sample_late,
            seed=args.test_sample_seed,
            output_dir=os.path.join(out_dir, 'train_test_sample_visualizations')
        )


if __name__ == '__main__':
    main()
