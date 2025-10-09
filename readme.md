# Pig Detection Contest

This repository contains the code for training and evaluating object detection models for pig detection in both grayscale and color modes, with ensemble prediction capabilities.

## Environment Setup

### Prerequisites
- Python 3.11
- CUDA-compatible GPU (recommended)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Dependencies
The following packages are required (see `requirements.txt`):
- `ultralytics` - YOLO model framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `ensemble-boxes` - Weighted box fusion
- `opencv-python` - Image processing
- `albumentations` - Data augmentation
- `torch` - PyTorch framework

## Training

### 1. Grayscale Mode Training

Train a model on grayscale images:

```bash
bash train_grayscale.sh
```

### 2. Color Mode Training

Train a model on color images:

```bash
bash train_color.sh
```

### Training Parameters
- `--color_mode`: Training mode (grayscale/color)
- `--pretrain_epochs`: Number of pretraining epochs
- `--batch_size`: Batch size for training
- `--device`: GPU device ID (0, 1, etc.)
- `--model`: Model architecture (yolov11x)
- `--train_all`: Train all images

## Evaluation and Submission

### Generate Predictions

Generate predictions with Test Time Augmentation:

```bash
# Generate predictions if using grayscale mode training
bash eval_grayscale.sh

# Generate predictions if using color mode training
bash eval_color.sh
```

These scripts will create:
- `50_grayscale.csv` - Grayscale model predictions
- `50_color.csv` - Color model predictions


## Ensemble Predictions

Combine predictions from multiple models using Weighted Box Fusion (WBF):

### Basic Usage
```bash
python ensemble_predictions.py grayscale_predictions.csv color_predictions.csv
```

### Advanced Usage
```bash
python ensemble_predictions.py \
    grayscale_predictions.csv \
    color_predictions.csv \
    --output ensemble_predictions.csv \
    --iou-threshold 0.8 \
    --skip-threshold 0.0001
```

### Parameters
- `--output`: Output CSV file path (default: ensemble_predictions.csv)
- `--iou-threshold`: IoU threshold for box fusion (default: 0.8)
- `--skip-threshold`: Skip boxes with confidence below this threshold (default: 0.0001)

### Input Format
Input CSV files should have the format:
```csv
Image_ID,PredictionString
1,0.868596 219.66 132.46 128.57 153.69 0 0.891083 181.54 63.20 122.32 74.12 0
2,0.864339 179.78 62.17 125.6 74.7 0
```

Where each PredictionString contains space-separated values: `score x y w h class_id` for each detection.

### Create Final Submission

After generating individual predictions, create ensemble predictions:

```bash
# Create ensemble predictions
python ensemble_predictions.py 50_grayscale.csv 50_color.csv --output submission.csv
```

This will create `submission.csv` - the final submission file.

## File Structure

```
pigpigcontest/
├── train_domain_adaptation.py    # Main training and prediction script
├── ensemble_predictions.py       # Ensemble prediction script
├── requirements.txt              # Python dependencies
├── readme.md                     # This file
├── runs/                        # Training outputs
│   ├── domain_adapt_grayscale/
│   └── domain_adapt_color/
├── yolo_dataset_pretrain_grayscale/  # Grayscale dataset
├── yolo_dataset_pretrain_color/      # Color dataset
└── test_images/                  # Test images
```