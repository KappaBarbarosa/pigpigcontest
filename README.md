# TAICA CVPDL 2025 Homework 1 - Object Detection Project

This project implements object detection models for computer vision tasks using PyTorch and YOLOv10.

## Project Structure

```
taica-cvpdl-2025-hw-1/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py     # Data loading utilities
│   ├── model.py           # Model definitions
│   ├── train.py           # Training utilities
│   └── inference.py       # Inference utilities
├── train/                 # Training data
│   ├── img/              # Training images
│   └── gt.txt            # Ground truth annotations
├── test/                 # Test data
│   └── img/              # Test images
├── yolov10/              # YOLOv10 implementation
├── main.py               # Main execution script
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

## Installation

This project uses `uv` for dependency management. To set up the environment:

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage

### Training

Train a custom model:
```bash
python main.py --mode train --model_type custom --epochs 20 --batch_size 16
```

### Inference

Run inference on test images:
```bash
python main.py --mode inference --model_type yolov10 --confidence_threshold 0.5
```

### Available Options

- `--mode`: Choose between 'train' or 'inference'
- `--model_type`: Choose between 'yolov10' or 'custom'
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--device`: Device to use ('cpu', 'cuda', or 'auto')
- `--confidence_threshold`: Confidence threshold for inference (default: 0.5)
- `--output_file`: Output file for predictions (default: 'predictions.csv')

## Data Format

### Training Data
- Images: JPG format in `train/img/`
- Annotations: Text file `train/gt.txt` with format:
  ```
  image_id,x,y,width,height
  ```
  Where:
  - `image_id`: Image identifier
  - `x, y`: Top-left corner coordinates of bounding box
  - `width, height`: Width and height of bounding box

### Test Data
- Images: JPG format in `test/img/`

### Output Format
Predictions are saved in CSV format with columns:
- `Image_ID`: Image identifier
- `PredictionString`: Space-separated predictions in format:
  ```
  confidence x1 y1 x2 y2 class_id
  ```
  Where:
  - `confidence`: Detection confidence score (0-1)
  - `x1, y1, x2, y2`: Bounding box coordinates (top-left and bottom-right)
  - `class_id`: Object class identifier

## Evaluation Metrics

The project uses standard object detection evaluation metrics:

### Key Metrics
- **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth bounding boxes
- **Precision**: Ratio of correct predictions to total predictions
- **Recall**: Ratio of correct predictions to total ground truth objects
- **F1 Score**: Harmonic mean of precision and recall
- **mAP (Mean Average Precision)**: Average precision across all images and confidence thresholds

### Evaluation Script
```bash
# Evaluate predictions against ground truth
uv run python evaluate.py --pred_file predictions.csv --gt_file train/gt.txt

# With custom thresholds
uv run python evaluate.py --pred_file predictions.csv --iou_threshold 0.5 --confidence_threshold 0.3
```

## Dependencies

- PyTorch 2.8.0
- OpenCV 4.12.0
- Matplotlib 3.10.6
- Pandas 2.3.3
- Scikit-learn 1.7.2
- Jupyter 1.1.1
- And more (see pyproject.toml)

## YOLOv10 Integration

This project includes YOLOv10 implementation in the `yolov10/` directory. You can use it directly for training and inference:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo10n.pt')

# Train
model.train(data='your_dataset.yaml', epochs=100)

# Inference
results = model('path/to/image.jpg')
```

## Examples

### Basic Training
```bash
# Train custom model for 20 epochs
python main.py --mode train --model_type custom --epochs 20 --batch_size 16 --learning_rate 0.001
```

### Inference with Different Confidence Thresholds
```bash
# High confidence predictions
python main.py --mode inference --confidence_threshold 0.8 --output_file high_conf_predictions.csv

# Low confidence predictions
python main.py --mode inference --confidence_threshold 0.3 --output_file low_conf_predictions.csv
```

### Using GPU
```bash
# Force GPU usage
python main.py --mode train --device cuda --model_type custom
```

## Development

To add new features or modify the code:

1. Make changes in the `src/` directory
2. Test your changes:
```bash
python main.py --mode inference --model_type custom
```
3. Run linting:
```bash
uv run ruff check src/
```

## License

This project is part of TAICA CVPDL 2025 coursework.
