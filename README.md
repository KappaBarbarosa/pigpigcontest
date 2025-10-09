# Pig Detection Contest - Domain Adaptation Project

This project implements a domain adaptation approach for pig detection using YOLO models, specifically designed for a computer vision contest. The system uses a two-stage training strategy to adapt models from a general domain to a target domain (last 60 images).

## 🎯 Project Overview

This is a computer vision project focused on pig detection using domain adaptation techniques. The project implements a sophisticated training pipeline that:

- Uses YOLO models (YOLOv10/YOLOv11) for object detection
- Implements domain adaptation with two-stage training
- Supports both color and grayscale image processing
- Includes test-time augmentation (TTA) for improved performance
- Provides comprehensive evaluation metrics

## 🏗️ Project Structure

```
pigpigcontest/
├── train/                          # Training data
│   ├── img/                       # Training images
│   ├── labels/                     # YOLO format labels
│   └── gt.txt                     # Ground truth annotations
├── test/                          # Test data
│   └── img/                       # Test images
├── data_splits/                   # Dataset configuration files
├── yolo_dataset_*/               # Generated datasets for different modes
├── runs/                         # Training outputs and model checkpoints
├── evaluation.py                 # Evaluation metrics implementation
├── train_domain_adaptation.py    # Main training script
├── eval.sh                       # Evaluation script
├── finetune.sh                   # Fine-tuning script
├── requirements.txt              # Python dependencies
└── pyproject.toml               # Project configuration
```

## 🚀 Key Features

### 1. Domain Adaptation Strategy
- **Two-stage training**: Pretrain on all data → Fine-tune on target domain
- **Target domain**: Last 60 images (similar to test domain)
- **Training split**: 50 train + 10 validation from last 60 images

### 2. Multi-Modal Support
- **Color mode**: Full RGB image processing
- **Grayscale mode**: RGB format with R=G=B channels
- **Automatic conversion**: Built-in grayscale transformation

### 3. Advanced Augmentation
- **Pretrain stage**: Comprehensive augmentations (perspective, lighting, blur, compression)
- **Fine-tune stage**: Minimal augmentations to preserve target domain characteristics
- **Test-time augmentation**: Multiple inference passes for improved accuracy

### 4. Model Support
- **YOLOv10**: YOLOv10n, YOLOv10s, YOLOv10m, YOLOv10l, YOLOv10x
- **YOLOv11**: YOLOv11n, YOLOv11s, YOLOv11m, YOLOv11l, YOLOv11x
- **TTA support**: Available for YOLOv11 models

## 📋 Requirements

### System Requirements
- Python >= 3.11
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for training

### Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.8.0`
- `torchvision>=0.23.0`
- `ultralytics>=8.3.205`
- `opencv-python>=4.12.0.88`
- `albumentations`
- `numpy`, `pandas`, `matplotlib`

## 🎮 Usage

### 1. Training (Two-Stage Domain Adaptation)

#### Color Mode Training
```bash
python train_domain_adaptation.py \
    --color_mode color \
    --model yolov11x \
    --device cuda:0 \
    --pretrain_epochs 100 \
    --finetune_epochs 5 \
    --batch 4
```

#### Grayscale Mode Training
```bash
python train_domain_adaptation.py \
    --color_mode grayscale \
    --model yolov11x \
    --device cuda:0 \
    --pretrain_epochs 100 \
    --finetune_epochs 5 \
    --batch 4
```

### 2. Evaluation Only

#### Load Trained Model and Evaluate
```bash
python train_domain_adaptation.py \
    --eval_only \
    --checkpoint runs/domain_adapt_color/yolov11x_finetune/weights/best.pt \
    --color_mode color \
    --use_tta
```

#### Test Set Sample Evaluation
```bash
python train_domain_adaptation.py \
    --eval_only \
    --checkpoint runs/domain_adapt_color/yolov11x_finetune/weights/best.pt \
    --color_mode color \
    --eval_test_sample \
    --test_sample_early 3 \
    --test_sample_late 2
```

#### Export Test Predictions to CSV
```bash
python train_domain_adaptation.py \
    --eval_only \
    --checkpoint runs/domain_adapt_color/yolov11x_finetune/weights/best.pt \
    --color_mode color \
    --export_test_csv \
    --test_csv_path predictions.csv \
    --test_vis
```

### 3. Quick Fine-tuning (Skip Pretraining)
```bash
python train_domain_adaptation.py \
    --skip_pretrain \
    --checkpoint pretrained_model.pt \
    --color_mode color \
    --finetune_epochs 10
```

## 📊 Evaluation Metrics

The project implements comprehensive evaluation using COCO-style metrics:

- **mAP@[.5:.95]**: Mean Average Precision across IoU thresholds 0.5-0.95
- **AP50**: Average Precision at IoU=0.5
- **AP75**: Average Precision at IoU=0.75
- **Confidence thresholds**: Evaluation at 0.01, 0.1, 0.2

### Test-Time Augmentation (TTA)
- **YOLOv11 models**: Full TTA support with multiple inference passes
- **YOLOv10 models**: Standard inference only
- **Performance boost**: Typically 2-5% improvement in mAP

## 🔧 Configuration

### Training Parameters
- **Image size**: 640x640 (configurable)
- **Batch size**: 4 (adjustable based on GPU memory)
- **Learning rates**: 
  - Pretrain: 0.01 → 0.01
  - Fine-tune: 0.0001 → 0.00001
- **Optimizer**: AdamW
- **Augmentation**: Albumentations pipeline

### Dataset Splits
- **Pretrain**: All 1270 images (90% train, 10% val)
- **Fine-tune**: Last 60 images (50 train, 10 val)
- **Validation**: First 10% + last 10 images (never used for training)

## 📁 Output Structure

```
runs/domain_adapt_{color_mode}/{name}/
├── {model}_pretrain/              # Pretraining results
│   ├── weights/
│   │   ├── best.pt                # Best model
│   │   └── last.pt                # Last epoch
│   ├── results.png                 # Training curves
│   └── confusion_matrix.png       # Confusion matrix
├── {model}_finetune/              # Fine-tuning results
│   ├── weights/
│   │   ├── best.pt                # Final model
│   │   └── last.pt                # Last epoch
│   └── results.png                # Training curves
├── eval_val_visualizations_*/      # Validation visualizations
├── eval_test_sample_*/            # Test sample visualizations
└── predictions.csv                # Test predictions (if exported)
```

## 🎨 Visualization Features

- **Ground truth vs predictions**: Side-by-side comparison
- **Domain labeling**: Target vs non-target domain identification
- **Confidence scores**: Visual confidence indicators
- **TTA indicators**: Shows whether TTA was used
- **Batch processing**: Efficient visualization generation

## 🚀 Performance Tips

### For Better Results
1. **Use YOLOv11 models** for TTA support
2. **Enable TTA** for final predictions
3. **Adjust batch size** based on GPU memory
4. **Monitor validation metrics** during training
5. **Use appropriate color mode** for your data

### For Faster Training
1. **Reduce image size** (e.g., 512x512)
2. **Use smaller models** (yolov11n, yolov11s)
3. **Increase batch size** if GPU allows
4. **Skip pretraining** if starting from good checkpoint

## 🔍 Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or image size
2. **TTA not working**: Ensure using YOLOv11 models
3. **Low mAP**: Check data quality and augmentation settings
4. **Overfitting**: Reduce learning rate or increase regularization

### Debug Mode
```bash
python train_domain_adaptation.py --eval_only --checkpoint model.pt --color_mode color --eval_test_sample --test_vis
```

## 📈 Results Interpretation

- **mAP@[.5:.95]**: Primary metric for model comparison
- **AP50**: Good for understanding detection quality
- **TTA improvement**: Shows potential performance gain
- **Domain performance**: Compare target vs non-target domain results

## 🤝 Contributing

This project is designed for a specific contest context. For modifications:

1. Adjust dataset paths in scripts
2. Modify augmentation parameters
3. Change model configurations
4. Update evaluation metrics as needed

## 📄 License

This project is part of a computer vision contest submission. Please refer to contest guidelines for usage terms.

---

**Note**: This project implements a sophisticated domain adaptation pipeline specifically designed for pig detection in contest scenarios. The two-stage training approach and comprehensive evaluation tools make it suitable for similar object detection tasks requiring domain adaptation.
