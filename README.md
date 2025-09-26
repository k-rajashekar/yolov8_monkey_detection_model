# YOLOv8s Monkey Detection for Wildlife Monitoring

A comprehensive implementation of YOLOv8s architecture for automated monkey detection in wildlife monitoring applications, focusing on efficiency-accuracy balance for edge deployment scenarios.

## Overview

This project implements YOLOv8s for monkey detection, achieving improved performance over YOLOv5 baselines while maintaining computational efficiency suitable for resource-constrained environments. The research addresses human-wildlife conflicts through automated monitoring systems capable of continuous operation within computational constraints.

## Key Features

- **YOLOv8s Implementation**: Optimized for monkey detection with custom hyperparameters
- **Comprehensive Evaluation**: Full performance metrics including mAP@0.5, precision, recall, F1-score
- **Visualization Suite**: Training curves, precision-recall plots, detection examples
- **Efficiency Analysis**: Model size, inference time, and parameter count optimization
- **Research-Ready**: Complete pipeline from training to publication-quality results

## Performance Highlights

- **Model Architecture**: YOLOv8s (11.2M parameters, 21.5MB)
- **Dataset**: 2,244 annotated monkey images across diverse environments
- **Training**: 150 epochs with advanced augmentation strategies
- **mAP@0.5**: 0.52 (+8.3% improvement over YOLOv5 baseline of 0.48)
- **Precision**: 0.89 (vs YOLOv5: 0.85)
- **Recall**: 0.78 (vs YOLOv5: 0.80)
- **F1-Score**: 0.83 (vs YOLOv5: 0.82)
- **Inference Time**: 5.5ms (vs YOLOv5: 6.3ms, 12.7% faster)
- **Model Size Reduction**: 20% smaller (21.5MB vs 27MB)

## Dataset

### Download from Kaggle
Download the dataset: [Monkey Species Annotated Images Direct Training](https://www.kaggle.com/datasets/codinginraj/monkey-species-annotated-images-direct-training)

### Dataset Structure
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```


### Dataset Statistics
- **Total Images**: 2,244
- **Train Split**: 70% (1,750 images)
- **Validation Split**: 20% (500 images)
- **Test Split**: 10% (250 images)
- **Classes**: Single class (monkey)
- **Format**: YOLOv8 compatible with bounding box annotations
- **Environment Coverage**: Urban settings (35%), forest canopies (40%), varied lighting (25%)
- **Image Resolutions**: 416×416 to 1920×1080 pixels, normalized to 640×640

## Installation & Setup

### 1. Clone Repository
git clone https://github.com/YOUR_USERNAME/yolov8-monkey-detection.git
cd yolov8-monkey-detection

### 2. Open in Google Colab

Upload the notebook to Google Colab
Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)

### 3. Install Dependencies
Run Cell 1 to install required packages:

ultralytics==8.3.184
opencv-python-headless
matplotlib, seaborn, pandas
scikit-learn
torch, torchvision

### 4. Dataset Setup

Download dataset from Kaggle link above
Upload zip file to Colab
Run extraction code in Cell 2
Update DATASET_PATH variable if needed



Usage
Training Pipeline
# Run cells in sequence:
#### Cell 1: Environment setup
#### Cell 2: Google Drive and dataset setup
#### Cell 3: Dataset analysis
#### Cell 4a: Training configuration and execution
#### Cell 4a_results: Extract training results
#### Cells 5-8: Analysis and visualization


