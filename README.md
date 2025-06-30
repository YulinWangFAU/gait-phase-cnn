# CNN-based Gait Phase Heatmap Classification for Parkinson’s Disease

## Overview
This project aims to classify Parkinson's Disease (PD) vs. Healthy Controls (HC) using gait signals transformed into phase heatmaps and a CNN model.

## Dataset Description
- Source: GaitPDB (PhysioNet)
- Each file: 16-channel vertical ground reaction force (VGRF) sampled at 100Hz
- Format: .txt files

## Phase Plot Method
- Threshold and filter gait signals
- Compute Hilbert transform → TFS → Interpolated phase trajectory
- Convert into 2D heatmaps via histogram + Gaussian filter

## CNN Architecture
See `models/cnn.py`. A simple 3-layer CNN used for binary classification.

## How to Run

1. Generate heatmaps:
```bash
python scripts/generate_heatmaps.py
```

2. Train CNN:
```bash
python scripts/train_cnn.py
```

3. Evaluate:
```bash
python scripts/evaluate.py
```

4. Grad-CAM visualization:
```bash
python scripts/gradcam_visualize.py
```

## Results & Evaluation
- Metrics: Accuracy, Confusion Matrix, ROC-AUC
- Grad-CAM helps visualize model attention on heatmaps.

## Author
Yulin Wang  
FAU Erlangen-Nürnberg, 2025
