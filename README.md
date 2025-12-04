# CNN-Based Classification of Parkinson’s Disease from Gait Phase-Plot Heatmaps

This repository implements a complete pipeline for classifying Parkinson’s disease (PD) vs. healthy controls using vertical ground reaction force (VGRF) gait signals converted into phase-plot heatmaps and a lightweight CNN model.

## Pipeline Overview
1. Generate phase-plot heatmaps  
2. Create subject-stratified K-fold splits  
3. Train CNN models across all folds  
4. Save metrics, ROC curves, and confusion matrices  

## Key Scripts
- `scripts/generate_heatmaps_all_experiments_kfold.py`  
  → Generate heatmaps for all experiments and preprocessing settings  
- `scripts/generate_subject_stratified_kfold_final.py`  
  → Create subject-wise K-fold splits  
- `scripts/cnn_phaseplot_model_kfold.py`  
  → Train the CNN for one experiment  
- `scripts/run_all_kfold_train.py`  
  → Run full K-fold training for all configurations  

## How to Run
```bash
python scripts/generate_heatmaps_all_experiments_kfold.py
python scripts/generate_subject_stratified_kfold_final.py
python scripts/run_all_kfold_train.py
