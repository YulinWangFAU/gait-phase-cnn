# -*- coding: utf-8 -*-
"""
config_preproc_train.py
Training configuration for gait-phase CNN (Yulin Wang)
Used by train.py, independent of preprocessing version files.
"""

import os
from datetime import datetime

class Config:
    ROOT_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data"
    I_POINTS = 1500
    GAUSS_SMOOTH = 6
    VERSION_TAG = f"i{I_POINTS}_s{GAUSS_SMOOTH}"
    BASE_DIR = ROOT_DIR

    LABEL_CSV_PATH = os.path.join(BASE_DIR, "labels_fullsignal.csv")
    MODEL_NAME = "cnn_model_paper"
    TAG = os.path.basename(LABEL_CSV_PATH).replace("labels_", "").replace(".csv", "")
    TAGGED_FOLDER = f"hilbert_tfs_cnn_{VERSION_TAG}_{TAG}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", TAGGED_FOLDER, MODEL_NAME)
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs", TAGGED_FOLDER, MODEL_NAME)
    HEATMAP_DIR = os.path.join(BASE_DIR, "heatmaps_fullsignal_default")

    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'
    INPUT_SHAPE = (1, 256, 256)

    @classmethod
    def initialize(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, "runs"), exist_ok=True)
        print(f"\n🧭 Config Loaded for Training:")
        print(f"   → Version tag: {cls.VERSION_TAG}")
        print(f"   → Base dir: {cls.BASE_DIR}")
        print(f"   → Label CSV: {cls.LABEL_CSV_PATH}")
        print(f"   → Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"   → TensorBoard logs: {cls.TENSORBOARD_LOG_DIR}")
        print(f"   → Heatmaps (default): {cls.HEATMAP_DIR}\n")
