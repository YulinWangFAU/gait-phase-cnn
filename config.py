# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 23:43

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# config.py

import os

class Config:
    # Paths
    LABEL_CSV_PATH = "data/labels.csv"
    CHECKPOINT_DIR = "checkpoints"
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = "runs/phase-cnn"

    # Training settings
    BATCH_SIZE = 8
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 5
    VAL_SPLIT = 0.2

    # Data
    INPUT_SHAPE = (1, 256, 256)
