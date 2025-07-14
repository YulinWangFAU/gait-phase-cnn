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
    # LABEL_CSV_PATH = "data/labels_win400_step200.csv"
    # CHECKPOINT_DIR = "checkpoints"
    # MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    # TENSORBOARD_LOG_DIR = "runs/phase-cnn"
    # LABEL_CSV_PATH = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data/labels_win400_step200.csv"
    # CHECKPOINT_DIR = "/home/woody/rlvl/rlvl144v/gait-phase-cnn/checkpoints"
    # MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    # TENSORBOARD_LOG_DIR = "/home/woody/rlvl/rlvl144v/gait-phase-cnn/runs/phase-cnn"

    # # Base directory
    # BASE_DATA_DIR = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data"
    #
    # # Placeholder, 将由 pipeline 中动态修改
    # TAG = "win400_step200"  # 默认标签
    # LABEL_CSV_PATH = os.path.join(BASE_DATA_DIR, f"labels_{TAG}.csv")
    # CHECKPOINT_DIR = os.path.join(BASE_DATA_DIR, "checkpoints", TAG)
    # MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    # TENSORBOARD_LOG_DIR = os.path.join(BASE_DATA_DIR, "runs", TAG)
    # === 通用配置 ===
    LABEL_CSV_PATH = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data/labels_win400_step200.csv"
    MODEL_NAME = "cnn_model_bn"
    # === 根据 LABEL_CSV_PATH 自动提取 TAG ===
    TAG = os.path.basename(LABEL_CSV_PATH).replace("labels_", "").replace(".csv", "")  # 例如 win400_step200 或 fullsignal

    # === 自动命名的输出文件夹，基于预处理参数
    I_POINTS = 3000
    GAUSS_SMOOTH = 8
    TAGGED_FOLDER = f"hilbert_tfs_cnn_i{I_POINTS}_s{GAUSS_SMOOTH}_{TAG}"  # 自动拼接

    BASE_DIR = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data"
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", TAGGED_FOLDER, MODEL_NAME)
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs", TAGGED_FOLDER, MODEL_NAME)
    # Training settings
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4  # ← NEW
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'
    VAL_SPLIT = 0.25

    # Data
    INPUT_SHAPE = (1, 256, 256)
