# -*- coding: utf-8 -*-
"""
Final config.py for gait-phase-CNN (Yulin Wang)
"""

import os
from datetime import datetime

class Config:
    # === 数据根目录 ===
    BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data"

    # === 标签文件路径 ===
    LABEL_CSV_PATH = os.path.join(BASE_DIR, "labels_fullsignal.csv")

    # === 模型名称（对应使用的网络结构文件）===
    MODEL_NAME = "cnn_model_paper"

    # === 数据与预处理标识（方便区分不同实验） ===
    I_POINTS = 3000       # 插值点数
    GAUSS_SMOOTH = 8      # 高斯平滑核
    TAG = os.path.basename(LABEL_CSV_PATH).replace("labels_", "").replace(".csv", "")
    TAGGED_FOLDER = f"hilbert_tfs_cnn_i{I_POINTS}_s{GAUSS_SMOOTH}_{TAG}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # === 自动生成输出路径（模型、日志等） ===
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", TAGGED_FOLDER, MODEL_NAME)
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs", TAGGED_FOLDER, MODEL_NAME)

    # === 训练超参数 ===
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'

    # === 数据集划分比例 ===
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # === 输入形状 ===
    INPUT_SHAPE = (1, 256, 256)  # 单通道热力图
