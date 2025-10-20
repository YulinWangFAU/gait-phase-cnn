# -*- coding: utf-8 -*-
"""
Final config.py for gait-phase-CNN (Yulin Wang)
Version-managed configuration supporting multiple preprocessing settings.
This version avoids premature I/O creation at import time and provides
a clear initialization structure.
"""

import os
from datetime import datetime


class Config:
    # === 通用根目录 ===
    ROOT_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data"

    # === 数据与预处理参数 ===
    I_POINTS = 1500       # 插值点数
    GAUSS_SMOOTH = 6      # 高斯平滑核

    # === 版本标签 ===
    VERSION_TAG = f"i{I_POINTS}_s{GAUSS_SMOOTH}"

    # === 主数据目录 ===
    BASE_DIR = ROOT_DIR   # 保持所有数据在同一目录下（不要嵌套 version 子文件夹）

    # === 文件与目录结构 ===
    LABEL_CSV_PATH = os.path.join(BASE_DIR, "labels_fullsignal.csv")
    MODEL_NAME = "cnn_model_paper"
    TAG = os.path.basename(LABEL_CSV_PATH).replace("labels_", "").replace(".csv", "")
    TAGGED_FOLDER = f"hilbert_tfs_cnn_{VERSION_TAG}_{TAG}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # === 输出路径（自动区分版本） ===
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", TAGGED_FOLDER, MODEL_NAME)
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs", TAGGED_FOLDER, MODEL_NAME)
    HEATMAP_DIR = os.path.join(BASE_DIR, f"heatmaps_fullsignal_{VERSION_TAG}")

    # === 数据集划分比例 ===
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # === 训练超参数 ===
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'

    # === 输入形状 ===
    INPUT_SHAPE = (1, 256, 256)  # 单通道热力图

    # === 初始化方法（确保目录存在 + 打印配置） ===
    @classmethod
    def initialize(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, "runs"), exist_ok=True)
        os.makedirs(cls.HEATMAP_DIR, exist_ok=True)

        print(f"\n🧭 Config Loaded:")
        print(f"   → Version tag: {cls.VERSION_TAG}")
        print(f"   → Base dir: {cls.BASE_DIR}")
        print(f"   → Label CSV: {cls.LABEL_CSV_PATH}")
        print(f"   → Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"   → TensorBoard logs: {cls.TENSORBOARD_LOG_DIR}")
        print(f"   → Heatmaps: {cls.HEATMAP_DIR}\n")
