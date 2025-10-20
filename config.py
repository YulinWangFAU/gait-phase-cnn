# -*- coding: utf-8 -*-
"""
Final config.py for gait-phase-CNN (Yulin Wang)
Version-managed configuration supporting multiple preprocessing settings
"""

import os
from datetime import datetime


class Config:
    # === 通用根目录 ===
    ROOT_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data"
    BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data"
    # === 数据与预处理参数 ===
    I_POINTS = 1500       # 插值点数
    GAUSS_SMOOTH = 6      # 高斯平滑核

    # === 版本标签（根据插值和平滑核自动生成） ===
    VERSION_TAG = f"i{I_POINTS}_s{GAUSS_SMOOTH}"

    # === 版本专属目录（自动生成） ===
    #BASE_DIR = os.path.join(ROOT_DIR, f"version_{VERSION_TAG}")

    # === 确保版本目录存在 ===
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "runs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "heatmaps_fullsignal"), exist_ok=True)

    # === 标签文件路径 ===
    LABEL_CSV_PATH = os.path.join(BASE_DIR, "labels_fullsignal.csv")

    # === 模型名称（对应使用的网络结构文件）===
    MODEL_NAME = "cnn_model_paper"

    # === 时间戳标识（用于生成唯一输出文件夹） ===
    TAG = os.path.basename(LABEL_CSV_PATH).replace("labels_", "").replace(".csv", "")
    TAGGED_FOLDER = f"hilbert_tfs_cnn_{VERSION_TAG}_{TAG}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # === 自动生成输出路径（模型、日志等） ===
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", TAGGED_FOLDER, MODEL_NAME)
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "cnn_best.pt")
    TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "runs", TAGGED_FOLDER, MODEL_NAME)

    # === 训练超参数 ===
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_DELTA = 0.001
    EARLY_STOPPING_MODE = 'max'

    # === 数据集划分比例 ===
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15

    # === 输入形状 ===
    INPUT_SHAPE = (1, 256, 256)  # 单通道热力图

    # === 调试打印（显示当前版本配置） ===
    print(f"\n🧭 Config Loaded:")
    print(f"   → Version tag: {VERSION_TAG}")
    print(f"   → Base dir: {BASE_DIR}")
    print(f"   → Label CSV: {LABEL_CSV_PATH}")
    print(f"   → Checkpoints: {CHECKPOINT_DIR}")
    print(f"   → TensorBoard logs: {TENSORBOARD_LOG_DIR}\n")
