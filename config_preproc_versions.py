# -*- coding: utf-8 -*-
"""
config_preproc_versions.py
Preprocessing configuration manager for gait-phase CNN project
Author: Yulin Wang

Usage:
    from config_preproc_versions import PreprocConfig
    cfg = PreprocConfig(version="pd_sensitive")
    print(cfg.VERSION_TAG, cfg.THRESHOLD_MODE, cfg.LOWPASS_CUTOFF)
"""

import os
from datetime import datetime

class PreprocConfig:
    """
    Manage preprocessing parameter versions for gait heatmap generation.
    """

    def __init__(self, version="baseline"):
        self.version = version.lower().strip()

        # === Common Paths (linked to your main Config) ===
        from config import Config
        Config.initialize()
        self.BASE_DIR = Config.BASE_DIR
        self.HEATMAP_DIR = Config.HEATMAP_DIR

        # === Version-dependent settings ===
        if self.version == "baseline":
            # === Version A: Standard baseline ===
            self.VERSION_TAG = "baseline"
            self.THRESHOLD_MODE = "fixed"   # 固定阈值
            self.THRESHOLD_VALUE = 20.0
            self.LOWPASS_CUTOFF = 10        # Hz
            self.NORMALIZE_MODE = "per_channel"
            self.GAUSS_SMOOTH = 8
            self.HILBERT_TWICE = True
            self.SIGNAL_TYPE = "both"

        elif self.version == "pd_sensitive":
            # === Version B: 保留病理特征版 ===
            self.VERSION_TAG = "pd_sensitive"
            self.THRESHOLD_MODE = "adaptive"   # 动态阈值（分位数）
            self.THRESHOLD_VALUE = 5           # percentile
            self.LOWPASS_CUTOFF = 15           # Hz
            self.NORMALIZE_MODE = "global"
            self.GAUSS_SMOOTH = 6
            self.HILBERT_TWICE = False         # 只一次Hilbert
            self.SIGNAL_TYPE = "left_right"    # 同时生成左右脚版本

        elif self.version == "smooth":
            # === Version C: 过度平滑版（对照组） ===
            self.VERSION_TAG = "smooth"
            self.THRESHOLD_MODE = "fixed"
            self.THRESHOLD_VALUE = 25.0
            self.LOWPASS_CUTOFF = 8
            self.NORMALIZE_MODE = "per_channel"
            self.GAUSS_SMOOTH = 12
            self.HILBERT_TWICE = True
            self.SIGNAL_TYPE = "both"

        else:
            raise ValueError(f"Unknown version name: {self.version}")

        # === Auto path generation ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.OUTPUT_TAG = f"{self.VERSION_TAG}_{timestamp}"
        self.OUTPUT_DIR = os.path.join(self.HEATMAP_DIR, f"heatmaps_{self.OUTPUT_TAG}")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # === Interpolation & sampling (same across versions) ===
        self.I_POINTS = Config.I_POINTS
        self.FS = 100

    def summary(self):
        """
        Print a human-readable summary of configuration parameters.
        """
        print("\n🧭 Preprocessing Configuration Summary")
        print(f"   Version: {self.VERSION_TAG}")
        print(f"   Threshold: {self.THRESHOLD_MODE} ({self.THRESHOLD_VALUE})")
        print(f"   Lowpass cutoff: {self.LOWPASS_CUTOFF} Hz")
        print(f"   Normalize mode: {self.NORMALIZE_MODE}")
        print(f"   Hilbert twice:  {self.HILBERT_TWICE}")
        print(f"   Gaussian σ:     {self.GAUSS_SMOOTH}")
        print(f"   Signal type:    {self.SIGNAL_TYPE}")
        print(f"   Output dir:     {self.OUTPUT_DIR}\n")
