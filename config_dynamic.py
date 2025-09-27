# -*- coding: utf-8 -*-
"""
Created on 2025/7/13 21:55
一次性跑多个不同 win/step 实验
@author: Yulin Wang
@email: yulin.wang@fau.de
"""

# config_dynamic.py
import os

class Config:
    def __init__(self, win, step):
        if win == 0 and step == 0:
            self.name = "fullsignal"
        else:
            self.name = f"win{win}_step{step}"

        base_dir = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data"
        self.LABEL_CSV_PATH = os.path.join(base_dir, f"labels_{self.name}.csv")
        self.CHECKPOINT_DIR = os.path.join(base_dir, f"checkpoints_{self.name}")
        self.MODEL_SAVE_PATH = os.path.join(self.CHECKPOINT_DIR, "cnn_best.pt")
        self.TENSORBOARD_LOG_DIR = os.path.join(base_dir, f"runs_{self.name}")

        self.BATCH_SIZE = 4
        self.EPOCHS = 50
        self.LEARNING_RATE = 5e-4
        self.WEIGHT_DECAY = 1e-4
        self.EARLY_STOPPING_PATIENCE = 10
        self.EARLY_STOPPING_DELTA = 0.001
        self.EARLY_STOPPING_MODE = 'max'
        self.VAL_SPLIT = 0.25

        self.INPUT_SHAPE = (1, 256, 256)
