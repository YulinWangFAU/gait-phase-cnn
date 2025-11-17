# -*- coding: utf-8 -*-
"""
Created on 2025/11/17 14:56

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# create_global_split.py

import os, json, random
import numpy as np

DATA_DIR = "/Users/wangyulin/Time Series/629/gait-phase-cnn/data/raw"
EXPERIMENTS = ["Ga", "Ju", "Si"]
TASK_MODES = ["normal", "dual"]

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

random.seed(42)
np.random.seed(42)

def detect_task_mode(fname):
    return "dual" if "_10" in fname else "normal"

def split_subjects(subjects):
    n = len(subjects)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return {
        "train": subjects[:n_train],
        "val": subjects[n_train:n_train+n_val],
        "test": subjects[n_train+n_val:]
    }

GLOBAL_SPLIT = {}

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

for EXP in EXPERIMENTS:
    exp_files = [f for f in all_files if f.startswith(EXP)]

    for TASK in TASK_MODES:
        if TASK == "normal":
            selected = [f for f in exp_files if "_10" not in f]
        else:
            selected = [f for f in exp_files if "_10" in f]

        if len(selected) == 0:
            continue

        subjects = sorted(list(set([f[:6] for f in selected])))

        # 全局固定 split（不 shuffle）
        split = split_subjects(subjects)
        GLOBAL_SPLIT[f"{EXP}-{TASK}"] = split

        print(f"[{EXP}-{TASK}] subjects = {len(subjects)} | "
              f"train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")

with open("global_split.json", "w") as f:
    json.dump(GLOBAL_SPLIT, f, indent=4)

print("\n✅ Saved global_split.json")
