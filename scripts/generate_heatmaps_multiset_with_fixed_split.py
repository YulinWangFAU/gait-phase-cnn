# -*- coding: utf-8 -*-
"""
Created on 2025/11/17 15:34

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
generate_heatmaps_multiset_with_fixed_split.py
--------------------------------------------
Generate heatmaps using the SAME balanced subject split,
saved in subject_split.json for reproducible experiments.

Three preprocessing configs:
    1) σ=8,  interp=2000
    2) σ=10, interp=4000
    3) σ=12, interp=5000

This ensures fair comparison of different preprocessing settings.
"""

import os
import json
import numpy as np
import pandas as pd
import random
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm


# ================= PATH =====================

DATA_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw"
OUT_BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_balanced_multiset"
os.makedirs(OUT_BASE, exist_ok=True)

SPLIT_JSON_PATH = os.path.join(OUT_BASE, "subject_split.json")

# ================= CONFIG ===================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

METHODS = ["rawphase", "tfs"]
SIGNAL_TYPES = ["left", "right", "both"]
EXPERIMENTS = ["Ga", "Ju", "Si"]
TASK_MODES = ["normal", "dual"]

# 3 套参数组合
PREPROCESSING_CONFIGS = [
    {"SIGMA": 8,  "INTERP": 2000},
    {"SIGMA": 10, "INTERP": 4000},
    {"SIGMA": 12, "INTERP": 5000},
]

BINS = 248
PAD = 8


# ================ Helper Functions ==================

def read_gait_data(filename):
    data = np.loadtxt(filename)
    time = data[:, 0]
    sensors = data[:, 1:17]
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return time, sensors


def get_gait_signal(sensors, signal_type='both'):
    if signal_type == 'left':
        signal = np.linalg.norm(sensors[:, 0:8], axis=1)
    elif signal_type == 'right':
        signal = np.linalg.norm(sensors[:, 8:16], axis=1)
    else:  # both
        signal = np.sqrt(np.sum(sensors**2, axis=1))

    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance /= distance[-1]

    alpha = np.linspace(0, 1, n_points)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)


def get_heat(signal, method, sigma, interp_points):
    if method == "tfs":
        analytic = hilbert(signal)
        env = np.abs(analytic)
        tfs = signal / env
        analytic = hilbert(tfs)
    else:
        analytic = hilbert(signal)

    x, y = analytic.real.copy(), analytic.imag.copy()
    points = np.vstack([x, y]).T

    ixy = inter2D(points, interp_points)

    heatmap, _, _ = np.histogram2d(ixy[:, 0], ixy[:, 1], bins=BINS)

    p = int(PAD / 2)
    hmap = np.zeros((BINS + PAD, BINS + PAD))
    hmap[p:-p, p:-p] = heatmap

    return gaussian_filter(hmap, sigma=sigma).T


def save_img(hmap, path):
    plt.imsave(path, hmap, cmap=cm.hot, origin='lower')


# ===== Balanced split =====

def split_subjects_balanced(subjects):
    co = [s for s in subjects if "Co" in s]
    pt = [s for s in subjects if "Pt" in s]

    random.shuffle(co)
    random.shuffle(pt)

    def split(arr):
        n = len(arr)
        t = int(TRAIN_RATIO * n)
        v = int(VAL_RATIO * n)
        return arr[:t], arr[t:t+v], arr[t+v:]

    tr_co, va_co, te_co = split(co)
    tr_pt, va_pt, te_pt = split(pt)

    train = tr_co + tr_pt
    val = va_co + va_pt
    test = te_co + te_pt

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ============== JSON I/O ==============

def save_split_json(split_dict):
    with open(SPLIT_JSON_PATH, "w") as f:
        json.dump(split_dict, f, indent=4)
    print(f"📁 Saved NEW split JSON → {SPLIT_JSON_PATH}")


def load_split_json():
    if not os.path.exists(SPLIT_JSON_PATH):
        return None
    with open(SPLIT_JSON_PATH, "r") as f:
        print(f"📁 Loaded existing split JSON → {SPLIT_JSON_PATH}")
        return json.load(f)


# =================== MAIN =====================

def main():
    print("🚀 Starting MULTI-PARAM balanced heatmap generation...\n")

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    # 先尝试加载 split
    split_json = load_split_json()
    split_dict = {} if split_json is None else split_json

    # ----------- Loop over experiment & task mode -----------

    for EXP in EXPERIMENTS:

        exp_files = [f for f in all_files if f.startswith(EXP)]
        if not exp_files:
            continue

        for TASK in TASK_MODES:

            # 选 normal / dual 文件
            if TASK == "normal":
                selected_files = [f for f in exp_files if "_10" not in f]
            else:
                selected_files = [f for f in exp_files if "_10" in f]

            if not selected_files:
                continue

            subjects = sorted(list(set([f[:6] for f in selected_files])))
            split_key = f"{EXP}_{TASK}"

            # ---- (A) 如果已有 split.json，直接用 ----
            if split_json is not None:
                train_subj = split_json[split_key]["train"]
                val_subj   = split_json[split_key]["val"]
                test_subj  = split_json[split_key]["test"]

            # ---- (B) 第一次运行：生成 split ----
            else:
                train_subj, val_subj, test_subj = split_subjects_balanced(subjects)
                split_dict[split_key] = {
                    "train": train_subj,
                    "val": val_subj,
                    "test": test_subj
                }

            # =========== 三套 preprocessing configs ===========

            for cfg in PREPROCESSING_CONFIGS:
                SIGMA = cfg["SIGMA"]
                INTERP = cfg["INTERP"]

                for METHOD in METHODS:
                    for SIGTYPE in SIGNAL_TYPES:

                        OUT_DIR = os.path.join(
                            OUT_BASE,
                            f"heatmaps_{METHOD}_{SIGTYPE}_σ{SIGMA}_i{INTERP}_{EXP}_{TASK}_balanced"
                        )
                        CSV_PATH = os.path.join(
                            OUT_BASE,
                            f"labels_{METHOD}_{SIGTYPE}_σ{SIGMA}_i{INTERP}_{EXP}_{TASK}_balanced.csv"
                        )

                        os.makedirs(OUT_DIR, exist_ok=True)
                        for sp in ["train", "val", "test"]:
                            os.makedirs(os.path.join(OUT_DIR, sp), exist_ok=True)

                        records = []

                        # ------ Generate heatmaps ------
                        for fname in selected_files:

                            sid = fname[:6]
                            group = "Co" if "Co" in fname else "Pt"
                            walk = fname.split("_")[-1].replace(".txt", "")

                            if sid in train_subj:
                                split = "train"
                            elif sid in val_subj:
                                split = "val"
                            else:
                                split = "test"

                            filepath = os.path.join(DATA_DIR, fname)

                            _, sensors = read_gait_data(filepath)
                            signal = get_gait_signal(sensors, SIGTYPE)

                            heat = get_heat(signal, METHOD, SIGMA, INTERP)
                            save_path = os.path.join(
                                OUT_DIR, split, fname.replace(".txt", ".png")
                            )
                            save_img(heat, save_path)

                            records.append([
                                fname.replace(".txt", ".png"),
                                sid, group, EXP,
                                SIGTYPE, METHOD, walk, split, TASK
                            ])

                        df = pd.DataFrame(records, columns=[
                            "filename", "subject_id", "group", "experiment",
                            "signal_type", "method", "walk_number",
                            "split", "task_mode"
                        ])
                        df.to_csv(CSV_PATH, index=False)

                        print(f"✔ Done: {EXP}-{TASK}  σ{SIGMA}-i{INTERP}  {METHOD}-{SIGTYPE}")

    # --- 保存 JSON（仅第一次生成时） ---
    if split_json is None:
        save_split_json(split_dict)

    print("\n🎉 ALL FINISHED — Multi-set + fixed split generation completed!")


if __name__ == "__main__":
    main()
