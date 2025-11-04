# -*- coding: utf-8 -*-
"""
generate_heatmaps_auto_split_all.py
-----------------------------------
Generate phase plot heatmaps for gait signals (GaitPDB dataset)
for all combinations of:
    - method: RawPhase / TFS
    - signal_type: left / right / both
    - experiment: Ga / Ju / Si
    - task_mode: normal / dual
Automatically splits train/val/test (80/10/10),
and saves CSV labels with all metadata.

# === Path Layout ===
# Raw input files:     /home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw
# Generated heatmaps:  /home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data
# Project code:        /home/hpc/iwi5/iwi5325h/gait-phase-cnn
# ====================

"""

import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm
import random

# ============ CONFIGURATION ============
DATA_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw"
OUT_BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data"
os.makedirs(OUT_BASE, exist_ok=True)

SIGMA = 8
INTERP_POINTS = 2000
BINS = 248
PAD = 8
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

METHODS = ["rawphase", "tfs"]
SIGNAL_TYPES = ["left", "right", "both"]
EXPERIMENTS = ["Ga", "Ju", "Si"]
TASK_MODES = ["normal", "dual"]  # normal: _01–_09, dual: _10 only

# ---------- Helper Functions ----------

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
    elif signal_type == 'both':
        signal = np.sqrt(np.sum(sensors ** 2, axis=1))
    else:
        raise ValueError("signal_type must be 'left' / 'right' / 'both'")
    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points=2000):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]
    alpha = np.linspace(0, 1, n_points)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)


def get_heat(signal, method="rawphase", sigma=8, bins=248, interp_points=2000):
    """Generate phase plot heatmap."""
    if method == "tfs":
        analytic_signal = hilbert(signal)
        env = np.abs(analytic_signal)
        tfs = signal / env
        analytic_signal = hilbert(tfs)
    else:  # rawphase
        analytic_signal = hilbert(signal)

    x, y = analytic_signal.real.copy(), analytic_signal.imag.copy()
    points = np.vstack([x, y]).T
    ixy = inter2D(points, interp_points)
    x, y = ixy[:, 0], ixy[:, 1]
    heatmap, _, _ = np.histogram2d(x, y, bins=bins)
    p = int(PAD / 2)
    hmap = np.zeros([bins + PAD, bins + PAD])
    hmap[p:-p, p:-p] = heatmap
    return gaussian_filter(hmap, sigma=sigma).T


def save_heatmap_image(heatmap, save_path):
    plt.imsave(save_path, heatmap, cmap=cm.hot, origin='lower')


def split_subjects(subjects):
    n = len(subjects)
    random.shuffle(subjects)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:]
    return train, val, test


# ---------- Main Process ----------

def main():
    print("🚀 Starting heatmap generation...")
    print(f"Input folder: {DATA_DIR}")
    print(f"Output base:  {OUT_BASE}\n")

    all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".txt")]
    print(f"Total .txt files found: {len(all_files)}\n")

    for EXPERIMENT in EXPERIMENTS:
        exp_files = [f for f in all_files if f.startswith(EXPERIMENT)]
        if len(exp_files) == 0:
            print(f"⚠️ No files found for experiment {EXPERIMENT}, skipping.")
            continue

        for TASK_MODE in TASK_MODES:
            if TASK_MODE == "normal":
                selected_files = [f for f in exp_files if "_10" not in f]
            else:
                selected_files = [f for f in exp_files if "_10" in f]

            if len(selected_files) == 0:
                continue

            subjects = sorted(list(set([f[0:6] for f in selected_files])))  # e.g., GaCo01
            train_subj, val_subj, test_subj = split_subjects(subjects)

            for METHOD in METHODS:
                for SIGNAL_TYPE in SIGNAL_TYPES:
                    OUT_DIR = os.path.join(
                        OUT_BASE,
                        f"heatmaps_{METHOD}_{SIGNAL_TYPE}_σ{SIGMA}_i{INTERP_POINTS}_{EXPERIMENT}_{TASK_MODE}"
                    )
                    CSV_PATH = os.path.join(
                        OUT_BASE,
                        f"labels_{METHOD}_{SIGNAL_TYPE}_σ{SIGMA}_i{INTERP_POINTS}_{EXPERIMENT}_{TASK_MODE}.csv"
                    )

                    os.makedirs(OUT_DIR, exist_ok=True)
                    for split in ["train", "val", "test"]:
                        os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

                    with open(os.path.join(OUT_DIR, "README.txt"), "w") as f:
                        f.write(f"Method: {METHOD}\n")
                        f.write(f"Signal type: {SIGNAL_TYPE}\n")
                        f.write(f"Sigma (Gaussian): {SIGMA}\n")
                        f.write(f"Interpolation points: {INTERP_POINTS}\n")
                        f.write(f"Experiment: {EXPERIMENT}\n")
                        f.write(f"Task mode: {TASK_MODE}\n")
                        f.write(f"Sampling rate: 100 Hz\n")

                    records = []
                    for fname in selected_files:
                        subject_id = fname[:6]
                        group = "Co" if "Co" in fname else "Pt"
                        walk_number = fname.split("_")[-1].replace(".txt", "")

                        if subject_id in train_subj:
                            split = "train"
                        elif subject_id in val_subj:
                            split = "val"
                        elif subject_id in test_subj:
                            split = "test"
                        else:
                            continue

                        file_path = os.path.join(DATA_DIR, fname)
                        try:
                            _, sensors = read_gait_data(file_path)
                            signal = get_gait_signal(sensors, SIGNAL_TYPE)
                            heatmap = get_heat(signal, method=METHOD, sigma=SIGMA,
                                               bins=BINS, interp_points=INTERP_POINTS)
                            save_path = os.path.join(OUT_DIR, split, fname.replace(".txt", ".png"))
                            save_heatmap_image(heatmap, save_path)
                            records.append([fname.replace(".txt", ".png"), subject_id, group,
                                            EXPERIMENT, SIGNAL_TYPE, METHOD, walk_number,
                                            split, TASK_MODE])
                        except Exception as e:
                            print(f"⚠️ Error processing {fname}: {e}")

                    df = pd.DataFrame(records, columns=[
                        "filename", "subject_id", "group", "experiment",
                        "signal_type", "method", "walk_number", "split", "task_mode"
                    ])
                    df.to_csv(CSV_PATH, index=False)

                    print(f"✅ [{EXPERIMENT}-{TASK_MODE}-{METHOD}-{SIGNAL_TYPE}] "
                          f"-> {len(records)} heatmaps saved.\n")


if __name__ == "__main__":
    main()
