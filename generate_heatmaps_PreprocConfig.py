# -*- coding: utf-8 -*-
"""
Generate gait-phase heatmaps for CNN classification (PD vs HC)
Supports multiple preprocessing versions via PreprocConfig:
  - baseline (standard)
  - pd_sensitive (retain pathological features)
  - smooth (over-smoothed control)
Author: Yulin Wang
Date: 2025-10-27
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm

from config_preproc_versions import PreprocConfig
from config_preproc_train import Config

# === Select version here ===
preproc = PreprocConfig(version="pd_sensitive")  # baseline / pd_sensitive / smooth
preproc.summary()

# === Global paths ===
Config.initialize()
DATA_DIR = os.path.join(Config.BASE_DIR, "raw")
INDEX_FILES = ["index_ga.csv", "index_ju.csv", "index_si.csv"]

output_dir = preproc.OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)
records = []

# === Statistics counters ===
group_counts = {"GA": 0, "JU": 0, "SI": 0}

# ----------------------------------------------------------
#                SIGNAL PROCESSING FUNCTIONS
# ----------------------------------------------------------
def apply_threshold(signal, mode="fixed", fixed_threshold=20.0, percentile=5):
    """
    Apply either fixed or adaptive (percentile-based) threshold per channel.
    """
    if mode == "adaptive":
        thresholds = np.percentile(signal, percentile, axis=0)
        mask = signal < thresholds
        signal[mask] = 0
    else:
        signal[signal < fixed_threshold] = 0
    return signal


def lowpass_filter(signal, cutoff=10, fs=100, order=4):
    """
    Apply Butterworth lowpass filter to remove high-frequency noise.
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)


def read_signal(filepath):
    """
    Read and preprocess 16-channel VGRF signal from text file.
    """
    data = np.loadtxt(filepath)
    sensors = data[:, 1:17]

    # === Thresholding ===
    sensors = apply_threshold(
        sensors,
        mode=preproc.THRESHOLD_MODE,
        fixed_threshold=preproc.THRESHOLD_VALUE,
        percentile=preproc.THRESHOLD_VALUE
    )

    # === Filtering ===
    sensors = lowpass_filter(sensors, cutoff=preproc.LOWPASS_CUTOFF, fs=preproc.FS)

    # === Normalization ===
    sensors = sensors - np.mean(sensors, axis=0)
    if preproc.NORMALIZE_MODE == "per_channel":
        max_vals = np.max(np.abs(sensors), axis=0)
        max_vals[max_vals == 0] = 1
        sensors = sensors / max_vals
    elif preproc.NORMALIZE_MODE == "global":
        norm_factor = np.max(np.abs(sensors))
        if norm_factor == 0:
            norm_factor = 1
        sensors = sensors / norm_factor

    return sensors


def get_gait_signal(sensors, signal_type='both'):
    """
    Combine left/right/both foot VGRF channels into a single scalar signal.
    """
    if signal_type == 'left':
        signal = np.linalg.norm(sensors[:, 0:8], axis=1)
    elif signal_type == 'right':
        signal = np.linalg.norm(sensors[:, 8:16], axis=1)
    elif signal_type == 'both':
        signal = np.sqrt(np.sum(sensors ** 2, axis=1))
    else:
        raise ValueError("Invalid signal_type")

    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points=Config.I_POINTS):
    """
    Interpolate a 2D trajectory (x,y) to have fixed length n_points.
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]
    alpha = np.linspace(0, 1, n_points)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)


def get_heat(signal, hilbert_twice=True):
    """
    Generate 2D phase heatmap using (possibly double) Hilbert transform.
    """
    bins = 248
    s = preproc.GAUSS_SMOOTH

    analytic_signal = hilbert(signal)
    env = np.abs(analytic_signal)
    env[env < 1e-6] = 1e-6
    tss = signal / env

    if hilbert_twice:
        analytic_signal = hilbert(tss)
        x, y = analytic_signal.real.copy(), analytic_signal.imag.copy()
    else:
        x, y = tss.real.copy(), tss.imag.copy()

    points = np.vstack([x, y]).T
    ixy = inter2D(points)
    x, y = ixy[:, 0], ixy[:, 1]
    heatmap, _, _ = np.histogram2d(x, y, bins=bins)

    pad = 32
    p = int(pad / 2)
    hmap = np.zeros([bins + pad, bins + pad])
    hmap[p:-p, p:-p] = heatmap
    return gaussian_filter(hmap, sigma=s).T


# ----------------------------------------------------------
#                    MAIN EXECUTION LOOP
# ----------------------------------------------------------
print(f"\n🧭 Running Heatmap Generation for version: {preproc.VERSION_TAG}")
print(f"📂 Input directory : {DATA_DIR}")
print(f"📁 Output directory: {output_dir}\n")

for index_file in INDEX_FILES:
    group_name = index_file.split("_")[1].split(".")[0].upper()  # GA / JU / SI
    index_path = os.path.join(Config.BASE_DIR, index_file)

    if not os.path.exists(index_path):
        print(f"⚠️ Index file not found: {index_path}, skipping...")
        continue

    print(f"\n🚀 Processing group: {group_name} | Version: {preproc.VERSION_TAG}")
    df = pd.read_csv(index_path)
    group_counts[group_name] = len(df)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{group_name}"):
        fname = row["filename"]
        label = row["label"]
        basename = fname.replace(".txt", "")
        filepath = os.path.join(DATA_DIR, fname)

        try:
            sensors = read_signal(filepath)

            # 支持多个信号类型
            if preproc.SIGNAL_TYPE == "left_right":
                signal_types = ["left", "right"]
            else:
                signal_types = [preproc.SIGNAL_TYPE]

            for side in signal_types:
                signal = get_gait_signal(sensors, signal_type=side)
                heatmap = get_heat(signal, hilbert_twice=preproc.HILBERT_TWICE)

                out_name = f"{basename}_{side}_{preproc.VERSION_TAG}.png"
                out_path = os.path.join(output_dir, out_name)
                plt.imsave(out_path, heatmap, cmap=cm.hot)
                records.append({"filename": out_path, "label": label})

        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

# ----------------------------------------------------------
#                    SAVE LABEL FILE
# ----------------------------------------------------------
label_df = pd.DataFrame(records)
label_csv_path = os.path.join(Config.BASE_DIR, f"labels_{preproc.OUTPUT_TAG}.csv")
label_df.to_csv(label_csv_path, index=False)

# ----------------------------------------------------------
#                    FINAL SUMMARY
# ----------------------------------------------------------
print(f"\n✅ Generation Summary ({preproc.VERSION_TAG})")
print(f"🗂️ Total samples generated: {len(records)}")
for g in ["GA", "JU", "SI"]:
    print(f"  ├─ {g}: {group_counts[g]} samples")
print(f"💾 Labels saved to: {label_csv_path}")
print(f"📁 Heatmaps saved to: {output_dir}\n")
print("🎯 Done.")
