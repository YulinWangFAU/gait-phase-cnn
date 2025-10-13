# -*- coding: utf-8 -*-
"""
Created on 2025/7/13

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm

# === Global Config ===
DATA_DIR = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data/raw"  # 原始 .txt 步态信号目录
INDEX_FILES = ["index_ga.csv", "index_ju.csv", "index_si.csv"]  #  3 个索引 CSV，列出要处理的文件名和标签
BASE_OUTPUT_DIR = "/home/woody/rlvl/rlvl144v/gaitphasecnn_raw_data"
SIGNAL_TYPE = 'both'  # 'left', 'right', or 'both'
FS = 100  # sampling frequency in Hz

# === Output directory ===
config_name = "fullsignal"
output_dir = os.path.join(BASE_OUTPUT_DIR, f"heatmaps_{config_name}")
os.makedirs(output_dir, exist_ok=True)
records = []

# === Processing Functions ===
def apply_threshold(signal, threshold=20.0):
    signal[signal < threshold] = 0
    return signal

def lowpass_filter(signal, cutoff=10, fs=100, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)

def read_signal(filepath, threshold=20.0, apply_filter=True):
    data = np.loadtxt(filepath)
    sensors = data[:, 1:17]
    sensors = apply_threshold(sensors, threshold)
    if apply_filter:
        sensors = lowpass_filter(sensors)
    sensors = sensors - np.mean(sensors, axis=0)
    max_vals = np.max(np.abs(sensors), axis=0)
    max_vals[max_vals == 0] = 1
    sensors = sensors / max_vals
    return sensors

def get_gait_signal(sensors, signal_type='both'):
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

def inter2D(points):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]
    alpha = np.linspace(0, 1, 3000)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)

def get_heat(signal):
    bins = 248
    s = 8
    analytic_signal = hilbert(signal)
    env = np.abs(analytic_signal)
    env[env < 1e-6] = 1e-6
    tss = signal / env
    analytic_signal = hilbert(tss)
    x, y = analytic_signal.real.copy(), analytic_signal.imag.copy()
    points = np.vstack([x, y]).T
    ixy = inter2D(points)
    x, y = ixy[:, 0], ixy[:, 1]
    heatmap, _, _ = np.histogram2d(x, y, bins=bins)
    pad = 32
    p = int(pad / 2)
    hmap = np.zeros([bins + pad, bins + pad])
    hmap[p:-p, p:-p] = heatmap
    return gaussian_filter(hmap, sigma=s).T

# === Main Loop ===
print(f"\n🚀 Processing config: {config_name}")
for index_file in INDEX_FILES:
    index_path = os.path.join(BASE_OUTPUT_DIR, index_file)
    df = pd.read_csv(index_path)

    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{index_file}"):
        fname = row["filename"]
        label = row["label"]
        basename = fname.replace(".txt", "")
        filepath = os.path.join(DATA_DIR, fname)

        try:
            sensors = read_signal(filepath)
            signal = get_gait_signal(sensors, signal_type=SIGNAL_TYPE)

            # === No sliding window: whole signal ===
            heatmap = get_heat(signal)
            out_name = f"{basename}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.imsave(out_path, heatmap, cmap=cm.hot)
            records.append({"filename": out_path, "label": label})
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

# === Save label file ===
label_df = pd.DataFrame(records)
label_csv_path = os.path.join(BASE_OUTPUT_DIR, f"labels_{config_name}.csv")
label_df.to_csv(label_csv_path, index=False)
print(f"✅ Saved {len(records)} samples to: {label_csv_path}")
