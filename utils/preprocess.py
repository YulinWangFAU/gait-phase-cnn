# -*- coding: utf-8 -*-
"""
Created on 2025/6/30 09:56

@author: Yulin Wang
@email: yulin.wang@fau.de
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from tqdm import tqdm

DATA_DIR = "../data"
OUT_DIR = "../heatmaps"
CSV_PATH = "../train_labels.csv"
WIN_SIZE = 256
STEP_SIZE = 128

def apply_threshold(signal, threshold=20.0):
    signal[signal < threshold] = 0
    return signal

def lowpass_filter(signal, cutoff=10, fs=100, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal, axis=0)

def normalize_sensors(sensors):
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return sensors

def get_gait_signal(sensors, signal_type="both"):
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

def inter2D(points, num_points=3000):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]
    alpha = np.linspace(0, 1, num_points)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)

def get_heat(signal, bins=248, sigma=8):
    analytic_signal = hilbert(signal)
    env = np.abs(analytic_signal)
    env[env < 1e-6] = 1e-6
    tss = signal / env
    analytic_signal = hilbert(tss)
    x = analytic_signal.real.copy()
    y = analytic_signal.imag.copy()
    points = np.vstack([x, y]).T
    ixy = inter2D(points)
    x, y = ixy[:, 0], ixy[:, 1]
    heatmap, _, _ = np.histogram2d(x, y, bins=bins)
    pad = 32
    p = int(pad / 2)
    hmap = np.zeros((bins + pad, bins + pad))
    hmap[p:-p, p:-p] = heatmap
    return gaussian_filter(hmap, sigma=sigma)

def extract_windows(signal, size, step):
    n_frames = int((len(signal) - size) / step)
    return [signal[i*step:i*step+size] for i in range(n_frames)]

def process_file(filepath):
    basename = os.path.basename(filepath).replace('.txt', '')
    label = 1 if "Pt" in basename else 0

    data = np.loadtxt(filepath)
    sensors = data[:, 1:17]
    sensors = apply_threshold(sensors)
    sensors = lowpass_filter(sensors)
    sensors = normalize_sensors(sensors)
    signal = get_gait_signal(sensors)

    windows = extract_windows(signal, WIN_SIZE, STEP_SIZE)

    records = []
    for idx, win in enumerate(windows):
        heatmap = get_heat(win)
        fname = f"{basename}_win{idx}.png"
        out_path = os.path.join(OUT_DIR, fname)

        plt.imsave(out_path, heatmap, cmap="hot", format="png")
        records.append((fname, label))

    return records

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records = []
    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]

    for fname in tqdm(txt_files, desc="Processing files"):
        full_path = os.path.join(DATA_DIR, fname)
        records = process_file(full_path)
        all_records.extend(records)

    df = pd.DataFrame(all_records, columns=["filename", "label"])
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} samples to {CSV_PATH}")

if __name__ == "__main__":
    main()
