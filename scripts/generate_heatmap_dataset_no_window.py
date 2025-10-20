# -*- coding: utf-8 -*-
"""
Generate heatmap dataset for gait signals (no windowing)
Versioned by Config.I_POINTS and Config.GAUSS_SMOOTH
Automatically summarizes results and prevents overwriting between versions.
Created by Yulin Wang, modified 2025/10/20
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))          # 当前 scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 上级 gait-phase-cnn/

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm
from config import Config
from config import Config
Config.initialize()
# === 全局路径与参数 ===
INDEX_PREFIXES = ["ga", "ju", "si"]  # 支持三个分组
SIGNAL_TYPE = 'both'                 # 可选：'left'、'right'、'both'
FS = 100                             # 采样率（Hz）

config_name = f"fullsignal_i{Config.I_POINTS}_s{Config.GAUSS_SMOOTH}"
DATA_DIR = os.path.join(Config.BASE_DIR, "raw")
output_dir = os.path.join(Config.BASE_DIR, f"heatmaps_{config_name}")
os.makedirs(output_dir, exist_ok=True)

print("\n🧭 Running Heatmap Generation")
print(f"   ➤ I_POINTS = {Config.I_POINTS}")
print(f"   ➤ GAUSS_SMOOTH = {Config.GAUSS_SMOOTH}")
print(f"📂 Input directory : {DATA_DIR}")
print(f"📁 Output directory: {output_dir}\n")

records = []

# === 功能函数 ===
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

def inter2D(points, n_points=Config.I_POINTS):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / distance[-1]
    alpha = np.linspace(0, 1, n_points)
    interpolator = interp1d(distance, points, kind='cubic', axis=0)
    return interpolator(alpha)

def get_heat(signal):
    bins = 248
    s = Config.GAUSS_SMOOTH
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

# === 主处理循环 ===
for prefix in INDEX_PREFIXES:
    index_file = f"index_{prefix}.csv"
    index_path = os.path.join(Config.BASE_DIR, index_file)
    if not os.path.exists(index_path):
        print(f"⚠️ Index file not found: {index_path}, skipping...")
        continue

    print(f"\n🚀 Processing group: {prefix.upper()} | Config: {config_name}")
    df = pd.read_csv(index_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{prefix.upper()}"):
        fname = row["filename"]
        label = row["label"]
        basename = fname.replace(".txt", "")
        filepath = os.path.join(DATA_DIR, fname)

        try:
            sensors = read_signal(filepath)
            signal = get_gait_signal(sensors, signal_type=SIGNAL_TYPE)
            heatmap = get_heat(signal)
            out_name = f"{basename}.png"
            out_path = os.path.join(output_dir, out_name)
            plt.imsave(out_path, heatmap, cmap=cm.hot)
            records.append({"filename": out_path, "label": label})
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")

# === 保存标签文件 ===
label_csv_path = os.path.join(Config.BASE_DIR, f"labels_{config_name}.csv")
pd.DataFrame(records).to_csv(label_csv_path, index=False)

# === 输出统计信息 ===
print("\n✅ Generation Summary")
print(f"🗂️ Total samples generated: {len(records)}")
for prefix in INDEX_PREFIXES:
    count = len([r for r in records if f'/{prefix.upper()}' in r['filename'] or f'_{prefix}' in r['filename']])
    print(f"  ├─ {prefix.upper()}: {count} samples")
print(f"💾 Labels saved to: {label_csv_path}")
print(f"📁 Heatmaps saved to: {output_dir}\n")
print("🎯 Done.\n")
