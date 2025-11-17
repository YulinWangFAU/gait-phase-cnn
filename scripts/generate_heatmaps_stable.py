# -*- coding: utf-8 -*-
"""
Created on 2025/11/17 15:00

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# generate_heatmaps_stable.py

import os, json, numpy as np, pandas as pd
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm

DATA_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw"

# ⭐ 多 σ/i 配置，一次性全部生成
CONFIGS = [
    (8, 2000),
    (10, 4000),
    (12, 5000),
]

METHODS = ["rawphase", "tfs"]
SIGNAL_TYPES = ["left", "right", "both"]
EXPERIMENTS = ["Ga", "Ju", "Si"]
TASK_MODES = ["normal", "dual"]

BINS = 248
PAD = 8

# ⭐ 载入全局固定 split
GLOBAL_SPLIT = json.load(open("global_split.json"))

def read_gait_data(filename):
    data = np.loadtxt(filename)
    time = data[:, 0]
    sensors = data[:, 1:17]
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return time, sensors

def get_gait_signal(sensors, t):
    if t == "left":  v = np.linalg.norm(sensors[:,0:8], axis=1)
    elif t == "right": v = np.linalg.norm(sensors[:,8:16], axis=1)
    else: v = np.sqrt(np.sum(sensors**2, axis=1))
    v = v - np.mean(v)
    v = v / np.max(np.abs(v))
    return v

def inter2D(points, n):
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0) / dist[-1]
    alpha = np.linspace(0, 1, n)
    return interp1d(dist, points, axis=0, kind='cubic')(alpha)

def get_heat(signal, method, sigma, bins, interp_points):
    if method == "tfs":
        analytic = hilbert(signal)
        env = np.abs(analytic)
        signal = signal / env
        analytic = hilbert(signal)
    else:
        analytic = hilbert(signal)

    points = np.vstack([analytic.real, analytic.imag]).T
    xy = inter2D(points, interp_points)
    heatmap,_,_ = np.histogram2d(xy[:,0], xy[:,1], bins=bins)
    hmap = np.zeros([bins+PAD, bins+PAD])
    p = PAD//2
    hmap[p:-p,p:-p] = heatmap
    return gaussian_filter(hmap, sigma=sigma).T

def save_heatmap(h, path):
    plt.imsave(path, h, cmap=cm.hot, origin="lower")

def main():
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

    for (SIGMA, INTERP) in CONFIGS:
        out_root = f"/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_stable_sigma{SIGMA}_i{INTERP}"
        os.makedirs(out_root, exist_ok=True)

        print(f"\n===== Generating sigma={SIGMA}, interp={INTERP} =====")

        for EXP in EXPERIMENTS:
            exp_files = [f for f in all_files if f.startswith(EXP)]

            for TASK in TASK_MODES:
                if TASK == "normal":
                    selected = [f for f in exp_files if "_10" not in f]
                else:
                    selected = [f for f in exp_files if "_10" in f]

                if len(selected)==0: continue

                split = GLOBAL_SPLIT[f"{EXP}-{TASK}"]
                train_subj = split["train"]
                val_subj   = split["val"]
                test_subj  = split["test"]

                for M in METHODS:
                    for S in SIGNAL_TYPES:

                        out_dir = os.path.join(out_root,
                                f"heatmaps_{M}_{S}_σ{SIGMA}_i{INTERP}_{EXP}_{TASK}")
                        csv_path = os.path.join(out_root,
                                f"labels_{M}_{S}_σ{SIGMA}_i{INTERP}_{EXP}_{TASK}.csv")

                        os.makedirs(out_dir, exist_ok=True)
                        for sp in ["train","val","test"]:
                            os.makedirs(os.path.join(out_dir,sp), exist_ok=True)

                        rows = []

                        for fname in selected:
                            subj = fname[:6]
                            if subj in train_subj: split_name="train"
                            elif subj in val_subj: split_name="val"
                            elif subj in test_subj: split_name="test"
                            else: continue

                            group = "Co" if "Co" in fname else "Pt"
                            path = os.path.join(DATA_DIR, fname)

                            try:
                                _, sensors = read_gait_data(path)
                                sig = get_gait_signal(sensors, S)
                                h = get_heat(sig, M, SIGMA, BINS, INTERP)
                                save_path = os.path.join(out_dir, split_name, fname.replace(".txt",".png"))
                                save_heatmap(h, save_path)

                                rows.append([fname.replace(".txt",".png"), subj, group,
                                             EXP, S, M, split_name, TASK])
                            except Exception as e:
                                print("Error:", fname, e)

                        df = pd.DataFrame(rows, columns=[
                            "filename","subject_id","group","experiment",
                            "signal_type","method","split","taskmode"
                        ])
                        df.to_csv(csv_path, index=False)
                        print(f"Saved: {csv_path}")

    print("\n🎉 DONE — All stable heatmaps generated.")

if __name__ == "__main__":
    main()
