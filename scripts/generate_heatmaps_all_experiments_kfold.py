# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 20:45

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
generate_heatmaps_all_experiments_kfold.py
--------------------------------------------
Generate heatmaps separately for 4 experiments:
    1) Ga Normal   (GaCo*, GaPt* WITHOUT "_10")
    2) Ga Dual     (GaCo*, GaPt* WITH "_10")
    3) Ju Normal   (all JuCo* JuPt*)
    4) Si Normal   (all SiCo* SiPt*)

Each experiment will have its own heatmap folders and meta CSV.

After generating this, you can run split_into_kfold.py
to generate K-fold train/val/test per experiment.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt, cm


# ============ PATHS ============
DATA_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_raw_data/raw"
OUT_BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"
os.makedirs(OUT_BASE, exist_ok=True)

# ============ PARAMETER GROUPS ============
PARAM_GROUPS = [
    {"SIGMA": 8, "INTERP": 2000},
    {"SIGMA": 10, "INTERP": 4000},
    {"SIGMA": 12, "INTERP": 5000},
]

BINS = 248
PAD = 8

METHODS = ["rawphase", "tfs"]
SIGNAL_TYPES = ["left", "right", "both"]


# ============ Helper functions ============
def read_gait_data(filename):
    data = np.loadtxt(filename)
    sensors = data[:, 1:17]
    sensors = sensors - np.mean(sensors, axis=0)
    sensors = sensors / np.max(np.abs(sensors), axis=0)
    return sensors


def get_gait_signal(sensors, signal_type='both'):
    if signal_type == "left":
        signal = np.linalg.norm(sensors[:, 0:8], axis=1)
    elif signal_type == "right":
        signal = np.linalg.norm(sensors[:, 8:16], axis=1)
    else:
        signal = np.sqrt(np.sum(sensors ** 2, axis=1))

    signal = signal - np.mean(signal)
    signal = signal / np.max(np.abs(signal))
    return signal


def inter2D(points, n_points):
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    dist /= dist[-1]

    alpha = np.linspace(0, 1, n_points)
    fn = interp1d(dist, points, kind="cubic", axis=0)
    return fn(alpha)


def get_heat(signal, method, sigma, interp_points):
    if method == "tfs":
        analytic = hilbert(signal)
        env = np.abs(analytic)
        tfs = signal / env
        analytic = hilbert(tfs)
    else:
        analytic = hilbert(signal)

    points = np.vstack([analytic.real, analytic.imag]).T
    ixy = inter2D(points, interp_points)

    heatmap, _, _ = np.histogram2d(ixy[:, 0], ixy[:, 1], bins=BINS)

    p = int(PAD / 2)
    hmap = np.zeros((BINS + PAD, BINS + PAD))
    hmap[p:-p, p:-p] = heatmap
    return gaussian_filter(hmap, sigma=sigma).T


def save_img(hmap, path):
    plt.imsave(path, hmap, cmap=cm.hot, origin="lower")


# ============ Experiment assignment rules ============
def assign_experiment(fname):

    if fname.startswith("Ga"):
        if "_10" in fname:
            return "GaDual"
        else:
            return "GaNormal"

    if fname.startswith("Ju"):
        return "JuNormal"

    if fname.startswith("Si"):
        return "SiNormal"

    return None


# ============ MAIN ============
def main():
    all_txt = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"📂 Found {len(all_txt)} txt files.")

    # ---- Group txt by experiment ----
    EXP_DICT = {"GaNormal": [], "GaDual": [], "JuNormal": [], "SiNormal": []}

    for f in all_txt:
        exp = assign_experiment(f)
        if exp:
            EXP_DICT[exp].append(f)

    print("\n======== Experiment File Counts ========")
    for exp, files in EXP_DICT.items():
        print(f"{exp}: {len(files)} files")
    print("========================================\n")

    # ---- Generate heatmaps per experiment ----
    for exp_name, txt_files in EXP_DICT.items():

        print(f"\n\n#############################################")
        print(f"🔥 Generating EXPERIMENT: {exp_name}")
        print("#############################################\n")

        exp_root = os.path.join(OUT_BASE, exp_name)
        os.makedirs(exp_root, exist_ok=True)

        for cfg in PARAM_GROUPS:
            SIGMA = cfg["SIGMA"]
            INTERP = cfg["INTERP"]

            group_dir = os.path.join(exp_root, f"sigma{SIGMA}_i{INTERP}_kfold")
            os.makedirs(group_dir, exist_ok=True)

            records = []

            for method in METHODS:
                for sig in SIGNAL_TYPES:

                    subdir = os.path.join(group_dir, f"{method}_{sig}")
                    os.makedirs(subdir, exist_ok=True)

                    for fname in txt_files:

                        subject = fname[:6]        # e.g., GaCo01
                        group = "Co" if "Co" in fname else "Pt"
                        walk = fname.split("_")[-1].replace(".txt", "")

                        sensors = read_gait_data(os.path.join(DATA_DIR, fname))
                        signal = get_gait_signal(sensors, sig)
                        heat = get_heat(signal, method, SIGMA, INTERP)

                        out_name = fname.replace(".txt", f"_{method}_{sig}.png")
                        out_path = os.path.join(subdir, out_name)
                        save_img(heat, out_path)

                        records.append([out_name, subject, group, sig,
                                        method, walk, out_path])

                    print(f"✔ {exp_name}: Finished {method}-{sig}")

            df = pd.DataFrame(records, columns=[
                "png", "subject", "group", "signal_type",
                "method", "walk", "path"
            ])

            df.to_csv(os.path.join(group_dir, f"meta_{exp_name}_sigma{SIGMA}_i{INTERP}.csv"), index=False)

            print(f"🎉 Completed {exp_name} σ={SIGMA}, interp={INTERP}")
            print(f"📄 meta saved: meta_{exp_name}_sigma{SIGMA}_i{INTERP}.csv\n")


if __name__ == "__main__":
    main()
