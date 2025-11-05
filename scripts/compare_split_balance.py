# -*- coding: utf-8 -*-
"""
Created on 2025/11/6 00:30

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
compare_split_balance.py
------------------------
Compare the distribution of PD (Pt) vs HC (Co)
between random-split and balanced-split CSV label files.

Author: Yulin Wang
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

OLD_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data"
NEW_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_balanced"
OUT_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_split_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

def load_csvs(base_dir):
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    csv_paths = [os.path.join(base_dir, f) for f in csv_files]
    return {os.path.basename(p): pd.read_csv(p) for p in csv_paths}

def plot_comparison(df_old, df_new, title, save_path):
    splits = ["train", "val", "test"]
    groups = ["Co", "Pt"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, split in enumerate(splits):
        sub_old = df_old[df_old["split"] == split]
        sub_new = df_new[df_new["split"] == split]

        old_counts = sub_old["group"].value_counts().reindex(groups, fill_value=0)
        new_counts = sub_new["group"].value_counts().reindex(groups, fill_value=0)

        axes[i].bar([0, 1], old_counts.values, width=0.4, label="Old", color="lightgray")
        axes[i].bar([x + 0.4 for x in [0, 1]], new_counts.values, width=0.4, label="Balanced", color="#F97306")
        axes[i].set_xticks([0.2, 1.2])
        axes[i].set_xticklabels(groups)
        axes[i].set_title(split.capitalize())
        axes[i].legend()

        # Print comparison
        print(f"\n=== {title} | {split.upper()} ===")
        print("Old split:")
        print(old_counts)
        print("Balanced split:")
        print(new_counts)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def main():
    df_old_all = load_csvs(OLD_DIR)
    df_new_all = load_csvs(NEW_DIR)

    for name_new, df_new in df_new_all.items():
        # 找到对应的旧版本（不带 _balanced）
        name_old = name_new.replace("_balanced", "")
        if name_old not in df_old_all:
            print(f"⚠️ Skipping {name_new} — no matching old version found.")
            continue

        df_old = df_old_all[name_old]
        title = name_new.replace(".csv", "")
        save_path = os.path.join(OUT_DIR, f"{title}_comparison.png")
        plot_comparison(df_old, df_new, title, save_path)

    print(f"\n✅ All comparisons saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
