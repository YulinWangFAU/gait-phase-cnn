# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 21:14

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
split_into_kfold_experiments.py
----------------------------------------
Generate subject-level 10-fold cross-validation splits for all experiments:

Experiments:
    - GaNormal
    - GaDual
    - JuNormal
    - SiNormal

Inside each experiment:
    For each param group σ,i
    For each method (rawphase/tfs)
    For each signal_type (left/right/both)
Do 10-fold CV and save train/val/test CSV into:

    <method>_<signal_type>/folds/foldX/train.csv
"""

import os
import pandas as pd
from sklearn.model_selection import KFold

BASE_DIR = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"

EXPERIMENTS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
METHODS = ["rawphase", "tfs"]
SIGNALS = ["left", "right", "both"]

KFOLDS = 10


def parse_subject_from_png(fname):
    """
    Example filename:
        GaCo01_01_rawphase_left.png
        → subject = GaCo01
    """
    return fname.split("_")[0]


def build_meta_df(folder):
    """
    Scan folder for PNG images, extract subject + label.
    """
    pngs = [f for f in os.listdir(folder) if f.endswith(".png")]

    records = []
    for f in pngs:
        subject = parse_subject_from_png(f)

        # label
        if "Co" in f:
            label = 0
        else:
            label = 1

        records.append([f, subject, label])

    df = pd.DataFrame(records, columns=["png", "subject", "label"])
    return df


def make_kfold_splits(df, out_dir):
    """
    Create 10 folds with subject-level splits.
    test = 10% subjects
    val = 10% subjects (from train subjects)
    train = remaining subjects
    """
    subjects = sorted(df["subject"].unique())
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(subjects)):
        fold_dir = os.path.join(out_dir, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        test_subj = [subjects[i] for i in test_idx]
        trainval_subj = [subjects[i] for i in trainval_idx]

        # ------------ val split (10% of trainval) ------------
        n_val = max(1, int(len(trainval_subj) * 0.1))
        val_subj = trainval_subj[:n_val]
        train_subj = trainval_subj[n_val:]

        # ------------ map subjects to rows ------------
        train_df = df[df["subject"].isin(train_subj)]
        val_df = df[df["subject"].isin(val_subj)]
        test_df = df[df["subject"].isin(test_subj)]

        # ------------ write CSV ------------
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        print(f"⭐ Saved fold{fold_id}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")


def main():
    print("============================================")
    print("     🚀 Generating K-Fold Splits (10-fold)")
    print("============================================\n")

    for exp in EXPERIMENTS:
        exp_dir = os.path.join(BASE_DIR, exp)
        print(f"\n\n############################################")
        print(f"🔥 Experiment: {exp}")
        print("############################################")

        param_groups = [d for d in os.listdir(exp_dir) if d.startswith("sigma")]

        for param in param_groups:
            param_dir = os.path.join(exp_dir, param)
            print(f"\n-- Parameter group: {param}")

            for method in METHODS:
                for sig in SIGNALS:

                    folder = os.path.join(param_dir, f"{method}_{sig}")
                    if not os.path.exists(folder):
                        print(f"❌ Skip: {folder}")
                        continue

                    print(f"   → Processing {method}_{sig}")

                    df = build_meta_df(folder)

                    # output path:
                    out_dir = os.path.join(folder, "folds")
                    os.makedirs(out_dir, exist_ok=True)

                    make_kfold_splits(df, out_dir)

    print("\n============================================")
    print("🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY!")
    print("============================================")


if __name__ == "__main__":
    main()
