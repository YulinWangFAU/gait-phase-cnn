# -*- coding: utf-8 -*-
"""
Created on 2025/11/19 15:01

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
split_into_kfold.py
--------------------------------------------
Step 2 of K-fold pipeline:
Take generated heatmaps (no split yet)
and create subject-wise K-fold (default K=10).

- train = 81%
- val   = 9%
- test  = 10%  (1 fold)

Directory structure after split:

sigma10_i4000_kfold/
    rawphase_left/
        fold0/train/
        fold0/val/
        fold0/test/
        labels_rawphase_left_kfold0.csv
        ...
        fold9/
    rawphase_right/
    rawphase_both/
    tfs_left/
    tfs_right/
    tfs_both/
"""

import os
import argparse
import shutil
import pandas as pd
from sklearn.model_selection import KFold


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split_one_group(group_dir, meta_df, k=10):
    """
    group_dir: e.g.  sigma10_i4000_kfold/rawphase_left
    meta_df  : the meta CSV containing all png+subject info
    """
    method, signal = os.path.basename(group_dir).split("_")

    print(f"\n==============================")
    print(f" Processing {method}-{signal}")
    print(f"==============================")

    # 1) collect all rows belonging to this method + signal
    df = meta_df[(meta_df["method"] == method) &
                 (meta_df["signal_type"] == signal)].reset_index(drop=True)

    if len(df) == 0:
        print(f"⚠ No data for {method}-{signal}, skip.")
        return

    subjects = df["subject"].unique()
    subjects.sort()

    print(f" Total subjects: {len(subjects)}")
    print(f" Total samples : {len(df)}")

    # 2) make K folds on subjects
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(subjects)):
        fold_dir = os.path.join(group_dir, f"fold{fold_id}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir   = os.path.join(fold_dir, "val")
        test_dir  = os.path.join(fold_dir, "test")

        ensure_dir(train_dir)
        ensure_dir(val_dir)
        ensure_dir(test_dir)

        test_subjects = subjects[test_idx]
        trainval_subjects = subjects[trainval_idx]

        # train/val split inside trainval_subject group
        # train = 90% of trainval, val = 10%
        n_trainval = len(trainval_subjects)
        n_train = int(n_trainval * 0.90)
        train_subjects = trainval_subjects[:n_train]
        val_subjects   = trainval_subjects[n_train:]

        # assign rows
        train_rows = df[df["subject"].isin(train_subjects)]
        val_rows   = df[df["subject"].isin(val_subjects)]
        test_rows  = df[df["subject"].isin(test_subjects)]

        # 3）copy files
        for _, row in train_rows.iterrows():
            shutil.copy(row["path"], os.path.join(train_dir, row["png"]))

        for _, row in val_rows.iterrows():
            shutil.copy(row["path"], os.path.join(val_dir, row["png"]))

        for _, row in test_rows.iterrows():
            shutil.copy(row["path"], os.path.join(test_dir, row["png"]))

        # save CSV
        out_csv = os.path.join(group_dir, f"labels_{method}_{signal}_kfold{fold_id}.csv")
        pd.concat([
            train_rows.assign(split="train"),
            val_rows.assign(split="val"),
            test_rows.assign(split="test")
        ]).to_csv(out_csv, index=False)

        print(f" ✔ fold{fold_id}: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Root folder: gaitphasecnn_middle_data_kfold")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    ROOT = args.root
    K = args.k

    groups = [d for d in os.listdir(ROOT) if d.startswith("sigma")]
    groups.sort()

    print("\n====================================")
    print("   K-fold split (subject-wise)")
    print("====================================")
    print(f"Root = {ROOT}")
    print(f"K = {K}")

    for g in groups:
        group_path = os.path.join(ROOT, g)
        meta_files = [f for f in os.listdir(group_path) if f.startswith("meta_")]

        if len(meta_files) != 1:
            print(f"⚠ meta CSV not found in {group_path}, skip.")
            continue

        meta_csv = os.path.join(group_path, meta_files[0])
        print(f"\n📄 Loading meta: {meta_csv}")
        meta_df = pd.read_csv(meta_csv)

        # process 6 subfolders
        for sub in ["rawphase_left","rawphase_right","rawphase_both",
                    "tfs_left","tfs_right","tfs_both"]:
            sub_path = os.path.join(group_path, sub)
            if os.path.exists(sub_path):
                split_one_group(sub_path, meta_df, K)
            else:
                print(f"⚠ {sub_path} does not exist, skip.")

    print("\n🎉 All K folds completed!\n")


if __name__ == "__main__":
    main()
