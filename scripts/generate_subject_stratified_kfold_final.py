# -*- coding: utf-8 -*-
"""
Created on 2025/11/20 11:08

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
generate_subject_stratified_kfold_final.py
-------------------------------------------------------
✔ Subject-level splits (no leakage)
✔ Stratified (Co/Pt both appear in every fold)
✔ Test = 10% subjects  (outer 10-fold)
✔ Train = 81% subjects
✔ Val   = 9% subjects  (from remaining 90%)
✔ Works for: GaNormal, GaDual, JuNormal, SiNormal
✔ Works for: rawphase/tfs + left/right/both
✔ Replaces your old KFold version (NOT safe to keep)
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"

EXPS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
SIGMAS = ["sigma8_i2000_kfold", "sigma10_i4000_kfold", "sigma12_i5000_kfold"]
METHODS = [
    "rawphase_left", "rawphase_right", "rawphase_both",
    "tfs_left", "tfs_right", "tfs_both"
]

KF = 10  # test folds


def generate_splits():
    for exp in EXPS:
        # ==================================================
        # ONLY RUN JuNormal
        # ==================================================
        if exp != "JuNormal":
            continue
        # ==================================================
        exp_dir = os.path.join(BASE, exp)

        for sigma in SIGMAS:
            sigma_dir = os.path.join(exp_dir, sigma)

            # load meta CSV
            meta_csv = os.path.join(
                sigma_dir,
                f"meta_{exp}_{sigma.replace('_kfold','')}.csv"
            )
            if not os.path.exists(meta_csv):
                print(f"❌ meta not found: {meta_csv}")
                continue

            meta = pd.read_csv(meta_csv)

            for method in METHODS:
                print(f"\n📌 Processing {exp} / {sigma} / {method}")

                # filter by method (path column includes folder name)
                df = meta[meta["path"].str.contains(method)]
                if len(df) == 0:
                    print(f"❌ No images for {method}")
                    continue

                # SUBJECT-LEVEL data
                subjects = df["subject"].unique()

                # assign label per subject
                subj_labels = []
                for s in subjects:
                    subj_labels.append(0 if "Co" in s else 1)
                subj_labels = pd.Series(subj_labels)

                # outer: 10-fold (test = 10%)
                skf_outer = StratifiedKFold(n_splits=KF, shuffle=True, random_state=42)

                folds_root = os.path.join(sigma_dir, method, "folds")
                os.makedirs(folds_root, exist_ok=True)

                for fold_id, (trainval_idx, test_idx) in enumerate(skf_outer.split(subjects, subj_labels)):

                    fold_path = os.path.join(folds_root, f"fold{fold_id}")
                    os.makedirs(fold_path, exist_ok=True)

                    # -------- subjects in each split --------
                    test_subj = set(subjects[test_idx])
                    trainval_subj = set(subjects[trainval_idx])

                    df_test = df[df["subject"].isin(test_subj)]
                    df_trainval = df[df["subject"].isin(trainval_subj)]

                    # Now split trainval into 90/10 → 81% / 9%
                    tv_subjects = list(df_trainval["subject"].unique())
                    tv_labels = [0 if "Co" in s else 1 for s in tv_subjects]

                    skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=fold_id+100)

                    # take first split for train/val
                    inner_train_idx, inner_val_idx = next(skf_inner.split(tv_subjects, tv_labels))

                    train_subj = set([tv_subjects[i] for i in inner_train_idx])
                    val_subj = set([tv_subjects[i] for i in inner_val_idx])

                    df_train = df[df["subject"].isin(train_subj)]
                    df_val = df[df["subject"].isin(val_subj)]

                    # -------- save CSVs --------
                    df_train.to_csv(os.path.join(fold_path, "train.csv"), index=False)
                    df_val.to_csv(os.path.join(fold_path, "val.csv"), index=False)
                    df_test.to_csv(os.path.join(fold_path, "test.csv"), index=False)

                    print(f"✔ Fold {fold_id}: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")


if __name__ == "__main__":
    generate_splits()
    print("\n🎉 DONE: Subject-level Stratified 10-fold successfully created!\n")
