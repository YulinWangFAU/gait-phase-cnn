# -*- coding: utf-8 -*-
"""
generate_subject_stratified_kfold_final.py (Universal Auto-KF Version)
---------------------------------------------------------
✔ Auto-adaptive K-fold for ALL experiments:
      KF = min(10, #Co_subjects, #Pt_subjects)
✔ Guarantees every fold has both Co & Pt (no AUC NaN)
✔ Subject-level splits (no leakage)
✔ Valid for: GaNormal, GaDual, JuNormal, SiNormal
✔ Works with: rawphase/tfs + left/right/both
---------------------------------------------------------
YOU ONLY NEED TO CHANGE:
    EXPS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
or:
    EXPS = ["GaDual"]
---------------------------------------------------------
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ================================
# MODIFY THIS LINE ONLY
# ================================
#EXPS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
EXPS = ["GaDual"]   # ← example: run only GaDual
# ================================

BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"

SIGMAS = ["sigma8_i2000_kfold", "sigma10_i4000_kfold", "sigma12_i5000_kfold"]

METHODS = [
    "rawphase_left", "rawphase_right", "rawphase_both",
    "tfs_left", "tfs_right", "tfs_both"
]


def generate_splits():
    for exp in EXPS:
        print(f"\n===============================")
        print(f"   🔥 Running EXPERIMENT: {exp}")
        print("===============================\n")

        exp_dir = os.path.join(BASE, exp)

        for sigma in SIGMAS:
            sigma_dir = os.path.join(exp_dir, sigma)

            meta_csv = os.path.join(
                sigma_dir,
                f"meta_{exp}_{sigma.replace('_kfold','')}.csv"
            )
            if not os.path.exists(meta_csv):
                print(f"❌ meta not found: {meta_csv}")
                continue

            meta = pd.read_csv(meta_csv)
            print(f"✔ Loaded meta: {meta_csv} ({len(meta)} rows)")

            for method in METHODS:

                df = meta[meta["path"].str.contains(method)]
                if len(df) == 0:
                    print(f"⚠️ No images for {method}")
                    continue

                subjects = df["subject"].unique()
                subj_labels = [0 if "Co" in s else 1 for s in subjects]

                # ------------------------------------------
                # AUTO KF SELECTION
                # ------------------------------------------
                num_co = sum(1 for s in subjects if "Co" in s)
                num_pt = sum(1 for s in subjects if "Pt" in s)

                KF_effective = min(10, num_co, num_pt)

                if KF_effective < 2:
                    print(f"❌ ERROR: {exp}/{method}: Not enough subjects for ANY split.")
                    continue

                print(f"➡ {exp}/{sigma}/{method}: Co={num_co}, Pt={num_pt} → KF={KF_effective}")

                # Outer split
                skf_outer = StratifiedKFold(
                    n_splits=KF_effective,
                    shuffle=True,
                    random_state=42
                )

                folds_root = os.path.join(sigma_dir, method, "folds")
                os.makedirs(folds_root, exist_ok=True)

                for fold_id, (trainval_idx, test_idx) in enumerate(
                    skf_outer.split(subjects, subj_labels)
                ):

                    fold_path = os.path.join(folds_root, f"fold{fold_id}")
                    os.makedirs(fold_path, exist_ok=True)

                    test_subj = set(subjects[test_idx])
                    trainval_subj = set(subjects[trainval_idx])

                    df_test = df[df["subject"].isin(test_subj)]
                    df_trainval = df[df["subject"].isin(trainval_subj)]

                    # --------------------------------------
                    # Inner split (train / val)
                    # --------------------------------------
                    tv_subjects = list(df_trainval["subject"].unique())
                    tv_labels = [0 if "Co" in s else 1 for s in tv_subjects]

                    skf_inner = StratifiedKFold(
                        n_splits=KF_effective,
                        shuffle=True,
                        random_state=fold_id + 100
                    )

                    inner_train_idx, inner_val_idx = next(
                        skf_inner.split(tv_subjects, tv_labels)
                    )

                    train_subj = set(tv_subjects[i] for i in inner_train_idx)
                    val_subj   = set(tv_subjects[i] for i in inner_val_idx)

                    df_train = df[df["subject"].isin(train_subj)]
                    df_val   = df[df["subject"].isin(val_subj)]

                    # Save CSVs
                    df_train.to_csv(os.path.join(fold_path, "train.csv"), index=False)
                    df_val.to_csv(os.path.join(fold_path, "val.csv"), index=False)
                    df_test.to_csv(os.path.join(fold_path, "test.csv"), index=False)

                    print(f"✔ Fold {fold_id} saved → Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")


if __name__ == "__main__":
    generate_splits()
    print("\n🎉 DONE: Auto-adaptive stratified K-fold completed!\n")
