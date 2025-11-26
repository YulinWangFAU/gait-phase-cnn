# -*- coding: utf-8 -*-
"""
Created on 2025/11/26 23:08

@author: Yulin Wang
@email: yulin.wang@fau.de
"""
# -*- coding: utf-8 -*-
"""
check_folds.py
----------------------------------------
检查每个 K-fold 的 train/val/test 是否包含 Co 和 Pt。
确认不会出现某一类在某个 fold 中消失。
"""

import os
import pandas as pd

BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"

EXPS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
SIGMAS = ["sigma8_i2000_kfold", "sigma10_i4000_kfold", "sigma12_i5000_kfold"]
METHODS = [
    "rawphase_left", "rawphase_right", "rawphase_both",
    "tfs_left", "tfs_right", "tfs_both"
]

def count_labels(df):
    """统计 Co 和 Pt 个数"""
    co = (df["group"] == "Co").sum()
    pt = (df["group"] == "Pt").sum()
    return co, pt

def main():

    for exp in EXPS:
        print(f"\n===============================")
        print(f"🔍 Checking EXPERIMENT: {exp}")
        print("===============================")

        for sigma in SIGMAS:
            for method in METHODS:

                fold_root = os.path.join(BASE, exp, sigma, method, "folds")
                if not os.path.exists(fold_root):
                    continue

                print(f"\n➡ {exp}/{sigma}/{method}")

                for fold in range(20):   # 多写一点，根据实际 folder 自动停
                    fold_dir = os.path.join(fold_root, f"fold{fold}")
                    if not os.path.exists(fold_dir):
                        break

                    train_csv = os.path.join(fold_dir, "train.csv")
                    val_csv = os.path.join(fold_dir, "val.csv")
                    test_csv = os.path.join(fold_dir, "test.csv")

                    if not os.path.exists(train_csv):
                        continue

                    df_train = pd.read_csv(train_csv)
                    df_val = pd.read_csv(val_csv)
                    df_test = pd.read_csv(test_csv)

                    co_tr, pt_tr = count_labels(df_train)
                    co_va, pt_va = count_labels(df_val)
                    co_te, pt_te = count_labels(df_test)

                    print(f"  Fold {fold}:")
                    print(f"    Train: Co={co_tr:2d}, Pt={pt_tr:2d}")
                    print(f"    Val  : Co={co_va:2d}, Pt={pt_va:2d}")
                    print(f"    Test : Co={co_te:2d}, Pt={pt_te:2d}")

                    # 如果某个 split 缺失类别 → 报警告
                    if (co_tr == 0 or pt_tr == 0 or
                        co_va == 0 or pt_va == 0 or
                        co_te == 0 or pt_te == 0):
                        print("    ❌ WARNING: 某个 split 缺失 Co 或 Pt !!!!!")
                    else:
                        print("    ✔ OK (both classes present)")

if __name__ == "__main__":
    main()
