# -*- coding: utf-8 -*-
"""
summarize_kfold_results.py
----------------------------------------
Collect AUC / Accuracy from all folds:
- 4 experiments: GaNormal, GaDual, JuNormal, SiNormal
- 3 σ groups: sigma8_i2000_kfold, sigma10_i4000_kfold, sigma12_i5000_kfold
- 6 methods: rawphase/tfs × left/right/both
- 3 FC sizes: 128 / 256 / 512

Generates summary_kfold_results.csv
"""

import os
import re
import numpy as np
import pandas as pd

BASE = "/home/woody/iwi5/iwi5325h/gaitphasecnn_middle_data_kfold_experiments"

EXPERIMENTS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]
PARAM_GROUPS = [
    "sigma8_i2000_kfold",
    "sigma10_i4000_kfold",
    "sigma12_i5000_kfold",
]
METHODS = [
    "rawphase_left",
    "rawphase_right",
    "rawphase_both",
    "tfs_left",
    "tfs_right",
    "tfs_both",
]
FC_SIZES = [128, 256, 512]

def extract_metrics(report_path):
    """Extract Accuracy and AUC from report.txt"""
    if not os.path.exists(report_path):
        return None, None

    acc, auc_val = None, None

    with open(report_path, "r") as f:
        text = f.read()

    # Extract accuracy from classification report "accuracy" line
    m_acc = re.search(r"accuracy\s+([\d\.]+)", text)
    if m_acc:
        acc = float(m_acc.group(1))

    # Extract AUC=0.8765
    m_auc = re.search(r"AUC=([\d\.]+)", text)
    if m_auc:
        auc_val = float(m_auc.group(1))

    return acc, auc_val


def main():
    records = []

    for exp in EXPERIMENTS:
        for param in PARAM_GROUPS:
            for method in METHODS:
                for fc in FC_SIZES:

                    fold_acc = []
                    fold_auc = []

                    method_dir = os.path.join(BASE, exp, param, method)
                    res_dir = os.path.join(method_dir, f"results_fc{fc}")
                    if not os.path.exists(res_dir):
                        continue

                    for fold in range(10):
                        report_path = os.path.join(
                            res_dir, f"fold{fold}", "report.txt"
                        )
                        acc, auc_val = extract_metrics(report_path)
                        if acc is not None and auc_val is not None:
                            fold_acc.append(acc)
                            fold_auc.append(auc_val)

                    if len(fold_acc) == 0:
                        continue

                    records.append([
                        exp,
                        param,
                        method,
                        fc,
                        len(fold_acc),
                        np.mean(fold_acc),
                        np.std(fold_acc),
                        np.mean(fold_auc),
                        np.std(fold_auc)
                    ])

    df = pd.DataFrame(records, columns=[
        "Experiment", "ParamGroup", "Method", "FC",
        "NumFolds",
        "Acc_mean", "Acc_std",
        "AUC_mean", "AUC_std"
    ])

    out_csv = "summary_kfold_results.csv"
    df.to_csv(out_csv, index=False)
    print("\n=====================================")
    print("  ✅ Summary saved:", out_csv)
    print("=====================================\n")


if __name__ == "__main__":
    main()
