# -*- coding: utf-8 -*-
"""
summarize_kfold_results.py
--------------------------------------------
Summarizes 10-fold results for all experiments:
- GaNormal, GaDual, JuNormal, SiNormal
- 3 param groups (sigma8_i2000, sigma10_i4000, sigma12_i5000)
- 6 methods
- 3 FC sizes
Outputs:
    summary_all_results.csv
    <experiment>_summary.csv
    <experiment>.tex  (LaTeX tables)
"""

import os
import pandas as pd
import numpy as np


BASE = "/Users/wangyulin/Time Series/gaitphasecnn_results_kfold/"

EXPERIMENTS = ["GaNormal", "GaDual", "JuNormal", "SiNormal"]

PARAM_GROUPS = [
    "sigma8_i2000",
    "sigma10_i4000",
    "sigma12_i5000"
]

METHODS = [
    "rawphase_left",
    "rawphase_right",
    "rawphase_both",
    "tfs_left",
    "tfs_right",
    "tfs_both"
]

FC_SIZES = [128, 256, 512]


def extract_metrics(report_path, pred_csv):
    """Reads AUC from report.txt, and computes accuracy/sens/spec."""
    if not os.path.exists(report_path):
        return None, None, None, None

    # ----- AUC from report.txt -----
    with open(report_path, "r") as f:
        text = f.read()
    auc_line = [x for x in text.split("\n") if "AUC" in x]
    if len(auc_line) == 0:
        auc_val = None
    else:
        auc_val = float(auc_line[-1].split("=")[-1])

    # ----- Metrics from test_predictions.csv -----
    df = pd.read_csv(pred_csv)
    y_true = df["true"]
    y_pred = df["pred"]

    acc = (y_true == y_pred).mean()

    # confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    sens = tp / (tp + fn) if (tp + fn) > 0 else None
    spec = tn / (tn + fp) if (tn + fp) > 0 else None

    return auc_val, acc, sens, spec


def main():

    all_records = []

    for exp in EXPERIMENTS:
        print(f"\n========================================")
        print(f"   🔍 Summarizing: {exp}")
        print("========================================\n")

        exp_records = []

        for param in PARAM_GROUPS:
            for method in METHODS:
                for fc in FC_SIZES:

                    results_root = os.path.join(BASE, exp, param, method, f"fc{fc}")

                    if not os.path.exists(results_root):
                        continue

                    auc_list, acc_list, sens_list, spec_list = [], [], [], []

                    for fold in range(10):
                        fold_dir = os.path.join(results_root, f"fold{fold}")
                        report_path = os.path.join(fold_dir, "report.txt")
                        pred_csv = os.path.join(fold_dir, "test_predictions.csv")

                        if not os.path.exists(report_path):
                            continue

                        auc_v, acc_v, sens_v, spec_v = extract_metrics(report_path, pred_csv)

                        if auc_v is not None:
                            auc_list.append(auc_v)
                        if acc_v is not None:
                            acc_list.append(acc_v)
                        if sens_v is not None:
                            sens_list.append(sens_v)
                        if spec_v is not None:
                            spec_list.append(spec_v)

                    # if no folds → skip
                    if len(auc_list) == 0:
                        continue

                    record = {
                        "Experiment": exp,
                        "Param": param,
                        "Method": method,
                        "FC": fc,
                        "AUC_mean": np.mean(auc_list),
                        "AUC_std": np.std(auc_list),
                        "ACC_mean": np.mean(acc_list),
                        "ACC_std": np.std(acc_list),
                        "Sens_mean": np.mean(sens_list),
                        "Spec_mean": np.mean(spec_list),
                    }

                    all_records.append(record)
                    exp_records.append(record)

        # save each experiment summary
        df_exp = pd.DataFrame(exp_records)
        df_exp.to_csv(f"{exp}_summary.csv", index=False)

        # Export LaTeX also
        with open(f"{exp}_table.tex", "w") as f:
            f.write(df_exp.to_latex(index=False, float_format="%.3f"))

        print(f"✔ Saved: {exp}_summary.csv")
        print(f"✔ Saved LaTeX: {exp}_table.tex\n")

    # Save global summary
    df_all = pd.DataFrame(all_records)
    df_all.to_csv("summary_all_results.csv", index=False)
    print("🎉 ALL DONE → summary_all_results.csv generated!\n")


if __name__ == "__main__":
    main()
